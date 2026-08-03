// MLVC Cam: NativeActivity (no Java) that captures the back camera at 720p,
// encodes on the HTP NPU with the same QnnRunner/rANS stack as mlvc_encode,
// and streams the .mlvb v2 container (frames=0 => live/unbounded) over TCP.
//
// Launch:
//   am start -n com.mlvc.cam/android.app.NativeActivity \
//      -e target "[2401:...:4cdf]:8712" [-e q 63] [-e qmin 21] [-e qmax 63]
//
// Watch: adb logcat -s MLVC

#include <android/asset_manager.h>
#include <android/native_window.h>
#include <android/rect.h>
#include <android/sensor.h>
#include <android/surface_control.h>
#include <android/window.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "android_util.h"
#include "decode_loop.h"
#include "mlvc_codec.h"
#include "net_util.h"
#include "qnn_runner.h"
#include "srt_transport.h"
#include "stream_feedback.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MLVC", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MLVC", __VA_ARGS__)

namespace {

constexpr int kWidth = 1280, kHeight = 720;

double msNow() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Latest-frame slot: capture thread deposits, encode thread takes. Newest
// wins — a live encoder must drop stale camera frames, never queue them.
struct FrameSlot {
  std::mutex m;
  std::condition_variable cv;
  AImage* img = nullptr;
  std::atomic<bool> stop{false};
  void put(AImage* newImg) {
    std::lock_guard<std::mutex> l(m);
    if (img) AImage_delete(img);  // drop the unconsumed older frame
    img = newImg;
    cv.notify_one();
  }
  AImage* take() {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [&] { return img != nullptr || stop.load(); });
    AImage* out = img;
    img = nullptr;
    return out;
  }
};

FrameSlot gSlot;

void onImage(void* /*ctx*/, AImageReader* reader) {
  AImage* img = nullptr;
  if (AImageReader_acquireLatestImage(reader, &img) == AMEDIA_OK && img) {
    gSlot.put(img);
  }
}

// Physical device tilt (0/90/180/270), independent of the Activity's window
// orientation - the Activity stays locked landscape (AndroidManifest.xml)
// so surfaces/sessions never need to be torn down on rotation, but the
// USER can still tilt the phone to portrait, and both the encoded video
// and the local preview should follow that. Read from the gravity sensor
// (falls back to raw accelerometer) rather than AConfiguration, since the
// latter is frozen by the orientation lock.
struct OrientationSensor {
  ASensorManager* mgr = nullptr;
  ASensorEventQueue* queue = nullptr;
  const ASensor* sensor = nullptr;
  std::atomic<int> bucket{0};  // degrees content must rotate CW to be upright, RELATIVE to launch pose
  int pending = 0;
  int stableCount = 0;
  bool calibrated = false;
  int zeroRaw = 0;

  void init(android_app* app) {
    mgr = ASensorManager_getInstanceForPackage("com.mlvc.cam");
    if (!mgr) { LOGE("no ASensorManager - orientation stuck at 0"); return; }
    sensor = ASensorManager_getDefaultSensor(mgr, ASENSOR_TYPE_GRAVITY);
    if (!sensor) sensor = ASensorManager_getDefaultSensor(mgr, ASENSOR_TYPE_ACCELEROMETER);
    if (!sensor) { LOGE("no gravity/accelerometer sensor - orientation stuck at 0"); return; }
    queue = ASensorManager_createEventQueue(mgr, app->looper, LOOPER_ID_USER, nullptr, nullptr);
    ASensorEventQueue_enableSensor(queue, sensor);
    ASensorEventQueue_setEventRate(queue, sensor, 100000);  // 10Hz; a tilt is not a fast gesture
    LOGI("orientation sensor ready (%s)", ASensor_getName(sensor));
  }

  // Called from android_main's poll loop when ALooper_pollOnce returns
  // LOOPER_ID_USER.
  //
  // The raw accelerometer angle is relative to the device's absolute
  // natural (portrait) frame, not to however the user is holding this
  // landscape-locked app - holding it normally already reads ~90/270 in
  // that frame, not 0. Rather than hardcode which of the two landscape
  // holds this device/camera combo calls "normal" (front vs back camera
  // sensorOrientation differ, and it's easy to get the sign backwards),
  // self-calibrate: whatever pose the phone is in when the first valid
  // sample arrives (i.e. app launch) becomes the zero reference, and
  // bucket tracks only the DELTA from there.
  void drain() {
    if (!queue) return;
    ASensorEvent ev;
    while (ASensorEventQueue_getEvents(queue, &ev, 1) > 0) {
      const float x = ev.acceleration.x, y = ev.acceleration.y;
      if (x * x + y * y < 9.0f) continue;  // near-flat on a table - no reliable signal, keep last
      float deg = std::atan2(-x, y) * 180.0f / static_cast<float>(M_PI);
      if (deg < 0) deg += 360.0f;
      const int nearest = (static_cast<int>(std::lround(deg / 90.0f)) % 4) * 90;
      if (!calibrated) {
        zeroRaw = nearest;
        calibrated = true;
        LOGI("orientation calibrated: launch pose (raw %d) = upright", zeroRaw);
        continue;
      }
      const int rel = ((nearest - zeroRaw) % 360 + 360) % 360;
      if (rel == pending) {
        if (++stableCount >= 3 && bucket.load() != rel) {  // ~3 samples debounce
          bucket.store(rel);
          LOGI("orientation -> %d", rel);
        }
      } else {
        pending = rel;
        stableCount = 0;
      }
    }
  }

  void shutdown() {
    if (queue) { ASensorManager_destroyEventQueue(mgr, queue); queue = nullptr; }
  }
};

OrientationSensor gOrient;

// Packs 8-bit RGBA into the in-memory word AHARDWAREBUFFER_FORMAT_R8G8B8A8
// expects on a little-endian CPU: R at the lowest byte address, so R must
// land in the low bits of the uint32_t.
constexpr uint32_t packRgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8) | r;
}

// Standard 7-segment codes (A=bit0 .. G=bit6), the same table used in
// countless embedded display drivers - 0x3F,0x06,0x5B... for 0-9.
constexpr uint8_t kSevenSeg[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};

// A CPU-locked HUD buffer plus its TRUE reported bounds (from
// AHardwareBuffer_describe - width/stride can legitimately differ, and on
// at least one device+driver combo seen here, a rect write that looked
// in-bounds against the requested logical size still segfaulted, so every
// primitive clamps against the buffer's own reported width/height rather
// than trusting caller geometry).
struct Canvas {
  uint32_t* px;
  uint32_t stride;
  int w, h;
};

// Fills an axis-aligned rect, clamped to the canvas's actual bounds.
void fillRect(const Canvas& c, int x, int y, int w, int h, uint32_t color) {
  const int x0 = std::clamp(x, 0, c.w), x1 = std::clamp(x + w, 0, c.w);
  const int y0 = std::clamp(y, 0, c.h), y1 = std::clamp(y + h, 0, c.h);
  for (int yy = y0; yy < y1; ++yy) {
    uint32_t* row = c.px + static_cast<size_t>(yy) * c.stride;
    for (int xx = x0; xx < x1; ++xx) row[xx] = color;
  }
}

// One 7-segment digit in a [x,y,w,h] cell. Segment thickness scales with
// cell width so the same code works for the small stat digits and the
// bigger button digits.
void drawDigit(const Canvas& c, int x, int y, int w, int h, int digit, uint32_t color) {
  if (digit < 0 || digit > 9) return;
  const uint8_t seg = kSevenSeg[static_cast<size_t>(digit)];
  const int t = std::max(3, w / 5);
  const int midY = y + h / 2;
  if (seg & 0x01) fillRect(c, x, y, w, t, color);                  // A top
  if (seg & 0x02) fillRect(c, x + w - t, y, t, h / 2, color);      // B top-right
  if (seg & 0x04) fillRect(c, x + w - t, midY, t, h / 2, color);   // C bottom-right
  if (seg & 0x08) fillRect(c, x, y + h - t, w, t, color);          // D bottom
  if (seg & 0x10) fillRect(c, x, midY, t, h / 2, color);           // E bottom-left
  if (seg & 0x20) fillRect(c, x, y, t, h / 2, color);              // F top-left
  if (seg & 0x40) fillRect(c, x, midY - t / 2, w, t, color);       // G middle
}

// Right-aligned multi-digit integer, cell pitch `w`+gap per digit.
void drawNumber(const Canvas& c, int rightX, int y, int w, int h, int gap, int value,
                uint32_t color) {
  if (value < 0) value = 0;
  char buf[12];
  snprintf(buf, sizeof(buf), "%d", value);
  const int n = static_cast<int>(strlen(buf));
  int cx = rightX - w;
  for (int i = n - 1; i >= 0; --i) {
    drawDigit(c, cx, y, w, h, buf[i] - '0', color);
    cx -= (w + gap);
  }
}

// On-screen HUD: two buttons (camera flip, lens/zoom cycle) plus a live
// bitrate/fps/quality meter with a frame counter below it. The camera HAL
// writes directly into app->window as its own buffer producer (see
// CameraRig::open) - drawing on that same layer would race with it - so
// this is a separate ASurfaceControl child layer, composited by
// SurfaceFlinger above the camera preview, drawn into with a plain
// AHardwareBuffer (no GL, no Java/dex, no font library: digits are
// classic 7-segment bars, see drawDigit above).
struct Hud {
  ASurfaceControl* root = nullptr;
  ASurfaceControl* layer = nullptr;
  AHardwareBuffer* buf = nullptr;
  int W = 640, H = 400;      // HUD canvas size
  int posX = 24, posY = 24;  // top-left placement in real screen pixels

  int camBtnX = 20, camBtnY = 20, camBtnW = 170, camBtnH = 130;
  int lensBtnX = 210, lensBtnY = 20, lensBtnW = 170, lensBtnH = 130;

  // Updated by the encode thread's existing periodic stats block, read by
  // redraw() - a handful of small values redrawn a few times/sec, not
  // worth a lock.
  std::atomic<int> statFps10{0};   // fps * 10
  std::atomic<int> statKbps{0};
  std::atomic<int> statQ{0};
  std::atomic<int> statFrames{0};
  bool loggedDesc = false;

  bool init(android_app* app) {
    root = ASurfaceControl_createFromWindow(app->window, "mlvc_root");
    if (!root) { LOGE("hud: no root surface control"); return false; }
    layer = ASurfaceControl_create(root, "mlvc_hud");
    if (!layer) { LOGE("hud: no child surface control"); return false; }

    AHardwareBuffer_Desc desc{};
    desc.width = static_cast<uint32_t>(W);
    desc.height = static_cast<uint32_t>(H);
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    if (AHardwareBuffer_allocate(&desc, &buf) != 0) { LOGE("hud: buffer alloc failed"); return false; }

    ASurfaceTransaction* t = ASurfaceTransaction_create();
    ASurfaceTransaction_setZOrder(t, layer, 1);  // above the camera preview layer
    const ARect src{0, 0, W, H};
    const ARect dst{posX, posY, posX + W, posY + H};
    ASurfaceTransaction_setGeometry(t, layer, src, dst, ANATIVEWINDOW_TRANSFORM_IDENTITY);
    ASurfaceTransaction_apply(t);
    ASurfaceTransaction_delete(t);
    return true;
  }

  // touchX/Y are in real screen coordinates (same space as onInput). 0 =
  // no button, 1 = camera-flip button, 2 = lens button.
  int hitTest(float touchX, float touchY) const {
    const int lx = static_cast<int>(touchX) - posX, ly = static_cast<int>(touchY) - posY;
    if (lx >= camBtnX && lx < camBtnX + camBtnW && ly >= camBtnY && ly < camBtnY + camBtnH)
      return 1;
    if (lx >= lensBtnX && lx < lensBtnX + lensBtnW && ly >= lensBtnY && ly < lensBtnY + lensBtnH)
      return 2;
    return 0;
  }

  void redraw(int camIdx, float zoomRatio) {
    if (!buf) return;
    void* addr = nullptr;
    if (AHardwareBuffer_lock(buf, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &addr) != 0)
      return;
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(buf, &desc);
    if (!loggedDesc) {
      LOGI("hud buffer: requested %dx%d, described %ux%u stride=%u", W, H, desc.width, desc.height,
           desc.stride);
      loggedDesc = true;
    }
    // Clamp against the buffer's OWN reported width/height, not the W/H we
    // requested - on this device+driver a rect that looked in-bounds
    // against the requested size still segfaulted (see qnn-in-app... no,
    // see PLAN.md HUD entry), so every primitive clips to what the
    // allocator actually gave us.
    const Canvas c{static_cast<uint32_t*>(addr), desc.stride, static_cast<int>(desc.width),
                   static_cast<int>(desc.height)};

    const uint32_t clear = packRgba(0, 0, 0, 0);
    fillRect(c, 0, 0, W, H, clear);

    const uint32_t panelBg = packRgba(20, 22, 28, 190);
    const uint32_t btnBg = packRgba(45, 50, 60, 220);
    const uint32_t digitColor = packRgba(240, 240, 245, 255);
    const uint32_t barBg = packRgba(60, 64, 72, 220);
    const uint32_t barBr = packRgba(56, 189, 248, 255);   // bitrate: sky blue
    const uint32_t barFps = packRgba(74, 222, 128, 255);  // fps: green
    const uint32_t barQ = packRgba(250, 204, 21, 255);    // quality: amber

    fillRect(c, 0, 0, W, H, panelBg);
    fillRect(c, camBtnX, camBtnY, camBtnW, camBtnH, btnBg);
    fillRect(c, lensBtnX, lensBtnY, lensBtnW, lensBtnH, btnBg);

    // Button contents: big single digit for camera index, "N.N" (or "10")
    // for zoom ratio.
    drawDigit(c, camBtnX + camBtnW / 2 - 28, camBtnY + 15, 56, 100, camIdx, digitColor);
    {
      const int dw = 40, dh = 80, gap = 10, dotW = 14;
      const bool whole = zoomRatio >= 9.95f;
      const int lo = static_cast<int>(std::lround((zoomRatio - static_cast<int>(zoomRatio)) * 10)) % 10;
      const int hi = static_cast<int>(zoomRatio);
      int cx = lensBtnX + lensBtnW - 20;
      if (!whole) {
        cx -= dw; drawDigit(c, cx, lensBtnY + 25, dw, dh, lo, digitColor);
        cx -= dotW; fillRect(c, cx + dotW / 2 - 4, lensBtnY + 25 + dh - 12, 8, 8, digitColor);
        cx -= gap;
      }
      cx -= dw; drawDigit(c, cx, lensBtnY + 25, dw, dh, hi % 10, digitColor);
      if (hi >= 10) {
        cx -= gap + dw;
        drawDigit(c, cx, lensBtnY + 25, dw, dh, hi / 10, digitColor);
      }
    }

    // Meter rows: colored dot, horizontal fill bar, right-aligned number.
    // Number cell is 34x48 - the first cut used 18x30 and was technically
    // correct but too small to read at arm's length (see PLAN.md HUD entry).
    const int rowY0 = camBtnY + camBtnH + 20, rowH = 40, rowGap = 16;
    const int numW = 34, numH = 48;
    const int barX = 56, barW = W - barX - 130, numX = W - 20;
    auto meterRow = [&](int row, uint32_t color, int fillPct, int value) {
      const int y = rowY0 + row * (rowH + rowGap);
      fillRect(c, 20, y + rowH / 2 - 8, 16, 16, color);
      fillRect(c, barX, y, barW, rowH, barBg);
      const int fw = std::clamp(fillPct, 0, 100) * barW / 100;
      if (fw > 0) fillRect(c, barX, y, fw, rowH, color);
      drawNumber(c, numX, y + rowH / 2 - numH / 2, numW, numH, 8, value, digitColor);
    };
    const int kbps = statKbps.load();
    const int fps10 = statFps10.load();
    const int q = statQ.load();
    meterRow(0, barBr, kbps * 100 / 1500, kbps);
    meterRow(1, barFps, fps10 * 100 / 300, fps10 / 10);
    meterRow(2, barQ, q * 100 / 63, q);

    // Frame count, plain readout beneath the meter rows.
    const int statsY = rowY0 + 3 * (rowH + rowGap) + 8;
    drawNumber(c, W - 20, statsY, 22, 32, 6, statFrames.load(), digitColor);

    AHardwareBuffer_unlock(buf, nullptr);

    ASurfaceTransaction* t = ASurfaceTransaction_create();
    ASurfaceTransaction_setBuffer(t, layer, buf, -1);
    ASurfaceTransaction_apply(t);
    ASurfaceTransaction_delete(t);
  }
};

Hud gHud;

// YUV_420_888 (arbitrary row/pixel strides) -> fp16 [1,3,H,W] in [0,1].
// Chroma is nearest-upsampled 2x, matching the file encoder's convention.
void image888ToTensor(AImage* img, uint16_t* out) {
  static const auto kLut = [] {
    std::array<uint16_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
      __fp16 h = static_cast<__fp16>(static_cast<float>(i) / 255.0f);
      memcpy(&t[static_cast<size_t>(i)], &h, 2);
    }
    return t;
  }();
  const size_t hw = static_cast<size_t>(kWidth) * kHeight;
  uint8_t* data = nullptr;
  int len = 0, rowStride = 0, pixStride = 0;

  AImage_getPlaneData(img, 0, &data, &len);
  AImage_getPlaneRowStride(img, 0, &rowStride);
  for (int r = 0; r < kHeight; ++r) {
    const uint8_t* src = data + static_cast<size_t>(r) * rowStride;
    uint16_t* dst = out + static_cast<size_t>(r) * kWidth;
    for (int c = 0; c < kWidth; ++c) dst[c] = kLut[src[c]];
  }
  for (int plane = 1; plane <= 2; ++plane) {
    AImage_getPlaneData(img, plane, &data, &len);
    AImage_getPlaneRowStride(img, plane, &rowStride);
    AImage_getPlanePixelStride(img, plane, &pixStride);
    uint16_t* base = out + static_cast<size_t>(plane) * hw;
    for (int r = 0; r < kHeight; ++r) {
      const uint8_t* src = data + static_cast<size_t>(r / 2) * rowStride;
      uint16_t* dst = base + static_cast<size_t>(r) * kWidth;
      for (int c = 0; c < kWidth; c += 2) {
        const uint16_t v = kLut[src[(c / 2) * pixStride]];
        dst[c] = v;
        dst[c + 1] = v;
      }
    }
  }
}

// Rotates+letterboxes the [3,H,W] fp16 tensor in `src` to appear upright
// when the phone is tilted `bucket` degrees from its default landscape
// hold, writing into `dst` (same size). 0 is a no-op; 180 is a plain
// flip, still full-frame. 90/270 need scaling down: the rotated content
// is HxW=720x1280 which does not fit the model's fixed 1280x720 input
// as-is, so it is scaled to fit and gray-letterboxed (fp16 0.5, same
// convention as the I-frame gray reference). Nearest-neighbor sampling,
// matching the chroma upsampling above - no bilinear anywhere in this
// pipeline. A genuine portrait-shaped model would need a second AI-Hub
// export (see PLAN.md); this keeps the existing 1280x720 model as-is.
void rotateLetterboxTensor(const uint16_t* src, uint16_t* dst, int bucket) {
  constexpr uint16_t kGray = 0x3800;  // fp16 0.5
  constexpr int W = kWidth, H = kHeight;
  const size_t hw = static_cast<size_t>(W) * H;
  if (bucket == 0) {
    memcpy(dst, src, 3 * hw * sizeof(uint16_t));
    return;
  }
  if (bucket == 180) {
    for (int ch = 0; ch < 3; ++ch) {
      const uint16_t* s = src + static_cast<size_t>(ch) * hw;
      uint16_t* d = dst + static_cast<size_t>(ch) * hw;
      for (size_t i = 0; i < hw; ++i) d[i] = s[hw - 1 - i];
    }
    return;
  }
  // 90 or 270: fit the rotated (HxW) content inside the original WxH
  // canvas, centered, gray-padded on the sides that don't fill.
  const float scale = std::min(static_cast<float>(W) / H, static_cast<float>(H) / W);
  const int outW = static_cast<int>(std::lround(H * scale));  // fitted width of rotated content
  const int outH = static_cast<int>(std::lround(W * scale));  // fitted height of rotated content
  const int offX = (W - outW) / 2, offY = (H - outH) / 2;
  for (int ch = 0; ch < 3; ++ch) {
    const uint16_t* s = src + static_cast<size_t>(ch) * hw;
    uint16_t* d = dst + static_cast<size_t>(ch) * hw;
    std::fill(d, d + hw, kGray);
    for (int oy = 0; oy < outH; ++oy) {
      for (int ox = 0; ox < outW; ++ox) {
        // Map output(ox,oy) in the fitted upright rectangle back to a
        // source pixel in the original landscape buffer.
        const float u = static_cast<float>(ox) / outW;  // [0,1) across fitted width
        const float v = static_cast<float>(oy) / outH;  // [0,1) across fitted height
        int sx, sy;
        if (bucket == 90) {
          sx = static_cast<int>(v * W);
          sy = static_cast<int>((1.0f - u) * H);
        } else {  // 270
          sx = static_cast<int>((1.0f - v) * W);
          sy = static_cast<int>(u * H);
        }
        sx = std::clamp(sx, 0, W - 1);
        sy = std::clamp(sy, 0, H - 1);
        d[static_cast<size_t>(offY + oy) * W + (offX + ox)] = s[static_cast<size_t>(sy) * W + sx];
      }
    }
  }
}

// Packs a YUV_420_888 AImage into planar I420 (the CLI encoder's input
// format), honoring row/pixel strides.
void image888ToI420(AImage* img, uint8_t* out) {
  const size_t hw = static_cast<size_t>(kWidth) * kHeight;
  uint8_t* data = nullptr;
  int len = 0, rowStride = 0, pixStride = 0;
  AImage_getPlaneData(img, 0, &data, &len);
  AImage_getPlaneRowStride(img, 0, &rowStride);
  for (int r = 0; r < kHeight; ++r) {
    memcpy(out + static_cast<size_t>(r) * kWidth, data + static_cast<size_t>(r) * rowStride,
           kWidth);
  }
  uint8_t* dstU = out + hw;
  uint8_t* dstV = out + hw + hw / 4;
  for (int plane = 1; plane <= 2; ++plane) {
    AImage_getPlaneData(img, plane, &data, &len);
    AImage_getPlaneRowStride(img, plane, &rowStride);
    AImage_getPlanePixelStride(img, plane, &pixStride);
    uint8_t* dst = plane == 1 ? dstU : dstV;
    for (int r = 0; r < kHeight / 2; ++r) {
      const uint8_t* src = data + static_cast<size_t>(r) * rowStride;
      for (int c = 0; c < kWidth / 2; ++c) dst[static_cast<size_t>(r) * (kWidth / 2) + c] = src[c * pixStride];
    }
  }
}

// Bridge mode: forward raw I420 frames over localhost TCP to the CLI, which
// owns the NPU and the SRT transport. Historically believed mandatory ("app
// sandbox blocks QNN") - disproven: the real blocker was libcdsprpc.so not
// resolving in the app's linker namespace, fixed by the manifest's
// <uses-native-library> entry. Bridge remains useful because the in-app
// encode loop below is TCP-only while the CLI speaks SRT.
void bridgeLoop(uint16_t port, const std::string& dumpPath) {
  bool dumped = false;
  const int fd = mlvc::tcpConnect("127.0.0.1", port);
  if (fd < 0) { LOGE("bridge connect 127.0.0.1:%u failed (CLI listening?)", port); return; }
  const uint32_t hdr[3] = {0x30323449u /* 'I420' */, kWidth, kHeight};
  if (!mlvc::sendAll(fd, hdr, sizeof(hdr))) { close(fd); return; }
  LOGI("bridge connected on port %u", port);
  const size_t frameBytes = static_cast<size_t>(kWidth) * kHeight * 3 / 2;
  std::vector<uint8_t> buf(frameBytes);
  int f = 0;
  while (!gSlot.stop.load()) {
    AImage* img = gSlot.take();
    if (!img) break;
    image888ToI420(img, buf.data());
    AImage_delete(img);
    if (!dumped && !dumpPath.empty()) {
      FILE* f = fopen(dumpPath.c_str(), "wb");
      if (f) { fwrite(buf.data(), 1, buf.size(), f); fclose(f); LOGI("dumped raw I420 frame to %s", dumpPath.c_str()); }
      dumped = true;
    }
    if (!mlvc::sendAll(fd, buf.data(), frameBytes)) { LOGI("bridge peer closed"); break; }
    if (++f % 150 == 0) LOGI("bridge fed %d frames", f);
  }
  close(fd);
  LOGI("bridge ended after %d frames", f);
}

// intentExtra/nativeLibDir/materializeAsset moved to android_util.h/cpp -
// decode_loop.cpp (the receiver path) needs them too.
using mlvc::intentExtra;
using mlvc::materializeAsset;
using mlvc::nativeLibDir;

void encodeLoop(android_app* app, const std::string& target, int q0, int qMin, int qMax,
                const std::string& enc1Path, const std::string& enc2Path,
                const std::string& pmfPath, const std::string& backendLib,
                bool useSrt, int srtLatencyMs) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int c = 4; c < 8; ++c) CPU_SET(c, &mask);
  sched_setaffinity(gettid(), sizeof(mask), &mask);

  mlvc::PmfTables tables;
  if (!tables.load(pmfPath)) { LOGE("pmf load failed"); return; }
  mlvc::QnnRunner enc1, enc2;
  LOGI("diag: backend=%s", backendLib.c_str());
  if (!enc1.init(backendLib, "libQnnSystem.so", enc1Path, true)) {
    LOGE("enc1 init: %s", enc1.error().c_str());
    if (backendLib != "libQnnHtp.so") {
      LOGI("diag: CPU backend test ends here (expected — CPU can't execute an "
           "HTP-compiled binary); the diagnostic signal is WHICH step failed, see above");
    }
    return;
  }
  LOGI("diag: enc1 init SUCCEEDED with backend=%s", backendLib.c_str());
  if (!enc2.init(backendLib, "libQnnSystem.so", enc2Path, false)) {
    LOGE("enc2 init: %s", enc2.error().c_str());
    return;
  }
  const std::string g1 = enc1.graphs()[0].graphName;
  const std::string g2 = enc2.graphs()[0].graphName;

  std::string host;
  uint16_t port = 0;
  if (!mlvc::splitHostPort(target, host, port)) { LOGE("bad target %s", target.c_str()); return; }
  SRTSOCKET srt = SRT_INVALID_SOCK;
  int fd = -1;
  if (useSrt) {
    mlvc::srtStartup();
    srt = mlvc::srtConnect(host, port, srtLatencyMs);
    if (srt == SRT_INVALID_SOCK) { LOGE("srt connect %s failed", target.c_str()); return; }
    LOGI("streaming to %s over SRT (latency %d ms)", target.c_str(), srtLatencyMs);
  } else {
    fd = mlvc::tcpConnect(host, port);
    if (fd < 0) { LOGE("connect %s failed", target.c_str()); return; }
    LOGI("streaming to %s", target.c_str());
  }

  mlvc::ModelDims dims;
  const size_t hw = static_cast<size_t>(kWidth) * kHeight;
  const size_t featN = static_cast<size_t>(dims.featureCh) * (kHeight / 8) * (kWidth / 8);
  const size_t zN = static_cast<size_t>(dims.zCh) * dims.zH * dims.zW;
  const size_t yHalfN = static_cast<size_t>(dims.latentCh / 2) * dims.yH * dims.yW;
  std::vector<uint16_t> x(hw * 3), xRot(hw * 3), refFrame(hw * 3, 0x3800), refFeature(featN, 0);
  std::vector<uint16_t> feature(featN), zRaw(zN), yRaw0(yHalfN), yRaw1(yHalfN), xHat(hw * 3);
  int32_t qShifted = 0;
  uint16_t refExists = 0;

  auto orderOf = [](const mlvc::GraphIo& g, const char* n) -> size_t {
    for (size_t i = 0; i < g.inputs.size(); ++i)
      if (g.inputs[i].name == n) return i;
    return 0;
  };
  const auto& io1 = enc1.graphs()[0];
  std::vector<void*> in1(io1.inputs.size());
  std::vector<void*> out1 = {feature.data(), zRaw.data(), yRaw0.data(), yRaw1.data()};
  const auto& io2 = enc2.graphs()[0];
  std::vector<void*> in2(io2.inputs.size());
  std::vector<void*> out2 = {xHat.data()};

  // Container v4 on both transports, byte-identical to mlvc_encode's wire
  // format (header: magic,version,w,h,q,frames; per frame: size, qFrame,
  // curIdx, sender-timestamp, payload). SRT gets the header as a repeated
  // out-of-band message (srtSendHeader) so TSBPD startup loss can't kill
  // the session; per-frame data is accumulated and sent as chunked messages.
  std::vector<uint8_t> srtBuf;
  uint32_t srtFrameIdx = 0;
  bool sendOk = true;
  auto sendBytes = [&](const void* p, size_t n) {
    if (srt != SRT_INVALID_SOCK) {
      const auto* b = static_cast<const uint8_t*>(p);
      srtBuf.insert(srtBuf.end(), b, b + n);
    } else if (!mlvc::sendAll(fd, p, n)) {
      sendOk = false;
    }
  };
  auto wrU32 = [&](uint32_t v) { sendBytes(&v, 4); };
  auto flushFrame = [&] {
    if (srt == SRT_INVALID_SOCK || srtBuf.empty()) return;
    if (!mlvc::srtSendFrame(srt, srtFrameIdx++, srtBuf.data(), srtBuf.size())) sendOk = false;
    srtBuf.clear();
  };
  {
    uint32_t hdr[6] = {0x424C564Du, 4, kWidth, kHeight, static_cast<uint32_t>(q0), 0};
    if (srt != SRT_INVALID_SOCK) {
      mlvc::srtSendHeader(srt, hdr, sizeof(hdr));
    } else {
      sendBytes(hdr, sizeof(hdr));
    }
  }

  mlvc::RateController rc;
  rc.q = q0; rc.qMin = qMin; rc.qMax = qMax;
  mlvc::CoderWorkspace ws;
  ws.enc.reserve(zN + 2 * yHalfN);
  int f = 0;
  int curIdx = 0;  // resets at every I-frame; drives qpShift and the schedule
  double statT0 = msNow(), busyMs = 0;
  size_t statBytes = 0;

  while (!gSlot.stop.load()) {
    AImage* img = gSlot.take();
    if (!img) break;
    const double t0 = msNow();
    image888ToTensor(img, x.data());
    AImage_delete(img);
    const int orientBucket = gOrient.bucket.load();
    const uint16_t* xFinal = x.data();
    if (orientBucket != 0) {
      rotateLetterboxTensor(x.data(), xRot.data(), orientBucket);
      xFinal = xRot.data();
    }

    mlvc::Feedback fb{}, latest{};
    bool got = false;
    if (srt != SRT_INVALID_SOCK) {
      // Live-mode SRT hard-rejects any recv buffer smaller than
      // SRTO_PAYLOADSIZE even for tiny messages (LiveCC checkTransArgs).
      char fbuf[mlvc::kSrtChunkPayload];
      int n;
      while ((n = srt_recv(srt, fbuf, sizeof(fbuf))) >= (int)sizeof(mlvc::Feedback)) {
        memcpy(&fb, fbuf, sizeof(fb));
        if (fb.magic == mlvc::kFeedbackMagic) { latest = fb; got = true; }
      }
    } else {
      while (mlvc::recvNonBlocking(fd, &fb, sizeof(fb))) {
        if (fb.magic == mlvc::kFeedbackMagic) { latest = fb; got = true; }
      }
    }
    // I-frame recovery is unconditional; rate adaptation rides the same
    // feedback. Same split as mlvc_encode.
    if (got && latest.needIframe) curIdx = 0;
    if (got) {
      if (latest.verdict < 0) rc.maybeCongested(latest.waitMs);
      else if (latest.verdict > 0) rc.healthy();
      else rc.hold();
    }

    // Reference refresh, identical schedule to mlvc_encode and the decoder.
    if (mlvc::isIframe(curIdx)) {
      curIdx = 0;
      std::fill(refFrame.begin(), refFrame.end(), 0x3800);  // fp16 0.5 gray
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else if (mlvc::isFeatureReset(curIdx)) {
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else {
      refExists = 0x3C00;  // fp16 1.0
    }

    const int qFrame = rc.q;
    qShifted = qFrame + mlvc::qpShift(curIdx);

    in1[orderOf(io1, "x")] = const_cast<void*>(static_cast<const void*>(xFinal));
    in1[orderOf(io1, "ref_frame")] = refFrame.data();
    in1[orderOf(io1, "ref_feature")] = refFeature.data();
    in1[orderOf(io1, "q_index_shifted")] = &qShifted;
    in1[orderOf(io1, "ref_exists")] = &refExists;
    if (!enc1.execute(g1, in1, out1)) { LOGE("enc1: %s", enc1.error().c_str()); break; }

    std::vector<uint8_t> payload;
    std::thread rans([&] {
      payload = mlvc::encodeBitstream(tables, dims, yRaw0.data(), yRaw1.data(), zRaw.data(),
                                      qFrame, ws);
    });
    in2[orderOf(io2, "feature")] = feature.data();
    in2[orderOf(io2, "q_index_shifted")] = &qShifted;
    const bool ok2 = enc2.execute(g2, in2, out2);
    rans.join();
    if (!ok2) { LOGE("enc2: %s", enc2.error().c_str()); break; }

    const double tSend0 = msNow();
    wrU32(static_cast<uint32_t>(payload.size()));
    wrU32(static_cast<uint32_t>(qFrame));
    wrU32(static_cast<uint32_t>(curIdx));  // v4: transmitted schedule index
    const uint64_t tsMs = static_cast<uint64_t>(msNow());
    sendBytes(&tsMs, sizeof(tsMs));
    sendBytes(payload.data(), payload.size());
    flushFrame();
    if (!sendOk) {
      LOGI("receiver closed, stopping");
      break;
    }
    const double sendMs = msNow() - tSend0;
    // TCP-only local congestion signal: a blocking send() means the socket
    // buffer is full. SRT never blocks here (TLPKTDROP drops instead).
    if (fd >= 0 && sendMs > 33.0) rc.congested(static_cast<uint32_t>(sendMs));
    rc.tick();

    refFeature.swap(feature);
    refFrame.swap(xHat);
    out1[0] = feature.data();
    out2[0] = xHat.data();
    ++curIdx;

    statBytes += payload.size();
    busyMs += msNow() - t0;
    ++f;
    gHud.statFrames.store(f);
    gHud.statQ.store(rc.q);
    if (f % 60 == 0) {
      const double wall = msNow() - statT0;
      const double fps = 60000.0 / wall, kbps = statBytes * 8.0 / wall;
      gHud.statFps10.store(static_cast<int>(std::lround(fps * 10.0)));
      gHud.statKbps.store(static_cast<int>(std::lround(kbps)));
      LOGI("f=%d q=%d fps=%.1f busy=%.1fms/frame kbps=%.0f", f, rc.q, fps, busyMs / 60.0, kbps);
      statT0 = msNow();
      busyMs = 0;
      statBytes = 0;
    }
  }
  if (srt != SRT_INVALID_SOCK) {
    mlvc::srtDrainSend(srt);  // closing early discards the in-flight tail
    srt_close(srt);
    mlvc::srtCleanup();
  } else {
    shutdown(fd, SHUT_WR);
    char drain[256];
    while (recv(fd, drain, sizeof(drain), 0) > 0) {}
    close(fd);
  }
  LOGI("encode loop ended after %d frames", f);
}

// One physical camera the device exposes (front, back main, back ultrawide,
// back tele, ...) plus what's needed to reopen a session on it.
struct CamInfo {
  std::string id;
  uint8_t facing = ACAMERA_LENS_FACING_BACK;
  int32_t sensorOrientation = 90;  // degrees, sensor-native vs device "natural" orientation
  // ACAMERA_CONTROL_ZOOM_RATIO_RANGE. On multi-lens phones the OEM fuses
  // wide/ultrawide/tele into ONE logical camera id and picks the physical
  // lens internally based on zoom ratio - min<1.0 means an ultrawide is
  // fused in, max>1.0 means a tele is. Default [1,1] (tag unsupported or
  // single-lens) makes zoom cycling below a no-op.
  float zoomMin = 1.0f;
  float zoomMax = 1.0f;
};

// Owns the camera device/session and can tear down + reopen on a different
// physical camera without restarting the process or the encoder thread -
// tap the preview to cycle. One AImageReader is reused across switches since
// the capture format/size never changes (the NPU model is a fixed 1280x720).
struct CameraRig {
  ACameraManager* mgr = nullptr;
  AImageReader* reader = nullptr;
  ANativeWindow* readerWin = nullptr;
  ANativeWindow* previewWindow = nullptr;  // set once from app->window
  std::vector<CamInfo> cams;
  int current = -1;
  std::vector<float> zoomPresets{1.0f};  // rebuilt per-camera in open(), see below
  int zoomIdx = 0;

  ACameraDevice* device = nullptr;
  ACameraCaptureSession* session = nullptr;
  ACaptureRequest* request = nullptr;
  ACameraOutputTarget* readerTarget = nullptr;
  ACameraOutputTarget* previewTarget = nullptr;
  ACaptureSessionOutputContainer* outputs = nullptr;
  ACaptureSessionOutput* readerOutput = nullptr;
  ACaptureSessionOutput* previewOutput = nullptr;

  void enumerate() {
    ACameraIdList* ids = nullptr;
    ACameraManager_getCameraIdList(mgr, &ids);
    for (int i = 0; i < ids->numCameras; ++i) {
      ACameraMetadata* meta = nullptr;
      ACameraManager_getCameraCharacteristics(mgr, ids->cameraIds[i], &meta);
      CamInfo info;
      info.id = ids->cameraIds[i];
      ACameraMetadata_const_entry e{};
      if (ACameraMetadata_getConstEntry(meta, ACAMERA_LENS_FACING, &e) == ACAMERA_OK)
        info.facing = e.data.u8[0];
      if (ACameraMetadata_getConstEntry(meta, ACAMERA_SENSOR_ORIENTATION, &e) == ACAMERA_OK)
        info.sensorOrientation = e.data.i32[0];
      if (ACameraMetadata_getConstEntry(meta, ACAMERA_CONTROL_ZOOM_RATIO_RANGE, &e) == ACAMERA_OK &&
          e.count >= 2) {
        info.zoomMin = e.data.f[0];
        info.zoomMax = e.data.f[1];
      }
      ACameraMetadata_free(meta);
      cams.push_back(info);
      LOGI("camera[%d] id=%s facing=%d sensorOrient=%d zoom=[%.2f,%.2f]", i, info.id.c_str(),
           info.facing, info.sensorOrientation, info.zoomMin, info.zoomMax);
    }
    ACameraManager_deleteCameraIdList(ids);
  }

  void closeSession() {
    if (session) { ACameraCaptureSession_close(session); session = nullptr; }
    if (request) { ACaptureRequest_free(request); request = nullptr; }
    if (readerTarget) { ACameraOutputTarget_free(readerTarget); readerTarget = nullptr; }
    if (previewTarget) { ACameraOutputTarget_free(previewTarget); previewTarget = nullptr; }
    if (readerOutput) { ACaptureSessionOutput_free(readerOutput); readerOutput = nullptr; }
    if (previewOutput) { ACaptureSessionOutput_free(previewOutput); previewOutput = nullptr; }
    if (outputs) { ACaptureSessionOutputContainer_free(outputs); outputs = nullptr; }
    if (device) { ACameraDevice_close(device); device = nullptr; }
  }

  bool open(int index) {
    if (index < 0 || index >= static_cast<int>(cams.size())) return false;
    closeSession();
    const CamInfo& info = cams[static_cast<size_t>(index)];

    static ACameraDevice_StateCallbacks devCbs = {
        nullptr,
        [](void*, ACameraDevice*) { LOGE("camera disconnected"); },
        [](void*, ACameraDevice*, int err) { LOGE("camera error %d", err); },
    };
    if (ACameraManager_openCamera(mgr, info.id.c_str(), &devCbs, &device) != ACAMERA_OK) {
      LOGE("openCamera(%s) failed", info.id.c_str());
      return false;
    }

    ACaptureSessionOutputContainer_create(&outputs);
    ACaptureSessionOutput_create(readerWin, &readerOutput);
    ACaptureSessionOutputContainer_add(outputs, readerOutput);
    if (previewWindow) {
      ACaptureSessionOutput_create(previewWindow, &previewOutput);
      ACaptureSessionOutputContainer_add(outputs, previewOutput);
    }
    static ACameraCaptureSession_stateCallbacks sessCbs = {
        nullptr,
        [](void*, ACameraCaptureSession*) { LOGI("session closed"); },
        [](void*, ACameraCaptureSession*) {},
        [](void*, ACameraCaptureSession*) {},
    };
    if (ACameraDevice_createCaptureSession(device, outputs, &sessCbs, &session) != ACAMERA_OK) {
      LOGE("createCaptureSession failed");
      return false;
    }

    ACameraDevice_createCaptureRequest(device, TEMPLATE_RECORD, &request);
    // TEMPLATE_RECORD does not guarantee 3A is enabled, and without it the
    // sensor keeps default fixed exposure/WB: measured luma averaged 54/255
    // (range 11-115) in a normally lit room, which reads as a dark,
    // colourless picture no codec setting can recover.
    {
      const uint8_t aeMode = ACAMERA_CONTROL_AE_MODE_ON;
      ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_AE_MODE, 1, &aeMode);
      const uint8_t awbMode = ACAMERA_CONTROL_AWB_MODE_AUTO;
      ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_AWB_MODE, 1, &awbMode);
      const uint8_t afMode = ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO;
      ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_AF_MODE, 1, &afMode);
      const uint8_t ctrlMode = ACAMERA_CONTROL_MODE_AUTO;
      ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_MODE, 1, &ctrlMode);
      // Cap the AE frame-rate range so it cannot buy brightness by dropping
      // to 10fps in dim light - this is a 30fps live encoder.
      const int32_t fpsRange[2] = {24, 30};
      ACaptureRequest_setEntry_i32(request, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE, 2, fpsRange);
      const uint8_t antiBand = ACAMERA_CONTROL_AE_ANTIBANDING_MODE_50HZ;  // India mains
      ACaptureRequest_setEntry_u8(request, ACAMERA_CONTROL_AE_ANTIBANDING_MODE, 1, &antiBand);
    }
    // Zoom-ratio presets for this camera's fused lens range - candidates that
    // fall inside [zoomMin, zoomMax] become tap-to-cycle stops (two-finger
    // tap, see onInput). 1.0x is always included so the default view is the
    // primary wide lens.
    {
      zoomPresets.clear();
      for (float z : {info.zoomMin, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, info.zoomMax}) {
        if (z >= info.zoomMin && z <= info.zoomMax) zoomPresets.push_back(z);
      }
      std::sort(zoomPresets.begin(), zoomPresets.end());
      zoomPresets.erase(std::unique(zoomPresets.begin(), zoomPresets.end(),
                                     [](float a, float b) { return std::fabs(a - b) < 0.01f; }),
                         zoomPresets.end());
      if (zoomPresets.empty()) zoomPresets.push_back(1.0f);
      zoomIdx = 0;
      for (size_t i = 0; i < zoomPresets.size(); ++i) {
        if (std::fabs(zoomPresets[i] - 1.0f) < 0.01f) { zoomIdx = static_cast<int>(i); break; }
      }
      const float z0 = zoomPresets[static_cast<size_t>(zoomIdx)];
      ACaptureRequest_setEntry_float(request, ACAMERA_CONTROL_ZOOM_RATIO, 1, &z0);
    }
    ACameraOutputTarget_create(readerWin, &readerTarget);
    ACaptureRequest_addTarget(request, readerTarget);
    if (previewWindow) {
      ACameraOutputTarget_create(previewWindow, &previewTarget);
      ACaptureRequest_addTarget(request, previewTarget);
    }
    ACameraCaptureSession_setRepeatingRequest(session, nullptr, 1, &request, nullptr);
    current = index;
    LOGI("using camera %s (facing=%d) [%d/%zu]", info.id.c_str(), info.facing, index,
         cams.size());
    return true;
  }

  void cycle() { open((current + 1) % static_cast<int>(cams.size())); }

  // Cycles the zoom-ratio preset on the CURRENT camera - no session/device
  // teardown needed, just mutate the standing request and resubmit. This is
  // how a fused-lens camera (e.g. wide+ultrawide+tele under one logical id)
  // switches physical lens on this device.
  void cycleZoom() {
    if (!request || !session || zoomPresets.size() <= 1) return;
    zoomIdx = (zoomIdx + 1) % static_cast<int>(zoomPresets.size());
    const float z = zoomPresets[static_cast<size_t>(zoomIdx)];
    ACaptureRequest_setEntry_float(request, ACAMERA_CONTROL_ZOOM_RATIO, 1, &z);
    ACameraCaptureSession_setRepeatingRequest(session, nullptr, 1, &request, nullptr);
    LOGI("zoom -> %.2fx [%d/%zu]", z, zoomIdx, zoomPresets.size());
  }
};

CameraRig gRig;


// One tap on the camera-flip button: next camera (cycles front/back
// logical ids - "whatever cams I have"). One tap on the lens button: next
// zoom/lens preset on the CURRENT camera (cycles fused wide/ultrawide/tele
// lenses where the device has them). Taps outside both buttons do nothing.
int32_t onInput(android_app*, AInputEvent* event) {
  if (AInputEvent_getType(event) != AINPUT_EVENT_TYPE_MOTION) return 0;
  const int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
  if (action != AMOTION_EVENT_ACTION_DOWN) return 0;
  const int hit = gHud.hitTest(AMotionEvent_getX(event, 0), AMotionEvent_getY(event, 0));
  if (hit == 1) {
    gRig.cycle();
    gHud.redraw(gRig.current, gRig.zoomPresets[static_cast<size_t>(gRig.zoomIdx)]);
    return 1;
  }
  if (hit == 2) {
    gRig.cycleZoom();
    gHud.redraw(gRig.current, gRig.zoomPresets[static_cast<size_t>(gRig.zoomIdx)]);
    return 1;
  }
  return 0;
}

}  // namespace

void android_main(android_app* app) {
  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);

  const std::string target = intentExtra(app, env, "target");
  const std::string qs = intentExtra(app, env, "q");
  const std::string qmins = intentExtra(app, env, "qmin");
  const std::string qmaxs = intentExtra(app, env, "qmax");
  const int q0 = qs.empty() ? 63 : atoi(qs.c_str());
  const int qMin = qmins.empty() ? 21 : atoi(qmins.c_str());
  const int qMax = qmaxs.empty() ? 63 : atoi(qmaxs.c_str());
  const bool useSrt = intentExtra(app, env, "srt") == "1";
  const std::string srtLatS = intentExtra(app, env, "srt_latency");
  const int srtLatencyMs = srtLatS.empty() ? 200 : atoi(srtLatS.c_str());
  const std::string mode = intentExtra(app, env, "mode");
  const std::string backendLib = [&] {
    const std::string b = intentExtra(app, env, "backend");
    return b == "cpu" ? "libQnnCpu.so" : "libQnnHtp.so";
  }();
  // Launched straight from the home-screen icon (no intent extras at all)
  // used to hard-exit here - the app only ever ran as an adb-launched dev
  // tool. A real, installable app has to at least show its camera and be
  // usable when tapped cold, so a missing target now just means "skip
  // streaming, run as a standalone viewfinder" instead of quitting.
  const bool hasStreamTarget = !target.empty() || mode == "bridge";
  if (!hasStreamTarget) {
    LOGI("no -e target given - running standalone (camera/HUD only, no streaming)");
  }

  // Samsung's camera HAL kills ImageReader-only sessions (~35s, error 4);
  // a visible preview surface keeps it alive, so wait for the window and add
  // it as a second output target. Keep-screen-on stops lock-screen teardown.
  ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
  while (!app->window && !app->destroyRequested) {
    android_poll_source* src = nullptr;
    if (ALooper_pollOnce(100, nullptr, nullptr, reinterpret_cast<void**>(&src)) >= 0 && src) {
      src->process(app, src);
    }
  }
  LOGI("window ready %p", (void*)app->window);

  // fastrpc resolves the DSP skel through this env var at first use. The
  // unsigned skel packaged in our APK's native-lib dir DOES load on this
  // retail S24U (unsigned-PD offload is permitted for apps here) - the
  // earlier in-app failure was never the skel path or SELinux at all, but
  // libcdsprpc.so failing to resolve in the app's classloader namespace
  // (fixed via <uses-native-library> in the manifest; the lib is on
  // /vendor/etc/public.libraries.txt, apps just must opt in since API 31).
  const std::string adspOverride = intentExtra(app, env, "adsp_path");
  setenv("ADSP_LIBRARY_PATH",
         adspOverride.empty() ? nativeLibDir(app, env).c_str() : adspOverride.c_str(), 1);

  const std::string enc1Path = materializeAsset(app, "s24_720p_enc1_v75.bin");
  const std::string enc2Path = materializeAsset(app, "s24_720p_enc2_v75.bin");
  const std::string pmfPath = materializeAsset(app, "pmf_tables.bin");

  // Receiver: no camera, no HUD - this instance owns the whole window and
  // spends it on the decoded picture. See decode_loop.cpp.
  if (mode == "receive") {
    const std::string decPath = materializeAsset(app, "s24_720p_dec_v75.bin");
    const std::string portS = intentExtra(app, env, "listen_port");
    const int listenPort = portS.empty() ? 8720 : atoi(portS.c_str());
    const std::string bindAddr = intentExtra(app, env, "bind");
    mlvc::decodeLoop(app, listenPort, decPath, pmfPath, backendLib, srtLatencyMs, bindAddr);
    return;
  }

  // --- camera ---
  gRig.mgr = ACameraManager_create();
  gRig.previewWindow = app->window;
  AImageReader_new(kWidth, kHeight, AIMAGE_FORMAT_YUV_420_888, 4, &gRig.reader);
  AImageReader_ImageListener listener{nullptr, onImage};
  AImageReader_setImageListener(gRig.reader, &listener);
  AImageReader_getWindow(gRig.reader, &gRig.readerWin);

  gRig.enumerate();
  if (gRig.cams.empty()) { LOGE("no cameras"); return; }
  // Initial pick: "front"/"back" (first match), or a numeric index into the
  // enumerated list logged above. Tap the preview to cycle at runtime.
  {
    const std::string want = intentExtra(app, env, "camera");
    int start = 0;
    if (want == "front" || want == "back") {
      const uint8_t f = want == "front" ? ACAMERA_LENS_FACING_FRONT : ACAMERA_LENS_FACING_BACK;
      for (size_t i = 0; i < gRig.cams.size(); ++i)
        if (gRig.cams[i].facing == f) { start = static_cast<int>(i); break; }
    } else if (!want.empty()) {
      start = atoi(want.c_str()) % static_cast<int>(gRig.cams.size());
    }
    if (!gRig.open(start)) { LOGE("initial camera open failed"); return; }
  }
  app->onInputEvent = onInput;
  LOGI("camera streaming at %dx%d (tap camera/lens buttons to switch)", kWidth, kHeight);
  gOrient.init(app);
  const bool hudOk = gHud.init(app);
  if (hudOk) {
    gHud.redraw(gRig.current, gRig.zoomPresets[static_cast<size_t>(gRig.zoomIdx)]);
  } else {
    LOGE("hud init failed - camera/lens switching falls back to no on-screen buttons");
  }

  const std::string bridgePortS = intentExtra(app, env, "bridge_port");
  const uint16_t bridgePort =
      bridgePortS.empty() ? 8901 : static_cast<uint16_t>(atoi(bridgePortS.c_str()));
  std::thread enc;
  if (hasStreamTarget) {
    enc = std::thread([&] {
      if (mode == "bridge") {
        const std::string dumpPath = std::string(app->activity->internalDataPath) + "/dump.i420";
        bridgeLoop(bridgePort, dumpPath);
      } else {
        encodeLoop(app, target, q0, qMin, qMax, enc1Path, enc2Path, pmfPath, backendLib,
                   useSrt, srtLatencyMs);
      }
    });
  }

  // Pump lifecycle events until the activity is destroyed. LOOPER_ID_USER is
  // the orientation sensor queue (registered in gOrient.init); everything
  // else comes back as a glue-owned android_poll_source.
  int lastPreviewBucket = -1;
  int hudTick = 0;
  while (!app->destroyRequested) {
    android_poll_source* source = nullptr;
    const int ident = ALooper_pollOnce(200, nullptr, nullptr, reinterpret_cast<void**>(&source));
    if (ident == LOOPER_ID_USER) {
      gOrient.drain();
    } else if (ident >= 0 && source) {
      source->process(app, source);
    }
    // Stats (bitrate/fps/quality/frame count) refresh here on a plain
    // timer; camera/lens button taps redraw immediately in onInput for
    // responsiveness instead of waiting on this tick.
    if (hudOk && ++hudTick >= 3) {  // ~600ms at the 200ms poll timeout above
      hudTick = 0;
      gHud.redraw(gRig.current, gRig.zoomPresets[static_cast<size_t>(gRig.zoomIdx)]);
    }
    const int bucket = gOrient.bucket.load();
    if (bucket != lastPreviewBucket && gRig.previewWindow) {
      // Rotate ONLY the on-screen preview to match; the camera HAL still
      // writes unrotated sensor-orientation buffers into it (this is a
      // SurfaceFlinger composition hint, not a pixel touch), separate from
      // the encode-side tensor rotation above.
      const int32_t transform = bucket == 90    ? ANATIVEWINDOW_TRANSFORM_ROTATE_90
                                 : bucket == 180 ? ANATIVEWINDOW_TRANSFORM_ROTATE_180
                                 : bucket == 270 ? ANATIVEWINDOW_TRANSFORM_ROTATE_270
                                                  : ANATIVEWINDOW_TRANSFORM_IDENTITY;
      ANativeWindow_setBuffersTransform(gRig.previewWindow, transform);
      lastPreviewBucket = bucket;
    }
  }
  gOrient.shutdown();
  gSlot.stop.store(true);
  gSlot.cv.notify_all();
  if (gRig.session) ACameraCaptureSession_stopRepeating(gRig.session);
  gRig.closeSession();
  AImageReader_delete(gRig.reader);
  ACameraManager_delete(gRig.mgr);
  if (enc.joinable()) enc.join();
}
