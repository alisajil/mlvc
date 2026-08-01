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
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mlvc_codec.h"
#include "net_util.h"
#include "qnn_runner.h"
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

// Same AIMD controller as mlvc_encode.cpp (kept local: the CLI and the app
// are separate binaries and this is 25 lines).
struct RateController {
  int q, qMin, qMax;
  int cleanRun = 0, raiseStep = 2;
  static constexpr int kRaiseAfter = 10, kRaiseMax = 8;
  void congested(uint32_t waitMs) {
    q = std::max(qMin, q - std::clamp(static_cast<int>(waitMs / 15), 4, 16));
    cleanRun = 0;
    raiseStep = 2;
  }
  void healthy() {
    if (++cleanRun >= kRaiseAfter) {
      q = std::min(qMax, q + raiseStep);
      raiseStep = std::min(kRaiseMax, raiseStep * 2);
      cleanRun = 0;
    }
  }
  void hold() { cleanRun = 0; }
};

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

// Bridge mode: no QNN in-app (blocked by the app sandbox) — forward raw I420
// frames over localhost TCP to the shell-domain CLI, which owns the NPU.
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

// Reads a string extra from the launching intent (JNI: NativeActivity has no
// native accessor for extras).
std::string intentExtra(android_app* app, JNIEnv* env, const char* name) {
  jobject activity = app->activity->clazz;
  jclass actCls = env->GetObjectClass(activity);
  jmethodID getIntent = env->GetMethodID(actCls, "getIntent", "()Landroid/content/Intent;");
  jobject intent = env->CallObjectMethod(activity, getIntent);
  if (!intent) return "";
  jclass intCls = env->GetObjectClass(intent);
  jmethodID getStr =
      env->GetMethodID(intCls, "getStringExtra", "(Ljava/lang/String;)Ljava/lang/String;");
  jstring key = env->NewStringUTF(name);
  auto val = static_cast<jstring>(env->CallObjectMethod(intent, getStr, key));
  if (!val) return "";
  const char* c = env->GetStringUTFChars(val, nullptr);
  std::string s(c);
  env->ReleaseStringUTFChars(val, c);
  return s;
}

std::string nativeLibDir(android_app* app, JNIEnv* env) {
  jobject activity = app->activity->clazz;
  jclass actCls = env->GetObjectClass(activity);
  jmethodID getAppInfo =
      env->GetMethodID(actCls, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
  jobject info = env->CallObjectMethod(activity, getAppInfo);
  jfieldID fid =
      env->GetFieldID(env->GetObjectClass(info), "nativeLibraryDir", "Ljava/lang/String;");
  auto dir = static_cast<jstring>(env->GetObjectField(info, fid));
  const char* c = env->GetStringUTFChars(dir, nullptr);
  std::string s(c);
  env->ReleaseStringUTFChars(dir, c);
  return s;
}

// Copies an APK asset to the app's private files dir; returns the path.
std::string materializeAsset(android_app* app, const char* name) {
  const std::string path = std::string(app->activity->internalDataPath) + "/" + name;
  AAsset* a = AAssetManager_open(app->activity->assetManager, name, AASSET_MODE_STREAMING);
  if (!a) { LOGE("asset %s missing", name); return ""; }
  const off_t want = AAsset_getLength(a);
  FILE* existing = fopen(path.c_str(), "rb");
  if (existing) {
    fseek(existing, 0, SEEK_END);
    const long have = ftell(existing);
    fclose(existing);
    if (have == want) { AAsset_close(a); return path; }
  }
  FILE* f = fopen(path.c_str(), "wb");
  char buf[65536];
  int n = 0;
  while ((n = AAsset_read(a, buf, sizeof(buf))) > 0) fwrite(buf, 1, static_cast<size_t>(n), f);
  fclose(f);
  AAsset_close(a);
  return path;
}

void encodeLoop(android_app* app, const std::string& target, int q0, int qMin, int qMax,
                const std::string& enc1Path, const std::string& enc2Path,
                const std::string& pmfPath, const std::string& backendLib) {
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
  const int fd = mlvc::tcpConnect(host, port);
  if (fd < 0) { LOGE("connect %s failed", target.c_str()); return; }
  LOGI("streaming to %s", target.c_str());

  mlvc::ModelDims dims;
  const size_t hw = static_cast<size_t>(kWidth) * kHeight;
  const size_t featN = static_cast<size_t>(dims.featureCh) * (kHeight / 8) * (kWidth / 8);
  const size_t zN = static_cast<size_t>(dims.zCh) * dims.zH * dims.zW;
  const size_t yHalfN = static_cast<size_t>(dims.latentCh / 2) * dims.yH * dims.yW;
  std::vector<uint16_t> x(hw * 3), refFrame(hw * 3, 0x3800), refFeature(featN, 0);
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

  auto wrU32 = [&](uint32_t v) { return mlvc::sendAll(fd, &v, 4); };
  wrU32(0x424C564Du);
  wrU32(2);
  wrU32(kWidth);
  wrU32(kHeight);
  wrU32(static_cast<uint32_t>(q0));
  wrU32(0);  // frames=0: live stream, until socket closes

  RateController rc{q0, qMin, qMax};
  mlvc::CoderWorkspace ws;
  ws.enc.reserve(zN + 2 * yHalfN);
  int f = 0;
  double statT0 = msNow(), busyMs = 0;
  size_t statBytes = 0;

  while (!gSlot.stop.load()) {
    AImage* img = gSlot.take();
    if (!img) break;
    const double t0 = msNow();
    image888ToTensor(img, x.data());
    AImage_delete(img);

    mlvc::Feedback fb{}, latest{};
    bool got = false;
    while (mlvc::recvNonBlocking(fd, &fb, sizeof(fb))) {
      if (fb.magic == mlvc::kFeedbackMagic) { latest = fb; got = true; }
    }
    if (got) {
      if (latest.verdict < 0) rc.congested(latest.waitMs);
      else if (latest.verdict > 0) rc.healthy();
      else rc.hold();
    }
    const int qFrame = rc.q;
    qShifted = qFrame + mlvc::qpShift(f);
    refExists = (f == 0) ? 0 : 0x3C00;

    in1[orderOf(io1, "x")] = x.data();
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
    if (!wrU32(static_cast<uint32_t>(payload.size())) ||
        !wrU32(static_cast<uint32_t>(qFrame)) ||
        !mlvc::sendAll(fd, payload.data(), payload.size())) {
      LOGI("receiver closed, stopping");
      break;
    }
    const double sendMs = msNow() - tSend0;
    if (sendMs > 33.0) rc.congested(static_cast<uint32_t>(sendMs));

    refFeature.swap(feature);
    refFrame.swap(xHat);
    out1[0] = feature.data();
    out2[0] = xHat.data();

    statBytes += payload.size();
    busyMs += msNow() - t0;
    ++f;
    if (f % 60 == 0) {
      const double wall = msNow() - statT0;
      LOGI("f=%d q=%d fps=%.1f busy=%.1fms/frame kbps=%.0f", f, rc.q, 60000.0 / wall,
           busyMs / 60.0, statBytes * 8.0 / wall);
      statT0 = msNow();
      busyMs = 0;
      statBytes = 0;
    }
  }
  shutdown(fd, SHUT_WR);
  char drain[256];
  while (recv(fd, drain, sizeof(drain), 0) > 0) {}
  close(fd);
  LOGI("encode loop ended after %d frames", f);
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
  if (target.empty() && intentExtra(app, env, "mode") != "bridge") {
    LOGE("no -e target given; exiting");
    ANativeActivity_finish(app->activity);
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

  // fastrpc resolves the DSP skel through this env var at first use.
  // The skel shipped in the QAIRT SDK's hexagon-v75/unsigned/ folder is only
  // trusted by the DSP loader from allow-listed paths (e.g. /data/local/tmp
  // on this device) on a retail build, NOT from an app's private native-lib
  // dir — shipping this for real needs Qualcomm's signed skel or a testsig.
  // ponytail: hardcoded dev path, replace with signed skel before any real release.
  const std::string adspOverride = intentExtra(app, env, "adsp_path");
  setenv("ADSP_LIBRARY_PATH",
         adspOverride.empty() ? nativeLibDir(app, env).c_str() : adspOverride.c_str(), 1);

  const std::string enc1Path = materializeAsset(app, "s24_720p_enc1_v75.bin");
  const std::string enc2Path = materializeAsset(app, "s24_720p_enc2_v75.bin");
  const std::string pmfPath = materializeAsset(app, "pmf_tables.bin");

  // --- camera ---
  const bool wantFront = intentExtra(app, env, "camera") == "front";
  ACameraManager* mgr = ACameraManager_create();
  ACameraIdList* ids = nullptr;
  ACameraManager_getCameraIdList(mgr, &ids);
  std::string camId;
  const uint8_t wantFacing = wantFront ? ACAMERA_LENS_FACING_FRONT : ACAMERA_LENS_FACING_BACK;
  for (int i = 0; i < ids->numCameras; ++i) {
    ACameraMetadata* meta = nullptr;
    ACameraManager_getCameraCharacteristics(mgr, ids->cameraIds[i], &meta);
    ACameraMetadata_const_entry facing{};
    if (ACameraMetadata_getConstEntry(meta, ACAMERA_LENS_FACING, &facing) == ACAMERA_OK &&
        facing.data.u8[0] == wantFacing) {
      camId = ids->cameraIds[i];
      ACameraMetadata_free(meta);
      break;
    }
    ACameraMetadata_free(meta);
  }
  LOGI("using camera %s (%s)", camId.c_str(), wantFront ? "front" : "back");

  AImageReader* reader = nullptr;
  AImageReader_new(kWidth, kHeight, AIMAGE_FORMAT_YUV_420_888, 4, &reader);
  AImageReader_ImageListener listener{nullptr, onImage};
  AImageReader_setImageListener(reader, &listener);
  ANativeWindow* readerWin = nullptr;
  AImageReader_getWindow(reader, &readerWin);

  ACameraDevice* device = nullptr;
  ACameraDevice_StateCallbacks devCbs{};
  devCbs.onDisconnected = [](void*, ACameraDevice*) { LOGE("camera disconnected"); };
  devCbs.onError = [](void*, ACameraDevice*, int err) { LOGE("camera error %d", err); };
  if (ACameraManager_openCamera(mgr, camId.c_str(), &devCbs, &device) != ACAMERA_OK) {
    LOGE("openCamera failed (permission granted?)");
    return;
  }

  ACaptureSessionOutputContainer* outputs = nullptr;
  ACaptureSessionOutputContainer_create(&outputs);
  ACaptureSessionOutput* output = nullptr;
  ACaptureSessionOutput_create(readerWin, &output);
  ACaptureSessionOutputContainer_add(outputs, output);
  ACaptureSessionOutput* previewOutput = nullptr;
  if (app->window) {
    ACaptureSessionOutput_create(app->window, &previewOutput);
    ACaptureSessionOutputContainer_add(outputs, previewOutput);
  }
  ACameraCaptureSession* session = nullptr;
  ACameraCaptureSession_stateCallbacks sessCbs{};
  sessCbs.onClosed = [](void*, ACameraCaptureSession*) { LOGI("session closed"); };
  sessCbs.onReady = [](void*, ACameraCaptureSession*) {};
  sessCbs.onActive = [](void*, ACameraCaptureSession*) {};
  if (ACameraDevice_createCaptureSession(device, outputs, &sessCbs, &session) != ACAMERA_OK) {
    LOGE("createCaptureSession failed");
    return;
  }
  ACaptureRequest* request = nullptr;
  ACameraDevice_createCaptureRequest(device, TEMPLATE_RECORD, &request);
  ACameraOutputTarget* target_ = nullptr;
  ACameraOutputTarget_create(readerWin, &target_);
  ACaptureRequest_addTarget(request, target_);
  ACameraOutputTarget* previewTarget = nullptr;
  if (app->window) {
    ACameraOutputTarget_create(app->window, &previewTarget);
    ACaptureRequest_addTarget(request, previewTarget);
  }
  ACameraCaptureSession_setRepeatingRequest(session, nullptr, 1, &request, nullptr);
  LOGI("camera streaming at %dx%d", kWidth, kHeight);

  const std::string mode = intentExtra(app, env, "mode");
  const std::string bridgePortS = intentExtra(app, env, "bridge_port");
  const uint16_t bridgePort =
      bridgePortS.empty() ? 8901 : static_cast<uint16_t>(atoi(bridgePortS.c_str()));
  const std::string backendLib = [&] {
    const std::string b = intentExtra(app, env, "backend");
    return b == "cpu" ? "libQnnCpu.so" : "libQnnHtp.so";
  }();
  std::thread enc([&] {
    if (mode == "bridge") {
      const std::string dumpPath = std::string(app->activity->internalDataPath) + "/dump.i420";
      bridgeLoop(bridgePort, dumpPath);
    } else {
      encodeLoop(app, target, q0, qMin, qMax, enc1Path, enc2Path, pmfPath, backendLib);
    }
  });

  // Pump lifecycle events until the activity is destroyed.
  while (!app->destroyRequested) {
    android_poll_source* source = nullptr;
    if (ALooper_pollOnce(200, nullptr, nullptr, reinterpret_cast<void**>(&source)) >= 0 &&
        source) {
      source->process(app, source);
    }
  }
  gSlot.stop.store(true);
  gSlot.cv.notify_all();
  ACameraCaptureSession_stopRepeating(session);
  ACameraDevice_close(device);
  ACameraManager_delete(mgr);
  enc.join();
}
