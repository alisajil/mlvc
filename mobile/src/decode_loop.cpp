#include "decode_loop.h"

#include <android/log.h>
#include <android/native_window.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "mlvc_codec.h"
#include "qnn_runner.h"
#include "srt_transport.h"
#include "stream_feedback.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MLVC", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MLVC", __VA_ARGS__)

namespace mlvc {
namespace {

double msNow() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

size_t orderOf(const std::vector<TensorDesc>& t, const char* n) {
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i].name == n) return i;
  return 0;
}

// The compiled decoder graph keeps its ONNX input names (z_raw, ref_frame,
// ...) but renames both outputs generically (output_0/output_1) - byte size
// disambiguates them unambiguously instead: x_hat is hw*3 fp16 elements,
// feature is featN fp16 elements, and the two never collide in practice.
size_t orderOfSize(const std::vector<TensorDesc>& t, size_t byteSize) {
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i].byteSize == byteSize) return i;
  return 0;
}

// fp16 [1,3,H,W] x_hat (Y,U,V planes, chroma already full-res in the tensor -
// no need to redo the encoder's 2x2 chroma averaging, that step exists only
// to produce a real 4:2:0 file format) -> RGBA_8888 into a locked window
// buffer. Standard BT.601 full-range YUV->RGB, fixed-point (matches the
// no-bilinear-anywhere convention already used for chroma upsampling
// elsewhere in this codec).
void xhatToWindow(const uint16_t* xhat, int w, int h, const ANativeWindow_Buffer& buf) {
  const size_t hw = static_cast<size_t>(w) * h;
  auto* dst = static_cast<uint32_t*>(buf.bits);
  auto f2b = [](uint16_t h16) -> int {
    __fp16 f;
    memcpy(&f, &h16, 2);
    const float v = static_cast<float>(f) * 255.0f + 0.5f;
    return v < 0 ? 0 : v > 255 ? 255 : static_cast<int>(v);
  };
  auto clamp8 = [](int x) { return x < 0 ? 0 : x > 255 ? 255 : x; };
  for (int r = 0; r < h; ++r) {
    uint32_t* row = dst + static_cast<size_t>(r) * static_cast<size_t>(buf.stride);
    for (int c = 0; c < w; ++c) {
      const size_t idx = static_cast<size_t>(r) * w + c;
      const int y = f2b(xhat[idx]);
      const int u = f2b(xhat[hw + idx]) - 128;
      const int v = f2b(xhat[2 * hw + idx]) - 128;
      const int rr = clamp8(y + ((91881 * v) >> 16));
      const int gg = clamp8(y - ((22554 * u + 46802 * v) >> 16));
      const int bb = clamp8(y + ((116130 * u) >> 16));
      row[c] = (0xFFu << 24) | (static_cast<uint32_t>(bb) << 16) |
               (static_cast<uint32_t>(gg) << 8) | static_cast<uint32_t>(rr);
    }
  }
}

}  // namespace

void decodeLoop(android_app* app, int listenPort, const std::string& decPath,
                const std::string& pmfPath, const std::string& backendLib,
                int srtLatencyMs, const std::string& bindAddr) {
  PmfTables tables;
  if (!tables.load(pmfPath)) { LOGE("pmf load failed"); return; }

  QnnRunner dec;
  LOGI("dec diag: backend=%s", backendLib.c_str());
  if (!dec.init(backendLib, "libQnnSystem.so", decPath, true)) {
    LOGE("dec init: %s", dec.error().c_str());
    return;
  }
  LOGI("dec init SUCCEEDED with backend=%s", backendLib.c_str());
  const std::string gName = dec.graphs()[0].graphName;
  const auto& io = dec.graphs()[0];
  for (size_t i = 0; i < io.inputs.size(); ++i) {
    const auto& t = io.inputs[i];
    LOGI("dec input[%zu]: %s dims=%s dtype=%u bytes=%zu", i, t.name.c_str(),
         [&] { std::string s; for (auto d : t.dims) s += std::to_string(d) + ","; return s; }()
             .c_str(),
         t.dataType, t.byteSize);
  }
  for (size_t i = 0; i < io.outputs.size(); ++i) {
    const auto& t = io.outputs[i];
    LOGI("dec output[%zu]: %s dims=%s dtype=%u bytes=%zu", i, t.name.c_str(),
         [&] { std::string s; for (auto d : t.dims) s += std::to_string(d) + ","; return s; }()
             .c_str(),
         t.dataType, t.byteSize);
  }

  srtStartup();
  LOGI("listening for sender on [%s]:%d ...", bindAddr.empty() ? "any" : bindAddr.c_str(),
       listenPort);
  SRTSOCKET srt = srtAcceptOne(static_cast<uint16_t>(listenPort), srtLatencyMs, bindAddr);
  if (srt == SRT_INVALID_SOCK) { LOGE("srt listen failed"); return; }
  LOGI("sender connected");

  std::vector<uint8_t> hdrBuf;
  if (!srtRecvHeader(srt, hdrBuf) || hdrBuf.size() < 24) {
    LOGE("srt header receive failed");
    return;
  }
  auto u32At = [&](size_t off) {
    uint32_t v;
    memcpy(&v, hdrBuf.data() + off, 4);
    return v;
  };
  const uint32_t magic = u32At(0), version = u32At(4);
  const int width = static_cast<int>(u32At(8)), height = static_cast<int>(u32At(12));
  if (magic != 0x424C564Du || version != 4) {
    LOGE("bad/unsupported header: magic=0x%08x version=%u (this receiver is v4-only)", magic,
         version);
    return;
  }
  if (width != 1280 || height != 720) {
    LOGE("unsupported resolution %dx%d (this build's decoder model is 720p-only)", width, height);
    return;
  }
  LOGI("bitstream: %dx%d live", width, height);

  ANativeWindow_setBuffersGeometry(app->window, width, height, WINDOW_FORMAT_RGBA_8888);

  ModelDims dims;
  const size_t hw = static_cast<size_t>(width) * height;
  const size_t featN = static_cast<size_t>(dims.featureCh) * (height / 8) * (width / 8);
  const size_t zN = static_cast<size_t>(dims.zCh) * dims.zH * dims.zW;
  const size_t yHalfN = static_cast<size_t>(dims.latentCh / 2) * dims.yH * dims.yW;

  std::vector<uint16_t> zRaw(zN), yRaw0(yHalfN), yRaw1(yHalfN);
  std::vector<uint16_t> refFrame(hw * 3, 0x3800), refFeature(featN, 0);
  std::vector<uint16_t> xHat(hw * 3), feature(featN);
  int32_t qShifted = 0;
  uint16_t refExists = 0;

  std::vector<void*> in(io.inputs.size());
  std::vector<void*> out(io.outputs.size());
  out[orderOfSize(io.outputs, hw * 3 * sizeof(uint16_t))] = xHat.data();
  out[orderOfSize(io.outputs, featN * sizeof(uint16_t))] = feature.data();

  CoderWorkspace ws;
  std::vector<uint8_t> msg, payload;
  int curIdx = 0;
  int decoded = 0;
  double statT0 = msNow();
  int statFrames = 0;

  for (;;) {
    const double tWait0 = msNow();
    uint32_t frameIdx = 0;
    bool lost = false;
    if (!srtRecvFrame(srt, msg, frameIdx, lost)) {
      LOGI("sender disconnected after %d frames", decoded);
      break;
    }
    if (msg.size() < 20) continue;  // gap report only, no payload to decode

    size_t off = 0;
    auto rd32 = [&]() {
      uint32_t v;
      memcpy(&v, msg.data() + off, 4);
      off += 4;
      return v;
    };
    const uint32_t sz = rd32();
    const int qFrame = static_cast<int>(rd32());
    curIdx = static_cast<int>(rd32());
    uint64_t sendTsMs = 0;
    memcpy(&sendTsMs, msg.data() + off, 8);
    off += 8;
    if (off + sz > msg.size()) {
      LOGE("truncated frame %u (have %zu, want %zu)", frameIdx, msg.size(), off + sz);
      continue;
    }
    payload.assign(msg.begin() + static_cast<long>(off), msg.begin() + static_cast<long>(off + sz));

    const double t0 = msNow();
    // Receiver -> sender congestion feedback, same protocol the Mac decoder
    // and encodeLoop already speak (stream_feedback.h) - keeps this receiver
    // interchangeable with either sender.
    Feedback fb{};
    fb.frameIdx = frameIdx;
    fb.waitMs = static_cast<uint32_t>(t0 - tWait0);
    fb.verdict = fb.waitMs > kStarvedMs ? -1 : (fb.waitMs < kHealthyMs ? 1 : 0);
    if (lost) fb.needIframe = 1;
    srt_send(srt, reinterpret_cast<const char*>(&fb), sizeof(fb));

    if (!decodeLatents(tables, dims, payload.data(), payload.size(), qFrame, zRaw.data(),
                       yRaw0.data(), yRaw1.data(), ws)) {
      LOGE("frame %u: corrupt bitstream", frameIdx);
      continue;
    }
    // Must mirror the sender's schedule exactly (mlvc_codec.h) or references
    // diverge - encoder and decoder run on different silicon with nothing
    // else resetting them in sync.
    if (isIframe(curIdx)) {
      curIdx = 0;
      std::fill(refFrame.begin(), refFrame.end(), 0x3800);
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else if (isFeatureReset(curIdx)) {
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else {
      refExists = 0x3C00;
    }
    qShifted = qFrame + qpShift(curIdx);

    in[orderOf(io.inputs, "z_raw")] = zRaw.data();
    in[orderOf(io.inputs, "y_raw_0")] = yRaw0.data();
    in[orderOf(io.inputs, "y_raw_1")] = yRaw1.data();
    in[orderOf(io.inputs, "ref_frame")] = refFrame.data();
    in[orderOf(io.inputs, "ref_feature")] = refFeature.data();
    in[orderOf(io.inputs, "ref_exists")] = &refExists;
    in[orderOf(io.inputs, "q_index_shifted")] = &qShifted;
    if (!dec.execute(gName, in, out)) {
      LOGE("dec: %s", dec.error().c_str());
      continue;
    }
    refFrame = xHat;
    refFeature = feature;
    ++curIdx;

    ANativeWindow_Buffer winBuf;
    if (ANativeWindow_lock(app->window, &winBuf, nullptr) == 0) {
      xhatToWindow(xHat.data(), width, height, winBuf);
      ANativeWindow_unlockAndPost(app->window);
    }

    ++decoded;
    if (++statFrames >= 60) {
      const double wall = msNow() - statT0;
      LOGI("decoded=%d fps=%.1f", decoded, statFrames * 1000.0 / wall);
      statT0 = msNow();
      statFrames = 0;
    }
  }

  srtDrainSend(srt);
  srt_close(srt);
  srtCleanup();
}

}  // namespace mlvc
