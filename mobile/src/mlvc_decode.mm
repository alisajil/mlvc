// mlvc_decode: M2 Mac receiver. Reads an .mlvb bitstream produced by the phone
// encoder, entropy-decodes latents (shared rans/mlvc_codec code), runs the
// CoreML MLVCDecoder on the Apple Neural Engine, and reports PSNR against the
// original clip plus per-stage timing. Optionally writes reconstructed YUV420.
//
// Usage:
//   mlvc_decode --bitstream clip.mlvb --model MLVCDecoder.mlmodelc \
//               --pmf pmf_tables.bin [--ref original.yuv] [--out recon.yuv]

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <pthread.h>

#include "mlvc_codec.h"
#include "net_util.h"
#include "srt_transport.h"
#include "stream_feedback.h"

namespace {

// Reads the .mlvb byte stream from a file or a live socket.
struct Source {
  SRTSOCKET srt = SRT_INVALID_SOCK;
  std::vector<uint8_t> msg;   // current reassembled frame
  size_t msgPos = 0;
  bool frameLost = false;     // set when SRT abandoned chunks of a frame
  std::ifstream file;
  int fd = -1;
  bool ok = true;
  bool read(void* data, size_t n) {
    if (srt != SRT_INVALID_SOCK) {
      while (msgPos + n > msg.size()) {   // need the next message
        uint32_t idx = 0;
        bool lost = false;
        std::vector<uint8_t> next;
        if (!mlvc::srtRecvFrame(srt, next, idx, lost)) return ok = false;
        if (lost) frameLost = true;
        if (next.empty()) continue;       // gap report, no payload
        msg = std::move(next);
        msgPos = 0;
      }
      memcpy(data, msg.data() + msgPos, n);
      msgPos += n;
      return ok = true;
    }
    if (fd >= 0) return ok = mlvc::recvAll(fd, data, n);
    file.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
    return ok = static_cast<bool>(file);
  }
  uint32_t u32() { uint32_t v = 0; read(&v, 4); return v; }
};

double msNow() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

float fp16ToF32(uint16_t h) { return (float)*(__fp16*)&h; }

// PSNR of fp16 plane vs 8-bit source plane (float [0,1] domain).
double psnrPlane(const uint16_t* rec, const uint8_t* src, size_t n, int strideRec,
                 int strideSrc, int w, int rows) {
  double mse = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < w; ++c) {
      float d = fp16ToF32(rec[(size_t)r * strideRec + c]) - src[(size_t)r * strideSrc + c] / 255.0f;
      mse += (double)d * d;
    }
  }
  mse /= (double)w * rows;
  (void)n;
  return mse > 0 ? 10.0 * log10(1.0 / mse) : 99.0;
}

}  // namespace

int main(int argc, char** argv) {
  // A live receiver must not be starved by desktop load; ask for performance
  // cores rather than accepting the default utility QoS.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered: a live tool must
                                        // report progress as it happens
  std::string bsPath, modelPath, pmfPath, refPath, outPath;
  int listenPort = 0;
  bool computeAll = false;
  bool bindAll = false;
  bool verbose = false;
  bool useSrt = false;
  int srtLatencyMs = 200;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--bitstream") bsPath = next();
    else if (a == "--model") modelPath = next();
    else if (a == "--pmf") pmfPath = next();
    else if (a == "--ref") refPath = next();
    else if (a == "--out") outPath = next();
    else if (a == "--listen") listenPort = atoi(next());
    else if (a == "--compute-all") computeAll = true;
    else if (a == "--public") bindAll = true;
    else if (a == "--verbose") verbose = true;
    else if (a == "--srt") useSrt = true;
    else if (a == "--srt-latency") srtLatencyMs = atoi(next());
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if ((bsPath.empty() && listenPort == 0) || modelPath.empty() || pmfPath.empty()) {
    fprintf(stderr, "usage: mlvc_decode --bitstream x.mlvb --model y.mlmodelc --pmf pmf.bin\n");
    return 2;
  }

  // --- container header ---
  Source bs;
  if (listenPort > 0) {
    printf("listening on %s:%d ...\n", bindAll ? "0.0.0.0" : "127.0.0.1", listenPort);
    fflush(stdout);
    if (useSrt) {
      mlvc::srtStartup();
      printf("SRT listening on :%d (latency budget %d ms)\n", listenPort, srtLatencyMs);
      fflush(stdout);
      bs.srt = mlvc::srtAcceptOne((uint16_t)listenPort, srtLatencyMs);
      if (bs.srt == SRT_INVALID_SOCK) { fprintf(stderr, "srt listen failed\n"); return 1; }
    } else {
      bs.fd = mlvc::tcpAcceptOne((uint16_t)listenPort, bindAll);
      if (bs.fd < 0) { fprintf(stderr, "listen failed\n"); return 1; }
    }
    printf("sender connected\n");
  } else {
    bs.file.open(bsPath, std::ios::binary);
    if (!bs.file) { fprintf(stderr, "cannot open %s\n", bsPath.c_str()); return 1; }
  }
  auto rdU32 = [&]() { return bs.u32(); };
  if (rdU32() != 0x424C564Du) { fprintf(stderr, "bad magic\n"); return 1; }
  const uint32_t version = rdU32();
  if (version != 1 && version != 2 && version != 3) {
    fprintf(stderr, "unsupported container version %u\n", version);
    return 1;
  }
  const int width = (int)rdU32(), height = (int)rdU32();
  const int qIndex = (int)rdU32(), frames = (int)rdU32();
  if (!bs.ok) { fprintf(stderr, "short header\n"); return 1; }
  // Untrusted input once the port is forwarded: reject anything not matching
  // the model this binary was built to decode.
  if (width != 1280 || height != 720 || qIndex < 0 || qIndex >= 72 ||
      frames < 0 || frames > 100000) {
    fprintf(stderr, "rejected header: %dx%d q=%d frames=%d\n", width, height, qIndex, frames);
    return 1;
  }
  const bool live = frames == 0;  // v2 live stream: run until the sender closes
  printf("bitstream: %dx%d q=%d frames=%s\n", width, height, qIndex,
         live ? "live" : std::to_string(frames).c_str());

  mlvc::ModelDims dims;
  dims.width = width; dims.height = height;
  mlvc::PmfTables tables;
  if (!tables.load(pmfPath)) { fprintf(stderr, "pmf load failed\n"); return 1; }

  // --- CoreML model ---
  // No Xcode on this machine, so compile .mlpackage -> .mlmodelc at runtime
  // via the framework API directly (coremlcompiler is just a CLI wrapper
  // around this same call).
  NSError* err = nil;
  NSURL* modelUrl = [NSURL fileURLWithPath:@(modelPath.c_str())];
  const bool isPackage =
      modelPath.size() >= 10 && modelPath.compare(modelPath.size() - 10, 10, ".mlpackage") == 0;
  if (isPackage) {
    // Compiling costs ~1s, so keep the result next to the package and reuse it.
    NSString* cached = [@(modelPath.c_str()) stringByReplacingOccurrencesOfString:@".mlpackage"
                                                                      withString:@".mlmodelc"];
    NSFileManager* fm = NSFileManager.defaultManager;
    if ([fm fileExistsAtPath:cached]) {
      modelUrl = [NSURL fileURLWithPath:cached];
    } else {
      NSURL* compiledUrl = [MLModel compileModelAtURL:modelUrl error:&err];
      if (!compiledUrl) {
        fprintf(stderr, "compile: %s\n", err.description.UTF8String);
        return 1;
      }
      NSURL* cachedUrl = [NSURL fileURLWithPath:cached];
      modelUrl = [fm moveItemAtURL:compiledUrl toURL:cachedUrl error:nil] ? cachedUrl : compiledUrl;
      printf("compiled -> %s\n", modelUrl.path.UTF8String);
    }
  }
  MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
  // ANE + CPU only: MLComputeUnitsAll lets CoreML schedule onto the GPU, which
  // contends with desktop rendering and made decode latency swing 16->40ms.
  cfg.computeUnits = computeAll ? MLComputeUnitsAll : MLComputeUnitsCPUAndNeuralEngine;
  MLModel* model = [MLModel modelWithContentsOfURL:modelUrl configuration:cfg error:&err];
  if (!model) { fprintf(stderr, "model load: %s\n", err.description.UTF8String); return 1; }

  const size_t hw = (size_t)width * height;
  const size_t featN = (size_t)dims.featureCh * (height / 8) * (width / 8);
  const size_t zN = (size_t)dims.zCh * dims.zH * dims.zW;
  const size_t yHalfN = (size_t)(dims.latentCh / 2) * dims.yH * dims.yW;

  std::vector<uint16_t> zRaw(zN), yRaw0(yHalfN), yRaw1(yHalfN);
  std::vector<uint16_t> refFrame(hw * 3, 0x3800), refFeature(featN, 0);
  std::vector<uint16_t> xHat(hw * 3), feature(featN);
  int32_t qShifted = 0;
  uint16_t refExists = 0;

  auto arr = [&](void* data, NSArray<NSNumber*>* shape, MLMultiArrayDataType dt,
                 size_t elemBytes, size_t n) -> MLMultiArray* {
    NSMutableArray* strides = [NSMutableArray array];
    size_t s = n;
    for (NSNumber* dim in shape) { s /= dim.unsignedLongValue; [strides addObject:@(s)]; }
    (void)elemBytes;
    return [[MLMultiArray alloc] initWithDataPointer:data shape:shape dataType:dt
                                             strides:strides deallocator:nil error:nil];
  };
  auto fp16Arr = [&](std::vector<uint16_t>& v, NSArray<NSNumber*>* shape) {
    return arr(v.data(), shape, MLMultiArrayDataTypeFloat16, 2, v.size());
  };

  MLMultiArray* mZ = fp16Arr(zRaw, @[@1, @(dims.zCh), @(dims.zH), @(dims.zW)]);
  MLMultiArray* mY0 = fp16Arr(yRaw0, @[@1, @(dims.latentCh / 2), @(dims.yH), @(dims.yW)]);
  MLMultiArray* mY1 = fp16Arr(yRaw1, @[@1, @(dims.latentCh / 2), @(dims.yH), @(dims.yW)]);
  MLMultiArray* mRefF = fp16Arr(refFrame, @[@1, @3, @(height), @(width)]);
  MLMultiArray* mRefFeat = fp16Arr(refFeature, @[@1, @(dims.featureCh), @(height / 8), @(width / 8)]);
  MLMultiArray* mRefEx = arr(&refExists, @[@1], MLMultiArrayDataTypeFloat16, 2, 1);
  MLMultiArray* mQ = arr(&qShifted, @[@1], MLMultiArrayDataTypeInt32, 4, 1);

  // --- reference clip for PSNR ---
  std::ifstream ref;
  std::vector<uint8_t> srcFrame(hw * 3 / 2);
  const bool havePsnr = !refPath.empty();
  if (havePsnr) {
    ref.open(refPath, std::ios::binary);
    if (!ref) { fprintf(stderr, "cannot open %s\n", refPath.c_str()); return 1; }
  }
  std::ofstream outYuv;
  if (!outPath.empty()) outYuv.open(outPath, std::ios::binary);

  mlvc::CoderWorkspace ws;
  double sumRans = 0, sumNpu = 0, sumPsnr = 0;
  double sumWait = 0, sumGap = 0, maxGap = 0, lastArrival = 0;
  int starvedFrames = 0, qMinSeen = 999, qMaxSeen = -1;
  double minDelta = 1e18, maxDelta = -1e18, sumDelta = 0;
  int lostFrames = 0;
  long qSum = 0;
  double psnrMin = 1e9;
  std::vector<uint8_t> payload;

  int decoded = 0;
  int curIdx = 0;  // mirrors the encoder's cur_frame_idx
  for (int f = 0; live || f < frames; ++f) {
    const double tWait0 = msNow();
    const uint32_t sz = rdU32();
    if (live && !bs.ok) { printf("stream ended by sender\n"); break; }
    if (!bs.ok || sz < 8 || sz > (1u << 22)) {  // 4 MB ceiling; 720p frames are ~7 KB
      fprintf(stderr, "rejected frame %d payload size %u\n", f, sz);
      return 1;
    }
    // v2 carries q_index per frame, so the sender can change quality mid-stream.
    int qFrame = qIndex;
    if (version >= 2) {
      qFrame = (int)rdU32();
      if (!bs.ok || qFrame < 0 || qFrame >= 72) {
        fprintf(stderr, "rejected frame %d q_index %d\n", f, qFrame);
        return 1;
      }
    }
    uint64_t sendTsMs = 0;
    if (version >= 3) bs.read(&sendTsMs, sizeof(sendTsMs));
    payload.resize(sz);
    if (!bs.read(payload.data(), sz)) { fprintf(stderr, "truncated at frame %d\n", f); return 1; }
    if (version >= 3) {
      const double delta = msNow() - (double)sendTsMs;  // unsynced clocks: use spread
      if (delta < minDelta) minDelta = delta;
      if (delta > maxDelta) maxDelta = delta;
      sumDelta += delta;
    }
    const double t0 = msNow();
    sumWait += t0 - tWait0;
    if (f > 0) {
      const double gap = t0 - lastArrival;
      sumGap += gap;
      if (gap > maxGap) maxGap = gap;
    }
    lastArrival = t0;
    if (bs.fd >= 0 || bs.srt != SRT_INVALID_SOCK) {
      // Receiver starvation is the congestion signal: a healthy link leaves
      // this near zero, a saturated one leaves the decoder waiting.
      const double waitMs = t0 - tWait0;
      mlvc::Feedback fb{};
      fb.frameIdx = (uint32_t)f;
      fb.waitMs = (uint32_t)waitMs;
      fb.verdict = waitMs > mlvc::kStarvedMs ? -1 : (waitMs < mlvc::kHealthyMs ? 1 : 0);
      if (bs.frameLost) { fb.needIframe = 1; bs.frameLost = false; ++lostFrames; }
      if (bs.srt != SRT_INVALID_SOCK) srt_send(bs.srt, (const char*)&fb, sizeof(fb));
      else mlvc::sendNonBlocking(bs.fd, &fb, sizeof(fb));
      if (fb.verdict < 0) ++starvedFrames;
      qSum += qFrame;
      if (qFrame < qMinSeen) qMinSeen = qFrame;
      if (qFrame > qMaxSeen) qMaxSeen = qFrame;
    }
    if (!mlvc::decodeLatents(tables, dims, payload.data(), payload.size(), qFrame,
                             zRaw.data(), yRaw0.data(), yRaw1.data(), ws)) {
      fprintf(stderr, "frame %d: corrupt bitstream\n", f);
      return 1;
    }
    // Must mirror the encoder's schedule exactly, or references diverge.
    if (mlvc::isIframe(curIdx)) {
      curIdx = 0;
      std::fill(refFrame.begin(), refFrame.end(), 0x3800);
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else if (mlvc::isFeatureReset(curIdx)) {
      std::fill(refFeature.begin(), refFeature.end(), 0);
      refExists = 0;
    } else {
      refExists = 0x3C00;
    }
    qShifted = qFrame + mlvc::qpShift(curIdx);
    if (getenv("MLVC_QLOG")) fprintf(stderr, "DEC f=%d i=%d q=%d qs=%d\n", f, curIdx, qFrame, qShifted);
    const double t1 = msNow();

    NSDictionary* feats = @{
      @"z_raw" : mZ, @"y_raw_0" : mY0, @"y_raw_1" : mY1, @"ref_frame" : mRefF,
      @"ref_feature" : mRefFeat, @"ref_exists" : mRefEx, @"q_index_shifted" : mQ,
    };
    MLDictionaryFeatureProvider* in =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feats error:&err];
    id<MLFeatureProvider> outFeats = [model predictionFromFeatures:in error:&err];
    if (!outFeats) { fprintf(stderr, "predict: %s\n", err.description.UTF8String); return 1; }
    const double t2 = msNow();

    MLMultiArray* oX = [outFeats featureValueForName:@"x_hat"].multiArrayValue;
    MLMultiArray* oF = [outFeats featureValueForName:@"feature"].multiArrayValue;
    memcpy(xHat.data(), oX.dataPointer, hw * 3 * 2);
    memcpy(feature.data(), oF.dataPointer, featN * 2);

    // reference loop
    memcpy(refFrame.data(), xHat.data(), hw * 3 * 2);
    memcpy(refFeature.data(), feature.data(), featN * 2);

    if (havePsnr) {
      ref.read((char*)srcFrame.data(), (std::streamsize)srcFrame.size());
      // Y at full res; U/V: compare 2x2-mean-downsampled recon vs source.
      double py = psnrPlane(xHat.data(), srcFrame.data(), hw, width, width, width, height);
      double mseU = 0, mseV = 0;
      const uint8_t* srcU = srcFrame.data() + hw;
      const uint8_t* srcV = srcFrame.data() + hw + hw / 4;
      for (int r = 0; r < height / 2; ++r) {
        for (int c = 0; c < width / 2; ++c) {
          auto down = [&](const uint16_t* plane) {
            const size_t p = (size_t)(2 * r) * width + 2 * c;
            return 0.25f * (fp16ToF32(plane[p]) + fp16ToF32(plane[p + 1]) +
                            fp16ToF32(plane[p + width]) + fp16ToF32(plane[p + width + 1]));
          };
          float du = down(xHat.data() + hw) - srcU[(size_t)r * (width / 2) + c] / 255.0f;
          float dv = down(xHat.data() + 2 * hw) - srcV[(size_t)r * (width / 2) + c] / 255.0f;
          mseU += (double)du * du;
          mseV += (double)dv * dv;
        }
      }
      mseU /= (double)(hw / 4); mseV /= (double)(hw / 4);
      double pu = 10 * log10(1.0 / mseU), pv = 10 * log10(1.0 / mseV);
      double p = (6 * py + pu + pv) / 8.0;
      sumPsnr += p;
      if (p < psnrMin) psnrMin = p;
    }
    if (outYuv.is_open()) {
      // fp16 [0,1] -> 8-bit YUV420 (Y full res, U/V 2x2 mean)
      std::vector<uint8_t> o(hw * 3 / 2);
      for (size_t i = 0; i < hw; ++i) {
        float v = fp16ToF32(xHat[i]) * 255.0f + 0.5f;
        o[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
      }
      size_t w2 = 0;
      for (int pl = 1; pl <= 2; ++pl) {
        const uint16_t* plane = xHat.data() + (size_t)pl * hw;
        for (int r = 0; r < height / 2; ++r) {
          for (int c = 0; c < width / 2; ++c) {
            const size_t p = (size_t)(2 * r) * width + 2 * c;
            float m = 0.25f * (fp16ToF32(plane[p]) + fp16ToF32(plane[p + 1]) +
                               fp16ToF32(plane[p + width]) + fp16ToF32(plane[p + width + 1]));
            float v = m * 255.0f + 0.5f;
            o[hw + w2++] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
          }
        }
      }
      outYuv.write((char*)o.data(), (std::streamsize)o.size());
    }
    if (verbose) {
      printf("  frame %3d q=%2d %6u B wait=%6.1f ms decode=%5.1f ms\n", f, qFrame, sz,
             t0 - tWait0, t2 - t1);
    }
    ++curIdx;
    sumRans += t1 - t0;
    sumNpu += t2 - t1;
    ++decoded;
  }
  const int statFrames = decoded > 0 ? decoded : 1;

  printf("decoded %d frames\n", decoded);
  printf("decode ms/frame: rans=%.2f coreml=%.2f total=%.2f (%.1f fps)\n",
         sumRans / statFrames, sumNpu / statFrames, (sumRans + sumNpu) / statFrames,
         1000.0 / ((sumRans + sumNpu) / statFrames));
  if (havePsnr) printf("PSNR (6:1:1): mean=%.2f dB min=%.2f dB\n", sumPsnr / statFrames, psnrMin);
  if (bs.fd >= 0) {
    printf("live: wait-for-frame=%.2f ms avg | inter-frame gap avg=%.2f ms max=%.2f ms "
           "(%.1f fps sustained)\n",
           sumWait / statFrames, sumGap / std::max(1, statFrames - 1), maxGap,
           1000.0 / (sumGap / std::max(1, statFrames - 1)));
    if (version >= 3 && decoded > 0) {
      printf("latency: queueing delay above best case: avg=%.0f ms peak=%.0f ms "
             "(clock offset removed)\n",
             sumDelta / statFrames - minDelta, maxDelta - minDelta);
    }
    if (lostFrames) printf("loss: %d frames incomplete, I-frame recovery requested\n", lostFrames);
    printf("adaptive: q ranged %d..%d (mean %.1f) | %d/%d frames starved\n",
           qMinSeen, qMaxSeen, (double)qSum / statFrames, starvedFrames, decoded);
  }
  return 0;
}
