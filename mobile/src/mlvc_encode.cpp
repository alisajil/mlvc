// mlvc_encode: M1 offline encoder. Reads a raw YUV420p clip, runs the MLVC-S
// encoder graphs on the HTP NPU, entropy-codes the latents with rANS, and
// writes a bitstream file. Reports per-stage timing and bpp.
//
// Bitstream container (little-endian):
//   'MLVB' u32 | version u32 | width u32 | height u32 | qIndex u32 | frames u32
//   per frame: payloadSize u32 | rANS payload bytes
//
// Usage:
//   mlvc_encode --yuv in.yuv --width 1280 --height 720 --frames 48 --q 63 \
//               --enc1 enc1.bin --enc2 enc2.bin --pmf pmf.bin --out clip.mlvb \
//               [--verify] [--no-burst]

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "mlvc_codec.h"
#include "net_util.h"
#include "srt_transport.h"
#include "qnn_runner.h"
#include "stream_feedback.h"

namespace {

// Writes the .mlvb byte stream to a file, a socket, or both.
struct Sink {
  SRTSOCKET srt = SRT_INVALID_SOCK;
  uint32_t srtFrameIdx = 0;
  std::vector<uint8_t> srtBuf;   // one frame accumulated, then sent chunked
  std::ofstream file;
  int fd = -1;
  bool ok = true;
  void write(const void* data, size_t n) {
    if (file.is_open()) file.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (srt != SRT_INVALID_SOCK) {  // message transport: buffer, send per frame
      const auto* p = static_cast<const uint8_t*>(data);
      srtBuf.insert(srtBuf.end(), p, p + n);
      return;
    }
    if (fd >= 0 && !mlvc::sendAll(fd, data, n)) ok = false;
  }
  void u32(uint32_t v) { write(&v, 4); }
  // SRT is message-oriented: flush the accumulated frame as chunks.
  void flushFrame() {
    if (srt == SRT_INVALID_SOCK || srtBuf.empty()) return;
    if (!mlvc::srtSendFrame(srt, srtFrameIdx++, srtBuf.data(), srtBuf.size())) ok = false;
    srtBuf.clear();
  }
};

// Pins the calling thread to the big cores (4-7 on SM8650-class parts) so the
// scheduler cannot park hot loops on LITTLE cores while the NPU runs.
void pinBigCores() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int c = 4; c < 8; ++c) CPU_SET(c, &mask);
  sched_setaffinity(gettid(), sizeof(mask), &mask);
}

double msNow() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Stats {
  double sum = 0, mx = 0;
  int n = 0;
  void add(double v) { sum += v; if (v > mx) mx = v; ++n; }
  double avg() const { return n ? sum / n : 0; }
};

}  // namespace

int main(int argc, char** argv) {
  pinBigCores();
  std::string yuvPath, enc1Path, enc2Path, pmfPath, outPath, streamTarget;
  bool useSrt = false;
  int srtLatencyMs = 200;
  int yuvListenPort = 0;  // live camera bridge: raw I420 frames over localhost
  std::string backend = "libQnnHtp.so", system = "libQnnSystem.so";
  int width = 1280, height = 720, frames = 0, qIndex = 63;
  bool verify = false, burst = true, adaptive = false, loop = false;
  int qMin = 21, qMax = 63;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--yuv") yuvPath = next();
    else if (a == "--yuv-listen") yuvListenPort = atoi(next());
    else if (a == "--srt") useSrt = true;
    else if (a == "--srt-latency") srtLatencyMs = atoi(next());
    else if (a == "--width") width = atoi(next());
    else if (a == "--height") height = atoi(next());
    else if (a == "--frames") frames = atoi(next());
    else if (a == "--q") qIndex = atoi(next());
    else if (a == "--enc1") enc1Path = next();
    else if (a == "--enc2") enc2Path = next();
    else if (a == "--pmf") pmfPath = next();
    else if (a == "--out") outPath = next();
    else if (a == "--stream") streamTarget = next();
    else if (a == "--adaptive") adaptive = true;
    else if (a == "--loop") loop = true;
    else if (a == "--q-min") qMin = atoi(next());
    else if (a == "--q-max") qMax = atoi(next());
    else if (a == "--verify") verify = true;
    else if (a == "--no-burst") burst = false;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  const bool liveInput = yuvListenPort > 0;
  if ((yuvPath.empty() && !liveInput) || enc1Path.empty() || enc2Path.empty() ||
      pmfPath.empty() || (outPath.empty() && streamTarget.empty()) ||
      (frames <= 0 && !liveInput)) {
    fprintf(stderr, "missing required args\n");
    return 2;
  }

  mlvc::ModelDims dims;
  dims.width = width;
  dims.height = height;

  mlvc::PmfTables tables;
  if (!tables.load(pmfPath)) { fprintf(stderr, "pmf load failed\n"); return 1; }

  mlvc::QnnRunner enc1, enc2;
  if (!enc1.init(backend, system, enc1Path, burst)) {
    fprintf(stderr, "enc1 init: %s\n", enc1.error().c_str());
    return 1;
  }
  if (!enc2.init(backend, system, enc2Path, /*burstMode=*/false)) {
    fprintf(stderr, "enc2 init: %s\n", enc2.error().c_str());
    return 1;
  }
  const std::string g1 = enc1.graphs()[0].graphName;
  const std::string g2 = enc2.graphs()[0].graphName;

  const size_t hw = static_cast<size_t>(width) * height;
  const size_t frameBytes = hw * 3 / 2;  // YUV420p 8-bit
  std::ifstream yuv;
  int bridgeFd = -1;
  if (liveInput) {
    printf("waiting for camera bridge on 127.0.0.1:%d ...\n", yuvListenPort);
    fflush(stdout);
    bridgeFd = mlvc::tcpAcceptOne(static_cast<uint16_t>(yuvListenPort), false);
    if (bridgeFd < 0) { fprintf(stderr, "bridge listen failed\n"); return 1; }
    uint32_t hdr[3] = {0, 0, 0};
    if (!mlvc::recvAll(bridgeFd, hdr, sizeof(hdr)) || hdr[0] != 0x30323449u ||
        static_cast<int>(hdr[1]) != width || static_cast<int>(hdr[2]) != height) {
      fprintf(stderr, "bad bridge header\n");
      return 1;
    }
    printf("camera bridge connected (%ux%u)\n", hdr[1], hdr[2]);
    frames = 0;  // live: run until the camera stops
  } else {
    yuv.open(yuvPath, std::ios::binary);
    if (!yuv) { fprintf(stderr, "cannot open %s\n", yuvPath.c_str()); return 1; }
  }

  // Buffers. Graph tensor order (from context introspection):
  // enc1 in:  x, ref_frame, q_index_shifted, ref_feature, ref_exists
  // enc1 out: feature, z_raw(output_1), y_raw_0(output_2), y_raw_1(output_3)
  // enc2 in:  feature, q_index_shifted   out: x_hat
  const size_t featN = static_cast<size_t>(dims.featureCh) * (height / 8) * (width / 8);
  const size_t zN = static_cast<size_t>(dims.zCh) * dims.zH * dims.zW;
  const size_t yHalfN = static_cast<size_t>(dims.latentCh / 2) * dims.yH * dims.yW;
  std::vector<uint16_t> x(hw * 3), refFrame(hw * 3), refFeature(featN);
  std::vector<uint16_t> feature(featN), zRaw(zN), yRaw0(yHalfN), yRaw1(yHalfN);
  std::vector<uint16_t> xHat(hw * 3);
  int32_t qShifted = 0;
  uint16_t refExists = 0;

  // First-frame references: gray frame (0.5), zero feature. 0x3800 == fp16 0.5.
  for (auto& v : refFrame) v = 0x3800;
  memset(refFeature.data(), 0, refFeature.size() * 2);

  Sink out;
  if (!outPath.empty()) out.file.open(outPath, std::ios::binary);
  if (!streamTarget.empty()) {
    std::string host;
    uint16_t port = 0;
    if (!mlvc::splitHostPort(streamTarget, host, port)) {
      fprintf(stderr, "bad --stream target %s (use host:port or [v6::addr]:port)\n",
              streamTarget.c_str());
      return 1;
    }
    if (useSrt) {
      mlvc::srtStartup();
      if (getenv("MLVC_SRT_DEBUG")) {
        srt_setloglevel(LOG_DEBUG);
        srt_setlogflags(0);
        static auto handler = [](void*, int level, const char* file, int line,
                                 const char* area, const char* msg) {
          fprintf(stderr, "[srt %d] %s:%d %s: %s\n", level, file, line, area, msg);
        };
        srt_setloghandler(nullptr, +handler);
      }
      out.srt = mlvc::srtConnect(host, port, srtLatencyMs);
      out.fd = -1;
      if (out.srt == SRT_INVALID_SOCK) {
        fprintf(stderr, "srt connect to %s failed\n", streamTarget.c_str());
        return 1;
      }
      printf("streaming to %s over SRT (latency budget %d ms)\n",
             streamTarget.c_str(), srtLatencyMs);
    } else
    out.fd = mlvc::tcpConnect(host, port);
    if (!useSrt && out.fd < 0) {
      fprintf(stderr, "connect to %s failed (receiver listening? router IPv6 "
                      "firewall may block inbound port %u)\n", streamTarget.c_str(), port);
      return 1;
    }
    printf("streaming to %s\n", streamTarget.c_str());
  }
  out.u32(0x424C564Du); out.u32(3);  // v3: per-frame q_index + send timestamp
  out.u32(static_cast<uint32_t>(width)); out.u32(static_cast<uint32_t>(height));
  out.u32(static_cast<uint32_t>(qIndex));
  out.u32(liveInput ? 0u : static_cast<uint32_t>(frames));

  mlvc::RateController rc;
  rc.q = qIndex; rc.qMin = qMin; rc.qMax = qMax;
  int qFloorSeen = qIndex, qCeilSeen = qIndex, adaptations = 0, iframesForced = 0;

  std::vector<uint8_t> frameBuf(frameBytes);
  mlvc::CoderWorkspace ws;
  ws.enc.reserve(zN + 2 * yHalfN);
  Stats tPre, tNpu1, tNpu2, tRans, tTotal, tSendStat;
  size_t totalPayload = 0;
  bool allVerified = true;

  // Graph input order discovered at runtime by name; build index maps once.
  auto orderOf = [](const mlvc::GraphIo& g, const char* n) -> int {
    for (size_t i = 0; i < g.inputs.size(); ++i)
      if (g.inputs[i].name == n) return static_cast<int>(i);
    return -1;
  };
  const auto& io1 = enc1.graphs()[0];
  std::vector<void*> in1(io1.inputs.size());
  in1[static_cast<size_t>(orderOf(io1, "x"))] = x.data();
  in1[static_cast<size_t>(orderOf(io1, "ref_frame"))] = refFrame.data();
  in1[static_cast<size_t>(orderOf(io1, "ref_feature"))] = refFeature.data();
  in1[static_cast<size_t>(orderOf(io1, "q_index_shifted"))] = &qShifted;
  in1[static_cast<size_t>(orderOf(io1, "ref_exists"))] = &refExists;
  std::vector<void*> out1 = {feature.data(), zRaw.data(), yRaw0.data(), yRaw1.data()};
  const auto& io2 = enc2.graphs()[0];
  std::vector<void*> in2(io2.inputs.size());
  in2[static_cast<size_t>(orderOf(io2, "feature"))] = feature.data();
  in2[static_cast<size_t>(orderOf(io2, "q_index_shifted"))] = &qShifted;
  std::vector<void*> out2 = {xHat.data()};

  // Frame 0 preprocessed up front; frame f+1 is read+converted on a worker
  // thread while frame f runs on the NPU.
  std::vector<uint16_t> xNext(hw * 3);
  long inputDropped = 0;   // camera frames skipped to stay current
  long encodeSkipped = 0;  // frames not encoded because the link was backlogged
  auto readFrameWrapped = [&]() -> bool {
    if (bridgeFd >= 0) {
      if (!mlvc::recvAll(bridgeFd, frameBuf.data(), frameBytes)) return false;
      // Newest-wins on input: whole frames already queued behind this one mean
      // the encoder fell behind the camera, so skip to the freshest.
      while (mlvc::bytesAvailable(bridgeFd) >= static_cast<int>(frameBytes)) {
        if (!mlvc::recvAll(bridgeFd, frameBuf.data(), frameBytes)) return false;
        ++inputDropped;
      }
      return true;
    }
    yuv.read(reinterpret_cast<char*>(frameBuf.data()),
             static_cast<std::streamsize>(frameBytes));
    if (yuv) return true;
    if (!loop) return false;
    yuv.clear();               // EOF: wrap to the start of the clip
    yuv.seekg(0);
    yuv.read(reinterpret_cast<char*>(frameBuf.data()),
             static_cast<std::streamsize>(frameBytes));
    return static_cast<bool>(yuv);
  };
  if (!readFrameWrapped()) { fprintf(stderr, "yuv read failed at frame 0\n"); return 1; }
  mlvc::yuv420ToTensor(frameBuf.data(), width, height, x.data());

  // ~0.35s of video at the 1.3Mbps this link sustains. Large enough that TCP
  // keeps enough in flight for throughput, small enough that a stall is felt
  // within a few frames instead of accumulating seconds.
  // MLVC_NOGUARD=1 disables the latency guard, for A/B measurement only.
  const int kBacklogLimitBytes = getenv("MLVC_NOGUARD") ? (1 << 30) : 64 * 1024;
  bool liveEnded = false;
  int curIdx = 0;  // resets at every I-frame; drives qpShift and the schedule
  for (int f = 0; (liveInput ? !liveEnded : f < frames); ++f) {
    const double t0 = msNow();
    // Drain any feedback that arrived since the last frame; act on the newest.
    if (adaptive && out.fd >= 0) {
      mlvc::Feedback fb{}, latest{};
      bool got = false;
      if (out.srt != SRT_INVALID_SOCK) {
        char fbuf[sizeof(mlvc::Feedback)];
        while (srt_recv(out.srt, fbuf, sizeof(fbuf)) == (int)sizeof(fbuf)) {
          memcpy(&fb, fbuf, sizeof(fb));
          if (fb.magic == mlvc::kFeedbackMagic) { latest = fb; got = true; }
        }
      } else {
        while (mlvc::recvNonBlocking(out.fd, &fb, sizeof(fb))) {
          if (fb.magic == mlvc::kFeedbackMagic) { latest = fb; got = true; }
        }
      }
      if (got && latest.needIframe) {
        // The receiver could not decode a frame. Both reference chains must be
        // re-seeded or every later frame inherits the damage.
        curIdx = 0;
        ++iframesForced;
      }
      if (got) {
        const int before = rc.q;
        if (latest.verdict < 0) rc.maybeCongested(latest.waitMs);
        else if (latest.verdict > 0) rc.healthy();
        else rc.hold();
        if (rc.q != before) {
          ++adaptations;
          qFloorSeen = std::min(qFloorSeen, rc.q);
          qCeilSeen = std::max(qCeilSeen, rc.q);
        }
      }
    }
    // Debug: force a deterministic q swing to test per-frame quality changes
    // without any network in the loop.
    // Bound end-to-end latency. Dropping an INPUT frame is safe: the encoder
    // simply never codes it, so both ends stay on the same reference chain and
    // the stream just runs at a lower frame rate for a moment. Dropping an
    // ENCODED frame would be fatal - the decoder's reference would diverge.
    const bool backlogged =
        liveInput && out.fd >= 0 && mlvc::socketBacklog(out.fd) > kBacklogLimitBytes;
    if (backlogged) {
      ++encodeSkipped;
      if (!readFrameWrapped()) break;
      mlvc::yuv420ToTensor(frameBuf.data(), width, height, x.data());
      --f;  // this camera frame was never coded, so it consumes no frame index
      continue;
    }

    // Reference refresh, identical schedule on the decoder. curIdx restarts at
    // each I-frame, exactly as cur_frame_idx does in the Python frame loop.
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

    int qFrame = adaptive ? rc.q : qIndex;
    if (const char* sw = getenv("MLVC_QSWEEP")) {
      qFrame = (f % 2) ? std::max(12, qIndex - atoi(sw)) : qIndex;
    }
    qShifted = qFrame + mlvc::qpShift(curIdx);
    if (getenv("MLVC_QLOG")) fprintf(stderr, "ENC f=%d i=%d q=%d qs=%d\n", f, curIdx, qFrame, qShifted);

    bool preOk = true;
    std::thread preThread;
    if (liveInput || f + 1 < frames) {
      preThread = std::thread([&] {
        pinBigCores();
        if (!readFrameWrapped()) { preOk = false; return; }
        mlvc::yuv420ToTensor(frameBuf.data(), width, height, xNext.data());
      });
    }
    const double t1 = msNow();

    if (!enc1.execute(g1, in1, out1)) {
      fprintf(stderr, "enc1: %s\n", enc1.error().c_str());
      return 1;
    }
    const double t2 = msNow();

    // enc2 (NPU) and rANS (CPU) touch disjoint buffers — run them in parallel.
    std::vector<uint8_t> payload;
    std::thread ransThread([&] {
      pinBigCores();
      payload = mlvc::encodeBitstream(tables, dims, yRaw0.data(), yRaw1.data(),
                                      zRaw.data(), qFrame, ws);
    });
    bool enc2Ok = enc2.execute(g2, in2, out2);
    const double t3 = msNow();
    ransThread.join();
    if (!enc2Ok) {
      fprintf(stderr, "enc2: %s\n", enc2.error().c_str());
      return 1;
    }
    const double t4 = msNow();

    if (verify && !mlvc::verifyBitstream(tables, dims, payload, yRaw0.data(),
                                         yRaw1.data(), zRaw.data(), qFrame)) {
      fprintf(stderr, "frame %d bitstream verify FAILED\n", f);
      allVerified = false;
    }

    const double tSend0 = msNow();
    out.u32(static_cast<uint32_t>(payload.size()));
    out.u32(static_cast<uint32_t>(qFrame));
    // Sender clock. The two machines are not synchronised, so only the SPREAD
    // of (arrival - send) is meaningful - that spread is exactly the queueing
    // delay this guard exists to bound.
    const uint64_t tsMs = static_cast<uint64_t>(msNow());
    out.write(&tsMs, sizeof(tsMs));
    out.write(payload.data(), payload.size());
    if (!out.ok) { fprintf(stderr, "receiver disconnected at frame %d\n", f); return 1; }
    out.flushFrame();
    const double sendMs = msNow() - tSend0;
    tSendStat.add(sendMs);
    // send() blocking means the socket buffer is full: local congestion signal
    // that arrives sooner than receiver feedback can.
    if (adaptive && out.fd >= 0 && sendMs > 33.0) {
      const int before = rc.q;
      rc.congested(static_cast<uint32_t>(sendMs));
      if (rc.q != before) { ++adaptations; qFloorSeen = std::min(qFloorSeen, rc.q); }
    }
    totalPayload += payload.size();

    // Reference update by pointer swap (no copies), then rebind graph inputs.
    refFeature.swap(feature);
    refFrame.swap(xHat);
    if (preThread.joinable()) {
      preThread.join();
      if (!preOk) {
        if (liveInput) { liveEnded = true; }
        else { fprintf(stderr, "yuv read failed at frame %d\n", f + 1); return 1; }
      } else {
        x.swap(xNext);
      }
    }
    in1[static_cast<size_t>(orderOf(io1, "x"))] = x.data();
    in1[static_cast<size_t>(orderOf(io1, "ref_frame"))] = refFrame.data();
    in1[static_cast<size_t>(orderOf(io1, "ref_feature"))] = refFeature.data();
    out1[0] = feature.data();
    in2[static_cast<size_t>(orderOf(io2, "feature"))] = feature.data();
    out2[0] = xHat.data();

    ++curIdx;
    rc.tick();
    tPre.add(t1 - t0); tNpu1.add(t2 - t1); tNpu2.add(t3 - t2);
    tRans.add(t4 - t3); tTotal.add(t4 - t0);
  }

  const int encoded = tTotal.n > 0 ? tTotal.n : 1;
  const double bpp = static_cast<double>(totalPayload) * 8.0 /
                     (static_cast<double>(hw) * encoded);
  printf("frames=%d q=%d payload=%zuB bpp=%.4f kbps@30=%.0f\n", encoded, qIndex,
         totalPayload, bpp, bpp * static_cast<double>(hw) * 30.0 / 1000.0);
  printf("ms/frame avg (max): pre=%.2f(%.2f) enc1=%.2f(%.2f) enc2||rans=%.2f(%.2f) "
         "ransWait=%.2f(%.2f) total=%.2f(%.2f)\n",
         tPre.avg(), tPre.mx, tNpu1.avg(), tNpu1.mx, tNpu2.avg(), tNpu2.mx,
         tRans.avg(), tRans.mx, tTotal.avg(), tTotal.mx);
  if (out.srt != SRT_INVALID_SOCK) {
    srt_close(out.srt);
    mlvc::srtCleanup();
    printf("srt closed\n");
  }
  if (out.fd >= 0) {
    // Half-close and wait for the receiver to finish reading everything still
    // queued in the socket buffer; exiting immediately resets the connection
    // and silently discards buffered frames (soak test lost 833 of 1920).
    shutdown(out.fd, SHUT_WR);
    char drain[256];
    while (recv(out.fd, drain, sizeof(drain), 0) > 0) {}
    close(out.fd);
    printf("socket drained, receiver closed\n");
  }
  printf("send blocking: avg=%.2f ms max=%.2f ms\n", tSendStat.avg(), tSendStat.mx);
  if (iframesForced) printf("recovery: %d I-frames forced by receiver\n", iframesForced);
  if (liveInput) {
    printf("latency guard: %ld input frames skipped (stale), %ld encodes skipped "
           "(link backlogged)\n", inputDropped, encodeSkipped);
  }
  if (adaptive) {
    printf("adaptive: %d q changes, q ranged %d..%d (started %d, ended %d)\n",
           adaptations, qFloorSeen, qCeilSeen, qIndex, rc.q);
  }
  if (verify) printf("bitstream verify: %s\n", allVerified ? "ALL OK" : "FAILURES");
  return allVerified ? 0 : 1;
}
