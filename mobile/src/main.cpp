// mlvc_bench: load an MLVC QNN context binary and benchmark its graph on the
// HTP NPU with synthetic inputs. M1 step 1 — proves our own C++ QNN path
// before wiring preprocessing, rANS, and the recurrent state loop.
//
// Usage:
//   mlvc_bench --context <ctx.bin> [--backend libQnnHtp.so]
//              [--system libQnnSystem.so] [--loops 100] [--no-burst]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qnn_runner.h"

namespace {

double msNow() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Fills a buffer with plausible values per dtype: fp16 random [0,1) for image
// tensors, zeros for int32 (q_index 0 is a valid quality level).
void fillInput(void* buf, const mlvc::TensorDesc& d, std::mt19937& rng) {
  const size_t n = d.byteSize / mlvc::qnnElementSize(d.dataType);
  if (d.dataType == 0x0216 /* QNN_DATATYPE_FLOAT_16 */) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < n; ++i) {
      // float -> fp16 (round-to-nearest not required for benchmark data)
      float f = dist(rng);
      uint32_t bits;
      memcpy(&bits, &f, 4);
      uint16_t h = static_cast<uint16_t>(((bits >> 16) & 0x8000) |
                                         ((((bits >> 23) - 112) & 0x1F) << 10) |
                                         ((bits >> 13) & 0x3FF));
      p[i] = h;
    }
  } else {
    memset(buf, 0, d.byteSize);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string context, backend = "libQnnHtp.so", system = "libQnnSystem.so";
  int loops = 100;
  bool burst = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--context") context = next();
    else if (a == "--backend") backend = next();
    else if (a == "--system") system = next();
    else if (a == "--loops") loops = atoi(next());
    else if (a == "--no-burst") burst = false;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (context.empty()) {
    fprintf(stderr, "usage: mlvc_bench --context <ctx.bin> [--loops N]\n");
    return 2;
  }

  double t0 = msNow();
  mlvc::QnnRunner runner;
  if (!runner.init(backend, system, context, burst)) {
    fprintf(stderr, "init failed: %s\n", runner.error().c_str());
    return 1;
  }
  double initMs = msNow() - t0;

  std::mt19937 rng(42);
  int rc = 0;
  for (const auto& g : runner.graphs()) {
    printf("graph %s\n", g.graphName.c_str());
    std::vector<std::vector<uint8_t>> inBufs, outBufs;
    std::vector<void*> inPtrs, outPtrs;
    for (const auto& d : g.inputs) {
      printf("  in  %-16s %8zu B\n", d.name.c_str(), d.byteSize);
      inBufs.emplace_back(d.byteSize);
      fillInput(inBufs.back().data(), d, rng);
      inPtrs.push_back(inBufs.back().data());
    }
    for (const auto& d : g.outputs) {
      printf("  out %-16s %8zu B\n", d.name.c_str(), d.byteSize);
      outBufs.emplace_back(d.byteSize);
      outPtrs.push_back(outBufs.back().data());
    }

    for (int i = 0; i < 3; ++i) {  // warmup
      if (!runner.execute(g.graphName, inPtrs, outPtrs)) {
        fprintf(stderr, "execute failed: %s\n", runner.error().c_str());
        return 1;
      }
    }
    std::vector<double> times;
    times.reserve(static_cast<size_t>(loops));
    for (int i = 0; i < loops; ++i) {
      double s = msNow();
      if (!runner.execute(g.graphName, inPtrs, outPtrs)) {
        fprintf(stderr, "execute failed: %s\n", runner.error().c_str());
        return 1;
      }
      times.push_back(msNow() - s);
    }
    std::sort(times.begin(), times.end());
    double sum = 0;
    for (double t : times) sum += t;
    printf("  init=%.1fms  runs=%d  min=%.2fms  median=%.2fms  mean=%.2fms  p95=%.2fms\n",
           initMs, loops, times.front(), times[times.size() / 2],
           sum / static_cast<double>(times.size()),
           times[static_cast<size_t>(static_cast<double>(times.size()) * 0.95)]);
  }
  return rc;
}
