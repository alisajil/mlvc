// Records raw I420 frames from the camera bridge to a file, for capturing a
// fair natural-content comparison clip (the synthetic torture test clips used
// for QP-sweep validation are a known worst case for a temporally-tuned
// codec - see PLAN.md - not representative of real content).
//
// Usage: yuv_recorder --listen 8901 --width 1280 --height 720 --frames 300 \
//                      --out capture.yuv
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "net_util.h"

int main(int argc, char** argv) {
  int listenPort = 0, width = 1280, height = 720, frames = 0;
  std::string outPath;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--listen") listenPort = atoi(next());
    else if (a == "--width") width = atoi(next());
    else if (a == "--height") height = atoi(next());
    else if (a == "--frames") frames = atoi(next());
    else if (a == "--out") outPath = next();
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (listenPort == 0 || frames <= 0 || outPath.empty()) {
    fprintf(stderr, "usage: yuv_recorder --listen PORT --width W --height H --frames N --out FILE\n");
    return 2;
  }

  fprintf(stderr, "waiting for camera bridge on 127.0.0.1:%d ...\n", listenPort);
  const int fd = mlvc::tcpAcceptOne(static_cast<uint16_t>(listenPort), false);
  if (fd < 0) { fprintf(stderr, "bridge listen failed\n"); return 1; }

  uint32_t hdr[3] = {0, 0, 0};
  if (!mlvc::recvAll(fd, hdr, sizeof(hdr)) || hdr[0] != 0x30323449u ||
      static_cast<int>(hdr[1]) != width || static_cast<int>(hdr[2]) != height) {
    fprintf(stderr, "bad bridge header (magic=0x%08x %ux%u, expected %dx%d)\n",
            hdr[0], hdr[1], hdr[2], width, height);
    return 1;
  }
  fprintf(stderr, "camera bridge connected (%ux%u)\n", hdr[1], hdr[2]);

  FILE* out = fopen(outPath.c_str(), "wb");
  if (!out) { fprintf(stderr, "cannot open %s for writing\n", outPath.c_str()); return 1; }

  const size_t frameBytes = static_cast<size_t>(width) * height * 3 / 2;
  std::vector<uint8_t> buf(frameBytes);
  for (int f = 0; f < frames; ++f) {
    if (!mlvc::recvAll(fd, buf.data(), frameBytes)) {
      fprintf(stderr, "camera feed ended at frame %d\n", f);
      break;
    }
    fwrite(buf.data(), 1, buf.size(), out);
    if ((f + 1) % 30 == 0) fprintf(stderr, "recorded %d/%d frames\n", f + 1, frames);
  }
  fclose(out);
  fprintf(stderr, "done: %s\n", outPath.c_str());
  return 0;
}
