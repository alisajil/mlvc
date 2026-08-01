#pragma once
// MLVC-S frame pipeline pieces shared by the encoder CLI: PMF blob loading,
// scale extraction from z, checkerboard packing, YUV420->444 preprocess, and
// the per-frame QP shift schedule. Mirrors conversion/_split_model semantics.

#include <cstdint>
#include <string>
#include <vector>

#include "rans.h"

namespace mlvc {

// Model geometry for 720p MLVC-S (from exported metadata.json).
struct ModelDims {
  int width = 1280, height = 720;
  int latentCh = 48, featureCh = 96, zCh = 48;
  int yH = 45, yW = 80;   // height/16, width/16
  int zH = 12, zW = 20;   // padded height/64, width/64
  int channelRepeat = 4;  // y_scale_repeat
  int spatialRepeat = 4;  // downsample_hyperprior
};

struct PmfTables {
  PmfSet gaussian;
  PmfSet bitest;
  uint32_t scaleLevels = 0;
  uint32_t qpNum = 0;
  uint32_t zChannels = 0;
  bool load(const std::string& path);
};

// q_index_shifted schedule: qp_shift[frame_index_map[(idx+1) % 8]]
int qpShift(int frameIdx);

// Reference-refresh schedule, from the model's own metadata and mirrored from
// the Python reference (conversion/_frame_loop.py). Both ends derive this
// independently from the frame index, so it needs no signalling — but it is
// NOT optional: the encoder references its own enc2 reconstruction while the
// decoder references its own output, on different silicon, so without these
// periodic resets the two drift apart and the picture destroys itself after a
// few hundred frames.
constexpr int kIframePeriod = 96;   // full reference clear (gray frame, no feature)
constexpr int kResetPeriod = 32;    // feature-only reset, reference frame kept
inline bool isIframe(int curFrameIdx) { return curFrameIdx % kIframePeriod == 0; }
inline bool isFeatureReset(int curFrameIdx) {
  return (curFrameIdx + 1) % kResetPeriod == 0;
}

// z_raw fp16 [1,48,12,20] -> integer scale indices packed into checkerboard
// halves scales0/scales1, each [24,45,80] int32 (clipped to scaleLevels-1).
void extractScales(const uint16_t* zRaw, const ModelDims& d, uint32_t scaleLevels,
                   std::vector<int32_t>& scales0, std::vector<int32_t>& scales1);

// 8-bit YUV420p frame -> fp16 [1,3,H,W] in [0,1] (UV nearest-upsampled 2x).
void yuv420ToTensor(const uint8_t* frame, int width, int height, uint16_t* out);

// Reusable per-frame scratch: keeps vector capacities across frames so the
// steady-state encode loop allocates nothing.
struct CoderWorkspace {
  RansEncoder enc;
  std::vector<int32_t> scales0, scales1;
};

// Entropy-codes one frame's latents. y/z buffers are the fp16 graph outputs
// (integer-valued). Returns the rANS bitstream (z, y0, y1 push order).
std::vector<uint8_t> encodeBitstream(const PmfTables& t, const ModelDims& d,
                                     const uint16_t* yRaw0, const uint16_t* yRaw1,
                                     const uint16_t* zRaw, int qIndex,
                                     CoderWorkspace& ws);

// Round-trip check: decodes the bitstream and compares against the original
// latents. Returns true when everything matches and EOF lands clean.
bool verifyBitstream(const PmfTables& t, const ModelDims& d,
                     const std::vector<uint8_t>& bs, const uint16_t* yRaw0,
                     const uint16_t* yRaw1, const uint16_t* zRaw, int qIndex);

// Receiver path: entropy-decodes one frame's latents from the bitstream into
// fp16 buffers (z first, scales derived from it, then y0/y1). Returns false
// on a corrupt stream (EOF check fails).
bool decodeLatents(const PmfTables& t, const ModelDims& d, const uint8_t* bs,
                   size_t bsSize, int qIndex, uint16_t* zRaw, uint16_t* yRaw0,
                   uint16_t* yRaw1, CoderWorkspace& ws);

}  // namespace mlvc
