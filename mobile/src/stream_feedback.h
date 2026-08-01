#pragma once
// Receiver -> sender congestion feedback. Both ends must agree on this layout,
// so it lives in one file. Sent over the same TCP connection as the media,
// travelling receiver->sender (downlink to the phone), which is far less
// congested than the uplink carrying video.

#include <cstdint>

namespace mlvc {

constexpr uint32_t kFeedbackMagic = 0x46564C4D;  // 'MLVF'

// Receiver starvation thresholds. Measured on real cellular: a healthy link
// leaves the receiver waiting ~6 ms per frame, a saturated one ~170 ms.
constexpr double kStarvedMs = 50.0;
constexpr double kHealthyMs = 12.0;

#pragma pack(push, 1)
struct Feedback {
  uint32_t magic = kFeedbackMagic;
  uint32_t frameIdx = 0;
  int32_t verdict = 0;   // -1 shed bitrate, 0 hold, +1 headroom to probe up
  uint32_t waitMs = 0;   // how long the receiver sat idle waiting for this frame
};
#pragma pack(pop)

}  // namespace mlvc
