#pragma once
// Receiver -> sender congestion feedback. Both ends must agree on this layout,
// so it lives in one file. Sent over the same TCP connection as the media,
// travelling receiver->sender (downlink to the phone), which is far less
// congested than the uplink carrying video.

#include <algorithm>
#include <cstdint>

namespace mlvc {

constexpr uint32_t kFeedbackMagic = 0x46564C4D;  // 'MLVF'

// Receiver starvation thresholds. Measured on real cellular: a healthy link
// leaves the receiver waiting ~6 ms per frame, a saturated one ~170 ms.
// kStarvedMs is ~3 frame intervals at 30fps: below that it is ordinary
// cellular jitter, not a link that cannot keep up.
constexpr double kStarvedMs = 90.0;
constexpr double kHealthyMs = 25.0;

#pragma pack(push, 1)
struct Feedback {
  uint32_t magic = kFeedbackMagic;
  uint32_t frameIdx = 0;
  int32_t verdict = 0;   // -1 shed bitrate, 0 hold, +1 headroom to probe up
  uint32_t waitMs = 0;   // how long the receiver sat idle waiting for this frame
  uint32_t needIframe = 0;  // a frame arrived incomplete: re-seed both chains
};
#pragma pack(pop)

// AIMD rate control over q_index, shared by the CLI encoder and the app so the
// two cannot drift apart. Per-frame q costs nothing in this codec (q_index is
// just a graph input), so quality changes need no keyframe.
//
// The control law is dominated by feedback lag: when q drops, frames already
// in flight were encoded at the OLD rate, so their feedback keeps reporting
// congestion. Reacting to that is a cascade straight to the floor - measured
// at mean q39/63 from only 28 starved frames in 632. Hence the cooldown
// (absorb one change before judging the next) and the debounce (a single late
// frame is jitter, not congestion).
struct RateController {
  int q = 63, qMin = 21, qMax = 63;

  void congested(uint32_t waitMs) {
    starvedRun_ = 0;
    cleanRun_ = 0;
    if (cooldown_ > 0) return;  // still absorbing the previous change
    // Shed proportionally: barely late trims a little, deeply stalled cuts hard.
    const int step = std::clamp(static_cast<int>(waitMs / 20), 3, 12);
    q = std::max(qMin, q - step);
    raiseStep_ = kRaiseStepMin;
    cooldown_ = kCooldownFrames;
  }

  // Congestion signal that must repeat before it is believed.
  void maybeCongested(uint32_t waitMs) {
    cleanRun_ = 0;
    if (++starvedRun_ >= kStarvedRunNeeded) congested(waitMs);
  }

  void healthy() {
    starvedRun_ = 0;
    if (++cleanRun_ < kRaiseAfter) return;
    q = std::min(qMax, q + raiseStep_);
    raiseStep_ = std::min(kRaiseStepMax, raiseStep_ * 2);  // compounding probe
    cleanRun_ = 0;
  }

  void hold() { starvedRun_ = 0; cleanRun_ = 0; }
  void tick() { if (cooldown_ > 0) --cooldown_; }

 private:
  static constexpr int kRaiseAfter = 6;        // clean frames before probing up
  static constexpr int kRaiseStepMin = 2;
  static constexpr int kRaiseStepMax = 8;
  static constexpr int kStarvedRunNeeded = 3;  // debounce isolated jitter
  static constexpr int kCooldownFrames = 15;   // ~0.5s at 30fps
  int cleanRun_ = 0, starvedRun_ = 0, raiseStep_ = kRaiseStepMin, cooldown_ = 0;
};

}  // namespace mlvc
