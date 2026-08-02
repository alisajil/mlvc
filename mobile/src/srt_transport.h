#pragma once
// SRT transport. Unlike TCP, SRT live mode abandons packets that miss the
// latency budget instead of retransmitting them forever, which is what bounds
// the multi-second stalls TCP cannot avoid (measured peak ~2.4s).
//
// The cost is that loss becomes visible to us: SRT live mode caps a message at
// SRT_LIVE_DEF_PLSIZE (1316 B) while our frames are ~7 KB, so a frame is sent
// as several chunks and any missing chunk makes that frame undecodable. This
// codec is recurrent, so an undecodable frame must not simply be skipped - the
// receiver reports the gap and the encoder answers with an I-frame, which
// re-seeds both reference chains.

#include <srt/srt.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mlvc {

constexpr int kSrtChunkPayload = 1300;  // under SRT's 1316 live-mode limit

#pragma pack(push, 1)
struct SrtChunkHeader {
  uint32_t frameIdx;
  uint16_t chunkIdx;
  uint16_t chunkCount;
  uint32_t frameBytes;  // total payload size of this frame
};
#pragma pack(pop)

// One-time library init/teardown.
inline void srtStartup() { srt_startup(); }
inline void srtCleanup() { srt_cleanup(); }

// Configures a socket for live streaming with a fixed latency budget: SRT
// delivers what arrives within `latencyMs` and drops the rest rather than
// stalling the stream.
inline void srtConfigureLive(SRTSOCKET s, int latencyMs) {
  const int transtype = SRTT_LIVE;
  srt_setsockflag(s, SRTO_TRANSTYPE, &transtype, sizeof(transtype));
  srt_setsockflag(s, SRTO_LATENCY, &latencyMs, sizeof(latencyMs));
  const int payload = kSrtChunkPayload;
  srt_setsockflag(s, SRTO_PAYLOADSIZE, &payload, sizeof(payload));
  const int tlpktdrop = 1;  // drop packets that miss the deadline
  srt_setsockflag(s, SRTO_TLPKTDROP, &tlpktdrop, sizeof(tlpktdrop));
  const int tsbpd = 1;      // timestamp-based delivery, keeps pacing sane
  srt_setsockflag(s, SRTO_TSBPDMODE, &tsbpd, sizeof(tsbpd));
  // Cellular paths are commonly 1400-1428 B; the 1500 default fragments or
  // silently drops.
  const int mss = 1360;
  srt_setsockflag(s, SRTO_MSS, &mss, sizeof(mss));
}

// Caller side. Returns SRT_INVALID_SOCK on failure.
SRTSOCKET srtConnect(const std::string& host, uint16_t port, int latencyMs);

// Listener side; blocks until one caller connects.
//
// `bindAddr` should be the exact IPv6 address callers dial (empty = any).
// Binding the specific address matters on a multi-address host: with a
// wildcard bind the OS picks the reply source address by its own preference
// (macOS prefers the rotating RFC 4941 temporary address), and when that
// differs from the address the caller targeted, the caller discards the
// handshake response and times out. Binding pins the source. This was the
// entire "SRT never connects over cellular" failure.
SRTSOCKET srtAcceptOne(uint16_t port, int latencyMs,
                       const std::string& bindAddr = "");

// Sends one frame as chunked messages. Returns false only on a dead socket -
// individual dropped chunks are SRT's business, not an error here.
bool srtSendFrame(SRTSOCKET s, uint32_t frameIdx, const void* data, size_t n);

// Waits (up to maxWaitMs) for the send buffer to drain before closing.
// srt_close discards whatever is still in flight, which in live mode is a full
// latency budget of frames - measured: exactly 6 tail frames (200 ms at 30fps)
// lost on every clean shutdown without this.
void srtDrainSend(SRTSOCKET s, int maxWaitMs = 2000);

// Receives chunks until one frame is complete. Sets `lost` when chunks for a
// frame went missing (that frame is undecodable and the caller must recover).
// Returns false when the connection is gone.
bool srtRecvFrame(SRTSOCKET s, std::vector<uint8_t>& out, uint32_t& frameIdx,
                  bool& lost);

}  // namespace mlvc
