#include "srt_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mlvc {
namespace {

// Resolves host:port, preferring IPv6 (the only path that works phone->Mac
// here, since both ends sit behind carrier NAT on IPv4).
bool resolve(const std::string& host, uint16_t port, sockaddr_storage& out,
             socklen_t& outLen) {
  char portStr[8];
  snprintf(portStr, sizeof(portStr), "%u", port);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), portStr, &hints, &res) != 0 || !res) return false;
  bool ok = false;
  for (addrinfo* a = res; a; a = a->ai_next) {  // v6 first
    if (a->ai_family == AF_INET6) {
      memcpy(&out, a->ai_addr, a->ai_addrlen);
      outLen = a->ai_addrlen;
      ok = true;
      break;
    }
  }
  if (!ok) {
    memcpy(&out, res->ai_addr, res->ai_addrlen);
    outLen = res->ai_addrlen;
    ok = true;
  }
  freeaddrinfo(res);
  return ok;
}

}  // namespace

SRTSOCKET srtConnect(const std::string& host, uint16_t port, int latencyMs) {
  sockaddr_storage addr{};
  socklen_t addrLen = 0;
  if (!resolve(host, port, addr, addrLen)) return SRT_INVALID_SOCK;
  const SRTSOCKET s = srt_create_socket();
  if (s == SRT_INVALID_SOCK) return SRT_INVALID_SOCK;
  srtConfigureLive(s, latencyMs);
  const int conntimeo = 8000;  // cellular handshakes are slow; default 3s is tight
  srt_setsockflag(s, SRTO_CONNTIMEO, &conntimeo, sizeof(conntimeo));
  // The sender polls this socket for receiver feedback (needIframe) between
  // frames; a blocking recv would stall the encode loop. Must be set BEFORE
  // connect - setting it after produced a "LiveCC buffer size: 20 is too
  // small" warning and a disconnect a few frames later (some options only
  // take effect cleanly pre-handshake).
  const bool blocking = false;
  srt_setsockflag(s, SRTO_RCVSYN, &blocking, sizeof(blocking));
  // If this times out with no packets apparently arriving: the caller side
  // sends fine (verified by syscall trace on Android). The historical failure
  // was the LISTENER replying from a different source address than the one
  // dialed (multi-address IPv6 host, wildcard bind) - see srtAcceptOne.
  if (srt_connect(s, reinterpret_cast<sockaddr*>(&addr), addrLen) == SRT_ERROR) {
    fprintf(stderr, "srt_connect: %s\n", srt_getlasterror_str());
    srt_close(s);
    return SRT_INVALID_SOCK;
  }
  return s;
}

SRTSOCKET srtAcceptOne(uint16_t port, int latencyMs,
                       const std::string& bindAddr) {
  const SRTSOCKET srv = srt_create_socket();
  if (srv == SRT_INVALID_SOCK) return SRT_INVALID_SOCK;
  srtConfigureLive(srv, latencyMs);
  sockaddr_in6 a6{};
  a6.sin6_family = AF_INET6;
  a6.sin6_addr = in6addr_any;
  if (!bindAddr.empty() &&
      inet_pton(AF_INET6, bindAddr.c_str(), &a6.sin6_addr) != 1) {
    fprintf(stderr, "srtAcceptOne: bad bind address '%s'\n", bindAddr.c_str());
    srt_close(srv);
    return SRT_INVALID_SOCK;
  }
  a6.sin6_port = htons(port);
  const int v6only = 1;  // IPv6 only: the sole path that works phone->Mac here
  srt_setsockflag(srv, SRTO_IPV6ONLY, &v6only, sizeof(v6only));
  if (srt_bind(srv, reinterpret_cast<sockaddr*>(&a6), sizeof(a6)) == SRT_ERROR ||
      srt_listen(srv, 1) == SRT_ERROR) {
    fprintf(stderr, "srt bind/listen: %s\n", srt_getlasterror_str());
    srt_close(srv);
    return SRT_INVALID_SOCK;
  }
  const SRTSOCKET s = srt_accept(srv, nullptr, nullptr);
  srt_close(srv);
  return s;
}

namespace {

// ponytail: test hook, env-parsed once. MLVC_SRT_DROP=comma list of frame
// indices to send incomplete (chunk 0 withheld); MLVC_SRT_DROP_FULL=frames to
// withhold entirely. Simulates TLPKTDROP loss deterministically.
bool inDropList(const char* env, uint32_t frameIdx) {
  const char* v = getenv(env);
  if (!v) return false;
  for (const char* p = v; *p;) {
    char* end = nullptr;
    if (strtoul(p, &end, 10) == frameIdx) return true;
    if (!end || *end == '\0') break;
    p = end + 1;
  }
  return false;
}

}  // namespace

bool srtSendFrame(SRTSOCKET s, uint32_t frameIdx, const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  const size_t body = kSrtChunkPayload - sizeof(SrtChunkHeader);
  const uint16_t count = static_cast<uint16_t>((n + body - 1) / body);
  const bool dropFull = inDropList("MLVC_SRT_DROP_FULL", frameIdx);
  const bool dropOne = !dropFull && inDropList("MLVC_SRT_DROP", frameIdx);
  if (dropFull || dropOne)
    fprintf(stderr, "TEST: dropping %s of frame %u\n",
            dropFull ? "all chunks" : "chunk 0", frameIdx);
  if (dropFull) return true;
  std::vector<uint8_t> buf(kSrtChunkPayload);
  for (uint16_t i = 0; i < count; ++i) {
    if (dropOne && i == 0) continue;
    const size_t off = static_cast<size_t>(i) * body;
    const size_t len = std::min(body, n - off);
    SrtChunkHeader h{frameIdx, i, count, static_cast<uint32_t>(n)};
    memcpy(buf.data(), &h, sizeof(h));
    memcpy(buf.data() + sizeof(h), p + off, len);
    if (srt_send(s, reinterpret_cast<const char*>(buf.data()),
                 static_cast<int>(sizeof(h) + len)) == SRT_ERROR) {
      const int err = srt_getlasterror(nullptr);
      // A dropped or congested message is expected on a live link; only a
      // broken connection ends the stream.
      if (err == SRT_ECONNLOST || err == SRT_EINVSOCK || err == SRT_ECONNREJ) return false;
    }
  }
  return true;
}

void srtDrainSend(SRTSOCKET s, int maxWaitMs) {
  for (int waited = 0; waited < maxWaitMs; waited += 10) {
    size_t blocks = 0, bytes = 0;
    if (srt_getsndbuffer(s, &blocks, &bytes) == SRT_ERROR || bytes == 0) return;
    usleep(10000);
  }
}

bool srtRecvFrame(SRTSOCKET s, std::vector<uint8_t>& out, uint32_t& frameIdx,
                  bool& lost) {
  static std::vector<uint8_t> assembly;
  static std::vector<bool> got;
  static uint32_t curFrame = UINT32_MAX;
  static uint16_t curCount = 0;
  static uint16_t haveCount = 0;

  char buf[kSrtChunkPayload + 64];
  lost = false;
  for (;;) {
    const int n = srt_recv(s, buf, sizeof(buf));
    if (n == SRT_ERROR) {
      const int err = srt_getlasterror(nullptr);
      if (err == SRT_ECONNLOST || err == SRT_EINVSOCK) return false;
      continue;
    }
    if (n < static_cast<int>(sizeof(SrtChunkHeader))) {
      if (getenv("MLVC_SRT_TRACE"))
        fprintf(stderr, "[srtRecvFrame] runt message n=%d (< header %zu)\n", n,
                sizeof(SrtChunkHeader));
      continue;
    }
    SrtChunkHeader h{};
    memcpy(&h, buf, sizeof(h));
    const size_t body = kSrtChunkPayload - sizeof(SrtChunkHeader);
    if (getenv("MLVC_SRT_TRACE"))
      fprintf(stderr, "[srtRecvFrame] n=%d frameIdx=%u chunkIdx=%u/%u frameBytes=%u\n",
              n, h.frameIdx, h.chunkIdx, h.chunkCount, h.frameBytes);

    if (h.frameIdx != curFrame) {
      // Moving on with an incomplete frame means SRT abandoned chunks for it.
      if (curFrame != UINT32_MAX && haveCount < curCount) lost = true;
      curFrame = h.frameIdx;
      curCount = h.chunkCount;
      haveCount = 0;
      assembly.assign(h.frameBytes, 0);
      got.assign(h.chunkCount, false);
    }
    if (h.chunkIdx < got.size() && !got[h.chunkIdx]) {
      const size_t off = static_cast<size_t>(h.chunkIdx) * body;
      const size_t len = static_cast<size_t>(n) - sizeof(h);
      if (off + len <= assembly.size()) {
        memcpy(assembly.data() + off, buf + sizeof(h), len);
        got[h.chunkIdx] = true;
        ++haveCount;
      }
    }
    if (haveCount == curCount && curCount > 0) {
      out = assembly;
      frameIdx = curFrame;
      curFrame = UINT32_MAX;
      return true;
    }
    if (lost) {  // report the gap before delivering anything further
      out.clear();
      frameIdx = h.frameIdx;
      return true;
    }
  }
}

bool srtSendHeader(SRTSOCKET s, const void* data, size_t n, int repeats,
                   int gapMs) {
  if (n > kSrtChunkPayload - sizeof(SrtChunkHeader)) return false;  // header is tiny; sanity check
  std::vector<uint8_t> buf(sizeof(SrtChunkHeader) + n);
  // frameIdx = UINT32_MAX: matches srtRecvFrame's own "no frame in progress"
  // reset sentinel, so a header message can never masquerade as - or get
  // confused with - a real frame's reassembly state. chunkCount = 0 is what
  // actually marks this as a header rather than a frame chunk.
  const SrtChunkHeader h{UINT32_MAX, 0, 0, static_cast<uint32_t>(n)};
  memcpy(buf.data(), &h, sizeof(h));
  memcpy(buf.data() + sizeof(h), data, n);
  for (int i = 0; i < repeats; ++i) {
    if (srt_send(s, reinterpret_cast<const char*>(buf.data()),
                 static_cast<int>(buf.size())) == SRT_ERROR) {
      const int err = srt_getlasterror(nullptr);
      if (err == SRT_ECONNLOST || err == SRT_EINVSOCK || err == SRT_ECONNREJ) return false;
    }
    if (i + 1 < repeats) usleep(gapMs * 1000);
  }
  return true;
}

bool srtRecvHeader(SRTSOCKET s, std::vector<uint8_t>& out) {
  char buf[kSrtChunkPayload + 64];
  for (;;) {
    const int n = srt_recv(s, buf, sizeof(buf));
    if (n == SRT_ERROR) {
      const int err = srt_getlasterror(nullptr);
      if (err == SRT_ECONNLOST || err == SRT_EINVSOCK) return false;
      continue;
    }
    if (n < static_cast<int>(sizeof(SrtChunkHeader))) continue;  // runt, ignore
    SrtChunkHeader h{};
    memcpy(&h, buf, sizeof(h));
    if (h.chunkCount != 0) continue;  // an ordinary frame chunk, not the
                                       // header - the main loop's own loss
                                       // recovery will handle it from here
    out.assign(buf + sizeof(h), buf + sizeof(h) + h.frameBytes);
    return true;
  }
}

}  // namespace mlvc
