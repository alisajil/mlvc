#include "srt_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#include <cstdio>

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
  // srt_create_socket() with no hint appears to default the underlying UDP
  // socket to AF_INET: connecting it straight to an IPv6 destination produced
  // zero packets on the wire and a silent timeout (confirmed with a packet
  // capture and SRT's own debug log - not one handshake attempt logged).
  // Binding an IPv6 ANY address first forces the real socket to be IPv6.
  // NOTE (2026-08-02): connect over real cellular IPv6 times out with ZERO
  // packets on the wire (confirmed via tcpdump) despite raw UDP to the same
  // address working (confirmed with a plain nc round-trip). An explicit
  // pre-bind to force the IPv6 family was tried and rejected outright by SRT
  // (srt_bind -> "Operation not supported"), disproving that theory. Root
  // cause not yet found - see PLAN.md.
  if (srt_connect(s, reinterpret_cast<sockaddr*>(&addr), addrLen) == SRT_ERROR) {
    fprintf(stderr, "srt_connect: %s\n", srt_getlasterror_str());
    srt_close(s);
    return SRT_INVALID_SOCK;
  }
  return s;
}

SRTSOCKET srtAcceptOne(uint16_t port, int latencyMs) {
  const SRTSOCKET srv = srt_create_socket();
  if (srv == SRT_INVALID_SOCK) return SRT_INVALID_SOCK;
  srtConfigureLive(srv, latencyMs);
  sockaddr_in6 a6{};
  a6.sin6_family = AF_INET6;
  a6.sin6_addr = in6addr_any;
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

bool srtSendFrame(SRTSOCKET s, uint32_t frameIdx, const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  const size_t body = kSrtChunkPayload - sizeof(SrtChunkHeader);
  const uint16_t count = static_cast<uint16_t>((n + body - 1) / body);
  std::vector<uint8_t> buf(kSrtChunkPayload);
  for (uint16_t i = 0; i < count; ++i) {
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
    if (n < static_cast<int>(sizeof(SrtChunkHeader))) continue;
    SrtChunkHeader h{};
    memcpy(&h, buf, sizeof(h));
    const size_t body = kSrtChunkPayload - sizeof(SrtChunkHeader);

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

}  // namespace mlvc
