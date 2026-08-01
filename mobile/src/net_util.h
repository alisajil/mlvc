#pragma once
// Blocking TCP helpers shared by the encoder (sender) and receiver. POSIX
// sockets work unchanged on bionic and macOS. Address-family agnostic: IPv6 is
// the only path that works phone->Mac here, since both ends sit behind
// carrier-grade NAT on IPv4.
// ponytail: TCP + blocking IO. UDP/FEC/jitter buffer belongs here when the
// transport needs to survive real packet loss.

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace mlvc {

// Splits "host:port" or "[v6::addr]:port". Returns false if no port present.
inline bool splitHostPort(const std::string& target, std::string& host, uint16_t& port) {
  size_t colon;
  if (!target.empty() && target[0] == '[') {  // bracketed IPv6 literal
    const size_t end = target.find(']');
    if (end == std::string::npos) return false;
    host = target.substr(1, end - 1);
    colon = target.find(':', end);
  } else {
    colon = target.rfind(':');
    if (colon == std::string::npos) return false;
    host = target.substr(0, colon);
  }
  if (colon == std::string::npos) return false;
  port = static_cast<uint16_t>(atoi(target.c_str() + colon + 1));
  return port != 0;
}

// Connects to host:port, trying every resolved address (v6 first if present).
// Returns -1 on failure.
inline int tcpConnect(const std::string& host, uint16_t port) {
  char portStr[8];
  snprintf(portStr, sizeof(portStr), "%u", port);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), portStr, &hints, &res) != 0 || !res) return -1;

  int fd = -1;
  for (addrinfo* a = res; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd >= 0) {
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return fd;
}

// Listens on port and blocks until one client connects. Returns -1 on failure.
// bindAll=false restricts to IPv4 loopback (the adb-reverse USB tunnel);
// true opens a dual-stack socket on all interfaces, so the port is publicly
// reachable over IPv6 (there is no NAT to hide behind).
inline int tcpAcceptOne(uint16_t port, bool bindAll = false) {
  int srv;
  const int one = 1;
  if (bindAll) {
    srv = socket(AF_INET6, SOCK_STREAM, 0);
    if (srv < 0) return -1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    const int off = 0;  // dual-stack: accept IPv4-mapped clients too
    setsockopt(srv, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    sockaddr_in6 a6{};
    a6.sin6_family = AF_INET6;
    a6.sin6_addr = in6addr_any;
    a6.sin6_port = htons(port);
    if (bind(srv, reinterpret_cast<sockaddr*>(&a6), sizeof(a6)) != 0 || listen(srv, 1) != 0) {
      close(srv);
      return -1;
    }
  } else {
    srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return -1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a4{};
    a4.sin_family = AF_INET;
    a4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a4.sin_port = htons(port);
    if (bind(srv, reinterpret_cast<sockaddr*>(&a4), sizeof(a4)) != 0 || listen(srv, 1) != 0) {
      close(srv);
      return -1;
    }
  }
  const int fd = accept(srv, nullptr, nullptr);
  close(srv);
  if (fd >= 0) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  return fd;
}

inline bool sendAll(int fd, const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    const ssize_t w = send(fd, p, n, 0);
    if (w <= 0) return false;
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

// Reads one fixed-size message if one is fully buffered, else returns false
// without blocking. Used to drain feedback from the media socket.
inline bool recvNonBlocking(int fd, void* data, size_t n) {
  const ssize_t r = recv(fd, data, n, MSG_DONTWAIT);
  return r == static_cast<ssize_t>(n);
}

// Best-effort send that never blocks the caller; drops if the buffer is full.
inline void sendNonBlocking(int fd, const void* data, size_t n) {
  send(fd, data, n, MSG_DONTWAIT);
}

inline bool recvAll(int fd, void* data, size_t n) {
  auto* p = static_cast<uint8_t*>(data);
  while (n > 0) {
    const ssize_t r = recv(fd, p, n, 0);
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

}  // namespace mlvc
