#pragma once
// rANS entropy coder over quantized PMF tables (16-bit precision, implicit
// escape via each row's last bin + 2-bit bypass chunks). Self-consistent
// encoder/decoder pair; not byte-compatible with the closed msrtc stream.

#include <cstdint>
#include <vector>

namespace mlvc {

// One PMF family: flat table segmented by rowLengths, per-row symbol offset.
// Row i covers integer values [-offset[i], -offset[i] + length[i] - 2];
// the last bin is the escape symbol.
struct PmfSet {
  std::vector<int32_t> lengths;
  std::vector<int32_t> offsets;
  std::vector<uint32_t> rowStart;   // cumsum of lengths
  std::vector<uint32_t> cdf;        // per-row cumulative freq, rowStart-aligned,
                                    // cdf[rowStart[i] + k] = sum of freqs < k
  void build(const int32_t* lens, const int32_t* offs, size_t rows,
             const int32_t* table, size_t tableLen);
};

class RansEncoder {
 public:
  // Streaming: entropy-codes immediately. Decode order is the REVERSE of push
  // order — push blocks in reverse of the decoder's read order.
  void push(const PmfSet& pmf, uint32_t row, int32_t value);
  std::vector<uint8_t> flush();
  // Pre-sizes the word buffer (capacity survives flush; call once).
  void reserve(size_t symbols) { words_.reserve(symbols / 2 + 16); }

 private:
  void putSlot(uint32_t cum, uint32_t freq);
  uint64_t state_ = 1ull << 31;  // rans64 lower bound
  std::vector<uint32_t> words_;
};

class RansDecoder {
 public:
  explicit RansDecoder(const uint8_t* data, size_t size);
  int32_t decode(const PmfSet& pmf, uint32_t row);
  // True when the stream is fully consumed and the state returned to init.
  bool eofOk() const;

 private:
  uint32_t getBits(uint32_t nbits);
  const uint32_t* words_ = nullptr;
  size_t numWords_ = 0;
  size_t pos_ = 0;
  uint64_t state_ = 0;
  bool bad_ = false;
};

}  // namespace mlvc
