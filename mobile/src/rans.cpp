#include "rans.h"

#include <cassert>

namespace mlvc {
namespace {

constexpr uint32_t kPrecision = 16;            // symbolBits
constexpr uint32_t kBypassBits = 2;
constexpr uint32_t kMaxBypassVal = (1u << kBypassBits) - 1;
constexpr uint64_t kRansL = 1ull << 31;        // rans64 lower bound

}  // namespace

void PmfSet::build(const int32_t* lens, const int32_t* offs, size_t rows,
                   const int32_t* table, size_t tableLen) {
  lengths.assign(lens, lens + rows);
  offsets.assign(offs, offs + rows);
  rowStart.resize(rows);
  uint32_t acc = 0;
  for (size_t i = 0; i < rows; ++i) {
    rowStart[i] = acc;
    acc += static_cast<uint32_t>(lengths[i]);
  }
  assert(acc == tableLen);
  cdf.resize(tableLen + rows);  // one extra slot per row for the total
  size_t w = 0;
  for (size_t i = 0; i < rows; ++i) {
    uint32_t c = 0;
    for (int32_t k = 0; k < lengths[i]; ++k) {
      cdf[w++] = c;
      c += static_cast<uint32_t>(table[rowStart[i] + static_cast<uint32_t>(k)]);
    }
    cdf[w++] = c;  // == 1 << kPrecision for our tables
    assert(c == (1u << kPrecision));
  }
  // Re-point rowStart at the widened cdf layout (rows add one slot each).
  for (size_t i = 0; i < rows; ++i) {
    rowStart[i] += static_cast<uint32_t>(i);
  }
}

// Streaming encoder: symbols are entropy-coded immediately (no op buffer).
// Decode order is therefore the REVERSE of push order — callers must push
// blocks in reverse of the order the decoder will read them.

void RansEncoder::putSlot(uint32_t cum, uint32_t freq) {
  const uint64_t xMax = ((kRansL >> kPrecision) << 32) * freq;
  if (state_ >= xMax) {
    words_.push_back(static_cast<uint32_t>(state_));
    state_ >>= 32;
  }
  state_ = ((state_ / freq) << kPrecision) + (state_ % freq) + cum;
}

void RansEncoder::push(const PmfSet& pmf, uint32_t row, int32_t value) {
  const uint32_t base = pmf.rowStart[row];
  const int32_t rowLen = pmf.lengths[row];      // includes escape bin
  const int32_t maxDirect = rowLen - 2;         // last direct symbol index
  const int32_t idx = value + pmf.offsets[row];

  auto cdfSlot = [&](int32_t k) {
    const uint32_t c = pmf.cdf[base + static_cast<uint32_t>(k)];
    const uint32_t f = pmf.cdf[base + static_cast<uint32_t>(k) + 1] - c;
    putSlot(c, f);
  };
  constexpr uint32_t kBypassFreq = 1u << (kPrecision - kBypassBits);
  auto bypass = [&](uint32_t val) { putSlot(val * kBypassFreq, kBypassFreq); };

  if (idx >= 0 && idx <= maxDirect) {
    cdfSlot(idx);
    return;
  }
  // Escape. Decoder reads: escape slot, count seq, chunks 0..n-1 — so emit
  // everything in exact reverse: chunks n-1..0, count seq reversed, escape.
  const uint32_t raw = idx < 0 ? static_cast<uint32_t>(-idx) * 2 - 1
                               : static_cast<uint32_t>(idx - (maxDirect + 1)) * 2;
  uint32_t nChunks = 0;
  while ((raw >> (nChunks * kBypassBits)) != 0) ++nChunks;
  for (uint32_t j = nChunks; j-- > 0;) {
    bypass((raw >> (j * kBypassBits)) & kMaxBypassVal);
  }
  bypass(nChunks % kMaxBypassVal);
  for (uint32_t r = nChunks / kMaxBypassVal; r > 0; --r) bypass(kMaxBypassVal);
  cdfSlot(rowLen - 1);
}

std::vector<uint8_t> RansEncoder::flush() {
  words_.push_back(static_cast<uint32_t>(state_ >> 32));
  words_.push_back(static_cast<uint32_t>(state_));
  std::vector<uint8_t> out(words_.size() * 4);
  for (size_t i = 0; i < words_.size(); ++i) {
    const uint32_t w = words_[words_.size() - 1 - i];
    out[i * 4 + 0] = static_cast<uint8_t>(w);
    out[i * 4 + 1] = static_cast<uint8_t>(w >> 8);
    out[i * 4 + 2] = static_cast<uint8_t>(w >> 16);
    out[i * 4 + 3] = static_cast<uint8_t>(w >> 24);
  }
  words_.clear();
  state_ = kRansL;
  return out;
}

RansDecoder::RansDecoder(const uint8_t* data, size_t size) {
  words_ = reinterpret_cast<const uint32_t*>(data);
  numWords_ = size / 4;
  if (numWords_ < 2) {  // stream must carry at least the initial 64-bit state
    bad_ = true;
    state_ = kRansL;
    pos_ = 0;
    return;
  }
  // flush() pushes [.., high, low] then reverses: stream starts [low, high].
  state_ = (static_cast<uint64_t>(words_[1]) << 32) | words_[0];
  pos_ = 2;
}

uint32_t RansDecoder::getBits(uint32_t nbits) {
  const uint32_t f = 1u << (kPrecision - nbits);
  const uint32_t cum = static_cast<uint32_t>(state_ & ((1u << kPrecision) - 1));
  const uint32_t val = cum / f;
  state_ = f * (state_ >> kPrecision) + (cum % f);
  while (state_ < kRansL && pos_ < numWords_) {
    state_ = (state_ << 32) | words_[pos_++];
  }
  return val;
}

int32_t RansDecoder::decode(const PmfSet& pmf, uint32_t row) {
  const uint32_t base = pmf.rowStart[row];
  const int32_t rowLen = pmf.lengths[row];
  const uint32_t cum = static_cast<uint32_t>(state_ & ((1u << kPrecision) - 1));
  // linear scan; rows are <= ~40 bins
  int32_t k = 0;
  while (k + 1 < rowLen && pmf.cdf[base + static_cast<uint32_t>(k) + 1] <= cum) ++k;
  const uint32_t c = pmf.cdf[base + static_cast<uint32_t>(k)];
  const uint32_t f = pmf.cdf[base + static_cast<uint32_t>(k) + 1] - c;
  state_ = f * (state_ >> kPrecision) + (cum - c);
  while (state_ < kRansL && pos_ < numWords_) {
    state_ = (state_ << 32) | words_[pos_++];
  }
  const int32_t maxDirect = rowLen - 2;
  if (k <= maxDirect) return k - pmf.offsets[row];
  // escape: read chunk count (saturated), then chunks
  uint32_t nChunks = 0, v = 0;
  while ((v = getBits(kBypassBits)) == kMaxBypassVal) nChunks += kMaxBypassVal;
  nChunks += v;
  uint32_t raw = 0;
  for (uint32_t j = 0; j < nChunks; ++j) {
    raw |= getBits(kBypassBits) << (j * kBypassBits);
  }
  const int32_t idx = (raw & 1) ? -static_cast<int32_t>((raw + 1) / 2)
                                : maxDirect + 1 + static_cast<int32_t>(raw / 2);
  return idx - pmf.offsets[row];
}

bool RansDecoder::eofOk() const { return !bad_ && pos_ == numWords_ && state_ == kRansL; }

}  // namespace mlvc
