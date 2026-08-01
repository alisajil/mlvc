#include "mlvc_codec.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace mlvc {
namespace {

inline float fp16ToF32(uint16_t h) { return static_cast<float>(*reinterpret_cast<__fp16*>(&h)); }
inline uint16_t f32ToFp16(float f) {
  __fp16 h = static_cast<__fp16>(f);
  uint16_t out;
  memcpy(&out, &h, 2);
  return out;
}

}  // namespace

bool PmfTables::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  auto rdU32 = [&]() { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
  if (rdU32() != 0x4D504D46u) return false;
  if (rdU32() != 1) return false;

  const uint32_t gRows = rdU32();
  const uint32_t gLen = rdU32();
  scaleLevels = rdU32();
  std::vector<int32_t> lens(gRows), offs(gRows), table(gLen);
  f.read(reinterpret_cast<char*>(lens.data()), gRows * 4);
  f.read(reinterpret_cast<char*>(offs.data()), gRows * 4);
  f.read(reinterpret_cast<char*>(table.data()), gLen * 4);
  gaussian.build(lens.data(), offs.data(), gRows, table.data(), gLen);

  qpNum = rdU32();
  zChannels = rdU32();
  const uint32_t bRows = rdU32();
  const uint32_t bLen = rdU32();
  lens.resize(bRows); offs.resize(bRows); table.resize(bLen);
  f.read(reinterpret_cast<char*>(lens.data()), bRows * 4);
  f.read(reinterpret_cast<char*>(offs.data()), bRows * 4);
  f.read(reinterpret_cast<char*>(table.data()), bLen * 4);
  if (!f) return false;
  bitest.build(lens.data(), offs.data(), bRows, table.data(), bLen);
  return true;
}

int qpShift(int frameIdx) {
  static const int kFrameIndexMap[8] = {0, 1, 0, 2, 0, 2, 0, 2};
  static const int kQpShift[3] = {0, 8, 4};
  return kQpShift[kFrameIndexMap[(frameIdx + 1) % 8]];
}

void extractScales(const uint16_t* zRaw, const ModelDims& d, uint32_t scaleLevels,
                   std::vector<int32_t>& scales0, std::vector<int32_t>& scales1) {
  const int halfCh = d.latentCh / 2;                 // 24
  const int baseCh = d.latentCh / d.channelRepeat;   // 12
  (void)baseCh;
  const size_t plane = static_cast<size_t>(d.yH) * d.yW;
  scales0.resize(static_cast<size_t>(halfCh) * plane);
  scales1.resize(static_cast<size_t>(halfCh) * plane);

  auto upsampled = [&](int c, int h, int w) -> int32_t {
    // c/4 selects the base z channel; h/4, w/4 undo the spatial repeat.
    const int zc = c / d.channelRepeat;
    const int zh = h / d.spatialRepeat;
    const int zw = w / d.spatialRepeat;
    const float v = std::fabs(
        fp16ToF32(zRaw[(static_cast<size_t>(zc) * d.zH + zh) * d.zW + zw]));
    int32_t idx = static_cast<int32_t>(v);  // trunc, matches astype(int32)
    if (idx < 0) idx = 0;
    const int32_t maxIdx = static_cast<int32_t>(scaleLevels) - 1;
    return idx > maxIdx ? maxIdx : idx;
  };

  for (int k = 0; k < halfCh; ++k) {
    for (int h = 0; h < d.yH; ++h) {
      for (int w = 0; w < d.yW; ++w) {
        const bool even = ((h + w) & 1) == 0;  // micro mask [[1,0],[0,1]]
        const int32_t a = upsampled(k, h, w);
        const int32_t b = upsampled(k + halfCh, h, w);
        const size_t at = (static_cast<size_t>(k) * d.yH + h) * d.yW + w;
        scales0[at] = even ? a : b;
        scales1[at] = even ? b : a;
      }
    }
  }
}

void yuv420ToTensor(const uint8_t* frame, int width, int height, uint16_t* out) {
  // 256-entry LUT: fp16 bits of i/255. Turns the whole conversion into loads.
  static const auto kLut = [] {
    std::array<uint16_t, 256> t{};
    for (int i = 0; i < 256; ++i) t[static_cast<size_t>(i)] = f32ToFp16(static_cast<float>(i) / 255.0f);
    return t;
  }();
  const size_t hw = static_cast<size_t>(width) * height;
  const uint8_t* yPlane = frame;
  const uint8_t* uPlane = frame + hw;
  const uint8_t* vPlane = frame + hw + hw / 4;
  for (size_t i = 0; i < hw; ++i) out[i] = kLut[yPlane[i]];
  const int hw2 = width / 2;
  for (int h = 0; h < height; ++h) {
    const int uvRow = (h / 2) * hw2;
    uint16_t* uRow = out + hw + static_cast<size_t>(h) * width;
    uint16_t* vRow = out + 2 * hw + static_cast<size_t>(h) * width;
    for (int w = 0; w < width; w += 2) {  // each UV sample covers 2 pixels
      const int uv = uvRow + w / 2;
      const uint16_t u = kLut[uPlane[uv]];
      const uint16_t v = kLut[vPlane[uv]];
      uRow[w] = u; uRow[w + 1] = u;
      vRow[w] = v; vRow[w + 1] = v;
    }
  }
}

std::vector<uint8_t> encodeBitstream(const PmfTables& t, const ModelDims& d,
                                     const uint16_t* yRaw0, const uint16_t* yRaw1,
                                     const uint16_t* zRaw, int qIndex,
                                     CoderWorkspace& ws) {
  extractScales(zRaw, d, t.scaleLevels, ws.scales0, ws.scales1);
  auto& scales0 = ws.scales0;
  auto& scales1 = ws.scales1;
  RansEncoder& enc = ws.enc;
  // Streaming encoder reverses on decode, so push y1, y0, z backwards to give
  // the decoder z, y0, y1 forward (receiver derives y-scales from z first).
  const size_t yN = scales0.size();
  for (size_t i = yN; i-- > 0;) {
    enc.push(t.gaussian, static_cast<uint32_t>(scales1[i]),
             static_cast<int32_t>(fp16ToF32(yRaw1[i])));
  }
  for (size_t i = yN; i-- > 0;) {
    enc.push(t.gaussian, static_cast<uint32_t>(scales0[i]),
             static_cast<int32_t>(fp16ToF32(yRaw0[i])));
  }
  const size_t zN = static_cast<size_t>(d.zCh) * d.zH * d.zW;
  const size_t zPlane = static_cast<size_t>(d.zH) * d.zW;
  for (size_t i = zN; i-- > 0;) {
    const uint32_t row = static_cast<uint32_t>(qIndex) * t.zChannels +
                         static_cast<uint32_t>(i / zPlane);
    enc.push(t.bitest, row, static_cast<int32_t>(fp16ToF32(zRaw[i])));
  }
  return enc.flush();
}

bool decodeLatents(const PmfTables& t, const ModelDims& d, const uint8_t* bs,
                   size_t bsSize, int qIndex, uint16_t* zRaw, uint16_t* yRaw0,
                   uint16_t* yRaw1, CoderWorkspace& ws) {
  RansDecoder dec(bs, bsSize);
  const size_t zN = static_cast<size_t>(d.zCh) * d.zH * d.zW;
  const size_t zPlane = static_cast<size_t>(d.zH) * d.zW;
  for (size_t i = 0; i < zN; ++i) {
    const uint32_t row = static_cast<uint32_t>(qIndex) * t.zChannels +
                         static_cast<uint32_t>(i / zPlane);
    zRaw[i] = f32ToFp16(static_cast<float>(dec.decode(t.bitest, row)));
  }
  extractScales(zRaw, d, t.scaleLevels, ws.scales0, ws.scales1);
  const size_t yN = ws.scales0.size();
  for (size_t i = 0; i < yN; ++i) {
    yRaw0[i] = f32ToFp16(static_cast<float>(
        dec.decode(t.gaussian, static_cast<uint32_t>(ws.scales0[i]))));
  }
  for (size_t i = 0; i < yN; ++i) {
    yRaw1[i] = f32ToFp16(static_cast<float>(
        dec.decode(t.gaussian, static_cast<uint32_t>(ws.scales1[i]))));
  }
  return dec.eofOk();
}

bool verifyBitstream(const PmfTables& t, const ModelDims& d,
                     const std::vector<uint8_t>& bs, const uint16_t* yRaw0,
                     const uint16_t* yRaw1, const uint16_t* zRaw, int qIndex) {
  std::vector<int32_t> scales0, scales1;
  extractScales(zRaw, d, t.scaleLevels, scales0, scales1);
  RansDecoder dec(bs.data(), bs.size());

  const size_t zN = static_cast<size_t>(d.zCh) * d.zH * d.zW;
  const size_t zPlane = static_cast<size_t>(d.zH) * d.zW;
  for (size_t i = 0; i < zN; ++i) {
    const uint32_t row = static_cast<uint32_t>(qIndex) * t.zChannels +
                         static_cast<uint32_t>(i / zPlane);
    if (dec.decode(t.bitest, row) != static_cast<int32_t>(fp16ToF32(zRaw[i]))) {
      fprintf(stderr, "z mismatch at %zu\n", i);
      return false;
    }
  }
  const size_t yN = scales0.size();
  for (size_t i = 0; i < yN; ++i) {
    if (dec.decode(t.gaussian, static_cast<uint32_t>(scales0[i])) !=
        static_cast<int32_t>(fp16ToF32(yRaw0[i]))) {
      fprintf(stderr, "y0 mismatch at %zu\n", i);
      return false;
    }
  }
  for (size_t i = 0; i < yN; ++i) {
    if (dec.decode(t.gaussian, static_cast<uint32_t>(scales1[i])) !=
        static_cast<int32_t>(fp16ToF32(yRaw1[i]))) {
      fprintf(stderr, "y1 mismatch at %zu\n", i);
      return false;
    }
  }
  if (!dec.eofOk()) {
    fprintf(stderr, "EOF check failed\n");
    return false;
  }
  return true;
}

}  // namespace mlvc
