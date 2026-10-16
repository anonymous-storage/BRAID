#ifndef BRAID_KERNELS_VECTOR_COMMON_H_
#define BRAID_KERNELS_VECTOR_COMMON_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include "src/core/gf256.h"

namespace braid::kernels {

struct NibbleTables {
  std::array<std::array<uint8_t, 16>, 256> lo{};
  std::array<std::array<uint8_t, 16>, 256> hi{};
};

static inline const NibbleTables& GetNibbleTables() {
  static const NibbleTables tables = [] {
    NibbleTables t;
    for (int c = 0; c < 256; ++c) {
      for (int n = 0; n < 16; ++n) {
        t.lo[c][n] = braid::core::GF256::Multiply(static_cast<uint8_t>(c), static_cast<uint8_t>(n));
        t.hi[c][n] = braid::core::GF256::Multiply(static_cast<uint8_t>(c), static_cast<uint8_t>(n << 4));
      }
    }
    return t;
  }();
  return tables;
}

#if defined(__AVX2__)
static inline __m256i MultiplyConstAvx2(__m256i value, uint8_t coefficient) {
  const auto& tables = GetNibbleTables();
  const __m256i lo_tbl =
      _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tables.lo[coefficient].data())));
  const __m256i hi_tbl =
      _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tables.hi[coefficient].data())));
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i lo = _mm256_and_si256(value, mask);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(value, 4), mask);
  return _mm256_xor_si256(_mm256_shuffle_epi8(lo_tbl, lo), _mm256_shuffle_epi8(hi_tbl, hi));
}
#endif

#if defined(__AVX512F__)
static inline __m512i MultiplyConstAvx512(__m512i value, uint8_t coefficient) {
  const auto& tables = GetNibbleTables();
  const __m128i lo128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tables.lo[coefficient].data()));
  const __m128i hi128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tables.hi[coefficient].data()));
  const __m512i lo_tbl = _mm512_broadcast_i32x4(lo128);
  const __m512i hi_tbl = _mm512_broadcast_i32x4(hi128);
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i lo = _mm512_and_si512(value, mask);
  const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(value, 4), mask);
  return _mm512_xor_si512(_mm512_shuffle_epi8(lo_tbl, lo), _mm512_shuffle_epi8(hi_tbl, hi));
}
#endif

#if defined(__AVX2__)
static inline __m256i XtimeAvx2(__m256i value) {
  const __m256i high_mask = _mm256_set1_epi8(static_cast<char>(0x80));
  const __m256i poly = _mm256_set1_epi8(static_cast<char>(0x39));
  const __m256i high = _mm256_and_si256(value, high_mask);
  const __m256i carry = _mm256_cmpeq_epi8(high, high_mask);
  const __m256i shifted = _mm256_add_epi8(value, value);
  return _mm256_xor_si256(shifted, _mm256_and_si256(carry, poly));
}
#endif

#if defined(__AVX512F__)
static inline __m512i XtimeAvx512(__m512i value) {
  const __m512i high_mask = _mm512_set1_epi8(static_cast<char>(0x80));
  const __m512i high = _mm512_and_si512(value, high_mask);
  const __mmask64 carry_mask =
      _mm512_cmpneq_epi8_mask(high, _mm512_setzero_si512());
  const __m512i shifted = _mm512_add_epi8(value, value);
  const __m512i poly = _mm512_maskz_set1_epi8(carry_mask, static_cast<char>(0x39));
  return _mm512_xor_si512(shifted, poly);
}
#endif

static inline void XorBytesScalar(uint8_t* dst, const uint8_t* src, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    dst[i] ^= src[i];
  }
}

}  // namespace braid::kernels

#endif  // BRAID_KERNELS_VECTOR_COMMON_H_
