#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

#include "src/core/gfni_matrix.h"
#include "src/kernels/vector_common.h"
#include "src/kernels/vector_entrypoints.h"
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
#include "src/runtime/avx512_plan_jit.h"
#endif

namespace braid::kernels {

namespace {

inline bool UsesBaseXGenerator(const braid::core::DecompositionTable& table) {
  return table.generator == braid::core::GF256::MultiplyByX(1);
}

inline __m512i MultiplyXPowAvx512(
    const braid::core::DecompositionTable& table, __m512i value, uint8_t power) {
  if (power == 0) {
    return value;
  }
  const uint8_t x_exponent = table.x_exponents[power];
  if (x_exponent == 0) {
    return value;
  }
  __m512i out = value;
  if (x_exponent <= 4) {
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = XtimeAvx512(out);
    }
    return out;
  }
  return MultiplyConstAvx512(value, table.x_powers[power]);
}

inline __m512i MultiplyCoeffAvx512(__m512i value, uint8_t coefficient, uint8_t x_exponent) {
  if (x_exponent == 0) {
    return value;
  }
  if (x_exponent <= 4) {
    __m512i out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = XtimeAvx512(out);
    }
    return out;
  }
  return MultiplyConstAvx512(value, coefficient);
}

inline __m512i AdvanceByGapAvx512(
    const braid::core::DecompositionTable& table, __m512i value, uint8_t gap) {
  if (gap == 0) {
    return value;
  }
  if (UsesBaseXGenerator(table) && gap == 1) {
    return XtimeAvx512(value);
  }
  if (UsesBaseXGenerator(table) && gap == 2) {
    return XtimeAvx512(XtimeAvx512(value));
  }
  return MultiplyXPowAvx512(table, value, gap);
}

inline uint8_t AdvanceByGapScalar(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t gap) {
  if (gap == 0) {
    return value;
  }
  if (UsesBaseXGenerator(table) && gap == 1) {
    return braid::core::GF256::MultiplyByX(value);
  }
  if (UsesBaseXGenerator(table) && gap == 2) {
    return braid::core::GF256::MultiplyByX(braid::core::GF256::MultiplyByX(value));
  }
  const uint8_t x_exponent = table.x_exponents[gap];
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.x_powers[gap], value);
}

inline __m512i AdvanceBetweenPlanesAvx512(
    const braid::core::DecompositionTable& table, __m512i value, uint8_t from_plane, uint8_t to_plane) {
  return MultiplyCoeffAvx512(
      value, table.transition_coeffs[from_plane][to_plane], table.transition_x_exponents[from_plane][to_plane]);
}

#if defined(BRAID_AVX512_GFNI_OBJECT)
inline __m512i XtimeGfni512(__m512i value) {
  const __m512i matrix = _mm512_set1_epi64(
      static_cast<long long>(UINT64_C(0x8001028488902040)));
  return _mm512_gf2p8affine_epi64_epi8(value, matrix, 0);
}

inline __m512i MultiplyConstGfni512(
    __m512i value, uint8_t coefficient) {
  if (coefficient == 1) {
    return value;
  }
  const __m512i matrix = _mm512_set1_epi64(
      static_cast<long long>(
          braid::core::GfniAffineMatrices()[coefficient]));
  return _mm512_gf2p8affine_epi64_epi8(value, matrix, 0);
}

inline __m512i AdvanceBetweenPlanesGfni512(
    const braid::core::DecompositionTable& table,
    __m512i value,
    uint8_t from_plane,
    uint8_t to_plane) {
  const uint8_t steps =
      table.transition_x_exponents[from_plane][to_plane];
  if (steps <= 4) {
    for (uint8_t step = 0; step < steps; ++step) {
      value = XtimeGfni512(value);
    }
    return value;
  }
  return MultiplyConstGfni512(
      value, table.transition_coeffs[from_plane][to_plane]);
}
#endif

inline __m512i XtimeSelected512(
    __m512i value, bool use_gfni) {
#if defined(BRAID_AVX512_GFNI_OBJECT)
  return use_gfni ? XtimeGfni512(value) : XtimeAvx512(value);
#else
  (void)use_gfni;
  return XtimeAvx512(value);
#endif
}

inline __m512i AdvanceBetweenPlanesSelected512(
    const braid::core::DecompositionTable& table,
    __m512i value,
    uint8_t from_plane,
    uint8_t to_plane,
    bool use_gfni) {
#if defined(BRAID_AVX512_GFNI_OBJECT)
  return use_gfni
             ? AdvanceBetweenPlanesGfni512(
                   table, value, from_plane, to_plane)
             : AdvanceBetweenPlanesAvx512(
                   table, value, from_plane, to_plane);
#else
  (void)use_gfni;
  return AdvanceBetweenPlanesAvx512(table, value, from_plane, to_plane);
#endif
}

inline uint8_t AdvanceBetweenPlanesScalar(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[from_plane][to_plane];
  if (x_exponent == 0) {
    return value;
  }
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.transition_coeffs[from_plane][to_plane], value);
}

inline __m512i AdvanceDirectAvx512(
    const braid::core::DecompositionTable& table, __m512i value, uint8_t from_plane, uint8_t to_plane) {
  return MultiplyCoeffAvx512(
      value, table.transition_coeffs[to_plane][from_plane], table.transition_x_exponents[to_plane][from_plane]);
}

inline uint8_t AdvanceDirectScalar512(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[to_plane][from_plane];
  if (x_exponent == 0) {
    return value;
  }
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.transition_coeffs[to_plane][from_plane], value);
}







uint8_t* AcquireThreadScratch512(size_t bytes) {
  struct AlignedScratch {
    uint8_t* data = nullptr;
    size_t size = 0;

    ~AlignedScratch() {
      if (data != nullptr) {
        ::operator delete[](data, std::align_val_t(64));
      }
    }
  };

  thread_local AlignedScratch scratch;
  if (scratch.size < bytes) {
    auto* replacement = static_cast<uint8_t*>(::operator new[](bytes, std::align_val_t(64)));
    if (scratch.data != nullptr) {
      ::operator delete[](scratch.data, std::align_val_t(64));
    }
    scratch.data = replacement;
    scratch.size = bytes;
  }
  return scratch.data;
}

inline void XorScratchVector512(uint8_t* dst_ptr, __m512i value) {
  const __m512i dst = _mm512_load_si512(reinterpret_cast<const void*>(dst_ptr));
  _mm512_store_si512(reinterpret_cast<void*>(dst_ptr), _mm512_xor_si512(dst, value));
}

template <size_t TargetCount>
inline void AccumulateScratchTargets512(
    const uint8_t* src,
    const uint8_t* prefetch_src,
    const std::array<uint8_t*, 12>& targets,
    size_t vec_limit,
    size_t chunk) {
  static_assert(TargetCount >= 1 && TargetCount <= 12);
  constexpr size_t kVec = 64;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    if (prefetch_src != nullptr) {
      __builtin_prefetch(prefetch_src + pos, 0, 1);
    }
    const __m512i value = _mm512_loadu_si512(reinterpret_cast<const void*>(src + pos));
    XorScratchVector512(targets[0] + pos, value);
    if constexpr (TargetCount >= 2) XorScratchVector512(targets[1] + pos, value);
    if constexpr (TargetCount >= 3) XorScratchVector512(targets[2] + pos, value);
    if constexpr (TargetCount >= 4) XorScratchVector512(targets[3] + pos, value);
    if constexpr (TargetCount >= 5) XorScratchVector512(targets[4] + pos, value);
    if constexpr (TargetCount >= 6) XorScratchVector512(targets[5] + pos, value);
    if constexpr (TargetCount >= 7) XorScratchVector512(targets[6] + pos, value);
    if constexpr (TargetCount >= 8) XorScratchVector512(targets[7] + pos, value);
    if constexpr (TargetCount >= 9) XorScratchVector512(targets[8] + pos, value);
    if constexpr (TargetCount >= 10) XorScratchVector512(targets[9] + pos, value);
    if constexpr (TargetCount >= 11) XorScratchVector512(targets[10] + pos, value);
    if constexpr (TargetCount >= 12) XorScratchVector512(targets[11] + pos, value);
  }
  for (size_t target = 0; target < TargetCount; ++target) {
    XorBytesScalar(targets[target] + vec_limit, src + vec_limit, chunk - vec_limit);
  }
}

template <size_t TargetCount>
inline void InitializeOrAccumulateScratchTargets512(
    const uint8_t* src,
    const uint8_t* prefetch_src,
    const std::array<uint8_t*, 12>& targets,
    uint16_t initialize_mask,
    size_t vec_limit,
    size_t chunk) {
  static_assert(TargetCount >= 1 && TargetCount <= 12);
  constexpr size_t kVec = 64;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    if (prefetch_src != nullptr) {
      __builtin_prefetch(prefetch_src + pos, 0, 1);
    }
    const __m512i value = _mm512_loadu_si512(reinterpret_cast<const void*>(src + pos));
    for (size_t target = 0; target < TargetCount; ++target) {
      if ((initialize_mask & (1u << target)) != 0) {
        _mm512_store_si512(reinterpret_cast<void*>(targets[target] + pos), value);
      } else {
        XorScratchVector512(targets[target] + pos, value);
      }
    }
  }
  for (size_t target = 0; target < TargetCount; ++target) {
    if ((initialize_mask & (1u << target)) != 0) {
      std::memcpy(targets[target] + vec_limit, src + vec_limit, chunk - vec_limit);
    } else {
      XorBytesScalar(targets[target] + vec_limit, src + vec_limit, chunk - vec_limit);
    }
  }
}

template <size_t Batch>
inline void XorAccumulators512(uint8_t mask, __m512i value, std::array<__m512i, Batch>& accs) {
  if constexpr (Batch == 4) {
    switch (mask & 0x0f) {
      case 0x0: return;
      case 0x1: accs[0] = _mm512_xor_si512(accs[0], value); return;
      case 0x2: accs[1] = _mm512_xor_si512(accs[1], value); return;
      case 0x3:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        return;
      case 0x4: accs[2] = _mm512_xor_si512(accs[2], value); return;
      case 0x5:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
      case 0x6:
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
      case 0x7:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
      case 0x8: accs[3] = _mm512_xor_si512(accs[3], value); return;
      case 0x9:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xA:
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xB:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xC:
        accs[2] = _mm512_xor_si512(accs[2], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xD:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xE:
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
      case 0xF:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        accs[3] = _mm512_xor_si512(accs[3], value);
        return;
    }
  } else if constexpr (Batch == 3) {
    switch (mask & 0x07) {
      case 0x0: return;
      case 0x1: accs[0] = _mm512_xor_si512(accs[0], value); return;
      case 0x2: accs[1] = _mm512_xor_si512(accs[1], value); return;
      case 0x3:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        return;
      case 0x4: accs[2] = _mm512_xor_si512(accs[2], value); return;
      case 0x5:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
      case 0x6:
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
      case 0x7:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        accs[2] = _mm512_xor_si512(accs[2], value);
        return;
    }
  } else if constexpr (Batch == 2) {
    switch (mask & 0x03) {
      case 0x0: return;
      case 0x1: accs[0] = _mm512_xor_si512(accs[0], value); return;
      case 0x2: accs[1] = _mm512_xor_si512(accs[1], value); return;
      case 0x3:
        accs[0] = _mm512_xor_si512(accs[0], value);
        accs[1] = _mm512_xor_si512(accs[1], value);
        return;
    }
  } else {
    for (size_t idx = 0; idx < Batch; ++idx) {
      if (mask & (1u << idx)) {
        accs[idx] = _mm512_xor_si512(accs[idx], value);
      }
    }
  }
}

inline void XorIntoPlane512(
    int plane,
    __m512i value,
    __m512i& p0,
    __m512i& p1,
    __m512i& p2,
    __m512i& p3,
    __m512i& p4,
    __m512i& p5,
    __m512i& p6,
    __m512i& p7,
    __m512i& p8,
    __m512i& p9,
    __m512i& p10,
    __m512i& p11,
    __m512i& p12,
    __m512i& p13,
    __m512i& p14,
    __m512i& p15,
    __m512i& p16) {
  switch (plane) {
    case 0: p0 = _mm512_xor_si512(p0, value); break;
    case 1: p1 = _mm512_xor_si512(p1, value); break;
    case 2: p2 = _mm512_xor_si512(p2, value); break;
    case 3: p3 = _mm512_xor_si512(p3, value); break;
    case 4: p4 = _mm512_xor_si512(p4, value); break;
    case 5: p5 = _mm512_xor_si512(p5, value); break;
    case 6: p6 = _mm512_xor_si512(p6, value); break;
    case 7: p7 = _mm512_xor_si512(p7, value); break;
    case 8: p8 = _mm512_xor_si512(p8, value); break;
    case 9: p9 = _mm512_xor_si512(p9, value); break;
    case 10: p10 = _mm512_xor_si512(p10, value); break;
    case 11: p11 = _mm512_xor_si512(p11, value); break;
    case 12: p12 = _mm512_xor_si512(p12, value); break;
    case 13: p13 = _mm512_xor_si512(p13, value); break;
    case 14: p14 = _mm512_xor_si512(p14, value); break;
    case 15: p15 = _mm512_xor_si512(p15, value); break;
    case 16: p16 = _mm512_xor_si512(p16, value); break;
    default: break;
  }
}

inline __m512i ReducePlanes512(
    const braid::core::DecompositionTable& table,
    __m512i p0,
    __m512i p1,
    __m512i p2,
    __m512i p3,
    __m512i p4,
    __m512i p5,
    __m512i p6,
    __m512i p7,
    __m512i p8,
    __m512i p9,
    __m512i p10,
    __m512i p11,
    __m512i p12,
    __m512i p13,
    __m512i p14,
    __m512i p15,
    __m512i p16) {
  __m512i acc = p16;
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p15);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p14);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p13);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p12);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p11);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p10);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p9);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p8);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p7);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p6);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p5);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p4);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p3);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p2);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p1);
  acc = _mm512_xor_si512(AdvanceByGapAvx512(table, acc, 1), p0);
  return acc;
}

inline void ReduceTwoSparseRows512(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 34>& plane_ptrs,
    const std::array<std::array<uint8_t, 17>, 2>& used_planes,
    const std::array<uint8_t, 2>& used_plane_counts,
    size_t pos,
    bool use_gfni,
    std::array<__m512i, 2>& acc) {
  std::array<int, 2> next_idx{
      static_cast<int>(used_plane_counts[0]) - 2,
      static_cast<int>(used_plane_counts[1]) - 2,
  };
  std::array<uint8_t, 2> current_planes{};
  for (size_t b = 0; b < 2; ++b) {
    if (used_plane_counts[b] == 0) {
      acc[b] = _mm512_setzero_si512();
      current_planes[b] = 0;
      continue;
    }
    current_planes[b] = used_planes[b][used_plane_counts[b] - 1];
    acc[b] = _mm512_load_si512(
        reinterpret_cast<const void*>(plane_ptrs[b * 17 + static_cast<size_t>(current_planes[b])] + pos));
  }

  while (next_idx[0] >= 0 || next_idx[1] >= 0) {
    if (next_idx[0] >= 0) {
      const uint8_t next_plane = used_planes[0][static_cast<size_t>(next_idx[0]--)];
      acc[0] = AdvanceBetweenPlanesSelected512(
          table, acc[0], current_planes[0], next_plane, use_gfni);
      acc[0] = _mm512_xor_si512(
          acc[0],
          _mm512_load_si512(reinterpret_cast<const void*>(plane_ptrs[static_cast<size_t>(next_plane)] + pos)));
      current_planes[0] = next_plane;
    }
    if (next_idx[1] >= 0) {
      const uint8_t next_plane = used_planes[1][static_cast<size_t>(next_idx[1]--)];
      acc[1] = AdvanceBetweenPlanesSelected512(
          table, acc[1], current_planes[1], next_plane, use_gfni);
      acc[1] = _mm512_xor_si512(
          acc[1],
          _mm512_load_si512(
              reinterpret_cast<const void*>(plane_ptrs[17 + static_cast<size_t>(next_plane)] + pos)));
      current_planes[1] = next_plane;
    }
  }

  if (used_plane_counts[0] != 0 && current_planes[0] != 0) {
    acc[0] = AdvanceBetweenPlanesSelected512(
        table, acc[0], current_planes[0], 0, use_gfni);
  }
  if (used_plane_counts[1] != 0 && current_planes[1] != 0) {
    acc[1] = AdvanceBetweenPlanesSelected512(
        table, acc[1], current_planes[1], 0, use_gfni);
  }
}

inline void ReduceTwoSparseRowsScalar(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 34>& plane_ptrs,
    const std::array<std::array<uint8_t, 17>, 2>& used_planes,
    const std::array<uint8_t, 2>& used_plane_counts,
    size_t pos,
    std::array<uint8_t, 2>& acc) {
  std::array<int, 2> next_idx{
      static_cast<int>(used_plane_counts[0]) - 2,
      static_cast<int>(used_plane_counts[1]) - 2,
  };
  std::array<uint8_t, 2> current_planes{};
  for (size_t b = 0; b < 2; ++b) {
    if (used_plane_counts[b] == 0) {
      acc[b] = 0;
      current_planes[b] = 0;
      continue;
    }
    current_planes[b] = used_planes[b][used_plane_counts[b] - 1];
    acc[b] = plane_ptrs[b * 17 + static_cast<size_t>(current_planes[b])][pos];
  }

  while (next_idx[0] >= 0 || next_idx[1] >= 0) {
    if (next_idx[0] >= 0) {
      const uint8_t next_plane = used_planes[0][static_cast<size_t>(next_idx[0]--)];
      acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_planes[0], next_plane);
      acc[0] ^= plane_ptrs[static_cast<size_t>(next_plane)][pos];
      current_planes[0] = next_plane;
    }
    if (next_idx[1] >= 0) {
      const uint8_t next_plane = used_planes[1][static_cast<size_t>(next_idx[1]--)];
      acc[1] = AdvanceBetweenPlanesScalar(table, acc[1], current_planes[1], next_plane);
      acc[1] ^= plane_ptrs[17 + static_cast<size_t>(next_plane)][pos];
      current_planes[1] = next_plane;
    }
  }

  if (used_plane_counts[0] != 0 && current_planes[0] != 0) {
    acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_planes[0], 0);
  }
  if (used_plane_counts[1] != 0 && current_planes[1] != 0) {
    acc[1] = AdvanceBetweenPlanesScalar(table, acc[1], current_planes[1], 0);
  }
}

template <size_t Batch>
inline std::array<__m512i, Batch> ReconstructFactorOutputs512(
    const braid::runtime::FactorBatchSchedule& schedule,
    const std::array<__m512i, Batch>& transformed) {
  std::array<__m512i, Batch> outputs{};
  for (size_t output = 0; output < Batch; ++output) {
    const uint8_t mask = schedule.reconstruction_masks[output];
    bool initialized = false;
    for (size_t input = 0; input < Batch; ++input) {
      if ((mask & (1u << input)) == 0) {
        continue;
      }
      if (!initialized) {
        outputs[output] = transformed[input];
        initialized = true;
      } else {
        outputs[output] =
            _mm512_xor_si512(outputs[output], transformed[input]);
      }
    }
    if (!initialized) {
      outputs[output] = _mm512_setzero_si512();
    }
  }
  return outputs;
}

template <size_t Batch>
inline std::array<uint8_t, Batch> ReconstructFactorOutputsScalar(
    const braid::runtime::FactorBatchSchedule& schedule,
    const std::array<uint8_t, Batch>& transformed) {
  std::array<uint8_t, Batch> outputs{};
  for (size_t output = 0; output < Batch; ++output) {
    const uint8_t mask = schedule.reconstruction_masks[output];
    for (size_t input = 0; input < Batch; ++input) {
      if ((mask & (1u << input)) != 0) {
        outputs[output] ^= transformed[input];
      }
    }
  }
  return outputs;
}

void ExecuteBraidBatch1Register512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t out_idx = 0; out_idx < plan.rows.size(); ++out_idx) {
    const auto& row = plan.rows[out_idx];
    uint8_t* out = outputs[out_idx];
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      __m512i p0 = _mm512_setzero_si512();
      __m512i p1 = _mm512_setzero_si512();
      __m512i p2 = _mm512_setzero_si512();
      __m512i p3 = _mm512_setzero_si512();
      __m512i p4 = _mm512_setzero_si512();
      __m512i p5 = _mm512_setzero_si512();
      __m512i p6 = _mm512_setzero_si512();
      __m512i p7 = _mm512_setzero_si512();
      __m512i p8 = _mm512_setzero_si512();
      __m512i p9 = _mm512_setzero_si512();
      __m512i p10 = _mm512_setzero_si512();
      __m512i p11 = _mm512_setzero_si512();
      __m512i p12 = _mm512_setzero_si512();
      __m512i p13 = _mm512_setzero_si512();
      __m512i p14 = _mm512_setzero_si512();
      __m512i p15 = _mm512_setzero_si512();
      __m512i p16 = _mm512_setzero_si512();

      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t weight = row.alpha_weights[in];
        if (weight == 0) {
          continue;
        }
        if (in + 8 < plan.input_indices.size()) {
          __builtin_prefetch(inputs[in + 8] + pos, 0, 1);
        }
        const __m512i value = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + pos));
        const auto& positions = row.alpha_positions[in];
        XorIntoPlane512(
            positions[0], value, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
        if (weight >= 2) {
          XorIntoPlane512(
              positions[1], value, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
        }
        if (weight >= 3) {
          XorIntoPlane512(
              positions[2], value, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
        }
      }

      __m512i reduced = ReducePlanes512(
          table, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), reduced);
    }

    for (size_t pos = vec_limit; pos < block_size; ++pos) {
      uint8_t planes[17] = {};
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t weight = row.alpha_weights[in];
        if (weight == 0) {
          continue;
        }
        const uint8_t value = inputs[in][pos];
        const auto& positions = row.alpha_positions[in];
        planes[positions[0]] ^= value;
        if (weight >= 2) {
          planes[positions[1]] ^= value;
        }
        if (weight >= 3) {
          planes[positions[2]] ^= value;
        }
      }
      uint8_t acc = planes[16];
      for (int plane = 15; plane >= 0; --plane) {
        acc = AdvanceByGapScalar(table, acc, 1) ^ planes[static_cast<size_t>(plane)];
      }
      out[pos] = acc;
    }
  }
}

void ExecuteBraidPlaneStreaming512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t out_idx = 0; out_idx < plan.rows.size(); ++out_idx) {
    const auto& row = plan.rows[out_idx];
    uint8_t* out = outputs[out_idx];
    int highest_plane = 16;
    while (highest_plane >= 0 &&
           row.plane_input_indices[static_cast<size_t>(highest_plane)].empty()) {
      --highest_plane;
    }
    if (highest_plane < 0) {
      std::memset(out, 0, block_size);
      continue;
    }

    size_t pos = 0;
    for (; pos + 2 * kVec <= vec_limit; pos += 2 * kVec) {
      __m512i acc0 = _mm512_setzero_si512();
      __m512i acc1 = _mm512_setzero_si512();
      uint8_t current_plane = static_cast<uint8_t>(highest_plane);
      bool initialized = false;
      for (int plane = highest_plane; plane >= 0; --plane) {
        const auto& routes =
            row.plane_input_indices[static_cast<size_t>(plane)];
        if (routes.empty()) {
          continue;
        }
        __m512i plane0 = _mm512_setzero_si512();
        __m512i plane1 = _mm512_setzero_si512();
        for (uint16_t input : routes) {
          plane0 = _mm512_xor_si512(
              plane0,
              _mm512_loadu_si512(
                  reinterpret_cast<const void*>(inputs[input] + pos)));
          plane1 = _mm512_xor_si512(
              plane1,
              _mm512_loadu_si512(
                  reinterpret_cast<const void*>(inputs[input] + pos + kVec)));
        }
        if (!initialized) {
          acc0 = plane0;
          acc1 = plane1;
          initialized = true;
        } else {
          acc0 = _mm512_xor_si512(
              AdvanceBetweenPlanesAvx512(
                  table, acc0, current_plane, static_cast<uint8_t>(plane)),
              plane0);
          acc1 = _mm512_xor_si512(
              AdvanceBetweenPlanesAvx512(
                  table, acc1, current_plane, static_cast<uint8_t>(plane)),
              plane1);
        }
        current_plane = static_cast<uint8_t>(plane);
      }
      if (current_plane != 0) {
        acc0 = AdvanceBetweenPlanesAvx512(table, acc0, current_plane, 0);
        acc1 = AdvanceBetweenPlanesAvx512(table, acc1, current_plane, 0);
      }
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), acc0);
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos + kVec), acc1);
    }

    for (; pos < vec_limit; pos += kVec) {
      __m512i acc = _mm512_setzero_si512();
      uint8_t current_plane = static_cast<uint8_t>(highest_plane);
      bool initialized = false;
      for (int plane = highest_plane; plane >= 0; --plane) {
        const auto& routes =
            row.plane_input_indices[static_cast<size_t>(plane)];
        if (routes.empty()) {
          continue;
        }
        __m512i plane_acc = _mm512_setzero_si512();
        for (uint16_t input : routes) {
          plane_acc = _mm512_xor_si512(
              plane_acc,
              _mm512_loadu_si512(
                  reinterpret_cast<const void*>(inputs[input] + pos)));
        }
        if (!initialized) {
          acc = plane_acc;
          initialized = true;
        } else {
          acc = _mm512_xor_si512(
              AdvanceBetweenPlanesAvx512(
                  table, acc, current_plane, static_cast<uint8_t>(plane)),
              plane_acc);
        }
        current_plane = static_cast<uint8_t>(plane);
      }
      if (current_plane != 0) {
        acc = AdvanceBetweenPlanesAvx512(table, acc, current_plane, 0);
      }
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), acc);
    }

    for (; pos < block_size; ++pos) {
      uint8_t acc = 0;
      uint8_t current_plane = static_cast<uint8_t>(highest_plane);
      bool initialized = false;
      for (int plane = highest_plane; plane >= 0; --plane) {
        const auto& routes =
            row.plane_input_indices[static_cast<size_t>(plane)];
        if (routes.empty()) {
          continue;
        }
        uint8_t plane_acc = 0;
        for (uint16_t input : routes) {
          plane_acc ^= inputs[input][pos];
        }
        if (!initialized) {
          acc = plane_acc;
          initialized = true;
        } else {
          acc = AdvanceBetweenPlanesScalar(
                    table, acc, current_plane, static_cast<uint8_t>(plane)) ^
                plane_acc;
        }
        current_plane = static_cast<uint8_t>(plane);
      }
      if (current_plane != 0) {
        acc = AdvanceBetweenPlanesScalar(table, acc, current_plane, 0);
      }
      out[pos] = acc;
    }
  }
}

void ExecuteBraidPolynomialRegister512(
    const braid::runtime::Plan& plan,
    size_t base_out,
    size_t output_count,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const bool use_gfni =
      plan.microkernel_mode ==
      BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI;
  const size_t vec_limit = (block_size / kVec) * kVec;
  const size_t output_end = std::min(plan.rows.size(), base_out + output_count);
  for (size_t out_idx = base_out; out_idx < output_end; ++out_idx) {
    const auto& row = plan.rows[out_idx];
    uint8_t* out = outputs[out_idx];
    size_t pos = 0;
    for (; pos + 2 * kVec <= vec_limit; pos += 2 * kVec) {
      __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
      __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
      __m512i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
      __m512i b4 = a0, b5 = a0, b6 = a0, b7 = a0;

      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t coefficient = row.coefficients[in];
        if (coefficient == 0) {
          continue;
        }
        if (in + 8 < plan.input_indices.size()) {
          __builtin_prefetch(inputs[in + 8] + pos, 0, 1);
        }
        const __m512i value0 =
            _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + pos));
        const __m512i value1 =
            _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + pos + kVec));
        if (coefficient & 0x01u) { a0 = _mm512_xor_si512(a0, value0); b0 = _mm512_xor_si512(b0, value1); }
        if (coefficient & 0x02u) { a1 = _mm512_xor_si512(a1, value0); b1 = _mm512_xor_si512(b1, value1); }
        if (coefficient & 0x04u) { a2 = _mm512_xor_si512(a2, value0); b2 = _mm512_xor_si512(b2, value1); }
        if (coefficient & 0x08u) { a3 = _mm512_xor_si512(a3, value0); b3 = _mm512_xor_si512(b3, value1); }
        if (coefficient & 0x10u) { a4 = _mm512_xor_si512(a4, value0); b4 = _mm512_xor_si512(b4, value1); }
        if (coefficient & 0x20u) { a5 = _mm512_xor_si512(a5, value0); b5 = _mm512_xor_si512(b5, value1); }
        if (coefficient & 0x40u) { a6 = _mm512_xor_si512(a6, value0); b6 = _mm512_xor_si512(b6, value1); }
        if (coefficient & 0x80u) { a7 = _mm512_xor_si512(a7, value0); b7 = _mm512_xor_si512(b7, value1); }
      }

      __m512i acc0 = a7;
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a6);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a5);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a4);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a3);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a2);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a1);
      acc0 = _mm512_xor_si512(XtimeSelected512(acc0, use_gfni), a0);
      __m512i acc1 = b7;
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b6);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b5);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b4);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b3);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b2);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b1);
      acc1 = _mm512_xor_si512(XtimeSelected512(acc1, use_gfni), b0);
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), acc0);
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos + kVec), acc1);
    }

    for (; pos < vec_limit; pos += kVec) {
      __m512i p0 = _mm512_setzero_si512();
      __m512i p1 = _mm512_setzero_si512();
      __m512i p2 = _mm512_setzero_si512();
      __m512i p3 = _mm512_setzero_si512();
      __m512i p4 = _mm512_setzero_si512();
      __m512i p5 = _mm512_setzero_si512();
      __m512i p6 = _mm512_setzero_si512();
      __m512i p7 = _mm512_setzero_si512();

      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t coefficient = row.coefficients[in];
        if (coefficient == 0) {
          continue;
        }
        if (in + 8 < plan.input_indices.size()) {
          __builtin_prefetch(inputs[in + 8] + pos, 0, 1);
        }
        const __m512i value = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + pos));
        switch (coefficient & 0x0fu) {
          case 0x0: break;
          case 0x1: p0 = _mm512_xor_si512(p0, value); break;
          case 0x2: p1 = _mm512_xor_si512(p1, value); break;
          case 0x3: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); break;
          case 0x4: p2 = _mm512_xor_si512(p2, value); break;
          case 0x5: p0 = _mm512_xor_si512(p0, value); p2 = _mm512_xor_si512(p2, value); break;
          case 0x6: p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); break;
          case 0x7: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); break;
          case 0x8: p3 = _mm512_xor_si512(p3, value); break;
          case 0x9: p0 = _mm512_xor_si512(p0, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xa: p1 = _mm512_xor_si512(p1, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xb: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xc: p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xd: p0 = _mm512_xor_si512(p0, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xe: p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
          case 0xf: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
        }
        switch (coefficient >> 4) {
          case 0x0: break;
          case 0x1: p4 = _mm512_xor_si512(p4, value); break;
          case 0x2: p5 = _mm512_xor_si512(p5, value); break;
          case 0x3: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); break;
          case 0x4: p6 = _mm512_xor_si512(p6, value); break;
          case 0x5: p4 = _mm512_xor_si512(p4, value); p6 = _mm512_xor_si512(p6, value); break;
          case 0x6: p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); break;
          case 0x7: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); break;
          case 0x8: p7 = _mm512_xor_si512(p7, value); break;
          case 0x9: p4 = _mm512_xor_si512(p4, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xa: p5 = _mm512_xor_si512(p5, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xb: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xc: p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xd: p4 = _mm512_xor_si512(p4, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xe: p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
          case 0xf: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
        }
      }

      __m512i acc = p7;
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p6);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p5);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p4);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p3);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p2);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p1);
      acc = _mm512_xor_si512(XtimeSelected512(acc, use_gfni), p0);
      _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), acc);
    }

    for (size_t pos = vec_limit; pos < block_size; ++pos) {
      uint8_t planes[8] = {};
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t coefficient = row.coefficients[in];
        const uint8_t value = inputs[in][pos];
        for (size_t plane = 0; plane < 8; ++plane) {
          if ((coefficient & (1u << plane)) != 0) {
            planes[plane] ^= value;
          }
        }
      }
      uint8_t acc = planes[7];
      for (int plane = 6; plane >= 0; --plane) {
        acc = braid::core::GF256::MultiplyByX(acc) ^ planes[static_cast<size_t>(plane)];
      }
      out[pos] = acc;
    }
  }
}

inline void AccumulatePolynomialBasisSwitch512(
    uint8_t coefficient,
    __m512i value,
    __m512i& p0,
    __m512i& p1,
    __m512i& p2,
    __m512i& p3,
    __m512i& p4,
    __m512i& p5,
    __m512i& p6,
    __m512i& p7) {
  switch (coefficient & 0x0fu) {
    case 0x0: break;
    case 0x1: p0 = _mm512_xor_si512(p0, value); break;
    case 0x2: p1 = _mm512_xor_si512(p1, value); break;
    case 0x3: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); break;
    case 0x4: p2 = _mm512_xor_si512(p2, value); break;
    case 0x5: p0 = _mm512_xor_si512(p0, value); p2 = _mm512_xor_si512(p2, value); break;
    case 0x6: p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); break;
    case 0x7: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); break;
    case 0x8: p3 = _mm512_xor_si512(p3, value); break;
    case 0x9: p0 = _mm512_xor_si512(p0, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xa: p1 = _mm512_xor_si512(p1, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xb: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xc: p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xd: p0 = _mm512_xor_si512(p0, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xe: p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
    case 0xf: p0 = _mm512_xor_si512(p0, value); p1 = _mm512_xor_si512(p1, value); p2 = _mm512_xor_si512(p2, value); p3 = _mm512_xor_si512(p3, value); break;
  }
  switch (coefficient >> 4) {
    case 0x0: break;
    case 0x1: p4 = _mm512_xor_si512(p4, value); break;
    case 0x2: p5 = _mm512_xor_si512(p5, value); break;
    case 0x3: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); break;
    case 0x4: p6 = _mm512_xor_si512(p6, value); break;
    case 0x5: p4 = _mm512_xor_si512(p4, value); p6 = _mm512_xor_si512(p6, value); break;
    case 0x6: p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); break;
    case 0x7: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); break;
    case 0x8: p7 = _mm512_xor_si512(p7, value); break;
    case 0x9: p4 = _mm512_xor_si512(p4, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xa: p5 = _mm512_xor_si512(p5, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xb: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xc: p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xd: p4 = _mm512_xor_si512(p4, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xe: p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
    case 0xf: p4 = _mm512_xor_si512(p4, value); p5 = _mm512_xor_si512(p5, value); p6 = _mm512_xor_si512(p6, value); p7 = _mm512_xor_si512(p7, value); break;
  }
}

inline void AccumulatePolynomialBasisBits512(
    uint8_t coefficient,
    __m512i value,
    __m512i& p0,
    __m512i& p1,
    __m512i& p2,
    __m512i& p3,
    __m512i& p4,
    __m512i& p5,
    __m512i& p6,
    __m512i& p7) {
  if (coefficient & 0x01u) p0 = _mm512_xor_si512(p0, value);
  if (coefficient & 0x02u) p1 = _mm512_xor_si512(p1, value);
  if (coefficient & 0x04u) p2 = _mm512_xor_si512(p2, value);
  if (coefficient & 0x08u) p3 = _mm512_xor_si512(p3, value);
  if (coefficient & 0x10u) p4 = _mm512_xor_si512(p4, value);
  if (coefficient & 0x20u) p5 = _mm512_xor_si512(p5, value);
  if (coefficient & 0x40u) p6 = _mm512_xor_si512(p6, value);
  if (coefficient & 0x80u) p7 = _mm512_xor_si512(p7, value);
}

template <bool UseBitTests>
inline void AccumulatePolynomialBasis512(
    uint8_t coefficient,
    __m512i value,
    __m512i& p0,
    __m512i& p1,
    __m512i& p2,
    __m512i& p3,
    __m512i& p4,
    __m512i& p5,
    __m512i& p6,
    __m512i& p7) {
  if constexpr (UseBitTests) {
    AccumulatePolynomialBasisBits512(coefficient, value, p0, p1, p2, p3, p4, p5, p6, p7);
  } else {
    AccumulatePolynomialBasisSwitch512(coefficient, value, p0, p1, p2, p3, p4, p5, p6, p7);
  }
}

inline __m512i ReducePolynomialBasis512(
    __m512i p0,
    __m512i p1,
    __m512i p2,
    __m512i p3,
    __m512i p4,
    __m512i p5,
    __m512i p6,
    __m512i p7) {
  __m512i acc = p7;
  acc = _mm512_xor_si512(XtimeAvx512(acc), p6);
  acc = _mm512_xor_si512(XtimeAvx512(acc), p5);
  acc = _mm512_xor_si512(XtimeAvx512(acc), p4);
  acc = _mm512_xor_si512(XtimeAvx512(acc), p3);
  acc = _mm512_xor_si512(XtimeAvx512(acc), p2);
  acc = _mm512_xor_si512(XtimeAvx512(acc), p1);
  return _mm512_xor_si512(XtimeAvx512(acc), p0);
}

template <bool UseBitTests>
void ExecuteBraidPolynomialPair512(
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const auto& row0 = plan.rows[base_out];
  const auto& row1 = plan.rows[base_out + 1];
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __m512i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    __m512i b4 = a0, b5 = a0, b6 = a0, b7 = a0;
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      if (input + 8 < plan.input_indices.size()) {
        __builtin_prefetch(inputs[input + 8] + pos, 0, 1);
      }
      const __m512i value =
          _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[input] + pos));
      AccumulatePolynomialBasis512<UseBitTests>(
          row0.coefficients[input], value, a0, a1, a2, a3, a4, a5, a6, a7);
      AccumulatePolynomialBasis512<UseBitTests>(
          row1.coefficients[input], value, b0, b1, b2, b3, b4, b5, b6, b7);
    }
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out] + pos),
        ReducePolynomialBasis512(a0, a1, a2, a3, a4, a5, a6, a7));
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out + 1] + pos),
        ReducePolynomialBasis512(b0, b1, b2, b3, b4, b5, b6, b7));
  }

  for (size_t pos = vec_limit; pos < block_size; ++pos) {
    std::array<uint8_t, 8> planes0{};
    std::array<uint8_t, 8> planes1{};
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      const uint8_t value = inputs[input][pos];
      const uint8_t coefficient0 = row0.coefficients[input];
      const uint8_t coefficient1 = row1.coefficients[input];
      for (size_t plane = 0; plane < 8; ++plane) {
        if (coefficient0 & (1u << plane)) planes0[plane] ^= value;
        if (coefficient1 & (1u << plane)) planes1[plane] ^= value;
      }
    }
    uint8_t acc0 = planes0[7];
    uint8_t acc1 = planes1[7];
    for (int plane = 6; plane >= 0; --plane) {
      acc0 = braid::core::GF256::MultiplyByX(acc0) ^ planes0[static_cast<size_t>(plane)];
      acc1 = braid::core::GF256::MultiplyByX(acc1) ^ planes1[static_cast<size_t>(plane)];
    }
    outputs[base_out][pos] = acc0;
    outputs[base_out + 1][pos] = acc1;
  }
}

template <bool UseBitTests>
void ExecuteBraidPolynomialTriple512(
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const auto& row0 = plan.rows[base_out];
  const auto& row1 = plan.rows[base_out + 1];
  const auto& row2 = plan.rows[base_out + 2];
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __m512i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    __m512i b4 = a0, b5 = a0, b6 = a0, b7 = a0;
    __m512i c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    __m512i c4 = a0, c5 = a0, c6 = a0, c7 = a0;
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      if (input + 8 < plan.input_indices.size()) {
        __builtin_prefetch(inputs[input + 8] + pos, 0, 1);
      }
      const __m512i value =
          _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[input] + pos));
      AccumulatePolynomialBasis512<UseBitTests>(
          row0.coefficients[input], value, a0, a1, a2, a3, a4, a5, a6, a7);
      AccumulatePolynomialBasis512<UseBitTests>(
          row1.coefficients[input], value, b0, b1, b2, b3, b4, b5, b6, b7);
      AccumulatePolynomialBasis512<UseBitTests>(
          row2.coefficients[input], value, c0, c1, c2, c3, c4, c5, c6, c7);
    }
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out] + pos),
        ReducePolynomialBasis512(a0, a1, a2, a3, a4, a5, a6, a7));
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out + 1] + pos),
        ReducePolynomialBasis512(b0, b1, b2, b3, b4, b5, b6, b7));
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out + 2] + pos),
        ReducePolynomialBasis512(c0, c1, c2, c3, c4, c5, c6, c7));
  }

  for (size_t pos = vec_limit; pos < block_size; ++pos) {
    std::array<std::array<uint8_t, 8>, 3> planes{};
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      const uint8_t value = inputs[input][pos];
      const std::array<uint8_t, 3> coefficients{
          row0.coefficients[input], row1.coefficients[input], row2.coefficients[input]};
      for (size_t output = 0; output < 3; ++output) {
        for (size_t plane = 0; plane < 8; ++plane) {
          if (coefficients[output] & (1u << plane)) planes[output][plane] ^= value;
        }
      }
    }
    for (size_t output = 0; output < 3; ++output) {
      uint8_t acc = planes[output][7];
      for (int plane = 6; plane >= 0; --plane) {
        acc = braid::core::GF256::MultiplyByX(acc) ^ planes[output][static_cast<size_t>(plane)];
      }
      outputs[base_out + output][pos] = acc;
    }
  }
}

inline __m512i ReduceAdaptiveBasis512(
    const braid::runtime::OutputPlanRow& row,
    __m512i p0,
    __m512i p1,
    __m512i p2,
    __m512i p3,
    __m512i p4,
    __m512i p5,
    __m512i p6,
    __m512i p7) {
  __m512i result = ReducePolynomialBasis512(p0, p1, p2, p3, p4, p5, p6, p7);
  const uint8_t scale = row.basis_elements[0];
  return scale == 1 ? result : MultiplyConstAvx512(result, scale);
}

template <bool UseBitTests>
inline void AccumulateAdaptiveBasisRow512(
    const braid::runtime::OutputPlanRow& row,
    size_t input,
    __m512i value,
    std::array<__m512i, 8>& planes) {
  AccumulatePolynomialBasis512<UseBitTests>(
      row.basis_coordinates[input],
      value,
      planes[0],
      planes[1],
      planes[2],
      planes[3],
      planes[4],
      planes[5],
      planes[6],
      planes[7]);
}

inline __m512i ReduceAdaptiveBasisArray512(
    const braid::runtime::OutputPlanRow& row,
    const std::array<__m512i, 8>& planes) {
  return ReduceAdaptiveBasis512(
      row,
      planes[0],
      planes[1],
      planes[2],
      planes[3],
      planes[4],
      planes[5],
      planes[6],
      planes[7]);
}

template <bool UseBitTests>
void ExecuteBraidAdaptiveBasisPair512(
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const auto& row0 = plan.rows[base_out];
  const auto& row1 = plan.rows[base_out + 1];
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __m512i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    __m512i b4 = a0, b5 = a0, b6 = a0, b7 = a0;
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      if (input + 8 < plan.input_indices.size()) {
        __builtin_prefetch(inputs[input + 8] + pos, 0, 1);
      }
      const __m512i value =
          _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[input] + pos));
      AccumulatePolynomialBasis512<UseBitTests>(
          row0.basis_coordinates[input], value, a0, a1, a2, a3, a4, a5, a6, a7);
      AccumulatePolynomialBasis512<UseBitTests>(
          row1.basis_coordinates[input], value, b0, b1, b2, b3, b4, b5, b6, b7);
    }
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out] + pos),
        ReduceAdaptiveBasis512(row0, a0, a1, a2, a3, a4, a5, a6, a7));
    _mm512_storeu_si512(
        reinterpret_cast<void*>(outputs[base_out + 1] + pos),
        ReduceAdaptiveBasis512(row1, b0, b1, b2, b3, b4, b5, b6, b7));
  }

  for (size_t pos = vec_limit; pos < block_size; ++pos) {
    std::array<uint8_t, 8> planes0{};
    std::array<uint8_t, 8> planes1{};
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      const uint8_t value = inputs[input][pos];
      const uint8_t coordinate0 = row0.basis_coordinates[input];
      const uint8_t coordinate1 = row1.basis_coordinates[input];
      for (size_t plane = 0; plane < 8; ++plane) {
        if ((coordinate0 & (1u << plane)) != 0) planes0[plane] ^= value;
        if ((coordinate1 & (1u << plane)) != 0) planes1[plane] ^= value;
      }
    }
    uint8_t result0 = 0;
    uint8_t result1 = 0;
    for (size_t plane = 0; plane < 8; ++plane) {
      result0 ^= braid::core::GF256::Multiply(row0.basis_elements[plane], planes0[plane]);
      result1 ^= braid::core::GF256::Multiply(row1.basis_elements[plane], planes1[plane]);
    }
    outputs[base_out][pos] = result0;
    outputs[base_out + 1][pos] = result1;
  }
}

template <size_t Batch, bool UseBitTests>
void ExecuteBraidAdaptiveBasisLut512(
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  static_assert(Batch >= 1 && Batch <= 3);
  constexpr size_t kVec = 64;
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t pos = 0; pos < vec_limit; pos += kVec) {
    std::array<std::array<__m512i, 8>, Batch> planes{};
    const __m512i zero = _mm512_setzero_si512();
    for (auto& row_planes : planes) {
      row_planes.fill(zero);
    }
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      if (input + 8 < plan.input_indices.size()) {
        __builtin_prefetch(inputs[input + 8] + pos, 0, 1);
      }
      const __m512i value =
          _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[input] + pos));
      AccumulateAdaptiveBasisRow512<UseBitTests>(
          plan.rows[base_out], input, value, planes[0]);
      if constexpr (Batch >= 2) {
        AccumulateAdaptiveBasisRow512<UseBitTests>(
            plan.rows[base_out + 1], input, value, planes[1]);
      }
      if constexpr (Batch >= 3) {
        AccumulateAdaptiveBasisRow512<UseBitTests>(
            plan.rows[base_out + 2], input, value, planes[2]);
      }
    }
    for (size_t output = 0; output < Batch; ++output) {
      const __m512i result =
          ReduceAdaptiveBasisArray512(plan.rows[base_out + output], planes[output]);
      _mm512_storeu_si512(
          reinterpret_cast<void*>(outputs[base_out + output] + pos),
          result);
    }
  }

  for (size_t pos = vec_limit; pos < block_size; ++pos) {
    std::array<std::array<uint8_t, 8>, Batch> planes{};
    for (size_t input = 0; input < plan.input_indices.size(); ++input) {
      const uint8_t value = inputs[input][pos];
      for (size_t output = 0; output < Batch; ++output) {
        const uint8_t coordinate =
            plan.rows[base_out + output].basis_coordinates[input];
        for (size_t plane = 0; plane < 8; ++plane) {
          if ((coordinate & (1u << plane)) != 0) {
            planes[output][plane] ^= value;
          }
        }
      }
    }
    for (size_t output = 0; output < Batch; ++output) {
      const auto& row = plan.rows[base_out + output];
      uint8_t result = 0;
      for (size_t plane = 0; plane < 8; ++plane) {
        result ^= braid::core::GF256::Multiply(
            row.basis_elements[plane],
            planes[output][plane]);
      }
      outputs[base_out + output][pos] = result;
    }
  }
}

template <size_t Batch>
void ExecuteBraidDirectBatch512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  static_assert(Batch >= 2 && Batch <= 4);
  constexpr size_t kVec = 64;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t input_count = plan.input_indices.size();

  struct DirectSchedule {
    uint8_t count = 0;
    std::array<uint8_t, 17> planes{};
    std::array<uint8_t, 17> masks{};
  };

  std::vector<DirectSchedule> schedules(input_count);
  const size_t batch_size = std::min<size_t>(Batch, plan.rows.size() - base_out);
  for (size_t in = 0; in < input_count; ++in) {
    std::array<uint8_t, 17> plane_masks{};
    for (size_t b = 0; b < batch_size; ++b) {
      const auto& row = plan.rows[base_out + b];
      const uint8_t weight = row.alpha_weights[in];
      if (weight == 0) {
        continue;
      }
      const auto& positions = row.alpha_positions[in];
      plane_masks[positions[0]] |= static_cast<uint8_t>(1u << b);
      if (weight >= 2) {
        plane_masks[positions[1]] |= static_cast<uint8_t>(1u << b);
      }
      if (weight >= 3) {
        plane_masks[positions[2]] |= static_cast<uint8_t>(1u << b);
      }
    }
    auto& schedule = schedules[in];
    for (uint8_t plane = 0; plane < 17; ++plane) {
      if (plane_masks[plane] == 0) {
        continue;
      }
      schedule.planes[schedule.count] = plane;
      schedule.masks[schedule.count] = plane_masks[plane];
      ++schedule.count;
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m512i, Batch> accs{};
      for (size_t b = 0; b < batch_size; ++b) {
        accs[b] = _mm512_setzero_si512();
      }
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        const __m512i input_value = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + offset + pos));
        uint8_t plane = schedule.planes[0];
        __m512i current = AdvanceDirectAvx512(table, input_value, 0, plane);
        XorAccumulators512<Batch>(schedule.masks[0], current, accs);
        for (uint8_t idx = 1; idx < schedule.count; ++idx) {
          const uint8_t next_plane = schedule.planes[idx];
          current = AdvanceDirectAvx512(table, current, plane, next_plane);
          XorAccumulators512<Batch>(schedule.masks[idx], current, accs);
          plane = next_plane;
        }
      }
      for (size_t b = 0; b < batch_size; ++b) {
        _mm512_storeu_si512(reinterpret_cast<void*>(outputs[base_out + b] + offset + pos), accs[b]);
      }
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, Batch> accs{};
      for (size_t b = 0; b < batch_size; ++b) {
        accs[b] = 0;
      }
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        const uint8_t input_value = inputs[in][offset + pos];
        uint8_t plane = schedule.planes[0];
        uint8_t current = AdvanceDirectScalar512(table, input_value, 0, plane);
        for (size_t b = 0; b < batch_size; ++b) {
          if (schedule.masks[0] & (1u << b)) {
            accs[b] ^= current;
          }
        }
        for (uint8_t idx = 1; idx < schedule.count; ++idx) {
          const uint8_t next_plane = schedule.planes[idx];
          current = AdvanceDirectScalar512(table, current, plane, next_plane);
          for (size_t b = 0; b < batch_size; ++b) {
            if (schedule.masks[idx] & (1u << b)) {
              accs[b] ^= current;
            }
          }
          plane = next_plane;
        }
      }
      for (size_t b = 0; b < batch_size; ++b) {
        outputs[base_out + b][offset + pos] = accs[b];
      }
    }
  }
}

void ExecuteBraidPairDirectScheduled512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  constexpr size_t kMaxSteps = 6;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t input_count = plan.input_indices.size();
  const size_t pair_outputs = std::min<size_t>(2, plan.rows.size() - base_out);

  struct Step {
    uint8_t pos = 0;
    uint8_t mask = 0;
  };
  struct Schedule {
    uint8_t count = 0;
    std::array<Step, kMaxSteps> steps{};
  };

  std::vector<Schedule> schedules(input_count);
  for (size_t in = 0; in < input_count; ++in) {
    std::array<uint8_t, 17> plane_masks{};
    for (size_t local = 0; local < pair_outputs; ++local) {
      const auto& row = plan.rows[base_out + local];
      const uint8_t weight = row.alpha_weights[in];
      if (weight == 0) {
        continue;
      }
      const auto& positions = row.alpha_positions[in];
      plane_masks[positions[0]] |= static_cast<uint8_t>(1u << local);
      if (weight >= 2) {
        plane_masks[positions[1]] |= static_cast<uint8_t>(1u << local);
      }
      if (weight >= 3) {
        plane_masks[positions[2]] |= static_cast<uint8_t>(1u << local);
      }
    }
    auto& schedule = schedules[in];
    for (uint8_t plane = 0; plane < 17; ++plane) {
      if (plane_masks[plane] != 0) {
        schedule.steps[schedule.count++] = Step{plane, plane_masks[plane]};
      }
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m512i, 2> accs{_mm512_setzero_si512(), _mm512_setzero_si512()};
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        if (in + 8 < input_count) {
          __builtin_prefetch(inputs[in + 8] + offset + pos, 0, 1);
        }
        const __m512i input_value = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + offset + pos));
        __m512i current = AdvanceDirectAvx512(table, input_value, 0, schedule.steps[0].pos);
        XorAccumulators512<2>(schedule.steps[0].mask, current, accs);
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectAvx512(table, current, last_pos, next_pos);
          XorAccumulators512<2>(schedule.steps[step].mask, current, accs);
          last_pos = next_pos;
        }
      }
      for (size_t local = 0; local < pair_outputs; ++local) {
        _mm512_storeu_si512(reinterpret_cast<void*>(outputs[base_out + local] + offset + pos), accs[local]);
      }
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, 2> accs{};
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        uint8_t current = AdvanceDirectScalar512(table, inputs[in][offset + pos], 0, schedule.steps[0].pos);
        for (size_t local = 0; local < pair_outputs; ++local) {
          if (schedule.steps[0].mask & (1u << local)) {
            accs[local] ^= current;
          }
        }
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectScalar512(table, current, last_pos, next_pos);
          for (size_t local = 0; local < pair_outputs; ++local) {
            if (schedule.steps[step].mask & (1u << local)) {
              accs[local] ^= current;
            }
          }
          last_pos = next_pos;
        }
      }
      for (size_t local = 0; local < pair_outputs; ++local) {
        outputs[base_out + local][offset + pos] = accs[local];
      }
    }
  }
}

template <size_t Batch>
void ExecuteBraidFixedBatch512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
  size_t block_size) {
  static_assert(Batch >= 1 && Batch <= 4);
  constexpr size_t kVec = 64;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  // An odd cache-line stride colors adjacent factor planes across all L1 sets.
  // Power-of-two strides make every plane contend for the same set at a given offset.
  const size_t plane_stride = tile + kVec;
  uint8_t* scratch = AcquireThreadScratch512(Batch * 17 * plane_stride);

  const auto& schedule = plan.factor_schedules[base_out];
  const bool use_gfni =
      plan.microkernel_mode ==
      BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI;
  const bool initialize_on_first_write =
      Batch != 3 ||
      plan.microkernel_mode ==
          BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI;

  std::array<uint32_t, Batch> used_plane_masks{};
  std::array<uint8_t, Batch> max_planes{};
  std::array<std::array<uint8_t, 17>, Batch> used_planes{};
  std::array<uint8_t, Batch> used_plane_counts{};
  for (size_t b = 0; b < Batch; ++b) {
    used_plane_masks[b] = schedule.used_plane_masks[b];
    max_planes[b] = schedule.max_planes[b];
    used_planes[b] = schedule.used_planes[b];
    used_plane_counts[b] = schedule.used_plane_counts[b];
  }
  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    std::array<uint8_t*, Batch * 17> plane_ptrs{};
    for (size_t idx = 0; idx < Batch * 17; ++idx) {
      plane_ptrs[idx] = scratch + (idx * plane_stride);
    }
    if (!initialize_on_first_write) {
      if constexpr (Batch == 3) {
        for (size_t b = 0; b < Batch; ++b) {
          uint32_t mask = used_plane_masks[b];
          while (mask != 0) {
            const uint32_t plane =
                static_cast<uint32_t>(__builtin_ctz(mask));
            std::memset(plane_ptrs[b * 17 + plane], 0, chunk);
            mask &= (mask - 1);
          }
        }
      }
    }
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t in = 0; in < plan.input_indices.size(); ++in) {
      const auto& targets = schedule.target_lists[in];
      if (targets.count == 0) {
        continue;
      }
      const uint8_t* src = inputs[in] + offset;
      const uint8_t* prefetch_src =
          in + 8 < plan.input_indices.size() ? inputs[in + 8] + offset : nullptr;
      std::array<uint8_t*, 12> target_ptrs{};
      for (size_t target = 0; target < targets.count; ++target) {
        target_ptrs[target] = plane_ptrs[targets.ids[target]];
      }
      if (targets.initialize_mask != 0 &&
          initialize_on_first_write) {
        switch (targets.count) {
          case 1: InitializeOrAccumulateScratchTargets512<1>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 2: InitializeOrAccumulateScratchTargets512<2>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 3: InitializeOrAccumulateScratchTargets512<3>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 4: InitializeOrAccumulateScratchTargets512<4>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 5: InitializeOrAccumulateScratchTargets512<5>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 6: InitializeOrAccumulateScratchTargets512<6>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 7: InitializeOrAccumulateScratchTargets512<7>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 8: InitializeOrAccumulateScratchTargets512<8>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 9: InitializeOrAccumulateScratchTargets512<9>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 10: InitializeOrAccumulateScratchTargets512<10>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 11: InitializeOrAccumulateScratchTargets512<11>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          case 12: InitializeOrAccumulateScratchTargets512<12>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
          default: break;
        }
        continue;
      }
      switch (targets.count) {
        case 1: AccumulateScratchTargets512<1>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 2: AccumulateScratchTargets512<2>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 3: AccumulateScratchTargets512<3>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 4: AccumulateScratchTargets512<4>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 5: AccumulateScratchTargets512<5>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 6: AccumulateScratchTargets512<6>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 7: AccumulateScratchTargets512<7>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 8: AccumulateScratchTargets512<8>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 9: AccumulateScratchTargets512<9>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 10: AccumulateScratchTargets512<10>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 11: AccumulateScratchTargets512<11>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        case 12: AccumulateScratchTargets512<12>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
        default:
          break;
      }
    }

    constexpr size_t kReductionPairs = Batch / 2;
    std::array<std::array<uint8_t*, 34>, kReductionPairs> reduction_plane_ptrs{};
    std::array<std::array<std::array<uint8_t, 17>, 2>, kReductionPairs> reduction_used_planes{};
    std::array<std::array<uint8_t, 2>, kReductionPairs> reduction_used_plane_counts{};
    if constexpr (Batch == 2 || Batch == 4) {
      for (size_t pair = 0; pair < kReductionPairs; ++pair) {
        for (size_t local = 0; local < 2; ++local) {
          const size_t output = pair * 2 + local;
          for (size_t plane = 0; plane < 17; ++plane) {
            reduction_plane_ptrs[pair][local * 17 + plane] = plane_ptrs[output * 17 + plane];
          }
          reduction_used_planes[pair][local] = used_planes[output];
          reduction_used_plane_counts[pair][local] = used_plane_counts[output];
        }
      }
    }

    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m512i, Batch> acc{};
      if constexpr (Batch == 1) {
        const uint8_t count = used_plane_counts[0];
        if (count == 0) {
          acc[0] = _mm512_setzero_si512();
        } else {
          uint8_t current_plane = used_planes[0][count - 1];
          acc[0] = _mm512_load_si512(
              reinterpret_cast<const void*>(plane_ptrs[static_cast<size_t>(current_plane)] + pos));
          for (int idx = static_cast<int>(count) - 2; idx >= 0; --idx) {
              const uint8_t next_plane = used_planes[0][static_cast<size_t>(idx)];
              acc[0] = AdvanceBetweenPlanesSelected512(
                  table, acc[0], current_plane, next_plane, use_gfni);
              acc[0] = _mm512_xor_si512(
                  acc[0],
                  _mm512_load_si512(
                    reinterpret_cast<const void*>(plane_ptrs[static_cast<size_t>(next_plane)] + pos)));
            current_plane = next_plane;
          }
            if (current_plane != 0) {
              acc[0] = AdvanceBetweenPlanesSelected512(
                  table, acc[0], current_plane, 0, use_gfni);
            }
        }
      } else if constexpr (Batch == 2) {
        std::array<__m512i, 2> pair_acc{};
        ReduceTwoSparseRows512(
            table,
            reduction_plane_ptrs[0],
            reduction_used_planes[0],
            reduction_used_plane_counts[0],
            pos,
            use_gfni,
            pair_acc);
        acc[0] = pair_acc[0];
        acc[1] = pair_acc[1];
      } else if constexpr (Batch == 4) {
        for (size_t pair = 0; pair < 2; ++pair) {
          std::array<__m512i, 2> pair_acc{};
          ReduceTwoSparseRows512(
              table,
              reduction_plane_ptrs[pair],
              reduction_used_planes[pair],
              reduction_used_plane_counts[pair],
              pos,
              use_gfni,
              pair_acc);
          acc[pair * 2] = pair_acc[0];
          acc[pair * 2 + 1] = pair_acc[1];
        }
      } else {
        const uint8_t global_max = *std::max_element(max_planes.begin(), max_planes.end());
        for (int plane = static_cast<int>(global_max); plane >= 0; --plane) {
          for (size_t b = 0; b < Batch; ++b) {
            const int max_plane = static_cast<int>(max_planes[b]);
            if (plane > max_plane) {
              continue;
            }
            if (plane == max_plane) {
              acc[b] = _mm512_load_si512(
                  reinterpret_cast<const void*>(plane_ptrs[b * 17 + static_cast<size_t>(plane)] + pos));
              continue;
            }
            acc[b] = AdvanceBetweenPlanesSelected512(
                table,
                acc[b],
                static_cast<uint8_t>(plane + 1),
                static_cast<uint8_t>(plane),
                use_gfni);
            if (used_plane_masks[b] & (1u << plane)) {
              acc[b] = _mm512_xor_si512(
                  acc[b],
                  _mm512_load_si512(
                      reinterpret_cast<const void*>(plane_ptrs[b * 17 + static_cast<size_t>(plane)] + pos)));
            }
          }
        }
      }
      const auto reconstructed =
          ReconstructFactorOutputs512<Batch>(schedule, acc);
      for (size_t b = 0; b < Batch; ++b) {
        uint8_t* dst_ptr = outputs[base_out + b] + offset + pos;
        _mm512_storeu_si512(
            reinterpret_cast<void*>(dst_ptr), reconstructed[b]);
      }
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, Batch> acc{};
      if constexpr (Batch == 1) {
        const uint8_t count = used_plane_counts[0];
        if (count == 0) {
          acc[0] = 0;
        } else {
            uint8_t current_plane = used_planes[0][count - 1];
            acc[0] = plane_ptrs[static_cast<size_t>(current_plane)][pos];
            for (int idx = static_cast<int>(count) - 2; idx >= 0; --idx) {
              const uint8_t next_plane = used_planes[0][static_cast<size_t>(idx)];
              acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_plane, next_plane);
              acc[0] ^= plane_ptrs[static_cast<size_t>(next_plane)][pos];
              current_plane = next_plane;
            }
            if (current_plane != 0) {
              acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_plane, 0);
            }
        }
      } else if constexpr (Batch == 2) {
        std::array<uint8_t*, 34> pair_plane_ptrs{};
        std::copy(plane_ptrs.begin(), plane_ptrs.end(), pair_plane_ptrs.begin());
        std::array<std::array<uint8_t, 17>, 2> pair_used_planes{};
        std::array<uint8_t, 2> pair_used_plane_counts{};
        std::array<uint8_t, 2> pair_acc{};
        for (size_t b = 0; b < 2; ++b) {
          pair_used_planes[b] = used_planes[b];
          pair_used_plane_counts[b] = used_plane_counts[b];
        }
        ReduceTwoSparseRowsScalar(
            table,
            pair_plane_ptrs,
            pair_used_planes,
            pair_used_plane_counts,
            pos,
            pair_acc);
        acc[0] = pair_acc[0];
        acc[1] = pair_acc[1];
      } else {
        const uint8_t global_max = *std::max_element(max_planes.begin(), max_planes.end());
        for (int plane = static_cast<int>(global_max); plane >= 0; --plane) {
          for (size_t b = 0; b < Batch; ++b) {
            const int max_plane = static_cast<int>(max_planes[b]);
            if (plane > max_plane) {
              continue;
            }
            if (plane == max_plane) {
              acc[b] = plane_ptrs[b * 17 + static_cast<size_t>(plane)][pos];
              continue;
            }
            acc[b] = AdvanceBetweenPlanesScalar(
                table, acc[b], static_cast<uint8_t>(plane + 1), static_cast<uint8_t>(plane));
            if (used_plane_masks[b] & (1u << plane)) {
              acc[b] ^= plane_ptrs[b * 17 + static_cast<size_t>(plane)][pos];
            }
          }
        }
      }
      const auto reconstructed =
          ReconstructFactorOutputsScalar<Batch>(schedule, acc);
      for (size_t b = 0; b < Batch; ++b) {
        outputs[base_out + b][offset + pos] = reconstructed[b];
      }
    }
  }
}

void ExecuteBraidGroupedScratch512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t plane_stride = tile + kVec;
  const size_t batch =
      static_cast<size_t>(std::max(plan.output_batch, 1));

  for (size_t base_out = 0; base_out < plan.rows.size();
       base_out += batch) {
    if (base_out >= plan.factor_grouped_schedules.size()) {
      throw std::runtime_error(
          "BRAID-F grouped factor schedule is unavailable");
    }
    const auto& schedule = plan.factor_grouped_schedules[base_out];
    const size_t batch_size = schedule.batch_size;
    if (batch_size == 0 || batch_size > 4 || schedule.groups.empty()) {
      throw std::runtime_error(
          "BRAID-F grouped factor schedule is invalid");
    }
    uint8_t* scratch =
        AcquireThreadScratch512(schedule.groups.size() * plane_stride);
    std::vector<uint8_t*> group_ptrs(schedule.groups.size());
    for (size_t group = 0; group < schedule.groups.size(); ++group) {
      group_ptrs[group] = scratch + group * plane_stride;
    }

    for (size_t offset = 0; offset < block_size; offset += tile) {
      const size_t chunk = std::min(tile, block_size - offset);
      const size_t vec_limit = (chunk / kVec) * kVec;
      for (size_t input = 0; input < plan.input_indices.size(); ++input) {
        const auto& targets = schedule.target_lists[input];
        if (targets.count == 0) {
          continue;
        }
        const uint8_t* src = inputs[input] + offset;
        const uint8_t* prefetch_src =
            input + 8 < plan.input_indices.size()
                ? inputs[input + 8] + offset
                : nullptr;
        std::array<uint8_t*, 12> target_ptrs{};
        for (size_t target = 0; target < targets.count; ++target) {
          target_ptrs[target] = group_ptrs[targets.ids[target]];
        }
        if (targets.initialize_mask != 0) {
          switch (targets.count) {
            case 1: InitializeOrAccumulateScratchTargets512<1>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 2: InitializeOrAccumulateScratchTargets512<2>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 3: InitializeOrAccumulateScratchTargets512<3>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 4: InitializeOrAccumulateScratchTargets512<4>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 5: InitializeOrAccumulateScratchTargets512<5>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 6: InitializeOrAccumulateScratchTargets512<6>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 7: InitializeOrAccumulateScratchTargets512<7>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 8: InitializeOrAccumulateScratchTargets512<8>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 9: InitializeOrAccumulateScratchTargets512<9>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 10: InitializeOrAccumulateScratchTargets512<10>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 11: InitializeOrAccumulateScratchTargets512<11>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            case 12: InitializeOrAccumulateScratchTargets512<12>(src, prefetch_src, target_ptrs, targets.initialize_mask, vec_limit, chunk); break;
            default: throw std::runtime_error("too many grouped factor routes");
          }
        } else {
          switch (targets.count) {
            case 1: AccumulateScratchTargets512<1>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 2: AccumulateScratchTargets512<2>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 3: AccumulateScratchTargets512<3>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 4: AccumulateScratchTargets512<4>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 5: AccumulateScratchTargets512<5>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 6: AccumulateScratchTargets512<6>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 7: AccumulateScratchTargets512<7>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 8: AccumulateScratchTargets512<8>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 9: AccumulateScratchTargets512<9>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 10: AccumulateScratchTargets512<10>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 11: AccumulateScratchTargets512<11>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            case 12: AccumulateScratchTargets512<12>(src, prefetch_src, target_ptrs, vec_limit, chunk); break;
            default: throw std::runtime_error("too many grouped factor routes");
          }
        }
      }

      for (size_t pos = 0; pos < vec_limit; pos += kVec) {
        for (size_t output = 0; output < batch_size; ++output) {
          __m512i accumulator = _mm512_setzero_si512();
          int current_plane = -1;
          for (int plane = 16; plane >= 0; --plane) {
            const auto& groups =
                schedule.reduction_groups[output]
                                         [static_cast<size_t>(plane)];
            if (groups.empty()) {
              continue;
            }
            __m512i plane_value = _mm512_load_si512(
                reinterpret_cast<const void*>(
                    group_ptrs[groups.front()] + pos));
            for (size_t group = 1; group < groups.size(); ++group) {
              plane_value = _mm512_xor_si512(
                  plane_value,
                  _mm512_load_si512(
                      reinterpret_cast<const void*>(
                          group_ptrs[groups[group]] + pos)));
            }
            if (current_plane < 0) {
              accumulator = plane_value;
            } else {
              accumulator = _mm512_xor_si512(
                  AdvanceBetweenPlanesAvx512(
                      table,
                      accumulator,
                      static_cast<uint8_t>(current_plane),
                      static_cast<uint8_t>(plane)),
                  plane_value);
            }
            current_plane = plane;
          }
          if (current_plane > 0) {
            accumulator = AdvanceBetweenPlanesAvx512(
                table,
                accumulator,
                static_cast<uint8_t>(current_plane),
                0);
          }
          _mm512_storeu_si512(
              reinterpret_cast<void*>(
                  outputs[base_out + output] + offset + pos),
              accumulator);
        }
      }

      for (size_t pos = vec_limit; pos < chunk; ++pos) {
        for (size_t output = 0; output < batch_size; ++output) {
          uint8_t accumulator = 0;
          int current_plane = -1;
          for (int plane = 16; plane >= 0; --plane) {
            const auto& groups =
                schedule.reduction_groups[output]
                                         [static_cast<size_t>(plane)];
            if (groups.empty()) {
              continue;
            }
            uint8_t plane_value = 0;
            for (uint8_t group : groups) {
              plane_value ^= group_ptrs[group][pos];
            }
            if (current_plane < 0) {
              accumulator = plane_value;
            } else {
              accumulator =
                  AdvanceBetweenPlanesScalar(
                      table,
                      accumulator,
                      static_cast<uint8_t>(current_plane),
                      static_cast<uint8_t>(plane)) ^
                  plane_value;
            }
            current_plane = plane;
          }
          if (current_plane > 0) {
            accumulator = AdvanceBetweenPlanesScalar(
                table,
                accumulator,
                static_cast<uint8_t>(current_plane),
                0);
          }
          outputs[base_out + output][offset + pos] = accumulator;
        }
      }
    }
  }
}











// ── Parallel two-phase BRAID kernel (AVX-512 only) ──
// Phase 1: independent plane expansion per input (zero dependency chain)
// Phase 2: per-row plane reduction after all inputs in a tile
// Requires 32 zmm registers → AVX-512 exclusive.
// Processes 2 output rows at a time.

struct PlaneNibbleTable {
  __m512i lo;
  __m512i hi;
};

inline void LoadPlaneTables(
    const braid::core::DecompositionTable& table,
    const std::vector<uint8_t>& used_planes,
    std::array<PlaneNibbleTable, 17>& tbls) {
  const auto& nt = GetNibbleTables();
  for (uint8_t p : used_planes) {
    uint8_t coeff = table.x_powers[p];
    tbls[p].lo = _mm512_broadcast_i32x4(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(nt.lo[coeff].data())));
    tbls[p].hi = _mm512_broadcast_i32x4(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(nt.hi[coeff].data())));
  }
}

inline __m512i PlaneExpandAvx512(__m512i v, const PlaneNibbleTable& tbl) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i lo = _mm512_and_si512(v, mask);
  const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), mask);
  return _mm512_xor_si512(
      _mm512_shuffle_epi8(tbl.lo, lo),
      _mm512_shuffle_epi8(tbl.hi, hi));
}

void ExecuteBraidParallelPair512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  ExecuteBraidPairDirectScheduled512(table, plan, base_out, inputs, outputs, block_size);
  return;

  constexpr size_t kVec = 64;
  const size_t tile = std::max<size_t>(plan.tile_bytes, 64u << 10);  // DENSE-friendly tile
  const size_t input_count = plan.input_indices.size();
  const size_t pair_rows = std::min<size_t>(2, plan.rows.size() - base_out);

  // Build union of used planes across the 2 rows
  uint32_t union_mask = 0;
  std::array<uint8_t, 17> plane_row0, plane_row1;
  uint8_t plane_count0 = 0, plane_count1 = 0;
  std::array<const braid::runtime::OutputPlanRow*, 2> rows;
  for (size_t b = 0; b < pair_rows; ++b) {
    rows[b] = &plan.rows[base_out + b];
  }
  for (size_t in = 0; in < input_count; ++in) {
    for (size_t b = 0; b < pair_rows; ++b) {
      uint8_t w = rows[b]->alpha_weights[in];
      if (w == 0) continue;
      const auto& pos = rows[b]->alpha_positions[in];
      union_mask |= (1u << pos[0]);
      if (w >= 2) union_mask |= (1u << pos[1]);
      if (w >= 3) union_mask |= (1u << pos[2]);
    }
  }
  // Extract per-row plane lists (sorted ascending for reduction)
  for (uint8_t p = 0; p < 17; ++p) {
    if (union_mask & (1u << p)) {
      for (size_t b = 0; b < pair_rows; ++b) {
        uint32_t row_mask = 0;
        for (size_t in = 0; in < input_count; ++in) {
          uint8_t w = rows[b]->alpha_weights[in];
          if (w == 0) continue;
          const auto& pos = rows[b]->alpha_positions[in];
          if (pos[0] == p || (w >= 2 && pos[1] == p) || (w >= 3 && pos[2] == p)) {
            row_mask = 1;
            break;
          }
        }
        if (row_mask) {
          if (b == 0) plane_row0[plane_count0++] = p;
          else plane_row1[plane_count1++] = p;
        }
      }
    }
  }

  // Precompute nibble tables for each x^plane
  std::array<PlaneNibbleTable, 17> plane_tbls{};
  const auto& nt = GetNibbleTables();
  for (uint8_t p = 0; p < 17; ++p) {
    if (union_mask & (1u << p)) {
      uint8_t coeff = table.x_powers[p];
      plane_tbls[p].lo = _mm512_broadcast_i32x4(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(nt.lo[coeff].data())));
      plane_tbls[p].hi = _mm512_broadcast_i32x4(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(nt.hi[coeff].data())));
    }
  }
  // Build per-input flat plane index arrays (avoid ctz+switch in hot path)
  struct InputPlanes {
    uint8_t count = 0;
    uint8_t planes[17];
  };
  std::vector<InputPlanes> input_planes(input_count);
  for (size_t in = 0; in < input_count; ++in) {
    uint32_t mask = 0;
    for (size_t b = 0; b < pair_rows; ++b) {
      uint8_t w = rows[b]->alpha_weights[in];
      if (w == 0) continue;
      const auto& pos = rows[b]->alpha_positions[in];
      mask |= (1u << pos[0]);
      if (w >= 2) mask |= (1u << pos[1]);
      if (w >= 3) mask |= (1u << pos[2]);
    }
    auto& ip = input_planes[in];
    uint32_t m = mask;
    while (m) {
      ip.planes[ip.count++] = static_cast<uint8_t>(__builtin_ctz(m));
      m &= (m - 1);
    }
  }

  const __m512i vmask = _mm512_set1_epi8(0x0f);

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;

    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      // Phase 1: plane expansion — accumulate into 17 plane registers
      __m512i plane0 = _mm512_setzero_si512(), plane1 = _mm512_setzero_si512();
      __m512i plane2 = _mm512_setzero_si512(), plane3 = _mm512_setzero_si512();
      __m512i plane4 = _mm512_setzero_si512(), plane5 = _mm512_setzero_si512();
      __m512i plane6 = _mm512_setzero_si512(), plane7 = _mm512_setzero_si512();
      __m512i plane8 = _mm512_setzero_si512(), plane9 = _mm512_setzero_si512();
      __m512i plane10 = _mm512_setzero_si512(), plane11 = _mm512_setzero_si512();
      __m512i plane12 = _mm512_setzero_si512(), plane13 = _mm512_setzero_si512();
      __m512i plane14 = _mm512_setzero_si512(), plane15 = _mm512_setzero_si512();
      __m512i plane16 = _mm512_setzero_si512();

      __m512i* plane_ptrs[17] = {&plane0,&plane1,&plane2,&plane3,&plane4,&plane5,&plane6,
                                  &plane7,&plane8,&plane9,&plane10,&plane11,&plane12,
                                  &plane13,&plane14,&plane15,&plane16};

      for (size_t in = 0; in < input_count; ++in) {
        const auto& ip = input_planes[in];
        if (ip.count == 0) continue;
        if (in + 8 < input_count)
          __builtin_prefetch(inputs[in + 8] + offset + pos, 0, 1);

        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[in] + offset + pos));
        const __m512i lo = _mm512_and_si512(v, vmask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), vmask);

        // Branch-free vpshufb plane expansion.
        // plane=0 → identity (x^0=1), skip GF multiply entirely (~15-20% of planes)
        for (uint8_t i = 0; i < ip.count; ++i) {
          uint8_t p = ip.planes[i];
          if (p == 0) {
            *plane_ptrs[0] = _mm512_xor_si512(*plane_ptrs[0], v);
          } else {
            const auto& tbl = plane_tbls[p];
            __m512i prod = _mm512_xor_si512(_mm512_shuffle_epi8(tbl.lo, lo),
                                             _mm512_shuffle_epi8(tbl.hi, hi));
            *plane_ptrs[p] = _mm512_xor_si512(*plane_ptrs[p], prod);
          }
        }
      }

      // Phase 2: per-row plane reduction to outputs
      __m512i planes[17] = {plane0,plane1,plane2,plane3,plane4,plane5,plane6,plane7,
                            plane8,plane9,plane10,plane11,plane12,plane13,plane14,plane15,plane16};

      auto reduce_row = [&](const std::array<uint8_t, 17>& row_planes, uint8_t count) -> __m512i {
        if (count == 0) return _mm512_setzero_si512();
        uint8_t cur = row_planes[count - 1];
        __m512i acc = planes[cur];
        for (int idx = static_cast<int>(count) - 2; idx >= 0; --idx) {
          uint8_t nxt = row_planes[idx];
          uint8_t gap = cur - nxt;
          if (gap == 1) acc = XtimeAvx512(acc);
          else if (gap == 2) acc = XtimeAvx512(XtimeAvx512(acc));
          else acc = MultiplyXPowAvx512(table, acc, gap);
          acc = _mm512_xor_si512(acc, planes[nxt]);
          cur = nxt;
        }
        if (cur == 1) acc = XtimeAvx512(acc);
        else if (cur == 2) acc = XtimeAvx512(XtimeAvx512(acc));
        else if (cur >= 3) acc = MultiplyXPowAvx512(table, acc, cur);
        return acc;
      };

      __m512i out0 = reduce_row(plane_row0, plane_count0);
      __m512i out1 = (pair_rows >= 2) ? reduce_row(plane_row1, plane_count1) : _mm512_setzero_si512();

      _mm512_storeu_si512(reinterpret_cast<void*>(outputs[base_out] + offset + pos), out0);
      if (pair_rows >= 2)
        _mm512_storeu_si512(reinterpret_cast<void*>(outputs[base_out + 1] + offset + pos), out1);
    }

    // Scalar tail
    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      uint8_t planes[17] = {};
      for (size_t in = 0; in < input_count; ++in) {
        uint8_t v = inputs[in][offset + pos];
        for (size_t b = 0; b < pair_rows; ++b) {
          uint8_t w = rows[b]->alpha_weights[in];
          if (w == 0) continue;
          const auto& pos_arr = rows[b]->alpha_positions[in];
          planes[pos_arr[0]] ^= v;
          if (w >= 2) planes[pos_arr[1]] ^= v;
          if (w >= 3) planes[pos_arr[2]] ^= v;
        }
      }
      auto reduce_scalar = [&](const auto& rp, uint8_t cnt) -> uint8_t {
        if (cnt == 0) return 0;
        uint8_t cur = rp[cnt - 1], acc = planes[cur];
        for (int i = cnt - 2; i >= 0; --i) {
          uint8_t nxt = rp[i];
          for (uint8_t g = cur - nxt; g > 0; --g)
            acc = braid::core::GF256::MultiplyByX(acc);
          acc ^= planes[nxt];
          cur = nxt;
        }
        for (uint8_t g = cur; g > 0; --g)
          acc = braid::core::GF256::MultiplyByX(acc);
        return acc;
      };
      outputs[base_out][offset + pos] = reduce_scalar(plane_row0, plane_count0);
      if (pair_rows >= 2)
        outputs[base_out + 1][offset + pos] = reduce_scalar(plane_row1, plane_count1);
    }
  }
}

void ExecuteBraidVector512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 64;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const int batch = std::max(plan.output_batch, 1);
  const size_t global_block_size = plan.planned_block_size != 0 ? plan.planned_block_size : block_size;
  const bool can_use_progression_direct = table.progression_order;
  const bool is_encode = plan.type == braid::runtime::PlanType::kEncode;
  const bool is_decode = plan.type == braid::runtime::PlanType::kDecode;
  const size_t input_count = plan.input_indices.size();
  for (size_t base_out = 0; base_out < plan.rows.size(); base_out += batch) {
    const size_t batch_size = std::min<size_t>(batch, plan.rows.size() - base_out);

    // Pure BF dispatch. Every branch evaluates the representation-level
    // factorization; dense multiplication is reserved for the oracle path.

    if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_BASIS_LUT) {
      const bool use_bit_tests = input_count > 20;
      if (batch_size == 4) {
        if (use_bit_tests) {
          ExecuteBraidAdaptiveBasisPair512<true>(
              plan, base_out, inputs, outputs, block_size);
          ExecuteBraidAdaptiveBasisPair512<true>(
              plan, base_out + 2, inputs, outputs, block_size);
        } else {
          ExecuteBraidAdaptiveBasisPair512<false>(
              plan, base_out, inputs, outputs, block_size);
          ExecuteBraidAdaptiveBasisPair512<false>(
              plan, base_out + 2, inputs, outputs, block_size);
        }
      } else if (batch_size == 3) {
        if (use_bit_tests) {
          ExecuteBraidAdaptiveBasisLut512<3, true>(
              plan, base_out, inputs, outputs, block_size);
        } else {
          ExecuteBraidAdaptiveBasisLut512<3, false>(
              plan, base_out, inputs, outputs, block_size);
        }
      } else if (batch_size == 2) {
        if (use_bit_tests) {
          ExecuteBraidAdaptiveBasisPair512<true>(
              plan, base_out, inputs, outputs, block_size);
        } else {
          ExecuteBraidAdaptiveBasisPair512<false>(
              plan, base_out, inputs, outputs, block_size);
        }
      } else {
        if (use_bit_tests) {
          ExecuteBraidAdaptiveBasisLut512<1, true>(
              plan, base_out, inputs, outputs, block_size);
        } else {
          ExecuteBraidAdaptiveBasisLut512<1, false>(
              plan, base_out, inputs, outputs, block_size);
        }
      }
      continue;
    }

    if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_DIRECT) {
      if (batch_size == 4) {
        ExecuteBraidDirectBatch512<4>(table, plan, base_out, inputs, outputs, block_size);
      } else if (batch_size == 3) {
        ExecuteBraidDirectBatch512<3>(table, plan, base_out, inputs, outputs, block_size);
      } else if (batch_size == 2) {
        ExecuteBraidPairDirectScheduled512(table, plan, base_out, inputs, outputs, block_size);
      } else {
        ExecuteBraidPolynomialRegister512(plan, base_out, 1, inputs, outputs, block_size);
      }
      continue;
    }
    // Short chains use an eight-basis factorization. A pair keeps sixteen
    // accumulators resident and reuses each input load across both outputs.
    if (input_count <= 20 ||
        plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_POLYNOMIAL_FUSED) {
      const bool use_bit_tests = input_count > 40;
      if (batch_size == 3) {
        if (use_bit_tests) {
          ExecuteBraidPolynomialTriple512<true>(plan, base_out, inputs, outputs, block_size);
        } else {
          ExecuteBraidPolynomialTriple512<false>(plan, base_out, inputs, outputs, block_size);
        }
        continue;
      }
      size_t local = 0;
      while (local + 2 <= batch_size) {
        if (use_bit_tests) {
          ExecuteBraidPolynomialPair512<true>(
              plan, base_out + local, inputs, outputs, block_size);
        } else {
          ExecuteBraidPolynomialPair512<false>(
              plan, base_out + local, inputs, outputs, block_size);
        }
        local += 2;
      }
      if (local < batch_size) {
        ExecuteBraidPolynomialRegister512(plan, base_out + local, 1, inputs, outputs, block_size);
      }
      continue;
    }

    // A single long-chain output keeps all eight polynomial basis planes in registers.
    if ((plan.microkernel_mode ==
             BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI ||
         plan.microkernel_mode ==
             BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI) &&
        input_count > 20) {
      if (batch_size == 4) {
        ExecuteBraidFixedBatch512<4>(
            table, plan, base_out, inputs, outputs, block_size);
      } else if (batch_size == 3) {
        ExecuteBraidFixedBatch512<3>(
            table, plan, base_out, inputs, outputs, block_size);
      } else if (batch_size == 2) {
        ExecuteBraidFixedBatch512<2>(
            table, plan, base_out, inputs, outputs, block_size);
      } else {
        ExecuteBraidFixedBatch512<1>(
            table, plan, base_out, inputs, outputs, block_size);
      }
      continue;
    }

    if (batch_size == 1) {
      ExecuteBraidPolynomialRegister512(plan, base_out, 1, inputs, outputs, block_size);
      continue;
    }

    // Long chains amortize one input load across every output in the batch.
    if (input_count > 20) {
      if (batch_size == 4) {
        ExecuteBraidFixedBatch512<4>(table, plan, base_out, inputs, outputs, block_size);
      } else if (batch_size == 3) {
        ExecuteBraidFixedBatch512<3>(table, plan, base_out, inputs, outputs, block_size);
      } else {
        ExecuteBraidFixedBatch512<2>(table, plan, base_out, inputs, outputs, block_size);
      }
      continue;
    }
    uint8_t* scratch = AcquireThreadScratch512(batch_size * 17 * tile);
    for (size_t offset = 0; offset < block_size; offset += tile) {
      const size_t chunk = std::min(tile, block_size - offset);
      for (size_t plane = 0; plane < batch_size * 17; ++plane) {
        std::memset(scratch + plane * tile, 0, chunk);
      }
      const size_t vec_limit = (chunk / kVec) * kVec;
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t* src = inputs[in] + offset;
        for (size_t pos = 0; pos < vec_limit; pos += kVec) {
          const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(src + pos));
          for (size_t b = 0; b < batch_size; ++b) {
            const auto& row = plan.rows[base_out + b];
            const uint8_t weight = row.alpha_weights[in];
            if (weight == 0) {
              continue;
            }
            const auto& positions = row.alpha_positions[in];
            uint8_t* dst_ptr = scratch + ((b * 17 + static_cast<size_t>(positions[0])) * tile) + pos;
            __m512i dst = _mm512_loadu_si512(reinterpret_cast<const void*>(dst_ptr));
            _mm512_storeu_si512(reinterpret_cast<void*>(dst_ptr), _mm512_xor_si512(dst, v));
            if (weight >= 2) {
              dst_ptr = scratch + ((b * 17 + static_cast<size_t>(positions[1])) * tile) + pos;
              dst = _mm512_loadu_si512(reinterpret_cast<const void*>(dst_ptr));
              _mm512_storeu_si512(reinterpret_cast<void*>(dst_ptr), _mm512_xor_si512(dst, v));
            }
            if (weight >= 3) {
              dst_ptr = scratch + ((b * 17 + static_cast<size_t>(positions[2])) * tile) + pos;
              dst = _mm512_loadu_si512(reinterpret_cast<const void*>(dst_ptr));
              _mm512_storeu_si512(reinterpret_cast<void*>(dst_ptr), _mm512_xor_si512(dst, v));
            }
          }
        }
        for (size_t b = 0; b < batch_size; ++b) {
          const auto& row = plan.rows[base_out + b];
          const uint8_t weight = row.alpha_weights[in];
          if (weight == 0) {
            continue;
          }
          const auto& positions = row.alpha_positions[in];
          uint8_t* dst = scratch + ((b * 17 + static_cast<size_t>(positions[0])) * tile);
          for (size_t pos = vec_limit; pos < chunk; ++pos) {
            dst[pos] ^= src[pos];
          }
          if (weight >= 2) {
            dst = scratch + ((b * 17 + static_cast<size_t>(positions[1])) * tile);
            for (size_t pos = vec_limit; pos < chunk; ++pos) {
              dst[pos] ^= src[pos];
            }
          }
          if (weight >= 3) {
            dst = scratch + ((b * 17 + static_cast<size_t>(positions[2])) * tile);
            for (size_t pos = vec_limit; pos < chunk; ++pos) {
              dst[pos] ^= src[pos];
            }
          }
        }
      }
      for (size_t b = 0; b < batch_size; ++b) {
        uint8_t* out = outputs[base_out + b] + offset;
        const uint8_t* plane16 = scratch + ((b * 17 + 16) * tile);
        for (size_t pos = 0; pos < vec_limit; pos += kVec) {
          __m512i acc = _mm512_load_si512(reinterpret_cast<const void*>(plane16 + pos));
          for (int t = 15; t >= 0; --t) {
            acc = AdvanceBetweenPlanesAvx512(table, acc, static_cast<uint8_t>(t + 1), static_cast<uint8_t>(t));
            const uint8_t* src = scratch + ((b * 17 + static_cast<size_t>(t)) * tile) + pos;
            acc = _mm512_xor_si512(acc, _mm512_load_si512(reinterpret_cast<const void*>(src)));
          }
          _mm512_storeu_si512(reinterpret_cast<void*>(out + pos), acc);
        }
        for (size_t pos = vec_limit; pos < chunk; ++pos) {
          uint8_t acc = plane16[pos];
          for (int t = 15; t >= 0; --t) {
            const uint8_t* src = scratch + ((b * 17 + static_cast<size_t>(t)) * tile);
            acc = AdvanceBetweenPlanesScalar(table, acc, static_cast<uint8_t>(t + 1), static_cast<uint8_t>(t)) ^
                  src[pos];
          }
          out[pos] = acc;
        }
      }
    }
  }
}

}  // namespace




#if defined(BRAID_AVX512_GFNI_OBJECT)
void ExecuteBraidAVX512GFNI(
#else
void ExecuteBraidAVX512(
#endif
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  if (plan.microkernel_mode ==
      BRAID_MICROKERNEL_FACTOR_GROUPED_SCRATCH) {
    ExecuteBraidGroupedScratch512(
        table, plan, inputs, outputs, block_size);
    return;
  }
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT ||
      plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI) {
    braid::runtime::ExecuteAvx512PlanJit(
        plan, inputs, outputs, block_size);
    return;
  }
#endif
  if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_REGISTER) {
    ExecuteBraidBatch1Register512(table, plan, inputs, outputs, block_size);
    return;
  }
  if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_PLANE_STREAMING) {
    ExecuteBraidPlaneStreaming512(table, plan, inputs, outputs, block_size);
    return;
  }

  // A progression representation can reduce each row from eight polynomial
  // basis planes regardless of chain length. Keep this path independent of the
  // legacy direct schedule, whose transition metadata is representation-specific.
  if (plan.output_batch == 1 && table.progression_order) {
    if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_SCRATCH ||
        plan.microkernel_mode ==
            BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI) {
      for (size_t out = 0; out < plan.rows.size(); ++out) {
        ExecuteBraidFixedBatch512<1>(table, plan, out, inputs, outputs, block_size);
      }
    } else {
      ExecuteBraidPolynomialRegister512(plan, 0, plan.rows.size(), inputs, outputs, block_size);
    }
    return;
  }
  ExecuteBraidVector512(table, plan, inputs, outputs, block_size);
}

}  // namespace braid::kernels
