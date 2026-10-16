#ifndef BRAID_CORE_GFNI_MATRIX_H_
#define BRAID_CORE_GFNI_MATRIX_H_

#include <array>
#include <cstdint>

#include "src/core/gf256.h"

namespace braid::core {

// Packs the GF(0x139) multiplication transform in Intel GFNI affine format.
static inline uint64_t BuildGfniAffineMatrix(uint8_t coefficient) {
  uint64_t matrix = 0;
  for (int output_bit = 0; output_bit < 8; ++output_bit) {
    uint8_t row = 0;
    for (int input_bit = 0; input_bit < 8; ++input_bit) {
      const uint8_t product =
          GF256::Multiply(coefficient, static_cast<uint8_t>(1u << input_bit));
      row |= static_cast<uint8_t>(
          ((product >> output_bit) & 1u) << input_bit);
    }
    matrix |= static_cast<uint64_t>(row) << (8 * (7 - output_bit));
  }
  return matrix;
}

static inline const std::array<uint64_t, 256>& GfniAffineMatrices() {
  static const std::array<uint64_t, 256> matrices = [] {
    std::array<uint64_t, 256> result{};
    for (size_t coefficient = 0; coefficient < result.size();
         ++coefficient) {
      result[coefficient] =
          BuildGfniAffineMatrix(static_cast<uint8_t>(coefficient));
    }
    return result;
  }();
  return matrices;
}

}  // namespace braid::core

#endif  // BRAID_CORE_GFNI_MATRIX_H_
