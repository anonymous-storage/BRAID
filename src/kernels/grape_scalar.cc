#include "src/kernels/grape_scalar.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "src/core/gf256.h"

namespace braid::kernels {

void ExecuteBraidScalar(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  const size_t tile = std::max<size_t>(plan.tile_bytes, 64);
  const int batch = std::max(plan.output_batch, 1);
  for (size_t base_out = 0; base_out < plan.rows.size(); base_out += batch) {
    const size_t batch_size = std::min<size_t>(batch, plan.rows.size() - base_out);
    std::vector<uint8_t> scratch(batch_size * 17 * tile, 0);
    for (size_t offset = 0; offset < block_size; offset += tile) {
      const size_t chunk = std::min(tile, block_size - offset);
      for (size_t plane = 0; plane < batch_size * 17; ++plane) {
        std::memset(scratch.data() + plane * tile, 0, chunk);
      }
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const uint8_t* src = inputs[in] + offset;
        for (size_t b = 0; b < batch_size; ++b) {
          const auto& row = plan.rows[base_out + b];
          const uint8_t weight = row.alpha_weights[in];
          if (weight == 0) {
            continue;
          }
          const auto& positions = row.alpha_positions[in];
          uint8_t* dst = scratch.data() + ((b * 17 + static_cast<size_t>(positions[0])) * tile);
          for (size_t pos = 0; pos < chunk; ++pos) {
            dst[pos] ^= src[pos];
          }
          if (weight >= 2) {
            dst = scratch.data() + ((b * 17 + static_cast<size_t>(positions[1])) * tile);
            for (size_t pos = 0; pos < chunk; ++pos) {
              dst[pos] ^= src[pos];
            }
          }
          if (weight >= 3) {
            dst = scratch.data() + ((b * 17 + static_cast<size_t>(positions[2])) * tile);
            for (size_t pos = 0; pos < chunk; ++pos) {
              dst[pos] ^= src[pos];
            }
          }
        }
      }
      for (size_t b = 0; b < batch_size; ++b) {
        uint8_t* out = outputs[base_out + b] + offset;
        const uint8_t* plane16 = scratch.data() + ((b * 17 + 16) * tile);
        std::memcpy(out, plane16, chunk);
        for (int t = 15; t >= 0; --t) {
            const uint8_t* src = scratch.data() + ((b * 17 + static_cast<size_t>(t)) * tile);
          for (size_t pos = 0; pos < chunk; ++pos) {
            const uint8_t x_exp = table.transition_x_exponents[static_cast<size_t>(t + 1)][static_cast<size_t>(t)];
            if (x_exp <= 4) {
              for (uint8_t step = 0; step < x_exp; ++step) {
                out[pos] = braid::core::GF256::MultiplyByX(out[pos]);
              }
              out[pos] ^= src[pos];
            } else {
              out[pos] =
                  braid::core::GF256::Multiply(table.transition_coeffs[static_cast<size_t>(t + 1)][static_cast<size_t>(t)], out[pos]) ^
                  src[pos];
            }
          }
        }
      }
    }
  }
}

}  // namespace braid::kernels
