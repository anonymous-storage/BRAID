#include "src/core/decomposition.h"

#include <cstddef>
#include <limits>

#include "src/core/gf256.h"

namespace braid::core {

DecompositionTable BuildDecompositionTable() {
  return BuildDecompositionTable(GF256::MultiplyByX(1));
}

DecompositionTable BuildDecompositionTable(uint8_t generator) {
  std::array<uint8_t, 17> ordered_columns{};
  ordered_columns[0] = 1;
  for (int t = 1; t < 17; ++t) {
    ordered_columns[t] = GF256::Multiply(ordered_columns[t - 1], generator);
  }
  DecompositionTable table = BuildDecompositionTable(ordered_columns);
  table.generator = generator;
  table.progression_order = true;
  return table;
}

DecompositionTable BuildDecompositionTable(const std::array<uint8_t, 17>& ordered_columns) {
  DecompositionTable table;
  table.order = 17;
  const uint8_t base_x = GF256::MultiplyByX(1);
  std::array<uint8_t, 256> discrete_log{};
  discrete_log.fill(0xFF);
  uint8_t probe = 1;
  for (uint8_t exponent = 0; exponent < 17; ++exponent) {
    discrete_log[probe] = exponent;
    probe = GF256::Multiply(probe, base_x);
  }

  table.x_powers = ordered_columns;
  for (int t = 0; t < 17; ++t) {
    table.x_exponents[t] = discrete_log[ordered_columns[t]];
  }
  table.generator = ordered_columns[1];
  table.generator_x_exponent = table.x_exponents[1];
  table.progression_order = true;
  for (int t = 1; t < 17; ++t) {
    if (table.x_powers[t] != GF256::Multiply(table.x_powers[t - 1], table.generator)) {
      table.progression_order = false;
      break;
    }
  }

  for (int from = 0; from < 17; ++from) {
    for (int to = 0; to <= from; ++to) {
      table.transition_coeffs[from][to] = GF256::Divide(table.x_powers[from], table.x_powers[to]);
      table.transition_x_exponents[from][to] =
          static_cast<uint8_t>((table.x_exponents[from] + 17 - table.x_exponents[to]) % 17);
    }
  }

  for (int t = 0; t < 17; ++t) {
    uint8_t value = table.x_powers[t];
    table.common_factor_nnz += static_cast<int>(__builtin_popcount(static_cast<unsigned>(value)));
    for (int bit = 0; bit < 8; ++bit) {
      table.common_factor_columns[bit][t] = static_cast<uint8_t>((value >> bit) & 1U);
    }
  }

  for (uint32_t mask = 0; mask < (1u << 17u); ++mask) {
    uint8_t value = 0;
    for (int t = 0; t < 17; ++t) {
      if (mask & (1u << t)) {
        value ^= table.x_powers[t];
      }
    }
    const uint8_t weight = static_cast<uint8_t>(__builtin_popcount(mask));
    if (table.weights[value] == 0 && table.alpha_masks[value] == 0 && value != 0) {
      table.weights[value] = std::numeric_limits<uint8_t>::max();
    }
    if (value == 0) {
      if (mask == 0) {
        table.alpha_masks[0] = 0;
        table.weights[0] = 0;
      }
      continue;
    }
    if (weight < table.weights[value] ||
        (weight == table.weights[value] && mask < table.alpha_masks[value])) {
      table.weights[value] = weight;
      table.alpha_masks[value] = mask;
    }
  }

  for (int value = 0; value < 256; ++value) {
    uint32_t mask = table.alpha_masks[static_cast<size_t>(value)];
    int position = 0;
    while (mask != 0 && position < 3) {
      table.alpha_positions[static_cast<size_t>(value)][static_cast<size_t>(position)] =
          static_cast<uint8_t>(__builtin_ctz(mask));
      mask &= (mask - 1);
      ++position;
    }
  }

  return table;
}

uint8_t ApplyCoefficientBraid(const DecompositionTable& table, uint8_t coefficient, uint8_t value) {
  uint8_t out = 0;
  uint32_t mask = table.alpha_masks[coefficient];
  while (mask != 0) {
    const int t = __builtin_ctz(mask);
    out ^= GF256::Multiply(table.x_powers[t], value);
    mask &= (mask - 1);
  }
  return out;
}

}  // namespace braid::core
