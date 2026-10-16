#ifndef BRAID_CORE_DECOMPOSITION_H_
#define BRAID_CORE_DECOMPOSITION_H_

#include <array>
#include <cstdint>

namespace braid::core {

struct DecompositionTable {
  int order = 0;
  int common_factor_nnz = 0;
  uint8_t generator = 0;
  uint8_t generator_x_exponent = 0;
  bool progression_order = true;
  std::array<uint8_t, 17> x_powers{};
  std::array<uint8_t, 17> x_exponents{};
  std::array<std::array<uint8_t, 17>, 17> transition_coeffs{};
  std::array<std::array<uint8_t, 17>, 17> transition_x_exponents{};
  std::array<uint32_t, 256> alpha_masks{};
  std::array<uint8_t, 256> weights{};
  std::array<std::array<uint8_t, 3>, 256> alpha_positions{};
  std::array<std::array<uint8_t, 17>, 8> common_factor_columns{};
};

DecompositionTable BuildDecompositionTable();
DecompositionTable BuildDecompositionTable(uint8_t generator);
DecompositionTable BuildDecompositionTable(const std::array<uint8_t, 17>& ordered_columns);
uint8_t ApplyCoefficientBraid(const DecompositionTable& table, uint8_t coefficient, uint8_t value);

}  // namespace braid::core

#endif  // BRAID_CORE_DECOMPOSITION_H_
