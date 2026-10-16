#include "src/runtime/avx512_plan_jit.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include <asmjit/x86.h>

#include "src/core/decomposition.h"
#include "src/core/gfni_matrix.h"
#include "src/runtime/types.h"

namespace braid::runtime {

namespace {

asmjit::JitRuntime& Runtime() {
  static asmjit::JitRuntime runtime;
  return runtime;
}

bool BooleanEnvironmentOverride(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return value[0] != '\0' && value[0] != '0';
}

size_t SizeEnvironmentOverride(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return end != value && *end == '\0'
             ? static_cast<size_t>(parsed)
             : fallback;
}

void EmitXtime(
    asmjit::x86::Assembler& assembler,
    uint32_t accumulator_register,
    uint32_t zero_register,
    uint32_t polynomial_register,
    uint32_t high_bits_register) {
  using namespace asmjit;
  const x86::Vec accumulator = x86::zmm(accumulator_register);
  assembler.vpcmpb(
      x86::k1,
      accumulator,
      x86::zmm(zero_register),
      1);
  assembler.vpaddb(accumulator, accumulator, accumulator);
  assembler.vpmovm2b(x86::zmm(high_bits_register), x86::k1);
  assembler.vpternlogd(
      accumulator,
      x86::zmm(high_bits_register),
      x86::zmm(polynomial_register),
      0x78);
}

Avx512JitBatchFn CompileFactorizedSingle(
    const Plan& plan,
    size_t row_index,
    void** allocation) {
  using namespace asmjit;

  JitRuntime& runtime = Runtime();
  CodeHolder code;
  code.init(runtime.environment(), runtime.cpu_features());
  x86::Assembler assembler(&code);
  const OutputPlanRow& row = plan.rows[row_index];

  constexpr uint32_t kValueRegister = 17;
  constexpr uint32_t kZeroRegister = 18;
  constexpr uint32_t kPolynomialRegister = 19;
  constexpr uint32_t kHighBitsRegister = 20;
  std::array<bool, 17> used_planes{};
  for (size_t input = 0; input < row.coefficients.size(); ++input) {
    const uint8_t weight = row.alpha_weights[input];
    for (uint8_t term = 0; term < weight; ++term) {
      used_planes[row.alpha_positions[input][term]] = true;
    }
  }

  assembler.mov(x86::r8, x86::ptr(x86::rsi));
  assembler.vpxord(
      x86::zmm(kZeroRegister),
      x86::zmm(kZeroRegister),
      x86::zmm(kZeroRegister));
  assembler.mov(x86::eax, 0x39393939u);
  assembler.vpbroadcastd(x86::zmm(kPolynomialRegister), x86::eax);
  assembler.xor_(x86::rcx, x86::rcx);

  Label done = assembler.new_label();
  Label loop = assembler.new_label();
  assembler.test(x86::rdx, x86::rdx);
  assembler.jz(done);
  assembler.bind(loop);
  for (uint32_t plane = 0; plane < used_planes.size(); ++plane) {
    if (used_planes[plane]) {
      assembler.vpxord(
          x86::zmm(plane),
          x86::zmm(plane),
          x86::zmm(plane));
    }
  }

  for (size_t input = 0; input < row.coefficients.size(); ++input) {
    const uint8_t weight = row.alpha_weights[input];
    if (weight == 0) {
      continue;
    }
    assembler.mov(
        x86::rax,
        x86::ptr(x86::rdi, static_cast<int32_t>(input * sizeof(void*))));
    assembler.vmovdqu8(
        x86::zmm(kValueRegister), x86::ptr(x86::rax, x86::rcx));
    for (uint8_t term = 0; term < weight; ++term) {
      const uint32_t plane = row.alpha_positions[input][term];
      assembler.vpxord(
          x86::zmm(plane),
          x86::zmm(plane),
          x86::zmm(kValueRegister));
    }
  }

  int accumulator_register = -1;
  int current_plane = -1;
  for (int plane = 16; plane >= 0; --plane) {
    if (!used_planes[static_cast<size_t>(plane)]) {
      continue;
    }
    if (current_plane < 0) {
      accumulator_register = plane;
      current_plane = plane;
      continue;
    }
    const uint8_t steps =
        plan.table.transition_x_exponents[static_cast<size_t>(current_plane)]
                                                [static_cast<size_t>(plane)];
    for (uint8_t step = 0; step < steps; ++step) {
      EmitXtime(
          assembler,
          static_cast<uint32_t>(accumulator_register),
          kZeroRegister,
          kPolynomialRegister,
          kHighBitsRegister);
    }
    assembler.vpxord(
        x86::zmm(static_cast<uint32_t>(accumulator_register)),
        x86::zmm(static_cast<uint32_t>(accumulator_register)),
        x86::zmm(static_cast<uint32_t>(plane)));
    current_plane = plane;
  }
  if (current_plane >= 0) {
    const uint8_t steps =
        plan.table.transition_x_exponents[static_cast<size_t>(current_plane)][0];
    for (uint8_t step = 0; step < steps; ++step) {
      EmitXtime(
          assembler,
          static_cast<uint32_t>(accumulator_register),
          kZeroRegister,
          kPolynomialRegister,
          kHighBitsRegister);
    }
    assembler.vmovdqu8(
        x86::ptr(x86::r8, x86::rcx),
        x86::zmm(static_cast<uint32_t>(accumulator_register)));
  } else {
    assembler.vmovdqu8(
        x86::ptr(x86::r8, x86::rcx),
        x86::zmm(kZeroRegister));
  }

  assembler.add(x86::rcx, 64);
  assembler.cmp(x86::rcx, x86::rdx);
  assembler.jb(loop);
  assembler.bind(done);
  assembler.ret();

  Avx512JitBatchFn batch_function = nullptr;
  Error error = runtime.add(&batch_function, &code);
  if (error != Error::kOk) {
    throw std::runtime_error(
        "AsmJit failed to allocate factorized BRAID-F code");
  }
  *allocation = reinterpret_cast<void*>(batch_function);
  return batch_function;
}

uint64_t SingleRowRouteCost(const OutputPlanRow& row) {
  uint64_t cost = 0;
  for (uint8_t coefficient : row.coefficients) {
    cost += static_cast<uint64_t>(
        __builtin_popcount(static_cast<unsigned>(coefficient)));
  }
  return cost;
}

uint64_t PairRouteCost(
    const OutputPlanRow& row0,
    const OutputPlanRow& row1) {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (int transform = 0; transform <= 2; ++transform) {
    uint64_t cost = transform == 0 ? 0 : 1;
    uint8_t shared_mask = 0;
    for (size_t input = 0; input < row0.coefficients.size(); ++input) {
      const uint8_t original0 = row0.coefficients[input];
      const uint8_t original1 = row1.coefficients[input];
      const uint8_t coefficient0 =
          transform == 2 ? original1 : original0;
      const uint8_t coefficient1 =
          transform == 0
              ? original1
              : static_cast<uint8_t>(original0 ^ original1);
      cost += static_cast<uint64_t>(
          __builtin_popcount(
              static_cast<unsigned>(coefficient0 | coefficient1)));
      shared_mask |= coefficient0 & coefficient1;
    }
    cost += 2u * static_cast<uint64_t>(
                     __builtin_popcount(
                         static_cast<unsigned>(shared_mask)));
    best = std::min(best, cost);
  }
  return best;
}

std::vector<size_t> SelectPairwiseRowOrder(
    const std::vector<OutputPlanRow>& rows) {
  std::vector<size_t> best_order(rows.size());
  std::iota(best_order.begin(), best_order.end(), size_t{0});
  uint64_t best_cost = std::numeric_limits<uint64_t>::max();
  const auto consider = [&](std::initializer_list<size_t> order) {
    std::vector<size_t> candidate(order);
    uint64_t cost = 0;
    for (size_t index = 0; index + 1 < candidate.size(); index += 2) {
      cost += PairRouteCost(
          rows[candidate[index]], rows[candidate[index + 1]]);
    }
    if ((candidate.size() & 1u) != 0) {
      cost += SingleRowRouteCost(rows[candidate.back()]);
    }
    if (cost < best_cost ||
        (cost == best_cost && candidate < best_order)) {
      best_cost = cost;
      best_order = std::move(candidate);
    }
  };
  if (rows.size() == 3) {
    consider({0, 1, 2});
    consider({0, 2, 1});
    consider({1, 2, 0});
  } else if (rows.size() == 4) {
    consider({0, 1, 2, 3});
    consider({0, 2, 1, 3});
    consider({0, 3, 1, 2});
  }
  return best_order;
}

struct RegisterDictionary {
  struct TransformOp {
    uint8_t destination = 0;
    uint8_t source = 0;
    bool swap = false;
  };

  uint8_t plane_count = 0;
  uint64_t route_weight = std::numeric_limits<uint64_t>::max();
  std::array<uint8_t, 17> positions{};
  std::array<uint32_t, 256> coordinates{};
  std::array<uint8_t, 3> transform_rows{1, 2, 4};
  std::array<TransformOp, 12> inverse_ops{};
  uint8_t inverse_op_count = 0;
};

struct OutputTransform {
  std::array<uint8_t, 3> rows{1, 2, 4};
  std::array<RegisterDictionary::TransformOp, 12> inverse_ops{};
  uint8_t inverse_op_count = 0;
  uint8_t inverse_xor_count = 0;
  uint64_t coarse_cost = 0;
};

bool BuildInverseTransform(OutputTransform* transform, size_t batch_size) {
  std::array<uint8_t, 3> matrix = transform->rows;
  uint8_t operation_count = 0;
  uint8_t xor_count = 0;
  for (size_t column = 0; column < batch_size; ++column) {
    size_t pivot = column;
    while (pivot < batch_size &&
           (matrix[pivot] & (uint8_t{1} << column)) == 0) {
      ++pivot;
    }
    if (pivot == batch_size) {
      return false;
    }
    if (pivot != column) {
      std::swap(matrix[pivot], matrix[column]);
      transform->inverse_ops[operation_count++] = {
          static_cast<uint8_t>(pivot),
          static_cast<uint8_t>(column),
          true};
    }
    for (size_t row = 0; row < batch_size; ++row) {
      if (row == column ||
          (matrix[row] & (uint8_t{1} << column)) == 0) {
        continue;
      }
      matrix[row] ^= matrix[column];
      transform->inverse_ops[operation_count++] = {
          static_cast<uint8_t>(row),
          static_cast<uint8_t>(column),
          false};
      ++xor_count;
    }
  }
  transform->inverse_op_count = operation_count;
  transform->inverse_xor_count = xor_count;
  return true;
}

uint8_t TransformedCoefficient(
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    size_t output,
    size_t input,
    const OutputTransform& transform) {
  uint8_t coefficient = 0;
  uint8_t mask = transform.rows[output];
  while (mask != 0) {
    const uint32_t source = static_cast<uint32_t>(__builtin_ctz(mask));
    coefficient ^= rows[base_row + source].coefficients[input];
    mask &= static_cast<uint8_t>(mask - 1);
  }
  return coefficient;
}

std::vector<OutputTransform> BuildOutputTransforms(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    size_t batch_size) {
  std::vector<OutputTransform> transforms;
  const auto add_transform = [&](std::array<uint8_t, 3> masks) {
    OutputTransform transform;
    transform.rows = masks;
    if (!BuildInverseTransform(&transform, batch_size)) {
      return;
    }
    uint64_t cost = transform.inverse_xor_count;
    for (size_t output = 0; output < batch_size; ++output) {
      for (size_t input = 0;
           input < rows[base_row + output].coefficients.size();
           ++input) {
        cost += table.weights[TransformedCoefficient(
            rows, base_row, output, input, transform)];
      }
    }
    transform.coarse_cost = cost;
    transforms.push_back(transform);
  };

  if (batch_size == 1) {
    add_transform({1, 2, 4});
  } else if (batch_size == 2) {
    add_transform({1, 2, 4});
    add_transform({1, 3, 4});
    add_transform({2, 3, 4});
  } else {
    for (uint8_t row0 = 1; row0 < 8; ++row0) {
      for (uint8_t row1 = 1; row1 < 8; ++row1) {
        for (uint8_t row2 = 1; row2 < 8; ++row2) {
          add_transform({row0, row1, row2});
        }
      }
    }
  }
  std::stable_sort(
      transforms.begin(),
      transforms.end(),
      [](const OutputTransform& lhs, const OutputTransform& rhs) {
        if (lhs.coarse_cost != rhs.coarse_cost) {
          return lhs.coarse_cost < rhs.coarse_cost;
        }
        return lhs.rows < rhs.rows;
      });
  const size_t transform_limit =
      batch_size == 3
          ? std::clamp<size_t>(
                SizeEnvironmentOverride(
                    "BRAID_GFNI_OUTPUT_TRANSFORMS", 4),
                1,
                transforms.size())
          : transforms.size();
  if (transforms.size() > transform_limit) {
    transforms.resize(transform_limit);
  }
  return transforms;
}

OutputTransform BuildQuadPolynomialOutputTransform(
    const std::vector<OutputPlanRow>& rows,
    size_t base_row) {
  OutputTransform best;
  best.coarse_cost = std::numeric_limits<uint64_t>::max();
  for (uint8_t row0 = 1; row0 < 8; ++row0) {
    for (uint8_t row1 = 1; row1 < 8; ++row1) {
      for (uint8_t row2 = 1; row2 < 8; ++row2) {
        OutputTransform candidate;
        candidate.rows = {row0, row1, row2};
        if (!BuildInverseTransform(&candidate, 3)) {
          continue;
        }
        uint64_t cost = candidate.inverse_xor_count;
        const size_t input_count = rows[base_row].coefficients.size();
        for (size_t output = 0; output < 3; ++output) {
          for (size_t input = 0; input < input_count; ++input) {
            cost += static_cast<uint64_t>(__builtin_popcount(
                static_cast<unsigned>(TransformedCoefficient(
                    rows, base_row, output, input, candidate))));
          }
        }
        candidate.coarse_cost = cost;
        if (cost < best.coarse_cost ||
            (cost == best.coarse_cost && candidate.rows < best.rows)) {
          best = candidate;
        }
      }
    }
  }
  return best;
}

std::array<uint32_t, 256> BuildDictionaryCoordinates(
    const core::DecompositionTable& table,
    const std::array<uint8_t, 17>& positions,
    size_t plane_count) {
  const size_t subset_count = size_t{1} << plane_count;
  std::vector<uint8_t> values(subset_count);
  std::vector<uint8_t> weights(subset_count);
  std::array<uint8_t, 256> best_weights{};
  std::array<uint32_t, 256> coordinates{};
  best_weights.fill(std::numeric_limits<uint8_t>::max());
  best_weights[0] = 0;

  for (uint32_t mask = 1; mask < subset_count; ++mask) {
    const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(mask));
    const uint32_t previous = mask & (mask - 1);
    values[mask] = static_cast<uint8_t>(
        values[previous] ^ table.x_powers[positions[bit]]);
    weights[mask] = static_cast<uint8_t>(weights[previous] + 1);
    const uint8_t value = values[mask];
    if (weights[mask] < best_weights[value] ||
        (weights[mask] == best_weights[value] &&
         mask < coordinates[value])) {
      best_weights[value] = weights[mask];
      coordinates[value] = mask;
    }
  }
  return coordinates;
}

struct QuadRegisterDictionary {
  uint8_t plane_count = 0;
  uint64_t route_weight = std::numeric_limits<uint64_t>::max();
  std::array<uint8_t, 17> positions{};
  std::array<uint32_t, 256> coordinates{};
};

QuadRegisterDictionary BuildQuadRegisterDictionary(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row) {
  QuadRegisterDictionary best;
  const size_t plane_count = std::clamp<size_t>(
      SizeEnvironmentOverride("BRAID_GFNI_QUAD_PLANES", 9), 8, 10);
  best.plane_count = static_cast<uint8_t>(plane_count);

  std::array<uint32_t, 256> frequency{};
  for (size_t output = 0; output < 4; ++output) {
    for (uint8_t coefficient : rows[base_row + output].coefficients) {
      ++frequency[coefficient];
    }
  }
  const auto consider = [&](const std::array<uint8_t, 17>& positions) {
    const auto coordinates =
        BuildDictionaryCoordinates(table, positions, plane_count);
    uint64_t route_weight = 0;
    for (size_t value = 1; value < frequency.size(); ++value) {
      if (frequency[value] == 0) {
        continue;
      }
      const uint32_t coordinate = coordinates[value];
      if (coordinate == 0) {
        return;
      }
      route_weight += static_cast<uint64_t>(frequency[value]) *
                      static_cast<uint64_t>(__builtin_popcount(coordinate));
    }
    if (route_weight < best.route_weight ||
        (route_weight == best.route_weight && positions < best.positions)) {
      best.route_weight = route_weight;
      best.positions = positions;
      best.coordinates = coordinates;
    }
  };

  std::array<uint8_t, 17> positions{};
  std::function<void(uint8_t, size_t)> enumerate =
      [&](uint8_t next, size_t selected) {
        if (selected == plane_count) {
          consider(positions);
          return;
        }
        const size_t remaining = plane_count - selected;
        for (uint8_t position = next;
             static_cast<size_t>(position) + remaining <= 17;
             ++position) {
          positions[selected] = position;
          enumerate(static_cast<uint8_t>(position + 1), selected + 1);
        }
      };
  enumerate(0, 0);
  if (best.route_weight == std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error(
        "failed to construct a quad register-constrained BRAID-F dictionary");
  }
  return best;
}

RegisterDictionary BuildRegisterDictionary(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    size_t batch_size) {
  RegisterDictionary best;
  const size_t plane_count =
      batch_size == 1
          ? 17
          : (batch_size == 2
                 ? 14
                 : std::clamp<size_t>(
                       SizeEnvironmentOverride(
                           "BRAID_GFNI_TRIPLE_PLANES", 9),
                       8,
                       9));
  best.plane_count = static_cast<uint8_t>(plane_count);

  const std::vector<OutputTransform> transforms =
      BuildOutputTransforms(table, rows, base_row, batch_size);
  std::vector<std::array<uint32_t, 256>> frequencies(transforms.size());
  for (size_t transform_index = 0;
       transform_index < transforms.size();
       ++transform_index) {
    const OutputTransform& transform = transforms[transform_index];
    auto& frequency = frequencies[transform_index];
    for (size_t output = 0; output < batch_size; ++output) {
      for (size_t input = 0;
           input < rows[base_row + output].coefficients.size();
           ++input) {
        const uint8_t coefficient = TransformedCoefficient(
            rows, base_row, output, input, transform);
        ++frequency[coefficient];
      }
    }
  }

  // Coordinates depend only on the selected factor planes. Build them once
  // per subset and score every output transform against the same result.
  // The previous transform-major loop rebuilt this exponential table for
  // every candidate transform without changing the selected dictionary.
  const auto score_positions =
      [&](const std::array<uint8_t, 17>& positions,
          size_t selected_plane_count) {
    RegisterDictionary candidate_best;
    candidate_best.plane_count =
        static_cast<uint8_t>(selected_plane_count);
    const auto coordinates =
        BuildDictionaryCoordinates(
            table, positions, selected_plane_count);
    for (size_t transform_index = 0;
         transform_index < transforms.size();
         ++transform_index) {
      const OutputTransform& transform = transforms[transform_index];
      const auto& frequency = frequencies[transform_index];
      uint64_t route_weight = transform.inverse_xor_count;
      bool valid = true;
      for (size_t value = 1; value < frequency.size(); ++value) {
        if (frequency[value] == 0) {
          continue;
        }
        const uint32_t coordinate = coordinates[value];
        if (coordinate == 0) {
          valid = false;
          break;
        }
        route_weight +=
            static_cast<uint64_t>(frequency[value]) *
            static_cast<uint64_t>(__builtin_popcount(coordinate));
      }
      if (!valid) {
        continue;
      }
      if (route_weight < candidate_best.route_weight ||
          (route_weight == candidate_best.route_weight &&
           positions < candidate_best.positions)) {
        candidate_best.route_weight = route_weight;
        candidate_best.positions = positions;
        candidate_best.coordinates = coordinates;
        candidate_best.transform_rows = transform.rows;
        candidate_best.inverse_ops = transform.inverse_ops;
        candidate_best.inverse_op_count = transform.inverse_op_count;
      }
    }
    return candidate_best;
  };

  const auto better_dictionary =
      [](const RegisterDictionary& lhs,
         const RegisterDictionary& rhs) {
    return lhs.route_weight < rhs.route_weight ||
           (lhs.route_weight == rhs.route_weight &&
            lhs.positions < rhs.positions);
  };

  if (plane_count == 17) {
    std::array<uint8_t, 17> positions{};
    std::iota(positions.begin(), positions.end(), uint8_t{0});
    best = score_positions(positions, plane_count);
  } else if (BooleanEnvironmentOverride(
                 "BRAID_GFNI_EXACT_DICTIONARY",
                 plane_count != 9)) {
    std::array<uint8_t, 17> positions{};
    std::function<void(uint8_t, size_t)> enumerate =
        [&](uint8_t next, size_t selected) {
      if (selected == plane_count) {
        const RegisterDictionary candidate =
            score_positions(positions, plane_count);
        if (better_dictionary(candidate, best)) {
          best = candidate;
        }
        return;
      }
      const size_t remaining = plane_count - selected;
      for (uint8_t position = next;
           static_cast<size_t>(position) + remaining <= 17;
           ++position) {
        positions[selected] = position;
        enumerate(static_cast<uint8_t>(position + 1), selected + 1);
      }
    };
    enumerate(0, 0);
  } else {
    // Start from all 17 cyclic factors and greedily remove the factor whose
    // absence gives the lowest route cost. A bounded exchange refinement at
    // the target size recovers interactions missed by greedy elimination.
    // This changes plan construction only; the emitted kernel still uses the
    // same register-constrained representation and exact route scoring.
    std::vector<uint8_t> selected(17);
    std::iota(selected.begin(), selected.end(), uint8_t{0});
    while (selected.size() > plane_count) {
      RegisterDictionary step_best;
      std::vector<uint8_t> step_positions;
      for (size_t removed = 0; removed < selected.size(); ++removed) {
        std::array<uint8_t, 17> positions{};
        size_t destination = 0;
        for (size_t index = 0; index < selected.size(); ++index) {
          if (index != removed) {
            positions[destination++] = selected[index];
          }
        }
        const RegisterDictionary candidate =
            score_positions(positions, destination);
        if (better_dictionary(candidate, step_best)) {
          step_best = candidate;
          step_positions.assign(
              positions.begin(), positions.begin() + destination);
        }
      }
      selected = std::move(step_positions);
    }

    std::array<uint8_t, 17> selected_positions{};
    std::copy(selected.begin(), selected.end(), selected_positions.begin());
    best = score_positions(selected_positions, plane_count);
    for (int round = 0; round < 8; ++round) {
      RegisterDictionary round_best = best;
      std::vector<uint8_t> round_positions = selected;
      for (size_t slot = 0; slot < selected.size(); ++slot) {
        for (uint8_t replacement = 0; replacement < 17; ++replacement) {
          if (std::find(selected.begin(), selected.end(), replacement) !=
              selected.end()) {
            continue;
          }
          std::vector<uint8_t> trial = selected;
          trial[slot] = replacement;
          std::sort(trial.begin(), trial.end());
          std::array<uint8_t, 17> positions{};
          std::copy(trial.begin(), trial.end(), positions.begin());
          const RegisterDictionary candidate =
              score_positions(positions, plane_count);
          if (better_dictionary(candidate, round_best)) {
            round_best = candidate;
            round_positions = std::move(trial);
          }
        }
      }
      if (!better_dictionary(round_best, best)) {
        break;
      }
      best = std::move(round_best);
      selected = std::move(round_positions);
    }
  }

  if (best.route_weight == std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error(
        "failed to construct a register-constrained BRAID-F dictionary");
  }
  return best;
}

struct RegisterBatchStats {
  uint64_t xor_ops_before = 0;
  uint64_t xor_ops_after = 0;
  uint64_t potential_xor_ops = 0;
  uint32_t groups = 0;
  uint32_t potential_groups = 0;
  uint32_t triple_groups = 0;
};

Avx512JitBatchFn CompileRegisterConstrainedBatch(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    size_t batch_size,
    bool streaming_stores,
    bool enable_slp,
    bool allow_triples,
    size_t max_pair_distance,
    size_t prefetch_distance,
    uint32_t prefetch_hint,
    size_t prefetch_input_stride,
    const RegisterDictionary* prepared_dictionary,
    RegisterBatchStats* stats,
    void** allocation) {
  using namespace asmjit;

  if (!table.progression_order || batch_size == 0 || batch_size > 3 ||
      base_row + batch_size > rows.size()) {
    throw std::invalid_argument(
        "invalid register-constrained BRAID-F JIT batch");
  }
  const RegisterDictionary dictionary =
      prepared_dictionary != nullptr
          ? *prepared_dictionary
          : BuildRegisterDictionary(table, rows, base_row, batch_size);
  const uint32_t plane_count = dictionary.plane_count;
  const uint32_t accumulator_count =
      static_cast<uint32_t>(batch_size) * plane_count;
  const uint32_t value_register = accumulator_count;
  const uint32_t matrix_register = value_register + 1;
  const uint32_t paired_value_register = matrix_register + 1;
  const uint32_t combined_value_register = paired_value_register + 1;
  if (matrix_register >= 32 ||
      (enable_slp && combined_value_register >= 32)) {
    throw std::runtime_error(
        "register-constrained BRAID-F dictionary exceeds ZMM budget");
  }

  JitRuntime& runtime = Runtime();
  CodeHolder code;
  code.init(runtime.environment(), runtime.cpu_features());
  x86::Assembler assembler(&code);
  const std::array<x86::Gp, 3> output_ptrs{x86::r8, x86::r9, x86::r10};
  for (size_t output = 0; output < batch_size; ++output) {
    assembler.mov(
        output_ptrs[output],
        x86::ptr(
            x86::rsi,
            static_cast<int32_t>(output * sizeof(void*))));
  }
  assembler.mov(
      x86::rax,
      core::BuildGfniAffineMatrix(table.generator));
  assembler.vpbroadcastq(
      x86::zmm(matrix_register), x86::rax);
  assembler.xor_(x86::rcx, x86::rcx);

  Label done = assembler.new_label();
  Label loop = assembler.new_label();
  assembler.test(x86::rdx, x86::rdx);
  assembler.jz(done);
  assembler.bind(loop);

  std::array<bool, 51> initialized{};
  std::array<uint32_t, 3> used_plane_masks{};
  const auto accumulate =
      [&](uint32_t accumulator, uint32_t source_register) {
    if (initialized[accumulator]) {
      assembler.vpxord(
          x86::zmm(accumulator),
          x86::zmm(accumulator),
          x86::zmm(source_register));
    } else {
      assembler.vmovdqa64(
          x86::zmm(accumulator),
          x86::zmm(source_register));
      initialized[accumulator] = true;
    }
  };

  struct InputRoutes {
    std::array<uint32_t, 3> routes{};
    bool used = false;
  };
  const size_t input_count = rows[base_row].coefficients.size();
  std::vector<InputRoutes> input_routes(input_count);
  OutputTransform output_transform;
  output_transform.rows = dictionary.transform_rows;
  for (size_t input = 0; input < input_count; ++input) {
    InputRoutes& input_route = input_routes[input];
    for (size_t output = 0; output < batch_size; ++output) {
      const uint8_t coefficient = TransformedCoefficient(
          rows, base_row, output, input, output_transform);
      input_route.routes[output] =
          dictionary.coordinates[coefficient];
      used_plane_masks[output] |= input_route.routes[output];
      input_route.used |= input_route.routes[output] != 0;
      if (stats != nullptr) {
        stats->xor_ops_before += static_cast<uint64_t>(
            __builtin_popcount(input_route.routes[output]));
      }
    }
  }

  // BRAID's factor routes expose XOR common subexpressions across inputs.
  // An N-input group with S shared targets replaces N*S route updates with
  // S updates and N-1 XORs. This is a pure XOR-SLP over factorized routes.
  struct SlpCandidate {
    std::array<size_t, 3> members{};
    uint8_t member_count = 0;
    uint32_t savings = 0;
    uint32_t shared_targets = 0;
  };
  std::vector<SlpCandidate> candidates;
  for (size_t lhs = 0; lhs < input_count; ++lhs) {
    if (!input_routes[lhs].used) {
      continue;
    }
    for (size_t rhs = lhs + 1; rhs < input_count; ++rhs) {
      if (max_pair_distance != 0 &&
          rhs - lhs > max_pair_distance) {
        break;
      }
      if (!input_routes[rhs].used) {
        continue;
      }
      uint32_t shared_targets = 0;
      for (size_t output = 0; output < batch_size; ++output) {
        shared_targets += static_cast<uint32_t>(
            __builtin_popcount(
                input_routes[lhs].routes[output] &
                input_routes[rhs].routes[output]));
      }
      if (shared_targets >= 2) {
        candidates.push_back(
            {{{lhs, rhs, 0}}, 2, shared_targets - 1, shared_targets});
      }
      if (!enable_slp || !allow_triples ||
          max_pair_distance != 0) {
        continue;
      }
      for (size_t third = rhs + 1; third < input_count; ++third) {
        if (!input_routes[third].used) {
          continue;
        }
        uint32_t triple_shared_targets = 0;
        for (size_t output = 0; output < batch_size; ++output) {
          triple_shared_targets += static_cast<uint32_t>(
              __builtin_popcount(
                  input_routes[lhs].routes[output] &
                  input_routes[rhs].routes[output] &
                  input_routes[third].routes[output]));
        }
        if (triple_shared_targets >= 2) {
          candidates.push_back(
              {{{lhs, rhs, third}},
               3,
               2 * triple_shared_targets - 1,
               triple_shared_targets});
        }
      }
    }
  }
  std::stable_sort(
      candidates.begin(),
      candidates.end(),
      [](const SlpCandidate& lhs, const SlpCandidate& rhs) {
        if (lhs.savings != rhs.savings) {
          return lhs.savings > rhs.savings;
        }
        if (lhs.shared_targets != rhs.shared_targets) {
          return lhs.shared_targets > rhs.shared_targets;
        }
        if (lhs.member_count != rhs.member_count) {
          return lhs.member_count < rhs.member_count;
        }
        const size_t lhs_distance =
            lhs.members[lhs.member_count - 1] - lhs.members[0];
        const size_t rhs_distance =
            rhs.members[rhs.member_count - 1] - rhs.members[0];
        if (lhs_distance != rhs_distance) {
          return lhs_distance < rhs_distance;
        }
        return lhs.members < rhs.members;
      });
  const size_t kNoGroup = input_count;
  std::vector<size_t> input_group(input_count, kNoGroup);
  std::vector<SlpCandidate> groups;
  uint64_t selected_savings = 0;
  uint32_t selected_triple_groups = 0;
  for (const SlpCandidate& candidate : candidates) {
    bool available = true;
    for (size_t index = 0; index < candidate.member_count; ++index) {
      available &=
          input_group[candidate.members[index]] == kNoGroup;
    }
    if (!available) {
      continue;
    }
    const size_t group = groups.size();
    groups.push_back(candidate);
    for (size_t index = 0; index < candidate.member_count; ++index) {
      input_group[candidate.members[index]] = group;
    }
    selected_savings += candidate.savings;
    selected_triple_groups += candidate.member_count == 3 ? 1u : 0u;
    if (stats != nullptr) {
      ++stats->potential_groups;
    }
  }
  if (stats != nullptr) {
    stats->potential_xor_ops = stats->xor_ops_before - selected_savings;
  }
  if (!enable_slp) {
    std::fill(input_group.begin(), input_group.end(), kNoGroup);
    groups.clear();
    selected_savings = 0;
  } else if (stats != nullptr) {
    stats->groups = stats->potential_groups;
    stats->triple_groups = selected_triple_groups;
  }
  if (stats != nullptr) {
    stats->xor_ops_after = stats->xor_ops_before - selected_savings;
  }

  const auto shared_routes_for_group =
      [&](const SlpCandidate& group) {
    std::array<uint32_t, 3> shared_routes =
        input_routes[group.members[0]].routes;
    for (size_t member = 1; member < group.member_count; ++member) {
      for (size_t output = 0; output < batch_size; ++output) {
        shared_routes[output] &=
            input_routes[group.members[member]].routes[output];
      }
    }
    return shared_routes;
  };

  const auto load_input =
      [&](size_t input, uint32_t destination_register) {
    assembler.mov(
        x86::rax,
        x86::ptr(
            x86::rdi,
            static_cast<int32_t>(input * sizeof(void*))));
    if (prefetch_distance != 0 &&
        input % prefetch_input_stride == 0) {
      const auto prefetch_address = x86::ptr(
          x86::rax,
          x86::rcx,
          0,
          static_cast<int32_t>(prefetch_distance));
      switch (prefetch_hint) {
        case 0:
          assembler.prefetchnta(prefetch_address);
          break;
        case 2:
          assembler.prefetcht1(prefetch_address);
          break;
        case 3:
          assembler.prefetcht2(prefetch_address);
          break;
        default:
          assembler.prefetcht0(prefetch_address);
          break;
      }
    }
    assembler.vmovdqu8(
        x86::zmm(destination_register),
        x86::ptr(x86::rax, x86::rcx));
  };
  const auto route_value =
      [&](const std::array<uint32_t, 3>& routes,
          uint32_t source_register) {
    for (size_t output = 0; output < batch_size; ++output) {
      uint32_t route = routes[output];
      while (route != 0) {
        const uint32_t plane =
            static_cast<uint32_t>(__builtin_ctz(route));
        const uint32_t accumulator =
            static_cast<uint32_t>(output) * plane_count + plane;
        accumulate(accumulator, source_register);
        route &= route - 1;
      }
    }
  };
  for (size_t input = 0; input < input_count; ++input) {
    if (!input_routes[input].used) {
      continue;
    }
    const size_t group_index = input_group[input];
    if (group_index == kNoGroup) {
      load_input(input, value_register);
      route_value(input_routes[input].routes, value_register);
      continue;
    }
    const SlpCandidate& group = groups[group_index];
    if (input != group.members[0]) {
      continue;
    }
    const std::array<uint32_t, 3> shared_routes =
        shared_routes_for_group(group);
    if (group.member_count == 2) {
      const size_t paired_input = group.members[1];
      load_input(input, value_register);
      load_input(paired_input, paired_value_register);
      assembler.vpxord(
          x86::zmm(combined_value_register),
          x86::zmm(value_register),
          x86::zmm(paired_value_register));
      std::array<uint32_t, 3> input_only_routes{};
      std::array<uint32_t, 3> paired_only_routes{};
      for (size_t output = 0; output < batch_size; ++output) {
        input_only_routes[output] =
            input_routes[input].routes[output] & ~shared_routes[output];
        paired_only_routes[output] =
            input_routes[paired_input].routes[output] &
            ~shared_routes[output];
      }
      route_value(shared_routes, combined_value_register);
      route_value(input_only_routes, value_register);
      route_value(paired_only_routes, paired_value_register);
      continue;
    }

    const std::array<uint32_t, 3> member_registers{
        combined_value_register, value_register, paired_value_register};
    for (size_t member = 0; member < group.member_count; ++member) {
      const size_t member_input = group.members[member];
      load_input(member_input, member_registers[member]);
      std::array<uint32_t, 3> member_only_routes{};
      for (size_t output = 0; output < batch_size; ++output) {
        member_only_routes[output] =
            input_routes[member_input].routes[output] &
            ~shared_routes[output];
      }
      route_value(member_only_routes, member_registers[member]);
    }
    assembler.vpternlogd(
        x86::zmm(combined_value_register),
        x86::zmm(value_register),
        x86::zmm(paired_value_register),
        0x96);
    route_value(shared_routes, combined_value_register);
  }

  struct ReductionAction {
    int32_t xor_register = -1;
  };
  std::array<uint32_t, 3> result_registers{};
  std::array<std::vector<ReductionAction>, 3> reduction_actions;
  const bool needs_zero_result =
      std::any_of(
          used_plane_masks.begin(),
          used_plane_masks.begin() + batch_size,
          [](uint32_t mask) { return mask == 0; });
  if (needs_zero_result) {
    assembler.vpxord(
        x86::zmm(value_register),
        x86::zmm(value_register),
        x86::zmm(value_register));
  }
  for (size_t output = 0; output < batch_size; ++output) {
    uint32_t used = used_plane_masks[output];
    if (used == 0) {
      result_registers[output] = value_register;
      continue;
    }
    int current_plane =
        31 - __builtin_clz(static_cast<unsigned>(used));
    const uint32_t accumulator =
        static_cast<uint32_t>(output) * plane_count +
        static_cast<uint32_t>(current_plane);
    result_registers[output] = accumulator;
    used &= ~(1u << current_plane);
    while (used != 0) {
      const int next_plane =
          31 - __builtin_clz(static_cast<unsigned>(used));
      const int steps =
          static_cast<int>(
              dictionary.positions[static_cast<size_t>(current_plane)]) -
          static_cast<int>(
              dictionary.positions[static_cast<size_t>(next_plane)]);
      for (int step = 0; step < steps; ++step) {
        reduction_actions[output].push_back({-1});
      }
      reduction_actions[output].push_back(
          {static_cast<int32_t>(
              static_cast<uint32_t>(output) * plane_count +
              static_cast<uint32_t>(next_plane))});
      current_plane = next_plane;
      used &= ~(1u << next_plane);
    }
    for (uint8_t step = 0;
         step <
         dictionary.positions[static_cast<size_t>(current_plane)];
         ++step) {
      reduction_actions[output].push_back({-1});
    }
  }

  // Keep independent GFNI dependency chains adjacent. Ice Lake can overlap
  // their latency; reducing one complete output before the next serializes
  // sixteen affine transforms and erases the route-count benefit.
  size_t action_count = 0;
  for (size_t output = 0; output < batch_size; ++output) {
    action_count =
        std::max(action_count, reduction_actions[output].size());
  }
  for (size_t action_index = 0;
       action_index < action_count;
       ++action_index) {
    for (size_t output = 0; output < batch_size; ++output) {
      if (action_index >= reduction_actions[output].size()) {
        continue;
      }
      const auto action = reduction_actions[output][action_index];
      const uint32_t accumulator = result_registers[output];
      if (action.xor_register < 0) {
        assembler.vgf2p8affineqb(
            x86::zmm(accumulator),
            x86::zmm(accumulator),
            x86::zmm(matrix_register),
            0);
      } else {
        assembler.vpxord(
            x86::zmm(accumulator),
            x86::zmm(accumulator),
            x86::zmm(
                static_cast<uint32_t>(action.xor_register)));
      }
    }
  }

  for (uint8_t index = 0; index < dictionary.inverse_op_count; ++index) {
    const RegisterDictionary::TransformOp& operation =
        dictionary.inverse_ops[index];
    if (operation.swap) {
      std::swap(
          result_registers[operation.destination],
          result_registers[operation.source]);
      continue;
    }
    assembler.vpxord(
        x86::zmm(result_registers[operation.destination]),
        x86::zmm(result_registers[operation.destination]),
        x86::zmm(result_registers[operation.source]));
  }
  for (size_t output = 0; output < batch_size; ++output) {
    if (streaming_stores) {
      assembler.vmovntdq(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    } else {
      assembler.vmovdqu8(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    }
  }

  assembler.add(x86::rcx, 64);
  assembler.cmp(x86::rcx, x86::rdx);
  assembler.jb(loop);
  assembler.bind(done);
  assembler.ret();

  Avx512JitBatchFn batch_function = nullptr;
  const Error error = runtime.add(&batch_function, &code);
  if (error != Error::kOk) {
    throw std::runtime_error(
        "AsmJit failed to allocate register-constrained BRAID-F code");
  }
  *allocation = reinterpret_cast<void*>(batch_function);
  return batch_function;
}

Avx512JitBatchFn CompileQuadDeferredDictionaryBatch(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    const QuadRegisterDictionary& dictionary,
    bool streaming_stores,
    void** allocation) {
  using namespace asmjit;

  if (!table.progression_order || dictionary.plane_count != 8 ||
      base_row + 4 > rows.size()) {
    throw std::invalid_argument(
        "invalid deferred-plane BRAID-F JIT batch");
  }

  const size_t input_count = rows[base_row].coefficients.size();
  std::array<uint32_t, 8> last_row_frequency{};
  for (size_t input = 0; input < input_count; ++input) {
    uint32_t route = dictionary.coordinates[
        rows[base_row + 3].coefficients[input]];
    while (route != 0) {
      const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(route));
      ++last_row_frequency[plane];
      route &= route - 1;
    }
  }
  const uint32_t deferred_plane = static_cast<uint32_t>(
      std::min_element(
          last_row_frequency.begin(), last_row_frequency.end()) -
      last_row_frequency.begin());
  const auto physical_register =
      [&](size_t output, uint32_t plane) -> uint32_t {
    if (output < 3) {
      return static_cast<uint32_t>(output) * 8 + plane;
    }
    if (plane == deferred_plane) {
      return 31;
    }
    return 24 + plane - (plane > deferred_plane ? 1u : 0u);
  };

  JitRuntime& runtime = Runtime();
  CodeHolder code;
  code.init(runtime.environment(), runtime.cpu_features());
  x86::Assembler assembler(&code);
  const std::array<x86::Gp, 4> output_ptrs{
      x86::r8, x86::r9, x86::r10, x86::r11};
  for (size_t output = 0; output < 4; ++output) {
    assembler.mov(
        output_ptrs[output],
        x86::ptr(x86::rsi, static_cast<int32_t>(output * sizeof(void*))));
  }

  // The matrix remains in memory so all 32 ZMM registers can represent
  // factor planes. ZMM31 is the source register during the main pass and
  // becomes the one deferred plane before reduction.
  assembler.mov(x86::rsi, x86::rsp);
  assembler.and_(x86::rsp, -64);
  assembler.sub(x86::rsp, 64);
  assembler.mov(x86::rax, core::BuildGfniAffineMatrix(table.generator));
  assembler.vpbroadcastq(x86::zmm(31), x86::rax);
  assembler.vmovdqu8(x86::ptr(x86::rsp), x86::zmm(31));
  assembler.xor_(x86::rcx, x86::rcx);

  Label done = assembler.new_label();
  Label loop = assembler.new_label();
  assembler.test(x86::rdx, x86::rdx);
  assembler.jz(done);
  assembler.bind(loop);

  std::array<bool, 32> initialized{};
  std::array<uint32_t, 4> used_masks{};
  std::vector<size_t> deferred_inputs;
  for (size_t input = 0; input < input_count; ++input) {
    std::array<uint32_t, 4> routes{};
    bool used = false;
    for (size_t output = 0; output < 4; ++output) {
      routes[output] = dictionary.coordinates[
          rows[base_row + output].coefficients[input]];
      used_masks[output] |= routes[output];
      used |= routes[output] != 0;
    }
    if ((routes[3] & (1u << deferred_plane)) != 0) {
      deferred_inputs.push_back(input);
      routes[3] &= ~(1u << deferred_plane);
    }
    if (!used) {
      continue;
    }
    assembler.mov(
        x86::rax,
        x86::ptr(x86::rdi, static_cast<int32_t>(input * sizeof(void*))));
    assembler.vmovdqu8(x86::zmm(31), x86::ptr(x86::rax, x86::rcx));
    for (size_t output = 0; output < 4; ++output) {
      uint32_t route = routes[output];
      while (route != 0) {
        const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(route));
        const uint32_t logical =
            static_cast<uint32_t>(output) * 8 + plane;
        const uint32_t target = physical_register(output, plane);
        if (initialized[logical]) {
          assembler.vpxord(
              x86::zmm(target), x86::zmm(target), x86::zmm(31));
        } else {
          assembler.vmovdqa64(x86::zmm(target), x86::zmm(31));
          initialized[logical] = true;
        }
        route &= route - 1;
      }
    }
  }

  const uint32_t deferred_logical = 24 + deferred_plane;
  if (deferred_inputs.empty()) {
    assembler.vpxord(x86::zmm(31), x86::zmm(31), x86::zmm(31));
  } else {
    assembler.mov(
        x86::rax,
        x86::ptr(
            x86::rdi,
            static_cast<int32_t>(deferred_inputs.front() * sizeof(void*))));
    assembler.vmovdqu8(x86::zmm(31), x86::ptr(x86::rax, x86::rcx));
    for (size_t index = 1; index < deferred_inputs.size(); ++index) {
      assembler.mov(
          x86::rax,
          x86::ptr(
              x86::rdi,
              static_cast<int32_t>(deferred_inputs[index] * sizeof(void*))));
      assembler.vpxord(
          x86::zmm(31), x86::zmm(31), x86::ptr(x86::rax, x86::rcx));
    }
  }
  initialized[deferred_logical] = true;

  struct ReductionAction {
    int32_t xor_register = -1;
  };
  std::array<uint32_t, 4> result_registers{};
  std::array<std::vector<ReductionAction>, 4> reduction_actions;
  for (size_t output = 0; output < 4; ++output) {
    uint32_t used = used_masks[output];
    if (used == 0) {
      const uint32_t result = physical_register(output, 0);
      assembler.vpxord(
          x86::zmm(result), x86::zmm(result), x86::zmm(result));
      result_registers[output] = result;
      continue;
    }
    int current_plane = 31 - __builtin_clz(static_cast<unsigned>(used));
    result_registers[output] = physical_register(
        output, static_cast<uint32_t>(current_plane));
    used &= ~(1u << current_plane);
    while (used != 0) {
      const int next_plane = 31 - __builtin_clz(static_cast<unsigned>(used));
      const int steps =
          static_cast<int>(dictionary.positions[current_plane]) -
          static_cast<int>(dictionary.positions[next_plane]);
      for (int step = 0; step < steps; ++step) {
        reduction_actions[output].push_back({-1});
      }
      reduction_actions[output].push_back(
          {static_cast<int32_t>(physical_register(
              output, static_cast<uint32_t>(next_plane)))});
      current_plane = next_plane;
      used &= ~(1u << next_plane);
    }
    for (uint8_t step = 0;
         step < dictionary.positions[current_plane];
         ++step) {
      reduction_actions[output].push_back({-1});
    }
  }

  size_t action_count = 0;
  for (const auto& actions : reduction_actions) {
    action_count = std::max(action_count, actions.size());
  }
  for (size_t action_index = 0; action_index < action_count; ++action_index) {
    for (size_t output = 0; output < 4; ++output) {
      if (action_index >= reduction_actions[output].size()) {
        continue;
      }
      const ReductionAction action = reduction_actions[output][action_index];
      const uint32_t accumulator = result_registers[output];
      if (action.xor_register < 0) {
        assembler.vgf2p8affineqb(
            x86::zmm(accumulator),
            x86::zmm(accumulator),
            x86::ptr(x86::rsp),
            0);
      } else {
        assembler.vpxord(
            x86::zmm(accumulator),
            x86::zmm(accumulator),
            x86::zmm(static_cast<uint32_t>(action.xor_register)));
      }
    }
  }

  for (size_t output = 0; output < 4; ++output) {
    if (streaming_stores) {
      assembler.vmovntdq(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    } else {
      assembler.vmovdqu8(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    }
  }

  assembler.add(x86::rcx, 64);
  assembler.cmp(x86::rcx, x86::rdx);
  assembler.jb(loop);
  assembler.bind(done);
  assembler.mov(x86::rsp, x86::rsi);
  assembler.ret();

  Avx512JitBatchFn batch_function = nullptr;
  const Error error = runtime.add(&batch_function, &code);
  if (error != Error::kOk) {
    throw std::runtime_error(
        "AsmJit failed to allocate deferred-plane BRAID-F code");
  }
  *allocation = reinterpret_cast<void*>(batch_function);
  return batch_function;
}

Avx512JitBatchFn CompileQuadRegisterConstrainedBatch(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    bool streaming_stores,
    void** allocation) {
  using namespace asmjit;

  if (!table.progression_order || base_row + 4 > rows.size()) {
    throw std::invalid_argument(
        "invalid quad register-constrained BRAID-F JIT batch");
  }
  const QuadRegisterDictionary dictionary =
      BuildQuadRegisterDictionary(table, rows, base_row);
  if (dictionary.plane_count == 8 &&
      BooleanEnvironmentOverride("BRAID_GFNI_QUAD_DEFERRED", true)) {
    return CompileQuadDeferredDictionaryBatch(
        table,
        rows,
        base_row,
        dictionary,
        streaming_stores,
        allocation);
  }
  const uint32_t plane_count = dictionary.plane_count;
  const uint32_t resident_last = 30 - 3 * plane_count;
  const uint32_t spill_count = plane_count - resident_last;
  const int32_t matrix_offset =
      static_cast<int32_t>(((spill_count * 64 + 63) / 64) * 64);
  const int32_t frame_size = matrix_offset + 64;
  const uint32_t scratch_register = 30;
  const uint32_t value_register = 31;

  JitRuntime& runtime = Runtime();
  CodeHolder code;
  code.init(runtime.environment(), runtime.cpu_features());
  x86::Assembler assembler(&code);
  const std::array<x86::Gp, 4> output_ptrs{
      x86::r8, x86::r9, x86::r10, x86::r11};
  for (size_t output = 0; output < 4; ++output) {
    assembler.mov(
        output_ptrs[output],
        x86::ptr(x86::rsi, static_cast<int32_t>(output * sizeof(void*))));
  }
  assembler.mov(x86::rsi, x86::rsp);
  assembler.and_(x86::rsp, -64);
  assembler.sub(x86::rsp, frame_size);
  assembler.mov(x86::rax, core::BuildGfniAffineMatrix(table.generator));
  assembler.vpbroadcastq(x86::zmm(scratch_register), x86::rax);
  assembler.vmovdqu8(
      x86::ptr(x86::rsp, matrix_offset), x86::zmm(scratch_register));
  assembler.xor_(x86::rcx, x86::rcx);

  Label done = assembler.new_label();
  Label loop = assembler.new_label();
  assembler.test(x86::rdx, x86::rdx);
  assembler.jz(done);
  assembler.bind(loop);

  std::array<bool, 40> initialized{};
  std::array<uint32_t, 4> used_masks{};
  const auto resident_register = [&](size_t output, uint32_t plane) {
    return output < 3
               ? static_cast<uint32_t>(output) * plane_count + plane
               : 3 * plane_count + plane;
  };
  const auto spill_slot = [&](uint32_t plane) {
    return static_cast<int32_t>((plane - resident_last) * 64);
  };
  const size_t input_count = rows[base_row].coefficients.size();
  for (size_t input = 0; input < input_count; ++input) {
    std::array<uint32_t, 4> routes{};
    bool used = false;
    for (size_t output = 0; output < 4; ++output) {
      routes[output] = dictionary.coordinates[
          rows[base_row + output].coefficients[input]];
      used_masks[output] |= routes[output];
      used |= routes[output] != 0;
    }
    if (!used) {
      continue;
    }
    assembler.mov(
        x86::rax,
        x86::ptr(x86::rdi, static_cast<int32_t>(input * sizeof(void*))));
    assembler.vmovdqu8(
        x86::zmm(value_register), x86::ptr(x86::rax, x86::rcx));
    for (size_t output = 0; output < 4; ++output) {
      uint32_t route = routes[output];
      while (route != 0) {
        const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(route));
        const uint32_t logical =
            static_cast<uint32_t>(output) * plane_count + plane;
        if (output == 3 && plane >= resident_last) {
          const int32_t offset = spill_slot(plane);
          if (!initialized[logical]) {
            assembler.vmovdqu8(
                x86::ptr(x86::rsp, offset), x86::zmm(value_register));
          } else {
            assembler.vmovdqu8(
                x86::zmm(scratch_register), x86::ptr(x86::rsp, offset));
            assembler.vpxord(
                x86::zmm(scratch_register),
                x86::zmm(scratch_register),
                x86::zmm(value_register));
            assembler.vmovdqu8(
                x86::ptr(x86::rsp, offset), x86::zmm(scratch_register));
          }
        } else {
          const uint32_t target = resident_register(output, plane);
          if (!initialized[logical]) {
            assembler.vmovdqa64(
                x86::zmm(target), x86::zmm(value_register));
          } else {
            assembler.vpxord(
                x86::zmm(target),
                x86::zmm(target),
                x86::zmm(value_register));
          }
        }
        initialized[logical] = true;
        route &= route - 1;
      }
    }
  }

  struct ReductionAction {
    int32_t xor_register = -1;
  };
  std::array<uint32_t, 3> result_registers{};
  std::array<std::vector<ReductionAction>, 3> actions;
  for (size_t output = 0; output < 3; ++output) {
    uint32_t used = used_masks[output];
    if (used == 0) {
      assembler.vpxord(
          x86::zmm(result_registers[output]),
          x86::zmm(result_registers[output]),
          x86::zmm(result_registers[output]));
      continue;
    }
    int current_plane =
        31 - __builtin_clz(static_cast<unsigned>(used));
    result_registers[output] =
        resident_register(output, static_cast<uint32_t>(current_plane));
    used &= ~(1u << current_plane);
    while (used != 0) {
      const int next_plane =
          31 - __builtin_clz(static_cast<unsigned>(used));
      const int steps =
          static_cast<int>(dictionary.positions[current_plane]) -
          static_cast<int>(dictionary.positions[next_plane]);
      for (int step = 0; step < steps; ++step) {
        actions[output].push_back({-1});
      }
      actions[output].push_back({static_cast<int32_t>(
          resident_register(output, static_cast<uint32_t>(next_plane)))});
      current_plane = next_plane;
      used &= ~(1u << next_plane);
    }
    for (uint8_t step = 0; step < dictionary.positions[current_plane]; ++step) {
      actions[output].push_back({-1});
    }
  }
  size_t action_count = 0;
  for (const auto& chain : actions) {
    action_count = std::max(action_count, chain.size());
  }
  for (size_t index = 0; index < action_count; ++index) {
    for (size_t output = 0; output < 3; ++output) {
      if (index >= actions[output].size()) {
        continue;
      }
      if (actions[output][index].xor_register < 0) {
        assembler.vgf2p8affineqb(
            x86::zmm(result_registers[output]),
            x86::zmm(result_registers[output]),
            x86::ptr(x86::rsp, matrix_offset),
            0);
      } else {
        assembler.vpxord(
            x86::zmm(result_registers[output]),
            x86::zmm(result_registers[output]),
            x86::zmm(static_cast<uint32_t>(
                actions[output][index].xor_register)));
      }
    }
  }
  for (size_t output = 0; output < 3; ++output) {
    if (streaming_stores) {
      assembler.vmovntdq(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    } else {
      assembler.vmovdqu8(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(result_registers[output]));
    }
  }

  std::array<uint32_t, 10> last_registers{};
  for (uint32_t plane = 0; plane < plane_count; ++plane) {
    if (plane < resident_last) {
      last_registers[plane] = resident_register(3, plane);
    } else {
      const uint32_t target = plane - resident_last;
      last_registers[plane] = target;
      assembler.vmovdqu8(
          x86::zmm(target), x86::ptr(x86::rsp, spill_slot(plane)));
    }
  }
  uint32_t used = used_masks[3];
  uint32_t last_result = scratch_register;
  if (used == 0) {
    assembler.vpxord(
        x86::zmm(last_result), x86::zmm(last_result), x86::zmm(last_result));
  } else {
    int current_plane =
        31 - __builtin_clz(static_cast<unsigned>(used));
    last_result = last_registers[static_cast<size_t>(current_plane)];
    used &= ~(1u << current_plane);
    while (used != 0) {
      const int next_plane =
          31 - __builtin_clz(static_cast<unsigned>(used));
      for (int step = dictionary.positions[current_plane];
           step > dictionary.positions[next_plane]; --step) {
        assembler.vgf2p8affineqb(
            x86::zmm(last_result),
            x86::zmm(last_result),
            x86::ptr(x86::rsp, matrix_offset),
            0);
      }
      assembler.vpxord(
          x86::zmm(last_result),
          x86::zmm(last_result),
          x86::zmm(last_registers[static_cast<size_t>(next_plane)]));
      current_plane = next_plane;
      used &= ~(1u << next_plane);
    }
    for (int step = dictionary.positions[current_plane]; step > 0; --step) {
      assembler.vgf2p8affineqb(
          x86::zmm(last_result),
          x86::zmm(last_result),
          x86::ptr(x86::rsp, matrix_offset),
          0);
    }
  }
  if (streaming_stores) {
    assembler.vmovntdq(
        x86::ptr(output_ptrs[3], x86::rcx), x86::zmm(last_result));
  } else {
    assembler.vmovdqu8(
        x86::ptr(output_ptrs[3], x86::rcx), x86::zmm(last_result));
  }

  assembler.add(x86::rcx, 64);
  assembler.cmp(x86::rcx, x86::rdx);
  assembler.jb(loop);
  assembler.bind(done);
  assembler.mov(x86::rsp, x86::rsi);
  assembler.ret();

  Avx512JitBatchFn batch_function = nullptr;
  const Error error = runtime.add(&batch_function, &code);
  if (error != Error::kOk) {
    throw std::runtime_error(
        "AsmJit failed to allocate quad register-constrained BRAID-F code");
  }
  *allocation = reinterpret_cast<void*>(batch_function);
  return batch_function;
}

Avx512JitBatchFn CompilePolynomialBatch(
    const core::DecompositionTable& table,
    const std::vector<OutputPlanRow>& rows,
    size_t base_row,
    size_t batch_size,
    bool streaming_stores,
    bool use_gfni,
    bool default_pin_input_pointers,
    bool enable_pair_frf,
    bool enable_quad_frf,
    bool enable_triple_frf,
    RegisterBatchStats* stats,
    void** allocation) {
  using namespace asmjit;

  if (batch_size == 0 || batch_size > 4 ||
      base_row + batch_size > rows.size()) {
    throw std::invalid_argument("invalid BRAID-F JIT batch");
  }
  const size_t input_count = rows[base_row].coefficients.size();
  const bool pin_input_pointers =
      batch_size <= 2 && input_count <= 10 &&
      BooleanEnvironmentOverride(
          "BRAID_GFNI_PIN_INPUT_POINTERS", default_pin_input_pointers);
  JitRuntime& runtime = Runtime();
  CodeHolder code;
  code.init(runtime.environment(), runtime.cpu_features());
  x86::Assembler assembler(&code);
  if (pin_input_pointers) {
    assembler.push(x86::rbx);
    assembler.push(x86::rbp);
    assembler.push(x86::r12);
    assembler.push(x86::r13);
    assembler.push(x86::r14);
    assembler.push(x86::r15);
  }
  const std::array<x86::Gp, 4> output_ptrs{
      x86::r8, x86::r9, x86::r10, x86::r11};
  for (size_t output = 0; output < batch_size; ++output) {
    assembler.mov(
        output_ptrs[output],
        x86::ptr(x86::rsi, static_cast<int32_t>(output * sizeof(void*))));
  }
  const std::array<x86::Gp, 10> pinned_input_ptrs{
      x86::r10, x86::r11, x86::rbx, x86::rbp, x86::r12,
      x86::r13, x86::r14, x86::r15, x86::rsi, x86::rdi};
  if (pin_input_pointers) {
    // Load rdi last because it is the input-pointer array base.
    for (size_t input = 0; input < input_count; ++input) {
      assembler.mov(
          pinned_input_ptrs[input],
          x86::ptr(x86::rdi, static_cast<int32_t>(input * sizeof(void*))));
    }
  }
  const auto store_output =
      [&](size_t output, uint32_t register_id) {
    if (streaming_stores) {
      assembler.vmovntdq(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(register_id));
    } else {
      assembler.vmovdqu8(
          x86::ptr(output_ptrs[output], x86::rcx),
          x86::zmm(register_id));
    }
  };

  const bool quad = batch_size == 4;
  const bool request_quad_output_transform =
      quad && use_gfni && BooleanEnvironmentOverride(
                               "BRAID_GFNI_QUAD_OUTPUT_TRANSFORM", false);
  const OutputTransform quad_output_transform =
      request_quad_output_transform
          ? BuildQuadPolynomialOutputTransform(rows, base_row)
          : OutputTransform{};
  bool use_quad_output_transform = request_quad_output_transform;
  if (use_quad_output_transform) {
    for (size_t output = 0; output < 3; ++output) {
      uint8_t used = 0;
      for (size_t input = 0;
           input < rows[base_row + output].coefficients.size();
           ++input) {
        used |= TransformedCoefficient(
            rows, base_row, output, input, quad_output_transform);
      }
      use_quad_output_transform &= used == 0xffu;
    }
  }
  int pair_transform = 0;
  if (batch_size == 2) {
    const auto route_score = [&](int transform) {
      uint32_t score = transform == 0 ? 0 : 1;
      uint8_t shared_mask = 0;
      for (size_t input = 0;
           input < rows[base_row].coefficients.size();
           ++input) {
        const uint8_t original0 =
            rows[base_row].coefficients[input];
        const uint8_t original1 =
            rows[base_row + 1].coefficients[input];
        const uint8_t coefficient0 =
            transform == 2 ? original1 : original0;
        const uint8_t coefficient1 =
            transform == 0 ? original1
                           : static_cast<uint8_t>(
                                 original0 ^ original1);
        score += static_cast<uint32_t>(
            __builtin_popcount(
                static_cast<unsigned>(coefficient0 | coefficient1)));
        shared_mask |= coefficient0 & coefficient1;
      }
      return score +
             2u * static_cast<uint32_t>(
                      __builtin_popcount(
                          static_cast<unsigned>(shared_mask)));
    };
    uint32_t best_score = route_score(0);
    for (int candidate = 1; candidate <= 2; ++candidate) {
      const uint32_t score = route_score(candidate);
      if (score < best_score) {
        best_score = score;
        pair_transform = candidate;
      }
    }
  }
  const auto coefficient_at =
      [&](size_t output, size_t input) -> uint8_t {
    if (use_quad_output_transform && output < 3) {
      return TransformedCoefficient(
          rows, base_row, output, input, quad_output_transform);
    }
    if (batch_size != 2 || pair_transform == 0) {
      return rows[base_row + output].coefficients[input];
    }
    const uint8_t original0 =
        rows[base_row].coefficients[input];
    const uint8_t original1 =
        rows[base_row + 1].coefficients[input];
    if (output == 0) {
      return pair_transform == 2 ? original1 : original0;
    }
    return static_cast<uint8_t>(original0 ^ original1);
  };
  std::array<uint8_t, 4> used_plane_masks{};
  uint8_t shared_plane_mask = 0;
  for (size_t output = 0; output < batch_size; ++output) {
    for (size_t input = 0;
         input < rows[base_row + output].coefficients.size();
         ++input) {
      used_plane_masks[output] |=
          coefficient_at(output, input);
    }
  }
  if (batch_size == 2) {
    for (size_t input = 0;
         input < rows[base_row].coefficients.size();
         ++input) {
      shared_plane_mask |=
          coefficient_at(0, input) &
          coefficient_at(1, input);
    }
  }
  std::array<uint32_t, 32> physical_register{};
  std::array<int8_t, 32> spill_slots{};
  spill_slots.fill(-1);
  for (uint32_t logical = 0; logical < physical_register.size(); ++logical) {
    physical_register[logical] = logical;
  }
  if (quad) {
    std::array<uint32_t, 32> route_frequency{};
    for (size_t output = 0; output < 4; ++output) {
      for (size_t input = 0;
           input < rows[base_row + output].coefficients.size();
           ++input) {
        const uint8_t coefficient = coefficient_at(output, input);
        for (size_t plane = 0; plane < 8; ++plane) {
          route_frequency[output * 8 + plane] +=
              (coefficient >> plane) & 1u;
        }
      }
    }
    std::array<uint32_t, 32> logical_order{};
    for (uint32_t logical = 0; logical < logical_order.size(); ++logical) {
      logical_order[logical] = logical;
    }
    std::stable_sort(
        logical_order.begin(),
        logical_order.end(),
        [&](uint32_t lhs, uint32_t rhs) {
          return route_frequency[lhs] < route_frequency[rhs];
        });
    spill_slots[logical_order[0]] = 0;
    spill_slots[logical_order[1]] = 1;
    uint32_t next_register = 0;
    for (uint32_t logical = 0; logical < physical_register.size(); ++logical) {
      if (spill_slots[logical] >= 0) {
        physical_register[logical] =
            static_cast<uint32_t>(30 + spill_slots[logical]);
      } else {
        physical_register[logical] = next_register++;
      }
    }
  }
  using QuadRoutes = std::array<uint8_t, 4>;
  using PairRoutes = std::array<uint8_t, 2>;
  struct PairFrfGroup {
    size_t first = 0;
    size_t second = 0;
    PairRoutes shared{};
    uint32_t savings = 0;
  };
  std::vector<PairRoutes> pair_routes(input_count);
  std::vector<PairFrfGroup> pair_groups;
  std::vector<size_t> pair_input_group(input_count, input_count);
  if (batch_size == 2 && enable_pair_frf) {
    const auto route_cost = [](const PairRoutes& routes) {
      return static_cast<uint32_t>(__builtin_popcount(
          static_cast<unsigned>(routes[0] | routes[1])));
    };
    for (size_t input = 0; input < input_count; ++input) {
      for (size_t output = 0; output < 2; ++output) {
        pair_routes[input][output] = coefficient_at(output, input);
      }
    }
    std::vector<PairFrfGroup> candidates;
    for (size_t first = 0; first < input_count; ++first) {
      for (size_t second = first + 1; second < input_count; ++second) {
        PairRoutes shared{};
        PairRoutes first_only{};
        PairRoutes second_only{};
        for (size_t output = 0; output < 2; ++output) {
          shared[output] = static_cast<uint8_t>(
              pair_routes[first][output] & pair_routes[second][output]);
          first_only[output] = static_cast<uint8_t>(
              pair_routes[first][output] & ~shared[output]);
          second_only[output] = static_cast<uint8_t>(
              pair_routes[second][output] & ~shared[output]);
        }
        const uint32_t before =
            route_cost(pair_routes[first]) + route_cost(pair_routes[second]);
        const uint32_t after = route_cost(shared) + route_cost(first_only) +
                               route_cost(second_only) + 1;
        if (before > after) {
          candidates.push_back({first, second, shared, before - after});
        }
      }
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const PairFrfGroup& lhs, const PairFrfGroup& rhs) {
          if (lhs.savings != rhs.savings) {
            return lhs.savings > rhs.savings;
          }
          return std::tie(lhs.first, lhs.second) <
                 std::tie(rhs.first, rhs.second);
        });
    for (const PairFrfGroup& candidate : candidates) {
      if (pair_input_group[candidate.first] != input_count ||
          pair_input_group[candidate.second] != input_count) {
        continue;
      }
      const size_t group = pair_groups.size();
      pair_groups.push_back(candidate);
      pair_input_group[candidate.first] = group;
      pair_input_group[candidate.second] = group;
    }
    if (stats != nullptr) {
      for (const PairRoutes& routes : pair_routes) {
        stats->xor_ops_before += route_cost(routes);
      }
      uint64_t savings = 0;
      for (const PairFrfGroup& group : pair_groups) {
        savings += group.savings;
      }
      stats->xor_ops_after = stats->xor_ops_before - savings;
      stats->potential_xor_ops = stats->xor_ops_after;
      stats->groups = static_cast<uint32_t>(pair_groups.size());
      stats->potential_groups = stats->groups;
    }
  }
  std::vector<QuadRoutes> quad_routes(input_count);
  struct QuadFrfGroup {
    size_t first = 0;
    size_t second = 0;
    QuadRoutes shared{};
    int32_t shared_spill_logical = -1;
    uint32_t savings = 0;
  };
  std::vector<QuadFrfGroup> quad_groups;
  std::vector<size_t> quad_input_group(input_count, input_count);
  if (quad && enable_quad_frf) {
    std::array<uint8_t, 4> spill_masks{};
    for (size_t output = 0; output < 4; ++output) {
      for (size_t plane = 0; plane < 8; ++plane) {
        const size_t logical = output * 8 + plane;
        if (spill_slots[logical] >= 0) {
          spill_masks[output] |= static_cast<uint8_t>(1u << plane);
        }
      }
    }
    for (size_t input = 0; input < input_count; ++input) {
      for (size_t output = 0; output < 4; ++output) {
        quad_routes[input][output] =
            coefficient_at(output, input);
      }
    }
    std::vector<QuadFrfGroup> candidates;
    for (size_t first = 0; first < input_count; ++first) {
      for (size_t second = first + 1; second < input_count; ++second) {
        QuadRoutes shared{};
        uint32_t shared_targets = 0;
        int32_t shared_spill_logical = -1;
        uint32_t shared_spill_count = 0;
        for (size_t output = 0; output < 4; ++output) {
          const uint8_t full_shared = static_cast<uint8_t>(
              quad_routes[first][output] & quad_routes[second][output]);
          const uint8_t spilled =
              static_cast<uint8_t>(full_shared & spill_masks[output]);
          shared_spill_count += static_cast<uint32_t>(
              __builtin_popcount(static_cast<unsigned>(spilled)));
          if (spilled != 0) {
            const uint32_t plane =
                static_cast<uint32_t>(__builtin_ctz(spilled));
            shared_spill_logical =
                static_cast<int32_t>(output * 8 + plane);
          }
          shared[output] = full_shared;
          shared_targets += static_cast<uint32_t>(
              __builtin_popcount(static_cast<unsigned>(shared[output])));
        }
        if (shared_spill_count > 1) {
          shared_targets = 0;
          shared_spill_logical = -1;
          for (size_t output = 0; output < 4; ++output) {
            shared[output] &= static_cast<uint8_t>(~spill_masks[output]);
            shared_targets += static_cast<uint32_t>(
                __builtin_popcount(static_cast<unsigned>(shared[output])));
          }
        }
        if (shared_targets >= 2) {
          candidates.push_back(
              {first, second, shared, shared_spill_logical,
               shared_targets - 1});
        }
      }
    }
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const QuadFrfGroup& lhs, const QuadFrfGroup& rhs) {
          if (lhs.savings != rhs.savings) {
            return lhs.savings > rhs.savings;
          }
          const size_t lhs_distance = lhs.second - lhs.first;
          const size_t rhs_distance = rhs.second - rhs.first;
          if (lhs_distance != rhs_distance) {
            return lhs_distance < rhs_distance;
          }
          return std::tie(lhs.first, lhs.second) <
                 std::tie(rhs.first, rhs.second);
        });
    std::vector<int32_t> candidate_index(input_count * input_count, -1);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const QuadFrfGroup& candidate = candidates[index];
      candidate_index[candidate.first * input_count + candidate.second] =
          static_cast<int32_t>(index);
    }
    const auto candidate_for =
        [&](size_t first, size_t second) -> const QuadFrfGroup* {
          if (first > second) {
            std::swap(first, second);
          }
          const int32_t index =
              candidate_index[first * input_count + second];
          return index < 0 ? nullptr : &candidates[static_cast<size_t>(index)];
        };
    for (const QuadFrfGroup& candidate : candidates) {
      if (quad_input_group[candidate.first] != input_count ||
          quad_input_group[candidate.second] != input_count) {
        continue;
      }
      const size_t group = quad_groups.size();
      quad_groups.push_back(candidate);
      quad_input_group[candidate.first] = group;
      quad_input_group[candidate.second] = group;
    }
    bool improved = true;
    while (improved) {
      improved = false;
      for (size_t lhs = 0; lhs < quad_groups.size() && !improved; ++lhs) {
        for (size_t rhs = lhs + 1; rhs < quad_groups.size(); ++rhs) {
          const std::array<size_t, 4> nodes{
              quad_groups[lhs].first, quad_groups[lhs].second,
              quad_groups[rhs].first, quad_groups[rhs].second};
          const uint32_t current =
              quad_groups[lhs].savings + quad_groups[rhs].savings;
          const std::array<std::array<size_t, 4>, 2> alternatives{{
              {{nodes[0], nodes[2], nodes[1], nodes[3]}},
              {{nodes[0], nodes[3], nodes[1], nodes[2]}},
          }};
          for (const auto& alternative : alternatives) {
            const QuadFrfGroup* first =
                candidate_for(alternative[0], alternative[1]);
            const QuadFrfGroup* second =
                candidate_for(alternative[2], alternative[3]);
            if (first == nullptr || second == nullptr ||
                first->savings + second->savings <= current) {
              continue;
            }
            quad_groups[lhs] = *first;
            quad_groups[rhs] = *second;
            improved = true;
            break;
          }
          if (improved) {
            break;
          }
        }
      }
    }
    std::fill(
        quad_input_group.begin(), quad_input_group.end(), input_count);
    for (size_t group = 0; group < quad_groups.size(); ++group) {
      quad_input_group[quad_groups[group].first] = group;
      quad_input_group[quad_groups[group].second] = group;
    }
    if (stats != nullptr) {
      for (const QuadRoutes& routes : quad_routes) {
        for (uint8_t route : routes) {
          stats->xor_ops_before += static_cast<uint64_t>(
              __builtin_popcount(static_cast<unsigned>(route)));
        }
      }
      uint64_t savings = 0;
      for (const QuadFrfGroup& group : quad_groups) {
        savings += group.savings;
      }
      stats->xor_ops_after = stats->xor_ops_before - savings;
      stats->potential_xor_ops = stats->xor_ops_after;
      stats->groups = static_cast<uint32_t>(quad_groups.size());
      stats->potential_groups = stats->groups;
    }
  }
  using TripleRoutes = std::array<uint8_t, 3>;
  struct TripleFrfGroup {
    size_t first = 0;
    size_t second = 0;
    TripleRoutes shared{};
    uint32_t savings = 0;
  };
  std::vector<TripleRoutes> triple_routes(input_count);
  std::vector<TripleFrfGroup> triple_groups;
  std::vector<size_t> triple_input_group(input_count, input_count);
  if (batch_size == 3 && enable_triple_frf) {
    for (size_t input = 0; input < input_count; ++input) {
      for (size_t output = 0; output < 3; ++output) {
        triple_routes[input][output] = coefficient_at(output, input);
      }
    }
    std::vector<TripleFrfGroup> candidates;
    for (size_t first = 0; first < input_count; ++first) {
      for (size_t second = first + 1; second < input_count; ++second) {
        TripleRoutes shared{};
        uint32_t targets = 0;
        for (size_t output = 0; output < 3; ++output) {
          shared[output] = static_cast<uint8_t>(
              triple_routes[first][output] &
              triple_routes[second][output]);
          targets += static_cast<uint32_t>(__builtin_popcount(
              static_cast<unsigned>(shared[output])));
        }
        if (targets >= 2) {
          candidates.push_back({first, second, shared, targets - 1});
        }
      }
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const TripleFrfGroup& lhs, const TripleFrfGroup& rhs) {
          if (lhs.savings != rhs.savings) {
            return lhs.savings > rhs.savings;
          }
          return std::tie(lhs.first, lhs.second) <
                 std::tie(rhs.first, rhs.second);
        });
    for (const TripleFrfGroup& candidate : candidates) {
      if (triple_input_group[candidate.first] != input_count ||
          triple_input_group[candidate.second] != input_count) {
        continue;
      }
      const size_t group = triple_groups.size();
      triple_groups.push_back(candidate);
      triple_input_group[candidate.first] = group;
      triple_input_group[candidate.second] = group;
    }
    if (stats != nullptr) {
      for (const TripleRoutes& routes : triple_routes) {
        for (uint8_t route : routes) {
          stats->xor_ops_before += static_cast<uint64_t>(__builtin_popcount(
              static_cast<unsigned>(route)));
        }
      }
      uint64_t savings = 0;
      for (const TripleFrfGroup& group : triple_groups) {
        savings += group.savings;
      }
      stats->xor_ops_after = stats->xor_ops_before - savings;
      stats->potential_xor_ops = stats->xor_ops_after;
      stats->groups = static_cast<uint32_t>(triple_groups.size());
      stats->potential_groups = stats->groups;
    }
  }
  if (quad) {
    // Four rows consume all 32 ZMM registers. Keep two planes in aligned
    // stack slots during accumulation, then spill the last row once so its
    // registers can hold the reduction constants.
    assembler.mov(x86::rsi, x86::rsp);
    assembler.and_(x86::rsp, -64);
    assembler.sub(x86::rsp, 640);
    if (use_gfni) {
      assembler.mov(
          x86::rax,
          core::BuildGfniAffineMatrix(2));
      assembler.vpbroadcastq(x86::zmm(30), x86::rax);
      assembler.vmovdqu8(x86::ptr(x86::rsp, 128), x86::zmm(30));
    }
  }

  const uint32_t value_register = quad ? 31u : 24u;
  const uint32_t spill_register = 30u;
  uint32_t zero_register = quad ? 24u : 25u;
  uint32_t polynomial_register = quad ? 25u : 26u;
  uint32_t high_bits_register = quad ? 27u : 28u;
  uint32_t gfni_matrix_register = quad ? 25u : 26u;
  if (!quad) {
    assembler.vpxord(
        x86::zmm(zero_register),
        x86::zmm(zero_register),
        x86::zmm(zero_register));
    if (use_gfni) {
      assembler.mov(
          x86::rax,
          core::BuildGfniAffineMatrix(2));
      assembler.vpbroadcastq(
          x86::zmm(gfni_matrix_register), x86::rax);
    } else {
      assembler.mov(x86::eax, 0x39393939u);
      assembler.vpbroadcastd(x86::zmm(polynomial_register), x86::eax);
    }
  }
  assembler.xor_(x86::rcx, x86::rcx);

  Label done = assembler.new_label();
  Label loop = assembler.new_label();
  assembler.test(x86::rdx, x86::rdx);
  assembler.jz(done);
  assembler.bind(loop);
  if (quad) {
    for (uint32_t logical = 0; logical < physical_register.size();
         ++logical) {
      const size_t output = logical / 8;
      const size_t plane = logical % 8;
      if ((used_plane_masks[output] & (1u << plane)) == 0 ||
          spill_slots[logical] >= 0) {
        continue;
      }
      const x86::Vec accumulator =
          x86::zmm(physical_register[logical]);
      assembler.vpxord(
          accumulator, accumulator, accumulator);
    }
  }
  if (quad) {
    assembler.vpxord(
        x86::zmm(spill_register),
        x86::zmm(spill_register),
        x86::zmm(spill_register));
    assembler.vmovdqu8(x86::ptr(x86::rsp, 0), x86::zmm(spill_register));
    assembler.vmovdqu8(x86::ptr(x86::rsp, 64), x86::zmm(spill_register));
  }

  std::array<bool, 32> accumulator_initialized{};
  const auto accumulate_value =
      [&](uint32_t logical, uint32_t physical, uint32_t source_register) {
    const x86::Vec accumulator = x86::zmm(physical);
    if (accumulator_initialized[logical]) {
      assembler.vpxord(
          accumulator, accumulator, x86::zmm(source_register));
    } else {
      assembler.vmovdqa64(
          accumulator, x86::zmm(source_register));
      accumulator_initialized[logical] = true;
    }
  };
  const auto load_quad_input =
      [&](size_t input, uint32_t destination) {
        if (pin_input_pointers) {
          assembler.vmovdqu8(
              x86::zmm(destination),
              x86::ptr(pinned_input_ptrs[input], x86::rcx));
        } else {
          assembler.mov(
              x86::rax,
              x86::ptr(
                  x86::rdi,
                  static_cast<int32_t>(input * sizeof(void*))));
          assembler.vmovdqu8(
              x86::zmm(destination), x86::ptr(x86::rax, x86::rcx));
        }
      };
  const auto route_pair =
      [&](const PairRoutes& routes, uint32_t source_register) {
        uint8_t remaining0 = routes[0];
        uint8_t remaining1 = routes[1];
        while ((remaining0 | remaining1) != 0) {
          const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(
              static_cast<unsigned>(remaining0 | remaining1)));
          const bool route0 = (remaining0 & (1u << plane)) != 0;
          const bool route1 = (remaining1 & (1u << plane)) != 0;
          const uint32_t logical =
              route0 && route1 ? 16 + plane : (route0 ? plane : 8 + plane);
          accumulate_value(logical, logical, source_register);
          remaining0 &= static_cast<uint8_t>(~(1u << plane));
          remaining1 &= static_cast<uint8_t>(~(1u << plane));
        }
      };
  const auto route_quad =
      [&](const QuadRoutes& routes,
          uint32_t source_register,
          uint32_t scratch) {
        for (size_t output = 0; output < 4; ++output) {
          uint8_t route = routes[output];
          while (route != 0) {
            const uint32_t plane =
                static_cast<uint32_t>(__builtin_ctz(route));
            const uint32_t logical =
                static_cast<uint32_t>(output * 8 + plane);
            if (spill_slots[logical] >= 0) {
              const int32_t spill_offset =
                  static_cast<int32_t>(spill_slots[logical] * 64);
              assembler.vmovdqu8(
                  x86::zmm(scratch), x86::ptr(x86::rsp, spill_offset));
              assembler.vpxord(
                  x86::zmm(scratch),
                  x86::zmm(scratch),
                  x86::zmm(source_register));
              assembler.vmovdqu8(
                  x86::ptr(x86::rsp, spill_offset), x86::zmm(scratch));
            } else {
              const x86::Vec accumulator =
                  x86::zmm(physical_register[logical]);
              assembler.vpxord(
                  accumulator, accumulator, x86::zmm(source_register));
            }
            route &= static_cast<uint8_t>(route - 1);
          }
        }
      };
  const auto route_triple =
      [&](const TripleRoutes& routes, uint32_t source_register) {
        for (size_t output = 0; output < 3; ++output) {
          uint8_t route = routes[output];
          while (route != 0) {
            const uint32_t plane =
                static_cast<uint32_t>(__builtin_ctz(route));
            const uint32_t logical =
                static_cast<uint32_t>(output * 8 + plane);
            accumulate_value(
                logical, physical_register[logical], source_register);
            route &= static_cast<uint8_t>(route - 1);
          }
        }
      };
  for (size_t input = 0; input < input_count; ++input) {
    if (batch_size == 2 && enable_pair_frf &&
        pair_input_group[input] != input_count) {
      const PairFrfGroup& group =
          pair_groups[pair_input_group[input]];
      if (input != group.first) {
        continue;
      }
      PairRoutes first_only{};
      PairRoutes second_only{};
      for (size_t output = 0; output < 2; ++output) {
        first_only[output] = static_cast<uint8_t>(
            pair_routes[group.first][output] & ~group.shared[output]);
        second_only[output] = static_cast<uint8_t>(
            pair_routes[group.second][output] & ~group.shared[output]);
      }
      constexpr uint32_t second_register = 29;
      constexpr uint32_t combined_register = 30;
      load_quad_input(group.first, value_register);
      load_quad_input(group.second, second_register);
      assembler.vpxord(
          x86::zmm(combined_register),
          x86::zmm(value_register),
          x86::zmm(second_register));
      route_pair(group.shared, combined_register);
      route_pair(first_only, value_register);
      route_pair(second_only, second_register);
      continue;
    }
    if (batch_size == 3 && enable_triple_frf &&
        triple_input_group[input] != input_count) {
      const TripleFrfGroup& group =
          triple_groups[triple_input_group[input]];
      if (input != group.first) {
        continue;
      }
      TripleRoutes first_only{};
      TripleRoutes second_only{};
      for (size_t output = 0; output < 3; ++output) {
        first_only[output] = static_cast<uint8_t>(
            triple_routes[group.first][output] & ~group.shared[output]);
        second_only[output] = static_cast<uint8_t>(
            triple_routes[group.second][output] & ~group.shared[output]);
      }
      constexpr uint32_t second_register = 29;
      constexpr uint32_t combined_register = 30;
      load_quad_input(group.first, value_register);
      load_quad_input(group.second, second_register);
      assembler.vpxord(
          x86::zmm(combined_register),
          x86::zmm(value_register),
          x86::zmm(second_register));
      route_triple(group.shared, combined_register);
      route_triple(first_only, value_register);
      route_triple(second_only, second_register);
      continue;
    }
    if (quad && enable_quad_frf &&
        quad_input_group[input] != input_count) {
      const QuadFrfGroup& group =
          quad_groups[quad_input_group[input]];
      if (input != group.first) {
        continue;
      }
      QuadRoutes first_only{};
      QuadRoutes second_only{};
      for (size_t output = 0; output < 4; ++output) {
        first_only[output] = static_cast<uint8_t>(
            quad_routes[group.first][output] & ~group.shared[output]);
        second_only[output] = static_cast<uint8_t>(
            quad_routes[group.second][output] & ~group.shared[output]);
      }
      load_quad_input(group.first, value_register);
      route_quad(first_only, value_register, spill_register);
      load_quad_input(group.second, spill_register);
      assembler.vpxord(
          x86::zmm(value_register),
          x86::zmm(value_register),
          x86::zmm(spill_register));
      QuadRoutes register_shared = group.shared;
      if (group.shared_spill_logical >= 0) {
        const size_t output =
            static_cast<size_t>(group.shared_spill_logical) / 8;
        const size_t plane =
            static_cast<size_t>(group.shared_spill_logical) % 8;
        register_shared[output] &=
            static_cast<uint8_t>(~(uint8_t{1} << plane));
      }
      route_quad(register_shared, value_register, spill_register);
      if (group.shared_spill_logical >= 0) {
        const int32_t spill_offset = static_cast<int32_t>(
            spill_slots[static_cast<size_t>(group.shared_spill_logical)] * 64);
        assembler.vpxord(
            x86::zmm(value_register),
            x86::zmm(value_register),
            x86::ptr(x86::rsp, spill_offset));
        assembler.vmovdqu8(
            x86::ptr(x86::rsp, spill_offset),
            x86::zmm(value_register));
      }
      route_quad(second_only, spill_register, value_register);
      continue;
    }
    bool used = false;
    for (size_t output = 0; output < batch_size; ++output) {
      used |= coefficient_at(output, input) != 0;
    }
    if (!used) {
      continue;
    }
    if (pin_input_pointers) {
      assembler.vmovdqu8(
          x86::zmm(value_register),
          x86::ptr(pinned_input_ptrs[input], x86::rcx));
    } else {
      assembler.mov(
          x86::rax,
          x86::ptr(x86::rdi, static_cast<int32_t>(input * sizeof(void*))));
      assembler.vmovdqu8(
          x86::zmm(value_register), x86::ptr(x86::rax, x86::rcx));
    }
    if (batch_size == 2) {
      const uint8_t coefficient0 = coefficient_at(0, input);
      const uint8_t coefficient1 = coefficient_at(1, input);
      for (size_t plane = 0; plane < 8; ++plane) {
        const bool route0 = (coefficient0 & (1u << plane)) != 0;
        const bool route1 = (coefficient1 & (1u << plane)) != 0;
        if (route0 && route1) {
          const uint32_t logical =
              static_cast<uint32_t>(16 + plane);
          accumulate_value(logical, logical, value_register);
        } else if (route0) {
          const uint32_t logical =
              static_cast<uint32_t>(plane);
          accumulate_value(logical, logical, value_register);
        } else if (route1) {
          const uint32_t logical =
              static_cast<uint32_t>(8 + plane);
          accumulate_value(logical, logical, value_register);
        }
      }
      continue;
    }
    for (size_t output = 0; output < batch_size; ++output) {
      const uint8_t coefficient =
          coefficient_at(output, input);
      for (size_t plane = 0; plane < 8; ++plane) {
        if ((coefficient & (1u << plane)) != 0) {
          const uint32_t logical =
              static_cast<uint32_t>(output * 8 + plane);
          if (quad && spill_slots[logical] >= 0) {
            const int32_t spill_offset =
                static_cast<int32_t>(spill_slots[logical] * 64);
            assembler.vmovdqu8(
                x86::zmm(spill_register),
                x86::ptr(x86::rsp, spill_offset));
            assembler.vpxord(
                x86::zmm(spill_register),
                x86::zmm(spill_register),
                x86::zmm(value_register));
            assembler.vmovdqu8(
                x86::ptr(x86::rsp, spill_offset),
                x86::zmm(spill_register));
            continue;
          }
          const x86::Vec accumulator =
              x86::zmm(physical_register[logical]);
          if (quad) {
            assembler.vpxord(
                accumulator, accumulator, x86::zmm(value_register));
          } else {
            accumulate_value(
                logical, physical_register[logical], value_register);
          }
        }
      }
    }
  }
  if (batch_size == 2) {
    for (uint32_t plane = 0; plane < 8; ++plane) {
      if ((shared_plane_mask & (1u << plane)) == 0) {
        continue;
      }
      const x86::Vec shared = x86::zmm(16 + plane);
      for (uint32_t output = 0; output < 2; ++output) {
        const uint32_t logical = output * 8 + plane;
        const x86::Vec accumulator = x86::zmm(logical);
        if (accumulator_initialized[logical]) {
          assembler.vpxord(
              accumulator, accumulator, shared);
        } else {
          assembler.vmovdqa64(accumulator, shared);
          accumulator_initialized[logical] = true;
        }
      }
    }
  }

  if (quad) {
    for (uint32_t logical = 0; logical < physical_register.size(); ++logical) {
      if (spill_slots[logical] >= 0) {
        assembler.vmovdqu8(
            x86::zmm(physical_register[logical]),
            x86::ptr(
                x86::rsp,
                static_cast<int32_t>(spill_slots[logical] * 64)));
      }
    }
    if (use_gfni) {
      int current_plane = 31 - __builtin_clz(
          static_cast<unsigned>(used_plane_masks[3]));
      const uint32_t accumulator_register =
          physical_register[24 + static_cast<size_t>(current_plane)];
      for (int plane = current_plane - 1; plane >= 0; --plane) {
        if ((used_plane_masks[3] & (1u << plane)) == 0) {
          continue;
        }
        for (int step = current_plane; step > plane; --step) {
          assembler.vgf2p8affineqb(
              x86::zmm(accumulator_register),
              x86::zmm(accumulator_register),
              x86::ptr(x86::rsp, 128),
              0);
        }
        assembler.vpxord(
            x86::zmm(accumulator_register),
            x86::zmm(accumulator_register),
            x86::zmm(physical_register[24 + static_cast<size_t>(plane)]));
        current_plane = plane;
      }
      for (int step = current_plane; step > 0; --step) {
        assembler.vgf2p8affineqb(
            x86::zmm(accumulator_register),
            x86::zmm(accumulator_register),
            x86::ptr(x86::rsp, 128),
            0);
      }
      store_output(3, accumulator_register);
    } else {
      for (size_t plane = 0; plane < 8; ++plane) {
        assembler.vmovdqu8(
            x86::ptr(x86::rsp, static_cast<int32_t>(128 + plane * 64)),
            x86::zmm(physical_register[24 + plane]));
      }
    }
    zero_register = physical_register[24];
    polynomial_register = physical_register[25];
    high_bits_register = physical_register[26];
    gfni_matrix_register = physical_register[25];
    assembler.vpxord(
        x86::zmm(zero_register),
        x86::zmm(zero_register),
        x86::zmm(zero_register));
    if (use_gfni) {
      assembler.mov(
          x86::rax,
          core::BuildGfniAffineMatrix(2));
      assembler.vpbroadcastq(
          x86::zmm(gfni_matrix_register), x86::rax);
    } else {
      assembler.mov(x86::eax, 0x39393939u);
      assembler.vpbroadcastd(x86::zmm(polynomial_register), x86::eax);
    }
  }

  const auto advance_accumulator =
      [&](uint32_t accumulator_register) {
    if (use_gfni) {
      assembler.vgf2p8affineqb(
          x86::zmm(accumulator_register),
          x86::zmm(accumulator_register),
          x86::zmm(gfni_matrix_register),
          0);
    } else {
      EmitXtime(
          assembler,
          accumulator_register,
          zero_register,
          polynomial_register,
          high_bits_register);
    }
  };

  const auto reduce_and_store =
      [&](size_t output,
          const std::array<uint32_t, 8>& plane_registers,
          uint8_t used_mask) {
    if (used_mask == 0) {
      store_output(output, zero_register);
      return;
    }
    int current_plane =
        31 - __builtin_clz(static_cast<unsigned>(used_mask));
    const uint32_t accumulator_register =
        plane_registers[static_cast<size_t>(current_plane)];
    const x86::Vec accumulator = x86::zmm(accumulator_register);
    for (int plane = current_plane - 1; plane >= 0; --plane) {
      if ((used_mask & (1u << plane)) == 0) {
        continue;
      }
      for (int step = current_plane; step > plane; --step) {
        advance_accumulator(accumulator_register);
      }
      assembler.vpxord(
          accumulator,
          accumulator,
          x86::zmm(plane_registers[static_cast<size_t>(plane)]));
      current_plane = plane;
    }
    for (int step = current_plane; step > 0; --step) {
      advance_accumulator(accumulator_register);
    }
    store_output(output, accumulator_register);
  };

  const size_t resident_outputs = quad ? 3 : batch_size;
  bool interleaved_full_reduction =
      (!quad || use_quad_output_transform || BooleanEnvironmentOverride(
                    "BRAID_GFNI_QUAD_INTERLEAVED_REDUCTION", false)) &&
      resident_outputs >= 2;
  for (size_t output = 0; output < resident_outputs; ++output) {
    interleaved_full_reduction &=
        used_plane_masks[output] == 0xffu;
  }
  if (interleaved_full_reduction) {
    std::array<uint32_t, 3> accumulators{};
    for (size_t output = 0; output < resident_outputs; ++output) {
      accumulators[output] =
          physical_register[output * 8 + 7];
    }
    for (int plane = 6; plane >= 0; --plane) {
      for (size_t output = 0; output < resident_outputs; ++output) {
        advance_accumulator(accumulators[output]);
      }
      for (size_t output = 0; output < resident_outputs; ++output) {
        const x86::Vec accumulator =
            x86::zmm(accumulators[output]);
        assembler.vpxord(
            accumulator,
            accumulator,
            x86::zmm(
                physical_register[output * 8 +
                                  static_cast<size_t>(plane)]));
      }
    }
    if (use_quad_output_transform) {
      for (uint8_t index = 0;
           index < quad_output_transform.inverse_op_count;
           ++index) {
        const auto& operation = quad_output_transform.inverse_ops[index];
        if (operation.swap) {
          std::swap(
              accumulators[operation.destination],
              accumulators[operation.source]);
        } else {
          assembler.vpxord(
              x86::zmm(accumulators[operation.destination]),
              x86::zmm(accumulators[operation.destination]),
              x86::zmm(accumulators[operation.source]));
        }
      }
      for (size_t output = 0; output < 3; ++output) {
        store_output(output, accumulators[output]);
      }
    } else if (pair_transform != 0) {
      assembler.vpxord(
          x86::zmm(accumulators[1]),
          x86::zmm(accumulators[1]),
          x86::zmm(accumulators[0]));
      const uint32_t output0 =
          pair_transform == 1 ? accumulators[0] : accumulators[1];
      const uint32_t output1 =
          pair_transform == 1 ? accumulators[1] : accumulators[0];
      store_output(0, output0);
      store_output(1, output1);
    } else {
      for (size_t output = 0; output < resident_outputs; ++output) {
        store_output(output, accumulators[output]);
      }
    }
  } else {
    for (size_t output = 0; output < resident_outputs; ++output) {
      std::array<uint32_t, 8> plane_registers{};
      for (size_t plane = 0; plane < 8; ++plane) {
        plane_registers[plane] =
            physical_register[output * 8 + plane];
      }
      reduce_and_store(
          output, plane_registers, used_plane_masks[output]);
    }
    if (pair_transform != 0) {
      const uint32_t accumulator0 = physical_register[
          static_cast<size_t>(
              31 - __builtin_clz(
                       static_cast<unsigned>(used_plane_masks[0])))];
      const uint32_t accumulator1 = physical_register[
          8 + static_cast<size_t>(
                  31 - __builtin_clz(
                           static_cast<unsigned>(used_plane_masks[1])))];
      assembler.vpxord(
          x86::zmm(accumulator1),
          x86::zmm(accumulator1),
          x86::zmm(accumulator0));
      const uint32_t output0 =
          pair_transform == 1 ? accumulator0 : accumulator1;
      const uint32_t output1 =
          pair_transform == 1 ? accumulator1 : accumulator0;
      store_output(0, output0);
      store_output(1, output1);
    }
  }
  if (quad && !use_gfni) {
    std::array<uint32_t, 8> reload_registers{};
    for (size_t plane = 0; plane < 8; ++plane) {
      reload_registers[plane] = physical_register[plane];
      assembler.vmovdqu8(
          x86::zmm(reload_registers[plane]),
          x86::ptr(x86::rsp, static_cast<int32_t>(128 + plane * 64)));
    }
    reduce_and_store(
        3, reload_registers, used_plane_masks[3]);
  }
  assembler.add(x86::rcx, 64);
  assembler.cmp(x86::rcx, x86::rdx);
  assembler.jb(loop);
  assembler.bind(done);
  if (quad) {
    assembler.mov(x86::rsp, x86::rsi);
  }
  if (pin_input_pointers) {
    assembler.pop(x86::r15);
    assembler.pop(x86::r14);
    assembler.pop(x86::r13);
    assembler.pop(x86::r12);
    assembler.pop(x86::rbp);
    assembler.pop(x86::rbx);
  }
  assembler.ret();

  Avx512JitBatchFn batch_function = nullptr;
  Error error = runtime.add(&batch_function, &code);
  if (error != Error::kOk) {
    throw std::runtime_error("AsmJit failed to allocate executable BRAID-F code");
  }
  *allocation = reinterpret_cast<void*>(batch_function);
  return batch_function;
}

}  // namespace

Avx512PlanJit::~Avx512PlanJit() {
  for (void* allocation : allocations) {
    Runtime().release(allocation);
  }
}

bool CompileAvx512PlanJit(Plan* plan) {
  if (plan == nullptr || plan->backend != BRAID_BACKEND_AVX512 ||
      plan->path != BRAID_PATH_BRAID ||
      (plan->microkernel_mode != BRAID_MICROKERNEL_FACTOR_JIT &&
       plan->microkernel_mode != BRAID_MICROKERNEL_FACTOR_JIT_GFNI)) {
    return false;
  }
  const bool use_gfni =
      plan->microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI;
#if defined(__GNUC__) || defined(__clang__)
  if (use_gfni && !__builtin_cpu_supports("gfni")) {
    return false;
  }
#endif

  const size_t input_count =
      plan->rows.empty() ? 0 : plan->rows.front().coefficients.size();
  if (use_gfni) {
    const bool short_cached_working_set =
        plan->execution_threads == 1 && input_count <= 10 &&
        plan->planned_block_size <= (size_t{1} << 20);
    const bool default_streaming_stores =
        !short_cached_working_set &&
        (plan->streaming_stores ||
         plan->planned_block_size >= (size_t{512} << 10));
    plan->streaming_stores = BooleanEnvironmentOverride(
        "BRAID_GFNI_STREAMING_STORES", default_streaming_stores);
  }

  auto jit = std::make_shared<Avx512PlanJit>();
  size_t default_max_batch = 4;
  if (use_gfni) {
    if (input_count > 48) {
      default_max_batch =
          plan->type == PlanType::kEncode &&
                  (plan->rows.size() == 3 || plan->rows.size() == 6)
              ? 3
              : 2;
    } else if (plan->execution_threads == 1 &&
               plan->rows.size() == 4 && input_count <= 10 &&
               plan->planned_block_size <= (size_t{1} << 20)) {
      default_max_batch = 2;
    } else if (plan->type == PlanType::kEncode &&
               plan->rows.size() == 4 && input_count <= 20) {
      default_max_batch = 2;
    } else if (plan->rows.size() == 6 && input_count > 32) {
      default_max_batch = 3;
    }
  } else if (plan->type == PlanType::kEncode && plan->rows.size() == 6 &&
             input_count >= 40 && input_count <= 48) {
    default_max_batch = 3;
  } else if (input_count > 48) {
    default_max_batch = 2;
  }
  const size_t max_batch = use_gfni
                               ? std::clamp<size_t>(
                                     SizeEnvironmentOverride(
                                         "BRAID_GFNI_MAX_BATCH",
                                         default_max_batch),
                                     1,
                                     4)
                               : default_max_batch;
  const size_t function_count =
      (plan->rows.size() + max_batch - 1) / max_batch;
  const bool default_allow_triples =
      plan->rows.size() == 3 ||
      (plan->type == PlanType::kDecode &&
       plan->rows.size() == 2 &&
       input_count >= 64) ||
      (plan->type == PlanType::kEncode &&
       (plan->rows.size() == 4 || plan->rows.size() == 6));
  const bool allow_triples = BooleanEnvironmentOverride(
      "BRAID_GFNI_SLP_TRIPLES", default_allow_triples);
  const bool enable_route_fusion = BooleanEnvironmentOverride(
      "BRAID_GFNI_FRF", true);
  const size_t max_pair_distance = SizeEnvironmentOverride(
      "BRAID_GFNI_SLP_PAIR_DISTANCE", 0);
  const size_t prefetch_distance = SizeEnvironmentOverride(
      "BRAID_GFNI_PREFETCH_DISTANCE",
      use_gfni && input_count >= 80
          ? size_t{64}
          : (use_gfni && input_count >= 64 ? size_t{128} : size_t{0}));
  const uint32_t prefetch_hint = static_cast<uint32_t>(
      SizeEnvironmentOverride(
          "BRAID_GFNI_PREFETCH_HINT", 1));
  const size_t prefetch_input_stride = std::max<size_t>(
      1,
      SizeEnvironmentOverride(
          "BRAID_GFNI_PREFETCH_INPUT_STRIDE", 1));
  jit->prefetch_distance = prefetch_distance;
  jit->prefetch_hint = prefetch_hint;
  jit->prefetch_input_stride = prefetch_input_stride;
  std::vector<size_t> row_order(plan->rows.size());
  std::iota(row_order.begin(), row_order.end(), size_t{0});
  if (use_gfni && input_count > 48 &&
      (plan->rows.size() == 3 || plan->rows.size() == 4)) {
    row_order =
        plan->type == PlanType::kEncode && input_count >= 80 &&
                plan->rows.size() == 4
            ? std::vector<size_t>{0, 3, 1, 2}
            : SelectPairwiseRowOrder(plan->rows);
  }
  std::vector<OutputPlanRow> jit_rows(plan->rows.size());
  for (size_t row = 0; row < row_order.size(); ++row) {
    jit_rows[row] = plan->rows[row_order[row]];
  }
  jit->batch_functions.reserve(function_count);
  if (plan->streaming_stores) {
    jit->cached_store_batch_functions.reserve(function_count);
  }
  jit->batch_sizes.reserve(function_count);
  jit->batch_output_indices.reserve(function_count);
  jit->allocations.reserve(function_count);
  const bool use_register_dictionary =
      input_count > 48 ||
      (plan->type == PlanType::kEncode && plan->rows.size() == 6 &&
       input_count > 32 && input_count <= 48) ||
      BooleanEnvironmentOverride("BRAID_GFNI_REGISTER_DICTIONARY", false);
  for (size_t base_row = 0; base_row < plan->rows.size(); base_row += max_batch) {
    const size_t batch_size =
        std::min(max_batch, plan->rows.size() - base_row);
    const bool register_dictionary_path =
        use_gfni && use_register_dictionary && batch_size <= 3 &&
        plan->table.progression_order;
    RegisterDictionary prepared_dictionary;
    if (register_dictionary_path) {
      // NT and cached-store twins differ only in the final store opcode.
      // Reuse the exact same representation search for both kernels.
      prepared_dictionary = BuildRegisterDictionary(
          plan->table, jit_rows, base_row, batch_size);
    }
    const auto compile_batch =
        [&](bool streaming_stores,
            RegisterBatchStats* stats,
            void** allocation) {
          if (use_gfni && batch_size == 4 &&
              plan->table.progression_order &&
              BooleanEnvironmentOverride(
                  "BRAID_GFNI_QUAD_DICTIONARY", false)) {
            return CompileQuadRegisterConstrainedBatch(
                plan->table,
                jit_rows,
                base_row,
                streaming_stores,
                allocation);
          }
          if (register_dictionary_path) {
            return CompileRegisterConstrainedBatch(
                plan->table,
                jit_rows,
                base_row,
                batch_size,
                streaming_stores,
                enable_route_fusion &&
                    (input_count < 80 ||
                     (plan->type == PlanType::kDecode &&
                      plan->rows.size() >= 2 &&
                      plan->rows.size() <= 3) ||
                     (plan->type == PlanType::kEncode &&
                      plan->rows.size() == 4)),
                allow_triples,
                max_pair_distance,
                prefetch_distance,
                prefetch_hint,
                prefetch_input_stride,
                &prepared_dictionary,
                stats,
                allocation);
          }
          return CompilePolynomialBatch(
              plan->table,
              jit_rows,
              base_row,
              batch_size,
              streaming_stores,
              use_gfni,
              plan->execution_threads == 1,
              enable_route_fusion &&
                  BooleanEnvironmentOverride(
                      "BRAID_GFNI_PAIR_POLY_FRF",
                      plan->execution_threads == 1 && input_count <= 10 &&
                          plan->planned_block_size <= (size_t{1} << 20)),
              enable_route_fusion &&
                  BooleanEnvironmentOverride(
                      "BRAID_GFNI_QUAD_FRF",
                      input_count >= 24 && input_count <= 32),
              enable_route_fusion &&
                  BooleanEnvironmentOverride(
                      "BRAID_GFNI_TRIPLE_POLY_FRF",
                      input_count >= 24 && input_count <= 32),
              stats,
              allocation);
        };
    void* allocation = nullptr;
    RegisterBatchStats batch_stats;
    jit->batch_functions.push_back(
        compile_batch(plan->streaming_stores, &batch_stats, &allocation));
    if (batch_stats.xor_ops_before != 0) {
      jit->slp_xor_ops_before += batch_stats.xor_ops_before;
      jit->slp_xor_ops_after += batch_stats.xor_ops_after;
      jit->slp_potential_xor_ops += batch_stats.potential_xor_ops;
      jit->slp_groups += batch_stats.groups;
      jit->slp_potential_groups += batch_stats.potential_groups;
      jit->slp_triple_groups += batch_stats.triple_groups;
    }
    if (plan->streaming_stores) {
      void* cached_allocation = nullptr;
      jit->cached_store_batch_functions.push_back(
          compile_batch(false, nullptr, &cached_allocation));
      jit->allocations.push_back(cached_allocation);
    }
    jit->batch_sizes.push_back(static_cast<uint8_t>(batch_size));
    std::array<uint8_t, 4> output_indices{};
    for (size_t local = 0; local < batch_size; ++local) {
      output_indices[local] =
          static_cast<uint8_t>(row_order[base_row + local]);
    }
    jit->batch_output_indices.push_back(output_indices);
    jit->allocations.push_back(allocation);
  }
  plan->avx512_jit = std::move(jit);
  return true;
}

void ExecuteAvx512PlanJit(
    const Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  if (!plan.avx512_jit ||
      plan.avx512_jit->batch_functions.size() !=
          plan.avx512_jit->batch_sizes.size() ||
      plan.avx512_jit->batch_functions.size() !=
          plan.avx512_jit->batch_output_indices.size()) {
    throw std::runtime_error("BRAID-F AVX-512 JIT plan is unavailable");
  }

  bool use_streaming_stores = plan.streaming_stores;
  if (use_streaming_stores) {
    for (size_t output = 0; output < plan.rows.size(); ++output) {
      use_streaming_stores &=
          reinterpret_cast<uintptr_t>(outputs[output]) % 64 == 0;
    }
  }
  const auto& batch_functions =
      !use_streaming_stores &&
              plan.avx512_jit->cached_store_batch_functions.size() ==
                  plan.avx512_jit->batch_functions.size()
          ? plan.avx512_jit->cached_store_batch_functions
          : plan.avx512_jit->batch_functions;

  const size_t vector_bytes = block_size & ~size_t{63};
  const size_t function_count =
      plan.avx512_jit->batch_functions.size();
  if (function_count == 1) {
    std::array<uint8_t*, 4> function_outputs{};
    const size_t batch_size = plan.avx512_jit->batch_sizes.front();
    for (size_t local = 0; local < batch_size; ++local) {
      function_outputs[local] =
          outputs[plan.avx512_jit->batch_output_indices.front()[local]];
    }
    batch_functions.front()(
        inputs, function_outputs.data(), vector_bytes);
  } else {
    // Complete every output group while the same input tile is still in L2.
    // A function-major traversal rereads the entire input cell per group.
    const size_t input_count = plan.input_indices.size();
    const size_t default_reuse_tile =
        plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI
            ? (plan.execution_threads == 1 && input_count >= 40
                   ? size_t{384}
                   : (input_count >= 80
                   ? size_t{1} << 10
                   : (input_count >= 64
                          ? size_t{2} << 10
                          : (input_count >= 40
                                 ? size_t{4} << 10
                                 : size_t{8} << 10))))
            : (plan.tile_bytes == 0 ? size_t{4} << 10
                                    : plan.tile_bytes);
    const size_t requested_tile = SizeEnvironmentOverride(
        "BRAID_GFNI_REUSE_TILE",
        default_reuse_tile);
    const size_t cache_reuse_cap =
        plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI
            ? size_t{8} << 10
            : size_t{4} << 10;
    const size_t reuse_tile =
        std::max<size_t>(
            64,
            (std::min(requested_tile, cache_reuse_cap) / 64) *
                64);
    thread_local std::vector<const uint8_t*> tile_inputs;
    tile_inputs.resize(plan.input_indices.size());
    for (size_t offset = 0; offset < vector_bytes; offset += reuse_tile) {
      const size_t chunk =
          std::min(reuse_tile, vector_bytes - offset);
      for (size_t input = 0; input < tile_inputs.size(); ++input) {
        tile_inputs[input] = inputs[input] + offset;
      }
      size_t base_row = 0;
      for (size_t function = 0; function < function_count; ++function) {
        std::array<uint8_t*, 4> function_outputs{};
        const size_t batch_size =
            plan.avx512_jit->batch_sizes[function];
        for (size_t local = 0; local < batch_size; ++local) {
          const size_t output =
              plan.avx512_jit->batch_output_indices[function][local];
          function_outputs[local] = outputs[output] + offset;
        }
        batch_functions[function](
            tile_inputs.data(), function_outputs.data(), chunk);
        base_row += batch_size;
      }
      if (base_row != plan.rows.size()) {
        throw std::runtime_error(
            "BRAID-F AVX-512 JIT batch layout mismatch");
      }
    }
  }

  for (size_t row_index = 0; row_index < plan.rows.size(); ++row_index) {
    const auto& row = plan.rows[row_index];
    for (size_t pos = vector_bytes; pos < block_size; ++pos) {
      uint8_t value = 0;
      for (size_t input = 0; input < row.coefficients.size(); ++input) {
        value ^= core::ApplyCoefficientBraid(
            plan.table, row.coefficients[input], inputs[input][pos]);
      }
      outputs[row_index][pos] = value;
    }
  }
}

}  // namespace braid::runtime
