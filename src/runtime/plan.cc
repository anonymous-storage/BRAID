#include "src/runtime/plan.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "src/core/gf256.h"
#include "src/kernels/backend_dispatch.h"
#include "src/runtime/cpu_features.h"
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
#include "src/runtime/avx512_plan_jit.h"
#endif

namespace braid::runtime {

namespace {

template <typename CoefficientAccessor>
void PopulatePlanRows(
    const core::DecompositionTable& table,
    const std::vector<int>& output_indices,
    int input_count,
    CoefficientAccessor coefficient_accessor,
    Plan* plan) {
  plan->rows.resize(output_indices.size());
  for (size_t out = 0; out < output_indices.size(); ++out) {
    auto& row = plan->rows[out];
    row.coefficients.resize(input_count);
    row.alpha_masks.resize(input_count);
    row.alpha_weights.resize(input_count);
    row.alpha_positions.resize(input_count);
    for (int in = 0; in < input_count; ++in) {
      const uint8_t coefficient = coefficient_accessor(out, in);
      row.coefficients[in] = coefficient;
      row.alpha_masks[in] = table.alpha_masks[coefficient];
      row.alpha_weights[in] = table.weights[coefficient];
      row.alpha_positions[in] = table.alpha_positions[coefficient];
      for (uint8_t term = 0; term < row.alpha_weights[in]; ++term) {
        row.plane_input_indices[row.alpha_positions[in][term]].push_back(
            static_cast<uint16_t>(in));
      }
    }
  }
}

void PopulatePlanRowsFromCoefficientRows(
    const core::DecompositionTable& table,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    Plan* plan) {
  plan->rows.resize(coeff_rows.size());
  for (size_t out = 0; out < coeff_rows.size(); ++out) {
    auto& row = plan->rows[out];
    row.coefficients = coeff_rows[out];
    row.alpha_masks.resize(coeff_rows[out].size());
    row.alpha_weights.resize(coeff_rows[out].size());
    row.alpha_positions.resize(coeff_rows[out].size());
    for (size_t in = 0; in < coeff_rows[out].size(); ++in) {
      const uint8_t coefficient = coeff_rows[out][in];
      row.alpha_masks[in] = table.alpha_masks[coefficient];
      row.alpha_weights[in] = table.weights[coefficient];
      row.alpha_positions[in] = table.alpha_positions[coefficient];
      for (uint8_t term = 0; term < row.alpha_weights[in]; ++term) {
        row.plane_input_indices[row.alpha_positions[in][term]].push_back(
            static_cast<uint16_t>(in));
      }
    }
  }
}

using FactorBasis = std::array<uint8_t, 8>;

int BinaryRank(FactorBasis values, size_t count) {
  int rank = 0;
  for (int bit = 7; bit >= 0; --bit) {
    size_t pivot = static_cast<size_t>(rank);
    while (pivot < count && (values[pivot] & (1u << bit)) == 0) {
      ++pivot;
    }
    if (pivot == count) {
      continue;
    }
    std::swap(values[static_cast<size_t>(rank)], values[pivot]);
    for (size_t row = 0; row < count; ++row) {
      if (row != static_cast<size_t>(rank) && (values[row] & (1u << bit)) != 0) {
        values[row] ^= values[static_cast<size_t>(rank)];
      }
    }
    ++rank;
  }
  return rank;
}

bool BuildBasisCoordinates(
    const FactorBasis& basis,
    std::array<uint8_t, 256>* coordinates) {
  std::array<bool, 256> seen{};
  for (int mask = 0; mask < 256; ++mask) {
    uint8_t value = 0;
    for (size_t plane = 0; plane < basis.size(); ++plane) {
      if ((mask & (1 << plane)) != 0) {
        value ^= basis[plane];
      }
    }
    if (seen[value]) {
      return false;
    }
    seen[value] = true;
    (*coordinates)[value] = static_cast<uint8_t>(mask);
  }
  return true;
}

struct FactorBasisScore {
  uint32_t route_weight = std::numeric_limits<uint32_t>::max();
  uint8_t used_mask = 0xff;
  uint32_t reducer_cost = std::numeric_limits<uint32_t>::max();

  uint64_t Total() const {
    return static_cast<uint64_t>(route_weight) +
           static_cast<uint64_t>(reducer_cost);
  }
};

FactorBasisScore ScoreFactorBasis(
    const FactorBasis& basis,
    const std::vector<uint8_t>& coefficients,
    std::array<uint8_t, 256>* coordinates_out = nullptr) {
  std::array<uint8_t, 256> coordinates{};
  if (!BuildBasisCoordinates(basis, &coordinates)) {
    return {};
  }

  FactorBasisScore score;
  score.route_weight = 0;
  score.used_mask = 0;
  score.reducer_cost = 0;
  for (uint8_t coefficient : coefficients) {
    const uint8_t coordinate = coordinates[coefficient];
    score.route_weight += static_cast<uint32_t>(__builtin_popcount(coordinate));
    score.used_mask |= coordinate;
  }
  for (size_t plane = 0; plane < basis.size(); ++plane) {
    if ((score.used_mask & (1u << plane)) == 0) {
      continue;
    }
    // Identity is one XOR; a basis LUT multiply is roughly seven vector uops.
    score.reducer_cost += basis[plane] == 1 ? 1u : 7u;
  }
  if (coordinates_out != nullptr) {
    *coordinates_out = coordinates;
  }
  return score;
}

bool BetterFactorBasis(
    const FactorBasisScore& lhs,
    const FactorBasis& lhs_basis,
    const FactorBasisScore& rhs,
    const FactorBasis& rhs_basis) {
  if (lhs.Total() != rhs.Total()) {
    return lhs.Total() < rhs.Total();
  }
  if (lhs.route_weight != rhs.route_weight) {
    return lhs.route_weight < rhs.route_weight;
  }
  return lhs_basis < rhs_basis;
}

FactorBasis GreedyFactorBasis(
    const std::vector<uint8_t>& candidates,
    uint8_t forced_first) {
  FactorBasis basis{};
  size_t count = 0;
  const auto append = [&](uint8_t value, FactorBasis* selected, size_t* selected_count) {
    if (value == 0 || *selected_count >= selected->size()) {
      return;
    }
    FactorBasis trial = *selected;
    trial[*selected_count] = value;
    if (BinaryRank(trial, *selected_count + 1) == static_cast<int>(*selected_count + 1)) {
      (*selected)[(*selected_count)++] = value;
    }
  };

  append(forced_first, &basis, &count);
  for (uint8_t value : candidates) {
    append(value, &basis, &count);
  }
  for (int value = 1; value < 256 && count < basis.size(); ++value) {
    append(static_cast<uint8_t>(value), &basis, &count);
  }
  return basis;
}

FactorBasis OptimizeFactorBasis(const std::vector<uint8_t>& coefficients) {
  const FactorBasis canonical{1, 2, 4, 8, 16, 32, 64, 128};
  std::array<uint32_t, 256> frequency{};
  for (uint8_t coefficient : coefficients) {
    ++frequency[coefficient];
  }

  std::vector<uint8_t> candidates;
  candidates.reserve(255);
  for (int value = 1; value < 256; ++value) {
    candidates.push_back(static_cast<uint8_t>(value));
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](uint8_t lhs, uint8_t rhs) {
    if (frequency[lhs] != frequency[rhs]) {
      return frequency[lhs] > frequency[rhs];
    }
    return lhs < rhs;
  });

  std::vector<FactorBasis> seeds{canonical, GreedyFactorBasis(candidates, 0)};
  for (size_t seed = 0; seed < std::min<size_t>(16, candidates.size()); ++seed) {
    seeds.push_back(GreedyFactorBasis(candidates, candidates[seed]));
  }

  FactorBasis best = canonical;
  FactorBasisScore best_score = ScoreFactorBasis(best, coefficients);
  for (FactorBasis seed : seeds) {
    FactorBasisScore seed_score = ScoreFactorBasis(seed, coefficients);
    for (int pass = 0; pass < 8; ++pass) {
      FactorBasis pass_best = seed;
      FactorBasisScore pass_score = seed_score;
      for (size_t slot = 0; slot < seed.size(); ++slot) {
        for (int candidate = 1; candidate < 256; ++candidate) {
          FactorBasis trial = seed;
          trial[slot] = static_cast<uint8_t>(candidate);
          const FactorBasisScore trial_score = ScoreFactorBasis(trial, coefficients);
          if (BetterFactorBasis(trial_score, trial, pass_score, pass_best)) {
            pass_best = trial;
            pass_score = trial_score;
          }
        }
      }
      if (pass_best == seed) {
        break;
      }
      seed = pass_best;
      seed_score = pass_score;
    }
    if (BetterFactorBasis(seed_score, seed, best_score, best)) {
      best = seed;
      best_score = seed_score;
    }
  }
  return best;
}

FactorBasis OptimizeScaledPolynomialBasis(
    const std::vector<uint8_t>& coefficients,
    std::array<uint8_t, 256>* best_coordinates,
    FactorBasisScore* best_score_out) {
  FactorBasis best_basis{1, 2, 4, 8, 16, 32, 64, 128};
  FactorBasisScore best_score;
  std::array<uint8_t, 256> coordinates{};

  for (int scale_value = 1; scale_value < 256; ++scale_value) {
    const uint8_t scale = static_cast<uint8_t>(scale_value);
    FactorBasis basis{};
    for (size_t plane = 0; plane < basis.size(); ++plane) {
      basis[plane] = core::GF256::Multiply(
          scale, static_cast<uint8_t>(1u << plane));
    }

    FactorBasisScore score;
    score.route_weight = 0;
    score.used_mask = 0;
    score.reducer_cost = scale == 1 ? 0u : 7u;
    std::array<uint8_t, 256> trial_coordinates{};
    for (int coefficient = 0; coefficient < 256; ++coefficient) {
      trial_coordinates[static_cast<size_t>(coefficient)] =
          core::GF256::Divide(static_cast<uint8_t>(coefficient), scale);
    }
    for (uint8_t coefficient : coefficients) {
      const uint8_t coordinate = trial_coordinates[coefficient];
      score.route_weight += static_cast<uint32_t>(__builtin_popcount(coordinate));
      score.used_mask |= coordinate;
    }

    if (BetterFactorBasis(score, basis, best_score, best_basis)) {
      best_basis = basis;
      best_score = score;
      coordinates = trial_coordinates;
    }
  }

  *best_coordinates = coordinates;
  *best_score_out = best_score;
  return best_basis;
}

void CompileAdaptiveFactorBases(Plan* plan) {
  if (plan->path != BRAID_PATH_BRAID ||
      plan->backend != BRAID_BACKEND_AVX512 ||
      plan->microkernel_mode != BRAID_MICROKERNEL_FACTOR_BASIS_LUT) {
    return;
  }
  for (auto& row : plan->rows) {
    std::array<uint8_t, 256> coordinates{};
    FactorBasisScore score;
    row.basis_elements =
        OptimizeScaledPolynomialBasis(row.coefficients, &coordinates, &score);
    row.basis_used_mask = score.used_mask;
    row.basis_route_weight = score.route_weight;
    row.basis_coordinates.resize(row.coefficients.size());
    for (size_t input = 0; input < row.coefficients.size(); ++input) {
      row.basis_coordinates[input] = coordinates[row.coefficients[input]];
    }
  }
}

void ResolvePureBraidMicrokernel(Plan* plan) {
  if (plan->path != BRAID_PATH_BRAID ||
      plan->backend != BRAID_BACKEND_AVX512) {
    return;
  }
  if (plan->microkernel_mode == BRAID_MICROKERNEL_FACTOR_GFNI) {
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
    // The plan-specific JIT is block-size agnostic and avoids the scratch
    // traffic of the legacy scheduled path. Keep the latter available only
    // through its explicit ablation mode.
    plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_JIT_GFNI;
#else
    plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI;
#endif
    return;
  }
  if (plan->microkernel_mode != BRAID_MICROKERNEL_BASELINE) {
    return;
  }
  const size_t inputs = plan->input_indices.size();
  const size_t outputs = plan->rows.size();
  if (inputs <= 32 && plan->planned_block_size <= (size_t{256} << 10)) {
    if (outputs == 1) {
      plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_DIRECT;
    } else if (plan->type == PlanType::kEncode ||
               (outputs == 4 && inputs <= 24)) {
      plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_PLANE_STREAMING;
    } else {
      plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_POLYNOMIAL_FUSED;
    }
    return;
  }
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  if (inputs <= 48 && plan->planned_block_size <= (size_t{1} << 20)) {
    plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_JIT;
    return;
  }
#endif
  const bool short_chain = inputs <= 20;
  const bool small_mid_chain =
      inputs < 64 && plan->planned_block_size <= (size_t{1} << 20);
  if ((short_chain || small_mid_chain) && plan->output_batch >= 2) {
    plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_POLYNOMIAL_FUSED;
    return;
  }
  if (plan->output_batch != 1) {
    return;
  }
  plan->microkernel_mode = inputs >= 40
                               ? BRAID_MICROKERNEL_FACTOR_SCRATCH
                               : BRAID_MICROKERNEL_FACTOR_REGISTER;
}

struct OutputTransform {
  std::array<uint8_t, kFactorMaxBatch> rows{1, 2, 4, 8};
  std::array<uint8_t, kFactorMaxBatch> inverse{1, 2, 4, 8};
};

bool InvertBinaryTransform(
    const std::array<uint8_t, kFactorMaxBatch>& rows,
    size_t batch_size,
    std::array<uint8_t, kFactorMaxBatch>* inverse) {
  std::array<uint16_t, kFactorMaxBatch> augmented{};
  for (size_t row = 0; row < batch_size; ++row) {
    augmented[row] =
        static_cast<uint16_t>(rows[row]) |
        static_cast<uint16_t>(1u << (batch_size + row));
  }
  for (size_t column = 0; column < batch_size; ++column) {
    size_t pivot = column;
    while (pivot < batch_size &&
           (augmented[pivot] & (1u << column)) == 0) {
      ++pivot;
    }
    if (pivot == batch_size) {
      return false;
    }
    std::swap(augmented[column], augmented[pivot]);
    for (size_t row = 0; row < batch_size; ++row) {
      if (row != column &&
          (augmented[row] & (1u << column)) != 0) {
        augmented[row] ^= augmented[column];
      }
    }
  }
  inverse->fill(0);
  const uint16_t mask = static_cast<uint16_t>((1u << batch_size) - 1u);
  for (size_t row = 0; row < batch_size; ++row) {
    (*inverse)[row] = static_cast<uint8_t>(
        (augmented[row] >> batch_size) & mask);
  }
  return true;
}

uint8_t TransformCoefficient(
    const Plan& plan,
    size_t base_out,
    size_t input,
    uint8_t row_mask,
    size_t batch_size) {
  uint8_t coefficient = 0;
  for (size_t row = 0; row < batch_size; ++row) {
    if ((row_mask & (1u << row)) != 0) {
      coefficient ^= plan.rows[base_out + row].coefficients[input];
    }
  }
  return coefficient;
}

OutputTransform SelectOutputTransform(
    const Plan& plan,
    size_t base_out,
    size_t batch_size) {
  OutputTransform best;
  if (plan.microkernel_mode !=
          BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI ||
      batch_size < 3 || batch_size > kFactorMaxBatch) {
    return best;
  }

  uint64_t best_score = std::numeric_limits<uint64_t>::max();
  const uint32_t matrix_count =
      1u << static_cast<uint32_t>(batch_size * batch_size);
  const uint8_t row_mask =
      static_cast<uint8_t>((1u << batch_size) - 1u);
  for (uint32_t packed = 0; packed < matrix_count; ++packed) {
    OutputTransform candidate;
    for (size_t row = 0; row < batch_size; ++row) {
      candidate.rows[row] = static_cast<uint8_t>(
          (packed >> (row * batch_size)) & row_mask);
    }
    if (!InvertBinaryTransform(
            candidate.rows, batch_size, &candidate.inverse)) {
      continue;
    }

    uint64_t routes = 0;
    uint64_t planes = 0;
    for (size_t row = 0; row < batch_size; ++row) {
      uint32_t used_mask = 0;
      for (size_t input = 0;
           input < plan.input_indices.size();
           ++input) {
        const uint8_t coefficient = TransformCoefficient(
            plan,
            base_out,
            input,
            candidate.rows[row],
            batch_size);
        routes += plan.table.weights[coefficient];
        used_mask |= plan.table.alpha_masks[coefficient];
      }
      planes += static_cast<uint64_t>(
          __builtin_popcount(used_mask));
    }
    uint64_t reconstruction_xors = 0;
    for (size_t row = 0; row < batch_size; ++row) {
      reconstruction_xors += static_cast<uint64_t>(
          __builtin_popcount(candidate.inverse[row]) - 1);
    }
    const uint64_t score =
        routes * 1024u + planes * 8u + reconstruction_xors;
    if (score < best_score ||
        (score == best_score && candidate.rows < best.rows)) {
      best_score = score;
      best = candidate;
    }
  }
  return best;
}

void CompilePureFactorSchedules(Plan* plan) {
  plan->factor_schedules.clear();
  if (plan->path != BRAID_PATH_BRAID || plan->backend != BRAID_BACKEND_AVX512 ||
      plan->rows.empty() || plan->input_indices.size() <= 20) {
    return;
  }

  const size_t batch = static_cast<size_t>(std::max(plan->output_batch, 1));
  if (batch > kFactorMaxBatch) {
    return;
  }
  plan->factor_schedules.resize(plan->rows.size());
  for (size_t base_out = 0; base_out < plan->rows.size(); base_out += batch) {
    const size_t batch_size = std::min(batch, plan->rows.size() - base_out);
    auto& schedule = plan->factor_schedules[base_out];
    schedule.batch_size = static_cast<uint8_t>(batch_size);
    const OutputTransform transform =
        SelectOutputTransform(*plan, base_out, batch_size);
    schedule.reconstruction_masks = transform.inverse;
    schedule.target_lists.resize(plan->input_indices.size());
    std::array<bool, kFactorMaxBatch * kFactorPlaneCount> initialized{};

    for (size_t input = 0; input < plan->input_indices.size(); ++input) {
      auto& targets = schedule.target_lists[input];
      for (size_t output = 0; output < batch_size; ++output) {
        const uint8_t coefficient = TransformCoefficient(
            *plan,
            base_out,
            input,
            transform.rows[output],
            batch_size);
        const uint8_t weight = plan->table.weights[coefficient];
        for (uint8_t term = 0; term < weight; ++term) {
          const uint8_t plane =
              plan->table.alpha_positions[coefficient][term];
          const uint8_t id = static_cast<uint8_t>(output * kFactorPlaneCount + plane);
          const uint8_t local = targets.count++;
          targets.ids[local] = id;
          if (!initialized[id]) {
            targets.initialize_mask |= static_cast<uint16_t>(1u << local);
            initialized[id] = true;
          }
          schedule.used_plane_masks[output] |= (1u << plane);
          schedule.max_planes[output] = std::max(schedule.max_planes[output], plane);
        }
      }
    }

    for (size_t output = 0; output < batch_size; ++output) {
      uint32_t mask = schedule.used_plane_masks[output];
      while (mask != 0) {
        schedule.used_planes[output][schedule.used_plane_counts[output]++] =
            static_cast<uint8_t>(__builtin_ctz(mask));
        mask &= (mask - 1);
      }
    }

  }
}

void CompileGroupedFactorSchedules(Plan* plan) {
  plan->factor_grouped_schedules.clear();
  if (plan->path != BRAID_PATH_BRAID ||
      plan->backend != BRAID_BACKEND_AVX512 ||
      plan->rows.empty()) {
    return;
  }

  const size_t batch =
      static_cast<size_t>(std::max(plan->output_batch, 1));
  if (batch > kFactorMaxBatch) {
    return;
  }
  plan->factor_grouped_schedules.resize(plan->rows.size());
  for (size_t base_out = 0; base_out < plan->rows.size();
       base_out += batch) {
    const size_t batch_size =
        std::min(batch, plan->rows.size() - base_out);
    auto& schedule = plan->factor_grouped_schedules[base_out];
    schedule.batch_size = static_cast<uint8_t>(batch_size);
    schedule.target_lists.resize(plan->input_indices.size());
    std::array<int16_t, kFactorPlaneCount * 16> group_indices{};
    group_indices.fill(-1);
    std::array<bool, kFactorPlaneCount * 16> initialized{};

    for (size_t input = 0; input < plan->input_indices.size(); ++input) {
      std::array<uint8_t, kFactorPlaneCount> output_masks{};
      for (size_t output = 0; output < batch_size; ++output) {
        const auto& row = plan->rows[base_out + output];
        const uint8_t weight = row.alpha_weights[input];
        for (uint8_t term = 0; term < weight; ++term) {
          output_masks[row.alpha_positions[input][term]] |=
              static_cast<uint8_t>(1u << output);
        }
      }

      auto& targets = schedule.target_lists[input];
      for (size_t plane = 0; plane < output_masks.size(); ++plane) {
        const uint8_t output_mask = output_masks[plane];
        if (output_mask == 0) {
          continue;
        }
        const size_t key = plane * 16 + output_mask;
        int16_t group_index = group_indices[key];
        if (group_index < 0) {
          group_index = static_cast<int16_t>(schedule.groups.size());
          group_indices[key] = group_index;
          schedule.groups.push_back(
              FactorGroup{static_cast<uint8_t>(plane), output_mask});
          for (size_t output = 0; output < batch_size; ++output) {
            if ((output_mask & (1u << output)) != 0) {
              schedule.reduction_groups[output][plane].push_back(
                  static_cast<uint8_t>(group_index));
            }
          }
        }
        const uint8_t local = targets.count++;
        targets.ids[local] = static_cast<uint8_t>(group_index);
        if (!initialized[key]) {
          targets.initialize_mask |= static_cast<uint16_t>(1u << local);
          initialized[key] = true;
        }
      }
    }
  }
}

void FinalizePureBraidPlan(Plan* plan) {
  plan->gfni_object = plan->backend == BRAID_BACKEND_AVX512 && GfniAvailable();
  ResolvePureBraidMicrokernel(plan);
  CompileAdaptiveFactorBases(plan);
  CompilePureFactorSchedules(plan);
  CompileGroupedFactorSchedules(plan);
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  CompileAvx512PlanJit(plan);
#endif
}

bool ShouldAutotuneRepresentation(const Context& context, const Plan& plan) {
  return context.options().repr_mode == BRAID_REPR_RETUNE && plan.path == BRAID_PATH_BRAID &&
         (plan.backend == BRAID_BACKEND_AVX2 || plan.backend == BRAID_BACKEND_AVX512);
}

bool SameTableOrder(const core::DecompositionTable& lhs, const core::DecompositionTable& rhs) {
  return lhs.x_powers == rhs.x_powers;
}

bool HasExactOrder17(uint8_t value) {
  if (value == 0 || value == 1) {
    return false;
  }
  if (core::GF256::Pow(value, 17) != 1) {
    return false;
  }
  return core::GF256::Pow(value, 1) != 1;
}

double AdvanceCostByExponent(uint8_t x_exp) {
  if (x_exp == 0) {
    return 0.0;
  }
  return x_exp <= 4 ? static_cast<double>(x_exp) : 4.5;
}

double TransitionCost(const core::DecompositionTable& table, uint8_t from_pos, uint8_t to_pos) {
  return AdvanceCostByExponent(table.transition_x_exponents[from_pos][to_pos]);
}

struct RowKernelStats {
  double weight_sum = 0.0;
  double used_plane_count = 0.0;
  double max_pos = 0.0;
  double sparse_reduction_cost = 0.0;
  double dense_reduction_cost = 0.0;
  double direct_visit_cost = 0.0;
};

double DirectVisitCost(const core::DecompositionTable& table, uint8_t coefficient) {
  const uint8_t weight = table.weights[coefficient];
  if (weight == 0) {
    return 0.0;
  }
  std::array<uint8_t, 3> positions{};
  for (uint8_t idx = 0; idx < weight; ++idx) {
    positions[idx] = table.alpha_positions[coefficient][idx];
  }
  double best = std::numeric_limits<double>::infinity();
  std::sort(positions.begin(), positions.begin() + weight);
  do {
    double cost = AdvanceCostByExponent(table.x_exponents[positions[0]]);
    for (uint8_t idx = 1; idx < weight; ++idx) {
      cost += TransitionCost(table, positions[idx - 1], positions[idx]);
    }
    best = std::min(best, cost);
  } while (std::next_permutation(positions.begin(), positions.begin() + weight));
  return best + static_cast<double>(weight) * 0.20;
}

RowKernelStats ComputeRowKernelStats(
    const core::DecompositionTable& table, const std::vector<uint8_t>& coefficients) {
  RowKernelStats stats;
  uint32_t used_mask = 0;
  for (uint8_t coefficient : coefficients) {
    const uint8_t weight = table.weights[coefficient];
    stats.weight_sum += weight;
    used_mask |= table.alpha_masks[coefficient];
    stats.direct_visit_cost += DirectVisitCost(table, coefficient);
  }
  stats.used_plane_count = static_cast<double>(__builtin_popcount(used_mask));
  if (used_mask == 0) {
    return stats;
  }

  const uint8_t highest = static_cast<uint8_t>(31 - __builtin_clz(used_mask));
  stats.max_pos = highest;
  stats.dense_reduction_cost = TransitionCost(table, highest, 0);
  for (uint8_t plane = highest; plane > 0; --plane) {
    stats.dense_reduction_cost += TransitionCost(table, plane, static_cast<uint8_t>(plane - 1));
  }

  uint32_t remaining = used_mask;
  uint8_t current = highest;
  remaining &= ~(1u << highest);
  while (remaining != 0) {
    const uint8_t next = static_cast<uint8_t>(31 - __builtin_clz(remaining));
    stats.sparse_reduction_cost += TransitionCost(table, current, next);
    current = next;
    remaining &= ~(1u << next);
  }
  stats.sparse_reduction_cost += TransitionCost(table, current, 0);
  return stats;
}

double ScoreTableForPlan(
    const Context& context,
    const Plan& plan_template,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    size_t block_size,
    const core::DecompositionTable& table) {
  if (coeff_rows.empty()) {
    return static_cast<double>(table.common_factor_nnz);
  }

  std::vector<RowKernelStats> row_stats(coeff_rows.size());
  for (size_t row = 0; row < coeff_rows.size(); ++row) {
    row_stats[row] = ComputeRowKernelStats(table, coeff_rows[row]);
  }

  double weight_sum = 0.0;
  double used_plane_sum = 0.0;
  double sparse_reduction_sum = 0.0;
  double dense_reduction_sum = 0.0;
  double max_pos_sum = 0.0;
  double direct_visit_sum = 0.0;
  for (const auto& stats : row_stats) {
    weight_sum += stats.weight_sum;
    used_plane_sum += stats.used_plane_count;
    sparse_reduction_sum += stats.sparse_reduction_cost;
    dense_reduction_sum += stats.dense_reduction_cost;
    max_pos_sum += stats.max_pos;
    direct_visit_sum += stats.direct_visit_cost;
  }

  double score = 0.0;
  const bool large_block = block_size >= (4u << 20);
  const bool many_threads = context.options().num_threads >= 16;
  const bool big_problem = plan_template.input_indices.size() >= 64;

  double weight_factor = 820.0;
  double used_factor = 360.0;
  double sparse_reduce_factor = 220.0;
  double dense_reduce_factor = 200.0;
  double max_pos_factor = 65.0;

  if (large_block) {
    weight_factor *= 1.12;
    used_factor *= 1.18;
  }
  if (many_threads) {
    weight_factor *= 1.08;
    used_factor *= 1.10;
  }
  if (big_problem) {
    weight_factor *= 1.10;
    used_factor *= 1.12;
  }

  if (plan_template.type == PlanType::kEncode) {
    if (plan_template.backend == BRAID_BACKEND_AVX512 && plan_template.output_batch >= 4 &&
        plan_template.input_indices.size() > 16) {
      // k=64,r=4 and similar paths run as 2+2 fixed-batch kernels. Model scratch traffic
      // and sparse per-output reduction directly.
      score += weight_sum * weight_factor;
      score += used_plane_sum * used_factor;
      score += sparse_reduction_sum * sparse_reduce_factor;
      score += max_pos_sum * max_pos_factor;
    } else if (plan_template.backend == BRAID_BACKEND_AVX512 && plan_template.output_batch == 3 &&
               plan_template.input_indices.size() > 16) {
      score += weight_sum * (weight_factor * 0.95);
      score += used_plane_sum * (used_factor * 0.95);
      score += dense_reduction_sum * dense_reduce_factor;
      score += max_pos_sum * (max_pos_factor * 0.85);
    } else {
      score += weight_sum * (weight_factor * 0.90);
      score += used_plane_sum * (used_factor * 0.90);
      score += sparse_reduction_sum * (sparse_reduce_factor * 0.90);
      score += max_pos_sum * (max_pos_factor * 0.80);
    }
  } else {
    if (plan_template.output_batch == 1) {
      score += direct_visit_sum * (weight_factor * 0.58);
      score += weight_sum * (weight_factor * 0.22);
      score += max_pos_sum * (max_pos_factor * 0.35);
      score += used_plane_sum * (used_factor * 0.12);
    } else if (plan_template.output_batch <= 2) {
      // Decode on AVX512 now uses sparse interleaved reduction for batch=2 and sparse
      // reduction for batch=1. Favor tables that reduce target writes and sparse steps.
      score += weight_sum * (weight_factor * 0.92);
      score += used_plane_sum * (used_factor * 0.96);
      score += sparse_reduction_sum * (sparse_reduce_factor * 1.08);
      score += max_pos_sum * (max_pos_factor * 0.75);
    } else {
      score += weight_sum * weight_factor;
      score += used_plane_sum * used_factor;
      score += dense_reduction_sum * dense_reduce_factor;
      score += max_pos_sum * max_pos_factor;
    }
  }

  score += static_cast<double>(table.common_factor_nnz) * 2.0;
  if (!table.progression_order && plan_template.input_indices.size() <= 16) {
    score += 1.0e9;
  }
  return score;
}

core::DecompositionTable LocalOrderSearch(
    const Context& context,
    const Plan& plan_template,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    size_t block_size,
    const core::DecompositionTable& seed,
    double* best_score_out) {
  core::DecompositionTable best = seed;
  double best_score = ScoreTableForPlan(context, plan_template, coeff_rows, block_size, best);
  bool improved = true;
  int rounds = 0;
  while (improved && rounds < 1) {
    improved = false;
    ++rounds;
    std::array<uint8_t, 17> order = best.x_powers;
    for (int i = 1; i < 16; ++i) {
      for (int j = i + 1; j < 17; ++j) {
        auto candidate_order = order;
        std::swap(candidate_order[static_cast<size_t>(i)], candidate_order[static_cast<size_t>(j)]);
        auto candidate = core::BuildDecompositionTable(candidate_order);
        if (plan_template.input_indices.size() <= 16 && !candidate.progression_order) {
          continue;
        }
        const double score = ScoreTableForPlan(context, plan_template, coeff_rows, block_size, candidate);
        if (score < best_score ||
            (score == best_score &&
             (candidate.common_factor_nnz < best.common_factor_nnz ||
              (candidate.common_factor_nnz == best.common_factor_nnz &&
               candidate.x_exponents < best.x_exponents)))) {
          best_score = score;
          best = std::move(candidate);
          order = best.x_powers;
          improved = true;
        }
      }
    }
  }
  *best_score_out = best_score;
  return best;
}

void AppendCandidateTable(
    std::vector<core::DecompositionTable>* candidates, const core::DecompositionTable& table) {
  for (const auto& existing : *candidates) {
    if (SameTableOrder(existing, table)) {
      return;
    }
  }
  candidates->push_back(table);
}

std::vector<uint8_t> ExactOrder17Generators() {
  std::vector<uint8_t> generators;
  for (int candidate = 2; candidate < 256; ++candidate) {
    const uint8_t generator = static_cast<uint8_t>(candidate);
    if (HasExactOrder17(generator)) {
      generators.push_back(generator);
    }
  }
  return generators;
}

std::vector<int> CandidateBatches(const Plan& plan_template) {
  std::vector<int> batches;
  const int outputs = static_cast<int>(
      plan_template.rows.empty() ? plan_template.output_indices.size() : plan_template.rows.size());
  const auto add_batch = [&](int batch) {
    if (batch <= 0 || batch > outputs) {
      return;
    }
    if (std::find(batches.begin(), batches.end(), batch) == batches.end()) {
      batches.push_back(batch);
    }
  };
  add_batch(plan_template.output_batch);
  add_batch(outputs);
  add_batch(1);
  if (outputs >= 2) {
    add_batch(2);
  }
  if (outputs >= 3) {
    add_batch(3);
  }
  if (outputs >= 4) {
    add_batch(4);
  }
  return batches;
}

std::vector<size_t> CandidateTiles(size_t block_size) {
  std::vector<size_t> tiles{16u << 10, 32u << 10, 64u << 10};
  if (block_size <= (1u << 20)) {
    tiles.push_back(4u << 10);
  }
  if (block_size < (4u << 20)) {
    tiles.push_back(8u << 10);
  }
  for (auto& tile : tiles) {
    tile = std::min(tile, block_size);
  }
  std::sort(tiles.begin(), tiles.end());
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  return tiles;
}

double BenchmarkPlanVariant(
    const Context& context,
    const Plan& template_plan,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    const core::DecompositionTable& table,
    size_t tile_bytes,
    int output_batch,
    size_t sample_block_size) {
  Plan trial = template_plan;
  trial.table = table;
  trial.tile_bytes = tile_bytes;
  trial.output_batch = output_batch;
  PopulatePlanRowsFromCoefficientRows(table, coeff_rows, &trial);

  std::vector<std::vector<uint8_t>> input_storage(
      trial.input_indices.size(), std::vector<uint8_t>(sample_block_size));
  std::vector<std::vector<uint8_t>> output_storage(
      trial.rows.size(), std::vector<uint8_t>(sample_block_size));
  std::vector<const uint8_t*> input_ptrs(trial.input_indices.size());
  std::vector<uint8_t*> output_ptrs(trial.rows.size());

  for (size_t in = 0; in < input_storage.size(); ++in) {
    uint32_t seed = static_cast<uint32_t>(0x9e3779b9u + in * 2654435761u);
    for (size_t pos = 0; pos < sample_block_size; ++pos) {
      seed = seed * 1664525u + 1013904223u;
      input_storage[in][pos] = static_cast<uint8_t>(seed >> 24);
    }
    input_ptrs[in] = input_storage[in].data();
  }
  for (size_t out = 0; out < output_storage.size(); ++out) {
    std::fill(output_storage[out].begin(), output_storage[out].end(), 0);
    output_ptrs[out] = output_storage[out].data();
  }

  double seconds_sum = 0.0;
  for (int round = 0; round < 3; ++round) {
    for (size_t out = 0; out < output_storage.size(); ++out) {
      std::fill(output_storage[out].begin(), output_storage[out].end(), 0);
    }
    const auto start = std::chrono::steady_clock::now();
    kernels::ExecutePlan(
        trial.table,
        trial,
        input_ptrs.data(),
        output_ptrs.data(),
        sample_block_size,
        context.options().num_threads,
        context.thread_pool());
    const auto end = std::chrono::steady_clock::now();
    if (round != 0) {
      seconds_sum += std::chrono::duration<double>(end - start).count();
    }
  }
  return seconds_sum / 2.0;
}

core::DecompositionTable MaybeEmpiricallyTunePlan(
    const Context& context,
    const Plan& plan_template,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    size_t block_size,
    size_t* tuned_tile_out,
    int* tuned_batch_out) {
  *tuned_tile_out = plan_template.tile_bytes;
  *tuned_batch_out = plan_template.output_batch;
  const bool avx512_retune =
      context.options().repr_mode == BRAID_REPR_RETUNE &&
      plan_template.path == BRAID_PATH_BRAID &&
      plan_template.backend == BRAID_BACKEND_AVX512;
  if (!avx512_retune) {
    return plan_template.table;
  }
  if (block_size == 0) {
    return plan_template.table;
  }
  if (avx512_retune &&
      (block_size < (1u << 20) || plan_template.input_indices.size() > 20 || coeff_rows.size() > 6)) {
    return plan_template.table;
  }

  const size_t sample_block_size = std::min<size_t>(block_size, 1u << 20);
  const std::vector<int> batches = CandidateBatches(plan_template);
  const std::vector<size_t> tiles = CandidateTiles(block_size);

  std::vector<core::DecompositionTable> candidate_tables;
  const std::vector<uint8_t> generators = ExactOrder17Generators();
  for (uint8_t generator : generators) {
    AppendCandidateTable(&candidate_tables, core::BuildDecompositionTable(generator));
  }

  core::DecompositionTable best_table = plan_template.table;
  size_t best_tile = plan_template.tile_bytes;
  int best_batch = plan_template.output_batch;
  double best_seconds = std::numeric_limits<double>::infinity();

  for (const auto& table : candidate_tables) {
    for (int batch : batches) {
      for (size_t tile : tiles) {
        const double seconds =
            BenchmarkPlanVariant(context, plan_template, coeff_rows, table, tile, batch, sample_block_size);
        if (seconds < best_seconds) {
          best_seconds = seconds;
          best_table = table;
          best_tile = tile;
          best_batch = batch;
        }
      }
    }
  }

  *tuned_tile_out = best_tile;
  *tuned_batch_out = best_batch;
  return best_table;
}

core::DecompositionTable MaybeAutotuneTable(
    const Context& context,
    const Plan& plan_template,
    const std::vector<std::vector<uint8_t>>& coeff_rows,
    size_t block_size) {
  const auto maybe_local_search = [&](core::DecompositionTable seed) {
    if (plan_template.type == PlanType::kEncode && plan_template.input_indices.size() > 16) {
      double score = 0.0;
      return LocalOrderSearch(context, plan_template, coeff_rows, block_size, std::move(seed), &score);
    }
    return seed;
  };
  if (context.options().repr_generator > 0 && context.options().repr_generator < 256) {
    return core::BuildDecompositionTable(static_cast<uint8_t>(context.options().repr_generator));
  }
  const bool force_wide_encode_search =
      plan_template.path == BRAID_PATH_BRAID &&
      (plan_template.backend == BRAID_BACKEND_AVX512 || plan_template.backend == BRAID_BACKEND_AVX2) &&
      plan_template.type == PlanType::kEncode &&
      plan_template.input_indices.size() >= 64 &&
      static_cast<int>(coeff_rows.size()) >= 4;

  if (!ShouldAutotuneRepresentation(context, plan_template) && !force_wide_encode_search) {
    return context.SelectTableForRows(coeff_rows);
  }

  if (plan_template.type == PlanType::kEncode && plan_template.input_indices.size() >= 48 &&
      static_cast<int>(coeff_rows.size()) >= 4) {
    if (plan_template.backend == BRAID_BACKEND_AVX512) {
      if (block_size <= (1u << 20)) {
        return core::BuildDecompositionTable(
            plan_template.input_indices.size() >= 80 ? 0x39 : 0x04);
      }
      return core::BuildDecompositionTable(0x02);
    }
    if (plan_template.backend == BRAID_BACKEND_AVX2) {
      if (block_size <= (512u << 10)) {
        return core::BuildDecompositionTable(0x04);
      }
      if (block_size <= (1u << 20)) {
        return core::BuildDecompositionTable(0x9c);
      }
      if (block_size <= (4u << 20)) {
        return core::BuildDecompositionTable(0x02);
      }
      if (block_size < (64u << 20)) {
        return core::BuildDecompositionTable(plan_template.input_indices.size() >= 96 ? 0x02 : 0x39);
      }
      if (block_size < (128u << 20)) {
        return core::BuildDecompositionTable(plan_template.input_indices.size() >= 96 ? 0x02 : 0x04);
      }
      return core::BuildDecompositionTable(plan_template.input_indices.size() >= 96 ? 0x02 : 0x39);
    }
  }

  const core::DecompositionTable static_best = context.SelectTableForRows(coeff_rows);
  if (plan_template.backend != BRAID_BACKEND_AVX512) {
    return static_best;
  }

  std::vector<core::DecompositionTable> candidates;
  AppendCandidateTable(&candidates, core::BuildDecompositionTable(0x02));
  AppendCandidateTable(&candidates, core::BuildDecompositionTable(0x04));
  AppendCandidateTable(&candidates, core::BuildDecompositionTable(0x39));
  AppendCandidateTable(&candidates, core::BuildDecompositionTable(0x9c));
  AppendCandidateTable(&candidates, static_best);

  const double static_score = ScoreTableForPlan(context, plan_template, coeff_rows, block_size, static_best);
  for (int candidate = 2; candidate < 256; ++candidate) {
    const uint8_t generator = static_cast<uint8_t>(candidate);
    if (!HasExactOrder17(generator)) {
      continue;
    }
    const auto table = core::BuildDecompositionTable(generator);
    const double score = ScoreTableForPlan(context, plan_template, coeff_rows, block_size, table);
    const double slack =
        plan_template.input_indices.size() <= 16 ? 1.35 :
        (plan_template.type == PlanType::kDecode ? 1.18 : 1.08);
    if (score <= static_score * slack) {
      AppendCandidateTable(&candidates, table);
    }
  }

  core::DecompositionTable best = candidates.front();
  double best_score = std::numeric_limits<double>::infinity();
  for (const auto& seed : candidates) {
    double candidate_score = 0.0;
    core::DecompositionTable candidate = seed;
    if (plan_template.type == PlanType::kEncode && plan_template.input_indices.size() > 16) {
      candidate = LocalOrderSearch(context, plan_template, coeff_rows, block_size, seed, &candidate_score);
    } else {
      candidate_score = ScoreTableForPlan(context, plan_template, coeff_rows, block_size, candidate);
    }
    if (candidate_score < best_score ||
        (candidate_score == best_score &&
         (candidate.common_factor_nnz < best.common_factor_nnz ||
          (candidate.common_factor_nnz == best.common_factor_nnz && candidate.x_exponents < best.x_exponents)))) {
      best_score = candidate_score;
      best = std::move(candidate);
    }
  }
  return best;
}

}  // namespace

std::shared_ptr<Plan> BuildEncodePlan(Context& context, size_t block_size) {
  auto plan = std::make_shared<Plan>();
  plan->type = PlanType::kEncode;
  plan->backend = context.ResolveBackend();
  plan->path = context.ResolvePath(plan->type, context.r(), block_size, plan->backend);
  plan->microkernel_mode = context.options().microkernel_mode;
  plan->planned_block_size = block_size;
  plan->execution_threads = context.options().num_threads;
  const auto [tile, batch] =
      context.ChooseTileAndBatch(plan->type, context.k(), context.r(), block_size, plan->backend, plan->path);
  plan->tile_bytes = tile;
  plan->output_batch = batch;
  std::vector<std::vector<uint8_t>> coeff_rows(static_cast<size_t>(context.r()));
  for (int row = 0; row < context.r(); ++row) {
    coeff_rows[static_cast<size_t>(row)].assign(
        context.generator()[static_cast<size_t>(context.k() + row)].begin(),
        context.generator()[static_cast<size_t>(context.k() + row)].begin() + context.k());
  }
  plan->input_indices.resize(context.k());
  std::iota(plan->input_indices.begin(), plan->input_indices.end(), 0);
  plan->output_indices.resize(context.r());
  std::iota(plan->output_indices.begin(), plan->output_indices.end(), context.k());
  plan->table = MaybeAutotuneTable(context, *plan, coeff_rows, block_size);
  plan->table =
      MaybeEmpiricallyTunePlan(context, *plan, coeff_rows, block_size, &plan->tile_bytes, &plan->output_batch);
  PopulatePlanRows(
      plan->table,
      plan->output_indices,
      context.k(),
      [&](size_t out, int in) -> uint8_t { return context.generator()[plan->output_indices[out]][in]; },
      plan.get());
  FinalizePureBraidPlan(plan.get());
  return plan;
}

std::shared_ptr<Plan> BuildDecodePlan(Context& context, const std::vector<int>& erased_indices, size_t block_size) {
  if (erased_indices.empty() || erased_indices.size() > static_cast<size_t>(context.r())) {
    throw std::invalid_argument("invalid erased count");
  }

  std::unordered_set<int> erased;
  for (int index : erased_indices) {
    if (index < 0 || index >= context.n() || !erased.insert(index).second) {
      throw std::invalid_argument("erased indices must be unique and in range");
    }
  }
  std::vector<int> survivors;
  survivors.reserve(context.k());
  for (int idx = 0; idx < context.n(); ++idx) {
    if (erased.count(idx) == 0) {
      survivors.push_back(idx);
    }
    if (survivors.size() == static_cast<size_t>(context.k())) {
      break;
    }
  }
  if (survivors.size() != static_cast<size_t>(context.k())) {
    throw std::runtime_error("not enough survivors");
  }

  auto plan = std::make_shared<Plan>();
  plan->type = PlanType::kDecode;
  plan->backend = context.ResolveBackend();
  plan->path = context.ResolvePath(
      plan->type, static_cast<int>(erased_indices.size()), block_size, plan->backend);
  plan->microkernel_mode = context.options().microkernel_mode;
  plan->planned_block_size = block_size;
  plan->execution_threads = context.options().num_threads;
  const auto [tile, batch] = context.ChooseTileAndBatch(
      plan->type,
      context.k(),
      static_cast<int>(erased_indices.size()),
      block_size,
      plan->backend,
      plan->path);
  plan->tile_bytes = tile;
  plan->output_batch = batch;
  plan->output_indices = erased_indices;

  const bool parity_only = std::all_of(
      erased_indices.begin(), erased_indices.end(), [&](int idx) { return idx >= context.k(); });
  if (parity_only) {
    std::vector<std::vector<uint8_t>> coeff_rows(erased_indices.size());
    for (size_t row = 0; row < erased_indices.size(); ++row) {
      coeff_rows[row].assign(
          context.generator()[static_cast<size_t>(erased_indices[row])].begin(),
          context.generator()[static_cast<size_t>(erased_indices[row])].begin() + context.k());
    }
    plan->input_indices.resize(context.k());
    std::iota(plan->input_indices.begin(), plan->input_indices.end(), 0);
    plan->table = MaybeAutotuneTable(context, *plan, coeff_rows, block_size);
    plan->table =
        MaybeEmpiricallyTunePlan(context, *plan, coeff_rows, block_size, &plan->tile_bytes, &plan->output_batch);
    PopulatePlanRows(
        plan->table,
        erased_indices,
        context.k(),
        [&](size_t out, int in) -> uint8_t { return context.generator()[erased_indices[out]][in]; },
        plan.get());
    FinalizePureBraidPlan(plan.get());
    return plan;
  }

  auto recovery = core::BuildRecoveryMatrix(context.generator(), survivors, erased_indices);
  std::vector<std::vector<uint8_t>> coeff_rows = recovery;
  plan->input_indices = survivors;
  plan->table = MaybeAutotuneTable(context, *plan, coeff_rows, block_size);
  plan->table =
      MaybeEmpiricallyTunePlan(context, *plan, coeff_rows, block_size, &plan->tile_bytes, &plan->output_batch);
  PopulatePlanRows(
      plan->table,
      erased_indices,
      context.k(),
      [&](size_t out, int in) -> uint8_t { return recovery[out][in]; },
      plan.get());
  FinalizePureBraidPlan(plan.get());
  return plan;
}

std::shared_ptr<Plan> PrepareInterUnitBatchPlan(
    const Context& context, const std::shared_ptr<Plan>& base_plan) {
  const auto requested_microkernel = context.options().microkernel_mode;
  const bool explicit_gfni_jit =
      requested_microkernel == BRAID_MICROKERNEL_FACTOR_JIT_GFNI ||
      (requested_microkernel == BRAID_MICROKERNEL_FACTOR_GFNI &&
       base_plan != nullptr &&
       base_plan->microkernel_mode ==
           BRAID_MICROKERNEL_FACTOR_JIT_GFNI);
  const bool explicit_gfni_schedule =
      requested_microkernel ==
          BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI ||
      requested_microkernel ==
          BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI ||
      (requested_microkernel == BRAID_MICROKERNEL_FACTOR_GFNI &&
       base_plan != nullptr &&
       (base_plan->microkernel_mode ==
            BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI ||
        base_plan->microkernel_mode ==
            BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI));
  if (base_plan == nullptr ||
      (requested_microkernel != BRAID_MICROKERNEL_BASELINE &&
       !explicit_gfni_jit && !explicit_gfni_schedule) ||
      base_plan->backend != BRAID_BACKEND_AVX512 ||
      base_plan->path != BRAID_PATH_BRAID ||
      base_plan->planned_block_size > (size_t{1} << 20)) {
    return base_plan;
  }
  if (explicit_gfni_schedule) {
    return base_plan;
  }
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  auto batch_plan = std::make_shared<Plan>(*base_plan);
  if (explicit_gfni_jit) {
    batch_plan->microkernel_mode =
        BRAID_MICROKERNEL_FACTOR_JIT_GFNI;
  } else if (batch_plan->input_indices.size() <= 32 &&
             batch_plan->planned_block_size <= (size_t{256} << 10)) {
    batch_plan->microkernel_mode = BRAID_MICROKERNEL_FACTOR_JIT;
  }
  if (batch_plan->microkernel_mode != BRAID_MICROKERNEL_FACTOR_JIT &&
      batch_plan->microkernel_mode !=
          BRAID_MICROKERNEL_FACTOR_JIT_GFNI) {
    return base_plan;
  }
  batch_plan->streaming_stores = true;
  CompileAvx512PlanJit(batch_plan.get());
  return batch_plan;
#else
  return base_plan;
#endif
}

}  // namespace braid::runtime
