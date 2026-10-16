#include "src/runtime/context.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "src/core/gf256.h"
#include "src/runtime/plan.h"
#include "src/runtime/cpu_features.h"

namespace braid::runtime {

namespace {

BackendCaps DetectCaps() {
  BackendCaps caps;
  caps.avx2 = BackendAvailable(BRAID_BACKEND_AVX2);
  caps.avx512 = BackendAvailable(BRAID_BACKEND_AVX512);
  return caps;
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

double ScoreTableForRows(
    const core::DecompositionTable& table, const std::vector<std::vector<uint8_t>>& rows) {
  if (rows.empty()) {
    return static_cast<double>(table.common_factor_nnz);
  }
  const int outputs = static_cast<int>(rows.size());
  const int inputs = static_cast<int>(rows.front().size());
  double sum_weight = 0.0;
  double sum_max_pos = 0.0;
  double sum_pos_sum = 0.0;
  double sum_reduction_cost = 0.0;
  size_t coeff_count = 0;
  for (int out = 0; out < outputs; ++out) {
    std::vector<uint8_t> row_positions;
    row_positions.reserve(17);
    for (int in = 0; in < inputs; ++in) {
      const uint8_t coefficient = rows[out][in];
      const uint8_t weight = table.weights[coefficient];
      sum_weight += weight;
      if (weight != 0) {
        sum_max_pos += table.alpha_positions[coefficient][weight - 1];
        for (uint8_t pos_idx = 0; pos_idx < weight; ++pos_idx) {
          const uint8_t position = table.alpha_positions[coefficient][pos_idx];
          sum_pos_sum += position;
          row_positions.push_back(position);
        }
      }
      ++coeff_count;
    }
    std::sort(row_positions.begin(), row_positions.end());
    row_positions.erase(std::unique(row_positions.begin(), row_positions.end()), row_positions.end());
    if (!row_positions.empty()) {
      uint8_t current = row_positions.back();
      sum_reduction_cost += TransitionCost(table, current, 0);
      for (int idx = static_cast<int>(row_positions.size()) - 2; idx >= 0; --idx) {
        const uint8_t next = row_positions[static_cast<size_t>(idx)];
        sum_reduction_cost += TransitionCost(table, current, next);
        current = next;
      }
    }
  }

  double pair_union = 0.0;
  double pair_span = 0.0;
  double pair_reduction_cost = 0.0;
  size_t pair_terms = 0;
  for (int out = 0; out < outputs; out += 2) {
    const int pair_outputs = std::min(2, outputs - out);
    for (int in = 0; in < inputs; ++in) {
      uint32_t union_mask = 0;
      uint8_t max_pos = 0;
      for (int local = 0; local < pair_outputs; ++local) {
        const uint8_t coefficient = rows[out + local][in];
        union_mask |= table.alpha_masks[coefficient];
        const uint8_t weight = table.weights[coefficient];
        if (weight != 0) {
          max_pos = std::max(max_pos, table.alpha_positions[coefficient][weight - 1]);
        }
      }
      pair_union += __builtin_popcount(union_mask);
      pair_span += max_pos;
      uint8_t current = max_pos;
      while (union_mask != 0) {
        const uint8_t next = static_cast<uint8_t>(31 - __builtin_clz(union_mask));
        union_mask &= ~(1u << next);
        if (next != current) {
          pair_reduction_cost += TransitionCost(table, current, next);
          current = next;
        }
      }
      pair_reduction_cost += TransitionCost(table, current, 0);
      ++pair_terms;
    }
  }

  const double avg_weight = coeff_count == 0 ? 0.0 : sum_weight / static_cast<double>(coeff_count);
  const double avg_max_pos = coeff_count == 0 ? 0.0 : sum_max_pos / static_cast<double>(coeff_count);
  const double avg_pos_sum = coeff_count == 0 ? 0.0 : sum_pos_sum / static_cast<double>(coeff_count);
  const double avg_reduction_cost = outputs == 0 ? 0.0 : sum_reduction_cost / static_cast<double>(outputs);
  const double avg_pair_union = pair_terms == 0 ? 0.0 : pair_union / static_cast<double>(pair_terms);
  const double avg_pair_span = pair_terms == 0 ? 0.0 : pair_span / static_cast<double>(pair_terms);
  const double avg_pair_reduction_cost = pair_terms == 0 ? 0.0 : pair_reduction_cost / static_cast<double>(pair_terms);
  return avg_pair_union * 1000.0 + avg_pair_span * 350.0 + avg_pair_reduction_cost * 220.0 +
         avg_reduction_cost * 180.0 + avg_max_pos * 150.0 + avg_pos_sum * 40.0 + avg_weight * 20.0 +
         static_cast<double>(table.common_factor_nnz);
}

}  // namespace

Context::Context(int k, int r, braid_options_t options)
    : k_(k),
      r_(r),
      n_(0),
      options_(options),
      caps_(DetectCaps()) {
  if (k_ <= 0 || r_ <= 0 || k_ > 255 || r_ > 255 || k_ > 255 - r_ ||
      options_.num_threads < 0 || options_.num_threads > 256) {
    throw std::invalid_argument("invalid code or thread parameters");
  }
  n_ = k_ + r_;
  if (options_.backend < BRAID_BACKEND_AUTO || options_.backend > BRAID_BACKEND_AVX512 ||
      (options_.path != BRAID_PATH_AUTO && options_.path != BRAID_PATH_BRAID) ||
      options_.tile_bytes > (size_t{16} << 20) || options_.output_batch < 0 ||
      options_.microkernel_mode < BRAID_MICROKERNEL_BASELINE ||
      options_.microkernel_mode > BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI ||
      (options_.repr_mode != BRAID_REPR_FIXED_X && options_.repr_mode != BRAID_REPR_RETUNE)) {
    throw std::invalid_argument("invalid or unsupported options");
  }
  if (options_.num_threads == 0) {
    options_.num_threads = 1;
  }
  if (!GfniAvailable() && options_.microkernel_mode >= BRAID_MICROKERNEL_FACTOR_JIT_GFNI) {
    options_.microkernel_mode = BRAID_MICROKERNEL_BASELINE;
  }
#if !defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  if (options_.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT ||
      options_.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI) {
    options_.microkernel_mode = BRAID_MICROKERNEL_BASELINE;
  }
#endif
  if (options_.repr_generator != 0 &&
      (options_.repr_generator < 0 || options_.repr_generator > 255 ||
       !HasExactOrder17(static_cast<uint8_t>(options_.repr_generator)))) {
    throw std::invalid_argument("representation generator must have exact order 17");
  }
  if (options_.pin_threads != 0) {
    try { topology_ = DetectTopology(); } catch (...) { topology_ = {}; }
  }
  if (options_.num_threads > 1) {
    auto cpu_order = BuildCpuBindingOrder(topology_, options_.num_threads, options_.numa_aware != 0);
    if (cpu_order.empty()) {
      cpu_order.resize(static_cast<size_t>(options_.num_threads), -1);
    } else if (options_.pin_threads == 0) {
      std::fill(cpu_order.begin(), cpu_order.end(), -1);
    }
    thread_pool_ = std::make_unique<ThreadPool>(std::move(cpu_order));
  }
  generator_ = core::BuildSystematicGeneratorMatrix(k_, r_);
  table_ = BuildSelectedDecompositionTable();
}

braid_backend_t Context::ResolveBackend() const {
  switch (options_.backend) {
    case BRAID_BACKEND_AVX512:
      return caps_.avx512 ? BRAID_BACKEND_AVX512 : (caps_.avx2 ? BRAID_BACKEND_AVX2 : BRAID_BACKEND_SCALAR);
    case BRAID_BACKEND_AVX2:
      return caps_.avx2 ? BRAID_BACKEND_AVX2 : BRAID_BACKEND_SCALAR;
    case BRAID_BACKEND_SCALAR:
      return BRAID_BACKEND_SCALAR;
    case BRAID_BACKEND_AUTO:
    default:
      if (caps_.avx512) {
        return BRAID_BACKEND_AVX512;
      }
      if (caps_.avx2) {
        return BRAID_BACKEND_AVX2;
      }
      return BRAID_BACKEND_SCALAR;
  }
}

braid_path_t Context::ResolvePath(PlanType type, int outputs, size_t block_size, braid_backend_t backend) const {
  if (options_.path != BRAID_PATH_AUTO) {
    return options_.path;
  }
  (void)type;
  (void)outputs;
  (void)block_size;
  (void)backend;
  return BRAID_PATH_BRAID;
}

std::pair<size_t, int> Context::ChooseTileAndBatch(
    PlanType type,
    int inputs,
    int outputs,
    size_t block_size,
    braid_backend_t backend,
    braid_path_t path) const {
  const size_t vector_width =
      backend == BRAID_BACKEND_AVX512 ? 64 : (backend == BRAID_BACKEND_AVX2 ? 32 : 16);
  const bool gfni_factor_requested =
      options_.microkernel_mode == BRAID_MICROKERNEL_FACTOR_GFNI ||
      options_.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI ||
      options_.microkernel_mode ==
          BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI ||
      options_.microkernel_mode ==
          BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI;
  const auto align_tile = [vector_width](size_t tile) {
    tile = std::max(tile, vector_width);
    return ((tile + vector_width - 1) / vector_width) * vector_width;
  };
  if (options_.tile_bytes != 0 && options_.output_batch > 0) {
    return {align_tile(options_.tile_bytes), std::min(outputs, options_.output_batch)};
  }
  int batch = 1;
  if (options_.output_batch > 0) {
    batch = std::min(outputs, options_.output_batch);
  } else if (path == BRAID_PATH_DENSE) {
    if (backend == BRAID_BACKEND_AVX512) {
      batch = std::min(outputs, inputs >= 48 ? 4 : 6);
    } else if (backend == BRAID_BACKEND_AVX2) {
      batch = std::min(outputs, inputs >= 48 ? 3 : 4);
    }
  } else {
    if (backend == BRAID_BACKEND_AVX512) {
      if (type == PlanType::kDecode && inputs <= 8 && outputs == 2 && block_size >= (4u << 20)) {
        batch = 2;
      } else if (type == PlanType::kDecode && inputs <= 20 && outputs == 3 && block_size >= (4u << 20)) {
        batch = 3;
      } else if (type == PlanType::kDecode && inputs <= 12 && outputs >= 4 && block_size >= (4u << 20)) {
        batch = std::min(outputs, 4);
      } else if (type == PlanType::kEncode && inputs <= 12 && outputs >= 4 && block_size >= (4u << 20)) {
        batch = 1;
      } else if (type == PlanType::kEncode && inputs <= 20 && outputs == 3 && block_size >= (4u << 20)) {
        batch = 3;
      } else if (type == PlanType::kDecode && inputs >= 64 && outputs >= 4) {
        if (inputs >= 80 && outputs == 4) {
          batch = 4;
        } else if (inputs < 80 && outputs == 4 &&
                   block_size <= (4u << 20)) {
          batch = 4;
        } else if (block_size <= (1u << 20)) {
          batch = 4;
        } else if (block_size <= (4u << 20)) {
          batch = 2;
        } else if (block_size == (8u << 20) && options_.num_threads >= 16) {
          batch = std::min(outputs, 4);
        } else if (block_size >= (8u << 20)) {
          batch = std::min(outputs, 4);
        } else {
          batch = 2;
        }
      } else if (type == PlanType::kEncode && inputs >= 64 && outputs >= 4) {
        if (inputs >= 80 && outputs == 4) {
          batch = 4;
        } else if (inputs < 80 && block_size <= (4u << 20)) {
          batch = std::min(outputs, 4);
        } else if (block_size <= (1u << 20)) {
          batch = std::min(outputs, 4);
        } else if (block_size <= (8u << 20)) {
          batch = 2;
        } else {
          batch = std::min(outputs, 4);
        }
      } else if (type == PlanType::kEncode && outputs >= 4) {
        batch = std::min(outputs, 4);
      } else if (type == PlanType::kEncode && inputs >= 64 && outputs == 3) {
        batch = 3;
      } else if (type == PlanType::kDecode && outputs <= 2) {
        batch = outputs;
      } else if (type == PlanType::kDecode && inputs < 64 && outputs >= 4) {
        // Keep four-output mid-chain recovery in one input pass. A 3+1 split
        // pays a second read and a second representation reduction.
        batch = std::min(outputs, 4);
      } else if (inputs >= 64 && outputs == 3) {
        batch = outputs;
      } else {
        batch = std::min(outputs, inputs >= 64 ? 2 : (inputs >= 24 ? 3 : 4));
      }
    } else if (backend == BRAID_BACKEND_AVX2) {
      if (type == PlanType::kEncode && inputs >= 64 && outputs >= 4) {
        batch = 1;
      } else if (type == PlanType::kDecode && inputs >= 64 && outputs >= 4) {
        batch = 1;
      } else {
        batch = std::min(outputs, inputs >= 64 ? 1 : 2);
      }
    }
  }
  batch = std::max(batch, 1);

  if (options_.tile_bytes != 0) {
    return {align_tile(options_.tile_bytes), batch};
  }

  size_t cache_budget = 16 * 1024;
  if (backend == BRAID_BACKEND_AVX512) {
    cache_budget = path == BRAID_PATH_BRAID ? 512 * 1024 : 256 * 1024;
  } else if (backend == BRAID_BACKEND_AVX2) {
    cache_budget = path == BRAID_PATH_BRAID ? 384 * 1024 : 192 * 1024;
  } else if (path == BRAID_PATH_BRAID) {
    cache_budget = 128 * 1024;
  }

  if (block_size >= (4u << 20)) {
    cache_budget = cache_budget * 5 / 4;
  }
  if (inputs >= 64) {
    cache_budget = cache_budget * 3 / 4;
  } else if (inputs <= 8 && outputs <= 2) {
    cache_budget = cache_budget * 3 / 2;
  }

  const size_t working_sets = path == BRAID_PATH_BRAID ? static_cast<size_t>(batch) * 17 : static_cast<size_t>(batch);
  size_t tile = cache_budget / std::max<size_t>(working_sets, 1);
  tile = std::max(tile, vector_width * 4);
  tile = (tile / vector_width) * vector_width;
  tile = std::min<size_t>(tile, 64 * 1024);

  if (path == BRAID_PATH_BRAID && backend == BRAID_BACKEND_AVX512) {
    if (inputs > 20 && inputs < 64 && block_size > (1u << 20)) {
      // Mid-chain factor scratch is an RMW workload over batch*17 planes.
      // Keep that active set near private cache instead of applying the
      // long-stripe byte-stream tile selected for register kernels.
      if (type == PlanType::kEncode) {
        tile = inputs <= 24 ? 1024 : 2 * 1024;
      } else if (outputs <= 2) {
        tile = inputs <= 24 ? 512 : 4 * 1024;
      } else if (outputs == 3) {
        tile = inputs <= 32 ? 512 : 1024;
      } else {
        tile = inputs <= 24 ? 512 : 2 * 1024;
      }
    } else if (type == PlanType::kDecode && inputs <= 8 && outputs == 2 && block_size >= (4u << 20)) {
      tile = 64 * 1024;
    } else if (type == PlanType::kDecode && inputs <= 20 && outputs == 3 && block_size >= (4u << 20)) {
      tile = 64 * 1024;
    } else if (type == PlanType::kDecode && inputs <= 12 && outputs >= 4 && block_size >= (4u << 20)) {
      tile = 64 * 1024;
    } else if (type == PlanType::kEncode && inputs <= 12 && outputs >= 4 && block_size >= (4u << 20)) {
      tile = 32 * 1024;
    } else if (type == PlanType::kEncode && inputs <= 20 && outputs == 3 && block_size >= (4u << 20)) {
      tile = 64 * 1024;
    } else if (type == PlanType::kDecode && inputs >= 64 && outputs >= 4) {
      if (gfni_factor_requested) {
        tile = 1024;
      } else {
        tile = (inputs >= 80 && outputs == 4) ||
                     (inputs < 80 && outputs == 4 &&
                      block_size <= (4u << 20)) ||
                     block_size <= (1u << 20)
                 ? 2 * 1024
                 : 8 * 1024;
      }
    } else if (type == PlanType::kEncode && inputs >= 64 && outputs >= 4) {
      if (gfni_factor_requested) {
        // Multiple register-resident output groups immediately reuse this
        // source window. Keep k*tile within a small fraction of private L2.
        tile = inputs < 80 ? 512 : 384;
      } else if (inputs >= 80 && outputs == 4) {
        tile = 2 * 1024;
      } else if (inputs < 80 && block_size <= (4u << 20)) {
        tile = 1024;
      } else {
        tile = block_size <= (1u << 20) ? 1024 : 8 * 1024;
      }
    } else if (type == PlanType::kEncode && inputs <= 16 && outputs >= 4) {
      tile = 16 * 1024;
    } else if (inputs >= 64 && outputs >= 4) {
      if (block_size >= (64u << 20) || options_.num_threads >= 32) {
        tile = 8 * 1024;
      } else if (block_size >= (16u << 20)) {
        tile = 16 * 1024;
      } else if (block_size <= (4u << 20) && options_.num_threads <= 4) {
        tile = 8 * 1024;
      } else {
        tile = 16 * 1024;
      }
    } else if (type == PlanType::kEncode && inputs >= 64 && outputs == 3) {
      tile = 16 * 1024;
    } else if (type == PlanType::kDecode && inputs >= 64 && outputs == 1) {
      tile = 12 * 1024;
    } else if (type == PlanType::kDecode && inputs >= 64 && outputs <= 3) {
      if (inputs < 80 && block_size <= (1u << 20)) {
        tile = outputs == 2 ? 768 : 512;
      } else {
        tile = 1024;
      }
    } else if (inputs >= 64 && outputs == 3) {
      tile = 16 * 1024;
    } else if (inputs >= 64 && batch <= 2) {
      tile = 24 * 1024;
    } else if (batch >= 4) {
      tile = 24 * 1024;
    } else if (batch == 3) {
      tile = 24 * 1024;
    }
  }

  if (path == BRAID_PATH_BRAID && backend == BRAID_BACKEND_AVX2) {
    if (type == PlanType::kEncode && inputs >= 64 && outputs >= 4) {
      if (block_size <= (512u << 10)) {
        tile = 8 * 1024;
      } else if (block_size <= (1u << 20)) {
        tile = 16 * 1024;
      } else if (block_size <= (64u << 20)) {
        tile = 16 * 1024;
      } else {
        tile = 16 * 1024;
      }
    } else if (type == PlanType::kDecode && inputs >= 64 && outputs >= 4) {
      tile = 32 * 1024;
    } else if (inputs >= 64) {
      tile = 16 * 1024;
    }
  }

  if (block_size != 0) {
    tile = std::min(tile, block_size);
  }
  return {align_tile(tile), batch};
}

std::string Context::BuildDecodeKey(const std::vector<int>& erased_indices, size_t block_size) const {
  std::ostringstream oss;
  oss << block_size << ':';
  for (size_t i = 0; i < erased_indices.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << erased_indices[i];
  }
  return oss.str();
}

core::DecompositionTable Context::BuildSelectedDecompositionTable() const {
  std::vector<std::vector<uint8_t>> encode_rows(static_cast<size_t>(r_));
  for (int row = 0; row < r_; ++row) {
    encode_rows[static_cast<size_t>(row)].assign(
        generator_[static_cast<size_t>(k_ + row)].begin(),
        generator_[static_cast<size_t>(k_ + row)].begin() + k_);
  }
  return SelectTableForRows(encode_rows);
}

core::DecompositionTable Context::SelectTableForRows(const std::vector<std::vector<uint8_t>>& rows) const {
  if (options_.repr_generator > 0 && options_.repr_generator < 256) {
    return core::BuildDecompositionTable(static_cast<uint8_t>(options_.repr_generator));
  }
  if (options_.repr_mode != BRAID_REPR_RETUNE) {
    return core::BuildDecompositionTable();
  }

  core::DecompositionTable best = core::BuildDecompositionTable();
  double best_score = ScoreTableForRows(best, rows);
  for (int candidate = 2; candidate < 256; ++candidate) {
    const uint8_t generator = static_cast<uint8_t>(candidate);
    if (!HasExactOrder17(generator)) {
      continue;
    }
    auto table = core::BuildDecompositionTable(generator);
    const double score = ScoreTableForRows(table, rows);
    if (score < best_score ||
        (score == best_score &&
         (table.common_factor_nnz < best.common_factor_nnz ||
          (table.common_factor_nnz == best.common_factor_nnz && generator < best.generator)))) {
      best_score = score;
      best = std::move(table);
    }
  }

  const bool enable_local_order_search =
      !rows.empty() && (rows.size() >= 4 || rows.front().size() >= 32);
  if (enable_local_order_search) {
    bool improved = true;
    int rounds = 0;
    while (improved && rounds < 1) {
      improved = false;
      ++rounds;
      std::array<uint8_t, 17> best_order = best.x_powers;
      for (int i = 1; i < 16; ++i) {
        for (int j = i + 1; j < 17; ++j) {
          auto candidate_order = best_order;
          std::swap(candidate_order[static_cast<size_t>(i)], candidate_order[static_cast<size_t>(j)]);
          auto table = core::BuildDecompositionTable(candidate_order);
          const double score = ScoreTableForRows(table, rows);
          if (score < best_score ||
              (score == best_score &&
               (table.common_factor_nnz < best.common_factor_nnz ||
                (table.common_factor_nnz == best.common_factor_nnz && table.x_exponents < best.x_exponents)))) {
            best_score = score;
            best = std::move(table);
            best_order = best.x_powers;
            improved = true;
          }
        }
      }
    }
  }
  return best;
}

std::shared_ptr<Plan> Context::BuildEncodePlanImpl(size_t block_size) { return BuildEncodePlan(*this, block_size); }

std::shared_ptr<Plan> Context::BuildDecodePlanImpl(const std::vector<int>& erased_indices, size_t block_size) {
  return BuildDecodePlan(*this, erased_indices, block_size);
}

std::shared_ptr<Plan> Context::GetOrBuildEncodePlan(size_t block_size) {
  if (!options_.enable_plan_cache) {
    return BuildEncodePlanImpl(block_size);
  }
  std::lock_guard<std::mutex> lock(plan_cache_mutex_);
  auto it = encode_plan_cache_.find(block_size);
  if (it != encode_plan_cache_.end()) {
    return it->second;
  }
  auto plan = BuildEncodePlanImpl(block_size);
  encode_plan_cache_[block_size] = plan;
  return plan;
}

std::shared_ptr<Plan> Context::GetOrBuildDecodePlan(const std::vector<int>& erased_indices, size_t block_size) {
  if (!options_.enable_plan_cache) {
    return BuildDecodePlanImpl(erased_indices, block_size);
  }
  std::lock_guard<std::mutex> lock(plan_cache_mutex_);
  const std::string key = BuildDecodeKey(erased_indices, block_size);
  auto it = decode_plan_cache_.find(key);
  if (it != decode_plan_cache_.end()) {
    return it->second;
  }
  auto plan = BuildDecodePlanImpl(erased_indices, block_size);
  decode_plan_cache_[key] = plan;
  return plan;
}

std::shared_ptr<Plan> Context::GetOrBuildEncodeBatchPlan(size_t block_size) {
  if (!options_.enable_plan_cache) {
    return PrepareInterUnitBatchPlan(*this, BuildEncodePlanImpl(block_size));
  }
  std::lock_guard<std::mutex> lock(plan_cache_mutex_);
  auto it = encode_batch_plan_cache_.find(block_size);
  if (it != encode_batch_plan_cache_.end()) {
    return it->second;
  }
  auto plan =
      PrepareInterUnitBatchPlan(*this, BuildEncodePlanImpl(block_size));
  encode_batch_plan_cache_[block_size] = plan;
  return plan;
}

std::shared_ptr<Plan> Context::GetOrBuildDecodeBatchPlan(
    const std::vector<int>& erased_indices, size_t block_size) {
  if (!options_.enable_plan_cache) {
    return PrepareInterUnitBatchPlan(
        *this, BuildDecodePlanImpl(erased_indices, block_size));
  }
  std::lock_guard<std::mutex> lock(plan_cache_mutex_);
  const std::string key = BuildDecodeKey(erased_indices, block_size);
  auto it = decode_batch_plan_cache_.find(key);
  if (it != decode_batch_plan_cache_.end()) {
    return it->second;
  }
  auto plan = PrepareInterUnitBatchPlan(
      *this, BuildDecodePlanImpl(erased_indices, block_size));
  decode_batch_plan_cache_[key] = plan;
  return plan;
}

}  // namespace braid::runtime
