#ifndef BRAID_RUNTIME_TYPES_H_
#define BRAID_RUNTIME_TYPES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "braid/braid.h"
#include "src/core/decomposition.h"

namespace braid::runtime {

struct Avx512PlanJit;

enum class PlanType {
  kEncode,
  kDecode,
};

struct BackendCaps {
  bool avx2 = false;
  bool avx512 = false;
};

constexpr size_t kFactorPlaneCount = 17;
constexpr size_t kFactorMaxBatch = 4;
constexpr size_t kFactorMaxTargetsPerInput = 3 * kFactorMaxBatch;

struct OutputPlanRow {
  std::vector<uint8_t> coefficients;
  std::vector<uint32_t> alpha_masks;
  std::vector<uint8_t> alpha_weights;
  std::vector<std::array<uint8_t, 3>> alpha_positions;
  std::array<std::vector<uint16_t>, kFactorPlaneCount> plane_input_indices;
  std::array<uint8_t, 8> basis_elements{1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<uint8_t> basis_coordinates;
  uint8_t basis_used_mask = 0;
  uint32_t basis_route_weight = 0;
};

struct FactorTargetList {
  uint8_t count = 0;
  uint16_t initialize_mask = 0;
  std::array<uint8_t, kFactorMaxTargetsPerInput> ids{};
};

// Immutable execution IR compiled from the representation-level factorization.
// Keeping this in the plan avoids rebuilding coefficient routes in every worker.
struct FactorBatchSchedule {
  uint8_t batch_size = 0;
  std::vector<FactorTargetList> target_lists;
  std::array<uint32_t, kFactorMaxBatch> used_plane_masks{};
  std::array<uint8_t, kFactorMaxBatch> max_planes{};
  std::array<std::array<uint8_t, kFactorPlaneCount>, kFactorMaxBatch> used_planes{};
  std::array<uint8_t, kFactorMaxBatch> used_plane_counts{};
  std::array<uint8_t, kFactorMaxBatch> reconstruction_masks{1, 2, 4, 8};
};

struct FactorGroup {
  uint8_t plane = 0;
  uint8_t output_mask = 0;
};

struct FactorGroupedSchedule {
  uint8_t batch_size = 0;
  std::vector<FactorTargetList> target_lists;
  std::vector<FactorGroup> groups;
  std::array<std::array<std::vector<uint8_t>, kFactorPlaneCount>,
             kFactorMaxBatch>
      reduction_groups;
};

struct Plan {
  PlanType type;
  braid_backend_t backend = BRAID_BACKEND_SCALAR;
  braid_path_t path = BRAID_PATH_BRAID;
  braid_microkernel_mode_t microkernel_mode = BRAID_MICROKERNEL_BASELINE;
  core::DecompositionTable table;
  size_t planned_block_size = 0;
  int execution_threads = 1;
  size_t tile_bytes = 0;
  int output_batch = 1;
  std::vector<int> input_indices;
  std::vector<int> output_indices;
  std::vector<OutputPlanRow> rows;
  std::vector<FactorBatchSchedule> factor_schedules;
  std::vector<FactorGroupedSchedule> factor_grouped_schedules;
  std::shared_ptr<Avx512PlanJit> avx512_jit;
  bool streaming_stores = false;
  bool gfni_object = false;
};

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_TYPES_H_
