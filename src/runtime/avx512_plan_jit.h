#ifndef BRAID_RUNTIME_AVX512_PLAN_JIT_H_
#define BRAID_RUNTIME_AVX512_PLAN_JIT_H_

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

namespace braid::runtime {

struct Plan;

using Avx512JitBatchFn =
    void (*)(const uint8_t* const* inputs, uint8_t* const* outputs, size_t vector_bytes);

struct Avx512PlanJit {
  std::vector<Avx512JitBatchFn> batch_functions;
  std::vector<Avx512JitBatchFn> cached_store_batch_functions;
  std::vector<uint8_t> batch_sizes;
  std::vector<std::array<uint8_t, 4>> batch_output_indices;
  std::vector<void*> allocations;
  uint64_t slp_xor_ops_before = 0;
  uint64_t slp_xor_ops_after = 0;
  uint64_t slp_potential_xor_ops = 0;
  uint32_t slp_groups = 0;
  uint32_t slp_potential_groups = 0;
  uint32_t slp_triple_groups = 0;
  size_t prefetch_distance = 0;
  uint32_t prefetch_hint = 0;
  size_t prefetch_input_stride = 1;

  ~Avx512PlanJit();
};

bool CompileAvx512PlanJit(Plan* plan);

void ExecuteAvx512PlanJit(
    const Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_AVX512_PLAN_JIT_H_
