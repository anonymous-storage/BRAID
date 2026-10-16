#ifndef BRAID_KERNELS_BACKEND_DISPATCH_H_
#define BRAID_KERNELS_BACKEND_DISPATCH_H_

#include <cstddef>
#include <cstdint>

#include "src/core/decomposition.h"
#include "src/runtime/thread_pool.h"
#include "src/runtime/types.h"

namespace braid::kernels {

void ExecutePlan(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size,
    int num_threads,
    braid::runtime::ThreadPool* thread_pool);

void ExecutePlanBatch(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* const* input_stripes,
    uint8_t* const* const* output_stripes,
    size_t stripe_count,
    size_t block_size,
    int num_threads,
    braid::runtime::ThreadPool* thread_pool);

}  // namespace braid::kernels

#endif  // BRAID_KERNELS_BACKEND_DISPATCH_H_
