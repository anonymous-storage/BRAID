#ifndef BRAID_KERNELS_NEON_KERNELS_H_
#define BRAID_KERNELS_NEON_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "src/core/decomposition.h"
#include "src/runtime/types.h"

namespace braid::kernels {

void ExecuteBraidNeon(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);



}  // namespace braid::kernels

#endif  // BRAID_KERNELS_NEON_KERNELS_H_
