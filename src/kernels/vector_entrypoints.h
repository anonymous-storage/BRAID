#ifndef BRAID_KERNELS_VECTOR_ENTRYPOINTS_H_
#define BRAID_KERNELS_VECTOR_ENTRYPOINTS_H_

#include <cstddef>
#include <cstdint>

#include "src/core/decomposition.h"
#include "src/runtime/types.h"

namespace braid::kernels {



void ExecuteBraidAVX2(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);



void ExecuteBraidAVX512(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);

void ExecuteBraidAVX512GFNI(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);



void ExecuteBraidNeon(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size);

}  // namespace braid::kernels

#endif  // BRAID_KERNELS_VECTOR_ENTRYPOINTS_H_

