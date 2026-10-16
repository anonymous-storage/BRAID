#ifndef BRAID_BRAID_H_
#define BRAID_BRAID_H_

#include <stddef.h>
#include <stdint.h>
#include "braid/export.h"
#include "braid/version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct braid_code braid_code_t;
typedef struct braid_plan braid_plan_t;

typedef enum braid_status {
    BRAID_STATUS_OK = 0,
    BRAID_STATUS_INVALID_ARGUMENT = 1,
    BRAID_STATUS_UNSUPPORTED = 2,
    BRAID_STATUS_INTERNAL_ERROR = 3,
} braid_status_t;

typedef enum braid_backend {
    BRAID_BACKEND_AUTO = 0,
    BRAID_BACKEND_SCALAR = 1,
    BRAID_BACKEND_AVX2 = 2,
    BRAID_BACKEND_AVX512 = 3,
    BRAID_BACKEND_NEON = 4, /* Reserved; unavailable in this release. */
} braid_backend_t;

typedef enum braid_path {
    BRAID_PATH_AUTO = 0,
    BRAID_PATH_BRAID = 1,
    BRAID_PATH_DENSE = 2, /* Reserved for compatibility; rejected at creation. */
} braid_path_t;

typedef enum braid_microkernel_mode {
    BRAID_MICROKERNEL_BASELINE = 0,
    BRAID_MICROKERNEL_PAIR_DIRECT = 1,
    BRAID_MICROKERNEL_FACTOR_SCRATCH = 2,
    BRAID_MICROKERNEL_FACTOR_REGISTER = 3,
    BRAID_MICROKERNEL_FACTOR_DIRECT = 4,
    BRAID_MICROKERNEL_FACTOR_POLYNOMIAL_FUSED = 5,
    BRAID_MICROKERNEL_FACTOR_BASIS_LUT = 6,
    BRAID_MICROKERNEL_FACTOR_PLANE_STREAMING = 7,
    BRAID_MICROKERNEL_FACTOR_JIT = 8,
    BRAID_MICROKERNEL_FACTOR_GROUPED_SCRATCH = 9,
    BRAID_MICROKERNEL_FACTOR_JIT_GFNI = 10,
    BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI = 11,
    BRAID_MICROKERNEL_FACTOR_GFNI = 12,
    BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI = 13,
} braid_microkernel_mode_t;

typedef enum braid_representation_mode {
    BRAID_REPR_FIXED_X = 0,
    BRAID_REPR_RETUNE = 1,
} braid_representation_mode_t;

typedef struct braid_options {
    braid_backend_t backend;
    braid_path_t path;
    size_t tile_bytes;
    int output_batch;
    int num_threads;
    int pin_threads;
    int numa_aware;
    int enable_plan_cache;
    braid_microkernel_mode_t microkernel_mode;
    braid_representation_mode_t repr_mode;
    int repr_generator;
} braid_options_t;

/* ABI v1: this structure and enum values are frozen; do not extend in place.
 * Initialize options before overriding individual members. Backend choices are
 * preferences: unavailable SIMD backends safely fall back to factorized scalar
 * or AVX2. Query availability or inspect the resolved plan backend if required.
 */
BRAID_API const char* braid_version(void);
BRAID_API unsigned braid_abi_version(void);
BRAID_API braid_status_t braid_options_init(braid_options_t* options, size_t options_size);
BRAID_API int braid_backend_available(braid_backend_t backend);
BRAID_API int braid_gfni_available(void);
BRAID_API const char* braid_status_string(braid_status_t status);
BRAID_API braid_status_t braid_create_code_ex(int k, int r, const braid_options_t* options,
                                            size_t options_size, braid_code_t** result);

BRAID_API braid_code_t* braid_create_code(int k, int r, const braid_options_t* options);
BRAID_API void braid_destroy_code(braid_code_t* code);

BRAID_API braid_status_t braid_encode(
    braid_code_t* code,
    const uint8_t* const* data_blocks,
    uint8_t* const* parity_blocks,
    size_t block_size);

BRAID_API braid_status_t braid_decode(
    braid_code_t* code,
    const uint8_t* const* available_blocks,
    const int* erased_indices,
    int erased_count,
    uint8_t* const* recovered_blocks,
    size_t block_size);

BRAID_API braid_status_t braid_encode_batch(
    braid_code_t* code,
    const uint8_t* const* const* data_stripes,
    uint8_t* const* const* parity_stripes,
    size_t stripe_count,
    size_t block_size);

BRAID_API braid_status_t braid_decode_batch(
    braid_code_t* code,
    const uint8_t* const* const* available_stripes,
    const int* erased_indices,
    int erased_count,
    uint8_t* const* const* recovered_stripes,
    size_t stripe_count,
    size_t block_size);

BRAID_API braid_plan_t* braid_build_encode_plan(braid_code_t* code);
BRAID_API braid_plan_t* braid_build_encode_plan_for_block_size(braid_code_t* code, size_t block_size);
BRAID_API braid_plan_t* braid_build_encode_batch_plan_for_block_size(
    braid_code_t* code, size_t block_size);
BRAID_API braid_plan_t* braid_build_decode_plan(
    braid_code_t* code,
    const int* erased_indices,
    int erased_count);
BRAID_API braid_plan_t* braid_build_decode_plan_for_block_size(
    braid_code_t* code,
    const int* erased_indices,
    int erased_count,
    size_t block_size);
BRAID_API braid_plan_t* braid_build_decode_batch_plan_for_block_size(
    braid_code_t* code,
    const int* erased_indices,
    int erased_count,
    size_t block_size);
BRAID_API void braid_destroy_plan(braid_plan_t* plan);
BRAID_API size_t braid_plan_tile_bytes(const braid_plan_t* plan);
BRAID_API int braid_plan_output_batch(const braid_plan_t* plan);
BRAID_API braid_backend_t braid_plan_backend(const braid_plan_t* plan);
BRAID_API braid_path_t braid_plan_path(const braid_plan_t* plan);
BRAID_API braid_microkernel_mode_t braid_plan_microkernel_mode(const braid_plan_t* plan);
BRAID_API int braid_plan_uses_streaming_stores(const braid_plan_t* plan);
BRAID_API int braid_plan_repr_generator(const braid_plan_t* plan);
BRAID_API int braid_code_repr_generator(const braid_code_t* code);
BRAID_API size_t braid_plan_slp_xor_ops_before(const braid_plan_t* plan);
BRAID_API size_t braid_plan_slp_xor_ops_after(const braid_plan_t* plan);
BRAID_API size_t braid_plan_slp_potential_xor_ops(const braid_plan_t* plan);
BRAID_API int braid_plan_slp_groups(const braid_plan_t* plan);
BRAID_API int braid_plan_slp_potential_groups(const braid_plan_t* plan);
BRAID_API int braid_plan_slp_triple_groups(const braid_plan_t* plan);
BRAID_API size_t braid_plan_gfni_prefetch_distance(const braid_plan_t* plan);
BRAID_API int braid_plan_gfni_prefetch_hint(const braid_plan_t* plan);
BRAID_API size_t braid_plan_gfni_prefetch_input_stride(const braid_plan_t* plan);

BRAID_API braid_status_t braid_execute_encode_plan(
    braid_code_t* code,
    braid_plan_t* plan,
    const uint8_t* const* data_blocks,
    uint8_t* const* output_blocks,
    size_t block_size);

BRAID_API braid_status_t braid_execute_decode_plan(
    braid_code_t* code,
    braid_plan_t* plan,
    const uint8_t* const* input_blocks,
    uint8_t* const* output_blocks,
    size_t block_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BRAID_BRAID_H_
