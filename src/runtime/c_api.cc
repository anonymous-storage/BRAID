#include "braid/braid.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/kernels/backend_dispatch.h"
#include "src/runtime/avx512_plan_jit.h"
#include "src/runtime/context.h"
#include "src/runtime/cpu_features.h"

struct braid_code {
  explicit braid_code(std::unique_ptr<braid::runtime::Context> impl)
      : owner_token(std::make_shared<int>(0)), impl(std::move(impl)) {}
  std::shared_ptr<const void> owner_token;
  std::unique_ptr<braid::runtime::Context> impl;
};

struct braid_plan {
  braid_plan(std::shared_ptr<const void> owner, std::shared_ptr<braid::runtime::Plan> impl)
      : owner(owner), impl(std::move(impl)) {}
  std::shared_ptr<const void> owner;
  std::shared_ptr<braid::runtime::Plan> impl;
};

namespace {

braid_options_t DefaultOptions() {
  braid_options_t options{};
  options.backend = BRAID_BACKEND_AUTO;
  options.path = BRAID_PATH_AUTO;
  options.tile_bytes = 0;
  options.output_batch = 0;
  options.num_threads = 1;
  options.pin_threads = 0;
  options.numa_aware = 0;
  options.enable_plan_cache = 1;
  options.microkernel_mode = BRAID_MICROKERNEL_BASELINE;
  options.repr_mode = BRAID_REPR_FIXED_X;
  options.repr_generator = 0;
  return options;
}

}  // namespace

extern "C" {

const char* braid_version(void) { return BRAID_VERSION_STRING; }
unsigned braid_abi_version(void) { return BRAID_ABI_VERSION; }
braid_status_t braid_options_init(braid_options_t* options, size_t size) {
  if (!options || size != sizeof(*options)) return BRAID_STATUS_INVALID_ARGUMENT;
  *options = DefaultOptions();
  return BRAID_STATUS_OK;
}
int braid_backend_available(braid_backend_t backend) {
  return braid::runtime::BackendAvailable(backend) ? 1 : 0;
}
int braid_gfni_available(void) { return braid::runtime::GfniAvailable() ? 1 : 0; }
const char* braid_status_string(braid_status_t status) {
  switch (status) {
    case BRAID_STATUS_OK: return "success";
    case BRAID_STATUS_INVALID_ARGUMENT: return "invalid argument";
    case BRAID_STATUS_UNSUPPORTED: return "unsupported";
    default: return "internal error";
  }
}
braid_status_t braid_create_code_ex(int k, int r, const braid_options_t* options,
                                  size_t size, braid_code_t** result) {
  if (!result) return BRAID_STATUS_INVALID_ARGUMENT;
  *result = nullptr;
  if ((options && size != sizeof(*options)) || (!options && size != 0))
    return BRAID_STATUS_INVALID_ARGUMENT;
  if (options && (options->path == BRAID_PATH_DENSE || options->backend == BRAID_BACKEND_NEON))
    return BRAID_STATUS_UNSUPPORTED;
  try {
    auto impl = std::make_unique<braid::runtime::Context>(k, r, options ? *options : DefaultOptions());
    *result = new braid_code(std::move(impl));
    return BRAID_STATUS_OK;
  } catch (const std::invalid_argument&) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  } catch (...) { return BRAID_STATUS_INTERNAL_ERROR; }
}

braid_code_t* braid_create_code(int k, int r, const braid_options_t* options) {
  try {
    braid_options_t resolved = options ? *options : DefaultOptions();
    auto impl = std::make_unique<braid::runtime::Context>(k, r, resolved);
    return new braid_code(std::move(impl));
  } catch (...) {
    return nullptr;
  }
}

void braid_destroy_code(braid_code_t* code) { delete code; }

braid_status_t braid_encode(
    braid_code_t* code,
    const uint8_t* const* data_blocks,
    uint8_t* const* parity_blocks,
    size_t block_size) {
  if (block_size > static_cast<size_t>(PTRDIFF_MAX) || code == nullptr || data_blocks == nullptr || parity_blocks == nullptr) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  for (int input = 0; input < code->impl->k(); ++input) {
    if (data_blocks[input] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  for (int output = 0; output < code->impl->r(); ++output) {
    if (parity_blocks[output] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    auto plan = code->impl->GetOrBuildEncodePlan(block_size);
    braid::kernels::ExecutePlan(
        plan->table,
        *plan,
        data_blocks,
        parity_blocks,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

braid_status_t braid_decode(
    braid_code_t* code,
    const uint8_t* const* available_blocks,
    const int* erased_indices,
    int erased_count,
    uint8_t* const* recovered_blocks,
    size_t block_size) {
  if (block_size > static_cast<size_t>(PTRDIFF_MAX) || code == nullptr || available_blocks == nullptr || erased_indices == nullptr || recovered_blocks == nullptr ||
      erased_count <= 0 || erased_count > code->impl->r()) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  for (int output = 0; output < erased_count; ++output) {
    if (recovered_blocks[output] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    auto plan =
        code->impl->GetOrBuildDecodePlan(std::vector<int>(erased_indices, erased_indices + erased_count), block_size);
    std::vector<const uint8_t*> ordered_inputs;
    ordered_inputs.reserve(plan->input_indices.size());
    for (int index : plan->input_indices) {
      if (available_blocks[index] == nullptr) {
        return BRAID_STATUS_INVALID_ARGUMENT;
      }
      ordered_inputs.push_back(available_blocks[index]);
    }
    braid::kernels::ExecutePlan(
        plan->table,
        *plan,
        ordered_inputs.data(),
        recovered_blocks,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (const std::invalid_argument&) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

braid_status_t braid_encode_batch(
    braid_code_t* code,
    const uint8_t* const* const* data_stripes,
    uint8_t* const* const* parity_stripes,
    size_t stripe_count,
    size_t block_size) {
  if (code == nullptr || data_stripes == nullptr || parity_stripes == nullptr ||
      stripe_count == 0 || stripe_count > static_cast<size_t>(INT32_MAX) || block_size == 0 ||
      block_size > static_cast<size_t>(PTRDIFF_MAX) / stripe_count ||
      block_size > std::numeric_limits<size_t>::max() / stripe_count) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  bool outputs_aligned = block_size % 64 == 0;
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    if (data_stripes[stripe] == nullptr || parity_stripes[stripe] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
    for (int input = 0; input < code->impl->k(); ++input) {
      if (data_stripes[stripe][input] == nullptr) {
        return BRAID_STATUS_INVALID_ARGUMENT;
      }
    }
    for (int output = 0; output < code->impl->r(); ++output) {
      if (parity_stripes[stripe][output] == nullptr) {
        return BRAID_STATUS_INVALID_ARGUMENT;
      }
      outputs_aligned &=
          reinterpret_cast<uintptr_t>(
              parity_stripes[stripe][output]) % 64 == 0;
    }
  }
  try {
    auto plan = code->impl->GetOrBuildEncodeBatchPlan(block_size);
    if (plan->streaming_stores && !outputs_aligned) {
      plan = code->impl->GetOrBuildEncodePlan(block_size);
    }
    braid::kernels::ExecutePlanBatch(
        plan->table,
        *plan,
        data_stripes,
        parity_stripes,
        stripe_count,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

braid_status_t braid_decode_batch(
    braid_code_t* code,
    const uint8_t* const* const* available_stripes,
    const int* erased_indices,
    int erased_count,
    uint8_t* const* const* recovered_stripes,
    size_t stripe_count,
    size_t block_size) {
  if (code == nullptr || available_stripes == nullptr || erased_indices == nullptr ||
      recovered_stripes == nullptr || erased_count <= 0 ||
      erased_count > code->impl->r() || stripe_count == 0 || stripe_count > static_cast<size_t>(INT32_MAX) || block_size == 0 ||
      block_size > static_cast<size_t>(PTRDIFF_MAX) / stripe_count ||
      block_size > std::numeric_limits<size_t>::max() / stripe_count) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto plan = code->impl->GetOrBuildDecodeBatchPlan(
        std::vector<int>(erased_indices, erased_indices + erased_count),
        block_size);
    bool outputs_aligned = block_size % 64 == 0;
    for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
      if (available_stripes[stripe] == nullptr ||
          recovered_stripes[stripe] == nullptr) {
        return BRAID_STATUS_INVALID_ARGUMENT;
      }
      for (int input_index : plan->input_indices) {
        if (available_stripes[stripe][input_index] == nullptr) {
          return BRAID_STATUS_INVALID_ARGUMENT;
        }
      }
      for (int output = 0; output < erased_count; ++output) {
        if (recovered_stripes[stripe][output] == nullptr) {
          return BRAID_STATUS_INVALID_ARGUMENT;
        }
        outputs_aligned &=
            reinterpret_cast<uintptr_t>(
                recovered_stripes[stripe][output]) % 64 == 0;
      }
    }
    if (plan->streaming_stores && !outputs_aligned) {
      plan = code->impl->GetOrBuildDecodePlan(
          std::vector<int>(
              erased_indices, erased_indices + erased_count),
          block_size);
    }
    braid::kernels::ExecutePlanBatch(
        plan->table,
        *plan,
        available_stripes,
        recovered_stripes,
        stripe_count,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (const std::invalid_argument&) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

braid_plan_t* braid_build_encode_plan(braid_code_t* code) {
  return braid_build_encode_plan_for_block_size(code, 0);
}

braid_plan_t* braid_build_encode_plan_for_block_size(braid_code_t* code, size_t block_size) {
  if (code == nullptr) {
    return nullptr;
  }
  try {
    return new braid_plan(code->owner_token, code->impl->GetOrBuildEncodePlan(block_size));
  } catch (...) {
    return nullptr;
  }
}

braid_plan_t* braid_build_encode_batch_plan_for_block_size(
    braid_code_t* code, size_t block_size) {
  if (code == nullptr) {
    return nullptr;
  }
  try {
    return new braid_plan(
        code->owner_token,
        code->impl->GetOrBuildEncodeBatchPlan(block_size));
  } catch (...) {
    return nullptr;
  }
}

braid_plan_t* braid_build_decode_plan(braid_code_t* code, const int* erased_indices, int erased_count) {
  return braid_build_decode_plan_for_block_size(code, erased_indices, erased_count, 0);
}

braid_plan_t* braid_build_decode_plan_for_block_size(
    braid_code_t* code,
    const int* erased_indices,
    int erased_count,
    size_t block_size) {
  if (code == nullptr || erased_indices == nullptr || erased_count <= 0 || erased_count > code->impl->r()) {
    return nullptr;
  }
  try {
    return new braid_plan(
        code->owner_token,
        code->impl->GetOrBuildDecodePlan(std::vector<int>(erased_indices, erased_indices + erased_count), block_size));
  } catch (...) {
    return nullptr;
  }
}

braid_plan_t* braid_build_decode_batch_plan_for_block_size(
    braid_code_t* code,
    const int* erased_indices,
    int erased_count,
    size_t block_size) {
  if (code == nullptr || erased_indices == nullptr || erased_count <= 0 || erased_count > code->impl->r()) {
    return nullptr;
  }
  try {
    return new braid_plan(
        code->owner_token,
        code->impl->GetOrBuildDecodeBatchPlan(
            std::vector<int>(
                erased_indices, erased_indices + erased_count),
            block_size));
  } catch (...) {
    return nullptr;
  }
}

void braid_destroy_plan(braid_plan_t* plan) { delete plan; }

size_t braid_plan_tile_bytes(const braid_plan_t* plan) {
  return plan ? plan->impl->tile_bytes : 0;
}

int braid_plan_output_batch(const braid_plan_t* plan) {
  return plan ? plan->impl->output_batch : 0;
}

braid_backend_t braid_plan_backend(const braid_plan_t* plan) {
  return plan ? plan->impl->backend : BRAID_BACKEND_AUTO;
}

braid_path_t braid_plan_path(const braid_plan_t* plan) {
  return plan ? plan->impl->path : BRAID_PATH_AUTO;
}

braid_microkernel_mode_t braid_plan_microkernel_mode(const braid_plan_t* plan) {
  return plan ? plan->impl->microkernel_mode : BRAID_MICROKERNEL_BASELINE;
}

int braid_plan_uses_streaming_stores(const braid_plan_t* plan) {
  return plan && plan->impl->streaming_stores ? 1 : 0;
}

int braid_plan_repr_generator(const braid_plan_t* plan) {
  return plan ? plan->impl->table.generator : 0;
}

int braid_code_repr_generator(const braid_code_t* code) {
  return code ? code->impl->repr_generator() : 0;
}

size_t braid_plan_slp_xor_ops_before(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? plan->impl->avx512_jit->slp_xor_ops_before
             : 0;
}

size_t braid_plan_slp_xor_ops_after(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? plan->impl->avx512_jit->slp_xor_ops_after
             : 0;
}

size_t braid_plan_slp_potential_xor_ops(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? plan->impl->avx512_jit->slp_potential_xor_ops
             : 0;
}

int braid_plan_slp_groups(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? static_cast<int>(plan->impl->avx512_jit->slp_groups)
             : 0;
}

int braid_plan_slp_potential_groups(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? static_cast<int>(
                   plan->impl->avx512_jit->slp_potential_groups)
             : 0;
}

int braid_plan_slp_triple_groups(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? static_cast<int>(
                   plan->impl->avx512_jit->slp_triple_groups)
             : 0;
}

size_t braid_plan_gfni_prefetch_distance(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? plan->impl->avx512_jit->prefetch_distance
             : 0;
}

int braid_plan_gfni_prefetch_hint(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? static_cast<int>(plan->impl->avx512_jit->prefetch_hint)
             : 0;
}

size_t braid_plan_gfni_prefetch_input_stride(const braid_plan_t* plan) {
  return plan && plan->impl->avx512_jit
             ? plan->impl->avx512_jit->prefetch_input_stride
             : 0;
}

braid_status_t braid_execute_encode_plan(
    braid_code_t* code,
    braid_plan_t* plan,
    const uint8_t* const* data_blocks,
    uint8_t* const* output_blocks,
    size_t block_size) {
  if (block_size > static_cast<size_t>(PTRDIFF_MAX) || code == nullptr || plan == nullptr || data_blocks == nullptr || output_blocks == nullptr) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  if (plan->owner != code->owner_token || plan->impl->type != braid::runtime::PlanType::kEncode ||
      (plan->impl->planned_block_size != 0 && plan->impl->planned_block_size != block_size)) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  for (size_t input = 0; input < plan->impl->input_indices.size(); ++input) {
    if (data_blocks[input] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  for (size_t output = 0; output < plan->impl->rows.size(); ++output) {
    if (output_blocks[output] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    braid::kernels::ExecutePlan(
        plan->impl->table,
        *plan->impl,
        data_blocks,
        output_blocks,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

braid_status_t braid_execute_decode_plan(
    braid_code_t* code,
    braid_plan_t* plan,
    const uint8_t* const* input_blocks,
    uint8_t* const* output_blocks,
    size_t block_size) {
  if (block_size > static_cast<size_t>(PTRDIFF_MAX) || code == nullptr || plan == nullptr || input_blocks == nullptr || output_blocks == nullptr) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  if (plan->owner != code->owner_token || plan->impl->type != braid::runtime::PlanType::kDecode ||
      (plan->impl->planned_block_size != 0 && plan->impl->planned_block_size != block_size)) {
    return BRAID_STATUS_INVALID_ARGUMENT;
  }
  for (size_t output = 0; output < plan->impl->rows.size(); ++output) {
    if (output_blocks[output] == nullptr) {
      return BRAID_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    std::vector<const uint8_t*> ordered_inputs;
    ordered_inputs.reserve(plan->impl->input_indices.size());
    for (int index : plan->impl->input_indices) {
      if (input_blocks[index] == nullptr) {
        return BRAID_STATUS_INVALID_ARGUMENT;
      }
      ordered_inputs.push_back(input_blocks[index]);
    }
    braid::kernels::ExecutePlan(
        plan->impl->table,
        *plan->impl,
        ordered_inputs.data(),
        output_blocks,
        block_size,
        code->impl->options().num_threads,
        code->impl->thread_pool());
    return BRAID_STATUS_OK;
  } catch (...) {
    return BRAID_STATUS_INTERNAL_ERROR;
  }
}

}  // extern "C"
