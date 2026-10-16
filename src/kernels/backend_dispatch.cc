#include "src/kernels/backend_dispatch.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <thread>
#include <vector>
#include <stdexcept>

#include "src/kernels/grape_scalar.h"
#include "src/kernels/vector_entrypoints.h"


namespace braid::kernels {

namespace {

inline void StreamingStoreFence() {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ volatile("sfence" ::: "memory");
#endif
}

size_t SizeEnvironmentOverride(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
    return fallback;
  }
  return static_cast<size_t>(parsed);
}

void ExecutePlanSerial(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  if (plan.path != BRAID_PATH_BRAID) {
    throw std::invalid_argument("only factorized execution is available");
  }
  // Complete this worker's non-temporal stores before publishing completion.
  struct StoreCompletion {
    bool streaming;
    ~StoreCompletion() { if (streaming) StreamingStoreFence(); }
  } completion{plan.streaming_stores};
  switch (plan.backend) {
#if defined(BRAID_HAVE_AVX512)
    case BRAID_BACKEND_AVX512:
#if defined(BRAID_HAVE_AVX512_GFNI)
      if (plan.gfni_object) {
        ExecuteBraidAVX512GFNI(table, plan, inputs, outputs, block_size);
        return;
      }
#endif
      ExecuteBraidAVX512(table, plan, inputs, outputs, block_size);
      return;
#endif
#if defined(BRAID_HAVE_AVX2)
    case BRAID_BACKEND_AVX2:
      ExecuteBraidAVX2(table, plan, inputs, outputs, block_size);
      return;
#endif
    case BRAID_BACKEND_SCALAR:
    case BRAID_BACKEND_AUTO:
    default:
      ExecuteBraidScalar(table, plan, inputs, outputs, block_size);
      return;
  }
}

}  // namespace

void ExecutePlan(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size,
    int num_threads,
    braid::runtime::ThreadPool* thread_pool) {
  const int requested_threads = std::max(num_threads, 1);
  size_t min_chunk = 256 * 1024;
  if (plan.path == BRAID_PATH_BRAID) {
    if (plan.backend == BRAID_BACKEND_AVX512) {
      if (block_size <= (1u << 20)) {
        min_chunk = 16 * 1024;
      } else if (plan.input_indices.size() <= 16 && plan.rows.size() >= 4) {
        min_chunk = 48 * 1024;
      } else {
        min_chunk = plan.rows.size() <= 4 ? 64 * 1024 : 128 * 1024;
      }
    } else if (plan.backend == BRAID_BACKEND_AVX2) {
      if (plan.type == braid::runtime::PlanType::kEncode &&
          plan.input_indices.size() >= 64 &&
          plan.rows.size() >= 4 &&
          block_size <= (1u << 20)) {
        min_chunk = 16 * 1024;
      } else if (plan.type == braid::runtime::PlanType::kDecode &&
                 plan.input_indices.size() >= 64 &&
                 plan.rows.size() >= 4 &&
                 block_size <= (4u << 20)) {
        min_chunk = 32 * 1024;
      } else {
        min_chunk = plan.rows.size() <= 4 ? 64 * 1024 : 128 * 1024;
      }
    } else {
      min_chunk = 128 * 1024;
    }
    const size_t tile_factor =
        (plan.backend == BRAID_BACKEND_AVX512 && plan.input_indices.size() <= 16 && plan.rows.size() >= 4) ? 1 : 2;
    min_chunk = std::max<size_t>(plan.tile_bytes * tile_factor, min_chunk);
  } else {
    min_chunk = std::max<size_t>(plan.tile_bytes * 4, min_chunk);
  }
  if (plan.microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT_GFNI) {
    // Mid-width 512 KiB cells carry enough coefficient work per 16 KiB
    // slice to use the full worker set.  The generic tile-derived floor
    // otherwise underfills the machine for this HDFS-sized regime.
    if (block_size <= (512u << 10) &&
        plan.input_indices.size() >= 40 &&
        plan.input_indices.size() <= 48) {
      min_chunk = 16 * 1024;
    }
    min_chunk = SizeEnvironmentOverride(
        "BRAID_GFNI_MIN_CHUNK", min_chunk);
  }
  int threads = requested_threads;
  if (plan.path == BRAID_PATH_BRAID &&
      plan.backend == BRAID_BACKEND_AVX512 &&
      plan.input_indices.size() >= 64 &&
      plan.rows.size() >= 4 &&
      block_size <= (1u << 20)) {
    threads = std::min(threads, 20);
  }
  if (block_size < min_chunk || requested_threads == 1) {
    threads = 1;
  } else {
    const size_t max_threads_by_work = (block_size + min_chunk - 1) / min_chunk;
    threads = std::min<int>(requested_threads, static_cast<int>(std::max<size_t>(1, max_threads_by_work)));
  }

  if (thread_pool == nullptr) {
    threads = 1;
  } else {
    threads = std::min(threads, thread_pool->capacity());
  }

  if (threads <= 1) {
    ExecutePlanSerial(table, plan, inputs, outputs, block_size);
    return;
  }

  size_t chunk =
      (block_size + static_cast<size_t>(threads) - 1) /
      static_cast<size_t>(threads);
  if (plan.streaming_stores) {
    chunk = ((chunk + 63) / 64) * 64;
  }

  auto execute_for_thread = [&](int tid) {
      thread_local std::vector<const uint8_t*> local_inputs;
      thread_local std::vector<uint8_t*> local_outputs;
      const size_t offset = static_cast<size_t>(tid) * chunk;
      if (offset >= block_size) {
        return;
      }
      const size_t local_size = std::min(chunk, block_size - offset);
      local_inputs.resize(plan.input_indices.size());
      local_outputs.resize(plan.rows.size());
      for (size_t i = 0; i < plan.input_indices.size(); ++i) {
        local_inputs[static_cast<size_t>(i)] = inputs[i] + offset;
      }
      for (size_t o = 0; o < plan.rows.size(); ++o) {
        local_outputs[static_cast<size_t>(o)] = outputs[o] + offset;
      }
      ExecutePlanSerial(table, plan, local_inputs.data(), local_outputs.data(), local_size);
  };

  const bool spin_wait = block_size <= (1u << 20);
  thread_pool->Run(threads, execute_for_thread, spin_wait);
}

void ExecutePlanBatch(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* const* input_stripes,
    uint8_t* const* const* output_stripes,
    size_t stripe_count,
    size_t block_size,
    int num_threads,
    braid::runtime::ThreadPool* thread_pool) {
  if (stripe_count == 0) {
    return;
  }
  const int requested_threads = std::max(num_threads, 1);
  int threads = std::min<int>(requested_threads, static_cast<int>(stripe_count));
  if (thread_pool == nullptr) {
    threads = 1;
  } else {
    threads = std::min(threads, thread_pool->capacity());
  }

  const auto execute_range = [&](int thread_id, int worker_count) {
    thread_local std::vector<const uint8_t*> local_inputs;
    thread_local std::vector<uint8_t*> local_outputs;
    local_inputs.resize(plan.input_indices.size());
    local_outputs.resize(plan.rows.size());
    const size_t total_bytes = stripe_count * block_size;
    const size_t vector_count = total_bytes / 64;
    size_t position =
        vector_count * static_cast<size_t>(thread_id) /
        static_cast<size_t>(worker_count) * 64;
    const size_t end =
        thread_id + 1 == worker_count
            ? total_bytes
            : vector_count * static_cast<size_t>(thread_id + 1) /
                  static_cast<size_t>(worker_count) * 64;
    while (position < end) {
      const size_t stripe = position / block_size;
      const size_t stripe_offset = position % block_size;
      const size_t local_size =
          std::min(end - position, block_size - stripe_offset);
      for (size_t input = 0; input < local_inputs.size(); ++input) {
        local_inputs[input] =
            input_stripes[stripe]
                         [static_cast<size_t>(plan.input_indices[input])] +
            stripe_offset;
      }
      for (size_t output = 0; output < local_outputs.size(); ++output) {
        local_outputs[output] =
            output_stripes[stripe][output] + stripe_offset;
      }
      ExecutePlanSerial(
          table,
          plan,
          local_inputs.data(),
          local_outputs.data(),
          local_size);
      position += local_size;
    }
    if (plan.streaming_stores) {
      StreamingStoreFence();
    }
  };

  if (threads <= 1) {
    execute_range(0, 1);
    return;
  }
  thread_pool->Run(
      threads,
      [&](int thread_id) { execute_range(thread_id, threads); },
      true);
}

}  // namespace braid::kernels
