#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include "src/kernels/vector_common.h"
#include "src/kernels/vector_entrypoints.h"

namespace braid::kernels {

namespace {

uint8_t* AcquireThreadScratch256(size_t bytes) {
  struct AlignedScratch {
    uint8_t* data = nullptr;
    size_t size = 0;

    ~AlignedScratch() {
      if (data != nullptr) {
        ::operator delete[](data, std::align_val_t(32));
      }
    }
  };

  thread_local AlignedScratch scratch;
  if (scratch.size < bytes) {
    auto* replacement = static_cast<uint8_t*>(::operator new[](bytes, std::align_val_t(32)));
    if (scratch.data != nullptr) {
      ::operator delete[](scratch.data, std::align_val_t(32));
    }
    scratch.data = replacement;
    scratch.size = bytes;
  }
  return scratch.data;
}

inline void XorScratchVector256(uint8_t* dst_ptr, __m256i value) {
  const __m256i dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst_ptr));
  _mm256_store_si256(reinterpret_cast<__m256i*>(dst_ptr), _mm256_xor_si256(dst, value));
}

template <size_t Batch>
inline void XorAccumulators256(uint8_t mask, __m256i value, std::array<__m256i, Batch>& accs) {
  if constexpr (Batch == 4) {
    switch (mask & 0x0f) {
      case 0x0: return;
      case 0x1: accs[0] = _mm256_xor_si256(accs[0], value); return;
      case 0x2: accs[1] = _mm256_xor_si256(accs[1], value); return;
      case 0x3:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[1] = _mm256_xor_si256(accs[1], value);
        return;
      case 0x4: accs[2] = _mm256_xor_si256(accs[2], value); return;
      case 0x5:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        return;
      case 0x6:
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        return;
      case 0x7:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        return;
      case 0x8: accs[3] = _mm256_xor_si256(accs[3], value); return;
      case 0x9:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xA:
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xB:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xC:
        accs[2] = _mm256_xor_si256(accs[2], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xD:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xE:
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
      case 0xF:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[1] = _mm256_xor_si256(accs[1], value);
        accs[2] = _mm256_xor_si256(accs[2], value);
        accs[3] = _mm256_xor_si256(accs[3], value);
        return;
    }
  } else if constexpr (Batch == 2) {
    switch (mask & 0x03) {
      case 0x0: return;
      case 0x1: accs[0] = _mm256_xor_si256(accs[0], value); return;
      case 0x2: accs[1] = _mm256_xor_si256(accs[1], value); return;
      case 0x3:
        accs[0] = _mm256_xor_si256(accs[0], value);
        accs[1] = _mm256_xor_si256(accs[1], value);
        return;
    }
  } else {
    for (size_t idx = 0; idx < Batch; ++idx) {
      if (mask & (1u << idx)) {
        accs[idx] = _mm256_xor_si256(accs[idx], value);
      }
    }
  }
}

inline __m256i MultiplyXPowAvx2(
    const braid::core::DecompositionTable& table, __m256i value, uint8_t power) {
  if (power == 0) {
    return value;
  }
  const uint8_t x_exponent = table.x_exponents[power];
  __m256i out = value;
  if (x_exponent <= 4) {
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = XtimeAvx2(out);
    }
    return out;
  }
  return MultiplyConstAvx2(value, table.x_powers[power]);
}

inline __m256i AdvanceBetweenPlanesAvx2(
    const braid::core::DecompositionTable& table, __m256i value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[from_plane][to_plane];
  if (x_exponent == 0) {
    return value;
  }
  __m256i out = value;
  if (x_exponent <= 4) {
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = XtimeAvx2(out);
    }
    return out;
  }
  return MultiplyConstAvx2(value, table.transition_coeffs[from_plane][to_plane]);
}

inline uint8_t AdvanceByGapScalar(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t gap) {
  if (gap == 0) {
    return value;
  }
  const uint8_t x_exponent = table.x_exponents[gap];
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.x_powers[gap], value);
}

inline uint8_t AdvanceBetweenPlanesScalar(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[from_plane][to_plane];
  if (x_exponent == 0) {
    return value;
  }
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.transition_coeffs[from_plane][to_plane], value);
}

inline __m256i AdvanceDirectAvx2(
    const braid::core::DecompositionTable& table, __m256i value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[to_plane][from_plane];
  if (x_exponent == 0) {
    return value;
  }
  __m256i out = value;
  if (x_exponent <= 4) {
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = XtimeAvx2(out);
    }
    return out;
  }
  return MultiplyConstAvx2(value, table.transition_coeffs[to_plane][from_plane]);
}

inline uint8_t AdvanceDirectScalar256(
    const braid::core::DecompositionTable& table, uint8_t value, uint8_t from_plane, uint8_t to_plane) {
  const uint8_t x_exponent = table.transition_x_exponents[to_plane][from_plane];
  if (x_exponent == 0) {
    return value;
  }
  if (x_exponent <= 4) {
    uint8_t out = value;
    for (uint8_t step = 0; step < x_exponent; ++step) {
      out = braid::core::GF256::MultiplyByX(out);
    }
    return out;
  }
  return braid::core::GF256::Multiply(table.transition_coeffs[to_plane][from_plane], value);
}

void ExecuteBraidPairDirectScheduled256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  constexpr size_t kMaxSteps = 6;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t input_count = plan.input_indices.size();
  const size_t pair_outputs = std::min<size_t>(2, plan.rows.size() - base_out);

  struct Step {
    uint8_t pos = 0;
    uint8_t mask = 0;
  };
  struct Schedule {
    uint8_t count = 0;
    std::array<Step, kMaxSteps> steps{};
  };

  std::vector<Schedule> schedules(input_count);
  for (size_t in = 0; in < input_count; ++in) {
    std::array<uint8_t, 17> plane_masks{};
    for (size_t local = 0; local < pair_outputs; ++local) {
      const auto& row = plan.rows[base_out + local];
      const uint8_t weight = row.alpha_weights[in];
      if (weight == 0) {
        continue;
      }
      const auto& positions = row.alpha_positions[in];
      plane_masks[positions[0]] |= static_cast<uint8_t>(1u << local);
      if (weight >= 2) {
        plane_masks[positions[1]] |= static_cast<uint8_t>(1u << local);
      }
      if (weight >= 3) {
        plane_masks[positions[2]] |= static_cast<uint8_t>(1u << local);
      }
    }
    auto& schedule = schedules[in];
    for (uint8_t plane = 0; plane < 17; ++plane) {
      if (plane_masks[plane] != 0) {
        schedule.steps[schedule.count++] = Step{plane, plane_masks[plane]};
      }
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      __m256i acc0 = _mm256_setzero_si256();
      __m256i acc1 = _mm256_setzero_si256();
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        const __m256i input_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs[in] + offset + pos));
        __m256i current = AdvanceDirectAvx2(table, input_value, 0, schedule.steps[0].pos);
        if (schedule.steps[0].mask & 0x1) {
          acc0 = _mm256_xor_si256(acc0, current);
        }
        if (schedule.steps[0].mask & 0x2) {
          acc1 = _mm256_xor_si256(acc1, current);
        }
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectAvx2(table, current, last_pos, next_pos);
          if (schedule.steps[step].mask & 0x1) {
            acc0 = _mm256_xor_si256(acc0, current);
          }
          if (schedule.steps[step].mask & 0x2) {
            acc1 = _mm256_xor_si256(acc1, current);
          }
          last_pos = next_pos;
        }
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out] + offset + pos), acc0);
      if (pair_outputs > 1) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out + 1] + offset + pos), acc1);
      }
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      uint8_t acc0 = 0;
      uint8_t acc1 = 0;
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        uint8_t current = AdvanceDirectScalar256(table, inputs[in][offset + pos], 0, schedule.steps[0].pos);
        if (schedule.steps[0].mask & 0x1) {
          acc0 ^= current;
        }
        if (schedule.steps[0].mask & 0x2) {
          acc1 ^= current;
        }
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectScalar256(table, current, last_pos, next_pos);
          if (schedule.steps[step].mask & 0x1) {
            acc0 ^= current;
          }
          if (schedule.steps[step].mask & 0x2) {
            acc1 ^= current;
          }
          last_pos = next_pos;
        }
      }
      outputs[base_out][offset + pos] = acc0;
      if (pair_outputs > 1) {
        outputs[base_out + 1][offset + pos] = acc1;
      }
    }
  }
}

void ExecuteBraidSingleDirectScheduled256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  constexpr size_t kMaxSteps = 3;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t input_count = plan.input_indices.size();
  const auto& row = plan.rows[base_out];

  struct Schedule {
    uint8_t count = 0;
    std::array<uint8_t, kMaxSteps> positions{};
  };

  std::vector<Schedule> schedules(input_count);
  for (size_t in = 0; in < input_count; ++in) {
    const uint8_t weight = row.alpha_weights[in];
    if (weight == 0) {
      continue;
    }
    auto& schedule = schedules[in];
    schedule.count = weight;
    const auto& positions = row.alpha_positions[in];
    for (uint8_t step = 0; step < weight; ++step) {
      schedule.positions[step] = positions[step];
    }
    std::sort(schedule.positions.begin(), schedule.positions.begin() + weight);
  }

  uint8_t* out = outputs[base_out];
  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      __m256i acc = _mm256_setzero_si256();
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        const __m256i input_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs[in] + offset + pos));
        __m256i current = AdvanceDirectAvx2(table, input_value, 0, schedule.positions[0]);
        acc = _mm256_xor_si256(acc, current);
        if (schedule.count >= 2) {
          current = AdvanceDirectAvx2(table, current, schedule.positions[0], schedule.positions[1]);
          acc = _mm256_xor_si256(acc, current);
        }
        if (schedule.count >= 3) {
          current = AdvanceDirectAvx2(table, current, schedule.positions[1], schedule.positions[2]);
          acc = _mm256_xor_si256(acc, current);
        }
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + offset + pos), acc);
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      uint8_t acc = 0;
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        uint8_t current = AdvanceDirectScalar256(table, inputs[in][offset + pos], 0, schedule.positions[0]);
        acc ^= current;
        if (schedule.count >= 2) {
          current = AdvanceDirectScalar256(table, current, schedule.positions[0], schedule.positions[1]);
          acc ^= current;
        }
        if (schedule.count >= 3) {
          current = AdvanceDirectScalar256(table, current, schedule.positions[1], schedule.positions[2]);
          acc ^= current;
        }
      }
      out[offset + pos] = acc;
    }
  }
}

void ExecuteBraidQuadDirectScheduled256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  constexpr size_t kMaxSteps = 12;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const size_t input_count = plan.input_indices.size();
  const size_t quad_outputs = std::min<size_t>(4, plan.rows.size() - base_out);

  struct Step {
    uint8_t pos = 0;
    uint8_t mask = 0;
  };
  struct Schedule {
    uint8_t count = 0;
    std::array<Step, kMaxSteps> steps{};
  };

  std::vector<Schedule> schedules(input_count);
  for (size_t in = 0; in < input_count; ++in) {
    std::array<uint8_t, 17> plane_masks{};
    for (size_t local = 0; local < quad_outputs; ++local) {
      const auto& row = plan.rows[base_out + local];
      const uint8_t weight = row.alpha_weights[in];
      if (weight == 0) {
        continue;
      }
      const auto& positions = row.alpha_positions[in];
      plane_masks[positions[0]] |= static_cast<uint8_t>(1u << local);
      if (weight >= 2) {
        plane_masks[positions[1]] |= static_cast<uint8_t>(1u << local);
      }
      if (weight >= 3) {
        plane_masks[positions[2]] |= static_cast<uint8_t>(1u << local);
      }
    }
    auto& schedule = schedules[in];
    for (uint8_t plane = 0; plane < 17; ++plane) {
      if (plane_masks[plane] != 0) {
        schedule.steps[schedule.count++] = Step{plane, plane_masks[plane]};
      }
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m256i, 4> accs{
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256()};
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        if (in + 8 < input_count) {
          __builtin_prefetch(inputs[in + 8] + offset + pos, 0, 1);
        }
        const __m256i input_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs[in] + offset + pos));
        __m256i current = AdvanceDirectAvx2(table, input_value, 0, schedule.steps[0].pos);
        XorAccumulators256<4>(schedule.steps[0].mask, current, accs);
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectAvx2(table, current, last_pos, next_pos);
          XorAccumulators256<4>(schedule.steps[step].mask, current, accs);
          last_pos = next_pos;
        }
      }
      for (size_t local = 0; local < quad_outputs; ++local) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out + local] + offset + pos), accs[local]);
      }
    }

    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, 4> accs{};
      for (size_t in = 0; in < input_count; ++in) {
        const auto& schedule = schedules[in];
        if (schedule.count == 0) {
          continue;
        }
        uint8_t current = AdvanceDirectScalar256(table, inputs[in][offset + pos], 0, schedule.steps[0].pos);
        for (size_t local = 0; local < quad_outputs; ++local) {
          if (schedule.steps[0].mask & (1u << local)) {
            accs[local] ^= current;
          }
        }
        uint8_t last_pos = schedule.steps[0].pos;
        for (uint8_t step = 1; step < schedule.count; ++step) {
          const uint8_t next_pos = schedule.steps[step].pos;
          current = AdvanceDirectScalar256(table, current, last_pos, next_pos);
          for (size_t local = 0; local < quad_outputs; ++local) {
            if (schedule.steps[step].mask & (1u << local)) {
              accs[local] ^= current;
            }
          }
          last_pos = next_pos;
        }
      }
      for (size_t local = 0; local < quad_outputs; ++local) {
        outputs[base_out + local][offset + pos] = accs[local];
      }
    }
  }
}









inline void ReduceTwoSparseRows256(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 34>& plane_ptrs,
    const std::array<std::array<uint8_t, 17>, 2>& used_planes,
    const std::array<uint8_t, 2>& used_plane_counts,
    size_t pos,
    std::array<__m256i, 2>& acc) {
  std::array<int, 2> next_idx{
      static_cast<int>(used_plane_counts[0]) - 2,
      static_cast<int>(used_plane_counts[1]) - 2,
  };
  std::array<uint8_t, 2> current_planes{};
  for (size_t b = 0; b < 2; ++b) {
    if (used_plane_counts[b] == 0) {
      current_planes[b] = 0;
      continue;
    }
    current_planes[b] = used_planes[b][used_plane_counts[b] - 1];
    acc[b] = _mm256_xor_si256(
        acc[b],
        _mm256_load_si256(
            reinterpret_cast<const __m256i*>(plane_ptrs[b * 17 + static_cast<size_t>(current_planes[b])] + pos)));
  }
  while (next_idx[0] >= 0 || next_idx[1] >= 0) {
    if (next_idx[0] >= 0) {
      const uint8_t next_plane = used_planes[0][static_cast<size_t>(next_idx[0]--)];
      acc[0] = AdvanceBetweenPlanesAvx2(table, acc[0], current_planes[0], next_plane);
      acc[0] = _mm256_xor_si256(
          acc[0],
          _mm256_load_si256(reinterpret_cast<const __m256i*>(plane_ptrs[static_cast<size_t>(next_plane)] + pos)));
      current_planes[0] = next_plane;
    }
    if (next_idx[1] >= 0) {
      const uint8_t next_plane = used_planes[1][static_cast<size_t>(next_idx[1]--)];
      acc[1] = AdvanceBetweenPlanesAvx2(table, acc[1], current_planes[1], next_plane);
      acc[1] = _mm256_xor_si256(
          acc[1],
          _mm256_load_si256(reinterpret_cast<const __m256i*>(
              plane_ptrs[17 + static_cast<size_t>(next_plane)] + pos)));
      current_planes[1] = next_plane;
    }
  }
  if (used_plane_counts[0] != 0 && current_planes[0] != 0) {
    acc[0] = AdvanceBetweenPlanesAvx2(table, acc[0], current_planes[0], 0);
  }
  if (used_plane_counts[1] != 0 && current_planes[1] != 0) {
    acc[1] = AdvanceBetweenPlanesAvx2(table, acc[1], current_planes[1], 0);
  }
}

inline void ReduceTwoSparseRowsScalar256(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 34>& plane_ptrs,
    const std::array<std::array<uint8_t, 17>, 2>& used_planes,
    const std::array<uint8_t, 2>& used_plane_counts,
    size_t pos,
    std::array<uint8_t, 2>& acc) {
  std::array<int, 2> next_idx{
      static_cast<int>(used_plane_counts[0]) - 2,
      static_cast<int>(used_plane_counts[1]) - 2,
  };
  std::array<uint8_t, 2> current_planes{};
  for (size_t b = 0; b < 2; ++b) {
    if (used_plane_counts[b] == 0) {
      current_planes[b] = 0;
      continue;
    }
    current_planes[b] = used_planes[b][used_plane_counts[b] - 1];
    acc[b] ^= plane_ptrs[b * 17 + static_cast<size_t>(current_planes[b])][pos];
  }
  while (next_idx[0] >= 0 || next_idx[1] >= 0) {
    if (next_idx[0] >= 0) {
      const uint8_t next_plane = used_planes[0][static_cast<size_t>(next_idx[0]--)];
      acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_planes[0], next_plane);
      acc[0] ^= plane_ptrs[static_cast<size_t>(next_plane)][pos];
      current_planes[0] = next_plane;
    }
    if (next_idx[1] >= 0) {
      const uint8_t next_plane = used_planes[1][static_cast<size_t>(next_idx[1]--)];
      acc[1] = AdvanceBetweenPlanesScalar(table, acc[1], current_planes[1], next_plane);
      acc[1] ^= plane_ptrs[17 + static_cast<size_t>(next_plane)][pos];
      current_planes[1] = next_plane;
    }
  }
  if (used_plane_counts[0] != 0 && current_planes[0] != 0) {
    acc[0] = AdvanceBetweenPlanesScalar(table, acc[0], current_planes[0], 0);
  }
  if (used_plane_counts[1] != 0 && current_planes[1] != 0) {
    acc[1] = AdvanceBetweenPlanesScalar(table, acc[1], current_planes[1], 0);
  }
}

inline void ReduceSparseRow256(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 17>& plane_ptrs,
    const std::array<uint8_t, 17>& used_planes,
    uint8_t used_plane_count,
    size_t pos,
    __m256i& acc) {
  if (used_plane_count == 0) {
    return;
  }
  uint8_t current_plane = used_planes[used_plane_count - 1];
  acc = _mm256_xor_si256(
      acc,
      _mm256_load_si256(reinterpret_cast<const __m256i*>(
          plane_ptrs[static_cast<size_t>(current_plane)] + pos)));
  for (int idx = static_cast<int>(used_plane_count) - 2; idx >= 0; --idx) {
    const uint8_t next_plane = used_planes[static_cast<size_t>(idx)];
    acc = AdvanceBetweenPlanesAvx2(table, acc, current_plane, next_plane);
    acc = _mm256_xor_si256(
        acc,
        _mm256_load_si256(reinterpret_cast<const __m256i*>(
            plane_ptrs[static_cast<size_t>(next_plane)] + pos)));
    current_plane = next_plane;
  }
  if (current_plane != 0) {
    acc = AdvanceBetweenPlanesAvx2(table, acc, current_plane, 0);
  }
}

inline void ReduceSparseRowScalar256(
    const braid::core::DecompositionTable& table,
    const std::array<uint8_t*, 17>& plane_ptrs,
    const std::array<uint8_t, 17>& used_planes,
    uint8_t used_plane_count,
    size_t pos,
    uint8_t& acc) {
  if (used_plane_count == 0) {
    return;
  }
  uint8_t current_plane = used_planes[used_plane_count - 1];
  acc ^= plane_ptrs[static_cast<size_t>(current_plane)][pos];
  for (int idx = static_cast<int>(used_plane_count) - 2; idx >= 0; --idx) {
    const uint8_t next_plane = used_planes[static_cast<size_t>(idx)];
    acc = AdvanceBetweenPlanesScalar(table, acc, current_plane, next_plane);
    acc ^= plane_ptrs[static_cast<size_t>(next_plane)][pos];
    current_plane = next_plane;
  }
  if (current_plane != 0) {
    acc = AdvanceBetweenPlanesScalar(table, acc, current_plane, 0);
  }
}

void ExecuteBraidFixedSingle256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t out_idx,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  uint8_t* scratch = AcquireThreadScratch256(17 * tile);

  const auto* row = &plan.rows[out_idx];
  uint32_t used_plane_mask = 0;
  std::array<uint8_t, 17> used_planes{};
  uint8_t used_plane_count = 0;

  struct TargetList {
    uint8_t count = 0;
    std::array<uint8_t, 3> ids{};
  };
  std::vector<TargetList> target_lists(plan.input_indices.size());
  for (size_t in = 0; in < plan.input_indices.size(); ++in) {
    const uint8_t weight = row->alpha_weights[in];
    if (weight == 0) {
      continue;
    }
    const auto& positions = row->alpha_positions[in];
    auto& targets = target_lists[in];
    targets.ids[targets.count++] = positions[0];
    used_plane_mask |= (1u << positions[0]);
    if (weight >= 2) {
      targets.ids[targets.count++] = positions[1];
      used_plane_mask |= (1u << positions[1]);
    }
    if (weight >= 3) {
      targets.ids[targets.count++] = positions[2];
      used_plane_mask |= (1u << positions[2]);
    }
  }

  uint32_t mask = used_plane_mask;
  while (mask != 0) {
    used_planes[used_plane_count++] = static_cast<uint8_t>(__builtin_ctz(mask));
    mask &= (mask - 1);
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    std::array<uint8_t*, 17> plane_ptrs{};
    for (size_t idx = 0; idx < plane_ptrs.size(); ++idx) {
      plane_ptrs[idx] = scratch + idx * tile;
    }
    mask = used_plane_mask;
    while (mask != 0) {
      const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(mask));
      std::memset(plane_ptrs[plane], 0, chunk);
      mask &= (mask - 1);
    }

    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t in = 0; in < plan.input_indices.size(); ++in) {
      const auto& targets = target_lists[in];
      if (targets.count == 0) {
        continue;
      }
      const uint8_t* src = inputs[in] + offset;
      for (size_t pos = 0; pos < vec_limit; pos += kVec) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pos));
        switch (targets.count) {
          case 3: XorScratchVector256(plane_ptrs[targets.ids[2]] + pos, value); [[fallthrough]];
          case 2: XorScratchVector256(plane_ptrs[targets.ids[1]] + pos, value); [[fallthrough]];
          case 1: XorScratchVector256(plane_ptrs[targets.ids[0]] + pos, value); break;
          default: break;
        }
      }
      switch (targets.count) {
        case 3: XorBytesScalar(plane_ptrs[targets.ids[2]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 2: XorBytesScalar(plane_ptrs[targets.ids[1]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 1: XorBytesScalar(plane_ptrs[targets.ids[0]] + vec_limit, src + vec_limit, chunk - vec_limit); break;
        default: break;
      }
    }

    uint8_t* out = outputs[out_idx] + offset;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      __m256i acc = _mm256_setzero_si256();
      ReduceSparseRow256(table, plane_ptrs, used_planes, used_plane_count, pos, acc);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + pos), acc);
    }
    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      uint8_t acc = 0;
      ReduceSparseRowScalar256(table, plane_ptrs, used_planes, used_plane_count, pos, acc);
      out[pos] = acc;
    }
  }
}

void ExecuteBraidFixedPair256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  uint8_t* scratch = AcquireThreadScratch256(2 * 17 * tile);
  constexpr size_t kMaxTargets = 6;

  const auto* row0 = &plan.rows[base_out];
  const auto* row1 = &plan.rows[base_out + 1];
  std::array<uint32_t, 2> used_plane_masks{};
  std::array<std::array<uint8_t, 17>, 2> used_planes{};
  std::array<uint8_t, 2> used_plane_counts{};

  struct TargetList {
    uint8_t count = 0;
    std::array<uint8_t, kMaxTargets> ids{};
  };
  std::vector<TargetList> target_lists(plan.input_indices.size());
  for (size_t in = 0; in < plan.input_indices.size(); ++in) {
    auto& targets = target_lists[in];
    const uint8_t w0 = row0->alpha_weights[in];
    if (w0 != 0) {
      const auto& positions = row0->alpha_positions[in];
      targets.ids[targets.count++] = positions[0];
      used_plane_masks[0] |= (1u << positions[0]);
      if (w0 >= 2) {
        targets.ids[targets.count++] = positions[1];
        used_plane_masks[0] |= (1u << positions[1]);
      }
      if (w0 >= 3) {
        targets.ids[targets.count++] = positions[2];
        used_plane_masks[0] |= (1u << positions[2]);
      }
    }
    const uint8_t w1 = row1->alpha_weights[in];
    if (w1 != 0) {
      const auto& positions = row1->alpha_positions[in];
      targets.ids[targets.count++] = static_cast<uint8_t>(17 + positions[0]);
      used_plane_masks[1] |= (1u << positions[0]);
      if (w1 >= 2) {
        targets.ids[targets.count++] = static_cast<uint8_t>(17 + positions[1]);
        used_plane_masks[1] |= (1u << positions[1]);
      }
      if (w1 >= 3) {
        targets.ids[targets.count++] = static_cast<uint8_t>(17 + positions[2]);
        used_plane_masks[1] |= (1u << positions[2]);
      }
    }
  }

  for (size_t b = 0; b < 2; ++b) {
    uint32_t mask = used_plane_masks[b];
    while (mask != 0) {
      used_planes[b][used_plane_counts[b]++] = static_cast<uint8_t>(__builtin_ctz(mask));
      mask &= (mask - 1);
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    std::array<uint8_t*, 34> plane_ptrs{};
    for (size_t idx = 0; idx < plane_ptrs.size(); ++idx) {
      plane_ptrs[idx] = scratch + idx * tile;
    }

    for (size_t b = 0; b < 2; ++b) {
      uint32_t mask = used_plane_masks[b];
      while (mask != 0) {
        const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(mask));
        std::memset(plane_ptrs[b * 17 + plane], 0, chunk);
        mask &= (mask - 1);
      }
    }

    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t in = 0; in < plan.input_indices.size(); ++in) {
      const auto& targets = target_lists[in];
      if (targets.count == 0) {
        continue;
      }
      const uint8_t* src = inputs[in] + offset;
      for (size_t pos = 0; pos < vec_limit; pos += kVec) {
        if (in + 8 < plan.input_indices.size()) {
          __builtin_prefetch(inputs[in + 8] + offset + pos, 0, 1);
        }
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pos));
        switch (targets.count) {
          case 6: XorScratchVector256(plane_ptrs[targets.ids[5]] + pos, value); [[fallthrough]];
          case 5: XorScratchVector256(plane_ptrs[targets.ids[4]] + pos, value); [[fallthrough]];
          case 4: XorScratchVector256(plane_ptrs[targets.ids[3]] + pos, value); [[fallthrough]];
          case 3: XorScratchVector256(plane_ptrs[targets.ids[2]] + pos, value); [[fallthrough]];
          case 2: XorScratchVector256(plane_ptrs[targets.ids[1]] + pos, value); [[fallthrough]];
          case 1: XorScratchVector256(plane_ptrs[targets.ids[0]] + pos, value); break;
          default: break;
        }
      }
      switch (targets.count) {
        case 6: XorBytesScalar(plane_ptrs[targets.ids[5]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 5: XorBytesScalar(plane_ptrs[targets.ids[4]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 4: XorBytesScalar(plane_ptrs[targets.ids[3]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 3: XorBytesScalar(plane_ptrs[targets.ids[2]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 2: XorBytesScalar(plane_ptrs[targets.ids[1]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 1: XorBytesScalar(plane_ptrs[targets.ids[0]] + vec_limit, src + vec_limit, chunk - vec_limit); break;
        default: break;
      }
    }

    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m256i, 2> acc{};
      ReduceTwoSparseRows256(table, plane_ptrs, used_planes, used_plane_counts, pos, acc);
      for (size_t b = 0; b < 2; ++b) {
        uint8_t* dst_ptr = outputs[base_out + b] + offset + pos;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr), acc[b]);
      }
    }
    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, 2> acc{};
      ReduceTwoSparseRowsScalar256(table, plane_ptrs, used_planes, used_plane_counts, pos, acc);
      for (size_t b = 0; b < 2; ++b) {
        outputs[base_out + b][offset + pos] = acc[b];
      }
    }
  }
}

void ExecuteBraidFixedQuad256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    size_t base_out,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  constexpr size_t kRows = 4;
  constexpr size_t kPlanes = 17;
  constexpr size_t kMaxTargets = 12;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  uint8_t* scratch = AcquireThreadScratch256(kRows * kPlanes * tile);

  std::array<const braid::runtime::OutputPlanRow*, kRows> rows{};
  std::array<uint32_t, kRows> used_plane_masks{};
  std::array<std::array<uint8_t, kPlanes>, kRows> used_planes{};
  std::array<uint8_t, kRows> used_plane_counts{};
  for (size_t b = 0; b < kRows; ++b) {
    rows[b] = &plan.rows[base_out + b];
  }

  struct TargetList {
    uint8_t count = 0;
    std::array<uint8_t, kMaxTargets> ids{};
  };
  std::vector<TargetList> target_lists(plan.input_indices.size());
  for (size_t in = 0; in < plan.input_indices.size(); ++in) {
    auto& targets = target_lists[in];
    for (size_t b = 0; b < kRows; ++b) {
      const uint8_t weight = rows[b]->alpha_weights[in];
      if (weight == 0) {
        continue;
      }
      const auto& positions = rows[b]->alpha_positions[in];
      targets.ids[targets.count++] = static_cast<uint8_t>(b * kPlanes + positions[0]);
      used_plane_masks[b] |= (1u << positions[0]);
      if (weight >= 2) {
        targets.ids[targets.count++] = static_cast<uint8_t>(b * kPlanes + positions[1]);
        used_plane_masks[b] |= (1u << positions[1]);
      }
      if (weight >= 3) {
        targets.ids[targets.count++] = static_cast<uint8_t>(b * kPlanes + positions[2]);
        used_plane_masks[b] |= (1u << positions[2]);
      }
    }
  }
  for (size_t b = 0; b < kRows; ++b) {
    uint32_t mask = used_plane_masks[b];
    while (mask != 0) {
      used_planes[b][used_plane_counts[b]++] = static_cast<uint8_t>(__builtin_ctz(mask));
      mask &= (mask - 1);
    }
  }

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    std::array<uint8_t*, kRows * kPlanes> plane_ptrs{};
    std::array<uint8_t*, 34> pair0_plane_ptrs{};
    std::array<uint8_t*, 34> pair1_plane_ptrs{};
    for (size_t idx = 0; idx < plane_ptrs.size(); ++idx) {
      plane_ptrs[idx] = scratch + idx * tile;
    }
    for (size_t plane = 0; plane < kPlanes; ++plane) {
      pair0_plane_ptrs[plane] = plane_ptrs[plane];
      pair0_plane_ptrs[17 + plane] = plane_ptrs[17 + plane];
      pair1_plane_ptrs[plane] = plane_ptrs[34 + plane];
      pair1_plane_ptrs[17 + plane] = plane_ptrs[51 + plane];
    }
    for (size_t b = 0; b < kRows; ++b) {
      uint32_t mask = used_plane_masks[b];
      while (mask != 0) {
        const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(mask));
        std::memset(plane_ptrs[b * kPlanes + plane], 0, chunk);
        mask &= (mask - 1);
      }
    }

    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t in = 0; in < plan.input_indices.size(); ++in) {
      const auto& targets = target_lists[in];
      if (targets.count == 0) {
        continue;
      }
      const uint8_t* src = inputs[in] + offset;
      for (size_t pos = 0; pos < vec_limit; pos += kVec) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pos));
        switch (targets.count) {
          case 12: XorScratchVector256(plane_ptrs[targets.ids[11]] + pos, value); [[fallthrough]];
          case 11: XorScratchVector256(plane_ptrs[targets.ids[10]] + pos, value); [[fallthrough]];
          case 10: XorScratchVector256(plane_ptrs[targets.ids[9]] + pos, value); [[fallthrough]];
          case 9: XorScratchVector256(plane_ptrs[targets.ids[8]] + pos, value); [[fallthrough]];
          case 8: XorScratchVector256(plane_ptrs[targets.ids[7]] + pos, value); [[fallthrough]];
          case 7: XorScratchVector256(plane_ptrs[targets.ids[6]] + pos, value); [[fallthrough]];
          case 6: XorScratchVector256(plane_ptrs[targets.ids[5]] + pos, value); [[fallthrough]];
          case 5: XorScratchVector256(plane_ptrs[targets.ids[4]] + pos, value); [[fallthrough]];
          case 4: XorScratchVector256(plane_ptrs[targets.ids[3]] + pos, value); [[fallthrough]];
          case 3: XorScratchVector256(plane_ptrs[targets.ids[2]] + pos, value); [[fallthrough]];
          case 2: XorScratchVector256(plane_ptrs[targets.ids[1]] + pos, value); [[fallthrough]];
          case 1: XorScratchVector256(plane_ptrs[targets.ids[0]] + pos, value); break;
          default: break;
        }
      }
      switch (targets.count) {
        case 12: XorBytesScalar(plane_ptrs[targets.ids[11]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 11: XorBytesScalar(plane_ptrs[targets.ids[10]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 10: XorBytesScalar(plane_ptrs[targets.ids[9]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 9: XorBytesScalar(plane_ptrs[targets.ids[8]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 8: XorBytesScalar(plane_ptrs[targets.ids[7]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 7: XorBytesScalar(plane_ptrs[targets.ids[6]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 6: XorBytesScalar(plane_ptrs[targets.ids[5]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 5: XorBytesScalar(plane_ptrs[targets.ids[4]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 4: XorBytesScalar(plane_ptrs[targets.ids[3]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 3: XorBytesScalar(plane_ptrs[targets.ids[2]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 2: XorBytesScalar(plane_ptrs[targets.ids[1]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
        case 1: XorBytesScalar(plane_ptrs[targets.ids[0]] + vec_limit, src + vec_limit, chunk - vec_limit); break;
        default: break;
      }
    }

    std::array<std::array<uint8_t, 17>, 2> pair0_used_planes{used_planes[0], used_planes[1]};
    std::array<std::array<uint8_t, 17>, 2> pair1_used_planes{used_planes[2], used_planes[3]};
    std::array<uint8_t, 2> pair0_counts{used_plane_counts[0], used_plane_counts[1]};
    std::array<uint8_t, 2> pair1_counts{used_plane_counts[2], used_plane_counts[3]};

    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      std::array<__m256i, 4> accs{
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256()};
      std::array<__m256i, 2> pair0_acc{accs[0], accs[1]};
      std::array<__m256i, 2> pair1_acc{accs[2], accs[3]};
      ReduceTwoSparseRows256(table, pair0_plane_ptrs, pair0_used_planes, pair0_counts, pos, pair0_acc);
      ReduceTwoSparseRows256(table, pair1_plane_ptrs, pair1_used_planes, pair1_counts, pos, pair1_acc);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out] + offset + pos), pair0_acc[0]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out + 1] + offset + pos), pair0_acc[1]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out + 2] + offset + pos), pair1_acc[0]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs[base_out + 3] + offset + pos), pair1_acc[1]);
    }
    for (size_t pos = vec_limit; pos < chunk; ++pos) {
      std::array<uint8_t, 4> accs{};
      std::array<uint8_t, 2> pair0_acc{accs[0], accs[1]};
      std::array<uint8_t, 2> pair1_acc{accs[2], accs[3]};
      ReduceTwoSparseRowsScalar256(table, pair0_plane_ptrs, pair0_used_planes, pair0_counts, pos, pair0_acc);
      ReduceTwoSparseRowsScalar256(table, pair1_plane_ptrs, pair1_used_planes, pair1_counts, pos, pair1_acc);
      outputs[base_out][offset + pos] = pair0_acc[0];
      outputs[base_out + 1][offset + pos] = pair0_acc[1];
      outputs[base_out + 2][offset + pos] = pair1_acc[0];
      outputs[base_out + 3][offset + pos] = pair1_acc[1];
    }
  }
}

void ExecuteBraidVector256(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 32;
  const size_t tile = std::max<size_t>(plan.tile_bytes, kVec * 4);
  const int batch = std::max(plan.output_batch, 1);
  const size_t global_block_size = plan.planned_block_size != 0 ? plan.planned_block_size : block_size;
  for (size_t base_out = 0; base_out < plan.rows.size(); base_out += batch) {
    const size_t batch_size = std::min<size_t>(batch, plan.rows.size() - base_out);
    if (batch_size == 4 && plan.type == braid::runtime::PlanType::kEncode &&
        plan.input_indices.size() >= 48 && global_block_size <= (1u << 20)) {
      ExecuteBraidQuadDirectScheduled256(table, plan, base_out, inputs, outputs, block_size);
      continue;
    }
    if (batch_size == 4 && plan.input_indices.size() > 16) {
      ExecuteBraidFixedQuad256(table, plan, base_out, inputs, outputs, block_size);
      continue;
    }
    if (batch_size == 2) {
      if (plan.microkernel_mode == BRAID_MICROKERNEL_PAIR_DIRECT &&
          plan.input_indices.size() >= 48 &&
          plan.type == braid::runtime::PlanType::kEncode) {
        ExecuteBraidPairDirectScheduled256(table, plan, base_out, inputs, outputs, block_size);
        continue;
      }
      ExecuteBraidFixedPair256(table, plan, base_out, inputs, outputs, block_size);
      continue;
    }
    if (batch_size == 1 && plan.input_indices.size() >= 48) {
      ExecuteBraidFixedSingle256(table, plan, base_out, inputs, outputs, block_size);
      continue;
    }
    if (batch_size == 1 && plan.microkernel_mode == BRAID_MICROKERNEL_PAIR_DIRECT &&
        plan.type == braid::runtime::PlanType::kEncode && plan.input_indices.size() >= 48) {
      ExecuteBraidSingleDirectScheduled256(table, plan, base_out, inputs, outputs, block_size);
      continue;
    }
    uint8_t* scratch = AcquireThreadScratch256(batch_size * 17 * tile);
    constexpr size_t kMaxTargets = 12;

    std::vector<const braid::runtime::OutputPlanRow*> rows(batch_size);
    std::vector<uint32_t> used_plane_masks(batch_size, 0);
    std::vector<uint8_t> max_planes(batch_size, 0);
    std::vector<std::array<uint8_t, 17>> used_planes(batch_size);
    std::vector<uint8_t> used_plane_counts(batch_size, 0);
    for (size_t b = 0; b < batch_size; ++b) {
      rows[b] = &plan.rows[base_out + b];
    }

    struct TargetList {
      uint8_t count = 0;
      std::array<uint8_t, kMaxTargets> ids{};
    };
    std::vector<TargetList> target_lists(plan.input_indices.size());
    for (size_t in = 0; in < plan.input_indices.size(); ++in) {
      auto& targets = target_lists[in];
      for (size_t b = 0; b < batch_size; ++b) {
        const uint8_t weight = rows[b]->alpha_weights[in];
        if (weight == 0) {
          continue;
        }
        const auto& positions = rows[b]->alpha_positions[in];
        targets.ids[targets.count++] = static_cast<uint8_t>(b * 17 + positions[0]);
        used_plane_masks[b] |= (1u << positions[0]);
        max_planes[b] = std::max(max_planes[b], positions[0]);
        if (weight >= 2) {
          targets.ids[targets.count++] = static_cast<uint8_t>(b * 17 + positions[1]);
          used_plane_masks[b] |= (1u << positions[1]);
          max_planes[b] = std::max(max_planes[b], positions[1]);
        }
        if (weight >= 3) {
          targets.ids[targets.count++] = static_cast<uint8_t>(b * 17 + positions[2]);
          used_plane_masks[b] |= (1u << positions[2]);
          max_planes[b] = std::max(max_planes[b], positions[2]);
        }
      }
    }
    for (size_t b = 0; b < batch_size; ++b) {
      uint32_t mask = used_plane_masks[b];
      while (mask != 0) {
        used_planes[b][used_plane_counts[b]++] = static_cast<uint8_t>(__builtin_ctz(mask));
        mask &= (mask - 1);
      }
    }

    for (size_t offset = 0; offset < block_size; offset += tile) {
      const size_t chunk = std::min(tile, block_size - offset);
      std::vector<uint8_t*> plane_ptrs(batch_size * 17);
      for (size_t idx = 0; idx < batch_size * 17; ++idx) {
        plane_ptrs[idx] = scratch + (idx * tile);
      }
      for (size_t b = 0; b < batch_size; ++b) {
        uint32_t mask = used_plane_masks[b];
        while (mask != 0) {
          const uint32_t plane = static_cast<uint32_t>(__builtin_ctz(mask));
          std::memset(plane_ptrs[b * 17 + plane], 0, chunk);
          mask &= (mask - 1);
        }
      }
      const size_t vec_limit = (chunk / kVec) * kVec;
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        const auto& targets = target_lists[in];
        if (targets.count == 0) {
          continue;
        }
        const uint8_t* src = inputs[in] + offset;
        for (size_t pos = 0; pos < vec_limit; pos += kVec) {
          const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pos));
          switch (targets.count) {
            case 12: XorScratchVector256(plane_ptrs[targets.ids[11]] + pos, v); [[fallthrough]];
            case 11: XorScratchVector256(plane_ptrs[targets.ids[10]] + pos, v); [[fallthrough]];
            case 10: XorScratchVector256(plane_ptrs[targets.ids[9]] + pos, v); [[fallthrough]];
            case 9: XorScratchVector256(plane_ptrs[targets.ids[8]] + pos, v); [[fallthrough]];
            case 8: XorScratchVector256(plane_ptrs[targets.ids[7]] + pos, v); [[fallthrough]];
            case 7: XorScratchVector256(plane_ptrs[targets.ids[6]] + pos, v); [[fallthrough]];
            case 6: XorScratchVector256(plane_ptrs[targets.ids[5]] + pos, v); [[fallthrough]];
            case 5: XorScratchVector256(plane_ptrs[targets.ids[4]] + pos, v); [[fallthrough]];
            case 4: XorScratchVector256(plane_ptrs[targets.ids[3]] + pos, v); [[fallthrough]];
            case 3: XorScratchVector256(plane_ptrs[targets.ids[2]] + pos, v); [[fallthrough]];
            case 2: XorScratchVector256(plane_ptrs[targets.ids[1]] + pos, v); [[fallthrough]];
            case 1: XorScratchVector256(plane_ptrs[targets.ids[0]] + pos, v); break;
            default: break;
          }
        }
        switch (targets.count) {
          case 12: XorBytesScalar(plane_ptrs[targets.ids[11]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 11: XorBytesScalar(plane_ptrs[targets.ids[10]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 10: XorBytesScalar(plane_ptrs[targets.ids[9]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 9: XorBytesScalar(plane_ptrs[targets.ids[8]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 8: XorBytesScalar(plane_ptrs[targets.ids[7]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 7: XorBytesScalar(plane_ptrs[targets.ids[6]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 6: XorBytesScalar(plane_ptrs[targets.ids[5]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 5: XorBytesScalar(plane_ptrs[targets.ids[4]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 4: XorBytesScalar(plane_ptrs[targets.ids[3]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 3: XorBytesScalar(plane_ptrs[targets.ids[2]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 2: XorBytesScalar(plane_ptrs[targets.ids[1]] + vec_limit, src + vec_limit, chunk - vec_limit); [[fallthrough]];
          case 1: XorBytesScalar(plane_ptrs[targets.ids[0]] + vec_limit, src + vec_limit, chunk - vec_limit); break;
          default: break;
        }
      }
      for (size_t b = 0; b < batch_size; ++b) {
        uint8_t* out = outputs[base_out + b] + offset;
        for (size_t pos = 0; pos < vec_limit; pos += kVec) {
          const uint8_t count = used_plane_counts[b];
          if (count == 0) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + pos), _mm256_setzero_si256());
            continue;
          }
          uint8_t current_plane = used_planes[b][count - 1];
          __m256i acc = _mm256_load_si256(
              reinterpret_cast<const __m256i*>(plane_ptrs[b * 17 + static_cast<size_t>(current_plane)] + pos));
          for (int idx = static_cast<int>(count) - 2; idx >= 0; --idx) {
            const uint8_t next_plane = used_planes[b][static_cast<size_t>(idx)];
            const uint8_t x_exp =
                table.transition_x_exponents[static_cast<size_t>(current_plane)][static_cast<size_t>(next_plane)];
            if (x_exp <= 4) {
              for (uint8_t step = 0; step < x_exp; ++step) {
                acc = XtimeAvx2(acc);
              }
            } else {
              acc = MultiplyConstAvx2(
                  acc, table.transition_coeffs[static_cast<size_t>(current_plane)][static_cast<size_t>(next_plane)]);
            }
            acc = _mm256_xor_si256(
                acc,
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    plane_ptrs[b * 17 + static_cast<size_t>(next_plane)] + pos)));
            current_plane = next_plane;
          }
          if (current_plane != 0) {
            const uint8_t x_exp = table.transition_x_exponents[static_cast<size_t>(current_plane)][0];
            if (x_exp <= 4) {
              for (uint8_t step = 0; step < x_exp; ++step) {
                acc = XtimeAvx2(acc);
              }
            } else {
              acc = MultiplyConstAvx2(acc, table.transition_coeffs[static_cast<size_t>(current_plane)][0]);
            }
          }
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + pos), acc);
        }
        for (size_t pos = vec_limit; pos < chunk; ++pos) {
          const uint8_t count = used_plane_counts[b];
          if (count == 0) {
            out[pos] = 0;
            continue;
          }
          uint8_t current_plane = used_planes[b][count - 1];
          uint8_t acc = plane_ptrs[b * 17 + static_cast<size_t>(current_plane)][pos];
          for (int idx = static_cast<int>(count) - 2; idx >= 0; --idx) {
            const uint8_t next_plane = used_planes[b][static_cast<size_t>(idx)];
            const uint8_t x_exp =
                table.transition_x_exponents[static_cast<size_t>(current_plane)][static_cast<size_t>(next_plane)];
            if (x_exp <= 4) {
              for (uint8_t step = 0; step < x_exp; ++step) {
                acc = braid::core::GF256::MultiplyByX(acc);
              }
            } else {
              acc = braid::core::GF256::Multiply(
                        table.transition_coeffs[static_cast<size_t>(current_plane)][static_cast<size_t>(next_plane)],
                        acc) ^
                    plane_ptrs[b * 17 + static_cast<size_t>(next_plane)][pos];
              current_plane = next_plane;
              continue;
            }
            acc ^= plane_ptrs[b * 17 + static_cast<size_t>(next_plane)][pos];
            current_plane = next_plane;
          }
          if (current_plane != 0) {
            const uint8_t x_exp = table.transition_x_exponents[static_cast<size_t>(current_plane)][0];
            if (x_exp <= 4) {
              for (uint8_t step = 0; step < x_exp; ++step) {
                acc = braid::core::GF256::MultiplyByX(acc);
              }
            } else {
              acc = braid::core::GF256::Multiply(table.transition_coeffs[static_cast<size_t>(current_plane)][0], acc);
            }
          }
          out[pos] = acc;
        }
      }
    }
  }
}

}  // namespace



void ExecuteBraidAVX2(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  ExecuteBraidVector256(table, plan, inputs, outputs, block_size);
}

}  // namespace braid::kernels
