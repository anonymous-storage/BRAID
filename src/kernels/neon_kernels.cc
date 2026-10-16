#include "src/kernels/neon_kernels.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <arm_neon.h>

#include "src/kernels/vector_common.h"
#include "src/kernels/vector_entrypoints.h"

namespace braid::kernels {

namespace {

inline uint8x16_t XtimeNeon(uint8x16_t value) {
  const uint8x16_t high_mask = vdupq_n_u8(0x80);
  const uint8x16_t poly = vdupq_n_u8(0x39);
  uint8x16_t high = vandq_u8(value, high_mask);
  uint8x16_t shifted = vshlq_n_u8(value, 1);
  uint8x16_t carry = vcgtq_u8(high, vdupq_n_u8(0x7f));
  return veorq_u8(shifted, vandq_u8(carry, poly));
}

inline uint8x16_t MultiplyConstNeon(uint8x16_t value, uint8_t coefficient) {
  if (coefficient == 0) return vdupq_n_u8(0);
  if (coefficient == 1) return value;
  const auto& tables = GetNibbleTables();
  uint8x16_t lo = vandq_u8(value, vdupq_n_u8(0x0f));
  uint8x16_t hi = vandq_u8(vshrq_n_u8(value, 4), vdupq_n_u8(0x0f));
  uint8x16_t lo_result = vqtbl1q_u8(
      vld1q_u8(reinterpret_cast<const uint8_t*>(tables.lo[coefficient].data())), lo);
  uint8x16_t hi_result = vqtbl1q_u8(
      vld1q_u8(reinterpret_cast<const uint8_t*>(tables.hi[coefficient].data())), hi);
  return veorq_u8(lo_result, hi_result);
}



inline void XorIntoPlaneNeon(int plane, uint8x16_t value,
    uint8x16_t& p0, uint8x16_t& p1, uint8x16_t& p2,
    uint8x16_t& p3, uint8x16_t& p4, uint8x16_t& p5,
    uint8x16_t& p6, uint8x16_t& p7, uint8x16_t& p8,
    uint8x16_t& p9, uint8x16_t& p10, uint8x16_t& p11,
    uint8x16_t& p12, uint8x16_t& p13, uint8x16_t& p14,
    uint8x16_t& p15, uint8x16_t& p16) {
  switch (plane) {
    case 0: p0 = veorq_u8(p0, value); break;
    case 1: p1 = veorq_u8(p1, value); break;
    case 2: p2 = veorq_u8(p2, value); break;
    case 3: p3 = veorq_u8(p3, value); break;
    case 4: p4 = veorq_u8(p4, value); break;
    case 5: p5 = veorq_u8(p5, value); break;
    case 6: p6 = veorq_u8(p6, value); break;
    case 7: p7 = veorq_u8(p7, value); break;
    case 8: p8 = veorq_u8(p8, value); break;
    case 9: p9 = veorq_u8(p9, value); break;
    case 10: p10 = veorq_u8(p10, value); break;
    case 11: p11 = veorq_u8(p11, value); break;
    case 12: p12 = veorq_u8(p12, value); break;
    case 13: p13 = veorq_u8(p13, value); break;
    case 14: p14 = veorq_u8(p14, value); break;
    case 15: p15 = veorq_u8(p15, value); break;
    case 16: p16 = veorq_u8(p16, value); break;
    default: break;
  }
}

inline uint8x16_t ReducePlanesNeon(
    uint8x16_t p0, uint8x16_t p1, uint8x16_t p2, uint8x16_t p3,
    uint8x16_t p4, uint8x16_t p5, uint8x16_t p6, uint8x16_t p7,
    uint8x16_t p8, uint8x16_t p9, uint8x16_t p10, uint8x16_t p11,
    uint8x16_t p12, uint8x16_t p13, uint8x16_t p14, uint8x16_t p15,
    uint8x16_t p16) {
  uint8x16_t acc = p16;
  acc = veorq_u8(XtimeNeon(acc), p15);
  acc = veorq_u8(XtimeNeon(acc), p14);
  acc = veorq_u8(XtimeNeon(acc), p13);
  acc = veorq_u8(XtimeNeon(acc), p12);
  acc = veorq_u8(XtimeNeon(acc), p11);
  acc = veorq_u8(XtimeNeon(acc), p10);
  acc = veorq_u8(XtimeNeon(acc), p9);
  acc = veorq_u8(XtimeNeon(acc), p8);
  acc = veorq_u8(XtimeNeon(acc), p7);
  acc = veorq_u8(XtimeNeon(acc), p6);
  acc = veorq_u8(XtimeNeon(acc), p5);
  acc = veorq_u8(XtimeNeon(acc), p4);
  acc = veorq_u8(XtimeNeon(acc), p3);
  acc = veorq_u8(XtimeNeon(acc), p2);
  acc = veorq_u8(XtimeNeon(acc), p1);
  acc = veorq_u8(XtimeNeon(acc), p0);
  return acc;
}

void ExecuteBraidBatch1RegisterNeon(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 16;
  const size_t vec_limit = (block_size / kVec) * kVec;
  for (size_t out_idx = 0; out_idx < plan.rows.size(); ++out_idx) {
    const auto& row = plan.rows[out_idx];
    uint8_t* out = outputs[out_idx];
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      uint8x16_t p0 = vdupq_n_u8(0), p1 = vdupq_n_u8(0), p2 = vdupq_n_u8(0);
      uint8x16_t p3 = vdupq_n_u8(0), p4 = vdupq_n_u8(0), p5 = vdupq_n_u8(0);
      uint8x16_t p6 = vdupq_n_u8(0), p7 = vdupq_n_u8(0), p8 = vdupq_n_u8(0);
      uint8x16_t p9 = vdupq_n_u8(0), p10 = vdupq_n_u8(0), p11 = vdupq_n_u8(0);
      uint8x16_t p12 = vdupq_n_u8(0), p13 = vdupq_n_u8(0), p14 = vdupq_n_u8(0);
      uint8x16_t p15 = vdupq_n_u8(0), p16 = vdupq_n_u8(0);

      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        uint8_t weight = row.alpha_weights[in];
        if (weight == 0) continue;
        uint8x16_t value = vld1q_u8(inputs[in] + pos);
        const auto& positions = row.alpha_positions[in];
        XorIntoPlaneNeon(positions[0], value, p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16);
        if (weight >= 2) XorIntoPlaneNeon(positions[1], value, p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16);
        if (weight >= 3) XorIntoPlaneNeon(positions[2], value, p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16);
      }

      uint8x16_t reduced = ReducePlanesNeon(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16);
      vst1q_u8(out + pos, reduced);
    }
    for (size_t pos = vec_limit; pos < block_size; ++pos) {
      uint8_t planes[17] = {};
      for (size_t in = 0; in < plan.input_indices.size(); ++in) {
        uint8_t weight = row.alpha_weights[in];
        if (weight == 0) continue;
        uint8_t value = inputs[in][pos];
        const auto& positions = row.alpha_positions[in];
        planes[positions[0]] ^= value;
        if (weight >= 2) planes[positions[1]] ^= value;
        if (weight >= 3) planes[positions[2]] ^= value;
      }
      uint8_t acc = planes[16];
      for (int plane = 15; plane >= 0; --plane)
        acc = braid::core::GF256::MultiplyByX(acc) ^ planes[plane];
      out[pos] = acc;
    }
  }
}

void ExecuteBraidParallelNeon(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  constexpr size_t kVec = 16;
  const size_t tile = std::max<size_t>(plan.tile_bytes, 64u << 10);
  const size_t input_count = plan.input_indices.size();
  const size_t rows = plan.rows.size();

  uint32_t union_mask = 0;
  for (size_t in = 0; in < input_count; ++in)
    for (size_t b = 0; b < rows; ++b) {
      uint8_t w = plan.rows[b].alpha_weights[in];
      if (w == 0) continue;
      const auto& pos = plan.rows[b].alpha_positions[in];
      union_mask |= (1u << pos[0]); if (w>=2) union_mask |= (1u<<pos[1]); if (w>=3) union_mask |= (1u<<pos[2]);
    }

  const auto& nt = GetNibbleTables();
  uint8x16_t lo_tbls[17], hi_tbls[17];
  for (uint8_t p = 0; p < 17; ++p) if (union_mask & (1u<<p)) {
    uint8_t c = table.x_powers[p];
    lo_tbls[p] = vld1q_u8(nt.lo[c].data()); hi_tbls[p] = vld1q_u8(nt.hi[c].data());
  }

  std::vector<uint32_t> imasks(input_count);
  // Precompute per-row plane lists (built ONCE, not per-vector!)
  std::vector<std::array<uint8_t,17>> row_planes(rows);
  std::vector<uint8_t> row_plane_counts(rows, 0);
  for (size_t in=0;in<input_count;++in){uint32_t m=0;
    for(size_t b=0;b<rows;++b){uint8_t w=plan.rows[b].alpha_weights[in];if(w==0)continue;
      const auto& pos=plan.rows[b].alpha_positions[in];
      m|=(1u<<pos[0]);if(w>=2)m|=(1u<<pos[1]);if(w>=3)m|=(1u<<pos[2]);}
    imasks[in]=m;}
  for (size_t b=0;b<rows;++b){uint32_t rm=0;
    for(size_t in=0;in<input_count;++in){uint8_t w=plan.rows[b].alpha_weights[in];if(w==0)continue;
      const auto& pos=plan.rows[b].alpha_positions[in];rm|=(1u<<pos[0]);if(w>=2)rm|=(1u<<pos[1]);if(w>=3)rm|=(1u<<pos[2]);}
    auto& rp=row_planes[b]; uint8_t& rc=row_plane_counts[b];
    for(uint8_t pp=0;pp<17;++pp) if(rm&(1u<<pp)) rp[rc++]=pp;}

  for (size_t offset = 0; offset < block_size; offset += tile) {
    const size_t chunk = std::min(tile, block_size - offset);
    const size_t vec_limit = (chunk / kVec) * kVec;
    for (size_t pos = 0; pos < vec_limit; pos += kVec) {
      uint8x16_t p[17];
      for(int i=0;i<17;++i) p[i]=vdupq_n_u8(0);
      for (size_t in = 0; in < input_count; ++in) {
        uint32_t m = imasks[in]; if (m == 0) continue;
        uint8x16_t v = vld1q_u8(inputs[in] + offset + pos);
        uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0f));
        uint8x16_t hi = vandq_u8(vshrq_n_u8(v, 4), vdupq_n_u8(0x0f));
        while (m) { uint8_t pn = __builtin_ctz(m); m &= (m - 1);
          p[pn] = veorq_u8(p[pn], veorq_u8(vqtbl1q_u8(lo_tbls[pn], lo), vqtbl1q_u8(hi_tbls[pn], hi)));
        }
      }
      for (size_t b = 0; b < rows; ++b) {
        const auto& rp = row_planes[b]; uint8_t rc = row_plane_counts[b];
        uint8x16_t acc; if(rc==0){acc=vdupq_n_u8(0);}else{uint8_t cur=rp[rc-1];acc=p[cur];
          for(int idx=rc-2;idx>=0;--idx){uint8_t nxt=rp[idx];
            for(uint8_t g=cur-nxt;g>0;--g) acc=XtimeNeon(acc);
            acc=veorq_u8(acc,p[nxt]);cur=nxt;}
          for(uint8_t g=cur;g>0;--g) acc=XtimeNeon(acc);}
        vst1q_u8(outputs[b] + offset + pos, acc);
      }
    }
    for (size_t pos=vec_limit;pos<chunk;++pos){uint8_t planes[17]={};
      for(size_t in=0;in<input_count;++in){uint8_t v=inputs[in][offset+pos];
        for(size_t b=0;b<rows;++b){uint8_t w=plan.rows[b].alpha_weights[in];if(w==0)continue;
          const auto& pos_arr=plan.rows[b].alpha_positions[in];planes[pos_arr[0]]^=v;
          if(w>=2)planes[pos_arr[1]]^=v;if(w>=3)planes[pos_arr[2]]^=v;}}
      for(size_t b=0;b<rows;++b){const auto& rp=row_planes[b];uint8_t rc=row_plane_counts[b];
        uint8_t acc=0;if(rc>0){acc=planes[rp[rc-1]];
          for(int idx=rc-2;idx>=0;--idx){for(uint8_t g=rp[rc-1]-rp[idx];g>0;--g)acc=braid::core::GF256::MultiplyByX(acc);
            acc^=planes[rp[idx]];}}
        outputs[b][offset+pos]=acc;
      }
    }
  }
}

}  // namespace

void ExecuteBraidNeon(
    const braid::core::DecompositionTable& table,
    const braid::runtime::Plan& plan,
    const uint8_t* const* inputs,
    uint8_t* const* outputs,
    size_t block_size) {
  // k>=8: Parallel two-phase (independent plane expansion, no gap chain)
  if (plan.input_indices.size() >= 8) {
    ExecuteBraidParallelNeon(table, plan, inputs, outputs, block_size);
    return;
  }
  // The row-wise register kernel is pure BRAID-F and handles every output.
  ExecuteBraidBatch1RegisterNeon(table, plan, inputs, outputs, block_size);
}



}  // namespace braid::kernels
