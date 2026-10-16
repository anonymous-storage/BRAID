#include "src/runtime/cpu_features.h"
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
namespace braid::runtime {
CpuFeatures DecodeX86Features(uint32_t c1, uint32_t b7, uint32_t c7, uint64_t xcr0) {
  CpuFeatures f;
  constexpr uint32_t avx_os = (1u << 26) | (1u << 27) | (1u << 28);
  if ((c1 & avx_os) != avx_os || (xcr0 & 0x6) != 0x6) return f;
  f.avx2 = (b7 & (1u << 5)) != 0;
  constexpr uint32_t zmm = (1u << 16) | (1u << 30) | (1u << 31);
  f.avx512 = f.avx2 && (b7 & zmm) == zmm && (xcr0 & 0xe6) == 0xe6;
  f.avx512_gfni = f.avx512 && (c7 & (1u << 8));
  return f;
}
CpuFeatures DetectCpuFeatures() {
#if defined(__x86_64__) || defined(__i386__)
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return {};
  const unsigned c1 = c;
  uint64_t xcr0 = 0;
  if ((c1 & ((1u << 26) | (1u << 27))) == ((1u << 26) | (1u << 27))) {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = (uint64_t{hi} << 32) | lo;
  }
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return {};
  return DecodeX86Features(c1, b, c, xcr0);
#else
  return {};
#endif
}
bool BackendAvailable(braid_backend_t backend) {
  const auto f = DetectCpuFeatures();
  switch (backend) {
    case BRAID_BACKEND_AUTO:
    case BRAID_BACKEND_SCALAR: return true;
#ifdef BRAID_HAVE_AVX2
    case BRAID_BACKEND_AVX2: return f.avx2;
#endif
#ifdef BRAID_HAVE_AVX512
    case BRAID_BACKEND_AVX512: return f.avx512;
#endif
    default: return false;
  }
}
bool GfniAvailable() {
#ifdef BRAID_HAVE_AVX512_GFNI
  return DetectCpuFeatures().avx512_gfni;
#else
  return false;
#endif
}
}
