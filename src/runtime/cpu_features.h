#ifndef BRAID_RUNTIME_CPU_FEATURES_H_
#define BRAID_RUNTIME_CPU_FEATURES_H_
#include <cstdint>
#include "braid/braid.h"
namespace braid::runtime {
struct CpuFeatures { bool avx2 = false; bool avx512 = false; bool avx512_gfni = false; };
// Pure decoder also permits testing missing CPU/OS features without executing ISA.
CpuFeatures DecodeX86Features(uint32_t leaf1_ecx, uint32_t leaf7_ebx,
                             uint32_t leaf7_ecx, uint64_t xcr0);
CpuFeatures DetectCpuFeatures();
bool BackendAvailable(braid_backend_t backend);
bool GfniAvailable();
}
#endif
