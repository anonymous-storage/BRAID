#include "src/runtime/cpu_features.h"
#include <iostream>
int main() {
  using braid::runtime::DecodeX86Features;
  const uint32_t c = (1u << 26) | (1u << 27) | (1u << 28);
  const uint32_t b = (1u << 5) | (1u << 16) | (1u << 30) | (1u << 31);
  const uint32_t g = 1u << 8;
  if (!DecodeX86Features(c, b, g, 0xe6).avx512_gfni) return 1;
  if (DecodeX86Features(c, b, 0, 0xe6).avx512_gfni) return 2;
  if (!DecodeX86Features(c, b, 0, 0xe6).avx512) return 7;
  for (int bit : {26, 27, 28})
    if (DecodeX86Features(c & ~(1u << bit), b, g, 0xe6).avx2) return 3;
  for (int bit : {1, 2, 5, 6, 7})
    if (DecodeX86Features(c, b, g, 0xe6 & ~(1u << bit)).avx512) return 4;
  for (int bit : {5, 16, 30, 31})
    if (DecodeX86Features(c, b & ~(1u << bit), g, 0xe6).avx512) return 5;
  if (!DecodeX86Features(c, b, 0, 6).avx2) return 6;
  std::cout << "CPU and OS feature gates OK\n";
}
