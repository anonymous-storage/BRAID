#ifndef BRAID_TEST_ORACLE_H_
#define BRAID_TEST_ORACLE_H_
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace oracle {
using Matrix = std::vector<std::vector<uint8_t>>;
// Independent bit-serial polynomial oracle. Never compiled into the library.
inline uint8_t mul(uint8_t a, uint8_t b) {
  unsigned x = a, y = b, v = 0;
  while (y) {
    if (y & 1) v ^= x;
    y >>= 1; x <<= 1;
    if (x & 256) x ^= 0x139;
  }
  return static_cast<uint8_t>(v);
}
inline uint8_t pow(uint8_t a, int n) {
  uint8_t v = 1;
  while (n--) v = mul(v, a);
  return v;
}
inline Matrix generator(int k, int r) {
  Matrix v(k + r, std::vector<uint8_t>(k));
  for (int i = 0; i < k + r; ++i)
    for (int j = 0; j < k; ++j) v[i][j] = pow(pow(3, i), j);
  Matrix a(v.begin(), v.begin() + k), inv(k, std::vector<uint8_t>(k));
  for (int i = 0; i < k; ++i) inv[i][i] = 1;
  for (int c = 0; c < k; ++c) {
    int p = c;
    while (p < k && !a[p][c]) ++p;
    if (p == k) throw std::runtime_error("singular oracle");
    std::swap(a[c], a[p]); std::swap(inv[c], inv[p]);
    uint8_t scale = pow(a[c][c], 254);
    for (int j = 0; j < k; ++j) { a[c][j] = mul(a[c][j], scale); inv[c][j] = mul(inv[c][j], scale); }
    for (int i = 0; i < k; ++i) if (i != c) {
      uint8_t f = a[i][c];
      for (int j = 0; j < k; ++j) { a[i][j] ^= mul(f, a[c][j]); inv[i][j] ^= mul(f, inv[c][j]); }
    }
  }
  Matrix out(k + r, std::vector<uint8_t>(k));
  for (int i = 0; i < k + r; ++i)
    for (int j = 0; j < k; ++j)
      for (int t = 0; t < k; ++t) out[i][j] ^= mul(v[i][t], inv[t][j]);
  return out;
}
}
#endif
