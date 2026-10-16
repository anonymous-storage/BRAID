#include "src/core/gf256.h"

namespace braid::core {

uint8_t GF256::Add(uint8_t a, uint8_t b) { return static_cast<uint8_t>(a ^ b); }

uint8_t GF256::Multiply(uint8_t a, uint8_t b) {
  uint8_t result = 0;
  uint8_t value = a;
  uint8_t factor = b;
  while (factor != 0) {
    if (factor & 1U) {
      result ^= value;
    }
    value = MultiplyByX(value);
    factor >>= 1U;
  }
  return result;
}

uint8_t GF256::Pow(uint8_t a, int exponent) {
  if (exponent == 0) {
    return 1;
  }
  if (a == 0) {
    return 0;
  }
  uint8_t result = 1;
  uint8_t base = a;
  int exp = exponent;
  while (exp > 0) {
    if (exp & 1) {
      result = Multiply(result, base);
    }
    base = Multiply(base, base);
    exp >>= 1;
  }
  return result;
}

uint8_t GF256::Inverse(uint8_t a) {
  if (a == 0) {
    return 0;
  }
  return Pow(a, 254);
}

uint8_t GF256::Divide(uint8_t a, uint8_t b) {
  if (a == 0) {
    return 0;
  }
  return Multiply(a, Inverse(b));
}

uint8_t GF256::MultiplyByX(uint8_t a) {
  const bool high_bit = (a & 0x80U) != 0;
  uint8_t result = static_cast<uint8_t>(a << 1U);
  if (high_bit) {
    result ^= static_cast<uint8_t>(kPolynomial & 0xFFU);
  }
  return result;
}

}  // namespace braid::core
