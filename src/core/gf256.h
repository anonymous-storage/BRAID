#ifndef BRAID_CORE_GF256_H_
#define BRAID_CORE_GF256_H_

#include <cstdint>

namespace braid::core {

class GF256 {
 public:
  static constexpr uint16_t kPolynomial = 0x139;
  static constexpr uint8_t kPrimitive = 0x03;

  static uint8_t Add(uint8_t a, uint8_t b);
  static uint8_t Multiply(uint8_t a, uint8_t b);
  static uint8_t Pow(uint8_t a, int exponent);
  static uint8_t Inverse(uint8_t a);
  static uint8_t Divide(uint8_t a, uint8_t b);
  static uint8_t MultiplyByX(uint8_t a);
};

}  // namespace braid::core

#endif  // BRAID_CORE_GF256_H_

