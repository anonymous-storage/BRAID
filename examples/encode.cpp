#include <braid/braid.hpp>
#include <array>
#include <iostream>
int main() {
  auto options = braid::default_options();
  auto code = braid::make_code(2, 1, &options);
  std::array<uint8_t, 65> a{}, b{}, parity{}, recovered{};
  a.fill(9); b.fill(31);
  const uint8_t* inputs[] = {a.data(), b.data()};
  uint8_t* outputs[] = {parity.data()};
  braid::plan_ptr plan(braid_build_encode_plan_for_block_size(code.get(), a.size()));
  if (!plan || braid_execute_encode_plan(code.get(), plan.get(), inputs, outputs, a.size()) != BRAID_STATUS_OK) return 1;
  const uint8_t* available[] = {a.data(), nullptr, parity.data()};
  uint8_t* targets[] = {recovered.data()};
  const int erased[] = {1};
  if (braid_decode(code.get(), available, erased, 1, targets, a.size()) != BRAID_STATUS_OK || recovered != b) return 2;
  std::cout << "BRAID C++17 RAII encode/decode OK\n";
}
