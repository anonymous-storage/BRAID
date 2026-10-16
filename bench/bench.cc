#include <braid/braid.hpp>
#include <chrono>
#include <iostream>
#include <vector>
int main() {
  constexpr size_t bytes = 1u << 20;
  auto code = braid::make_code(6, 3);
  std::vector<std::vector<uint8_t>> blocks(9, std::vector<uint8_t>(bytes, 37));
  const uint8_t* in[6]; uint8_t* out[3];
  for (int i = 0; i < 6; ++i) in[i] = blocks[i].data();
  for (int i = 0; i < 3; ++i) out[i] = blocks[i + 6].data();
  braid::plan_ptr plan(braid_build_encode_plan_for_block_size(code.get(), bytes));
  if (!plan) return 1;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i)
    if (braid_execute_encode_plan(code.get(), plan.get(), in, out, bytes) != BRAID_STATUS_OK) return 2;
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "local diagnostic only: backend=" << braid_plan_backend(plan.get())
            << " input_MB/s=" << (10.0 * 6 * bytes / 1e6 / elapsed) << '\n';
}
