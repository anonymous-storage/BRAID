#include <braid/braid.hpp>
#include "tests/oracle.h"
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

void exercise(braid_backend_t backend, int threads, size_t bytes,
              braid_microkernel_mode_t mode = BRAID_MICROKERNEL_BASELINE) {
  constexpr int k = 6, r = 3;
  auto options = braid::default_options();
  options.backend = backend; options.num_threads = threads;
  options.microkernel_mode = mode;
  auto code = braid::make_code(k, r, &options);
  braid::plan_ptr resolved(braid_build_encode_plan_for_block_size(code.get(), bytes));
  check(resolved != nullptr, "resolved plan");
  if (braid_backend_available(backend))
    check(braid_plan_backend(resolved.get()) == backend, "available backend must actually execute");
  if (!braid_gfni_available())
    check(braid_plan_microkernel_mode(resolved.get()) < BRAID_MICROKERNEL_FACTOR_JIT_GFNI, "no unsupported GFNI schedule");
  auto g = oracle::generator(k, r);
  std::vector<std::vector<uint8_t>> blocks(k + r, std::vector<uint8_t>(bytes + 2, 0xa7));
  std::vector<const uint8_t*> inputs(k), available(k + r);
  std::vector<uint8_t*> outputs(r);
  for (int i = 0; i < k; ++i) {
    for (size_t j = 0; j < bytes; ++j) blocks[i][j + 1] = static_cast<uint8_t>(i * 29 + j * 13 + (j >> 8));
    inputs[i] = blocks[i].data() + 1;
  }
  for (int i = 0; i < r; ++i) outputs[i] = blocks[k + i].data() + 1;
  check(braid_encode(code.get(), inputs.data(), outputs.data(), bytes) == BRAID_STATUS_OK, "encode");
  for (int i = 0; i < r; ++i) for (size_t j = 0; j < bytes; ++j) {
    uint8_t expected = 0;
    for (int t = 0; t < k; ++t) expected ^= oracle::mul(g[k + i][t], inputs[t][j]);
    check(expected == outputs[i][j], "independent oracle parity");
  }
  const int erased[] = {7, 0, 3};
  for (int i = 0; i < k + r; ++i) available[i] = blocks[i].data() + 1;
  for (int e : erased) available[e] = nullptr;
  std::vector<std::vector<uint8_t>> restored(r, std::vector<uint8_t>(bytes + 2, 0xa7));
  for (int i = 0; i < r; ++i) outputs[i] = restored[i].data() + 1;
  check(braid_decode(code.get(), available.data(), erased, r, outputs.data(), bytes) == BRAID_STATUS_OK, "decode");
  for (int i = 0; i < r; ++i) {
    check(std::memcmp(outputs[i], blocks[erased[i]].data() + 1, bytes) == 0, "erasure order");
    check(restored[i].front() == 0xa7 && restored[i].back() == 0xa7, "output canary");
  }
  for (const auto& b : blocks) check(b.front() == 0xa7 && b.back() == 0xa7, "block canary");
}

int main() {
  try {
    check(braid_abi_version() == BRAID_ABI_VERSION, "ABI version");
    check(std::strcmp(braid_version(), BRAID_VERSION_STRING) == 0, "version");
    auto o = braid::default_options();
    check(o.num_threads == 1 && !o.pin_threads && !o.numa_aware, "embedded defaults");
    check(braid_options_init(&o, 1) == BRAID_STATUS_INVALID_ARGUMENT, "options size");
    braid_code_t* raw = nullptr;
    check(braid_create_code_ex(INT_MAX, INT_MAX, nullptr, 0, &raw) == BRAID_STATUS_INVALID_ARGUMENT && !raw, "integer overflow");
    o.path = BRAID_PATH_DENSE;
    check(braid_create_code_ex(6, 3, &o, sizeof(o), &raw) == BRAID_STATUS_UNSUPPORTED && !raw, "no dense API");
    check(!braid_create_code(6, 3, &o), "no legacy dense API");
    o = braid::default_options(); o.backend = BRAID_BACKEND_NEON;
    check(!braid_backend_available(BRAID_BACKEND_NEON), "no unverified NEON claim");
    check(braid_create_code_ex(6, 3, &o, sizeof(o), &raw) == BRAID_STATUS_UNSUPPORTED, "NEON reserved");
    o = braid::default_options(); o.tile_bytes = std::numeric_limits<size_t>::max();
    check(!braid_create_code(6, 3, &o), "tile overflow");
    auto c = braid::make_code(2, 1);
    uint8_t byte = 0; const uint8_t* ins[] = {&byte, &byte}; uint8_t* outs[] = {&byte};
    check(braid_encode(c.get(), ins, outs, std::numeric_limits<size_t>::max()) == BRAID_STATUS_INVALID_ARGUMENT, "block overflow");
    auto p = braid::plan_ptr(braid_build_encode_plan(c.get()));
    if (!braid_backend_available(BRAID_BACKEND_AVX512)) check(braid_plan_backend(p.get()) != BRAID_BACKEND_AVX512, "AVX-512 gate");
    std::cout << "GFNI available: " << braid_gfni_available() << '\n';
    for (auto backend : {BRAID_BACKEND_SCALAR, BRAID_BACKEND_AVX2, BRAID_BACKEND_AVX512}) {
      std::cout << "backend " << backend << ": " << (braid_backend_available(backend) ? "available" : "unavailable; factorized fallback") << '\n';
      for (size_t n : {size_t{0}, size_t{1}, size_t{15}, size_t{31}, size_t{32}, size_t{63}, size_t{64}, size_t{65}, size_t{257}, size_t{4099}}) exercise(backend, 1, n);
      exercise(backend, 4, 524291);
    }
    for (auto mode : {BRAID_MICROKERNEL_FACTOR_JIT_GFNI, BRAID_MICROKERNEL_FACTOR_SCHEDULE_GFNI,
                      BRAID_MICROKERNEL_FACTOR_GFNI, BRAID_MICROKERNEL_FACTOR_HYBRID_GFNI})
      exercise(BRAID_BACKEND_AVX512, 1, 4099, mode);
    std::cout << "Release API, oracle, tail and parallel checks OK\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
