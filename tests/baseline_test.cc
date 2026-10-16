#include <atomic>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "braid/braid.h"
#include "src/core/decomposition.h"
#include "src/core/gfni_matrix.h"
#include "src/core/gf256.h"
#include "src/core/rs_matrix.h"
#include "src/runtime/context.h"
#include "src/runtime/plan.h"
#include "src/runtime/thread_pool.h"
#include "tests/oracle.h"

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void TestGF256Basic() {
  using braid::core::GF256;
  Expect(GF256::Add(0x12, 0x12) == 0x00, "GF256 add should xor");
  for (uint8_t a : {uint8_t{1}, uint8_t{2}, uint8_t{3}, uint8_t{17}, uint8_t{93}, uint8_t{255}}) {
    Expect(GF256::Multiply(a, 1) == a, "GF256 multiply by one");
    Expect(GF256::Multiply(0, a) == 0, "GF256 multiply by zero");
    if (a != 0) {
      Expect(GF256::Multiply(a, GF256::Inverse(a)) == 1, "GF256 inverse");
    }
  }
}

void TestGfniAffineMatrices() {
  using braid::core::BuildGfniAffineMatrix;
  Expect(
      BuildGfniAffineMatrix(1) == UINT64_C(0x0102040810204080),
      "GFNI identity matrix");
  Expect(
      BuildGfniAffineMatrix(2) == UINT64_C(0x8001028488902040),
      "GFNI GF(0x139) xtime matrix");
  Expect(
      BuildGfniAffineMatrix(4) == UINT64_C(0x40800142c4c89020),
      "GFNI GF(0x139) x^2 matrix");
  Expect(
      BuildGfniAffineMatrix(0x2f) == UINT64_C(0x2953a767e6e5ca94),
      "GFNI arbitrary coefficient matrix");
}

void TestDecompositionProperties() {
  braid::core::DecompositionTable table = braid::core::BuildDecompositionTable();
  Expect(table.order == 17, "Expected ord(0x139)=17");
  Expect(table.common_factor_nnz == 48, "Expected nnz(M)=48");
  for (int a = 0; a < 256; ++a) {
    for (int v = 0; v < 256; ++v) {
      const auto dense = oracle::mul(static_cast<uint8_t>(a), static_cast<uint8_t>(v));
      const auto factorized = braid::core::ApplyCoefficientBraid(table, static_cast<uint8_t>(a), static_cast<uint8_t>(v));
      Expect(dense == factorized, "Dense and factorized coefficient application mismatch");
    }
  }
}

void TestCustomOrderedDecompositionProperties() {
  auto base = braid::core::BuildDecompositionTable();
  auto reordered = base.x_powers;
  std::swap(reordered[3], reordered[11]);
  std::swap(reordered[5], reordered[14]);
  braid::core::DecompositionTable table = braid::core::BuildDecompositionTable(reordered);
  Expect(!table.progression_order, "Expected reordered table to break progression order");
  for (int a = 0; a < 256; ++a) {
    for (int v = 0; v < 256; ++v) {
      const auto dense = oracle::mul(static_cast<uint8_t>(a), static_cast<uint8_t>(v));
      const auto factorized = braid::core::ApplyCoefficientBraid(table, static_cast<uint8_t>(a), static_cast<uint8_t>(v));
      Expect(dense == factorized, "Custom ordered decomposition mismatch");
    }
  }
}

void TestGeneratorMatrix() {
  auto generator = braid::core::BuildSystematicGeneratorMatrix(6, 3);
  Expect(generator.size() == 9, "Generator rows");
  Expect(generator.front().size() == 6, "Generator cols");
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      const uint8_t expected = (i == j) ? 1 : 0;
      Expect(generator[i][j] == expected, "Systematic top rows must be identity");
    }
  }
}

void TestPublicApiSmoke() {
  braid_options_t options{};
  options.backend = BRAID_BACKEND_AUTO;
  options.path = BRAID_PATH_AUTO;
  options.enable_plan_cache = 1;

  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(6, 3, &options), braid_destroy_code);
  Expect(code != nullptr, "braid_create_code should succeed");

  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> plan(
      braid_build_encode_plan(code.get()), braid_destroy_plan);
  Expect(plan != nullptr, "braid_build_encode_plan should succeed");
  Expect(braid_plan_tile_bytes(plan.get()) > 0, "plan tile accessor");
  Expect(braid_plan_output_batch(plan.get()) > 0, "plan batch accessor");
  Expect(braid_plan_path(plan.get()) == BRAID_PATH_BRAID, "AUTO must resolve to pure BRAID-F");
}

void TestAutoPathNeverSelectsDense() {
  for (braid_backend_t backend :
       {BRAID_BACKEND_AUTO, BRAID_BACKEND_SCALAR, BRAID_BACKEND_AVX2, BRAID_BACKEND_AVX512}) {
    braid_options_t options{};
    options.backend = backend;
    options.path = BRAID_PATH_AUTO;
    options.num_threads = 4;
    braid::runtime::Context context(100, 4, options);
    for (size_t block_size : {size_t{1}, size_t{512} << 10, size_t{4} << 20, size_t{64} << 20}) {
      auto encode = braid::runtime::BuildEncodePlan(context, block_size);
      auto decode1 = braid::runtime::BuildDecodePlan(context, std::vector<int>{0}, block_size);
      auto decode4 = braid::runtime::BuildDecodePlan(context, std::vector<int>{0, 1, 2, 3}, block_size);
      Expect(encode->path == BRAID_PATH_BRAID, "AUTO encode selected dense");
      Expect(decode1->path == BRAID_PATH_BRAID, "AUTO e1 selected dense");
      Expect(decode4->path == BRAID_PATH_BRAID, "AUTO e4 selected dense");
      if (decode1->backend == BRAID_BACKEND_AVX512) {
        Expect(
            decode1->microkernel_mode == BRAID_MICROKERNEL_FACTOR_SCRATCH ||
                decode1->microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT,
            "AUTO long-chain e1 did not resolve to a pure factor microkernel");
      }
    }
  }

  if (braid_backend_available(BRAID_BACKEND_AVX512)) {
    braid_options_t options{};
    options.backend = BRAID_BACKEND_AVX512;
    options.path = BRAID_PATH_AUTO;
    options.num_threads = 4;
    braid::runtime::Context context(16, 4, options);
    auto encode = braid::runtime::BuildEncodePlan(context, size_t{4} << 20);
    Expect(encode->path == BRAID_PATH_BRAID, "short-chain AUTO encode selected dense");
    Expect(
        encode->microkernel_mode == BRAID_MICROKERNEL_FACTOR_POLYNOMIAL_FUSED ||
            encode->microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT,
        "short-chain AUTO encode did not expose a pure polynomial factor microkernel");

    braid::runtime::Context mid_context(44, 6, options);
    auto small_e4 = braid::runtime::BuildDecodePlan(
        mid_context, std::vector<int>{0, 1, 2, 3}, size_t{512} << 10);
    auto large_e4 = braid::runtime::BuildDecodePlan(
        mid_context, std::vector<int>{0, 1, 2, 3}, size_t{4} << 20);
    Expect(small_e4->output_batch == 4, "small mid-chain e4 did not resolve to batch=4");
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
    Expect(small_e4->microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT,
           "small mid-chain e4 did not resolve to the compiled factor microkernel");
#else
    Expect(small_e4->microkernel_mode != BRAID_MICROKERNEL_FACTOR_JIT,
           "disabled JIT must not be selected");
#endif
    Expect(large_e4->output_batch == 4, "large mid-chain e4 did not resolve to batch=4");
    Expect(
        large_e4->microkernel_mode == BRAID_MICROKERNEL_BASELINE,
        "large mid-chain e4 unexpectedly left the long factor schedule");

    braid::runtime::Context batch_context(24, 4, options);
    auto single_small =
        batch_context.GetOrBuildEncodePlan(size_t{128} << 10);
    auto batch_small =
        batch_context.GetOrBuildEncodeBatchPlan(size_t{128} << 10);
    Expect(
        single_small->microkernel_mode ==
            BRAID_MICROKERNEL_FACTOR_PLANE_STREAMING,
        "single-stripe small plan unexpectedly changed");
#if defined(BRAID_ENABLE_AVX512_PLAN_JIT)
    Expect(
        batch_small->microkernel_mode == BRAID_MICROKERNEL_FACTOR_JIT,
        "inter-unit small plan did not select the compiled factor kernel");
    Expect(
        batch_small->streaming_stores,
        "inter-unit aligned plan did not enable streaming stores");
#else
    Expect(batch_small->microkernel_mode == single_small->microkernel_mode,
           "no-JIT batching must preserve the factorized base plan");
#endif

    braid::runtime::Context long_context(74, 6, options);
    auto long_encode = braid::runtime::BuildEncodePlan(long_context, size_t{512} << 10);
    auto long_e4 = braid::runtime::BuildDecodePlan(
        long_context, std::vector<int>{0, 1, 2, 3}, size_t{512} << 10);
    auto long_e2 = braid::runtime::BuildDecodePlan(
        long_context, std::vector<int>{0, 1}, size_t{512} << 10);
    auto long_e3 = braid::runtime::BuildDecodePlan(
        long_context, std::vector<int>{0, 1, 2}, size_t{512} << 10);
    Expect(long_encode->tile_bytes == 1024, "small long-chain encode did not select 1 KiB tile");
    Expect(long_e2->tile_bytes == 768, "small long-chain e2 did not select 768-byte tile");
    Expect(long_e3->tile_bytes == 512, "small long-chain e3 did not select 512-byte tile");
    Expect(long_e4->output_batch == 4, "small long-chain e4 did not resolve to batch=4");
    Expect(long_e4->tile_bytes == 2048, "small long-chain e4 did not select 2 KiB tile");
  }
}

void TestEncodeDecodeRoundTrip() {
  constexpr int k = 6;
  constexpr int r = 3;
  constexpr int n = k + r;
  constexpr size_t block_size = 1024;

  braid_options_t options{};
  options.backend = BRAID_BACKEND_SCALAR;
  options.path = BRAID_PATH_BRAID;
  options.enable_plan_cache = 1;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "create code");

  std::vector<std::vector<uint8_t>> data(k, std::vector<uint8_t>(block_size));
  std::vector<std::vector<uint8_t>> parity(r, std::vector<uint8_t>(block_size, 0));
  std::mt19937 rng(12345);
  for (auto& block : data) {
    for (auto& byte : block) {
      byte = static_cast<uint8_t>(rng() & 0xFF);
    }
  }
  std::vector<const uint8_t*> data_ptrs(k);
  std::vector<uint8_t*> parity_ptrs(r);
  for (int i = 0; i < k; ++i) data_ptrs[i] = data[i].data();
  for (int i = 0; i < r; ++i) parity_ptrs[i] = parity[i].data();
  Expect(braid_encode(code.get(), data_ptrs.data(), parity_ptrs.data(), block_size) == BRAID_STATUS_OK, "encode");

  std::vector<const uint8_t*> all_blocks(n, nullptr);
  for (int i = 0; i < k; ++i) all_blocks[i] = data[i].data();
  for (int i = 0; i < r; ++i) all_blocks[k + i] = parity[i].data();

  int erased[2] = {1, 7};
  all_blocks[1] = nullptr;
  all_blocks[7] = nullptr;
  std::vector<std::vector<uint8_t>> recovered(2, std::vector<uint8_t>(block_size, 0));
  std::vector<uint8_t*> recovered_ptrs = {recovered[0].data(), recovered[1].data()};
  Expect(braid_decode(code.get(), all_blocks.data(), erased, 2, recovered_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "decode");
  Expect(std::memcmp(recovered[0].data(), data[1].data(), block_size) == 0, "recovered data block mismatch");
  Expect(std::memcmp(recovered[1].data(), parity[1].data(), block_size) == 0, "recovered parity block mismatch");
}

void TestInternalParallelRoundTrip() {
  constexpr int k = 6;
  constexpr int r = 3;
  constexpr int n = k + r;
  constexpr size_t block_size = 1 << 20;

  braid_options_t options{};
  options.backend = BRAID_BACKEND_AVX2;
  options.path = BRAID_PATH_BRAID;
  options.num_threads = 4;
  options.enable_plan_cache = 1;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "parallel create code");

  std::vector<std::vector<uint8_t>> data(k, std::vector<uint8_t>(block_size));
  std::vector<std::vector<uint8_t>> parity(r, std::vector<uint8_t>(block_size, 0));
  std::mt19937 rng(54321);
  for (auto& block : data) {
    for (auto& byte : block) {
      byte = static_cast<uint8_t>(rng() & 0xFF);
    }
  }

  std::vector<const uint8_t*> data_ptrs(k);
  std::vector<uint8_t*> parity_ptrs(r);
  for (int i = 0; i < k; ++i) data_ptrs[i] = data[i].data();
  for (int i = 0; i < r; ++i) parity_ptrs[i] = parity[i].data();
  Expect(braid_encode(code.get(), data_ptrs.data(), parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "parallel encode");

  std::vector<const uint8_t*> all_blocks(n, nullptr);
  for (int i = 0; i < k; ++i) all_blocks[i] = data[i].data();
  for (int i = 0; i < r; ++i) all_blocks[k + i] = parity[i].data();
  int erased[2] = {0, 6};
  all_blocks[0] = nullptr;
  all_blocks[6] = nullptr;
  std::vector<std::vector<uint8_t>> recovered(2, std::vector<uint8_t>(block_size, 0));
  std::vector<uint8_t*> recovered_ptrs = {recovered[0].data(), recovered[1].data()};
  Expect(braid_decode(code.get(), all_blocks.data(), erased, 2, recovered_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "parallel decode");
  Expect(std::memcmp(recovered[0].data(), data[0].data(), block_size) == 0, "parallel recovered data mismatch");
  Expect(std::memcmp(recovered[1].data(), parity[0].data(), block_size) == 0, "parallel recovered parity mismatch");
}

void TestVectorBackendsWhenAvailable() {
  constexpr int k = 6;
  constexpr int r = 3;
  constexpr size_t block_size = 4096;

  std::vector<std::vector<uint8_t>> data(k, std::vector<uint8_t>(block_size));
  std::mt19937 rng(2222);
  for (auto& block : data) {
    for (auto& byte : block) {
      byte = static_cast<uint8_t>(rng() & 0xFF);
    }
  }
  std::vector<const uint8_t*> data_ptrs(k);
  for (int i = 0; i < k; ++i) data_ptrs[i] = data[i].data();

  auto run_backend = [&](braid_backend_t backend) {
    braid_options_t options{};
    options.backend = backend;
    options.path = BRAID_PATH_BRAID;
    std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
        braid_create_code(k, r, &options), braid_destroy_code);
    Expect(code != nullptr, "vector backend create");
    std::vector<std::vector<uint8_t>> parity(r, std::vector<uint8_t>(block_size, 0));
    std::vector<uint8_t*> parity_ptrs(r);
    for (int i = 0; i < r; ++i) parity_ptrs[i] = parity[i].data();
    Expect(braid_encode(code.get(), data_ptrs.data(), parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
           "vector backend encode");
    return parity;
  };

  braid_options_t scalar_options{};
  scalar_options.backend = BRAID_BACKEND_SCALAR;
  scalar_options.path = BRAID_PATH_BRAID;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> scalar_code(
      braid_create_code(k, r, &scalar_options), braid_destroy_code);
  std::vector<std::vector<uint8_t>> scalar_parity(r, std::vector<uint8_t>(block_size, 0));
  std::vector<uint8_t*> scalar_ptrs(r);
  for (int i = 0; i < r; ++i) scalar_ptrs[i] = scalar_parity[i].data();
  Expect(braid_encode(scalar_code.get(), data_ptrs.data(), scalar_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "scalar encode");

  if (braid_backend_available(BRAID_BACKEND_AVX2)) {
    auto avx2 = run_backend(BRAID_BACKEND_AVX2);
    for (int i = 0; i < r; ++i) {
      Expect(std::memcmp(avx2[i].data(), scalar_parity[i].data(), block_size) == 0, "avx2 parity mismatch");
    }
  }
  if (braid_backend_available(BRAID_BACKEND_AVX512)) {
    auto avx512 = run_backend(BRAID_BACKEND_AVX512);
    for (int i = 0; i < r; ++i) {
      Expect(std::memcmp(avx512[i].data(), scalar_parity[i].data(), block_size) == 0, "avx512 parity mismatch");
    }
  }
}

void TestRetunedRepresentationRoundTrip() {
  constexpr int k = 6;
  constexpr int r = 3;
  constexpr int n = k + r;
  constexpr size_t block_size = 8192;

  braid_options_t options{};
  options.backend = BRAID_BACKEND_SCALAR;
  options.path = BRAID_PATH_BRAID;
  options.repr_mode = BRAID_REPR_RETUNE;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "retuned create code");
  Expect(braid_code_repr_generator(code.get()) != 0, "retuned generator should be selected");

  std::vector<std::vector<uint8_t>> data(k, std::vector<uint8_t>(block_size));
  std::vector<std::vector<uint8_t>> parity(r, std::vector<uint8_t>(block_size, 0));
  std::mt19937 rng(7777);
  for (auto& block : data) {
    for (auto& byte : block) {
      byte = static_cast<uint8_t>(rng() & 0xFF);
    }
  }
  std::vector<const uint8_t*> data_ptrs(k);
  std::vector<uint8_t*> parity_ptrs(r);
  for (int i = 0; i < k; ++i) data_ptrs[i] = data[i].data();
  for (int i = 0; i < r; ++i) parity_ptrs[i] = parity[i].data();
  Expect(braid_encode(code.get(), data_ptrs.data(), parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "retuned encode");

  std::vector<const uint8_t*> all_blocks(n, nullptr);
  for (int i = 0; i < k; ++i) all_blocks[i] = data[i].data();
  for (int i = 0; i < r; ++i) all_blocks[k + i] = parity[i].data();

  int erased[2] = {2, 7};
  all_blocks[2] = nullptr;
  all_blocks[7] = nullptr;
  std::vector<std::vector<uint8_t>> recovered(2, std::vector<uint8_t>(block_size, 0));
  std::vector<uint8_t*> recovered_ptrs = {recovered[0].data(), recovered[1].data()};
  Expect(braid_decode(code.get(), all_blocks.data(), erased, 2, recovered_ptrs.data(), block_size) == BRAID_STATUS_OK,
         "retuned decode");
  Expect(std::memcmp(recovered[0].data(), data[2].data(), block_size) == 0, "retuned recovered data mismatch");
  Expect(std::memcmp(recovered[1].data(), parity[1].data(), block_size) == 0, "retuned recovered parity mismatch");
}

void TestRetuneSingleOutputDecodeUsesProgressionOrder() {
  braid_options_t options{};
  options.backend = BRAID_BACKEND_AVX512;
  options.path = BRAID_PATH_BRAID;
  options.repr_mode = BRAID_REPR_RETUNE;

  braid::runtime::Context context(64, 4, options);
  auto plan = braid::runtime::BuildDecodePlan(context, std::vector<int>{0}, 16u << 20);
  Expect(plan != nullptr, "retune decode plan should build");
  Expect(plan->table.progression_order, "single-output retune decode should keep progression order");
}

braid_options_t BraidOptions(braid_backend_t backend, int threads = 1) {
  braid_options_t options{};
  options.backend = backend;
  options.path = BRAID_PATH_BRAID;
  options.tile_bytes = 257;
  options.output_batch = 4;
  options.num_threads = threads;
  options.enable_plan_cache = 1;
  return options;
}

std::vector<std::vector<uint8_t>> BuildDenseCodeword(int k, int r, size_t block_size, uint32_t seed) {
  const auto generator = braid::core::BuildSystematicGeneratorMatrix(k, r);
  std::vector<std::vector<uint8_t>> blocks(
      static_cast<size_t>(k + r), std::vector<uint8_t>(block_size));
  for (int input = 0; input < k; ++input) {
    for (size_t pos = 0; pos < block_size; ++pos) {
      seed = seed * 1664525u + 1013904223u;
      blocks[static_cast<size_t>(input)][pos] = static_cast<uint8_t>(seed >> 24);
    }
  }
  for (int parity = 0; parity < r; ++parity) {
    const auto& row = generator[static_cast<size_t>(k + parity)];
    for (size_t pos = 0; pos < block_size; ++pos) {
      uint8_t value = 0;
      for (int input = 0; input < k; ++input) {
        value ^= braid::core::GF256::Multiply(
            row[static_cast<size_t>(input)], blocks[static_cast<size_t>(input)][pos]);
      }
      blocks[static_cast<size_t>(k + parity)][pos] = value;
    }
  }
  return blocks;
}

void TestGfniFactorizedKernels() {
  if (!braid_gfni_available()) {
    return;
  }

  constexpr size_t block_size = 4109;
  for (const auto [k, r] :
       {std::pair<int, int>{10, 4},
        {24, 4},
        {44, 6},
        {74, 6},
        {100, 4}}) {
    const auto expected =
        BuildDenseCodeword(k, r, block_size, 0x9e3779b9u + k);
    std::vector<std::vector<uint8_t>> parity(
        static_cast<size_t>(r), std::vector<uint8_t>(block_size));
    std::vector<const uint8_t*> data(static_cast<size_t>(k));
    std::vector<uint8_t*> outputs(static_cast<size_t>(r));
    for (int input = 0; input < k; ++input) {
      data[static_cast<size_t>(input)] =
          expected[static_cast<size_t>(input)].data();
    }
    for (int output = 0; output < r; ++output) {
      outputs[static_cast<size_t>(output)] =
          parity[static_cast<size_t>(output)].data();
    }

    braid_options_t options{};
    options.backend = BRAID_BACKEND_AVX512;
    options.path = BRAID_PATH_BRAID;
    options.num_threads = 1;
    options.microkernel_mode = BRAID_MICROKERNEL_FACTOR_GFNI;
    std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
        braid_create_code(k, r, &options), braid_destroy_code);
    Expect(code != nullptr, "GFNI code creation failed");
    Expect(
        braid_encode(
            code.get(), data.data(), outputs.data(), block_size) ==
            BRAID_STATUS_OK,
        "GFNI encode failed");
    for (int output = 0; output < r; ++output) {
      Expect(
          parity[static_cast<size_t>(output)] ==
              expected[static_cast<size_t>(k + output)],
          "GFNI parity mismatch: k=" + std::to_string(k) +
              " r=" + std::to_string(r) +
              " output=" + std::to_string(output));
    }

    std::vector<const uint8_t*> available(
        static_cast<size_t>(k + r));
    for (int index = 0; index < k + r; ++index) {
      available[static_cast<size_t>(index)] =
          expected[static_cast<size_t>(index)].data();
    }
    for (int erased_count = 1;
         erased_count <= std::min(r, 4);
         ++erased_count) {
      std::vector<int> erased(static_cast<size_t>(erased_count));
      std::vector<std::vector<uint8_t>> recovered(
          static_cast<size_t>(erased_count),
          std::vector<uint8_t>(block_size));
      std::vector<uint8_t*> recovered_ptrs(
          static_cast<size_t>(erased_count));
      for (int erased_index = 0;
           erased_index < erased_count;
           ++erased_index) {
        erased[static_cast<size_t>(erased_index)] = erased_index;
        available[static_cast<size_t>(erased_index)] = nullptr;
        recovered_ptrs[static_cast<size_t>(erased_index)] =
            recovered[static_cast<size_t>(erased_index)].data();
      }
      Expect(
          braid_decode(
              code.get(),
              available.data(),
              erased.data(),
              erased_count,
              recovered_ptrs.data(),
              block_size) == BRAID_STATUS_OK,
          "GFNI decode failed");
      for (int erased_index = 0;
           erased_index < erased_count;
           ++erased_index) {
        Expect(
            recovered[static_cast<size_t>(erased_index)] ==
                expected[static_cast<size_t>(erased_index)],
            "GFNI recovery mismatch: k=" + std::to_string(k) +
                " r=" + std::to_string(r) +
                " erased_count=" + std::to_string(erased_count) +
                " output=" + std::to_string(erased_index));
        available[static_cast<size_t>(erased_index)] =
            expected[static_cast<size_t>(erased_index)].data();
      }
    }
  }
}

void TestGfniStreamingStoreAlignmentFallback() {
#if !defined(BRAID_ENABLE_AVX512_PLAN_JIT)
  return;
#endif
  if (!braid_gfni_available()) {
    return;
  }

  constexpr int k = 10;
  constexpr int r = 4;
  constexpr size_t block_size = size_t{4} << 20;
  const auto expected =
      BuildDenseCodeword(k, r, block_size, 0x13579bdfu);
  std::vector<const uint8_t*> inputs(static_cast<size_t>(k));
  for (int input = 0; input < k; ++input) {
    inputs[static_cast<size_t>(input)] =
        expected[static_cast<size_t>(input)].data();
  }

  braid_options_t options{};
  options.backend = BRAID_BACKEND_AVX512;
  options.path = BRAID_PATH_BRAID;
  options.num_threads = 1;
  options.microkernel_mode = BRAID_MICROKERNEL_FACTOR_GFNI;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "GFNI streaming-store code creation failed");
  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> short_plan(
      braid_build_encode_plan_for_block_size(code.get(), size_t{1} << 20),
      braid_destroy_plan);
  Expect(short_plan != nullptr, "GFNI short plan creation failed");
  Expect(
      braid_plan_uses_streaming_stores(short_plan.get()) == 0,
      "GFNI short-chain cache-resident plan did not select cached stores");
  braid_options_t parallel_options = options;
  parallel_options.num_threads = 4;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> parallel_code(
      braid_create_code(k, r, &parallel_options), braid_destroy_code);
  Expect(parallel_code != nullptr, "GFNI parallel code creation failed");
  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> parallel_plan(
      braid_build_encode_plan_for_block_size(
          parallel_code.get(), size_t{1} << 20),
      braid_destroy_plan);
  Expect(parallel_plan != nullptr, "GFNI parallel plan creation failed");
  Expect(
      braid_plan_uses_streaming_stores(parallel_plan.get()) != 0,
      "GFNI parallel short-chain plan did not select streaming stores");
  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> plan(
      braid_build_encode_plan_for_block_size(code.get(), block_size),
      braid_destroy_plan);
  Expect(plan != nullptr, "GFNI streaming-store plan creation failed");
  Expect(
      braid_plan_uses_streaming_stores(plan.get()) != 0,
      "GFNI large-block plan did not enable streaming stores");

  std::vector<std::vector<uint8_t>> storage(
      static_cast<size_t>(r), std::vector<uint8_t>(block_size + 65));
  std::vector<uint8_t*> outputs(static_cast<size_t>(r));
  const auto set_output_alignment = [&](size_t displacement) {
    for (int output = 0; output < r; ++output) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(
          storage[static_cast<size_t>(output)].data());
      const uintptr_t aligned = (base + 63u) & ~uintptr_t{63};
      outputs[static_cast<size_t>(output)] =
          reinterpret_cast<uint8_t*>(aligned + displacement);
    }
  };
  const auto verify = [&](const char* failure) {
    for (int output = 0; output < r; ++output) {
      Expect(
          std::memcmp(
              outputs[static_cast<size_t>(output)],
              expected[static_cast<size_t>(k + output)].data(),
              block_size) == 0,
          failure);
    }
  };

  set_output_alignment(0);
  Expect(
      braid_execute_encode_plan(
          code.get(),
          plan.get(),
          inputs.data(),
          outputs.data(),
          block_size) == BRAID_STATUS_OK,
      "aligned GFNI streaming-store encode failed");
  verify("aligned GFNI streaming-store parity mismatch");

  set_output_alignment(1);
  Expect(
      braid_execute_encode_plan(
          code.get(),
          plan.get(),
          inputs.data(),
          outputs.data(),
          block_size) == BRAID_STATUS_OK,
      "unaligned GFNI cached-store fallback encode failed");
  verify("unaligned GFNI cached-store fallback parity mismatch");
}

void TestBatchEncodeDecodeRoundTrip() {
  constexpr int k = 6;
  constexpr int r = 3;
  constexpr int n = k + r;
  constexpr size_t stripe_count = 5;
  constexpr size_t block_size = 777;

  auto options = BraidOptions(BRAID_BACKEND_SCALAR, 3);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "batch create code");

  std::vector<std::vector<std::vector<uint8_t>>> stripes;
  stripes.reserve(stripe_count);
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    stripes.push_back(BuildDenseCodeword(
        k, r, block_size, 0x10203040u + static_cast<uint32_t>(stripe)));
    for (int output = 0; output < r; ++output) {
      std::fill(
          stripes[stripe][static_cast<size_t>(k + output)].begin(),
          stripes[stripe][static_cast<size_t>(k + output)].end(),
          0);
    }
  }

  std::vector<std::vector<const uint8_t*>> data_ptrs(
      stripe_count, std::vector<const uint8_t*>(k));
  std::vector<std::vector<uint8_t*>> parity_ptrs(
      stripe_count, std::vector<uint8_t*>(r));
  std::vector<const uint8_t* const*> data_batch(stripe_count);
  std::vector<uint8_t* const*> parity_batch(stripe_count);
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    for (int input = 0; input < k; ++input) {
      data_ptrs[stripe][static_cast<size_t>(input)] =
          stripes[stripe][static_cast<size_t>(input)].data();
    }
    for (int output = 0; output < r; ++output) {
      parity_ptrs[stripe][static_cast<size_t>(output)] =
          stripes[stripe][static_cast<size_t>(k + output)].data();
    }
    data_batch[stripe] = data_ptrs[stripe].data();
    parity_batch[stripe] = parity_ptrs[stripe].data();
  }
  Expect(
      braid_encode_batch(
          code.get(),
          data_batch.data(),
          parity_batch.data(),
          stripe_count,
          block_size) == BRAID_STATUS_OK,
      "batch encode");

  const auto generator = braid::core::BuildSystematicGeneratorMatrix(k, r);
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    for (int output = 0; output < r; ++output) {
      for (size_t pos = 0; pos < block_size; ++pos) {
        uint8_t expected = 0;
        for (int input = 0; input < k; ++input) {
          expected ^= braid::core::GF256::Multiply(
              generator[static_cast<size_t>(k + output)]
                       [static_cast<size_t>(input)],
              stripes[stripe][static_cast<size_t>(input)][pos]);
        }
        Expect(
            stripes[stripe][static_cast<size_t>(k + output)][pos] == expected,
            "batch parity mismatch");
      }
    }
  }

  int erased[2] = {0, 1};
  std::vector<std::vector<const uint8_t*>> available(
      stripe_count, std::vector<const uint8_t*>(n));
  std::vector<std::vector<std::vector<uint8_t>>> recovered(
      stripe_count,
      std::vector<std::vector<uint8_t>>(
          2, std::vector<uint8_t>(block_size)));
  std::vector<std::vector<uint8_t*>> recovered_ptrs(
      stripe_count, std::vector<uint8_t*>(2));
  std::vector<const uint8_t* const*> available_batch(stripe_count);
  std::vector<uint8_t* const*> recovered_batch(stripe_count);
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    for (int index = 0; index < n; ++index) {
      available[stripe][static_cast<size_t>(index)] =
          index < 2 ? nullptr : stripes[stripe][static_cast<size_t>(index)].data();
    }
    for (int output = 0; output < 2; ++output) {
      recovered_ptrs[stripe][static_cast<size_t>(output)] =
          recovered[stripe][static_cast<size_t>(output)].data();
    }
    available_batch[stripe] = available[stripe].data();
    recovered_batch[stripe] = recovered_ptrs[stripe].data();
  }
  Expect(
      braid_decode_batch(
          code.get(),
          available_batch.data(),
          erased,
          2,
          recovered_batch.data(),
          stripe_count,
          block_size) == BRAID_STATUS_OK,
      "batch decode");
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    for (int output = 0; output < 2; ++output) {
      Expect(
          recovered[stripe][static_cast<size_t>(output)] ==
              stripes[stripe][static_cast<size_t>(output)],
          "batch recovery mismatch");
    }
  }
}

void TestBatchUnalignedOutputFallback() {
  if (!braid_backend_available(BRAID_BACKEND_AVX512)) {
    return;
  }
  constexpr int k = 24;
  constexpr int r = 4;
  constexpr size_t stripe_count = 2;
  constexpr size_t block_size = 1024;

  auto options = BraidOptions(BRAID_BACKEND_AVX512, 2);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "unaligned batch create code");

  std::vector<std::vector<std::vector<uint8_t>>> expected;
  std::vector<std::vector<std::vector<uint8_t>>> output_storage(
      stripe_count,
      std::vector<std::vector<uint8_t>>(
          r, std::vector<uint8_t>(block_size + 1)));
  std::vector<std::vector<const uint8_t*>> data_ptrs(
      stripe_count, std::vector<const uint8_t*>(k));
  std::vector<std::vector<uint8_t*>> output_ptrs(
      stripe_count, std::vector<uint8_t*>(r));
  std::vector<const uint8_t* const*> data_batch(stripe_count);
  std::vector<uint8_t* const*> output_batch(stripe_count);
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    expected.push_back(BuildDenseCodeword(
        k, r, block_size, 0x44556677u + static_cast<uint32_t>(stripe)));
    for (int input = 0; input < k; ++input) {
      data_ptrs[stripe][static_cast<size_t>(input)] =
          expected[stripe][static_cast<size_t>(input)].data();
    }
    for (int output = 0; output < r; ++output) {
      output_ptrs[stripe][static_cast<size_t>(output)] =
          output_storage[stripe][static_cast<size_t>(output)].data() + 1;
    }
    data_batch[stripe] = data_ptrs[stripe].data();
    output_batch[stripe] = output_ptrs[stripe].data();
  }

  Expect(
      braid_encode_batch(
          code.get(),
          data_batch.data(),
          output_batch.data(),
          stripe_count,
          block_size) == BRAID_STATUS_OK,
      "unaligned batch encode fallback");
  for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
    for (int output = 0; output < r; ++output) {
      Expect(
          std::memcmp(
              output_ptrs[stripe][static_cast<size_t>(output)],
              expected[stripe][static_cast<size_t>(k + output)].data(),
              block_size) == 0,
          "unaligned batch parity mismatch");
    }
  }
}

std::vector<int> MixedErasurePattern(int k, int erased_count) {
  switch (erased_count) {
    case 1:
      return {0};
    case 2:
      return {0, k};
    case 3:
      return {0, 1, k};
    default:
      return {0, 1, k, k + 1};
  }
}

void RunBraidCorrectnessCase(
    braid_backend_t backend,
    int k,
    int r,
    size_t block_size,
    uint32_t seed,
    int output_batch = 4,
    braid_microkernel_mode_t microkernel = BRAID_MICROKERNEL_BASELINE) {
  auto expected = BuildDenseCodeword(k, r, block_size, seed);
  auto options = BraidOptions(backend);
  options.output_batch = output_batch;
  options.microkernel_mode = microkernel;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "matrix create failed");

  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> encode_plan(
      braid_build_encode_plan_for_block_size(code.get(), block_size), braid_destroy_plan);
  Expect(encode_plan != nullptr, "matrix encode plan failed");
  const size_t vector_width = backend == BRAID_BACKEND_AVX512 ? 64 : 32;
  Expect(braid_plan_tile_bytes(encode_plan.get()) % vector_width == 0, "tile stride must be SIMD aligned");

  std::vector<const uint8_t*> data_ptrs(static_cast<size_t>(k));
  std::vector<std::vector<uint8_t>> parity(
      static_cast<size_t>(r), std::vector<uint8_t>(block_size, 0xa5));
  std::vector<uint8_t*> parity_ptrs(static_cast<size_t>(r));
  for (int input = 0; input < k; ++input) {
    data_ptrs[static_cast<size_t>(input)] = expected[static_cast<size_t>(input)].data();
  }
  for (int output = 0; output < r; ++output) {
    parity_ptrs[static_cast<size_t>(output)] = parity[static_cast<size_t>(output)].data();
  }
  Expect(
      braid_execute_encode_plan(
          code.get(), encode_plan.get(), data_ptrs.data(), parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
      "matrix encode failed");
  for (int output = 0; output < r; ++output) {
    Expect(
        parity[static_cast<size_t>(output)] == expected[static_cast<size_t>(k + output)],
        "matrix parity mismatch: backend=" + std::to_string(static_cast<int>(backend)) +
            " k=" + std::to_string(k) + " r=" + std::to_string(r) +
            " size=" + std::to_string(block_size) + " out=" + std::to_string(output));
  }

  const int max_erased = std::min(r, 4);
  for (int erased_count = 1; erased_count <= max_erased; ++erased_count) {
    const std::vector<int> erased = MixedErasurePattern(k, erased_count);
    std::vector<const uint8_t*> available(static_cast<size_t>(k + r));
    for (int index = 0; index < k + r; ++index) {
      available[static_cast<size_t>(index)] = expected[static_cast<size_t>(index)].data();
    }
    for (int index : erased) {
      available[static_cast<size_t>(index)] = nullptr;
    }
    std::vector<std::vector<uint8_t>> recovered(
        static_cast<size_t>(erased_count), std::vector<uint8_t>(block_size, 0x5a));
    std::vector<uint8_t*> recovered_ptrs(static_cast<size_t>(erased_count));
    for (int output = 0; output < erased_count; ++output) {
      recovered_ptrs[static_cast<size_t>(output)] = recovered[static_cast<size_t>(output)].data();
    }
    const braid_status_t status = braid_decode(
        code.get(),
        available.data(),
        erased.data(),
        erased_count,
        recovered_ptrs.data(),
        block_size);
    Expect(status == BRAID_STATUS_OK, "matrix decode failed");
    for (int output = 0; output < erased_count; ++output) {
      Expect(
          recovered[static_cast<size_t>(output)] == expected[static_cast<size_t>(erased[output])],
          "matrix recovery mismatch: backend=" + std::to_string(static_cast<int>(backend)) +
              " k=" + std::to_string(k) + " r=" + std::to_string(r) +
              " size=" + std::to_string(block_size) + " e=" + std::to_string(erased_count) +
              " out=" + std::to_string(output));
    }
  }
}

void TestLongChainPureFactorKernels() {
  if (!braid_backend_available(BRAID_BACKEND_AVX512)) {
    return;
  }
  RunBraidCorrectnessCase(
      BRAID_BACKEND_AVX512, 64, 4, 257, 0x640401u, 1, BRAID_MICROKERNEL_FACTOR_REGISTER);
  RunBraidCorrectnessCase(
      BRAID_BACKEND_AVX512, 100, 4, 4097, 0x1000401u, 1, BRAID_MICROKERNEL_FACTOR_SCRATCH);
}

void TestVectorCorrectnessMatrix() {
  const std::vector<std::pair<int, int>> codes = {
      {4, 2}, {10, 4}, {20, 4}, {44, 6}, {64, 4}, {100, 4}};
  const std::vector<size_t> block_sizes = {1, 31, 32, 63, 64, 65, 257, 4096};
  std::vector<braid_backend_t> backends;
  if (braid_backend_available(BRAID_BACKEND_AVX2)) {
    backends.push_back(BRAID_BACKEND_AVX2);
  }
  if (braid_backend_available(BRAID_BACKEND_AVX512)) {
    backends.push_back(BRAID_BACKEND_AVX512);
  }
  for (braid_backend_t backend : backends) {
    for (const auto& [k, r] : codes) {
      for (size_t block_size : block_sizes) {
        RunBraidCorrectnessCase(
            backend,
            k,
            r,
            block_size,
            static_cast<uint32_t>(backend * 100000 + k * 1000 + r * 10 + block_size));
      }
    }
  }
}

void TestLongChainParallelConsistency() {
  constexpr int k = 100;
  constexpr int r = 4;
  constexpr size_t block_size = 1u << 20;
  std::vector<std::vector<uint8_t>> data(k, std::vector<uint8_t>(block_size));
  uint32_t seed = 0x7212026u;
  for (auto& block : data) {
    for (uint8_t& value : block) {
      seed = seed * 1664525u + 1013904223u;
      value = static_cast<uint8_t>(seed >> 24);
    }
  }

  auto single_options = BraidOptions(BRAID_BACKEND_AVX512, 1);
  single_options.tile_bytes = 4096;
  auto parallel_options = single_options;
  parallel_options.num_threads = 4;
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> single(
      braid_create_code(k, r, &single_options), braid_destroy_code);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> parallel(
      braid_create_code(k, r, &parallel_options), braid_destroy_code);
  Expect(single != nullptr && parallel != nullptr, "long-chain context creation failed");

  std::vector<const uint8_t*> data_ptrs(k);
  for (int input = 0; input < k; ++input) {
    data_ptrs[static_cast<size_t>(input)] = data[static_cast<size_t>(input)].data();
  }
  std::vector<std::vector<uint8_t>> single_parity(r, std::vector<uint8_t>(block_size));
  std::vector<std::vector<uint8_t>> parallel_parity(r, std::vector<uint8_t>(block_size));
  std::vector<uint8_t*> single_parity_ptrs(r);
  std::vector<uint8_t*> parallel_parity_ptrs(r);
  for (int output = 0; output < r; ++output) {
    single_parity_ptrs[static_cast<size_t>(output)] = single_parity[static_cast<size_t>(output)].data();
    parallel_parity_ptrs[static_cast<size_t>(output)] = parallel_parity[static_cast<size_t>(output)].data();
  }
  Expect(
      braid_encode(single.get(), data_ptrs.data(), single_parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
      "single-thread long-chain encode failed");
  Expect(
      braid_encode(parallel.get(), data_ptrs.data(), parallel_parity_ptrs.data(), block_size) == BRAID_STATUS_OK,
      "parallel long-chain encode failed");
  Expect(single_parity == parallel_parity, "single/parallel long-chain encode mismatch");

  const std::vector<int> erased = {0, 1, 2, 3};
  std::vector<const uint8_t*> available(k + r);
  for (int index = 0; index < k; ++index) {
    available[static_cast<size_t>(index)] = data[static_cast<size_t>(index)].data();
  }
  for (int output = 0; output < r; ++output) {
    available[static_cast<size_t>(k + output)] = single_parity[static_cast<size_t>(output)].data();
  }
  for (int index : erased) {
    available[static_cast<size_t>(index)] = nullptr;
  }
  std::vector<std::vector<uint8_t>> single_recovered(r, std::vector<uint8_t>(block_size));
  std::vector<std::vector<uint8_t>> parallel_recovered(r, std::vector<uint8_t>(block_size));
  std::vector<uint8_t*> single_recovered_ptrs(r);
  std::vector<uint8_t*> parallel_recovered_ptrs(r);
  for (int output = 0; output < r; ++output) {
    single_recovered_ptrs[static_cast<size_t>(output)] = single_recovered[static_cast<size_t>(output)].data();
    parallel_recovered_ptrs[static_cast<size_t>(output)] = parallel_recovered[static_cast<size_t>(output)].data();
  }
  Expect(
      braid_decode(
          single.get(), available.data(), erased.data(), r, single_recovered_ptrs.data(), block_size) ==
          BRAID_STATUS_OK,
      "single-thread long-chain decode failed");
  Expect(
      braid_decode(
          parallel.get(), available.data(), erased.data(), r, parallel_recovered_ptrs.data(), block_size) ==
          BRAID_STATUS_OK,
      "parallel long-chain decode failed");
  Expect(single_recovered == parallel_recovered, "single/parallel long-chain decode mismatch");
  for (int output = 0; output < r; ++output) {
    Expect(
        single_recovered[static_cast<size_t>(output)] == data[static_cast<size_t>(output)],
        "long-chain recovered data mismatch");
  }
}

void TestApiValidation() {
  braid_options_t invalid_generator{};
  invalid_generator.backend = BRAID_BACKEND_AVX512;
  invalid_generator.path = BRAID_PATH_BRAID;
  invalid_generator.repr_generator = 3;
  Expect(braid_create_code(10, 4, &invalid_generator) == nullptr, "invalid generator must be rejected");
  Expect(braid_create_code(0, 4, nullptr) == nullptr, "invalid k must be rejected");
  Expect(braid_create_code(252, 4, nullptr) == nullptr, "n > 255 must be rejected");

  constexpr int k = 6;
  constexpr int r = 3;
  constexpr size_t block_size = 257;
  auto options = BraidOptions(BRAID_BACKEND_AVX512);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code_a(
      braid_create_code(k, r, &options), braid_destroy_code);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code_b(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code_a != nullptr && code_b != nullptr, "validation contexts failed");
  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> encode_plan(
      braid_build_encode_plan_for_block_size(code_a.get(), block_size), braid_destroy_plan);
  const int erased_index = 0;
  std::unique_ptr<braid_plan_t, decltype(&braid_destroy_plan)> decode_plan(
      braid_build_decode_plan_for_block_size(code_a.get(), &erased_index, 1, block_size), braid_destroy_plan);
  Expect(encode_plan != nullptr && decode_plan != nullptr, "validation plans failed");

  auto codeword = BuildDenseCodeword(k, r, block_size, 123u);
  std::vector<const uint8_t*> inputs(k + r);
  std::vector<std::vector<uint8_t>> outputs(r, std::vector<uint8_t>(block_size));
  std::vector<uint8_t*> output_ptrs(r);
  for (int index = 0; index < k + r; ++index) {
    inputs[static_cast<size_t>(index)] = codeword[static_cast<size_t>(index)].data();
  }
  for (int output = 0; output < r; ++output) {
    output_ptrs[static_cast<size_t>(output)] = outputs[static_cast<size_t>(output)].data();
  }
  Expect(
      braid_execute_encode_plan(
          code_b.get(), encode_plan.get(), inputs.data(), output_ptrs.data(), block_size) ==
          BRAID_STATUS_INVALID_ARGUMENT,
      "cross-context plan must be rejected");
  Expect(
      braid_execute_encode_plan(
          code_a.get(), decode_plan.get(), inputs.data(), output_ptrs.data(), block_size) ==
          BRAID_STATUS_INVALID_ARGUMENT,
      "decode plan used for encode must be rejected");
  Expect(
      braid_execute_encode_plan(
          code_a.get(), encode_plan.get(), inputs.data(), output_ptrs.data(), block_size + 1) ==
          BRAID_STATUS_INVALID_ARGUMENT,
      "planned block-size mismatch must be rejected");

  const int duplicate_erased[2] = {0, 0};
  Expect(
      braid_decode(
          code_a.get(), inputs.data(), duplicate_erased, 2, output_ptrs.data(), block_size) ==
          BRAID_STATUS_INVALID_ARGUMENT,
      "duplicate erasures must be rejected");
  const int out_of_range = k + r;
  Expect(
      braid_decode(
          code_a.get(), inputs.data(), &out_of_range, 1, output_ptrs.data(), block_size) ==
          BRAID_STATUS_INVALID_ARGUMENT,
      "out-of-range erasure must be rejected");
}

void TestConcurrentContextUse() {
  constexpr int k = 10;
  constexpr int r = 4;
  constexpr size_t block_size = 4096;
  constexpr int callers = 4;
  auto expected = BuildDenseCodeword(k, r, block_size, 0xabcdefu);
  auto options = BraidOptions(BRAID_BACKEND_AVX512, 4);
  std::unique_ptr<braid_code_t, decltype(&braid_destroy_code)> code(
      braid_create_code(k, r, &options), braid_destroy_code);
  Expect(code != nullptr, "concurrent context creation failed");

  std::vector<const uint8_t*> data_ptrs(k);
  for (int input = 0; input < k; ++input) {
    data_ptrs[static_cast<size_t>(input)] = expected[static_cast<size_t>(input)].data();
  }
  std::vector<std::vector<std::vector<uint8_t>>> parity(
      callers,
      std::vector<std::vector<uint8_t>>(r, std::vector<uint8_t>(block_size)));
  std::vector<braid_status_t> statuses(callers, BRAID_STATUS_INTERNAL_ERROR);
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (int caller = 0; caller < callers; ++caller) {
    threads.emplace_back([&, caller]() {
      std::vector<uint8_t*> output_ptrs(r);
      for (int output = 0; output < r; ++output) {
        output_ptrs[static_cast<size_t>(output)] =
            parity[static_cast<size_t>(caller)][static_cast<size_t>(output)].data();
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      statuses[static_cast<size_t>(caller)] =
          braid_encode(code.get(), data_ptrs.data(), output_ptrs.data(), block_size);
    });
  }
  while (ready.load(std::memory_order_acquire) != callers) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (int caller = 0; caller < callers; ++caller) {
    Expect(statuses[static_cast<size_t>(caller)] == BRAID_STATUS_OK, "concurrent encode failed");
    for (int output = 0; output < r; ++output) {
      Expect(
          parity[static_cast<size_t>(caller)][static_cast<size_t>(output)] ==
              expected[static_cast<size_t>(k + output)],
          "concurrent encode mismatch");
    }
  }
}

void TestThreadPoolGenerationStress() {
  const auto topology = braid::runtime::DetectTopology();
  const auto cpu_order = braid::runtime::BuildCpuBindingOrder(topology, 4, true);
  if (cpu_order.size() < 4) {
    return;
  }

  braid::runtime::ThreadPool pool(cpu_order);
  const int capacity = std::min<int>(4, static_cast<int>(cpu_order.size()));
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  constexpr int kStressRounds = 100;
#else
  constexpr int kStressRounds = 1000;
#endif
  for (int round = 0; round < kStressRounds; ++round) {
    const int threads = 2 + round % (capacity - 1);
    std::atomic<int> completed{0};
    pool.Run(
        threads,
        [&](int) { completed.fetch_add(1, std::memory_order_relaxed); },
        (round & 1) == 0);
    Expect(completed.load(std::memory_order_relaxed) == threads, "thread-pool generation lost work");
  }
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"GF256Basic", TestGF256Basic},
      {"GfniAffineMatrices", TestGfniAffineMatrices},
      {"GfniFactorizedKernels", TestGfniFactorizedKernels},
      {"GfniStreamingStoreAlignmentFallback", TestGfniStreamingStoreAlignmentFallback},
      {"DecompositionProperties", TestDecompositionProperties},
      {"CustomOrderedDecompositionProperties", TestCustomOrderedDecompositionProperties},
      {"GeneratorMatrix", TestGeneratorMatrix},
      {"PublicApiSmoke", TestPublicApiSmoke},
      {"AutoPathNeverSelectsDense", TestAutoPathNeverSelectsDense},
      {"EncodeDecodeRoundTrip", TestEncodeDecodeRoundTrip},
      {"InternalParallelRoundTrip", TestInternalParallelRoundTrip},
      {"BatchEncodeDecodeRoundTrip", TestBatchEncodeDecodeRoundTrip},
      {"BatchUnalignedOutputFallback", TestBatchUnalignedOutputFallback},
      {"VectorBackendsWhenAvailable", TestVectorBackendsWhenAvailable},
      {"RetunedRepresentationRoundTrip", TestRetunedRepresentationRoundTrip},
      {"RetuneSingleOutputDecodeUsesProgressionOrder", TestRetuneSingleOutputDecodeUsesProgressionOrder},
      {"VectorCorrectnessMatrix", TestVectorCorrectnessMatrix},
      {"LongChainPureFactorKernels", TestLongChainPureFactorKernels},
      {"LongChainParallelConsistency", TestLongChainParallelConsistency},
      {"ApiValidation", TestApiValidation},
      {"ConcurrentContextUse", TestConcurrentContextUse},
      {"ThreadPoolGenerationStress", TestThreadPoolGenerationStress},
  };

  int failed = 0;
  for (const auto& test : tests) {
    try {
      test.second();
      std::cout << "[PASS] " << test.first << '\n';
    } catch (const std::exception& ex) {
      ++failed;
      std::cerr << "[FAIL] " << test.first << ": " << ex.what() << '\n';
    }
  }

  if (failed != 0) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }

  std::cout << "All tests passed\n";
  return 0;
}
