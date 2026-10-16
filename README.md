# BRAID 0.1.0

Anonymous, unpublished engineering review candidate. **The main-library license
is pending rights-holder approval. This is not an authorized open-source release.**
No project URL is published. See `LICENSE`, `NOTICE` and `SOURCE_PROVENANCE.md`.

BRAID is a C ABI / C++17 implementation of systematic Reed-Solomon erasure coding
over GF(256), polynomial `0x139`, using pure factorized execution. The primary
algorithm and schedules are imported from a checksummed frozen baseline. 
ISA-L uses different field and matrix conventions. Reference multiplication is used only in tests;
Tests use reference multiplication. The library path is factorized execution.

## Build And Test

Requirements: Linux, CMake >= 3.19, GCC/Clang with C++17, a C compiler, pthreads.
The default offline build includes static/shared libraries, tests and examples.
No downloads, ISA-L installation or network access are needed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cmake -E chdir build ctest --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Use `BRAID_BUILD_STATIC=OFF` or `BRAID_BUILD_SHARED=OFF` for one library type.
`BRAID_ENABLE_X86=OFF` makes a scalar-only build.
`BRAID_ENABLE_AVX512_PLAN_JIT=OFF` removes the AsmJit dependency from the binary;
factorized non-JIT schedules remain available. `BRAID_BUILD_TESTS=OFF` and
`BRAID_BUILD_EXAMPLES=OFF` disable development executables. The diagnostic bench
requires explicit `BRAID_BUILD_BENCHMARKS=ON`; it has no ISA-L dependency and is
not a paper performance harness. Never add `-march=native` or global ISA flags
to a redistributable build, since that would bypass baseline dispatch safety.

## Use

Only `include/braid/` is public. No internal C++ or AsmJit headers are installed.
The shared object exports only the versioned `braid_*` C ABI. Its SONAME major
is 0; the layout ID reported by `braid_abi_version()` is 1. These are distinct.
ABI v1 options and enum values are frozen; future incompatible layouts must
use a new API and SONAME, not extend this struct in place.

```c
#include <braid/braid.h>
braid_options_t options;
braid_options_init(&options, sizeof(options));
options.num_threads = 1; /* Default: caller-controlled parallelism. */
braid_code_t *code = NULL;
braid_status_t status = braid_create_code_ex(6, 3, &options, sizeof(options), &code);
/* Check status, then braid_encode(code, data, parity, block_size). */
braid_destroy_code(code);
```

Complete runnable encode/recovery examples: `examples/encode.c` and
`examples/encode.cpp`. The optional `braid/braid.hpp` wrapper provides C++17 RAII.
See `docs/API.md` for pointer ordering, lifetimes and concurrency contracts.

```cmake
find_package(BRAID 0.1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE BRAID::braid)
# Or select BRAID::static / BRAID::shared when that variant is installed.
```

```sh
export PKG_CONFIG_PATH="$PWD/install/lib/pkgconfig"
cc examples/encode.c $(pkg-config --cflags --libs braid) -o example
LD_LIBRARY_PATH="$PWD/install/lib" ./example
# For a static-only installation, including the private C++/thread dependencies:
cc examples/encode.c $(pkg-config --cflags --libs --static braid) -o example-static
```

The latter selects a static archive only when a shared library is absent (or
the linker is explicitly directed to static BRAID); `--static` supplies private
dependencies, it does not itself select an archive. Both library variants are
PIC. CMake exports and pkg-config metadata are relocatable. Examples assume
`CMAKE_INSTALL_LIBDIR=lib`; adjust the search directory if using a multilib layout.


## Anonymous Source Package

Shipped paths are listed in `PACKAGE_FILES.txt`, with hashes in `SOURCE_SHA256SUMS`.
Third-party attributions remain with the upstream files. The archive is built
locally; this workflow does not upload or publish it. Build outputs, logs and
private provenance are omitted. A review candidate does not grant a license or
authorize public distribution. See `docs/RELEASE.md`.
