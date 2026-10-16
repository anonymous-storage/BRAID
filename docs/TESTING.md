# Validation Record

This is a sanitized local engineering record, not an experiment log or a
performance claim. No remote performance experiments were run. Raw build/test
logs and identity-bearing provenance are excluded from source packages.

Environment: Linux x86_64, GCC/G++ 11.4.0, CMake/CTest 3.19.0. The local CPU/OS
supports AVX2 and AVX-512 but does not expose GFNI. Builds used at most four
compiler jobs; CTest ran sequentially and internal test pools used at most four
workers. Main-library licensing remains PENDING regardless of test results.

## Completed Checks

| Configuration | Result |
| --- | --- |
| Release, static + shared, x86 SIMD objects and optional JIT compiled | 6/6 CTest passed |
| Release, scalar-only, static-only | 6/6 CTest passed |
| Release, x86 SIMD, JIT disabled, shared-only | 6/6 CTest passed |
| Debug, scalar-only static, AddressSanitizer + UndefinedBehaviorSanitizer | 6/6 CTest passed, leak detection enabled |
| Installed package copied to another prefix | C-only and C++17 downstream build/run passed |
| C-only consumer explicitly selecting static target | Build/run passed |
| Static archive linked into a C shared plugin | PIC/no-undefined link passed |
| Plain cc using relocated pkg-config shared metadata | Link/run passed |
| Plain cc using pkg-config --static private dependencies and explicit archive | Link/run passed |
| Actual anonymous source tar, extracted into a separate tree | Offline default build, 6/6 CTest, install, downstream C/C++ and static C passed |

The six CTest entries are: imported baseline suite (22 named checks), release API
and independent oracle tests, synthetic CPU/OS gate tests, thread-pool exception
and reuse tests, C example, and C++17 RAII example. Backend-dependent baseline
checks skip unsupported execution. A passing suite does **not** mean GFNI ran.

The independent bit-serial oracle checks full parity bytes and unsorted mixed
data/parity recovery. Cases cover zero-length single calls, unaligned buffers,
vector boundaries and tails, large parallel cells, canaries, invalid parameters,
cross-context plans, batches, concurrent context use and thread-pool generations.
The feature-gate test synthetically removes AVX2, AVX512F/BW/VL, GFNI, XSAVE,
OSXSAVE, AVX and XCR0 state bits. Hardware execution exercises scalar, AVX2 and
non-GFNI AVX-512;
AVX-512 preferences on the local non-GFNI CPU are verified to execute AVX-512,
including the ordinary JIT and the no-JIT build, rather than falling back to AVX2.
GFNI-only preferences on this CPU resolve to ordinary factorized plans.
Object disassembly verifies zero GFNI instructions in the ordinary AVX-512 object
and their presence in the separately compiled GFNI object. The release oracle
test asserts the resolved plan backend equals each available requested backend;
an AVX2 substitution therefore cannot masquerade as a passing AVX-512 test.

The shared library exposes 39 C function symbols plus its version node, and no
internal C++ or AsmJit symbols. No dense execution symbols remain in the static
archive. Dynamic inspection found no ISA-L/AsmJit shared dependency, TEXTREL or
embedded build RPATH/RUNPATH. The tested release library binaries contained no
original workspace prefix. Only source, never these binaries, is packaged.

The anonymous packaging gate was tested with a clean tree, identical repeated
packaging, and rejection of injected private identity, email, IPv4, home path,
ELF/NUL content, symbolic link, changed dependency attribution and path traversal.
Every member of the produced tar is re-read and scanned. The private report holds
the detailed checksums and policy match information. No identity terms are
repeated in this public record.

## Not Verified

GFNI numerical/JIT execution on a capable CPU; native ARM/NEON compilation
or execution; non-x86 scalar builds; Clang/libc++; Windows/macOS; 32-bit builds;
ThreadSanitizer; unusual NUMA/affinity restrictions; pool-construction resource
exhaustion; production performance or equivalence with published measurements.
Baseline SIMD vector-template attribute warnings remain in GCC builds; they did
not prevent the tested builds. The main license and ARM integration are release
follow-up items, not claims covered by these local results.
