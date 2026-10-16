# Source Provenance

This public manifest uses logical baseline IDs and SHA-256 only. It intentionally
contains no original filesystem paths, machine identifiers or author identities.
The private mapping is kept outside the release tree and is never packaged.

| Logical ID | Role |
| --- | --- |
| FROZEN-X86-01 | Primary frozen source and public header baseline |
| CANDIDATE-X86-01 | Candidate lineage, byte-identical src/ and include/ to the frozen baseline |
| CANDIDATE-TEST-01 | Original correctness suite imported as tests/baseline_test.cc |
| NATIVE-ARM-01 | Independently inspected native ARM fork; not merged or compiled |
| ASMJIT-SNAPSHOT-01 | Unmodified vendored dependency snapshot, separately licensed |

`BASELINE_SHA256SUMS` records individual original files, including inspected ARM
files that are not present in this release. These are **origin hashes**, not
checksums of edited release files. The package's `SOURCE_SHA256SUMS` identifies
the exact delivered files after engineering changes. The build pin accompanying
the local AsmJit snapshot named upstream commit
`0bd5787b54b575ed94bf32ac452153b34385c514`; no Git metadata is distributed and no
network verification of that commit was performed. File hashes are authoritative
for the actual dependency content, not an unverified commit-identity assertion.

## Deliberate Release Changes

- New offline CMake static/shared/install/export/pkg-config integration, C ABI
  symbol visibility, version accessors, initializer, status constructor, C++ RAII,
  examples, release tests, and anonymous allowlisted packaging.
- Runtime CPUID/XGETBV gate covers every ISA used by each compiled object. The
  same AVX-512 schedule source is built separately without and with GFNI, preserving
  the existing vector loops. A plan chooses its object once; unsupported GFNI
  preferences select the normal factorized planner, not a lower SIMD backend.
  Shared inline helper linkage is local to each ISA object to prevent cross-ISA
  COMDAT selection. Non-GFNI JIT emission remains independently usable.
- Dense entrypoints and dense SIMD implementations are removed. Dense public
  options are rejected; an independent bit-serial test oracle is test-only.
  Existing GF arithmetic for coefficient/planning and factorized tails remains.
- Startup defaults do not bind threads. Topology failure is nonfatal; integer
  sizes and options are checked, thread creation unwinds safely, and worker
  exceptions are joined and reported through the C boundary.
  SIMD scratch growth allocates before releasing the old buffer so an allocation
  failure cannot leave a dangling thread-local buffer for later use/destruction.
- Each worker fences non-temporal stores before exposing API completion; this
  repairs a library publication boundary without changing the GFNI schedule.
- Baseline tests use public availability checks and at most four internal workers.
  Factorized scalar/AVX2/AVX-512 schedule bodies are otherwise preserved.

The frozen NEON kernel exists but is disabled, not substituted for the independent
native ARM fork. Inspection found that the latter changes the planner, context,
types and dispatch and adds dedicated NEON IR/JIT files. Its kernel includes
`arm_neon.h`, its emitter uses AsmJit's AArch64 assembler, and its CMake integration
builds the dedicated factor IR and optional NEON JIT. Importing one file would
not reproduce that line. This candidate consequently does **not** claim native
ARM optimization coverage or coverage of every tested experimental branch.

The main-library rights and license require separate approval. Third-party
licenses are preserved under `third_party/` and summarized in `NOTICE`.
