# Native ARM Integration Assessment

Status: evaluated, not integrated or claimed supported. The checksummed
`NATIVE-ARM-01` lineage is an independently evolved implementation, not a renamed
copy of `FROZEN-X86-01`. This assessment is based on source/include diffs and
inspection of the native kernel, IR, emitter, planner and build rules, not on its
directory name. Original private paths are intentionally omitted.

## Coherent Import Scope

| Area | Required source-level work |
| --- | --- |
| Kernel | Native `neon_kernels.cc`: NEON factor-plane fallback plus native JIT execution |
| IR | Add `neon_factor_ir.h/.cc`: route dictionary, plane scheduling, inverse output transforms, packing/layout policies and factorized interpreter |
| Emitter | Add `neon_plan_jit.h/.cc`: AArch64 emission, executable allocation lifetime, packing, tails and layout metadata |
| Planning | Reconcile `context.cc/.h`, `plan.cc`, `types.h`: NEON caps, cache budget, tile selection, plan ownership and finalization |
| Runtime | Reconcile `backend_dispatch.cc`: NEON selection, work partition alignment and minimum-chunk rules; preserve the release exception/fence protections |
| Shared helpers | Reconcile architecture guards in `vector_common.h`; do not include x86 intrinsic headers in an ARM build |
| Public C ABI | Reserve backend value 4, decide microkernel value 14 and additive packing/layout queries; preserve existing structure layout and ownership contracts |
| Build/dependency | Build the AsmJit ARM emitter sources without disabling AArch64; verify dependency snapshot, PIC, hidden exports and older Linux runtime-library requirements |
| Tests | Import the native correctness suite separately; integrate full-byte oracle checks without pulling ISA-L or experiment harnesses into normal builds |

Relative to the frozen baseline, the native src/ tree has eight modified files
and four added IR/JIT files; the public header also changes. CMake and native tests
add further integration work. A single-kernel replacement would omit essential
plan state, ownership and instruction-generation dependencies.

The native public header also adds `braid_plan_uses_input_packing()` and
`braid_plan_neon_layout()`. Its diagnostics consume native route counts, while the
frozen diagnostics consume x86 JIT counts. Their meanings need explicit API
documentation, not an unreviewed wholesale header replacement. Its main options
structure layout does not itself need expansion for the observed additions.

## Integration Boundary

Keep the native IR and emitter in a separate backend module. Select and attach
that module only for an enabled AArch64/NEON build. Merge the native planner and
work-partition rules as a coherent platform policy, preserving the frozen x86
policy rather than substituting native defaults globally. Retain a separately
queryable scalar backend on ARM. Remove native dense wrappers during import;
any non-JIT/tail fallback must remain the native factorized path.

Do not transplant the archived frozen NEON implementation and call it the native
optimized line. Do not silently map explicit NEON-JIT failures to dense execution.
Review optional CPU instructions emitted by each native layout, OS feature gates,
executable-memory policy and publication barriers before advertising that layout.

## Required Validation

1. AArch64 cross/native compile and install, C-only/C++ consumers, shared/static
   PIC/export checks and an offline dependency/license audit.
2. Native hardware full-byte encode/recovery oracle checks for short/long codes,
   unaligned/tail sizes, mixed erasures, representations and each enabled layout.
3. JIT-on/off, input packing, cached plans, batches, multiple context workers,
   concurrent callers and failure/exception cleanup. No remote performance
   measurement is needed to establish correctness support.
4. An x86 regression run to prove the isolated native module did not alter x86
   schedule selection or byte results, then anonymous repack and tar rebuild.

Without an authorized native validation environment, this cannot be closed by a
source copy or by an x86-only build. No performance equivalence or complete native
optimization coverage is asserted by the current release candidate.
