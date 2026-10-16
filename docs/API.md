# C ABI Contract

Include `braid/braid.h` from C11 or C++; no public C++ runtime types cross the ABI.
Do not compile the public interface with `-fshort-enums` or nonstandard packing.
Use the matching installed headers and library. Status-returning functions do
not let C++ exceptions cross the boundary; pointer constructors return NULL on
failure. Use `braid_create_code_ex` for diagnostic status and options-size checks.
Allocation/thread/JIT failures may produce INTERNAL_ERROR without rolling back
partially written output buffers. Destroy functions accept NULL.

## Codes And Buffers

- `k >= 1`, `r >= 1`, `k + r <= 255`. Encode reads k data pointers and writes r
  parity pointers, each spanning `block_size` bytes. The systematic matrix is
  Vandermonde normalized by its first k rows, points `3^i` in GF(0x139).
- Decode takes an array of **k+r pointers indexed by original block number**.
  Only erased slots may be NULL. The planner chooses the first k survivors in
  ascending block-number order, not arbitrary compacted survivors. Supply every
  unerased block, even when a particular plan does not consume all of them.
- Erasure indices are unique, in `[0,k+r)`, with `1 <= erased_count <= r`.
  Recovered output pointers follow the **caller-supplied erasure order**, not
  sorted order. Parity erasures and mixed data/parity erasures are supported.
- Buffers may be unaligned and sizes need not be a vector multiple. All referenced
  pointers must be non-NULL, including zero-byte single-stripe calls; single
  calls with size zero are no-ops. Batch calls require nonzero count and size.
- Input and output memory ranges must not overlap; output ranges must be mutually
  disjoint, including across stripes. The caller supplies valid lengths and
  accessible memory. Pointer validity, allocation length and aliasing cannot be
  established by this API and are caller obligations.
- Sizes above PTRDIFF_MAX and overflowing batch spans are rejected. Tile sizes
  are auto-selected at zero or limited to 16 MiB; output batches are clamped to
  actual output count. Thread counts are 0 (treated as 1) or 1..256.

Batch APIs take an outer array of stripe pointers; each stripe uses the same
pointer indexing as the corresponding single API. Decode batches share one
erasure pattern. Unaligned streaming-store destinations select a factorized
cached-store schedule, never dense execution.

## Options And Plans

Call `braid_options_init(&options, sizeof(options))` before setting fields.
Passing NULL to a code constructor uses the same defaults. Paths AUTO and BRAID
both resolve to pure factorized BRAID. DENSE is an ABI-reserved value and is
rejected. NEON is a reserved unavailable backend. Unknown options are rejected.
GFNI availability is separate from AVX-512 availability. Unsupported GFNI-specific
microkernel preferences resolve to the ordinary factorized planner, which can
still execute AVX-512. Check `braid_gfni_available()` when actual GFNI is required.
Representation settings and microkernel tuning values are advanced interfaces
inherited from the frozen baseline; use defaults unless explicitly evaluating
them. Experimental `BRAID_*` environment overrides remain inherited from that
baseline and are outside the stable API; use an unset environment in production.

Plans belong to the context that created them. A plan cannot execute with another
context or with the wrong encode/decode function. A nonzero planned block size
must match execution size; zero means size-independent. `_batch_plan_` builders
only tune the plan for batch-like stores; explicit plan execution still handles
one stripe. To execute multiple stripes, use the batch code APIs.

The context caches plans when enabled. Applications with unbounded size/erasure
patterns should disable the cache or manage context lifetime; there is no eviction
policy in this baseline. Plan query functions accept NULL and return zero/default
sentinels. `braid_plan_path()` reports BRAID for valid plans. Exact schedule, tile,
batch and JIT diagnostics are observational, not performance/selection guarantees.

Keep the code and plan alive throughout each call. A plan may be destroyed after
its context, but cannot then be executed. Destruction must not race any operation.
Concurrent callers may share a context/plan if all writable buffers are disjoint.
No background work survives the completion of an execute call. Context destruction
joins its owned worker pool; callers retain ownership of all block storage.
