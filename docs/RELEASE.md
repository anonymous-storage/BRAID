# Review Candidate Checklist

The source pack is an anonymous **review candidate**, not a license grant and
not authorization to publish. Main-license selection and rights-holder approval
are outstanding. Do not publish, push or upload as part of this workflow.

## Build The Source Archive

Python 3.8+ is needed only for packaging, not for library builds. Maintain the
sorted explicit `PACKAGE_FILES.txt` allowlist when adding public files. Do not
replace it with a recursive archive of the working directory.

```sh
python3 scripts/package_source.py \
  --private-policy "$PRIVATE_POLICY" \
  --private-report "$PRIVATE_REPORT" \
  --out dist
```

Both private paths must be outside the source and archive trees. The policy is
a privately reviewed JSON object with `deny_tokens`: all self-identifying author,
account, institution, contact, host, remote and workspace identifiers, expressed
as literal case-insensitive strings. Do not check the policy into this source
distribution. The scan checks those tokens plus generic home paths, email, IP,
SSH remote and private-key patterns in every selected file and filename. It
rejects binary content, symlinks, unapproved file types and hidden/build paths.
The final tar is reopened and its actual members are checked again. Failed scans
do not produce a new archive; an existing archive from an earlier successful run
must not be treated as evidence for the changed tree.

The private report inventories each final member and checksum and records any
matched terms only privately. The public output prints a pass/fail summary and
archive hash, never identity terms. The archive has a generic top directory,
zero numeric ownership, empty owner/group names, normalized mode/mtime and no
absolute paths. `SOURCE_SHA256SUMS` checks the delivered source files. Repeated
packaging of identical source files is byte-for-byte reproducible.

Third-party source copyright, author names, URLs and contacts are not removed or
claimed as BRAID authorship. Generic attribution exceptions are restricted to
hash-verified unchanged `third_party/asmjit/` files and recorded separately in the
private report. Private identity deny-tokens still apply to those files. A
third-party file modification fails that whitelist and requires review.

This is a mechanical gate, not a guarantee against all forms of deanonymization.
The maintainer must review comments, dependency links, naming and policy coverage
as well as the generated report before any authorized submission.

## Required Before Public Release

1. Obtain the main-source rights/license decision and update LICENSE/NOTICE.
2. Review the anonymous source manifest and private report without adding private
   mappings, test logs, remote metadata or identities to the archive.
3. Run GFNI correctness/JIT tests on a GFNI-capable CPU. Local non-GFNI AVX-512
   numerical and ordinary JIT execution are verified; GFNI execution is not.
4. Separately integrate and validate the native ARM baseline if NEON is required;
   this candidate deliberately does not claim it. Validate any additional OS,
   architecture, compiler and ABI before widening the support statement.
5. Repeat build, CTest, relocation/install and downstream C/C++ tests from the
   source archive. Review exported symbols, static private dependencies and PIC.
6. Review baseline experimental environment overrides and cache bounds for the
   intended embedding environment. No performance equivalence is asserted.

Raw build/test output is private and excluded. Only a sanitized factual test
summary belongs in `docs/TESTING.md`. File-prefix/debug-prefix maps are applied
to library objects to avoid build paths in later binaries; this delivery is
source-only, so ELF objects, debug files and caches are never archived.
