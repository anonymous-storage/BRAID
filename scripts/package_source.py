#!/usr/bin/env python3
"""Local, allowlisted, anonymous source-only review packaging (no upload)."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOP = 'braid-0.1.0-review-candidate'
ALLOWED_DIRS = {'src', 'include', 'cmake', 'tests', 'examples', 'bench', 'scripts', 'docs', 'third_party'}
ALLOWED_ROOT = {'CMakeLists.txt', 'README.md', 'LICENSE', 'NOTICE', 'SOURCE_PROVENANCE.md',
                'BASELINE_SHA256SUMS', 'PACKAGE_FILES.txt', 'VERSION'}
ALLOWED_SUFFIXES = {'.h', '.hpp', '.c', '.cc', '.cpp', '.cmake', '.in', '.map', '.md', '.txt', '.py', '.natvis'}
BAD_PARTS = {'.git', '.codex', '.agents', '.braid-agents', '__pycache__', 'CMakeFiles', 'Testing', 'node_modules'}
PATTERNS = {
    'absolute-home': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+'),
    'email': re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}'),
    'ipv4': re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])'),
    'ssh-remote': re.compile(r'(?:ssh://|git@)[A-Za-z0-9][A-Za-z0-9.-]*(?::|/)[^\s]+'),
    'credential': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def inside(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private-policy', type=Path, required=True)
    ap.add_argument('--private-report', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'dist')
    args = ap.parse_args()
    out = args.out.resolve()
    report_path = args.private_report.resolve()
    policy_path = args.private_policy.resolve()
    if any(inside(p, ROOT) or inside(p, out) for p in (policy_path, report_path)):
        raise ValueError('private policy and report must remain outside the public tree and archive directory')
    policy = json.loads(policy_path.read_text())
    tokens = policy.get('deny_tokens', [])
    if not tokens or not all(isinstance(t, str) and len(t) >= 3 for t in tokens):
        raise ValueError('private policy must contain a reviewed nonempty deny_tokens list')
    tokens = [t.casefold() for t in tokens] + [str(ROOT).casefold()]
    names = (ROOT / 'PACKAGE_FILES.txt').read_text().splitlines()
    if names != sorted(set(names)) or not names:
        raise ValueError('source allowlist must be nonempty, sorted and unique')
    required = ALLOWED_ROOT | {'include/braid/braid.h', 'third_party/asmjit/LICENSE.md',
                               'scripts/package_source.py'}
    if not required.issubset(names):
        raise ValueError('source package is missing required metadata, API or dependency license')
    vendor_hashes = {}
    for line in (ROOT / 'BASELINE_SHA256SUMS').read_text().splitlines():
        sha, logical = line.split('  ', 1)
        prefix = 'ASMJIT-SNAPSHOT-01/'
        if logical.startswith(prefix):
            vendor_hashes['third_party/asmjit/' + logical[len(prefix):]] = sha
    failures, attributions, scanned = [], [], []

    def scan(name, data):
        # Identity terms apply even to third-party files. Generic attribution
        # exceptions require an exact hash match to the preserved dependency.
        text = data.decode('utf-8')
        both = name + '\n' + text
        vendor = name.startswith('third_party/asmjit/') and vendor_hashes.get(name) == digest(data)
        for token in tokens:
            if token in both.casefold():
                failures.append({'file': name, 'rule': 'private-token', 'token': token})
        if name.startswith('third_party/') and not vendor:
            failures.append({'file': name, 'rule': 'unverified-third-party'})
        for label, pattern in PATTERNS.items():
            matches = pattern.findall(both)
            if matches:
                entry = {'file': name, 'rule': label, 'matches': matches}
                (attributions if vendor and label != 'credential' else failures).append(entry)
        if '\x00' in text or data.startswith(b'\x7fELF'):
            failures.append({'file': name, 'rule': 'binary-content'})
        scanned.append({'file': name, 'sha256': digest(data), 'bytes': len(data), 'attribution_scope': vendor})

    files = {}
    for name in names:
        rel = PurePosixPath(name)
        if (rel.is_absolute() or '..' in rel.parts or str(rel) != name or
                any(p in BAD_PARTS or p.startswith('.') for p in rel.parts) or
                any(p.startswith('build') for p in rel.parts[:-1])):
            raise ValueError('unsafe source allowlist path')
        if len(rel.parts) == 1:
            if name not in ALLOWED_ROOT:
                raise ValueError('unapproved top-level source file')
        elif rel.parts[0] not in ALLOWED_DIRS or rel.suffix not in ALLOWED_SUFFIXES:
            raise ValueError('unapproved source type')
        path = ROOT / name
        if any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file():
            raise ValueError('source file must be a regular nonsymlink')
        data = path.read_bytes()
        scan(name, data)
        files[name] = data
    files['SOURCE_SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in sorted(files.items())).encode()
    scan('SOURCE_SHA256SUMS', files['SOURCE_SHA256SUMS'])
    report = {'status': 'FAIL' if failures else 'PASS', 'candidate': TOP,
              'scanned_files': scanned, 'failures': failures,
              'third_party_attribution_exceptions': attributions,
              'scope': 'every allowlisted source file and final tar member; original dependency hashes required'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def save_report():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        report_path.chmod(0o600)
    save_report()
    if failures:
        raise ValueError('anonymity gate failed; consult the private report (no archive produced)')
    out.mkdir(parents=True, exist_ok=True)
    archive = out / (TOP + '.tar.gz')
    temporary = out / (TOP + '.tar.gz.partial')
    # Deterministic owner, names, mode, timestamp, gzip header and archive order.
    with temporary.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(TOP + '/' + name)
                    info.size = len(data); info.mode = 0o644
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(data))
    # Reopen and scan the actual tar contents, not merely the staging files.
    scanned.clear()
    with tarfile.open(temporary, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files):
            raise ValueError('archive member count mismatch')
        for member in members:
            if not member.isfile() or not member.name.startswith(TOP + '/') or member.uid or member.gid or member.uname or member.gname:
                raise ValueError('unsafe archive metadata')
            name = member.name[len(TOP) + 1:]
            data = tar.extractfile(member).read()
            if name not in files or files[name] != data:
                raise ValueError('archive content mismatch')
            scan(name, data)
    if failures:
        temporary.unlink()
        report['status'] = 'FAIL'; save_report()
        raise ValueError('final archive anonymity gate failed')
    os.replace(temporary, archive)
    sha = digest(archive.read_bytes())
    (out / (archive.name + '.sha256')).write_text(f'{sha}  {archive.name}\n')
    report.update(status='PASS', archive_sha256=sha, archive_bytes=archive.stat().st_size)
    save_report()
    print(f'PASS: anonymous source candidate, {len(files)} files; SHA256 {sha}')

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, UnicodeError, KeyError) as error:
        # Deliberately do not print exception strings containing private paths.
        print('Packaging refused; inspect the private policy/report and source allowlist.')
        raise SystemExit(1) from None
