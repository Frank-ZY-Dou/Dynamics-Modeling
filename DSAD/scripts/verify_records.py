#!/usr/bin/env python
"""Check the released logs and models against their manifests.

    python scripts/verify_records.py --records records [--unlisted]

Each archive unpacks a manifest (*.manifest.jsonl) with one JSON object per line:

    a fit    {"study": ..., "tag": ..., "files": {<name>: <sha256>}, ...}: the files of
             <records>/<study>/<tag>/
    a file   {"file": <path>, "sha256": <sha256>}: a path under <records>

Missing files and differing hashes are reported, and the exit status is 1 if there are any. Files
that no manifest names are counted, and listed with --unlisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entries(root: Path, manifest: Path):
    """(path, expected sha256) of every file a manifest names; ValueError for a line of another form."""
    for k, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if isinstance(d.get("files"), dict):
            for name, digest in d["files"].items():
                yield root / d["study"] / d["tag"] / name, digest
        elif "file" in d and "sha256" in d:
            yield root / d["file"], d["sha256"]
        else:
            raise ValueError(f"{manifest}:{k}: neither a fit nor a file entry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, type=Path, help="the folder the archives were unpacked into")
    ap.add_argument("--unlisted", action="store_true", help="also list the files that no manifest names")
    args = ap.parse_args()
    root = args.records
    manifests = sorted(root.glob("*.manifest.jsonl"))
    if not manifests:
        sys.exit(f"{root}: no *.manifest.jsonl (were the archives unpacked here?)")
    want: dict[Path, set[str]] = {}
    for m in manifests:
        for path, digest in entries(root, m):
            want.setdefault(path, set()).add(digest)
    missing = sorted(p for p in want if not p.is_file())
    differ = sorted(p for p, d in want.items() if p.is_file() and (len(d) > 1 or sha256(p) not in d))
    listed = {p.resolve() for p in want}
    unlisted = sorted(p for p in root.rglob("*") if p.is_file() and p not in manifests
                      and p.resolve() not in listed)
    print(f"{len(manifests)} manifests name {len(want)} files: {len(want) - len(missing) - len(differ)} equal, "
          f"{len(missing)} missing, {len(differ)} with another sha256; {len(unlisted)} files named by no manifest")
    for kind, paths in (("missing", missing), ("another sha256", differ)) + ((("unlisted", unlisted),)
                                                                          if args.unlisted else ()):
        for p in paths:
            print(f"  {kind}: {p.relative_to(root)}")
    sys.exit(1 if missing or differ else 0)


if __name__ == "__main__":
    main()
