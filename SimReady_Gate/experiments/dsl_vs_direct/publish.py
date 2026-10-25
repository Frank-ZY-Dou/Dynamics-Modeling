"""Copy a finished comparison into results/dsl_vs_direct: per model the trials, the run's metadata,
system prompts and formats, finals.json, reuse.json and the drawings; the leak test; the tables.
Local directories and e-mail addresses are replaced on the way.

    python experiments/dsl_vs_direct/publish.py <scratch run dir> results/dsl_vs_direct
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def clean_text(text: str, roots: list[tuple[str, str]]) -> str:
    text = EMAIL.sub("<email>", text)
    for old, new in roots:
        text = text.replace(old, new)
    return text


def main(src: str, dst: str):
    src_p, dst_p = Path(src).resolve(), Path(dst)
    roots = [(str(src_p.parent), "<work_root>"), (str(Path.home()), "~")]
    dst_p.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted(p for p in src_p.iterdir() if p.is_dir() and (p / "trials").is_dir()):
        out = dst_p / model_dir.name
        (out / "trials").mkdir(parents=True, exist_ok=True)
        for f in sorted(model_dir.glob("trials/*.json")):
            (out / "trials" / f.name).write_text(clean_text(f.read_text(), roots))
        for name in ("run_meta.json", "finals.json", "reuse.json"):
            if (model_dir / name).exists():
                (out / name).write_text(clean_text((model_dir / name).read_text(), roots))
        for f in list(model_dir.glob("system_*.txt")) + list(model_dir.glob("schema_*.json")):
            (out / f.name).write_text(f.read_text())
        if (model_dir / "figures").is_dir():
            (out / "figures").mkdir(exist_ok=True)
            for f in model_dir.glob("figures/*.png"):
                shutil.copy2(f, out / "figures" / f.name)
    iso = src_p / "isolation_check.json"
    if iso.exists():
        (dst_p / "isolation_check.json").write_text(clean_text(iso.read_text(), roots))
    print(json.dumps({"published": str(dst_p)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
