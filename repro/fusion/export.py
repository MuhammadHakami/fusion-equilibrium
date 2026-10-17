"""Assemble the Fusion reproduction package as a clean directory tree (nothing is published here).

Sources = every Python file reachable by imports from the entry scripts named in repro/fusion/{train_all.sh, infer.sh, README.md} and
runs/final_fold.sh, plus those shell scripts, the repro files and the small runtime metadata. Weights and caches (MANIFEST_runtime.txt
sections A-C, F16K / F17K additions) are symlinked by default so the tree can be smoke-tested in place (--weights copy copies them);
--weights none writes the public release tree: the trained weights are listed in repro/fusion/WEIGHTS.sha256 (fetched from the release
assets by repro/fusion/fetch_weights.py), the training-derived caches in repro/fusion/CACHES.sha256 (rebuilt by the README data step).

    .venv/bin/python repro/fusion/export.py <out_dir> [--weights link|copy|none]
    -> <out_dir>/ (repo layout), <out_dir>/repro/fusion/MANIFEST_source.txt (sha256 of every source file)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEARCH = ["paper/strat/fusion", "paper/sophelio", "baseline/sophelio-starter/fusion_scoring", "baseline/sophelio-starter", "paper",
          "paper/strat/fusion/fb"]
SHELL = ["repro/fusion/train_all.sh", "repro/fusion/infer.sh", "runs/final_fold.sh"]
EXTRA = ["repro/fusion/README.md", "repro/fusion/requirements.lock", "repro/fusion/entry_configs.json", "repro/fusion/MANIFEST_runtime.txt",
         "repro/fusion/export.py", "repro/fusion/fetch_weights.py", "baseline/sophelio-starter/LICENSE"]
ROOT_FILES = {"repro/fusion/LICENSE": "LICENSE", "repro/fusion/PACKAGE_README.md": "README.md", "repro/fusion/gitignore": ".gitignore"}     # package root files
ENTRY_RE = re.compile(r"(?:\$F/|paper/strat/fusion/|paper/sophelio/)([A-Za-z0-9_]+\.py)")


def entry_scripts() -> list[Path]:
    text = "".join((ROOT / f).read_text() for f in SHELL + ["repro/fusion/README.md"])
    out = []
    for m in ENTRY_RE.finditer(text):
        hit = [ROOT / d / m.group(1) for d in SEARCH[:2] if (ROOT / d / m.group(1)).exists()]
        if not hit:
            raise FileNotFoundError(f"entry script {m.group(1)} not found under {SEARCH[:2]}")
        out.extend(hit[:1])
    return sorted(set(out + [ROOT / "paper/sophelio/sota.py"]))         # README data step: python -c "import sota; sota.build_cache()"


def local_module(name: str, here: Path) -> Path | None:
    for d in [here] + [ROOT / s for s in SEARCH]:
        for cand in (d / f"{name}.py", d / name / "__init__.py"):
            if cand.exists():
                return cand
    return None


def closure(entries: list[Path]) -> list[Path]:
    seen, todo = set(), list(entries)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for node in ast.walk(ast.parse(f.read_text(), str(f))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            for n in names:
                m = local_module(n.split(".")[0], f.parent)
                if m is not None and m not in seen:
                    todo.append(m)
    return sorted(seen)


def runtime_files() -> list[tuple[str, str, int, str]]:
    """(path, sha256, bytes, kind) per runtime file; kind: weights (A, F16K / F17K additions), meta (B), cache (C)."""
    out, kind = [], None
    for line in (ROOT / "repro/fusion/MANIFEST_runtime.txt").read_text().splitlines():
        if line.startswith("## "):
            kind = "meta" if line.startswith("## B") else "cache" if line.startswith("## C") else None if line.startswith("## D") else \
                   "weights" if line.startswith(("## A", "## F16K", "## F17K")) else kind
        parts = line.split()
        if len(parts) == 4 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            if kind is None:
                raise ValueError(f"runtime file outside a known manifest section: {parts[3]}")
            out.append((parts[3], parts[0], int(parts[1]), kind))
    return out


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--weights", choices=["link", "copy", "none"], default="link")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit("out_dir must lie outside the repository")
    src = closure(entry_scripts())
    rel_src = [str(p.relative_to(ROOT)) for p in src] + SHELL + EXTRA
    for r in rel_src:
        (out / r).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / r, out / r)
    for s_, d_ in ROOT_FILES.items():
        shutil.copy2(ROOT / s_, out / d_)
    linked, listed = 0, {"weights": [], "cache": []}
    for r, h, n, kind in runtime_files():
        if r in rel_src:
            continue
        dst = out / r
        if a.weights == "none" and kind in listed:
            listed[kind].append(f"{h}  {n:12d}  {r}\n"); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if a.weights == "link" and kind != "meta":
            os.symlink(ROOT / r, dst)
        else:
            shutil.copy2(ROOT / r, dst)
            if sha(dst) != h:
                raise ValueError(f"{r}: sha256 differs from MANIFEST_runtime.txt")
        linked += 1
    if a.weights == "none":
        for kind, fn in (("weights", "WEIGHTS.sha256"), ("cache", "CACHES.sha256")):
            (out / "repro/fusion" / fn).write_text("".join(listed[kind]))
    man = out / "repro/fusion/MANIFEST_source.txt"
    man.write_text("# sha256  bytes  path - every source / script / repro file of the package (export.py)\n" +
                   "".join(f"{sha(out / r)}  {(out / r).stat().st_size:9d}  {r}\n" for r in sorted(set(rel_src))))
    print(f"EXPORT_DONE {out}: {len(set(rel_src))} source files ({len(src)} Python), {linked} runtime files ({a.weights}), "
          f"{len(listed['weights'])} weights / {len(listed['cache'])} caches listed", flush=True)


if __name__ == "__main__":
    main()
