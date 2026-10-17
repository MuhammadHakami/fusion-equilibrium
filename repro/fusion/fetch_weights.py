"""Download the trained weights listed in repro/fusion/WEIGHTS.sha256 into their repository paths and verify every sha256.

The weights are the release assets of this repository (one asset per file, named by its basename):
    python repro/fusion/fetch_weights.py --base https://github.com/<owner>/<repo>/releases/download/<tag>
Files already present with the right sha256 are skipped; a mismatch is an error (nothing is overwritten silently).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True, help="release download URL prefix")
    ap.add_argument("--list", default=str(ROOT / "repro/fusion/WEIGHTS.sha256")); a = ap.parse_args()
    rows = [ln.split() for ln in Path(a.list).read_text().splitlines() if ln.strip()]
    for h, n, rel in rows:
        dst = ROOT / rel
        if dst.exists():
            if sha(dst) == h:
                print(f"ok      {rel}", flush=True); continue
            sys.exit(f"{rel} exists with a different sha256; remove it to re-download")
        dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(dst.suffix + ".part")
        url = f"{a.base.rstrip('/')}/{Path(rel).name}"; print(f"fetch   {rel} ({int(n) / 1e6:.0f} MB) <- {url}", flush=True)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
            for b in iter(lambda: r.read(1 << 20), b""):
                fh.write(b)
        if sha(tmp) != h:
            tmp.unlink(); sys.exit(f"{rel}: sha256 mismatch after download")
        tmp.rename(dst)
    print(f"FETCH_DONE {len(rows)} files verified", flush=True)


if __name__ == "__main__":
    main()
