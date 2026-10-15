"""Download one Hugging Face file with N parallel HTTP range requests (via hf-mirror.com) and stitch the parts.
Needed on throttled connections (NetFree gives ~100KB/s per connection); each part is resumable. Usage:
    python scripts/fetch_hf_parallel.py intfloat/multilingual-e5-base model.safetensors models/multilingual-e5-base --parts 8"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MIRROR = "https://hf-mirror.com"


def remote_size(url: str) -> int:
    with httpx.Client(follow_redirects=True, timeout=60) as c:
        r = c.head(url)
        r.raise_for_status()
        return int(r.headers["content-length"])


def fetch_part(url: str, dest: Path, start: int, end: int) -> None:
    """Resumable range download with curl (handles the redirect chain and retries)."""
    have = dest.stat().st_size if dest.exists() else 0
    want = end - start + 1
    while have < want:
        cmd = ["curl.exe" if sys.platform == "win32" else "curl", "-s", "-L", "--retry", "20", "--retry-delay", "5",
               "--retry-all-errors", "-r", f"{start + have}-{end}", "-o", str(dest) + ".tmp", url]
        subprocess.run(cmd, check=False)
        tmp = Path(str(dest) + ".tmp")
        if tmp.exists() and tmp.stat().st_size:
            with dest.open("ab") as out, tmp.open("rb") as inp:
                out.write(inp.read())
            tmp.unlink()
        have = dest.stat().st_size if dest.exists() else 0
        if have < want:
            time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("filename")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()

    url = f"{MIRROR}/{a.repo}/resolve/{a.revision}/{a.filename}"
    total = remote_size(url)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    final = a.out_dir / a.filename
    if final.exists() and final.stat().st_size == total:
        print(f"already complete: {final} ({total} bytes)")
        return 0
    size = (total + a.parts - 1) // a.parts
    parts = [(final.with_suffix(f".part{i}"), i * size, min((i + 1) * size, total) - 1) for i in range(a.parts)]
    print(f"{a.filename}: {total / 1e6:.0f} MB in {a.parts} parts")

    started = time.time()
    with ThreadPoolExecutor(max_workers=a.parts) as ex:
        futs = [ex.submit(fetch_part, url, p, s, e) for p, s, e in parts]
        while any(not f.done() for f in futs):
            got = sum(f.stat().st_size for p, _, _ in parts for f in (p, Path(str(p) + ".tmp")) if f.exists())
            rate = got / max(time.time() - started, 1)
            eta = (total - got) / rate / 60 if rate else 0
            print(f"\r{got / 1e6:8.0f}/{total / 1e6:.0f} MB  {rate / 1e3:6.0f} KB/s  eta {eta:5.1f} min", end="", flush=True)
            time.sleep(5)
        for f in futs:
            f.result()
    print()
    with final.open("wb") as out:
        for p, _, _ in parts:
            out.write(p.read_bytes())
            p.unlink()
    ok = final.stat().st_size == total
    print(f"{'OK' if ok else 'SIZE MISMATCH'}: {final} {final.stat().st_size} bytes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
