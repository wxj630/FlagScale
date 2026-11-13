#!/usr/bin/env python3
"""Download a ModelScope dataset with per-file concurrency.

Why this exists instead of just calling ``snapshot_download``: ModelScope
throttles each connection to roughly 130-155 kB/s, so the 4.4 GB Flickr30K dump
takes ~7 hours single-threaded.  The throttle is per connection, not per account,
so launching several files at once multiplies throughput (measured: 4 x ~145 kB/s
= ~580 kB/s, a 4.4x speedup).  ``snapshot_download`` fetches one file at a time,
which is why the first attempt looked like a broken network.

Usage::

    python tools/modelscope_parallel_download.py lmms-lab/Flickr30K \
        --dest /flagos-retrieval-datasets/flickr30k --workers 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Recursive=true"
FILE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"


def list_files(repo: str) -> list[dict]:
    """List a dataset's files via ``curl``, retrying a flaky endpoint.

    Deliberately not ``urllib``: inside the PPU container every python-side TLS
    read to modelscope timed out ("The read operation timed out" from ssl.py)
    while ``curl`` to the exact same URL worked.

    The listing endpoint is also intermittently slow rather than uniformly fast.
    Measured from the PPU container, three back-to-back requests to the same URL
    gave: 000 (24.8 s timeout), 200 (25.2 s), 000 (60 s timeout).  A single
    attempt therefore fails more often than it succeeds, so retry with a
    generous per-request timeout -- the file bodies themselves are throttled but
    reliable, it is only this metadata call that misbehaves.
    """

    url = API.format(repo=repo)
    last_error = "no attempt made"
    for attempt in range(5):
        try:
            completed = subprocess.run(
                ["curl", "-sL", "--connect-timeout", "30", "--max-time", "180",
                 "-H", "User-Agent: Mozilla/5.0", url],
                capture_output=True, text=True, timeout=200,
            )
        except subprocess.TimeoutExpired:
            last_error = "curl killed after 200 s"
        else:
            if completed.returncode == 0 and completed.stdout.strip():
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    last_error = f"non-JSON body: {completed.stdout[:120]!r}"
                else:
                    if payload.get("Code") == 200:
                        return [
                            entry
                            for entry in (payload.get("Data") or {}).get("Files") or []
                            if entry.get("Type") == "blob"
                        ]
                    last_error = f"API code {payload.get('Code')}: {payload.get('Message')}"
            else:
                last_error = f"curl exit {completed.returncode}"
        print(f"  list_files attempt {attempt + 1}/5 failed: {last_error}", flush=True)
        time.sleep(3)
    raise SystemExit(f"listing {repo} failed after 5 attempts (last: {last_error})")


def already_complete(path: Path, expected: int) -> bool:
    return path.is_file() and path.stat().st_size == expected and expected > 0


def fetch(repo: str, path: str, size: int, dest: Path) -> tuple[str, bool, str]:
    target = dest / path
    if already_complete(target, size):
        return path, True, "skipped (complete)"
    target.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes: a throttled download that dies halfway must not restart.
    cmd = [
        "curl",
        "-sL",
        "--retry", "5",
        "--retry-delay", "2",
        "--connect-timeout", "30",
        # A stalled transfer with no --max-time can hang forever: the connection
        # stays open, curl never returns, and the ThreadPoolExecutor worker is
        # pinned so the whole run never reaches its final "done" line.  Observed
        # on the PPU container, where two of nine files stalled mid-transfer and
        # the process sat there with zero child curls alive.
        "--max-time", "900",
        "-C", "-",
        "-o", str(target),
        FILE.format(repo=repo, path=path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=960)
    except subprocess.TimeoutExpired:
        return path, False, "curl killed after 960 s"
    if result.returncode != 0:
        return path, False, f"curl exit {result.returncode}: {result.stderr[:120]}"
    got = target.stat().st_size if target.is_file() else 0
    # ModelScope serves a JSON error body with HTTP 200 for unknown paths, so a
    # size check is the only reliable completion test.
    if got != size:
        return path, False, f"size {got} != {size}"
    return path, True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = list_files(args.repo)
    total = sum(entry.get("Size") or 0 for entry in files)
    print(f"{args.repo}: {len(files)} files, {total / 1e6:.1f} MB", flush=True)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, args.repo, entry["Path"], entry.get("Size") or 0, dest): entry
            for entry in files
        }
        done = 0
        for future in as_completed(futures):
            path, ok, note = future.result()
            done += 1
            status = "ok  " if ok else "FAIL"
            print(f"  [{done}/{len(files)}] {status} {path} ({note})", flush=True)
            if not ok:
                failed.append(path)

    if failed:
        print(f"\n{len(failed)} file(s) failed:", file=sys.stderr)
        for path in failed:
            print(f"  {path}", file=sys.stderr)
        raise SystemExit(1)
    print("done.")


if __name__ == "__main__":
    main()
