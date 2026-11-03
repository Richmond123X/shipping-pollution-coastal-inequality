#!/usr/bin/env python3
"""Resumable segmented HTTP downloader with byte-count verification.

Intended for the official CEPII archives, whose server throttles individual
connections. Each part is an explicit HTTP Range request. The final file is
published atomically only after every expected byte has been received.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


def remote_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if not size:
        raise RuntimeError(f"No Content-Length returned by {url}")
    return int(size)


def fetch_part(url: str, path: Path, start: int, end: int, retries: int) -> tuple[int, int]:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return start, expected

    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=(60, 180)) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range {start}-{end}: expected HTTP 206, got {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
                with tmp.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            actual = tmp.stat().st_size
            if actual != expected:
                raise RuntimeError(f"Range {start}-{end}: {actual} bytes, expected {expected}")
            os.replace(tmp, path)
            return start, actual
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--part-mib", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    total = remote_size(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size == total:
        print(f"Already complete: {args.output} ({total} bytes)", flush=True)
        print(f"SHA256 {sha256(args.output)}  {args.output.name}", flush=True)
        return

    part_dir = args.output.with_name(args.output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_size = args.part_mib * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, total, part_size)):
        end = min(total - 1, start + part_size - 1)
        ranges.append((index, start, end, part_dir / f"part_{index:05d}"))

    print(
        f"Downloading {args.url}\n"
        f"Target: {args.output}\n"
        f"Bytes: {total}; parts: {len(ranges)}; workers: {args.workers}",
        flush=True,
    )
    started = time.time()
    completed_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_part, args.url, path, start, end, args.retries): (index, start, end)
            for index, start, end, path in ranges
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            _, size = future.result()
            completed_bytes += size
            if count == 1 or count % 8 == 0 or count == len(ranges):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"parts {count}/{len(ranges)}; received this run "
                    f"{completed_bytes / 1024**2:.1f} MiB; "
                    f"{completed_bytes / elapsed / 1024**2:.2f} MiB/s",
                    flush=True,
                )

    assembled = args.output.with_suffix(args.output.suffix + ".assembling")
    with assembled.open("wb") as out:
        for index, _, _, path in ranges:
            with path.open("rb") as part:
                for block in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    out.write(block)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"Assembled size {assembled.stat().st_size}, expected {total}")
    os.replace(assembled, args.output)
    print(f"SHA256 {sha256(args.output)}  {args.output.name}", flush=True)


if __name__ == "__main__":
    main()
