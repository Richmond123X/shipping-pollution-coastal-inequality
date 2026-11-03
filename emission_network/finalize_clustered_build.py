"""Validate and finalize an already-built clustered AIS database.

This recovery helper is intentionally separate from the bulk rebuild.  It
opens the clustered target by itself for ``main.quick_check`` so corruption in
an attached source index cannot contaminate the target validation result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from rebuild_clustered_ais import Heartbeat, Reporter, _fingerprint, _readonly_uri, _size_gb


def _logged_inserted_rows(log_path: Path, month: str) -> int:
    pattern = re.compile(rf"\b{re.escape(month)} inserted rows=([0-9,]+)")
    found: list[int] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            found.append(int(match.group(1).replace(",", "")))
    if not found:
        raise RuntimeError(f"no inserted-row record for {month} in {log_path}")
    return found[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a clustered AIS .building.db")
    parser.add_argument("--month", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--building", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()

    month = args.month
    source = args.source.resolve()
    building = args.building.resolve()
    final = args.final.resolve()
    log_path = args.log.resolve()
    reporter = Reporter(log_path)

    if not source.is_file():
        raise FileNotFoundError(source)
    if not building.is_file():
        raise FileNotFoundError(building)
    if final.exists():
        raise FileExistsError(final)

    reporter.write("=" * 78)
    reporter.write(f"RECOVERY START {month}: target-only validation of {building}")

    # First validate only the new database.  Do not attach the source here.
    con = sqlite3.connect(_readonly_uri(building), uri=True, timeout=120)
    try:
        schema_row = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='ais_tracks'"
        ).fetchone()
        if not schema_row or "WITHOUT ROWID" not in str(schema_row[0]).upper():
            raise RuntimeError("target ais_tracks is not a WITHOUT ROWID table")
        no_temp_activity = building.parent / ".quick_check_no_temp"
        with Heartbeat(reporter, month, "TARGET-ONLY-QUICK-CHECK", building, no_temp_activity):
            quick = str(con.execute("PRAGMA main.quick_check").fetchone()[0])
        if quick.lower() != "ok":
            raise RuntimeError(f"target SQLite quick_check failed: {quick}")
        target_plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM ais_tracks "
            "WHERE MMSI=? ORDER BY UTCtime", (123456789,)
        ).fetchall()
    finally:
        con.close()

    # Reconfirm source row metadata and deterministic samples independently.
    src = sqlite3.connect(_readonly_uri(source), uri=True, timeout=120)
    dst = sqlite3.connect(_readonly_uri(building), uri=True, timeout=120)
    try:
        stat = src.execute(
            "SELECT stat FROM sqlite_stat1 WHERE tbl='ais_tracks' AND idx='idx_mmsi_time'"
        ).fetchone()
        if not stat:
            raise RuntimeError("source sqlite_stat1 row count is unavailable")
        source_rows = int(str(stat[0]).split()[0])
        inserted_rows = _logged_inserted_rows(log_path, month)
        if inserted_rows != source_rows:
            raise RuntimeError(
                f"row count mismatch: source={source_rows:,}, inserted={inserted_rows:,}"
            )

        source_ship_rows = int(src.execute("SELECT COUNT(*) FROM ship_info").fetchone()[0])
        target_ship_rows = int(dst.execute("SELECT COUNT(*) FROM ship_info").fetchone()[0])
        if source_ship_rows != target_ship_rows:
            raise RuntimeError(
                f"ship_info mismatch: source={source_ship_rows}, target={target_ship_rows}"
            )

        mmsi_values = [int(row[0]) for row in src.execute(
            "SELECT MMSI FROM ais_tracks INDEXED BY idx_mmsi_time "
            "GROUP BY MMSI LIMIT ?", (max(1, args.sample_count),)
        ).fetchall()]
        samples = []
        for mmsi in mmsi_values:
            src_fp = _fingerprint(src, "ais_tracks", mmsi)
            dst_fp = _fingerprint(dst, "ais_tracks", mmsi)
            if src_fp != dst_fp:
                raise RuntimeError(
                    f"sample mismatch MMSI={mmsi}: source={src_fp}, target={dst_fp}"
                )
            samples.append({"MMSI": mmsi, "fingerprint": list(src_fp)})
    finally:
        src.close()
        dst.close()

    verification = final.with_name(final.stem + ".verification.json")
    meta = {
        "month": month,
        "source": str(source),
        "target": str(final),
        "source_rows": source_rows,
        "inserted_rows": inserted_rows,
        "ship_info_rows": target_ship_rows,
        "source_size_gb": _size_gb(source),
        "target_size_gb": _size_gb(building),
        "sqlite_quick_check": quick,
        "target_query_plan": target_plan,
        "samples": samples,
        "recovered_after_source_quick_check_error": True,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    verification.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(building, final)
    reporter.write(
        f"RECOVERY COMPLETE {month}: {final} ({_size_gb(final):.1f} GB); "
        f"target quick_check={quick}; rows={inserted_rows:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
