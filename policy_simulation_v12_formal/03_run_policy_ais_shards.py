# -*- coding: utf-8 -*-
"""以全频、逐船数据库读取方式可恢复地运行年度政策AIS分片。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from policy_run_config import (
    AIS_DIR, AIS_PREFETCH_BATCH_SIZE, AIS_SAMPLE_INTERVAL_SEC,
    AIS_SHARD_SIZE, DIRECT_LOG_ROOT, DIRECT_MANIFEST, DIRECT_ROOT,
    GT_CROSSWALK, ROOT, SIMULATION_YEAR,
)


EMISSION_MAIN = Path(r"E:\transfer justice\emission_network\emission_network_main.py")
ZONES = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
GT = GT_CROSSWALK
SHARD_ROOT = DIRECT_ROOT / "shards"
LOG_ROOT = DIRECT_LOG_ROOT / "shards"
MANIFEST = DIRECT_MANIFEST

FORMAL_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]
# Policy geography is part of the cache key.  Bump this whenever the selected
# region membership or zone geometry changes so pre-geodesic shards cannot be
# silently reused merely because their output schema still matches.
PIPELINE_VERSION = (
    f"v12_direct_annual_shards_{SIMULATION_YEAR}_full_original_io_20260824"
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.casefold()).strip("_")


def _universe_counts() -> dict[str, int]:
    parts = []
    for p in sorted(AIS_DIR.glob(f"ais_{SIMULATION_YEAR}??.db")):
        conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True, timeout=120)
        d = pd.read_sql_query("SELECT MMSI, Ship_Type FROM ship_info", conn)
        conn.close()
        parts.append(d)
    u = pd.concat(parts, ignore_index=True).drop_duplicates("MMSI", keep="first")
    return {t: int((u["Ship_Type"] == t).sum()) for t in FORMAL_TYPES}


def _expected_files(root: Path, ship_type: str) -> list[Path]:
    out = []
    for month in range(1, 13):
        ym = f"{SIMULATION_YEAR}{month:02d}"
        d = root / ym / ship_type
        out.extend([
            d / f"city_policy_delta_{ship_type}_{ym}.csv",
            d / f"policy_delta_totals_{ship_type}_{ym}.csv",
            d / f"statistics_{ship_type}_{ym}.txt",
        ])
        if ship_type == "Container":
            out.extend([
                d / f"ops_calls_{ship_type}_{ym}.parquet",
                d / f"ops_call_city_{ship_type}_{ym}.parquet",
            ])
    return out


def _audit_complete(root: Path, ship_type: str) -> tuple[bool, str]:
    version = root / "_policy_pipeline_version.txt"
    if (not version.is_file()
            or version.read_text(encoding="utf-8").strip() != PIPELINE_VERSION):
        return False, "pipeline_version_missing_or_stale"
    missing = [p for p in _expected_files(root, ship_type) if not p.is_file()]
    if missing:
        return False, f"missing={len(missing)}"
    ym0 = f"{SIMULATION_YEAR}01"
    stat = root / ym0 / ship_type / f"statistics_{ship_type}_{ym0}.txt"
    text = stat.read_text(encoding="utf-8")
    m = re.search(r"Failed:(\d+)", text)
    if not m:
        return False, "statistics_unparseable"
    failed = int(m.group(1))
    if failed:
        return False, f"failed_ships={failed}"
    probe = root / ym0 / ship_type / f"policy_delta_totals_{ship_type}_{ym0}.csv"
    cols = set(pd.read_csv(probe, nrows=0).columns)
    required = {
        "NME_SO2_CURRENT", "NME_PM_CURRENT", "NME_NOx_CURRENT",
        "B_SO2", "B_PM", "B_NOx",
    }
    if not required.issubset(cols):
        return False, f"policy_columns_stale={sorted(required - cols)}"
    return True, "complete"


def _write_manifest(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(MANIFEST, index=False, encoding="utf-8-sig")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cores", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=AIS_SHARD_SIZE)
    ap.add_argument(
        "--prefetch-batch-size", type=int, default=AIS_PREFETCH_BATCH_SIZE,
        help="0=沿用原程序的工作进程逐船读取；正式流程固定为0",
    )
    ap.add_argument(
        "--sample-interval-sec", type=int, default=AIS_SAMPLE_INTERVAL_SEC,
        help="0=保留全部有效AIS点；正式流程固定为0",
    )
    ap.add_argument(
        "--ship-types", default=",".join(FORMAL_TYPES),
        help="逗号分隔；默认四类正式船型",
    )
    args = ap.parse_args()
    if args.n_cores < 1 or args.shard_size < 1:
        raise SystemExit("--n-cores 与 --shard-size 必须为正数")
    wanted_norm = {_norm(s) for s in args.ship_types.split(",") if s.strip()}
    wanted = [t for t in FORMAL_TYPES if _norm(t) in wanted_norm]
    if not wanted:
        raise SystemExit("--ship-types 未匹配正式船型")
    if not (ZONES.is_file() and GT.is_file() and EMISSION_MAIN.is_file()):
        raise SystemExit("正式政策区、GT交叉表或排放主程序缺失")
    if args.prefetch_batch_size != 0 or args.sample_interval_sec != 0:
        raise SystemExit("正式分片必须使用全频AIS和原逐船读取方式（两个参数均为0）")

    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    counts = _universe_counts()
    rows = []
    failed_shards: list[str] = []

    for ship_type in wanted:
        total = counts[ship_type]
        for start in range(0, total, args.shard_size):
            n = min(args.shard_size, total - start)
            shard_id = f"{_norm(ship_type)}_start_{start:05d}_n_{n:05d}"
            out_root = SHARD_ROOT / shard_id
            log_path = LOG_ROOT / f"{shard_id}.log"
            complete, note = _audit_complete(out_root, ship_type)
            rec = {
                "ship_type": ship_type,
                "start_index": start,
                "n_requested": n,
                "universe_n": total,
                "shard_id": shard_id,
                "output_root": str(out_root),
                "log_path": str(log_path),
                "status": "skipped_complete" if complete else "pending",
                "note": note,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            rows.append(rec)
            _write_manifest(rows)
            if complete:
                print(f"[skip] {shard_id}", flush=True)
                continue

            cmd = [
                sys.executable, "-u", str(EMISSION_MAIN),
                "--year", str(SIMULATION_YEAR),
                "--db-dir", str(AIS_DIR),
                "--results-root", str(out_root),
                "--only-ship-types", ship_type,
                "--ship-start-index", str(start),
                "--max-ships-per-type", str(n),
                "--n-cores", str(args.n_cores),
                "--prefetch-batch-size", str(args.prefetch_batch_size),
                "--prefetch-sample-interval-sec", str(args.sample_interval_sec),
                "--policy-zones", str(ZONES),
                # 计算层使用“区域×政策责任国”单元，几何并集与 policy_regions
                # 完全相同，但可为 Table 7 提供逐国精确分解。
                "--policy-zone-layer", "policy_units",
                "--ship-attributes-csv", str(GT),
                "--policy-only",
                "--strict-db-read",
                "--no-skip-done",
            ]
            print(f"[run] {shard_id}  cores={args.n_cores}", flush=True)
            rec["status"] = "running"
            rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_manifest(rows)
            with log_path.open("w", encoding="utf-8") as log:
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(EMISSION_MAIN.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=child_env,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            if code == 0:
                (out_root / "_policy_pipeline_version.txt").write_text(
                    PIPELINE_VERSION, encoding="utf-8"
                )
            complete, note = _audit_complete(out_root, ship_type)
            rec["status"] = "complete" if code == 0 and complete else "failed"
            rec["note"] = f"returncode={code};{note}"
            rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_manifest(rows)
            if rec["status"] != "complete":
                failed_shards.append(f"{shard_id} ({rec['note']})")
                print(
                    f"[shard-failed-continue] {failed_shards[-1]}", flush=True
                )

    complete_rows = sum(
        str(row.get("status", "")).casefold()
        in {"complete", "skipped_complete"}
        for row in rows
    )
    run_state = {
        "status": "complete" if not failed_shards and complete_rows == len(rows)
        else "failed",
        "execution_mode": "direct_annual_shards",
        "simulation_year": SIMULATION_YEAR,
        "ship_types": wanted,
        "universe_counts": counts,
        "shard_size": int(args.shard_size),
        "total_shards": len(rows),
        "complete_shards": int(complete_rows),
        "failed_shards": failed_shards,
        "config_key": {
            "simulation_year": SIMULATION_YEAR,
            "sample_interval_sec": int(args.sample_interval_sec),
            "prefetch_batch_size": int(args.prefetch_batch_size),
            "shard_size": int(args.shard_size),
            "pipeline_version": PIPELINE_VERSION,
            "policy_zone_layer": "policy_units",
        },
        "manifest": str(MANIFEST),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    DIRECT_ROOT.mkdir(parents=True, exist_ok=True)
    (DIRECT_ROOT / "direct_run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed_shards:
        raise SystemExit(
            "所有分片均已尝试，但以下分片未通过质量门:\n- "
            + "\n- ".join(failed_shards)
        )
    print(f"[complete] {len(rows)} shards  manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
