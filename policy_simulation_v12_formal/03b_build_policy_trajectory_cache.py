# -*- coding: utf-8 -*-
"""顺序扫描 2025 AIS 月库，构建 V12 政策模拟的 300 秒近岸轨迹缓存。

原库每月约 10^9 行且表记录按接收顺序交错。对数万艘船逐船索引回表会
产生严重随机 I/O；本脚本每月只顺序扫描一次，在政策区外扩 250 km 的保守
候选范围内保留四类正式船型，再按 MMSI×固定 UTC 时间桶保留首个真实点。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import contains_xy, make_valid, union_all


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
AIS_DIR = Path(r"D:\SQL_AIS_SingleShip")
ZONES = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
DEFAULT_CACHE = ROOT / "intermediate" / "trajectory_cache_300s_250km"
FORMAL_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=120)


def build_universe(months: list[str], cache: Path) -> set[int]:
    universe_path = cache / "formal_ship_universe.csv"
    if universe_path.is_file():
        saved = pd.read_csv(universe_path)
        required = {"MMSI", "Ship_Type"}
        if required.issubset(saved.columns):
            saved = saved[saved["Ship_Type"].isin(FORMAL_TYPES)].copy()
            ids = set(pd.to_numeric(
                saved["MMSI"], errors="coerce"
            ).dropna().astype(int))
            if ids:
                print(
                    f"[reuse universe] ships={len(ids):,} path={universe_path}",
                    flush=True,
                )
                return ids
    parts = []
    for ym in months:
        p = AIS_DIR / f"ais_{ym}.db"
        if not p.is_file() or not ym.startswith("2025"):
            continue
        conn = readonly(p)
        q = "SELECT MMSI, Ship_Type FROM ship_info WHERE Ship_Type IN (?,?,?,?)"
        parts.append(pd.read_sql_query(q, conn, params=FORMAL_TYPES))
        conn.close()
    ships = (pd.concat(parts, ignore_index=True)
             .drop_duplicates("MMSI", keep="first")
             .sort_values("MMSI"))
    cache.mkdir(parents=True, exist_ok=True)
    ships.to_csv(universe_path, index=False, encoding="utf-8-sig")
    return set(pd.to_numeric(ships["MMSI"], errors="coerce").dropna().astype(int))


def build_margin(margin_km: float, cache: Path):
    margin_path = cache / "policy_candidate_margin.gpkg"
    zone_sha256 = file_sha256(ZONES)
    if margin_path.is_file():
        saved = gpd.read_file(margin_path, layer="margin").to_crs(4326)
        if (
            len(saved) == 1
            and abs(float(saved.iloc[0]["margin_km"]) - margin_km) < 1e-9
            and "zone_sha256" in saved
            and str(saved.iloc[0]["zone_sha256"]) == zone_sha256
        ):
            return saved.geometry.iloc[0]
    zones = gpd.read_file(ZONES, layer="policy_geo_units").to_crs(6933)
    buffered = np.asarray([
        make_valid(g).buffer(float(margin_km) * 1000.0)
        for g in zones.geometry if g is not None and not g.is_empty
    ])
    margin = make_valid(union_all(buffered, grid_size=100.0))
    margin_wgs = gpd.GeoSeries([margin], crs=6933).to_crs(4326).iloc[0]
    gpd.GeoDataFrame(
        [{"margin_km": margin_km, "zone_sha256": zone_sha256,
          "geometry": margin_wgs}], crs=4326
    ).to_file(margin_path, layer="margin", driver="GPKG")
    return margin_wgs


def probe_recovery_rowid(db: Path, start_rowid: int) -> int | None:
    """在损坏页之后寻找首个可读取 rowid，并让跳过范围保持可审计。"""
    candidate = int(start_rowid)
    # 先以100个rowid为步长精细探测；若损坏范围较大再逐级扩大。
    for step, attempts in ((100, 100), (1_000, 100), (10_000, 100),
                           (100_000, 50)):
        for _ in range(attempts):
            candidate += step
            conn = readonly(db)
            try:
                row = conn.execute(
                    "SELECT rowid FROM ais_tracks "
                    "WHERE rowid >= ? ORDER BY rowid LIMIT 1",
                    (candidate,),
                ).fetchone()
                if row is not None:
                    return int(row[0])
                return None
            except sqlite3.DatabaseError:
                pass
            finally:
                conn.close()
    return None


def scan_month(
    ym: str,
    ids: set[int],
    margin,
    cache: Path,
    chunksize: int,
    n_parts: int,
    row_limit: int,
) -> dict:
    out = cache / "raw_margin" / ym
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    zone_sha256 = file_sha256(ZONES)
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "complete"
            and int(saved.get("row_limit", 0)) == row_limit
            and saved.get("zone_sha256") == zone_sha256
        ):
            print(f"[skip scan] {ym}", flush=True)
            return saved
    db = AIS_DIR / f"ais_{ym}.db"
    if not db.is_file():
        rec = {
            "status": "missing_db", "month": ym, "db": str(db),
            "zone_sha256": zone_sha256,
        }
        manifest_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec
    writers: dict[int, pq.ParquetWriter] = {}
    scanned = retained = chunks = 0
    corrupt_ranges: list[dict[str, int | str]] = []
    last_rowid = 0
    active_chunksize = int(chunksize)
    report_step = 50_000_000
    next_report = report_step
    t0 = time.time()
    try:
        finished = False
        while not finished:
            query = (
                "SELECT rowid AS _rowid, MMSI, UTCtime, Speed, Longitude, Latitude "
                "FROM ais_tracks WHERE rowid > ? ORDER BY rowid"
            )
            conn = readonly(db)
            try:
                iterator = pd.read_sql_query(
                    query, conn, params=(int(last_rowid),),
                    chunksize=int(active_chunksize),
                )
                for d in iterator:
                    if row_limit > 0:
                        remaining = int(row_limit) - scanned
                        if remaining <= 0:
                            finished = True
                            break
                        if len(d) > remaining:
                            d = d.iloc[:remaining].copy()
                            finished = True
                    chunks += 1
                    scanned += len(d)
                    last_rowid = int(d["_rowid"].iloc[-1])
                    m = (
                        d["MMSI"].isin(ids)
                        & d["Longitude"].between(-180, 180)
                        & d["Latitude"].between(-90, 90)
                    )
                    idx = np.flatnonzero(m.to_numpy())
                    keep = np.zeros(len(d), dtype=bool)
                    if len(idx):
                        keep[idx] = contains_xy(
                            margin,
                            d["Longitude"].to_numpy()[idx],
                            d["Latitude"].to_numpy()[idx],
                        )
                    d = d.loc[keep].drop(columns="_rowid").copy()
                    retained += len(d)
                    if len(d):
                        d["partition"] = (
                            pd.to_numeric(d["MMSI"], errors="coerce").astype("int64") % n_parts
                        ).astype("int16")
                        for part, g in d.groupby("partition", sort=False):
                            part = int(part)
                            table = pa.Table.from_pandas(
                                g.drop(columns="partition"), preserve_index=False)
                            if part not in writers:
                                writers[part] = pq.ParquetWriter(
                                    out / f"part_{part:03d}.parquet", table.schema,
                                    compression="zstd", compression_level=3,
                                    use_dictionary=True,
                                )
                            writers[part].write_table(table)
                    if scanned >= next_report:
                        elapsed = time.time() - t0
                        print(
                            f"  [{ym}] chunks={chunks:,} scanned={scanned:,} "
                            f"retained={retained:,} rate={scanned/elapsed:,.0f} rows/s",
                            flush=True,
                        )
                        while scanned >= next_report:
                            next_report += report_step
                    if finished:
                        break
                else:
                    finished = True
            except sqlite3.DatabaseError as exc:
                if active_chunksize > 1:
                    active_chunksize = max(1, active_chunksize // 100)
                    print(
                        f"[source corruption] {ym} after_rowid={last_rowid:,}; "
                        f"retry_chunksize={active_chunksize:,}; error={exc}",
                        flush=True,
                    )
                else:
                    skipped_start = int(last_rowid) + 1
                    recovered = probe_recovery_rowid(db, skipped_start)
                    if recovered is None:
                        raise RuntimeError(
                            f"{ym}: cannot recover after corrupted rowid {last_rowid:,}"
                        ) from exc
                    skipped_end = int(recovered) - 1
                    corrupt_ranges.append({
                        "rowid_start": skipped_start,
                        "rowid_end": skipped_end,
                        "rowid_span": skipped_end - skipped_start + 1,
                        "error": str(exc),
                    })
                    print(
                        f"[source corruption recovered] {ym} "
                        f"skipped_rowid={skipped_start:,}-{skipped_end:,} "
                        f"resume_rowid={recovered:,}",
                        flush=True,
                    )
                    last_rowid = skipped_end
                    active_chunksize = int(chunksize)
            finally:
                conn.close()
    finally:
        for writer in writers.values():
            writer.close()
    rec = {
        "status": "complete", "month": ym, "db": str(db),
        "rows_scanned": scanned, "rows_retained": retained,
        "retained_share": retained / scanned if scanned else 0.0,
        "chunks": chunks, "n_partitions_written": len(writers),
        "row_limit": row_limit, "zone_sha256": zone_sha256,
        "last_rowid": last_rowid,
        "source_corruption_detected": bool(corrupt_ranges),
        "source_corrupt_rowid_ranges": corrupt_ranges,
        "source_rowid_span_skipped": int(sum(
            int(x["rowid_span"]) for x in corrupt_ranges
        )),
        "seconds": time.time() - t0,
    }
    manifest_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[scan complete] {ym} {rec}", flush=True)
    return rec


def sample_month(
    ym: str, cache: Path, n_parts: int, interval_sec: int, row_limit: int,
) -> dict:
    raw = cache / "raw_margin" / ym
    out = cache / "sampled_month" / ym
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    zone_sha256 = file_sha256(ZONES)
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "complete"
            and int(saved.get("row_limit", 0)) == row_limit
            and int(saved.get("sample_interval_sec", -1)) == int(interval_sec)
            and saved.get("zone_sha256") == zone_sha256
        ):
            print(f"[skip sample] {ym}", flush=True)
            return saved
    raw_manifest = raw / "manifest.json"
    if not raw_manifest.is_file():
        raise RuntimeError(f"raw month not complete: {ym}")
    state = json.loads(raw_manifest.read_text(encoding="utf-8"))
    if state.get("zone_sha256") != zone_sha256:
        raise RuntimeError(f"raw month policy geography is stale: {ym}")
    if state.get("status") == "missing_db":
        rec = {"status": "missing_db", "month": ym,
               "zone_sha256": zone_sha256}
        manifest_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec
    before = after = 0
    t0 = time.time()
    for part in range(n_parts):
        src = raw / f"part_{part:03d}.parquet"
        if not src.is_file():
            continue
        d = pd.read_parquet(src)
        before += len(d)
        d = d.sort_values(["MMSI", "UTCtime"])
        d["_bucket"] = (
            pd.to_numeric(d["UTCtime"], errors="coerce").astype("int64")
            // int(interval_sec)
        )
        d = (d.drop_duplicates(["MMSI", "_bucket"], keep="first")
             .drop(columns="_bucket")
             .drop_duplicates(["MMSI", "UTCtime"], keep="first"))
        after += len(d)
        d.to_parquet(
            out / f"part_{part:03d}.parquet", index=False,
            compression="zstd",
        )
        if (part + 1) % 8 == 0:
            print(
                f"  [{ym} sample] part={part+1}/{n_parts} "
                f"raw={before:,} sampled={after:,}", flush=True)
    rec = {
        "status": "complete", "month": ym, "raw_rows": before,
        "sampled_rows": after, "sample_share": after / before if before else 0.0,
        "sample_interval_sec": interval_sec, "row_limit": row_limit,
        "zone_sha256": zone_sha256,
        "seconds": time.time() - t0,
    }
    manifest_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[sample complete] {ym} {rec}", flush=True)
    return rec


def consolidate_by_mmsi(
    months: list[str], cache: Path, n_parts: int, row_limit: int,
    interval_sec: int,
) -> dict:
    out = cache / "by_mmsi"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    zone_sha256 = file_sha256(ZONES)
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "complete"
            and int(saved.get("row_limit", 0)) == row_limit
            and int(saved.get("sample_interval_sec", -1)) == int(interval_sec)
            and saved.get("zone_sha256") == zone_sha256
        ):
            print("[skip consolidate] by_mmsi", flush=True)
            return saved
    rows = ships = 0
    t0 = time.time()
    for part in range(n_parts):
        files = [
            cache / "sampled_month" / ym / f"part_{part:03d}.parquet"
            for ym in months
        ]
        files = [p for p in files if p.is_file()]
        if not files:
            continue
        d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        d = (d.drop_duplicates(["MMSI", "UTCtime"], keep="first")
             .sort_values(["MMSI", "UTCtime"]))
        for mmsi, g in d.groupby("MMSI", sort=False):
            g.to_parquet(
                out / f"{int(mmsi):09d}.parquet", index=False,
                compression="zstd",
            )
            rows += len(g)
            ships += 1
        print(
            f"  [consolidate] part={part+1}/{n_parts} ships={ships:,} rows={rows:,}",
            flush=True,
        )
    rec = {
        "status": "complete", "months": months, "ships_with_candidate_rows": ships,
        "sampled_rows": rows, "row_limit": row_limit,
        "sample_interval_sec": interval_sec,
        "zone_sha256": zone_sha256,
        "seconds": time.time() - t0,
    }
    manifest_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def process_month(
    ym: str,
    ids: set[int],
    margin,
    cache: Path,
    chunksize: int,
    n_parts: int,
    row_limit: int,
    interval_sec: int,
) -> tuple[dict, dict]:
    """顺序扫描并抽样一个月；月份之间可并行，月内读取顺序保持不变。"""
    scan = scan_month(
        ym, ids, margin, cache, chunksize, n_parts, row_limit
    )
    sample = sample_month(
        ym, cache, n_parts, interval_sec, row_limit
    )
    return scan, sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--margin-km", type=float, default=250.0)
    ap.add_argument("--sample-interval-sec", type=int, default=300)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--partitions", type=int, default=64)
    ap.add_argument(
        "--jobs", type=int, default=1,
        help="并行处理的月份数；每个月内部仍为一次顺序扫描。",
    )
    ap.add_argument(
        "--scan-sample-only", action="store_true",
        help="只生成逐月扫描和抽样检查点，不写年度清单或按MMSI合并。",
    )
    # 正式基线严格使用 2025 年十二个月。2024-12 源库在顺序扫描约 1 亿行处
    # 报 SQLite 页损坏，而 2026-01 又不存在；因此两侧边界月都不混入正式
    # 缓存，避免只补前端、不补后端造成非对称边界处理。
    ap.add_argument("--months", default=",".join(f"2025{x:02d}" for x in range(1, 13)))
    ap.add_argument("--row-limit", type=int, default=0,
                    help="仅用于独立缓存根目录的冒烟测试；0=全库")
    args = ap.parse_args()
    cache = args.cache_root.resolve()
    months = [x.strip() for x in args.months.split(",") if x.strip()]
    cache.mkdir(parents=True, exist_ok=True)
    ids = build_universe(months, cache)
    margin = build_margin(args.margin_km, cache)
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    scan_by_month: dict[str, dict] = {}
    sample_by_month: dict[str, dict] = {}
    if args.jobs == 1:
        for ym in months:
            scan, sample = process_month(
                ym, ids, margin, cache, args.chunksize, args.partitions,
                args.row_limit, args.sample_interval_sec,
            )
            scan_by_month[ym] = scan
            sample_by_month[ym] = sample
    else:
        print(
            f"[parallel] months={len(months)} jobs={args.jobs}; "
            "each monthly SQLite scan remains sequential",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    process_month, ym, ids, margin, cache, args.chunksize,
                    args.partitions, args.row_limit, args.sample_interval_sec,
                ): ym
                for ym in months
            }
            for future in as_completed(futures):
                ym = futures[future]
                scan, sample = future.result()
                scan_by_month[ym] = scan
                sample_by_month[ym] = sample
                print(f"[month complete] {ym}", flush=True)
    scan_records = [scan_by_month[ym] for ym in months]
    sample_records = [sample_by_month[ym] for ym in months]
    if args.scan_sample_only:
        print(
            f"[scan-sample-only complete] months={','.join(months)}",
            flush=True,
        )
        return
    final = consolidate_by_mmsi(
        months, cache, args.partitions, args.row_limit, args.sample_interval_sec
    )
    pd.DataFrame(scan_records).to_csv(cache / "scan_manifest.csv", index=False)
    pd.DataFrame(sample_records).to_csv(cache / "sampling_manifest.csv", index=False)
    config = {
        "status": "complete", "source": str(AIS_DIR), "zones": str(ZONES),
        "zone_sha256": file_sha256(ZONES),
        "months": months, "margin_km": args.margin_km,
        "sample_interval_sec": args.sample_interval_sec,
        "n_formal_mmsi": len(ids), "n_partitions": args.partitions,
        "row_limit": args.row_limit, "by_mmsi": final,
        "boundary_months_excluded": ["202412", "202601"],
        "boundary_note": (
            "202412 excluded after SQLite malformed-page error; 202601 absent. "
            "Formal inventory uses only 202501-202512 symmetrically."
        ),
    }
    (cache / "cache_manifest.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[complete] cache={cache}\n{json.dumps(config, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
