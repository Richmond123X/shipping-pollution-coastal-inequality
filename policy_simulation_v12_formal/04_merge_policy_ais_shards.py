# -*- coding: utf-8 -*-
"""校验并合并 03 步产生的 V12 AIS 政策分片。"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from policy_run_config import (
    CONSOLIDATED_ROOT as OUT, DIRECT_MANIFEST as MANIFEST,
    ROOT, SIMULATION_YEAR,
)


TABLES = ROOT / "tables"
FORMAL_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]
DELTA = ["D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA"]
NO_MARGIN = ["NME_SO2_CURRENT", "NME_PM_CURRENT", "NME_NOx_CURRENT"]
BASE = ["B_SO2", "B_PM", "B_NOx"]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _require_complete_manifest() -> pd.DataFrame:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"分片清单不存在: {MANIFEST}")
    m = pd.read_csv(MANIFEST)
    bad = m[~m["status"].isin(["complete", "skipped_complete"])]
    if len(bad):
        raise RuntimeError(f"仍有 {len(bad)} 个分片未完成:\n{bad.to_string(index=False)}")
    if set(m["ship_type"]) != set(FORMAL_TYPES):
        raise RuntimeError(f"船型不完整: {sorted(set(m['ship_type']))}")
    if (
        "execution_mode" in m
        and m["execution_mode"].isin(["direct_monthly", "direct_annual"]).all()
    ):
        expected = {
            (f"{SIMULATION_YEAR}{x:02d}", st)
            for x in range(1, 13) for st in FORMAL_TYPES
        }
        actual = set(zip(m["year_month"].astype(str), m["ship_type"].astype(str)))
        if actual != expected or len(m) != len(expected):
            raise RuntimeError(
                f"直接运行清单不完整: rows={len(m)} missing={sorted(expected-actual)[:20]} "
                f"extra={sorted(actual-expected)[:20]}"
            )
        if m.duplicated(["year_month", "ship_type"]).any():
            raise RuntimeError("直接运行清单存在重复月份×船型")
        return m
    # 起点区间不得有缺口或重叠。
    for st, g in m.groupby("ship_type"):
        g = g.sort_values("start_index")
        expected = 0
        for row in g.itertuples():
            if int(row.start_index) != expected:
                raise RuntimeError(f"{st} 分片起点异常: expected={expected}, got={row.start_index}")
            expected += int(row.n_requested)
        if "universe_n" in g.columns:
            universe_values = set(g["universe_n"].astype(int))
            if len(universe_values) != 1 or expected != next(iter(universe_values)):
                raise RuntimeError(
                    f"{st} 分片末端未覆盖完整船舶宇宙: end={expected}, "
                    f"universe={sorted(universe_values)}"
                )
    return m


def _read_monthly_csv(root: Path, st: str, base: str) -> list[pd.DataFrame]:
    parts = []
    for month in range(1, 13):
        ym = f"{SIMULATION_YEAR}{month:02d}"
        p = root / ym / st / f"{base}_{st}_{ym}.csv"
        if not p.is_file():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        if len(d):
            parts.append(d)
    return parts


def _read_monthly_parquet(root: Path, st: str, base: str) -> list[pd.DataFrame]:
    parts = []
    for month in range(1, 13):
        ym = f"{SIMULATION_YEAR}{month:02d}"
        p = root / ym / st / f"{base}_{st}_{ym}.parquet"
        if not p.is_file():
            raise FileNotFoundError(p)
        d = pd.read_parquet(p)
        if len(d):
            parts.append(d)
    return parts


def main() -> None:
    manifest = _require_complete_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    city_parts = []
    total_parts = []
    ops_parts = []
    ops_city_parts = []
    stat_rows = []
    direct = (
        "execution_mode" in manifest
        and manifest["execution_mode"].isin(["direct_monthly", "direct_annual"]).all()
    )
    for row in manifest.itertuples(index=False):
        root = Path(row.output_root)
        st = row.ship_type
        if direct:
            ym = str(row.year_month)
            d = root / ym / st
            city_parts.append(pd.read_csv(d / f"city_policy_delta_{st}_{ym}.csv"))
            total_parts.append(pd.read_csv(d / f"policy_delta_totals_{st}_{ym}.csv"))
            if st == "Container":
                ops_parts.append(pd.read_parquet(d / f"ops_calls_{st}_{ym}.parquet"))
                ops_city_parts.append(pd.read_parquet(d / f"ops_call_city_{st}_{ym}.parquet"))
            stat = d / f"statistics_{st}_{ym}.txt"
        else:
            city_parts.extend(_read_monthly_csv(root, st, "city_policy_delta"))
            total_parts.extend(_read_monthly_csv(root, st, "policy_delta_totals"))
            if st == "Container":
                ops_parts.extend(_read_monthly_parquet(root, st, "ops_calls"))
                ops_city_parts.extend(_read_monthly_parquet(root, st, "ops_call_city"))
            ym0 = f"{SIMULATION_YEAR}01"
            stat = root / ym0 / st / f"statistics_{st}_{ym0}.txt"
        text = stat.read_text(encoding="utf-8")
        vals = {}
        for key in ("Success", "NoData", "Failed"):
            mm = re.search(rf"{key}:(\d+)", text)
            vals[key] = int(mm.group(1)) if mm else -1
        stat_rows.append({
            "ship_type": st,
            "year_month": str(row.year_month) if direct else "annual_shard",
            "shard_id": (
                f"direct_{row.year_month}_{_norm(st)}" if direct else row.shard_id
            ),
            "n_requested": (
                vals["Success"] + vals["NoData"] + vals["Failed"]
                if direct else row.n_requested
            ),
            **vals,
        })

    city = pd.concat(city_parts, ignore_index=True) if city_parts else pd.DataFrame()
    if len(city):
        city = city.groupby(
            ["node_id", "YearMonth", "Ship_Type", "Mode_Class", "Zone_ID"],
            as_index=False,
        )[DELTA].sum()
    city.to_parquet(
        OUT / f"city_policy_delta_{SIMULATION_YEAR}.parquet", index=False
    )

    totals = pd.concat(total_parts, ignore_index=True) if total_parts else pd.DataFrame()
    if len(totals):
        missing_cols = set(DELTA + NO_MARGIN + BASE) - set(totals.columns)
        if missing_cols:
            raise RuntimeError(f"政策总量列缺失: {sorted(missing_cols)}")
        bad_zone = ~totals["Zone_ID"].astype(str).str.match(r"^R\d+__[A-Z0-9]{3}$")
        if bad_zone.any():
            raise RuntimeError(
                "正式政策区标签不是区域×责任国单元:\n"
                + totals.loc[bad_zone, ["Zone_ID"]].drop_duplicates().head(20).to_string(index=False)
            )
        totals = totals.groupby(
            ["YearMonth", "Ship_Type", "Mode_Class", "Zone_ID"],
            as_index=False,
        )[DELTA + NO_MARGIN + BASE].sum()
    totals.to_parquet(
        OUT / f"policy_delta_totals_{SIMULATION_YEAR}.parquet", index=False
    )

    ops = pd.concat(ops_parts, ignore_index=True) if ops_parts else pd.DataFrame()
    if len(ops):
        dup = int(ops.duplicated("Call_ID").sum())
        if dup:
            raise RuntimeError(f"挂靠 Call_ID 跨分片重复: {dup}")
        ops = ops.sort_values(["YearMonth", "Port_Name", "Arrive_UTC"])
    ops.to_parquet(
        OUT / f"ops_calls_container_{SIMULATION_YEAR}.parquet", index=False
    )

    ops_city = (
        pd.concat(ops_city_parts, ignore_index=True)
        if ops_city_parts else pd.DataFrame()
    )
    if len(ops_city):
        val = [c for c in (
            "OPS_AE_kWh", "OPS_BO_kWh", "Berth_SO2_g", "Berth_PM_g",
            "Berth_NOx_g", *DELTA,
        ) if c in ops_city.columns]
        ops_city = ops_city.groupby(
            ["Call_ID", "Port_Name", "node_id", "YearMonth"], as_index=False
        )[val].sum()
    ops_city.to_parquet(
        OUT / f"ops_call_city_container_{SIMULATION_YEAR}.parquet", index=False
    )

    stats = pd.DataFrame(stat_rows)
    processed = stats[["Success", "NoData", "Failed"]].sum(axis=1)
    if not processed.eq(stats["n_requested"]).all():
        raise RuntimeError(
            "分片处理计数不守恒:\n"
            + stats.loc[~processed.eq(stats["n_requested"])].to_string(index=False)
        )
    stats.to_csv(TABLES / "ais_policy_shard_processing_qc.csv",
                 index=False, encoding="utf-8-sig")
    requested = int(stats["n_requested"].sum())
    failed = int(stats["Failed"].sum())
    gt_known = int(ops["Gross_Tonnage"].notna().sum()) if len(ops) else 0
    gt_missing = int(ops["Gross_Tonnage"].isna().sum()) if len(ops) else 0
    qc = pd.DataFrame([
        {"metric": (
            "requested_ship_month_records" if direct else "requested_unique_ships"
        ), "value": requested},
        {"metric": "successful_ships", "value": int(stats["Success"].sum())},
        {"metric": "no_data_ships", "value": int(stats["NoData"].sum())},
        {"metric": "failed_ships", "value": failed},
        {"metric": "city_delta_rows", "value": len(city)},
        {"metric": "zone_total_rows", "value": len(totals)},
        {"metric": "container_calls", "value": len(ops)},
        {"metric": "container_call_city_rows", "value": len(ops_city)},
        {"metric": "container_calls_gt_known", "value": gt_known},
        {"metric": "container_calls_gt_missing", "value": gt_missing},
        {"metric": "container_call_gt_match_rate",
         "value": gt_known / len(ops) if len(ops) else 0.0},
    ])
    qc.to_csv(TABLES / "ais_policy_consolidation_qc.csv",
              index=False, encoding="utf-8-sig")
    if failed:
        raise RuntimeError(f"正式合并发现失败船舶: {failed}")
    print(
        f"[complete] processed_units={requested:,} city_delta={len(city):,} "
        f"calls={len(ops):,} call_city={len(ops_city):,} OUT={OUT}"
    )


if __name__ == "__main__":
    main()
