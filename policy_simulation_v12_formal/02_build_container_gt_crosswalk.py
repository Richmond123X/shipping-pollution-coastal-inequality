# -*- coding: utf-8 -*-
"""构建 V12 岸电情景正式集装箱船 GT 交叉表。

原则：
1. GT 只来自原始船舶档案，不以 DWT 推算；
2. 优先直接 MMSI 匹配，其次通过 MMSI--IMO 对照匹配同一船舶；
3. 每条结果保留来源与匹配方法，冲突另表输出；
4. 用正式模拟年份12个月数据库中的 Container 船舶宇宙做覆盖率审计。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from policy_run_config import AIS_DIR, GT_CROSSWALK, ROOT, SIMULATION_YEAR


SHIP_DIR = Path(r"C:\Users\86176\Desktop\Ship Emission\Ship_File")
MMSI_DIR = Path(r"C:\Users\86176\Desktop\Ship Emission\MMSI")


def _num_key(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("\u00a0", "", regex=False).str.strip(),
        errors="coerce",
    ).astype("Int64")


def _read_sin_workbook(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    header_row = None
    header = None
    for ridx, row in enumerate(
        ws.iter_rows(min_row=1, max_row=min(8, ws.max_row), values_only=True),
        start=1,
    ):
        vals = [str(v).strip() if v is not None else "" for v in row]
        if "MMSI" in vals and "GT" in vals:
            header_row, header = ridx, vals
            break
    if header_row is None:
        return pd.DataFrame()
    rows = list(
        ws.iter_rows(min_row=header_row + 1, max_row=ws.max_row, values_only=True)
    )
    frame = pd.DataFrame(rows, columns=header)
    keep = [c for c in ("MMSI", "IMO", "GT", "Built") if c in frame.columns]
    frame = frame[keep].copy()
    frame["MMSI"] = _num_key(frame["MMSI"])
    if "IMO" in frame:
        frame["IMO"] = _num_key(frame["IMO"])
    else:
        frame["IMO"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["GT"] = pd.to_numeric(frame["GT"], errors="coerce")
    frame["BuildYear"] = pd.to_numeric(frame.get("Built"), errors="coerce")
    frame["source_file"] = path.name
    return frame[["MMSI", "IMO", "GT", "BuildYear", "source_file"]]


def _read_ship_files() -> pd.DataFrame:
    parts = []
    for p in (SHIP_DIR / "Ship_file_new.csv", SHIP_DIR / "Ship_file.csv"):
        d = pd.read_csv(
            p,
            usecols=lambda c: c in {"MMSI", "IMO", "GT", "BuildYear", "ShipType"},
            low_memory=False,
            encoding_errors="replace",
        )
        if "ShipType" in d:
            d = d[d["ShipType"].astype(str).str.casefold().str.contains("container")]
        d["MMSI"] = _num_key(d["MMSI"])
        d["IMO"] = _num_key(d["IMO"])
        d["GT"] = pd.to_numeric(d["GT"], errors="coerce")
        d["BuildYear"] = pd.to_numeric(d.get("BuildYear"), errors="coerce")
        d["source_file"] = p.name
        parts.append(d[["MMSI", "IMO", "GT", "BuildYear", "source_file"]])
    return pd.concat(parts, ignore_index=True)


def _read_imo_bridge() -> pd.DataFrame:
    p = MMSI_DIR / "Containership_MMSI_IMO.xlsx"
    d = pd.read_excel(p)
    d["MMSI"] = _num_key(d["MMSI"])
    d["IMO"] = _num_key(d["IMO"])
    return d.dropna(subset=["MMSI", "IMO"]).drop_duplicates("MMSI", keep="last")


def _read_container_universe() -> pd.DataFrame:
    parts = []
    for p in sorted(AIS_DIR.glob(f"ais_{SIMULATION_YEAR}??.db")):
        uri = p.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=120)
        d = pd.read_sql_query(
            "SELECT MMSI, DWT, Build_Year, Ship_Type FROM ship_info "
            "WHERE lower(replace(Ship_Type, ' ', '')) = 'container'",
            conn,
        )
        conn.close()
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["MMSI"] = _num_key(out["MMSI"])
    return out.drop_duplicates("MMSI", keep="first")


def _candidate_table(universe: pd.DataFrame) -> pd.DataFrame:
    ship = _read_ship_files()
    sin_parts = [
        _read_sin_workbook(p)
        for p in sorted(MMSI_DIR.glob("*.xlsx"))
        if p.name != "Containership_MMSI_IMO.xlsx" and "LNG" not in p.name.upper()
    ]
    sin = pd.concat([d for d in sin_parts if len(d)], ignore_index=True)
    bridge = _read_imo_bridge()

    candidates = []

    def direct(src: pd.DataFrame, priority: int, method: str) -> None:
        d = universe[["MMSI"]].merge(src, on="MMSI", how="inner")
        d["priority"] = priority
        d["match_method"] = method
        candidates.append(d)

    direct(ship, 1, "ship_file_direct_mmsi")
    direct(sin, 3, "sin_direct_mmsi")

    # 当前 MMSI -> IMO，再用 IMO 找 GT；适用于船舶更换 MMSI 的情况。
    uimo = universe[["MMSI"]].merge(
        bridge.rename(columns={"IMO": "bridge_IMO"}), on="MMSI", how="left"
    )
    for src, priority, method in (
        (ship, 2, "ship_file_via_imo"),
        (sin, 4, "sin_via_imo"),
    ):
        by_imo = src.dropna(subset=["IMO", "GT"]).copy()
        d = uimo.merge(
            by_imo,
            left_on="bridge_IMO",
            right_on="IMO",
            how="inner",
            suffixes=("", "_src"),
        )
        d["IMO"] = d["bridge_IMO"]
        d["priority"] = priority
        d["match_method"] = method
        candidates.append(
            d[["MMSI", "IMO", "GT", "BuildYear", "source_file",
               "priority", "match_method"]]
        )

    out = pd.concat(candidates, ignore_index=True)
    out = out.dropna(subset=["MMSI", "GT"])
    out = out[(out["GT"] > 0) & np.isfinite(out["GT"])]
    return out


def main() -> None:
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "tables").mkdir(parents=True, exist_ok=True)
    universe = _read_container_universe()
    cand = _candidate_table(universe)

    # 同一 MMSI 的显著 GT 冲突全部留痕；正式值按来源优先级确定。
    audit = (
        cand.groupby("MMSI", as_index=False)
        .agg(
            n_candidates=("GT", "size"),
            n_unique_gt=("GT", "nunique"),
            gt_min=("GT", "min"),
            gt_max=("GT", "max"),
            sources=("source_file", lambda s: ";".join(sorted(set(map(str, s))))),
            methods=("match_method", lambda s: ";".join(sorted(set(map(str, s))))),
        )
    )
    audit["relative_range"] = np.where(
        audit["gt_max"] > 0,
        (audit["gt_max"] - audit["gt_min"]) / audit["gt_max"],
        np.nan,
    )
    conflicts = audit[(audit["n_unique_gt"] > 1) & (audit["relative_range"] > 0.05)]

    chosen = (
        cand.sort_values(["MMSI", "priority", "GT"], ascending=[True, True, False])
        .drop_duplicates("MMSI", keep="first")
        .rename(columns={"source_file": "GT_Source", "match_method": "GT_Match_Method"})
    )
    chosen = universe[["MMSI", "DWT", "Build_Year"]].merge(
        chosen[["MMSI", "IMO", "GT", "BuildYear", "GT_Source", "GT_Match_Method"]],
        on="MMSI",
        how="left",
    )
    chosen["BuildYear"] = pd.to_numeric(chosen["BuildYear"], errors="coerce").fillna(
        pd.to_numeric(chosen["Build_Year"], errors="coerce")
    )
    chosen = chosen[["MMSI", "IMO", "GT", "BuildYear", "GT_Source", "GT_Match_Method"]]

    out_path = GT_CROSSWALK
    chosen.to_csv(out_path, index=False, encoding="utf-8-sig")
    conflicts.to_csv(
        ROOT / "tables" / "container_gt_conflicts_over5pct.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matched = int(chosen["GT"].notna().sum())
    summary = pd.DataFrame([
        {"metric": f"{SIMULATION_YEAR}_unique_container_mmsi", "value": len(chosen)},
        {"metric": "gt_matched_mmsi", "value": matched},
        {"metric": "gt_missing_mmsi", "value": len(chosen) - matched},
        {"metric": "gt_match_rate", "value": matched / len(chosen)},
        {"metric": "gt_conflicts_over5pct", "value": len(conflicts)},
    ])
    summary.to_csv(
        ROOT / "tables" / "container_gt_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[complete] Container MMSI={len(chosen):,} GT matched={matched:,} "
        f"({matched/len(chosen):.2%}) conflicts>5%={len(conflicts):,}\n{out_path}"
    )


if __name__ == "__main__":
    main()
