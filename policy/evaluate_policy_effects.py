# -*- coding: utf-8 -*-
"""研究 4 步骤 3：政策效果评估（方案 v6 7.3 节，分维度报告、不做综合评分）。

输入(排放主程序在政策情景启用下产出的评估年 12 个月):
  results_monthly/<YYYYMM>/<船型>/city_policy_delta_<船型>_<YYYYMM>.csv
  results_monthly/<YYYYMM>/<船型>/policy_delta_totals_<船型>_<YYYYMM>.csv
  results_monthly/<YYYYMM>/<船型>/city_emissions_<船型>_<YYYYMM>.csv   (基线分母)
以及 协变量面板(S1_pop / pop_density_km2 / S2_vuln_share, 取评估年) 和
policy_trigger_nodes.csv(高风险触发节点,标记受益去向)。

情景组装(无需单算): SECA=D_SO2_SECA+D_PM_SECA, NECA=D_NOx_NECA, BOTH=三列全取。

输出(--out-dir):
  policy_eval_zone_summary_<year>.csv  单元×情景×污染物×工况: 减排量/区内基线/减排率
  policy_eval_city_<year>.csv          城市级: 减排量/城市基线/减排率/暴露与脆弱指标
  policy_eval_equity_<year>.csv        单元×情景: 总减排、人口加权暴露削减、
                                       脆弱人群受益、高风险触发节点受益份额

用法： python policy/evaluate_policy_effects.py            # 全按 config.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "emission_network"))

DELTA_COLS = ["D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA"]
POL_OF_DELTA = {"D_SO2_SECA": "SO2", "D_PM_SECA": "PM", "D_NOx_NECA": "NOx"}
BASE_COLS = {"SO2": "B_SO2", "PM": "B_PM", "NOx": "B_NOx"}
SCENARIOS = {"SECA": ["SO2", "PM"], "NECA": ["NOx"], "BOTH": ["SO2", "PM", "NOx"]}
G_PER_T = 1e6


def _load_cfg():
    # 共享加载器：含跨机器路径自动迁移(用户名/盘符不同的电脑无需改 config.yaml)
    from config_loader import load_config
    return load_config()


def _months(year: int) -> list[str]:
    return [f"{year}{m:02d}" for m in range(1, 13)]


def _ship_dirs(results_root: str, ym: str, ship_types: list[str]) -> list[tuple[str, str]]:
    out = []
    for st in ship_types:
        d = os.path.join(results_root, ym, st)
        if os.path.isdir(d):
            out.append((st, d))
        else:  # 目录名可能被 _safe_folder_name 处理过, 容错 glob
            hits = glob.glob(os.path.join(results_root, ym, f"{st.replace(' ', '*')}*"))
            out.extend((st, h) for h in hits if os.path.isdir(h))
    return out


def _read_year(results_root: str, year: int, ship_types: list[str]):
    """拼全年 (delta_city, delta_tot, baseline_city, missing_items)。"""
    dc, dt, bc, missing = [], [], [], []
    for ym in _months(year):
        pairs = _ship_dirs(results_root, ym, ship_types)
        found_types = {st for st, _ in pairs}
        for ship_type in ship_types:
            if ship_type not in found_types:
                missing.append(f"{ym}/{ship_type}(directory)")
        for st, d in pairs:
            stn = os.path.basename(d)
            f_dc = os.path.join(d, f"city_policy_delta_{stn}_{ym}.csv")
            f_dt = os.path.join(d, f"policy_delta_totals_{stn}_{ym}.csv")
            f_bc = os.path.join(d, f"city_emissions_{stn}_{ym}.csv")
            if not (os.path.isfile(f_dc) and os.path.isfile(f_dt)):
                missing.append(f"{ym}/{stn}(policy)")
                continue
            a = pd.read_csv(f_dc)
            b = pd.read_csv(f_dt)
            if len(a):
                dc.append(a)
            if len(b):
                dt.append(b)
            if os.path.isfile(f_bc):
                c = pd.read_csv(f_bc)
                if len(c):
                    c["Ship_Type"] = stn
                    bc.append(c)
            else:
                missing.append(f"{ym}/{stn}(baseline)")
    if missing:
        print(f"  ⚠ 缺失 {len(missing)} 项(月份/船型政策或基线产物): {missing[:8]} ...")
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(dc), cat(dt), cat(bc), missing


def zone_summary(dt: pd.DataFrame) -> pd.DataFrame:
    """单元×情景×污染物×工况(含 ALL): 减排量(t)、区内基线(t)、减排率。"""
    rows = []
    grp_mode = dt.groupby(["Zone_ID", "Mode_Class"], as_index=False)[
        DELTA_COLS + list(BASE_COLS.values())].sum()
    grp_all = dt.groupby(["Zone_ID"], as_index=False)[
        DELTA_COLS + list(BASE_COLS.values())].sum()
    grp_all["Mode_Class"] = "ALL"
    both = pd.concat([grp_mode, grp_all], ignore_index=True)
    for scn, pols in SCENARIOS.items():
        for _, r in both.iterrows():
            for pol in pols:
                dcol = [k for k, v in POL_OF_DELTA.items() if v == pol][0]
                d_t = r[dcol] / G_PER_T
                b_t = r[BASE_COLS[pol]] / G_PER_T
                rows.append({
                    "Zone_ID": r["Zone_ID"], "Scenario": scn, "Pollutant": pol,
                    "Mode_Class": r["Mode_Class"],
                    "Delta_t": d_t, "Baseline_inzone_t": b_t,
                    "Reduction_pct": (100.0 * d_t / b_t) if b_t > 0 else np.nan,
                })
    return pd.DataFrame(rows).sort_values(
        ["Zone_ID", "Scenario", "Pollutant", "Mode_Class"])


def city_table(dc: pd.DataFrame, bc: pd.DataFrame, cov: pd.DataFrame,
               trigger_nodes: set) -> pd.DataFrame:
    """城市级年度表: 每(单元,节点,污染物)减排量、城市基线、减排率、暴露/脆弱指标。"""
    d = (dc.groupby(["Zone_ID", "node_id"], as_index=False)[DELTA_COLS].sum()
         .melt(id_vars=["Zone_ID", "node_id"], value_vars=DELTA_COLS,
               var_name="_d", value_name="Delta_g"))
    d["Pollutant"] = d["_d"].map(POL_OF_DELTA)
    d = d[d["Delta_g"] > 0].drop(columns="_d")

    base = pd.DataFrame()
    if len(bc):
        keep = [c for c in ("SO2_Total", "PM_Total", "NOx_Total") if c in bc.columns]
        base = (bc.groupby("node_id", as_index=False)[keep].sum()
                .melt(id_vars="node_id", var_name="_b", value_name="City_baseline_g"))
        base["Pollutant"] = base["_b"].str.replace("_Total", "", regex=False)
        base = base.drop(columns="_b")
    out = d.merge(base, on=["node_id", "Pollutant"], how="left") if len(base) else d
    if "City_baseline_g" not in out.columns:
        out["City_baseline_g"] = np.nan

    out = out.merge(cov, on="node_id", how="left")
    out["Delta_t"] = out["Delta_g"] / G_PER_T
    out["City_baseline_t"] = out["City_baseline_g"] / G_PER_T
    out["City_reduction_pct"] = np.where(
        out["City_baseline_g"] > 0, 100.0 * out["Delta_g"] / out["City_baseline_g"],
        np.nan)
    # 暴露削减代理 = Δ排放 × 人口密度(E2 人口端口径); 脆弱受益 = 再乘脆弱人群比例
    out["Exposure_reduction"] = out["Delta_g"] * out["pop_density_km2"]
    out["Vuln_exposure_reduction"] = out["Exposure_reduction"] * out["S2_vuln_share"]
    out["Is_trigger_node"] = out["node_id"].isin(trigger_nodes)
    cols = ["Zone_ID", "node_id", "Pollutant", "Delta_t", "City_baseline_t",
            "City_reduction_pct", "S1_pop", "pop_density_km2", "S2_vuln_share",
            "Exposure_reduction", "Vuln_exposure_reduction", "Is_trigger_node"]
    return out[cols].sort_values(["Zone_ID", "Pollutant", "Delta_t"],
                                 ascending=[True, True, False])


def equity_table(city: pd.DataFrame) -> pd.DataFrame:
    """单元×情景: 总减排/人口加权暴露削减/脆弱受益/触发节点受益份额。"""
    rows = []
    for scn, pols in SCENARIOS.items():
        sub = city[city["Pollutant"].isin(pols)]
        for zid, g in sub.groupby("Zone_ID"):
            trig = g[g["Is_trigger_node"]]
            rows.append({
                "Zone_ID": zid, "Scenario": scn,
                "Delta_total_t": g["Delta_t"].sum(),
                "N_cities_benefiting": g["node_id"].nunique(),
                "Exposure_reduction_sum": g["Exposure_reduction"].sum(),
                "Vuln_exposure_reduction_sum": g["Vuln_exposure_reduction"].sum(),
                "Trigger_node_delta_share": (
                    trig["Delta_t"].sum() / g["Delta_t"].sum()
                    if g["Delta_t"].sum() > 0 else np.nan),
                "N_trigger_nodes_benefiting": trig["node_id"].nunique(),
            })
    return pd.DataFrame(rows).sort_values(["Scenario", "Delta_total_t"],
                                          ascending=[True, False])


def main() -> None:
    cfg = _load_cfg()
    pol = cfg.get("policy") or {}
    paths = cfg["paths"]
    emis = cfg["emission_network"]

    ap = argparse.ArgumentParser(description="研究4: 政策效果评估(分维度)")
    ap.add_argument("--year", type=int, default=int(pol.get("eval_year", 2025)))
    ap.add_argument("--results-root", default=str(paths["results_root"]))
    ap.add_argument("--covariates-csv",
                    default=os.path.join(paths["covariates_dir"],
                                         "city_covariates_2019_2028.csv"))
    ap.add_argument("--trigger-nodes-csv", default=str(pol.get("trigger_nodes_csv", "")))
    ap.add_argument("--out-dir", default=str(pol.get(
        "eval_output_dir", os.path.join(HERE, "outputs", "evaluation"))))
    ap.add_argument("--ship-types", default=",".join(emis["ship_types"]))
    ap.add_argument(
        "--allow-incomplete", action="store_true",
        help="调试用：允许缺月/缺船型并生成部分年度汇总；正式 Fig.6 不应启用",
    )
    args = ap.parse_args()

    ship_types = [s.strip() for s in args.ship_types.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] 读取 {args.year} 年 12 个月政策产物({len(ship_types)} 船型)")
    dc, dt, bc, missing = _read_year(args.results_root, args.year, ship_types)
    if missing and not args.allow_incomplete:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... 另有 {len(missing)-20} 项" if len(missing) > 20 else ""
        raise SystemExit(
            "2025 政策评估输入不完整，已停止以防生成部分年度 Fig.6。缺失项：\n  "
            f"{preview}{suffix}\n若仅做调试可显式传入 --allow-incomplete。"
        )
    if len(dt) == 0:
        raise SystemExit("无任何 policy_delta_totals 产物——先在政策情景启用下跑评估年")
    print(f"  城市 delta 行 {len(dc):,}  区 totals 行 {len(dt):,}  基线行 {len(bc):,}")

    cov = pd.read_csv(args.covariates_csv,
                      usecols=["node_id", "year", "S1_pop", "S2_vuln_share",
                               "pop_density_km2"])
    cov = cov[cov["year"] == args.year].drop(columns="year")
    trigger_nodes: set = set()
    if args.trigger_nodes_csv and os.path.isfile(args.trigger_nodes_csv):
        trigger_nodes = set(pd.read_csv(args.trigger_nodes_csv)["node_id"])
        print(f"  触发节点 {len(trigger_nodes)} 个(用于受益份额)")

    print("[2/4] 单元级汇总(区内精确口径)")
    zs = zone_summary(dt)
    f1 = os.path.join(args.out_dir, f"policy_eval_zone_summary_{args.year}.csv")
    zs.to_csv(f1, index=False, encoding="utf-8-sig")

    print("[3/4] 城市级表(集水区口径 + 暴露/脆弱指标)")
    ct = city_table(dc, bc, cov, trigger_nodes)
    f2 = os.path.join(args.out_dir, f"policy_eval_city_{args.year}.csv")
    ct.to_csv(f2, index=False, encoding="utf-8-sig")

    print("[4/4] 公平性/受益去向表")
    eq = equity_table(ct)
    f3 = os.path.join(args.out_dir, f"policy_eval_equity_{args.year}.csv")
    eq.to_csv(f3, index=False, encoding="utf-8-sig")

    print("\n==== 单元级减排摘要(Scenario=BOTH, Mode=ALL) ====")
    show = zs[(zs["Scenario"] == "BOTH") & (zs["Mode_Class"] == "ALL")]
    print(show.to_string(index=False,
                         formatters={"Delta_t": "{:,.1f}".format,
                                     "Baseline_inzone_t": "{:,.1f}".format,
                                     "Reduction_pct": "{:.1f}%".format}))
    print(f"\n写出:\n  {f1}\n  {f2}\n  {f3}")


if __name__ == "__main__":
    main()
