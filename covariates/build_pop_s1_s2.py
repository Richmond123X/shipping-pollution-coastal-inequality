# -*- coding: utf-8 -*-
"""S1 人口总数 / S2 脆弱人群比例 / 人口密度（E2 人口端）（方案 3.9.5）。

数据：WorldPop 1km 栅格（注意：实际数据为 1km 而非方案所写 100m，写入 SI 说明）。
  data/人口/global_pop_<year>_CN_1km_R2025A_UA_v1.tif                 总人口
  data/人口结构/global_<pref>_<year>_CN_1km_R2025A_UA_v1.tif          年龄段
脆弱人群 = t_00(0-1) + t_01(1-4) + t_65..t_90(65 岁及以上)，共 8 段。
t_01 缺 2019：用 2020 值回填（慢变量；S2 评估只用最新观测，影响可忽略）。

口径：城市多边形本身（非集水区）分区求和；密度 = S1 / area_km2(EPSG:6933)。
实现：81 个全球栅格的分区统计**多进程并行**，逐栅格结果缓存到 cache_pop/，
      中断重跑自动跳过已完成栅格。

输出 outputs/pop_s1_s2.csv: node_id, year, S1_pop, S2_vuln_share, pop_density_km2
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import OUT, YEARS, load_config, load_cities

AGE_PREFIXES = ["t_00", "t_01", "t_65", "t_70", "t_75", "t_80", "t_85", "t_90"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_pop")
N_WORKERS = 8

_w_geoms = None  # worker 进程内的城市几何


def _init_worker():
    global _w_geoms
    cfg = load_config()
    _w_geoms = load_cities(cfg).geometry


def _run_one(job: tuple[str, str]) -> str:
    """job = (cache_key, tif_path)；分区求和并缓存。"""
    from rasterstats import zonal_stats
    cache_key, tif = job
    cache = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache):
        return f"[缓存] {cache_key}"
    zs = zonal_stats(_w_geoms, tif, stats="sum", all_touched=False, nodata=-99999)
    v = np.array([z["sum"] if z["sum"] is not None else np.nan for z in zs])
    np.save(cache, v)
    return f"[完成] {cache_key}"


def build_jobs(cfg: dict) -> list[tuple[str, str]]:
    pop_dir = cfg["paths"]["pop_dir"]
    age_dir = cfg["paths"]["pop_age_dir"]
    jobs = []
    for y in YEARS:
        jobs.append((f"pop_{y}",
                     os.path.join(pop_dir, f"global_pop_{y}_CN_1km_R2025A_UA_v1.tif")))
    for pref in AGE_PREFIXES:
        for y in YEARS:
            src_y = 2020 if (pref == "t_01" and y == 2019) else y
            key = f"{pref}_{src_y}"
            tif = os.path.join(age_dir, f"global_{pref}_{src_y}_CN_1km_R2025A_UA_v1.tif")
            if not os.path.exists(tif):
                raise FileNotFoundError(tif)
            if (key, tif) not in jobs:
                jobs.append((key, tif))
    # 去重（t_01 2019/2020 共用 t_01_2020）
    seen, uniq = set(), []
    for k, t in jobs:
        if k not in seen:
            seen.add(k)
            uniq.append((k, t))
    return uniq


def main() -> None:
    cfg = load_config()
    cities = load_cities(cfg)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 缓存指纹：节点数变化（如带宽校准后 1502→1425）则缓存按节点顺序失效，必须清空重算。
    fp_path = os.path.join(CACHE_DIR, "_node_count.txt")
    cur_fp = str(len(cities))
    old_fp = open(fp_path).read().strip() if os.path.exists(fp_path) else None
    if old_fp is not None and old_fp != cur_fp:
        import glob as _glob
        for f in _glob.glob(os.path.join(CACHE_DIR, "*.npy")):
            os.remove(f)
        print(f"节点数变化 {old_fp}→{cur_fp}，已清空 pop 缓存重算", flush=True)
    open(fp_path, "w").write(cur_fp)

    jobs = build_jobs(cfg)
    pending = [j for j in jobs if not os.path.exists(os.path.join(CACHE_DIR, f"{j[0]}.npy"))]
    print(f"分区统计任务: {len(jobs)} 个栅格, 待算 {len(pending)} 个, 并行 {N_WORKERS}", flush=True)
    if pending:
        try:
            with Pool(N_WORKERS, initializer=_init_worker) as pool:
                for i, msg in enumerate(pool.imap_unordered(_run_one, pending)):
                    print(f"{i + 1}/{len(pending)} {msg}", flush=True)
        except (OSError, PermissionError, Exception) as e:  # noqa: BLE001
            # Windows 下多进程 spawn 偶发 WinError 5 拒绝访问（句柄复制失败）；
            # 逐栅格缓存，已完成的会跳过，降级为串行续算保证最终完成。
            print(f"⚠ 并行失败 ({type(e).__name__}: {e})，降级串行续算", flush=True)
            _init_worker()
            still = [j for j in pending
                     if not os.path.exists(os.path.join(CACHE_DIR, f"{j[0]}.npy"))]
            for i, j in enumerate(still):
                print(f"{i + 1}/{len(still)} {_run_one(j)}", flush=True)

    def load(key: str) -> np.ndarray:
        return np.load(os.path.join(CACHE_DIR, f"{key}.npy"))

    # 小面/环礁在 all_touched=False、1km 分辨率下像元中心取不到 → S1=0（如基里巴斯实际12.7万人）。
    # 先按标准口径算，再对 S1==0 的节点用 all_touched=True 逐栅格重算（仅这些节点，不扰动其余）。
    pop_at = {}   # (key) -> all_touched 重算值（按需）
    zero_mask_any = np.zeros(len(cities), dtype=bool)
    s1_all = {}
    for y in YEARS:
        s1_all[y] = load(f"pop_{y}").copy()
        zero_mask_any |= (np.nan_to_num(s1_all[y], nan=0.0) <= 0)

    fix_idx = np.where(zero_mask_any)[0]
    uninhabited = set()
    if len(fix_idx):
        from rasterstats import zonal_stats
        from shapely.geometry import MultiPolygon
        sub_geom = cities.geometry.iloc[fix_idx]
        print(f"对 {len(fix_idx)} 个 S1=0 节点用 all_touched=True 重算 ...", flush=True)

        # 跨日界线节点（如基里巴斯，多边形跨 [-174,176]）整体 zonal 会读到近全球窗口、
        # 跨缝栅格化不稳定 → 按几何 part 拆分（每个 part 不跨缝），逐 part 求和再相加。
        def _parts(geom):
            if geom is None:
                return []
            return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

        def at_sum(tif):
            out = np.zeros(len(fix_idx))
            for k, g in enumerate(sub_geom.values):
                b = g.bounds if g is not None else (0, 0, 0, 0)
                if (b[2] - b[0]) > 180:  # 跨日界线 → 逐 part
                    s = 0.0
                    for part in _parts(g):
                        zs = zonal_stats([part], tif, stats="sum", all_touched=True, nodata=-99999)
                        s += zs[0]["sum"] or 0.0
                    out[k] = s
                else:
                    zs = zonal_stats([g], tif, stats="sum", all_touched=True, nodata=-99999)
                    out[k] = zs[0]["sum"] or 0.0
            return out

        for y in YEARS:
            tif = os.path.join(cfg["paths"]["pop_dir"],
                               f"global_pop_{y}_CN_1km_R2025A_UA_v1.tif")
            vals = at_sum(tif)
            s1_all[y][fix_idx] = np.where(s1_all[y][fix_idx] > 0, s1_all[y][fix_idx], vals)
            for pref in AGE_PREFIXES:
                src_y = 2020 if (pref == "t_01" and y == 2019) else y
                key = f"{pref}_{src_y}_AT"
                if key not in pop_at:
                    t2 = os.path.join(cfg["paths"]["pop_age_dir"],
                                      f"global_{pref}_{src_y}_CN_1km_R2025A_UA_v1.tif")
                    pop_at[key] = at_sum(t2)
        # 重算后仍为 0 的节点 = 真无人区（WorldPop 无像元）
        for k, i in enumerate(fix_idx):
            if all(s1_all[y][i] <= 0 for y in YEARS):
                uninhabited.add(cities["node_id"].iloc[i])
        print(f"  其中真无人区(重算后仍为0): {sorted(uninhabited)}", flush=True)

    fix_pos = {int(i): k for k, i in enumerate(fix_idx)}
    n, ny = len(cities), len(YEARS)
    # 构建 [节点, 年] 矩阵，便于做逐节点时间填补
    S1 = np.zeros((n, ny))
    VULN = np.zeros((n, ny))
    for yj, y in enumerate(YEARS):
        S1[:, yj] = s1_all[y]
        vuln = np.zeros(n)
        for pref in AGE_PREFIXES:
            src_y = 2020 if (pref == "t_01" and y == 2019) else y
            v = np.nan_to_num(load(f"{pref}_{src_y}"), nan=0.0)
            if len(fix_idx):
                at = pop_at[f"{pref}_{src_y}_AT"]
                for i in fix_idx:
                    v[i] = at[fix_pos[i]]
            vuln += v
        VULN[:, yj] = vuln

    # 逐年栅格覆盖间隙修复（如 WorldPop R2025A 仅 2025 层含基里巴斯环礁人口，其余年为 0）：
    # 节点若部分年 >0、部分年 =0（且非真无人区）→ 按时间线性插值 + 前后向延展填补 0 年。
    # 人口为慢变量，用相邻可得年份填补符合方案 3.8 "末值/最新可得值" 策略。
    pop_year_filled = set()
    for i in range(n):
        nz = S1[i] > 0
        if nz.any() and not nz.all():
            ser = pd.Series(np.where(nz, S1[i], np.nan))
            S1[i] = ser.interpolate().ffill().bfill().values
            vser = pd.Series(np.where(nz, VULN[i], np.nan))
            VULN[i] = vser.interpolate().ffill().bfill().values
            pop_year_filled.add(cities["node_id"].iloc[i])
    if pop_year_filled:
        print(f"  逐年栅格间隙时间填补节点: {sorted(pop_year_filled)}", flush=True)

    rows = []
    for yj, y in enumerate(YEARS):
        s1 = S1[:, yj]
        share = np.divide(VULN[:, yj], s1, out=np.full(n, np.nan),
                          where=np.nan_to_num(s1, nan=0.0) > 0)
        dens = s1 / cities["area_km2"].values
        for i in range(n):
            nid = cities["node_id"].iloc[i]
            flag = "uninhabited" if nid in uninhabited else (
                "pop_year_filled" if nid in pop_year_filled else "")
            s1v, shv, dv = s1[i], share[i], dens[i]
            if nid in uninhabited:
                s1v, shv, dv = 0.0, 0.0, 0.0
            rows.append({"node_id": nid, "year": y, "S1_pop": s1v,
                         "S2_vuln_share": shv, "pop_density_km2": dv,
                         "pop_flag": flag})

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "pop_s1_s2.csv"), index=False, encoding="utf-8-sig")
    bad = out[(out.S2_vuln_share < 0) | (out.S2_vuln_share > 1)]
    print(f"完成: {out.node_id.nunique()} 节点; S1 NaN {out.S1_pop.isna().sum()}; "
          f"S2 NaN {out.S2_vuln_share.isna().sum()}; 越界 {len(bad)}; "
          f"无人区 {len(uninhabited)}; S2 中位 {out.S2_vuln_share.median():.3f}", flush=True)


if __name__ == "__main__":
    main()
