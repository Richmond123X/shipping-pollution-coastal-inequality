# -*- coding: utf-8 -*-
"""Port2City L2 GIS 确定层：名称倒排索引 + 空间验证 + 面积带宽层级校准。

全部确定性、可复算。被 run_port2city.py 调用。

核心流程（研究方案 3.6.3 节步骤表）：
  步骤 2  对 World.shp 的 NAME_1-5、VARNAME_1-4、NL_NAME_1-3（含 NAME_0 城市国家）
          建归一化倒排索引（去重音、大小写无关、去标点、行政后缀变体）
  步骤 3  用 L1 城市名+别名查索引，按 country(GID_0) 过滤，按命中层级 GID 分组
  步骤 4  空间验证：候选面包含港口点，或边界距港口点 ≤ 25 km（测地线距离）
  步骤 5  命中多行按 GID 视作一个单元（节点生成阶段 dissolve）
  步骤 6  面积带宽校准：score=|log10(A/1e4)| 在层级链上取最小，带宽 [500, 80000] km²
  步骤 7  名称无命中 → 纯空间兜底（within → nearest），标低置信
"""
from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from pyproj import Geod
from shapely.geometry import Point
from shapely.ops import nearest_points
from unidecode import unidecode

GEOD = Geod(ellps="WGS84")

WORLD_COLS = (
    ["UID", "GID_0", "NAME_0"]
    + [f"GID_{i}" for i in range(1, 6)]
    + [f"NAME_{i}" for i in range(1, 6)]
    + [f"VARNAME_{i}" for i in range(1, 5)]
    + [f"NL_NAME_{i}" for i in range(1, 4)]
)

# 归一化时可剥离的行政后缀词（仅当剥离后仍有剩余 token 时生效）
SUFFIX_TOKENS = {
    "city", "county", "municipality", "district", "province", "region",
    "prefecture", "governorate", "state", "department", "parish",
    "borough", "division", "regency", "oblast", "krai", "voivodeship",
    "shi", "xian", "qu",
}

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize(name: str) -> str:
    """去重音、大小写无关、去标点、空白折叠。"""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = unidecode(name).casefold().translate(_PUNCT_TABLE)
    return " ".join(s.split())


def name_variants(name: str) -> set[str]:
    """归一化主形 + 行政后缀剥离变体（如 'harris county'→'harris'）。"""
    base = normalize(name)
    if not base:
        return set()
    out = {base}
    toks = base.split()
    while len(toks) > 1 and toks[-1] in SUFFIX_TOKENS:
        toks = toks[:-1]
        out.add(" ".join(toks))
    toks = base.split()
    while len(toks) > 1 and toks[0] in SUFFIX_TOKENS:
        toks = toks[1:]
        out.add(" ".join(toks))
    return out


# ---------------------------------------------------------------------------
# World.shp 加载与逐行面积缓存
# ---------------------------------------------------------------------------
def load_world(world_shp: str) -> gpd.GeoDataFrame:
    print(f"加载 World.shp ({world_shp}) ...")
    world = pyogrio.read_dataframe(world_shp, columns=WORLD_COLS)
    world = world.reset_index(drop=True)
    print(f"  {len(world)} 个面, CRS={world.crs}")
    return world


def ensure_row_areas(world: gpd.GeoDataFrame, cache_parquet: str,
                     equal_area_crs: str = "EPSG:6933",
                     chunk: int = 30000) -> gpd.GeoDataFrame:
    """逐行 EPSG:6933 面积（km²）与 4326 包围盒，按 UID 缓存到 parquet。"""
    if os.path.exists(cache_parquet):
        cached = pd.read_parquet(cache_parquet)
        if len(cached) == len(world) and (cached["UID"].values == world["UID"].values).all():
            print(f"  面积缓存命中: {cache_parquet}")
            for c in ["area_km2", "minx", "miny", "maxx", "maxy"]:
                world[c] = cached[c].values
            return world
        print("  面积缓存与当前 World.shp 不一致，重算")

    print("  计算逐行等积面积（分块投影 EPSG:6933）...")
    areas = np.empty(len(world))
    for i in range(0, len(world), chunk):
        j = min(i + chunk, len(world))
        areas[i:j] = world.geometry.iloc[i:j].to_crs(equal_area_crs).area.values / 1e6
        print(f"    {j}/{len(world)}")
    world["area_km2"] = areas
    b = world.geometry.bounds
    world[["minx", "miny", "maxx", "maxy"]] = b[["minx", "miny", "maxx", "maxy"]].values
    world[["UID", "area_km2", "minx", "miny", "maxx", "maxy"]].to_parquet(cache_parquet)
    print(f"  已缓存: {cache_parquet}")
    return world


# ---------------------------------------------------------------------------
# 倒排索引
# ---------------------------------------------------------------------------
def build_name_index(world: gpd.GeoDataFrame) -> dict[str, list[tuple[int, int]]]:
    """norm_name → [(row_idx, level), ...]。level 0 表示 NAME_0（城市国家整国单元）。"""
    print("构建归一化倒排索引 ...")
    index: dict[str, list[tuple[int, int]]] = {}

    def add(raw, idx: int, level: int) -> None:
        if not isinstance(raw, str) or not raw.strip():
            return
        for part in raw.split("|"):
            for v in name_variants(part):
                index.setdefault(v, []).append((idx, level))

    cols_by_level: list[list[str]] = [["NAME_0"]]
    for lv in range(1, 6):
        cols = [f"NAME_{lv}"]
        if lv <= 4:
            cols.append(f"VARNAME_{lv}")
        if lv <= 3:
            cols.append(f"NL_NAME_{lv}")
        cols_by_level.append(cols)

    for lv, cols in enumerate(cols_by_level):
        for col in cols:
            ser = world[col]
            for idx, raw in ser.items():
                add(raw, idx, lv)
    print(f"  索引词条 {len(index)}")
    return index


# ---------------------------------------------------------------------------
# 空间工具
# ---------------------------------------------------------------------------
def geodesic_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist / 1000.0


def point_to_geom_km(pt: Point, geom) -> float:
    """点到面边界的测地线距离（km）；点在面内返回 0。"""
    if geom.contains(pt):
        return 0.0
    try:
        p_on_geom, _ = nearest_points(geom, pt)
    except Exception:  # noqa: BLE001 退化几何
        p_on_geom = geom.representative_point()
    return geodesic_km(pt.x, pt.y, p_on_geom.x, p_on_geom.y)


# ---------------------------------------------------------------------------
# 匹配结果结构
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    method: str = ""            # name+spatial / spatial_within / spatial_nearest / unmatched
    hit_level: int = -1         # 名称命中层级
    final_gid: str = ""
    final_level: int = -1
    final_name: str = ""
    area_km2: float = np.nan
    dist_km: float = np.nan     # 港口点到命中单元的距离（0=在面内）
    confidence: str = "low"     # high / medium / low
    needs_l3: bool = False
    note: str = ""
    chain: list = field(default_factory=list)  # [(level, gid, name, area_km2, score)] L3 仲裁候选


class L2Matcher:
    def __init__(self, world: gpd.GeoDataFrame, cfg: dict):
        p2c = cfg["port2city"]
        self.world = world
        self.tol_km = float(p2c["spatial_tolerance_km"])
        self.target = float(p2c["area_target_km2"])
        self.amin = float(p2c["area_min_km2"])
        self.amax = float(p2c["area_max_km2"])
        self.index = build_name_index(world)
        print("构建空间索引 ...")
        self.sindex = world.sindex
        # 国家名 → GID_0 / 行集合
        self.name0_to_gid0 = dict(
            world[["NAME_0", "GID_0"]].drop_duplicates().itertuples(index=False, name=None))
        self.gid0_rows: dict[str, np.ndarray] = {
            g: idx.values for g, idx in world.groupby("GID_0").groups.items()}
        # GID_n → 行集合 的懒缓存
        self._gid_rows_cache: dict[tuple[int, str], np.ndarray] = {}
        self._gid_area_cache: dict[tuple[int, str], float] = {}

    # -- 基础访问 --------------------------------------------------------
    def rows_of_gid(self, level: int, gid: str) -> np.ndarray:
        key = (level, gid)
        if key not in self._gid_rows_cache:
            col = f"GID_{level}" if level > 0 else "GID_0"
            self._gid_rows_cache[key] = np.where(self.world[col].values == gid)[0]
        return self._gid_rows_cache[key]

    def area_of_gid(self, level: int, gid: str) -> float:
        key = (level, gid)
        if key not in self._gid_area_cache:
            rows = self.rows_of_gid(level, gid)
            self._gid_area_cache[key] = float(self.world["area_km2"].values[rows].sum())
        return self._gid_area_cache[key]

    def score(self, area: float) -> float:
        if area <= 0:
            return np.inf
        return abs(np.log10(area / self.target))

    # -- 名称候选 --------------------------------------------------------
    def name_candidates(self, names: list[str], gid0: str | None) -> dict[tuple[int, str], np.ndarray]:
        """查索引 → 按国家过滤 → 按 (level, GID) 分组返回行集合。"""
        hits: list[tuple[int, int]] = []
        for nm in names:
            for v in name_variants(nm):
                hits.extend(self.index.get(v, ()))
        if not hits:
            return {}
        cand: dict[tuple[int, str], set] = {}
        gid0_set = set(self.gid0_rows.get(gid0, ())) if gid0 else None
        for idx, lv in hits:
            if gid0_set is not None and idx not in gid0_set:
                continue
            col = f"GID_{lv}" if lv > 0 else "GID_0"
            gid = self.world[col].values[idx]
            if not isinstance(gid, str) or not gid:
                continue
            cand.setdefault((lv, gid), set()).add(idx)
        # 行集合扩展为该 GID 全部成员行（命中行可能只是单元的一个子面）
        return {k: self.rows_of_gid(k[0], k[1]) for k in cand}

    # -- 空间验证 --------------------------------------------------------
    def verify(self, pt: Point, rows: np.ndarray) -> tuple[bool, float]:
        """(通过?, 距离km)。包含→0；否则成员面最小测地线距离 ≤ tol。

        用包围盒距离下界预筛 + 早停，避免对大单元的全部成员面做精确计算。
        """
        w = self.world
        minx = w["minx"].values[rows]; miny = w["miny"].values[rows]
        maxx = w["maxx"].values[rows]; maxy = w["maxy"].values[rows]
        # 包围盒距离下界（度→km 粗换算，纬度 111km、经度按 cos(lat) 缩放）
        dx = np.maximum(np.maximum(minx - pt.x, pt.x - maxx), 0.0) * 111.0 * max(
            np.cos(np.radians(pt.y)), 0.05)
        dy = np.maximum(np.maximum(miny - pt.y, pt.y - maxy), 0.0) * 111.0
        lb = np.hypot(dx, dy)
        order = np.argsort(lb)

        # 包含检查：仅对包围盒含点的行
        for k in order:
            if lb[k] > 0:
                break
            g = w.geometry.values[rows[k]]
            if g is not None and g.contains(pt):
                return True, 0.0

        d_best = np.inf
        for k in order:
            if lb[k] * 0.8 > min(d_best, self.tol_km * 3):  # 下界粗糙，留 20% 余量
                break
            g = w.geometry.values[rows[k]]
            if g is None:
                continue
            d = point_to_geom_km(pt, g)
            if d < d_best:
                d_best = d
            if d_best == 0.0:
                break
        return bool(d_best <= self.tol_km), float(d_best)

    # -- 层级链与面积带宽 -------------------------------------------------
    def build_chain(self, level: int, gid: str, pt: Point) -> list[tuple[int, str, str, float, float]]:
        """沿层级链（祖先↑ + 含港口点的后代↓）枚举 (level, gid, name, area, score)。"""
        rows = self.rows_of_gid(level, gid)
        first = rows[0]
        chain: list[tuple[int, str, str, float, float]] = []

        # 祖先（含自身、含整国 level 0）
        for lv in range(0, level + 1):
            col_g = f"GID_{lv}" if lv > 0 else "GID_0"
            col_n = f"NAME_{lv}" if lv > 0 else "NAME_0"
            g = self.world[col_g].values[first]
            if not isinstance(g, str) or not g:
                continue
            a = self.area_of_gid(lv, g)
            chain.append((lv, g, str(self.world[col_n].values[first]), a, self.score(a)))

        # 后代：成员行中包含港口点的行所属的更细 GID；
        # 港口点在海上（无包含行）时退而取最近成员行（单元已通过 25km 空间验证）
        contain_row = -1
        for r in rows:
            geom = self.world.geometry.values[r]
            if geom is not None and geom.contains(pt):
                contain_row = r
                break
        if contain_row < 0:
            contain_row = self._nearest_member_row(pt, rows)
        if contain_row >= 0:
            for lv in range(level + 1, 6):
                g = self.world[f"GID_{lv}"].values[contain_row]
                if not isinstance(g, str) or not g:
                    break
                a = self.area_of_gid(lv, g)
                chain.append((lv, g, str(self.world[f"NAME_{lv}"].values[contain_row]),
                              a, self.score(a)))
        return chain

    def _nearest_member_row(self, pt, rows: np.ndarray) -> int:
        """成员行中距港口点最近的行（包围盒下界预筛 + 早停）。"""
        w = self.world
        minx = w["minx"].values[rows]; miny = w["miny"].values[rows]
        maxx = w["maxx"].values[rows]; maxy = w["maxy"].values[rows]
        dx = np.maximum(np.maximum(minx - pt.x, pt.x - maxx), 0.0) * 111.0 * max(
            np.cos(np.radians(pt.y)), 0.05)
        dy = np.maximum(np.maximum(miny - pt.y, pt.y - maxy), 0.0) * 111.0
        lb = np.hypot(dx, dy)
        order = np.argsort(lb)
        best, d_best = -1, np.inf
        for k in order:
            if lb[k] * 0.8 > d_best:
                break
            g = w.geometry.values[rows[k]]
            if g is None:
                continue
            d = point_to_geom_km(pt, g)
            if d < d_best:
                best, d_best = int(rows[k]), d
            if d_best == 0.0:
                break
        return best

    def calibrate(self, chain: list[tuple[int, str, str, float, float]],
                  anchor_level: int) -> tuple[int, str, str, float, bool]:
        """面积带宽校准：带内取 score 最小层；带内无解取全链 score 最小层并标记。

        anchor_level: 名称命中层级——校准在不偏离锚定层过远的层级链上选择。
        返回 (level, gid, name, area, in_band)。
        """
        in_band = [c for c in chain if self.amin <= c[3] <= self.amax]
        pool = in_band if in_band else chain
        lv, gid, name, area, _ = min(pool, key=lambda c: (c[4], abs(c[0] - anchor_level)))
        return lv, gid, name, area, bool(in_band)

    # -- 纯空间兜底 -------------------------------------------------------
    def spatial_fallback(self, pt: Point) -> tuple[int, float]:
        """返回 (row_idx, dist_km)；within 优先，失败取 nearest。row_idx=-1 表示彻底失败。"""
        cands = list(self.sindex.query(pt, predicate="within"))
        if cands:
            return int(cands[0]), 0.0
        # nearest：扩张包围盒搜索
        for deg in (0.5, 2.0, 8.0):
            box = (pt.x - deg, pt.y - deg, pt.x + deg, pt.y + deg)
            near = list(self.sindex.intersection(box))
            if near:
                best, d_best = -1, np.inf
                for r in near:
                    g = self.world.geometry.values[r]
                    if g is None:
                        continue
                    d = point_to_geom_km(pt, g)
                    if d < d_best:
                        best, d_best = r, d
                return best, float(d_best)
        return -1, np.inf

    # -- 主匹配入口 -------------------------------------------------------
    def match(self, port_name: str, lon: float, lat: float,
              l1_city: str, l1_alts: list[str], l1_country: str) -> MatchResult:
        pt = Point(lon, lat)
        gid0 = self.name0_to_gid0.get(l1_country)
        res = MatchResult()

        # ---- 名称路径 ----
        cands = self.name_candidates([l1_city] + list(l1_alts), gid0)
        verified: list[tuple[float, int, str]] = []   # (dist, level, gid)
        for (lv, gid), rows in cands.items():
            ok, dist = self.verify(pt, rows)
            if ok:
                verified.append((dist, lv, gid))
        if verified:
            # 同距离时优先更细层级（粗层级作为锚定会丢失带内候选）
            verified.sort(key=lambda t: (round(t[0], 3), -t[1]))
            dist, lv, gid = verified[0]
            chain = self.build_chain(lv, gid, pt)
            f_lv, f_gid, f_name, f_area, in_band = self.calibrate(chain, lv)
            res.method = "name+spatial"
            res.hit_level = lv
            res.final_gid, res.final_level, res.final_name = f_gid, f_lv, f_name
            res.area_km2, res.dist_km = f_area, dist
            res.chain = chain
            if in_band:
                res.confidence = "high"
            else:
                res.confidence = "medium"
                res.needs_l3 = True
                res.note = "area_out_of_band"
            return res

        # ---- 纯空间兜底 ----
        row, dist = self.spatial_fallback(pt)
        if row < 0:
            res.method = "unmatched"
            res.needs_l3 = True
            res.note = "no_name_hit; spatial_fallback_failed"
            return res
        # 该行最细有效层级
        finest = 0
        for lv in range(5, 0, -1):
            g = self.world[f"GID_{lv}"].values[row]
            if isinstance(g, str) and g:
                finest = lv
                break
        gid = (self.world[f"GID_{finest}"] if finest > 0 else self.world["GID_0"]).values[row]
        chain = self.build_chain(finest, gid, pt)
        if not chain:  # 点不在面内（nearest 兜底）时 build_chain 仍给祖先链
            chain = [(finest, gid, str(self.world[f"NAME_{finest}"].values[row]),
                      self.area_of_gid(finest, gid), self.score(self.area_of_gid(finest, gid)))]
        f_lv, f_gid, f_name, f_area, in_band = self.calibrate(chain, finest)
        res.method = "spatial_within" if dist == 0.0 else "spatial_nearest"
        res.hit_level = -1
        res.final_gid, res.final_level, res.final_name = f_gid, f_lv, f_name
        res.area_km2, res.dist_km = f_area, dist
        res.chain = chain
        res.confidence = "medium" if (dist == 0.0 and in_band) else "low"
        res.needs_l3 = True
        res.note = "no_name_hit"
        if not in_band:
            res.note += "; area_out_of_band"
        if dist > 0:
            res.note += f"; nearest_{dist:.1f}km"
        # 国家一致性检查
        if gid0 and self.world["GID_0"].values[self.rows_of_gid(f_lv, f_gid)[0]] != gid0:
            res.note += "; country_mismatch"
        return res
