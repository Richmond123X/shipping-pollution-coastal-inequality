# -*- coding: utf-8 -*-
"""Port2City L1 批次管理：生成待判定批次 / 合并判定结果到缓存。

L1 常识层由 Claude Code 会话内分批判定（不调 API）：
  python l1_batches.py make    # 依据缓存差集生成 l1_pending/batch_NNN.json
  python l1_batches.py merge   # 把 l1_results/batch_NNN_result.json 合入 llm_city_cache.json
  python l1_batches.py status  # 进度统计

缓存键 = "Port_Name|Country_Region"。缓存值结构：
  {"city": 英文城市名(面积与中国地级市相当的行政单元),
   "alt_names": [别名/本地名/旧称...],
   "country": GADM NAME_0 口径国家名,
   "source": "L1" | "L3",
   "batch": 批次号}
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PENDING_DIR = os.path.join(HERE, "l1_pending")
RESULTS_DIR = os.path.join(HERE, "l1_results")


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_path(cfg: dict) -> str:
    return cfg["paths"]["llm_city_cache"]


def load_cache(cfg: dict) -> dict:
    p = cache_path(cfg)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cfg: dict, cache: dict) -> None:
    p = cache_path(cfg)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def port_key(port_name: str, country_region: str) -> str:
    return f"{port_name}|{country_region}"


def make_batches() -> None:
    cfg = load_config()
    batch_size = int(cfg["port2city"]["l1_batch_size"])
    ports = pd.read_csv(cfg["paths"]["coastal_ports_csv"])
    cache = load_cache(cfg)

    pending = []
    for _, r in ports.iterrows():
        k = port_key(r["Port_Name"], r["Country_Region"])
        if k not in cache:
            pending.append({
                "port_name": r["Port_Name"],
                "country_region": r["Country_Region"],
                "lon": round(float(r["Lon_port"]), 4),
                "lat": round(float(r["Lat_port"]), 4),
            })

    os.makedirs(PENDING_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(PENDING_DIR, "batch_*.json")):
        os.remove(f)

    for i in range(0, len(pending), batch_size):
        bid = i // batch_size + 1
        with open(os.path.join(PENDING_DIR, f"batch_{bid:03d}.json"), "w", encoding="utf-8") as f:
            json.dump({"batch_id": bid, "ports": pending[i:i + batch_size]},
                      f, ensure_ascii=False, indent=1)
    n_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0
    print(f"缓存已有 {len(cache)} 港；待判定 {len(pending)} 港 → {n_batches} 个批次写入 {PENDING_DIR}")


def merge_results() -> None:
    cfg = load_config()
    cache = load_cache(cfg)
    gadm = set(pd.read_csv(os.path.join(HERE, "gadm_countries.csv"))["NAME_0"])
    with open(os.path.join(HERE, "country_name_map.json"), encoding="utf-8") as f:
        cmap = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    n_new, n_bad = 0, 0
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "batch_*_result.json")))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            res = json.load(f)
        bid = res.get("batch_id")
        for item in res.get("results", []):
            try:
                k = port_key(item["port_name"], item["country_region"])
                # 国家名以确定性映射为默认；agent 显式给出且合法时允许覆盖
                cr = str(item["country_region"])
                country = cmap.get(cr, cr)
                agent_country = str(item.get("country") or "").strip()
                if agent_country and agent_country in gadm:
                    country = agent_country
                entry = {
                    "city": str(item["city"]).strip(),
                    "alt_names": [str(a).strip() for a in item.get("alt_names", []) if str(a).strip()],
                    "country": country,
                    "source": "L1",
                    "batch": bid,
                }
                if not entry["city"]:
                    raise ValueError("empty city")
                if entry["country"] not in gadm:
                    raise ValueError(f"country not in GADM: {entry['country']}")
                if k not in cache:
                    n_new += 1
                cache[k] = entry
            except Exception as e:  # noqa: BLE001
                n_bad += 1
                print(f"  跳过无效条目 ({fp}): {item} ({e})")
    save_cache(cfg, cache)
    print(f"合并 {len(files)} 个结果文件: 新增 {n_new} 条, 无效 {n_bad} 条, 缓存总计 {len(cache)} 条")


def status() -> None:
    cfg = load_config()
    ports = pd.read_csv(cfg["paths"]["coastal_ports_csv"])
    cache = load_cache(cfg)
    keys = {port_key(r["Port_Name"], r["Country_Region"]) for _, r in ports.iterrows()}
    done = len(keys & set(cache))
    print(f"进度: {done}/{len(keys)} ({done / max(len(keys), 1):.1%})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"make": make_batches, "merge": merge_results, "status": status}[cmd]()
