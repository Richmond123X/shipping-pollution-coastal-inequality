# -*- coding: utf-8 -*-
"""排放网络配置加载：所有路径与参数来自项目根 config.yaml（命令行可覆盖）。

跨机器路径自适应：config.yaml 里写死的绝对路径在别的电脑上（用户名不同,
如 86176 vs HP;或项目/数据换了盘符）会自动重定位,无需改配置文件。规则按序:
  1. 展开 ~ 与环境变量;路径存在 → 原样使用;
  2. 前缀为 <盘符>:/<项目文件夹名>/ → 重定位到当前项目根
     (输出目录尚不存在也适用,如 results_monthly);
  3. <盘符>:/Users/<别的用户名>/... → 尝试当前用户目录下同结构,存在才用;
  4. 整条路径换盘符扫描(如 D:/SQL_AIS_SingleShip → E:/SQL_AIS_SingleShip);
  5. 都不中 → 保留原路径,由调用方自然报"文件不存在"。
发生迁移时在主进程打印一次映射清单,便于核对。
"""
from __future__ import annotations

import multiprocessing
import os
import re
import string

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CONFIG_PATH = os.environ.get("EMISSION_CONFIG", os.path.join(_ROOT, "config.yaml"))

_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def _existing_drives() -> list[str]:
    return [f"{d}:" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _resolve_path(p: str, root: str, migrations: list) -> str:
    q = os.path.expandvars(os.path.expanduser(p))
    if not _WIN_ABS.match(q) or os.path.exists(q):
        return q
    parts = [s for s in re.split(r"[\\/]+", q) if s]
    # 2) 项目根重定位: <盘符>:/<项目文件夹名>/rest → <当前项目根>/rest
    #    无条件生效(结果输出目录首跑前不存在,也要指到当前项目根)
    proj = os.path.basename(os.path.normpath(root))
    if len(parts) >= 2 and parts[1].casefold() == proj.casefold():
        cand = os.path.join(root, *parts[2:])
        if not _same_path(cand, q):
            migrations.append((p, cand))
        return cand
    # 3) 用户目录重定位: X:/Users/<名>/rest → <当前用户目录>/rest(存在才用)
    if len(parts) >= 3 and parts[1].casefold() == "users":
        cand = os.path.join(os.path.expanduser("~"), *parts[3:])
        if os.path.exists(cand):
            migrations.append((p, cand))
            return cand
    # 4) 换盘符: 同结构路径在其他盘(存在才用)
    for drv in _existing_drives():
        if drv.casefold() == parts[0].casefold():
            continue
        cand = os.path.join(drv + os.sep, *parts[1:])
        if os.path.exists(cand):
            migrations.append((p, cand))
            return cand
    return q


def _walk(obj, root: str, migrations: list):
    if isinstance(obj, dict):
        return {k: _walk(v, root, migrations) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, root, migrations) for v in obj]
    if isinstance(obj, str) and _WIN_ABS.match(obj.strip()):
        return _resolve_path(obj.strip(), root, migrations)
    return obj


def load_config(path: str | None = None) -> dict:
    p = path or CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.path.dirname(os.path.abspath(p))
    migrations: list = []
    cfg = _walk(cfg, root, migrations)
    # Pool 子进程 import 时会重复加载配置,迁移清单只在主进程打印一次
    if migrations and multiprocessing.current_process().name == "MainProcess":
        print("[config] 跨机器路径自动迁移(config.yaml 未改动,仅运行时映射):")
        for a, b in migrations:
            print(f"  {a}  ->  {b}")
    cfg["_config_path"] = p
    return cfg


def get_paths(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return cfg["paths"]


def get_emission_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return cfg["emission_network"]
