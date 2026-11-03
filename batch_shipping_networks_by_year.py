#!/usr/bin/env python3
"""
按自然年循环调用 shipping_network_main.py，处理 ais_<YYYYMM>.db（每月一库）。

默认：每年 12 个月 × 每种船型各跑一次，输出
  <results_root>/<YYYYMM>/<船型文件夹>/
换年：改 --year 即可，例如 --year 2025。

优化特性:
  - 批处理层并行：同时运行多个 month×ship_type 组合（--batch-parallel）
  - 断点续传：自动跳过已有输出文件的任务（--skip-done，默认开启）
  - 子进程内 workers 数可调（--workers，默认 8）
  - 实时输出：每个任务的进度独占一行，通过 ANSI 转义码原地更新

示例:
  python batch_shipping_networks_by_year.py --year 2019 --results-root "C:\\results"
  python batch_shipping_networks_by_year.py --year 2019 --batch-parallel 3 --workers 6
  python batch_shipping_networks_by_year.py --year 2019 --no-skip-done   # 强制重跑已完成任务
  python batch_shipping_networks_by_year.py --year 2024 --combined-monthly

排放网络（季度、全船型，调用 emission_network_main.py，默认输出 results_new/）:
  python batch_shipping_networks_by_year.py --emission-network --years 2019,2020,2021
  python batch_shipping_networks_by_year.py --emission-network --year 2024 --only-quarters 1,2
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =========================================================================
# Windows 10+ 启用 ANSI 转义码
# =========================================================================
def _enable_ansi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


# =========================================================================
# 多任务进度显示管理器
# =========================================================================
class ProgressDisplay:
    """
    管理多个并行任务的实时进度显示。
    - 每个任务的 \\r 进度独占一行，通过 ANSI 光标移动原地更新
    - 正常 \\n 输出打印在进度区域上方，不被覆盖
    - 线程安全
    """

    CSI_CLEAR_LINE = "\033[K"   # 清除光标到行尾

    def __init__(self):
        self._lock = threading.Lock()
        self._slots: dict[str, str] = {}   # tag → 最新进度文本
        self._order: list[str] = []        # 插入顺序
        self._n_shown: int = 0             # 当前屏幕上显示的进度行数

    # ---------- 公开接口 ----------

    def update_progress(self, tag: str, text: str) -> None:
        """更新某任务的 \\r 进度行（原地覆盖）。"""
        with self._lock:
            if tag not in self._slots:
                self._order.append(tag)
            self._slots[tag] = text
            self._redraw()

    def print_line(self, tag: str, text: str) -> None:
        """打印一条正常输出行（在进度区域上方插入）。"""
        with self._lock:
            self._cursor_to_top()
            sys.stdout.write(f"[{tag}] {text}{self.CSI_CLEAR_LINE}\n")
            self._redraw_slots()
            sys.stdout.flush()

    def print_untagged(self, text: str) -> None:
        """打印一条无标签的信息行（如任务完成汇总）。"""
        with self._lock:
            self._cursor_to_top()
            sys.stdout.write(f"{text}{self.CSI_CLEAR_LINE}\n")
            self._redraw_slots()
            sys.stdout.flush()

    def remove_task(self, tag: str) -> None:
        """任务结束后移除其进度行。"""
        with self._lock:
            if tag not in self._slots:
                return
            self._cursor_to_top()
            del self._slots[tag]
            self._order.remove(tag)
            self._redraw_slots()
            # 清除因行数减少而多出来的空行
            extra = self._n_shown - len(self._order)
            for _ in range(extra):
                sys.stdout.write(f"{self.CSI_CLEAR_LINE}\n")
            # 光标回到进度区域末尾
            if extra > 0:
                sys.stdout.write(f"\033[{extra}A")
            self._n_shown = len(self._order)
            sys.stdout.flush()

    # ---------- 内部方法 ----------

    def _cursor_to_top(self) -> None:
        """将光标移到进度区域首行。"""
        if self._n_shown > 0:
            sys.stdout.write(f"\033[{self._n_shown}A\r")

    def _redraw(self) -> None:
        self._cursor_to_top()
        self._redraw_slots()
        sys.stdout.flush()

    def _redraw_slots(self) -> None:
        for t in self._order:
            text = self._slots.get(t, "")
            sys.stdout.write(f"[{t}] {text}{self.CSI_CLEAR_LINE}\n")
        self._n_shown = len(self._order)


# 全局单例
_display = ProgressDisplay()


# =========================================================================
# 工具函数
# =========================================================================
def safe_folder_name(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "_", str(name).strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "unknown_ship_type"


def year_month_codes(year: int) -> list[str]:
    return [f"{year}{m:02d}" for m in range(1, 13)]


def year_quarter_codes(year: int) -> list[tuple[str, list[str]]]:
    """返回 [(label, [YYYYMM, ...]), ...]，如 ("2024Q1", ["202401","202402","202403"])."""
    out = []
    for q in range(1, 5):
        m_start, m_end = (q - 1) * 3 + 1, q * 3
        out.append((f"{year}Q{q}", [f"{year}{m:02d}" for m in range(m_start, m_end + 1)]))
    return out


def list_ship_types(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path, timeout=120)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ship_info'")
        if not cur.fetchone():
            raise SystemExit(f"库中无 ship_info 表: {db_path}")
        q = """
        SELECT DISTINCT Ship_Type AS st FROM ship_info
        WHERE Ship_Type IS NOT NULL AND TRIM(CAST(Ship_Type AS TEXT)) != ''
        ORDER BY st
        """
        rows = cur.execute(q).fetchall()
    finally:
        conn.close()
    return [str(r[0]) for r in rows]


def first_existing_monthly_db(db_dir: str, year: int, prefer_month: int = 1) -> str | None:
    """用于读取船型列表：优先 year 年 prefer_month 月，否则按 1..12 找第一个存在的库。"""
    order = [prefer_month] + [m for m in range(1, 13) if m != prefer_month]
    for mo in order:
        ym = f"{year}{mo:02d}"
        p = os.path.join(db_dir, f"ais_{ym}.db")
        if os.path.isfile(p):
            return p
    return None


# -----------------------------------------------------------------------
# 断点续传
# -----------------------------------------------------------------------
def _task_already_done(out_dir: Path, label: str) -> bool:
    """如果输出目录中已存在节点特征 CSV 和边 CSV，则视为已完成。"""
    nodes_csv = out_dir / f"port_node_features_{label}.csv"
    edges_csv = out_dir / f"network_edge_DWT_{label}.csv"
    if nodes_csv.is_file() and edges_csv.is_file():
        if nodes_csv.stat().st_size > 100 and edges_csv.stat().st_size > 100:
            return True
    return False


# -----------------------------------------------------------------------
# 子进程输出读取 → 分发到 ProgressDisplay
# -----------------------------------------------------------------------
def _stream_to_display(pipe, tag: str, log_file) -> None:
    """
    逐块读取子进程管道，区分 \\n 正常输出和 \\r 进度刷新：
    - \\n 行 → _display.print_line（在进度区域上方换行打印）
    - \\r 段 → _display.update_progress（原地覆盖对应任务行）
    """
    fd = pipe.fileno()
    buf = b""

    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk

        # 1) 处理完整的 \n 行
        while b"\n" in buf:
            segment, buf = buf.split(b"\n", 1)
            # segment 内若有 \r，只保留最后一段（模拟终端覆盖）
            if b"\r" in segment:
                segment = segment.rsplit(b"\r", 1)[-1]
            line = segment.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            _display.print_line(tag, line)
            if log_file:
                log_file.write(line + "\n")
                log_file.flush()

        # 2) 剩余 buf 无 \n，但可能有 \r 进度 → 原地更新该任务的进度行
        if b"\r" in buf:
            latest = buf.rsplit(b"\r", 1)[-1]
            line = latest.decode("utf-8", errors="replace").rstrip()
            if line:
                _display.update_progress(tag, line)
            buf = latest

    # 3) 残余
    if buf.strip():
        line = buf.decode("utf-8", errors="replace").rstrip()
        if line:
            _display.print_line(tag, line)
            if log_file:
                log_file.write(line + "\n")
                log_file.flush()

    # 任务输出结束，移除进度行
    _display.remove_task(tag)
    pipe.close()


# -----------------------------------------------------------------------
# 子任务运行函数
# -----------------------------------------------------------------------
def _run_one_task(task: dict) -> dict:
    """执行一个 month×ship_type 任务，实时转发输出到 ProgressDisplay。"""
    ym = task["ym"]
    ship_type = task["ship_type"]
    out_dir = task["out_dir"]
    cmd = task["cmd"]
    cwd = task["cwd"]
    env = task["env"]
    dry_run = task["dry_run"]

    tag = f"{ym}/{ship_type}" if ship_type else ym

    if dry_run:
        return {"ym": ym, "ship_type": ship_type, "success": True,
                "dry_run": True, "cmd": " ".join(cmd)}

    t0 = time.time()
    log_path = Path(out_dir) / f"_batch_log_{ym}.txt"
    try:
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        reader = threading.Thread(
            target=_stream_to_display,
            args=(proc.stdout, tag, log_f),
            daemon=True,
        )
        reader.start()
        proc.wait()
        reader.join(timeout=10)
        log_f.close()

        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"ym": ym, "ship_type": ship_type, "success": False,
                    "elapsed": elapsed, "error": f"退出码 {proc.returncode}",
                    "log": str(log_path)}
        return {"ym": ym, "ship_type": ship_type, "success": True,
                "elapsed": elapsed, "log": str(log_path)}
    except Exception as e:
        return {"ym": ym, "ship_type": ship_type, "success": False,
                "elapsed": time.time() - t0, "error": str(e)}


# =========================================================================
# 主程序
# =========================================================================
def main() -> None:
    _enable_ansi()

    root = Path(__file__).resolve().parent
    main_py = root / "shipping_network_main.py"

    p = argparse.ArgumentParser(
        description="按年循环处理航运网络（12 个月，可换 --year）——支持批处理并行与断点续传"
    )
    p.add_argument("--year", type=int, default=None,
                   help="四位年份；与 --years 二选一（航运批处理仅支持单年）")
    p.add_argument("--years", type=str, default="",
                   help="多个年份逗号分隔，如 2019,2020,2021；需与 --emission-network 联用")
    p.add_argument("--db-dir", type=str, default=r"E:\SQL_AIS_SingleShip",
                    help="含 ais_YYYYMM.db 的目录")
    p.add_argument("--results-root", type=str, default="",
                    help="结果根目录；默认：<本脚本所在目录>/results")
    p.add_argument("--combined-monthly", action="store_true",
                    help="每月只调用一次主程序（全船型合并）；默认按船型分子目录分别跑")
    p.add_argument("--types-from-month", type=int, default=1, metavar="M",
                    help="从该年 M 月（1-12）的库读取船型列表；若该月库不存在则自动换其他月")
    p.add_argument("--only-months", type=str, default="",
                    help="只处理这些月，逗号分隔：1-12 或 YYYYMM，如 1,2,3 或 201901,201902")
    p.add_argument("--by-quarter", action="store_true",
                    help="按季度合并：每季度 3 个月的 AIS 合并到一次主程序调用，"
                         "输出到 <results_root>/YYYYQn/<船型>/。--only-months 在此模式下被忽略，"
                         "改用 --only-quarters")
    p.add_argument("--only-quarters", type=str, default="",
                   help="--by-quarter 或 --emission-network：如 Q1,Q3 或 1,3；留空 = 全部 4 个季度")
    p.add_argument("--port-xlsx", type=str,
                    default=r"C:\Users\HP\Desktop\Port Data\port_data_clean.xlsx")
    p.add_argument(
        "--laf-xlsx", type=str,
        default=r"C:\Users\HP\Desktop\Ship Emission\Ship_File\Low load adjustment factor.xlsx",
        help="仅 --emission-network",
    )
    p.add_argument(
        "--eca-base", type=str,
        default=r"C:\Users\HP\Desktop\Ship Emission",
        help="仅 --emission-network",
    )
    p.add_argument(
        "--emission-network", action="store_true",
        help="调用 emission_network_main.py：按季度、全船型计算排放网络",
    )
    p.add_argument("--workers", type=int, default=8,
                    help="shipping：子进程 workers；emission-network：--n-cores")
    p.add_argument("--batch-parallel", type=int, default=2, metavar="N",
                    help="同时运行的 month×ship_type 子任务数（默认 2）；"
                         "设 1 则退化为串行。建议 2~3，过多会争抢 I/O")
    p.add_argument("--no-skip-done", action="store_true",
                    help="禁用断点续传，强制重跑所有任务（默认自动跳过已完成的）")
    p.add_argument("--berth-radius-km", type=float, default=None,
                    help="港口区域构建时每个泊位的缓冲半径（km），默认 8.0；传给 shipping_network_main.py")
    p.add_argument("--berth-km", type=float, default=None)
    p.add_argument("--canal-km", type=float, default=None)
    p.add_argument("--berth-gap-tolerance-sec", type=float, default=None,
                    help="靠泊中断容忍时间（秒），传递给 shipping_network_main.py（默认 600）")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    years_list: list[int] = []
    if (args.years or "").strip():
        for x in args.years.replace(";", ",").split(","):
            x = x.strip()
            if not x:
                continue
            if not x.isdigit() or len(x) != 4:
                raise SystemExit(f"--years 中无效年份: {x!r}")
            yy = int(x)
            if yy < 1990 or yy > 2100:
                raise SystemExit(f"年份不合理: {yy}")
            years_list.append(yy)
        if not years_list:
            raise SystemExit("--years 解析后为空")
    elif args.year is not None:
        if args.year < 1990 or args.year > 2100:
            raise SystemExit(f"年份不合理: {args.year}")
        years_list = [args.year]
    else:
        raise SystemExit("请指定 --year 或 --years")

    if len(years_list) > 1 and not args.emission_network:
        raise SystemExit("多个年份请使用 --emission-network，或对航运批处理每次只指定一个 --year")

    db_dir = os.path.abspath(args.db_dir)

    if args.emission_network:
        emission_py = root / "emission_network_main.py"
        if not emission_py.is_file():
            raise SystemExit(f"未找到: {emission_py}")
        results_em = (
            Path(args.results_root.strip()).resolve()
            if args.results_root.strip()
            else (root / "results_new").resolve()
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print(f"模式: 排放网络  年: {years_list}")
        print(f"数据库: {db_dir}")
        print(f"输出: {results_em}")
        for y in years_list:
            cmd = [
                sys.executable, str(emission_py),
                "--year", str(y),
                "--db-dir", db_dir,
                "--results-root", str(results_em),
                "--port-xlsx", os.path.abspath(args.port_xlsx),
                "--laf-xlsx", os.path.abspath(args.laf_xlsx),
                "--eca-base", os.path.abspath(args.eca_base),
                "--n-cores", str(max(1, int(args.workers))),
            ]
            if (args.only_quarters or "").strip():
                cmd.extend(["--only-quarters", args.only_quarters.strip()])
            if args.no_skip_done:
                cmd.append("--no-skip-done")
            print(f"\n{'=' * 60}\n运行: {' '.join(cmd)}\n{'=' * 60}", flush=True)
            rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
            if rc != 0:
                raise SystemExit(f"emission_network_main.py --year {y} 退出码 {rc}")
        print(f"\n完成: {years_list}")
        return

    year = years_list[0]
    results_root = Path(args.results_root.strip() or str(root / "results"))
    months_all = year_month_codes(year)
    skip_done = not args.no_skip_done

    # ---------- 解析 --only-months / --only-quarters ----------
    if args.by_quarter:
        if args.only_months.strip():
            print("提示: --by-quarter 模式下 --only-months 被忽略，请用 --only-quarters")
        if args.only_quarters.strip():
            raw = [x.strip().upper().lstrip("Q")
                   for x in args.only_quarters.replace(";", ",").split(",") if x.strip()]
            wanted = set()
            for x in raw:
                if x.isdigit() and 1 <= int(x) <= 4:
                    wanted.add(int(x))
                else:
                    raise SystemExit(f"--only-quarters 项无法解析: {x!r}（用 1-4 或 Q1-Q4）")
        else:
            wanted = {1, 2, 3, 4}
        all_quarters = year_quarter_codes(year)
        quarters_sel: list[tuple[str, list[str]]] = [
            (lbl, ms) for lbl, ms in all_quarters if int(lbl[-1]) in wanted
        ]
        months = []  # 按季度模式下不直接按月遍历
    else:
        if args.only_months.strip():
            raw = [x.strip() for x in args.only_months.replace(";", ",").split(",") if x.strip()]
            sel: list[str] = []
            for x in raw:
                if len(x) == 6 and x.isdigit():
                    if not x.startswith(str(year)):
                        raise SystemExit(f"--only-months 中的 {x} 与 --year {year} 不一致")
                    sel.append(x)
                elif x.isdigit() and 1 <= int(x) <= 12:
                    sel.append(f"{year}{int(x):02d}")
                else:
                    raise SystemExit(f"无法解析月份项: {x!r}（用 1-12 或 YYYYMM）")
            months = sorted(set(sel))
        else:
            months = months_all
        quarters_sel = []

    # ---------- 船型列表 ----------
    ref_db = first_existing_monthly_db(db_dir, year,
                                        prefer_month=max(1, min(12, int(args.types_from_month))))
    split_by_ship_type = not args.combined_monthly
    ship_types: list[str] = []
    if split_by_ship_type:
        if not ref_db:
            raise SystemExit(f"在 {db_dir} 未找到 {year} 年任一 ais_{year}MM.db，无法读取船型")
        ship_types = list_ship_types(ref_db)
        print(f"船型列表来自: {os.path.basename(ref_db)} ，共 {len(ship_types)} 种")

    batch_parallel = max(1, args.batch_parallel)
    if args.by_quarter:
        print(f"年份: {year}  季度数: {len(quarters_sel)}  分船型: {split_by_ship_type}  "
              f"模式: 按季度（每季度 3 个月 AIS 合并到一次主程序调用）")
    else:
        print(f"年份: {year}  月份数: {len(months)}  分船型: {split_by_ship_type}")
    print(f"批处理并行度: {batch_parallel}  子任务 workers: {args.workers}")
    print(f"断点续传: {'开启' if skip_done else '关闭'}")
    print(f"数据库目录: {db_dir}")
    print(f"结果根目录: {results_root}")

    # 环境变量：子进程不缓冲
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # =====================================================================
    # 构建任务列表
    # =====================================================================
    def _build_cmd(ym: str, out_dir: str, label: str,
                   ship_type: str | None) -> list[str]:
        cmd: list[str] = [
            sys.executable,
            str(main_py),
            "--months", ym,
            "--db-dir", db_dir,
            "--output-dir", out_dir,
            "--label", label,
            "--port-xlsx", args.port_xlsx,
            "--workers", str(max(1, int(args.workers))),
        ]
        if ship_type:
            cmd.extend(["--ship-type", ship_type])
        if args.berth_radius_km is not None:
            cmd.extend(["--berth-radius-km", str(args.berth_radius_km)])
        if args.berth_km is not None:
            cmd.extend(["--berth-km", str(args.berth_km)])
        if args.canal_km is not None:
            cmd.extend(["--canal-km", str(args.canal_km)])
        if args.berth_gap_tolerance_sec is not None:
            cmd.extend(["--berth-gap-tolerance-sec", str(args.berth_gap_tolerance_sec)])
        return cmd

    tasks: list[dict] = []
    skipped = 0

    if args.by_quarter:
        # ---------- 按季度构建任务 ----------
        for lbl, ms in quarters_sel:
            existing = [m for m in ms if os.path.isfile(os.path.join(db_dir, f"ais_{m}.db"))]
            if not existing:
                print(f"警告: 跳过 {lbl}（季度内无任一 ais_YYYYMM.db）")
                continue
            if len(existing) < len(ms):
                missing = sorted(set(ms) - set(existing))
                print(f"  注意 {lbl}: 仅找到 {existing}，缺 {missing}（按存在的月计算）")
            months_csv = ",".join(existing)

            if not split_by_ship_type:
                qd = results_root / lbl
                if skip_done and _task_already_done(qd, lbl):
                    print(f"  [跳过-已完成] {lbl}")
                    skipped += 1
                    continue
                qd.mkdir(parents=True, exist_ok=True)
                tasks.append({
                    "ym": lbl, "ship_type": None,
                    "out_dir": str(qd),
                    "cmd": _build_cmd(months_csv, str(qd), lbl, None),
                    "cwd": str(root), "env": env, "dry_run": args.dry_run,
                })
            else:
                for st in ship_types:
                    sub = safe_folder_name(st)
                    od = results_root / lbl / sub
                    if skip_done and _task_already_done(od, lbl):
                        print(f"  [跳过-已完成] {lbl}/{st}")
                        skipped += 1
                        continue
                    od.mkdir(parents=True, exist_ok=True)
                    tasks.append({
                        "ym": lbl, "ship_type": st,
                        "out_dir": str(od),
                        "cmd": _build_cmd(months_csv, str(od), lbl, st),
                        "cwd": str(root), "env": env, "dry_run": args.dry_run,
                    })
    else:
        # ---------- 按月构建任务 ----------
        for ym in months:
            db_m = os.path.join(db_dir, f"ais_{ym}.db")
            if not os.path.isfile(db_m):
                print(f"警告: 跳过 {ym}（无 {db_m}）")
                continue

            if not split_by_ship_type:
                month_dir = results_root / ym
                if skip_done and _task_already_done(month_dir, ym):
                    print(f"  [跳过-已完成] {ym}")
                    skipped += 1
                    continue
                month_dir.mkdir(parents=True, exist_ok=True)
                tasks.append({
                    "ym": ym, "ship_type": None,
                    "out_dir": str(month_dir),
                    "cmd": _build_cmd(ym, str(month_dir), ym, None),
                    "cwd": str(root), "env": env, "dry_run": args.dry_run,
                })
            else:
                for st in ship_types:
                    sub = safe_folder_name(st)
                    od = results_root / ym / sub
                    if skip_done and _task_already_done(od, ym):
                        print(f"  [跳过-已完成] {ym}/{st}")
                        skipped += 1
                        continue
                    od.mkdir(parents=True, exist_ok=True)
                    tasks.append({
                        "ym": ym, "ship_type": st,
                        "out_dir": str(od),
                        "cmd": _build_cmd(ym, str(od), ym, st),
                        "cwd": str(root), "env": env, "dry_run": args.dry_run,
                    })

    total = len(tasks) + skipped
    print(f"\n任务总计: {total}（待运行: {len(tasks)}，已跳过: {skipped}）")

    if not tasks:
        print("无待运行任务，退出。")
        return

    if args.dry_run:
        print("\n[dry-run] 任务列表:")
        for t in tasks:
            tag = f"{t['ym']}/{t['ship_type']}" if t["ship_type"] else t["ym"]
            print(f"  {tag}: {' '.join(t['cmd'])}")
        return

    # =====================================================================
    # 执行
    # =====================================================================
    t_start = time.time()
    done_ok = 0
    done_fail = 0
    failed_tasks: list[dict] = []

    if batch_parallel == 1:
        # ---------- 串行模式 ----------
        for i, task in enumerate(tasks):
            tag = f"{task['ym']}/{task['ship_type']}" if task["ship_type"] else task["ym"]
            _display.print_untagged(f"\n========== [{i+1}/{len(tasks)}] {tag} ==========")
            result = _run_one_task(task)
            if result["success"]:
                done_ok += 1
                elapsed_m = result.get("elapsed", 0) / 60
                _display.print_untagged(f"  ✓ 完成，耗时 {elapsed_m:.1f} 分钟")
            else:
                done_fail += 1
                failed_tasks.append(result)
                _display.print_untagged(f"  ✗ 失败: {result.get('error', '未知')}")
                _display.print_untagged(f"    日志: {result.get('log', 'N/A')}")
    else:
        # ---------- 并行模式 ----------
        _display.print_untagged(
            f"启动并行执行，最多 {batch_parallel} 个子任务同时运行...\n")
        with ThreadPoolExecutor(max_workers=batch_parallel) as executor:
            future_to_task = {}
            for task in tasks:
                fut = executor.submit(_run_one_task, task)
                future_to_task[fut] = task

            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                tag = (f"{task['ym']}/{task['ship_type']}"
                       if task["ship_type"] else task["ym"])
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"ym": task["ym"], "ship_type": task["ship_type"],
                              "success": False, "error": str(e)}

                if result["success"]:
                    done_ok += 1
                    elapsed_m = result.get("elapsed", 0) / 60
                    total_elapsed_m = (time.time() - t_start) / 60
                    _display.print_untagged(
                        f"  ✓ [{done_ok + done_fail}/{len(tasks)}] {tag} "
                        f"完成 ({elapsed_m:.1f}min)  总耗时 {total_elapsed_m:.1f}min")
                else:
                    done_fail += 1
                    failed_tasks.append(result)
                    _display.print_untagged(
                        f"  ✗ [{done_ok + done_fail}/{len(tasks)}] {tag} "
                        f"失败: {result.get('error', '未知')}")

    # =====================================================================
    # 汇总
    # =====================================================================
    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"全部完成。成功: {done_ok}，失败: {done_fail}，跳过: {skipped}，"
          f"总耗时: {total_min:.1f} 分钟")

    if failed_tasks:
        print(f"\n失败任务列表:")
        for r in failed_tasks:
            tag = f"{r['ym']}/{r['ship_type']}" if r.get("ship_type") else r["ym"]
            print(f"  {tag}: {r.get('error', '未知')}  日志: {r.get('log', 'N/A')}")


if __name__ == "__main__":
    main()
