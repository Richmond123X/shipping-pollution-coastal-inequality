# -*- coding: utf-8 -*-
"""按质量闸门顺序执行 V12 政策流水线（年度可恢复分片版本）。

不构建或读取永久轨迹缓存，不做时间抽样和批量预取。每个分片调用既有排放
主程序，以14个月窗口逐船完成全年计算；已完成分片可恢复。AIS分片全部尝试
后再进入合并与政策情景质量门。
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from policy_run_config import (
    AIS_PREFETCH_BATCH_SIZE, AIS_SAMPLE_INTERVAL_SEC, AIS_SHARD_SIZE,
    ROOT, SIMULATION_YEAR,
)


LOGS = ROOT / "logs"
STATUS = ROOT / "formal_pipeline_status.json"


def update(stage: str, status: str, **extra) -> None:
    rec = {
        "stage": stage,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    STATUS.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def run_step(name: str, script: str, *args: str) -> None:
    update(name, "running")
    log = LOGS / f"07_{name}.log"
    cmd = [sys.executable, "-u", str(ROOT / script), *args]
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"{name} 失败(returncode={proc.returncode})\n{tail}")
    update(name, "complete", log=str(log))


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        run_step("clustering_supplement_tables", "00c_build_clustering_supplement_tables.py")
        run_step(
            "ais_direct_annual_shards", "03_run_policy_ais_shards.py",
            "--n-cores", "4",
            "--shard-size", str(AIS_SHARD_SIZE),
            "--prefetch-batch-size", str(AIS_PREFETCH_BATCH_SIZE),
            "--sample-interval-sec", str(AIS_SAMPLE_INTERVAL_SEC),
        )
        run_step("merge", "04_merge_policy_ais_shards.py")
        run_step("scenarios", "05_compute_policy_scenarios.py")
        run_step("table7", "05a_build_table7_diagnostics.py")
        run_step("redesigned_figures", "06b_make_policy_redesign_panels.py")
        run_step("overlap_figures", "06a_make_overlap_figures.py")
        run_step("input_validation", "00_validate_policy_inputs.py")
        run_step("formal_verification", "08_verify_formal_outputs.py")
        update(
            "pipeline", "complete_awaiting_visual_qa",
            simulation_year=SIMULATION_YEAR,
        )
    except Exception as exc:
        update(
            "pipeline", "failed", error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
