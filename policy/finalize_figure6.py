# -*- coding: utf-8 -*-
"""Strictly evaluate 2025 policy outputs and finish all five Figure 6 panels.

This command intentionally stops before any calculation when the 12-month ×
ship-type inventory is incomplete. It never creates placeholder or partial-year
policy figures and does not build the final PowerPoint; the deck is assembled only
after all independent PNG panels have passed visual QA.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policy"
DEFAULT_RESULTS = ROOT / "results_monthly"
DEFAULT_EVAL = POLICY / "outputs" / "evaluation"
DEFAULT_OUTPUT_ROOT = ROOT / "study_run_2019_2023" / "figures_formal_panels"
EXPECTED_FIG6 = {
    "fig6a_policy_response_map.png",
    "fig6b_scenario_pollutant_lollipop.png",
    "fig6c_city_risk_change_distribution.png",
    "fig6d_regional_policy_effect_dumbbell.png",
    "fig6e_policy_benefit_concentration.png",
}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _verify_pngs(run_dir: Path) -> None:
    figure6_dir = run_dir / "figure_6"
    found = {path.name for path in figure6_dir.glob("*.png")} if figure6_dir.is_dir() else set()
    missing = sorted(EXPECTED_FIG6 - found)
    unexpected = sorted(found - EXPECTED_FIG6)
    if missing or unexpected:
        raise SystemExit(f"Figure 6 PNG set mismatch; missing={missing}, unexpected={unexpected}")
    for path in sorted(figure6_dir.glob("*.png")):
        with Image.open(path) as image:
            width, height = image.size
            if width < 1000 or height < 1000:
                raise SystemExit(f"Figure 6 panel is unexpectedly small: {path} ({width}×{height})")
            image.verify()

    all_files = [path for path in run_dir.rglob("*") if path.is_file()]
    non_png = [path for path in all_files if path.suffix.lower() != ".png"]
    pngs = [path for path in all_files if path.suffix.lower() == ".png"]
    if non_png:
        raise SystemExit(f"Clean panel directory contains non-PNG files: {non_png[:10]}")
    if len(pngs) != 34:
        raise SystemExit(f"Expected 34 main-text PNG panels after Figure 6, found {len(pngs)}")
    print(f"\nQA passed: {len(pngs)} independent PNG panels; Figure 6 = 5/5; no non-PNG files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish and verify the five real-data Figure 6 panels")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--run-name", default="main_text_panels_v3")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=330)
    args = parser.parse_args()
    if args.year != 2025:
        raise SystemExit("The current Figure 6 design is explicitly the 2025 policy evaluation")

    evaluator = _module(POLICY / "evaluate_policy_effects.py", "policy_evaluator")
    config_loader_dir = ROOT / "emission_network"
    if str(config_loader_dir) not in sys.path:
        sys.path.insert(0, str(config_loader_dir))
    from config_loader import load_config
    cfg = load_config()
    ship_types = list(cfg["emission_network"]["ship_types"])
    _, _, _, missing = evaluator._read_year(str(args.results_root), args.year, ship_types)
    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... plus {len(missing)-20} items" if len(missing) > 20 else ""
        raise SystemExit(
            f"Figure 6 is not ready: {len(missing)} monthly/ship-type products are missing:\n  "
            f"{preview}{suffix}"
        )

    args.eval_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    python = sys.executable
    _run([
        python, str(POLICY / "evaluate_policy_effects.py"),
        "--year", str(args.year),
        "--results-root", str(args.results_root),
        "--out-dir", str(args.eval_dir),
    ], env)
    _run([
        python, str(POLICY / "build_policy_spri_scenarios.py"),
        "--year", str(args.year),
        "--results-root", str(args.results_root),
        "--policy-city-csv", str(args.eval_dir / f"policy_eval_city_{args.year}.csv"),
        "--out-csv", str(args.eval_dir / f"policy_spri_scenarios_{args.year}.csv"),
    ], env)
    _run([
        python, "-m", "study_run_2019_2023.figure_pipeline",
        "--figures", "6",
        "--run-name", args.run_name,
        "--output-dir", str(args.output_root),
        "--formats", "png",
        "--dpi", str(args.dpi),
        "--strict",
    ], env)
    _verify_pngs(args.output_root / args.run_name)
    print("All independent main-text panels are ready for final visual QA and PPT assembly.")


if __name__ == "__main__":
    main()
