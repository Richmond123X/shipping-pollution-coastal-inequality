# -*- coding: utf-8 -*-
"""Build a non-destructive reconstructed monthly-emission data root.

Canonical corrections
---------------------
* 201905, 202305 and 202310 are reconstructed as the arithmetic mean of the
  preceding and following months at the finest available output key.
* All 2025 emission quantities are multiplied by 1.25, preserving the observed
  spatial, ship-type, pollutant and operating-mode composition.
* 202512 OD-edge emissions receive an additional pollutant-specific boundary
  correction.  The correction restores the December edge/true-total ratio to
  the robust median ratio observed during 202501--202511, while retaining the
  observed December topology and relative edge weights.

Only downstream-required CSV products are copied.  The source tree is opened
read-only and is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(r"E:\transfer justice")
DEFAULT_SOURCE = WORKSPACE / "results_monthly"
DEFAULT_OUTPUT = WORKSPACE / "results_v10_reconstructed_2019_2027" / "reconstructed_monthly"

YEARS = range(2019, 2026)
SHIPS = ["Container", "General Cargo", "Oil Tanker", "Bulk Carrier"]
POLLUTANTS = ["SO2", "PM", "CO2", "NOx", "CO", "CH4", "N2O", "NMVOC"]
MAIN_POLLUTANTS = ["SO2", "PM", "NOx"]
TARGET_MONTHS = {
    "201905": ("201904", "201906"),
    "202305": ("202304", "202306"),
    "202310": ("202309", "202311"),
}
YEAR_2025_SCALE = 1.25

COPY_PREFIXES = (
    "city_emissions_",
    "city_edge_list_",
    "true_totals_",
    "port_operational_monthly_",
    "city_policy_delta_",
    "policy_delta_totals_",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ship_dir(root: Path, ym: str, ship: str) -> Path:
    direct = root / ym / ship
    if direct.is_dir():
        return direct
    matches = [p for p in (root / ym).iterdir() if p.is_dir() and p.name.lower() == ship.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve {ym}/{ship} below {root}")
    return matches[0]


def _target_name(name: str, source_ym: str, target_ym: str) -> str:
    if source_ym not in name:
        raise ValueError(f"Expected month token {source_ym} in {name}")
    return name.replace(source_ym, target_ym)


def _mean_series(a: pd.Series, b: pd.Series, missing_zero: bool) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    if missing_zero:
        return (aa.fillna(0.0) + bb.fillna(0.0)) / 2.0
    return pd.concat([aa, bb], axis=1).mean(axis=1, skipna=True)


def _mean_keyed_csv(before: Path, after: Path, output: Path, target_ym: str) -> dict:
    a = pd.read_csv(before)
    b = pd.read_csv(after)
    if list(a.columns) != list(b.columns):
        raise ValueError(f"Schema mismatch: {before} vs {after}")
    columns = list(a.columns)
    if output.name.startswith("city_emissions_"):
        keys = ["node_id"]
        zero_missing = {c for c in columns if c not in {"node_id", "YearMonth", "Ship_Type"}}
    elif output.name.startswith("city_edge_list_"):
        keys = ["From_City", "To_City"]
        zero_missing = {"Total_Emission"}
    elif output.name.startswith("port_operational_monthly_"):
        keys = ["Port_Name"]
        zero_missing = {c for c in columns if c == "Visit_Count" or c.startswith("Total_")}
    elif output.name.startswith("city_policy_delta_"):
        keys = [c for c in ["node_id", "Zone_ID"] if c in columns]
        zero_missing = {
            c for c in columns
            if c not in set(keys) | {"YearMonth", "Ship_Type"}
            and pd.api.types.is_numeric_dtype(a[c])
        }
    else:
        raise ValueError(f"Unsupported keyed file: {output.name}")

    if not keys:
        raise ValueError(f"No key columns for {output}")
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError(f"Duplicate keys in neighbors for {output.name}")
    ai = a.set_index(keys)
    bi = b.set_index(keys)
    idx = ai.index.union(bi.index, sort=False)
    ai = ai.reindex(idx)
    bi = bi.reindex(idx)
    out = pd.DataFrame(index=idx)
    for col in columns:
        if col in keys:
            continue
        if col == "YearMonth":
            out[col] = int(target_ym)
        elif col == "Ship_Type":
            out[col] = ai[col].combine_first(bi[col])
        elif pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            out[col] = _mean_series(ai[col], bi[col], col in zero_missing)
        else:
            out[col] = ai[col].combine_first(bi[col])
    out = out.reset_index()[columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    return {"rows_before": len(a), "rows_after": len(b), "rows_output": len(out)}


def _mean_true_totals(before: Path, after: Path, output: Path, target_ym: str) -> dict:
    a = pd.read_csv(before)
    b = pd.read_csv(after)
    if list(a.columns) != list(b.columns) or len(a) != 1 or len(b) != 1:
        raise ValueError(f"Unexpected true-total schema: {before} / {after}")
    out = a.copy()
    for col in out.columns:
        if col == "YearMonth":
            out.loc[0, col] = int(target_ym)
        elif col == "Ship_Type":
            out.loc[0, col] = a.loc[0, col]
        else:
            out.loc[0, col] = (
                pd.to_numeric(a.loc[0, col], errors="coerce")
                + pd.to_numeric(b.loc[0, col], errors="coerce")
            ) / 2.0
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    return {"rows_before": 1, "rows_after": 1, "rows_output": 1}


def _copy_required(source: Path, output: Path) -> list[dict]:
    audit: list[dict] = []
    for year in YEARS:
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            for ship in SHIPS:
                src_dir = _ship_dir(source, ym, ship)
                dst_dir = output / ym / ship
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in sorted(src_dir.glob("*.csv")):
                    if not src.name.startswith(COPY_PREFIXES):
                        continue
                    dst = dst_dir / src.name
                    shutil.copy2(src, dst)
                    audit.append({
                        "file": str(dst.relative_to(output)),
                        "operation": "copied",
                        "factor": 1.0,
                        "source_a": str(src),
                        "source_b": "",
                        "rows_output": sum(1 for _ in src.open("rb")) - 1,
                    })
    return audit


def _reconstruct_bad_months(source: Path, output: Path, audit: list[dict]) -> None:
    for target_ym, (before_ym, after_ym) in TARGET_MONTHS.items():
        for ship in SHIPS:
            target_dir = _ship_dir(source, target_ym, ship)
            before_dir = _ship_dir(source, before_ym, ship)
            after_dir = _ship_dir(source, after_ym, ship)
            out_dir = output / target_ym / ship
            for target_source in sorted(target_dir.glob("*.csv")):
                if not target_source.name.startswith(COPY_PREFIXES):
                    continue
                before = before_dir / _target_name(target_source.name, target_ym, before_ym)
                after = after_dir / _target_name(target_source.name, target_ym, after_ym)
                if not before.exists() or not after.exists():
                    # Optional products such as true_totals were not emitted in
                    # the early pipeline years; do not invent a new schema.
                    continue
                out = out_dir / target_source.name
                if out.name.startswith("true_totals_") or out.name.startswith("policy_delta_totals_"):
                    stats = _mean_true_totals(before, after, out, target_ym)
                else:
                    stats = _mean_keyed_csv(before, after, out, target_ym)
                audit.append({
                    "file": str(out.relative_to(output)),
                    "operation": "neighbor_mean",
                    "factor": 1.0,
                    "source_a": str(before),
                    "source_b": str(after),
                    **stats,
                })


def _true_total(output: Path, ym: str, pollutant: str) -> float:
    value = 0.0
    for ship in SHIPS:
        path = output / ym / ship / f"true_totals_{ship}_{ym}.csv"
        frame = pd.read_csv(path)
        value += pd.to_numeric(frame[pollutant], errors="coerce").fillna(0).sum()
    return float(value)


def _edge_total(output: Path, ym: str, pollutant: str) -> float:
    value = 0.0
    for ship in SHIPS:
        path = output / ym / ship / f"city_edge_list_{pollutant}_{ship}_{ym}.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["From_City"].astype(str) != frame["To_City"].astype(str)]
        value += pd.to_numeric(frame["Total_Emission"], errors="coerce").fillna(0).sum()
    return float(value)


def _december_boundary_factors(output: Path) -> dict[str, float]:
    factors: dict[str, float] = {}
    for pollutant in POLLUTANTS:
        ratios = []
        for month in range(1, 12):
            ym = f"2025{month:02d}"
            total = _true_total(output, ym, pollutant)
            if total > 0:
                ratios.append(_edge_total(output, ym, pollutant) / total)
        dec_true = _true_total(output, "202512", pollutant)
        dec_ratio = _edge_total(output, "202512", pollutant) / dec_true if dec_true > 0 else np.nan
        target_ratio = float(np.median(ratios))
        factor = target_ratio / dec_ratio
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Invalid 202512 boundary factor for {pollutant}: {factor}")
        factors[pollutant] = float(factor)
    return factors


def _scale_frame(path: Path, factor: float) -> int:
    frame = pd.read_csv(path)
    if path.name.startswith("city_emissions_"):
        columns = [c for c in frame.columns if c not in {"node_id", "YearMonth", "Ship_Type"}]
    elif path.name.startswith("city_edge_list_"):
        columns = ["Total_Emission"]
    elif path.name.startswith("true_totals_"):
        columns = [c for c in frame.columns if c not in {"YearMonth", "Ship_Type"}]
    elif path.name.startswith(("city_policy_delta_", "policy_delta_totals_")):
        protected = {"node_id", "YearMonth", "Ship_Type", "Zone_ID"}
        columns = [c for c in frame.columns if c not in protected and pd.api.types.is_numeric_dtype(frame[c])]
    else:
        return 0
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0) * factor
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return len(frame)


def _scale_2025(output: Path, audit: list[dict]) -> dict[str, float]:
    boundary = _december_boundary_factors(output)
    for month in range(1, 13):
        ym = f"2025{month:02d}"
        for ship in SHIPS:
            folder = output / ym / ship
            for path in sorted(folder.glob("*.csv")):
                if not path.name.startswith((
                    "city_emissions_", "city_edge_list_", "true_totals_",
                    "city_policy_delta_", "policy_delta_totals_",
                )):
                    continue
                factor = YEAR_2025_SCALE
                if ym == "202512" and path.name.startswith("city_edge_list_"):
                    pollutant = path.name.removeprefix("city_edge_list_").split("_", 1)[0]
                    factor *= boundary[pollutant]
                rows = _scale_frame(path, factor)
                audit.append({
                    "file": str(path.relative_to(output)),
                    "operation": "scaled_2025" if factor == YEAR_2025_SCALE else "scaled_2025_and_dec_boundary",
                    "factor": factor,
                    "source_a": str(path),
                    "source_b": "",
                    "rows_output": rows,
                })
    return boundary


def _validate(source: Path, output: Path, boundary: dict[str, float]) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    expected_city = 7 * 12 * len(SHIPS)
    expected_edge = expected_city * len(POLLUTANTS)
    city_files = list(output.glob("20????/*/city_emissions_*.csv"))
    edge_files = list(output.glob("20????/*/city_edge_list_*.csv"))
    add("city emission file count", len(city_files) == expected_city, len(city_files))
    add("edge emission file count", len(edge_files) == expected_edge, len(edge_files))

    for target, (before, after) in TARGET_MONTHS.items():
        for pollutant in MAIN_POLLUTANTS:
            target_sum = before_sum = after_sum = 0.0
            for ship in SHIPS:
                for ym, bucket in [(target, "target"), (before, "before"), (after, "after")]:
                    root = output if ym == target else source
                    folder = _ship_dir(root, ym, ship)
                    frame = pd.read_csv(folder / f"city_emissions_{ship}_{ym}.csv")
                    val = pd.to_numeric(frame[f"{pollutant}_Total"], errors="coerce").fillna(0).sum()
                    if bucket == "target": target_sum += val
                    elif bucket == "before": before_sum += val
                    else: after_sum += val
            expected = (before_sum + after_sum) / 2.0
            rel = abs(target_sum - expected) / max(abs(expected), 1.0)
            add(f"{target} {pollutant} city neighbor mean", rel < 1e-10, rel)

    for month in range(1, 12):
        ym = f"2025{month:02d}"
        for pollutant in MAIN_POLLUTANTS:
            reconstructed = original = 0.0
            for ship in SHIPS:
                a = pd.read_csv(output / ym / ship / f"city_emissions_{ship}_{ym}.csv")
                b = pd.read_csv(_ship_dir(source, ym, ship) / f"city_emissions_{ship}_{ym}.csv")
                reconstructed += pd.to_numeric(a[f"{pollutant}_Total"], errors="coerce").fillna(0).sum()
                original += pd.to_numeric(b[f"{pollutant}_Total"], errors="coerce").fillna(0).sum()
            ratio = reconstructed / original
            add(f"{ym} {pollutant} node scale", abs(ratio - YEAR_2025_SCALE) < 1e-10, ratio)

    add("202512 boundary factors finite", all(np.isfinite(list(boundary.values()))), boundary)
    add("no negative city emissions", all(
        not (pd.read_csv(path).select_dtypes(include=[np.number]).drop(columns=["YearMonth"], errors="ignore") < 0).any().any()
        for path in city_files
    ))
    add("no negative edge emissions", all(
        pd.to_numeric(pd.read_csv(path)["Total_Emission"], errors="coerce").fillna(0).ge(0).all()
        for path in edge_files
    ))

    required_policy_baselines = [
        output / f"2025{month:02d}" / ship / f"city_emissions_{ship}_2025{month:02d}.csv"
        for month in range(1, 13) for ship in SHIPS
    ]
    add("2025 policy baseline complete", all(path.is_file() for path in required_policy_baselines), len(required_policy_baselines))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    expected_parent = (WORKSPACE / "results_v10_reconstructed_2019_2027").resolve()
    if output.parent != expected_parent or output.name != "reconstructed_monthly":
        raise ValueError(f"Unexpected output root: {output}")
    output.mkdir(parents=True, exist_ok=True)

    audit = _copy_required(source, output)
    _reconstruct_bad_months(source, output, audit)
    boundary = _scale_2025(output, audit)
    checks = _validate(source, output, boundary)

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(expected_parent / "reconstruction_file_audit.csv", index=False, encoding="utf-8-sig")
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(expected_parent / "reconstruction_validation.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source_root": str(source),
        "output_root": str(output),
        "years": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "neighbor_mean_months": TARGET_MONTHS,
        "year_2025_scale": YEAR_2025_SCALE,
        "december_edge_boundary_method": "pollutant-specific median Jan-Nov edge/true-total ratio",
        "december_edge_boundary_factors": boundary,
        "policy_compatibility": {
            "baseline_city_emissions_2025_complete": True,
            "policy_delta_files_present": bool(list(output.glob("2025??/*/city_policy_delta_*.csv"))),
            "note": "Policy delta files are optional scenario outputs. If later supplied, this script reconstructs/scales them with the same rules.",
        },
        "validation_all_passed": bool(check_frame.passed.all()),
        "validation": checks,
        "audit_sha256": _sha256(expected_parent / "reconstruction_file_audit.csv"),
    }
    with (expected_parent / "reconstruction_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(check_frame.to_string(index=False))
    print("202512 boundary factors:", json.dumps(boundary, ensure_ascii=False, indent=2))
    if not check_frame.passed.all():
        raise SystemExit("reconstruction validation failed")


if __name__ == "__main__":
    main()
