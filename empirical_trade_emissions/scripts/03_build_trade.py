#!/usr/bin/env python3
"""Extract and aggregate official BACI 202601 and CEPII Gravity/GeoDist data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = tuple(range(2019, 2025))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_baci_subset(archive: Path, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt member in {archive}: {bad}")
        names = zf.namelist()
        wanted_patterns = [
            *(rf"(?:^|/)BACI_HS02_Y{year}_V202601\.csv$" for year in YEARS),
            r"(?:^|/)country_codes_V202601\.csv$",
            r"(?:^|/)product_codes_HS02_V202601\.csv$",
            r"(?:^|/)readme.*",
        ]
        members = []
        for pattern in wanted_patterns:
            match = [name for name in names if re.search(pattern, name, flags=re.I)]
            if not match and not pattern.lower().startswith("(?:^|/)readme"):
                raise RuntimeError(f"No BACI archive member matches {pattern}")
            members.extend(match)
        for name in sorted(set(members)):
            target = output / Path(name).name
            info = zf.getinfo(name)
            if not target.exists() or target.stat().st_size != info.file_size:
                with zf.open(name) as source, target.open("wb") as destination:
                    while block := source.read(8 * 1024 * 1024):
                        destination.write(block)
            if target.stat().st_size != info.file_size:
                raise RuntimeError(f"Extracted-size mismatch for {target}")
            selected[target.name] = target
    return selected


def identify_country_columns(frame: pd.DataFrame) -> tuple[str, str]:
    normalized = {re.sub(r"[^a-z0-9]", "", col.lower()): col for col in frame.columns}
    numeric_candidates = [
        "countrycode",
        "countryid",
        "iso3num",
        "iso3numeric",
        "countrycodenumeric",
    ]
    iso_candidates = [
        "countryiso3",
        "iso3digitaalpha",
        "iso3digitalpha",
        "iso3alpha",
        "iso3",
        "iso3code",
    ]
    numeric = next((normalized[key] for key in numeric_candidates if key in normalized), None)
    iso3 = next((normalized[key] for key in iso_candidates if key in normalized), None)
    if numeric is None or iso3 is None:
        raise RuntimeError(f"Cannot identify country-code columns: {frame.columns.tolist()}")
    return numeric, iso3


def load_baci_country_map(path: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    source = pd.read_csv(path)
    numeric_col, iso_col = identify_country_columns(source)
    out = source.copy()
    out["baci_numeric_code"] = pd.to_numeric(out[numeric_col], errors="coerce").astype("Int64")
    out["iso3_analysis"] = out[iso_col].astype("string").str.upper().str.strip()
    out.loc[out["iso3_analysis"].isin(["", "NAN", "NA"]), "iso3_analysis"] = pd.NA
    out["mapping_override"] = ""
    mask_490 = out["baci_numeric_code"].eq(490)
    if not mask_490.any():
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [{"baci_numeric_code": 490, "iso3_analysis": "TWN", "mapping_override": "BACI 490 -> TWN"}]
                ),
            ],
            ignore_index=True,
        )
    else:
        out.loc[mask_490, "iso3_analysis"] = "TWN"
        out.loc[mask_490, "mapping_override"] = "BACI 490 -> TWN"
    usable = out.dropna(subset=["baci_numeric_code", "iso3_analysis"])
    if usable["baci_numeric_code"].duplicated().any():
        raise RuntimeError("Duplicate BACI numeric code in official metadata")
    mapping = dict(zip(usable["baci_numeric_code"].astype(int), usable["iso3_analysis"].astype(str)))
    return out, mapping


def aggregate_baci_year(path: Path, mapping: dict[int, str], chunksize: int = 750_000) -> tuple[pd.DataFrame, dict]:
    pieces: list[pd.DataFrame] = []
    unmapped_codes: set[int] = set()
    stats = {
        "year": int(re.search(r"Y(\d{4})", path.name).group(1)),
        "source_file": str(path.resolve()),
        "source_bytes": path.stat().st_size,
        "rows": 0,
        "mapped_rows": 0,
        "q_reported_rows": 0,
        "trade_value_kusd": 0.0,
        "q_covered_trade_value_kusd": 0.0,
        "quantity_tonnes": 0.0,
    }
    reader = pd.read_csv(
        path,
        usecols=["t", "i", "j", "k", "v", "q"],
        dtype={"t": "int16", "i": "int32", "j": "int32", "k": "string"},
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk["v"] = pd.to_numeric(chunk["v"], errors="coerce")
        chunk["q"] = pd.to_numeric(chunk["q"], errors="coerce")
        chunk["i_iso3"] = chunk["i"].map(mapping)
        chunk["j_iso3"] = chunk["j"].map(mapping)
        mapped = chunk["i_iso3"].notna() & chunk["j_iso3"].notna()
        if not mapped.all():
            unmapped_codes.update(chunk.loc[chunk["i_iso3"].isna(), "i"].astype(int).tolist())
            unmapped_codes.update(chunk.loc[chunk["j_iso3"].isna(), "j"].astype(int).tolist())
        q_reported = chunk["q"].notna() & chunk["q"].gt(0)
        stats["rows"] += len(chunk)
        stats["mapped_rows"] += int(mapped.sum())
        stats["q_reported_rows"] += int(q_reported.sum())
        stats["trade_value_kusd"] += float(chunk["v"].sum(skipna=True))
        stats["q_covered_trade_value_kusd"] += float(chunk.loc[q_reported, "v"].sum(skipna=True))
        stats["quantity_tonnes"] += float(chunk["q"].sum(skipna=True))
        if not mapped.all():
            # Retain all failures in stats below; do not silently drop. The final
            # mapping threshold is enforced after the full file has been read.
            pass
        work = chunk.loc[mapped, ["t", "i_iso3", "j_iso3", "v", "q"]].copy()
        work["q_value_covered"] = work["v"].where(work["q"].notna() & work["q"].gt(0), 0.0)
        work["q_reported_flow"] = (work["q"].notna() & work["q"].gt(0)).astype("int32")
        work["flow"] = 1
        grouped = work.groupby(["i_iso3", "j_iso3", "t"], observed=True, as_index=False).agg(
            q_tonnes=("q", "sum"),
            v_kusd=("v", "sum"),
            q_covered_v_kusd=("q_value_covered", "sum"),
            flow_count=("flow", "sum"),
            q_reported_flow_count=("q_reported_flow", "sum"),
        )
        pieces.append(grouped)
    out = pd.concat(pieces, ignore_index=True)
    out = out.groupby(["i_iso3", "j_iso3", "t"], observed=True, as_index=False).sum(numeric_only=True)
    out = out.rename(columns={"t": "year"})
    out = out.loc[out["i_iso3"] != out["j_iso3"]].copy()
    out["q_cov"] = out["q_covered_v_kusd"] / out["v_kusd"].replace(0, np.nan)
    stats["mapping_rate"] = stats["mapped_rows"] / stats["rows"] if stats["rows"] else 1.0
    stats["q_reported_row_rate"] = stats["q_reported_rows"] / stats["rows"] if stats["rows"] else np.nan
    stats["q_covered_value_rate"] = (
        stats["q_covered_trade_value_kusd"] / stats["trade_value_kusd"]
        if stats["trade_value_kusd"]
        else np.nan
    )
    stats["unmapped_numeric_codes"] = ";".join(str(code) for code in sorted(unmapped_codes))
    if stats["mapping_rate"] < 0.995:
        raise RuntimeError(f"BACI mapping rate below 99.5% for {path}: {stats['mapping_rate']:.6%}")
    return out.sort_values(["i_iso3", "j_iso3", "year"], ignore_index=True), stats


def build_geodist(archive: Path) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt member in {archive}: {bad}")
        with zf.open("Gravity_V202211.csv") as handle:
            reader = pd.read_csv(
                handle,
                usecols=[
                    "year",
                    "iso3_o",
                    "iso3_d",
                    "country_exists_o",
                    "country_exists_d",
                    "contig",
                ],
                dtype={
                    "year": "int16",
                    "iso3_o": "string",
                    "iso3_d": "string",
                    "country_exists_o": "float64",
                    "country_exists_d": "float64",
                    "contig": "float64",
                },
                chunksize=1_000_000,
            )
            for chunk in reader:
                active_2020 = (
                    chunk["year"].eq(2020)
                    & chunk["country_exists_o"].eq(1)
                    & chunk["country_exists_d"].eq(1)
                )
                selected = chunk.loc[active_2020, ["iso3_o", "iso3_d", "contig"]]
                if len(selected):
                    pieces.append(selected)
    out = pd.concat(pieces, ignore_index=True)
    out = out.rename(columns={"iso3_o": "i_iso3", "iso3_d": "j_iso3"})
    out["contig"] = out["contig"].fillna(0).astype("int8")
    if out[["i_iso3", "j_iso3"]].duplicated().any():
        raise RuntimeError("Duplicate country pair in 2020 GeoDist extraction")
    return out.sort_values(["i_iso3", "j_iso3"], ignore_index=True)


def add_contiguity(trade: pd.DataFrame, geodist: pd.DataFrame) -> pd.DataFrame:
    out = trade.merge(geodist, on=["i_iso3", "j_iso3"], how="left", validate="many_to_one")
    out["geodist_matched"] = out["contig"].notna()
    # Missing pairs remain unknown, not silently coded as non-contiguous.
    out["q_noncontig_tonnes"] = out["q_tonnes"].where(out["contig"].eq(0), np.nan)
    return out


def build_country_trade(trade: pd.DataFrame) -> pd.DataFrame:
    exp = trade.groupby(["i_iso3", "year"], as_index=False).agg(
        q_exports_tonnes=("q_tonnes", "sum"),
        v_exports_kusd=("v_kusd", "sum"),
        q_noncontig_exports_tonnes=("q_noncontig_tonnes", "sum"),
    ).rename(columns={"i_iso3": "iso3"})
    imp = trade.groupby(["j_iso3", "year"], as_index=False).agg(
        q_imports_tonnes=("q_tonnes", "sum"),
        v_imports_kusd=("v_kusd", "sum"),
        q_noncontig_imports_tonnes=("q_noncontig_tonnes", "sum"),
    ).rename(columns={"j_iso3": "iso3"})
    out = exp.merge(imp, on=["iso3", "year"], how="outer", validate="one_to_one").fillna(0)
    out["q_total_tonnes"] = out["q_exports_tonnes"] + out["q_imports_tonnes"]
    out["v_total_kusd"] = out["v_exports_kusd"] + out["v_imports_kusd"]
    out["q_noncontig_total_tonnes"] = (
        out["q_noncontig_exports_tonnes"] + out["q_noncontig_imports_tonnes"]
    )
    return out.sort_values(["iso3", "year"], ignore_index=True)


def version_comparison(workspace: Path, current: pd.DataFrame) -> pd.DataFrame:
    old_path = workspace / "study_run_2019_2023" / "trade_correlation" / "input" / "baci_slim.parquet"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_parquet(old_path)
    old = old.groupby(["origin_iso", "dest_iso", "year"], observed=True, as_index=False)["qty_t"].sum()
    old = old.rename(columns={"origin_iso": "i_iso3", "dest_iso": "j_iso3", "qty_t": "q_202501"})
    new = current.loc[current["year"].le(2023), ["i_iso3", "j_iso3", "year", "q_tonnes"]].rename(
        columns={"q_tonnes": "q_202601"}
    )
    merged = old.merge(new, on=["i_iso3", "j_iso3", "year"], how="outer").fillna(0)
    rows = []
    for year, group in merged.groupby("year"):
        both = group.loc[group["q_202501"].gt(0) & group["q_202601"].gt(0)]
        rows.append(
            {
                "year": int(year),
                "union_dyads": len(group),
                "positive_in_both_dyads": len(both),
                "pearson_q_union_zeros": group["q_202501"].corr(group["q_202601"]),
                "pearson_log1p_union_zeros": np.log1p(group["q_202501"]).corr(np.log1p(group["q_202601"])),
                "pearson_log_positive_both": np.log(both["q_202501"]).corr(np.log(both["q_202601"])),
                "q_202501_total": group["q_202501"].sum(),
                "q_202601_total": group["q_202601"].sum(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / "empirical_trade_emissions"
    raw = base / "data_raw"
    interim = base / "data_interim" / "baci_202601"
    output = base / "data_processed"
    qc = base / "results" / "qc"
    output.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)

    archive = raw / "baci_202601" / "BACI_HS02_V202601.zip"
    gravity_archive = raw / "geodist_202211" / "Gravity_csv_V202211.zip"
    extracted = extract_baci_subset(archive, interim)
    country_file = extracted["country_codes_V202601.csv"]
    country_meta, mapping = load_baci_country_map(country_file)
    country_meta.to_csv(output / "baci_country_codes_V202601_with_overrides.csv", index=False, encoding="utf-8-sig")

    annual = []
    stats = []
    for year in YEARS:
        path = extracted[f"BACI_HS02_Y{year}_V202601.csv"]
        cache = interim / f"trade_dyad_Y{year}_V202601.parquet"
        stats_cache = interim / f"trade_dyad_Y{year}_V202601_qc.json"
        if cache.exists() and stats_cache.exists():
            aggregated = pd.read_parquet(cache)
            year_stats = json.loads(stats_cache.read_text(encoding="utf-8"))
        else:
            aggregated, year_stats = aggregate_baci_year(path, mapping)
            aggregated.to_parquet(cache, index=False)
            stats_cache.write_text(
                json.dumps(year_stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        annual.append(aggregated)
        stats.append(year_stats)
    trade = pd.concat(annual, ignore_index=True)

    geodist = build_geodist(gravity_archive)
    geodist.to_parquet(output / "geodist_contiguity_V202211.parquet", index=False)
    trade = add_contiguity(trade, geodist)
    trade.to_parquet(output / "trade_dyad_annual.parquet", index=False)
    country = build_country_trade(trade)
    country.to_parquet(output / "trade_country_annual.parquet", index=False)

    stats_frame = pd.DataFrame(stats)
    stats_frame.to_csv(qc / "baci_aggregation_qc.csv", index=False, encoding="utf-8-sig")
    comparison = version_comparison(workspace, trade)
    comparison.to_csv(qc / "baci_version_202601_vs_202501.csv", index=False, encoding="utf-8-sig")

    unmapped_codes = []
    for row in stats:
        text_codes = row.get("unmapped_numeric_codes", "")
        for code in text_codes.split(";") if text_codes else []:
            unmapped_codes.append({"year": row["year"], "baci_numeric_code": int(code)})
    pd.DataFrame(unmapped_codes, columns=["year", "baci_numeric_code"]).to_csv(
        qc / "baci_unmapped_numeric_codes.csv", index=False, encoding="utf-8-sig"
    )

    manifest_sources = [
        (archive, "https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS02_V202601.zip"),
        (gravity_archive, "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"),
        (raw / "baci_202601" / "release_notes_202601.pdf", "https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf"),
        (raw / "world_bank" / "OGHIST.xlsx", "https://databankfiles.worldbank.org/public/ddpext_download/site-content/OGHIST.xlsx"),
        (raw / "un_m49" / "UN_M49_overview.html", "https://unstats.un.org/unsd/methodology/m49/overview/"),
    ]
    method_archive = raw / "method_code" / "ppml_fe_bias_master.zip"
    if method_archive.exists():
        manifest_sources.append(
            (method_archive, "https://github.com/tomzylkin/ppml_fe_bias/archive/refs/heads/master.zip")
        )
    manifest_rows = []
    for path, source in manifest_sources:
        manifest_rows.append(
            {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path), "source_url": source}
        )
    pd.DataFrame(manifest_rows).to_csv(raw / "source_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "baci_archive_sha256": manifest_rows[0]["sha256"],
        "gravity_archive_sha256": manifest_rows[1]["sha256"],
        "trade_dyad_year_rows": int(len(trade)),
        "trade_country_year_rows": int(len(country)),
        "baci_iso3_count": int(len(set(trade["i_iso3"]) | set(trade["j_iso3"]))),
        "geodist_pair_count": int(len(geodist)),
        "geodist_trade_pair_match_rate": float(trade["geodist_matched"].mean()),
        "unmapped_baci_code_year_rows": len(unmapped_codes),
    }
    (qc / "trade_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
