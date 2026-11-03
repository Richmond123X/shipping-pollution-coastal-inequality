# Shipping pollution risks and unequal coastal protection

Code archive accompanying **Emerging shipping pollution risks expose unequal protection across global coastal cities**.

Authors: Jia Shi, Jihong Chen, Qingquan Li and Shouyang Wang.

This repository archives the research code available in the project workspace on 3 November 2026. Data, manuscript files, figures, model weights, credentials and local Git history are not included. The separately prepared Figshare data archive contains the selected source data.

## Code map

| Directory | Purpose | Reproduction scope |
| --- | --- | --- |
| `emission_network/` | AIS processing, emissions and city networks | Requires original AIS databases, vessel records and external spatial inputs |
| `port2city/` | Port-city assignment and catchment construction | Requires original spatial inputs and reviewed mappings |
| `covariates/` | Population, GDP, healthcare, regulation and governance inputs | Requires cited source datasets |
| `gnn/` | Available graph forecasting models and initial training framework | Earlier framework; not the missing final training driver |
| `empirical_trade_emissions/scripts/` | Earlier country and country-pair PPML workflow | Earlier specification; not the final city PPML driver |
| `results_v10_reconstructed_2019_2027/scripts/` | Result reconstruction, plotting and delivery verification | Some scripts require the unavailable analysis helpers noted below |
| `policy_clustering_v12_formal/` | Coastal regionalization with the exact-OR Max-p constraint | Requires spatial inputs and risk panel |
| `policy_simulation_v12_formal/` | Formal S1-S7 policy simulations and figures | AIS computation year 2024; risk-selection anchor 2025 |
| `figure_scripts/` | Final risk maps and transition diagrams | Requires archived panels and map inputs |
| `release_tools/` | Portable checks of the archived source data | Requires the separately downloaded Figshare source-data package |

## Reading and running the archive

Python dependencies are listed in `requirements-archive.txt`; these are dependency names, not a claimed historical lockfile. The R workflow uses `fixest`, `data.table` and the additional packages named in its scripts. Read each component README before running. Original scripts retain their historical Windows paths and configuration; update those paths for the execution machine. Some entry points support `TRANSFER_JUSTICE_ROOT`, `V10_OUT` and `V10_MONTHLY_ROOT` environment variables.

A portable verification of the Figshare source data can be run independently:

```bash
python release_tools/check_source_data.py --data-root /path/to/01_source_data --output verification.json
```

## Archive completeness

The original final analysis directory `results_v10_empirical_risk_2019_2026/scripts` was not present when this archive was made. Its `common.py` and final forecasting, city-level empirical and risk-analysis drivers therefore cannot be claimed as recovered. Available earlier code is retained with this limitation explicitly documented. The repository is an archive of available source, rather than a claim that every published analysis can currently be regenerated end to end. Saved forecasts, risk panels and method manifests are included in the separate data archive.

The original reconstruction manifest records neighbor-month reconstruction for May 2019, May 2023 and October 2023, a 1.25 scaling of 2025 emission outputs, and a December 2025 network boundary correction. These provenance records are retained in the data archive.

No new software license has been selected by this archival operation. Original third-party code and data remain subject to their respective terms.
