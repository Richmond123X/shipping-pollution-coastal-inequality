# 贸易—船舶排放实证研究（研究内容 1，4.6 节）

本目录用于复现 `research_plan_v8.md` 第 4.6 节的两套模型：有向国家对×年的航段 PPML，以及国家×年的沿海城市 30 km 集水区 PPML。研究窗口为 2019–2024；BACI 版本固定为 202601、HS 2002。

所有外部输入均保留下载地址、文件大小与 SHA-256。独立的 UN Comtrade 文件不属于本流程，也不会被脚本读取。

目录结构：

- `data_raw/`：官方原始压缩包、元数据与来源清单；
- `data_interim/`：从大文件提取的必要年度文件；
- `data_processed/`：六个方案规定的 Parquet 中间产品及分组表；
- `results/`：质量控制、样本损耗、回归系数、稳健性结果和报告；
- `scripts/`：从原始数据到最终结果的可复现脚本。

注意：模型二使用方案 4.6.2-B 允许的替代口径，即将 `city_emissions_*` 的城市总排放按国家求和。2019–2024 的点级排放中间产物不可完整复用，故无法在本轮对国家集水区并集重新空间连接；城市缓冲区重叠会造成重复计数，方法与结果报告会显式披露。

## 复现

原始官方文件已经存放在 `data_raw/`，运行前可用 `data_raw/source_manifest.csv` 核对地址、大小和 SHA-256。完整重跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\empirical_trade_emissions\run_all.ps1
```

若已确认 `03_build_trade.py` 生成的 BACI/GeoDist 中间表没有变化，可跳过约 2 GB BACI 压缩包的重新解压聚合：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\empirical_trade_emissions\run_all.ps1 -SkipTradeRebuild
```

`-SkipTradeRebuild` 只复用已生成的官方贸易中间表；排放、分组、面板、全部回归、偏误修正和报告仍会重新运行。

## 主要交付

- `results/empirical_report_zh.md`：中文实证报告；
- `results/table_1_main_results.md`：正文六列主表；
- `results/table_s1_sample_attrition_final.md`：最终样本损耗；
- `results/table_s2_robustness.md`：稳健性结果；
- `results/completion_audit.csv`：逐项完成性审计；
- `results/artifact_manifest.csv`：最终产物哈希。
