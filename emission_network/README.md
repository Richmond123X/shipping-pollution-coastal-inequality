# 排放网络（月度 / 城市节点版）

由原季度版改造而来，对应研究方案 v6 第 3.5、3.6.5、5.2 节。原排放计算逻辑（STEAM、berth-first 挂靠、运河处理）未动。
（注：根目录旧季度/港口版 `emission_network_main.py`、`emission_network_functions.py` 已删除，本目录为唯一现行版本。）

## 与季度版的差异

| 项 | 季度版 | 本版 |
|---|---|---|
| 时间尺度 | `<Year>Qn`（3 库拼接） | `YYYYMM`，**按年连续处理(年±1 边界月)再按月分箱**：每船读 14 个月(上年12月+本年12月+下年1月)整段算一次，跨月航段逐点分月，仅输出本年 12 月。消除月度归属的边界低估。`--only-months` 仅过滤输出月份 |
| 节点 | 港口（Port.Name） | **城市**（Port2City Agent 的 node_id）+ 港口级可选保留 |
| 节点排放归属 | 港口 buffer 内 visits | **AIS 点级排放 sjoin 30km 城市集水区**（重叠重复计入，方案 3.6.5），四工况 Berth/Anchored/Manoeuvring/Sea 拆分供 E3 |
| OD 边 | 港口→港口 | 港口边经 `port_city_mapping.csv` 归并为城市→城市，同城剔除 |
| 船型 | 全部 | 4 类商船白名单（Container/General Cargo/Oil Tanker/Bulk Carrier；`--all-ship-types` 关闭） |
| 路径 | 硬编码 | 全部来自根目录 `config.yaml`（命令行可覆盖） |

## 新增输出（results_monthly/<YYYYMM>/<船型>/）

- `city_emissions_<船型>_<YYYYMM>.csv`：node_id, YearMonth, Ship_Type, 每种污染物 `_Berth/_Anchored/_Manoeuvring/_Sea/_Total`（克）。E3 停泊占比 = (_Berth+_Anchored)/_Total
- `city_edge_list_<污染物>_<船型>_<YYYYMM>.csv`：From_City, To_City, Total_Emission
- 港口级旧输出在 `config.yaml: emission_network.keep_port_level_outputs: true` 时保留（SI 用）

## 运行（迁回 AIS 机器后）

1. 改 `config.yaml` 的 `paths.ais_db_dir / laf_xlsx / eca_base` 为该机实际路径；
   把 `port2city/outputs/`（city_catchments_30km.gpkg、port_city_mapping.csv）一并拷贝并更新路径
2. `python emission_network_main.py --year 2024`（断点续跑默认开）

## 政策情景（研究 4，方案 v6 第七章）

`--policy-zones <gpkg>`（或 config.yaml `policy.enabled: true`）启用后，同一遍轨迹处理
额外输出 SECA/NECA 情景减排量（**单遍多情景**：对基线列做精确比例变换，不复制排放公式）：

- `city_policy_delta_<船型>_<YYYYMM>.csv`：node_id × Mode_Class × Zone_ID 的
  D_SO2_SECA / D_PM_SECA / D_NOx_NECA（克，集水区口径）
- `policy_delta_totals_<船型>_<YYYYMM>.csv`：Zone_ID × Mode_Class 的 delta + 区内基线
  B_SO2/B_PM/B_NOx（守恒口径，供区级减排率）

语义：区内燃油硫率 → 0.1%（HFO 换 MDO 油、SFC 同步切换；与现行 ECA 重叠取更严）；
Build_Year ≥ `policy.neca_build_year`(2016) 的船主机 NOx → Tier III EF。政策区生成与
效果评估见根目录 `policy/README.md`。启用政策时断点续跑要求 delta 文件也齐全，
旧产出月份会自动重算。

## 冒烟测试（本机可跑）

```
python tests/make_synthetic_ais.py
python tests/test_policy_smoke.py
```
前者：月度标签 / 城市集水区点级归属 / 停泊航行拆分 / OD 城市映射 / 同城边剔除。
后者：政策区点识别 / SECA 0.8 精确削减 / PM 仿射比 / NECA 建造年门槛 / delta 城市聚合。
