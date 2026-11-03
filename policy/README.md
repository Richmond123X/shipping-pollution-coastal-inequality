# 研究 4：高风险区域 SECA/NECA 政策模拟

对应方案 v6 第七章。四步流水线：

## 1. 政策区生成（本机可跑）

```
python policy/build_policy_zones.py
```

- 输入：研究 3 SPRI 表（`config.yaml: policy.spri_csv`，需含 node_id、SPRI 两列）、
  协变量面板（node→国家映射）、GADM World.shp
- 筛选：SPRI 降序前 20%（`top_pct`）→ 按国内高风险节点数取前 N 国（`top_n_countries`）；
  `policy.blocs` 可加区域经济体联合单元
- 几何：国家陆地面（aeqd 局地投影）向海缓冲 12 海里，扣除本国+邻国陆地 → 纯海域带
- 输出：`policy/outputs/candidate_policy_zones.gpkg` + `policy_trigger_nodes.csv`
- 研究 3 未产出时可 `--countries CHN,VNM` 直接指定单元调试
- 已知限制：跨反子午线的国家（如 RUS/FJI）aeqd 缓冲不可靠，自动跳过并告警

## 2. 情景排放重算（AIS 机器，评估年 2025）

```
python emission_network/emission_network_main.py --year 2025 --policy-zones policy/outputs/candidate_policy_zones.gpkg
```

（或 config.yaml 设 `policy.enabled: true` 后正常跑。）单遍多情景：基线照常产出，
每月每船型额外写两个 delta 文件（见 emission_network/README.md）。
组合情景无需单算：SECA 只动 SO₂/PM、NECA 只动 NOx，无交互。

## 3. 效果评估（本机可跑）

```
python policy/evaluate_policy_effects.py
```

正式运行默认执行完整性门控：12 个月×配置中的全部船型，只要缺少政策 delta 或
城市基线文件就停止，不写部分年度结果。`--allow-incomplete` 仅供调试，不能用于 Fig.6。

- 输出到 `policy/outputs/evaluation/`：
  - `policy_eval_zone_summary_<year>.csv`：单元×情景×污染物×工况的减排量/区内基线/减排率
  - `policy_eval_city_<year>.csv`：城市级减排 + 人口加权暴露削减 + 脆弱人群受益
  - `policy_eval_equity_<year>.csv`：单元×情景的总减排、暴露削减、触发节点受益份额
- 分维度报告，不合成单一评分；实施成本暂不纳入（方案 7.3）

## 4. 固定基线重算政策情景 SPRI（本机可跑）

完成第 2、3 步后运行：

```
python policy/build_policy_spri_scenarios.py
```

- 用 2025 全年真实 AIS 城市排放替换 `spri_panel.csv` 中的 2025 预测暴露；
- 按 5PM:2SO₂:1NOx 将城市减排量换算为 E1/E2 变化，S 与 AC 维度保持不变；
- 标准化边界与 CRITIC 权重只在基线估计一次，并固定应用到 SECA、NECA、BOTH；
- 假定政策持续，将 2025 城市减排比例延伸到 2026–2027 预测暴露，避免滚动未来项造成虚假反弹；
- 输出 `policy/outputs/evaluation/policy_spri_scenarios_2025.csv`，供 Fig.6c 使用。

## 一键完成并验收 Figure 6

2025 排放主程序全部结束后，可用以下命令依次执行效果评估、情景 SPRI、5 张
Figure 6 独立 PNG，并验证正文目录最终为 34 张 PNG、无非 PNG 文件：

```
python policy/finalize_figure6.py --results-root "E:/transfer justice/results_monthly"
```

该命令不会生成 PPT；PPT 仅在 34 张独立图片完成视觉终检后统一组合。

## 冒烟测试（本机）

```
cd emission_network && python tests/test_policy_smoke.py
```

断言：区内点识别、SECA 0.8 精确削减（0.5%→0.1%）、PM 仿射比削减、
NECA 建造年门槛（<2016 零削减）、城市聚合与月度分箱。
