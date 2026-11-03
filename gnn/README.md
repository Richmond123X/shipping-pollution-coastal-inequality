# GNN 排放网络预测框架（研究 2，方案第五章）

**状态：框架搭建完成，未训练**（按计划：评价指标对比模块 MAE/RMSE/MAPE 暂未实现；
训练待 Part B 月度数据在 AIS 机器上产出后，在 GPU 服务器执行）。

## 结构

| 文件 | 职责 |
|---|---|
| `dataset.py` | 读 `results_monthly/` 月度城市排放 → 张量 `[84, N, 12]`（4 船型×SO2/PM/NOx）+ 辅助特征（月份 sin/cos、SECA/NECA 月度虚拟变量）；z-score（仅训练段统计量，持久化反变换）；滑窗 12→12；缺测 mask |
| `adjacency.py` | 月度综合边权（5:2:1）→ 训练期均值 → log1p；DCRNN 双向扩散 / GWNet 对称归一化支撑 |
| `models/dcrnn.py` | 双向扩散卷积 + GRU（Li 2018） |
| `models/gwnet.py` | 自适应邻接 + 时间扩张因果卷积 + skip（Wu 2019） |
| `models/agcrn.py` | NAPL 节点自适应参数 + 数据自适应图 + GRU（Bai 2020），无需预定义图 |
| `models/heads.py` | 共享编码器 → **12 个并行任务头**；12 任务 Masked MAE 求和损失 |
| `baselines/` | HA（同月历史均值）/ 逐城市 ARIMA（SARIMAX 12 期季节）/ 逐城市 LSTM |
| `train.py` | 训练骨架：Adam(1e-3, wd 1e-5)、ReduceLROnPlateau(10)、batch 32、≤200 epoch、早停 20、梯度裁剪、特征扰动增强、checkpoint |
| `mc_dropout.py` | 推断保持 dropout、100 次采样 → 均值/方差 |
| `aggregate_predictions.py` | 月度预测 → 5:2:1 加权 → 船型加和 → 年度(2026/2027) ±95%CI → `f_input_2026_2027.csv`（研究 3 F 项输入） |

实现为**纯 PyTorch 稠密邻接**（N≈1500 可行），不依赖 torch-geometric。
超参集中在根 `config.yaml: gnn`；数据划分 2019.1-2024.12 / 2025.1-6 / 2025.7-12。

## 冒烟检查（本机已通过）

```
python tests/test_shapes.py   # 4 模型前向 shape + 梯度 + MC Dropout
```

## 后续使用顺序

1. AIS 机器跑完 84 个月 Part B 输出 → 拷回/挂载 `results_monthly/`
2. `python train.py --model dcrnn|gwnet|agcrn|lstm`（GPU）
3. （待补）评价模块跑 holdout 测试集选主模型
4. 全量重训 → 推 2026-2027 → `aggregate_predictions.py` 产出 F 项
