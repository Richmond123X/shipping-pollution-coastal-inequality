# V12政策模拟正式流水线

本目录使用研究方案V12、正式Max-p区域结果和固定规模参数 `X=50%`，是S1–S7政策模拟的唯一正式工作区。

正式处置范围：SPRI排名前22个沿海区域、75个去重政策责任国，实际覆盖全球沿海人口54.30%。

目录约定：

- `inputs/`：正式12海里政策区和外部电网强度表；
- `intermediate/`：AIS政策增量和逐次挂靠级岸电数据；
- `tables/`：Table 6、Table 7及补充表；
- `figures/`：Figure 7各幅独立PNG；
- `report/`：正式结果说明与完整性审计；
- `logs/`：长时间AIS运行日志。

所有情景按“先岸电，后硫含量管制/氮氧化物标准”处理重叠。S5–S7的岸电使用率固定为100%。

数据可行性审计见 `tables/policy_scenario_data_feasibility.csv` 与
`tables/policy_input_readiness.csv`。现有12个月2024年AIS、基线清单、政策区、
港口—城市映射和75国2022年官方电网强度可支持全部S1–S7；但正式重算必须读取
原始单船AIS，因为聚合网络不能恢复逐点12海里归属、船龄/燃料条件或逐次挂靠。
2024年集装箱船GT交叉表覆盖5001/5969个MMSI（83.78%），缺失GT的≥2小时挂靠保留为
“资格不确定”，不以DWT插补，也不静默归为不合格。

政策效果计算年份固定为2024，AIS来源固定为
`F:\SQL_AIS_SingleShip`。X=50%、22个入选区域、75个政策责任国、基于2025锚点
SPRI的区划与排序、S1–S7参数、2019分组切点和2022电网强度均保持不变。

## AIS执行方式（2026-08-23起）

正式流程不再生成逐船轨迹缓存，也不做时间抽样。
`03_run_policy_ais_shards.py` 通过原主程序自带的连续MMSI区间参数调用既有的
`emission_network/emission_network_main.py`，保留原模型使用的全部有效AIS点；
正式任务一次读取全年及两端边界月，每艘船全年连续处理一次，再按月分箱写出。
永久保存的仅是城市/政策区充分统计量和集装箱船逐次挂靠表。AIS读取也沿用
原程序的“工作进程逐船读取数据库”方式，不启用批量预取。数据库损坏、掉盘或单船计算失败均
计入严格质量门。原程序按MMSI顺序每500艘形成一个可恢复年度分片；每个分片
内部仍是全频、14个月窗口和全年连续计算，只增加检查点，不改变计算公式。
一个分片发生真正错误时仍继续尝试其他分片，全部结束后统一报告失败；已经
完整写出的分片自动跳过，外部中断最多重算当前500艘。
注册表中存在但有效AIS点不足3个的船归为“无可计算航段”，不计为程序失败，
MMSI与原因写入各船型的 `nodata_log_*.txt` 供后续核实。

## 一键运行

在PowerShell中只需执行：

```powershell
& "E:\transfer justice\policy_simulation_v12_formal\run_policy_2024.cmd"
```

该命令自动执行：2024 GT交叉表 → 输入审查 → 全年全频AIS连续分片计算 → 按月写出与合并
→ S1–S7 → Table 7 → 全部PNG → 正式终检。任何质量门失败
都会停止。中断后重新执行同一命令即可续跑，不会重算已完成的月份和船型。

可选分阶段命令：

```powershell
& "E:\transfer justice\policy_simulation_v12_formal\run_policy_2024.cmd" -Stage prepare
& "E:\transfer justice\policy_simulation_v12_formal\run_policy_2024.cmd" -Stage ais
& "E:\transfer justice\policy_simulation_v12_formal\run_policy_2024.cmd" -Stage post
```
