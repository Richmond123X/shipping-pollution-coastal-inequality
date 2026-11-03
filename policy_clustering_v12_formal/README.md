# V12 政策模拟正式聚类结果

本目录是后续政策模拟使用的唯一沿海区域聚类版本。前期实验目录和对照结果不属于正式输入。

## 目录

- `run_v12_formal_clustering.py`：正式执行入口，固定 200 个随机初始解。
- `maxp_clustering_core.py`：精确 OR Max-p、邻接构造与绘图公共函数；不单独执行。
- `cache/`：正式执行所需的 GADM 沿海几何缓存。
- `tables/`：区域成员、稳定性、累计人口截断及前30%/50%责任国名单。
- `figures/`：正式区域图、排序图及前30%/50%空间分布图，仅保留 PNG。
- `report/formal_clustering_summary.md`：正式结果摘要和解释边界。
- `formal_manifest.json`：输入、参数、运行时间及结果规模。

正式执行命令：

```powershell
python "E:\transfer justice\policy_clustering_v12_formal\run_v12_formal_clustering.py"
```

正式口径：197 个地理单元、150 个政策责任国、56 个沿海区域；前30%累计曲线审计点包含16个区域和59个责任国，正式前50%方案包含22个区域和75个责任国，实际覆盖沿海人口54.30%。全球海岸最短距离采用约5 km简化顶点的单位球弦长转大圆距离，以正确处理日期变更线。
