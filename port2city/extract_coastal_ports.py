# -*- coding: utf-8 -*-
"""Port2City 步骤 1：沿海港口筛选（继承 STEAM 规则）。

从 port_data_clean.xlsx（泊位级，~41,839 行）筛出沿海商业海港并去重到港口级：
  1. Port_Type == "Sea"（剔除内河港）
  2. 剔除离岸/海工设施（名称关键词 + Facility.Type 关键词）
  3. 剔除非港口节点（Anchorage/Strait/SPM/Oil Terminal/Lightering）
  4. 拆出运河条目（Suez/Panama，不进入城市节点）
  5. 泊位级 → 港口级去重，坐标取 Lon_port/Lat_port（参考点）

注意：Ningbo/Zhoushan 等组合港在此阶段保持原名（便于 L1 判定各自城市），
组合发生在节点生成阶段（run_port2city.py 按 config 的 node_merges 执行）。

输出 outputs/coastal_ports.csv：
  Port_Name, Country_Region, Lon_port, Lat_port, n_berths
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shipping_network_functions import (  # noqa: E402
    is_non_port_name, CANAL_PORT_NAMES, CANAL_WAITING_SUBPORTS,
)
from shipping_network_main import _row_is_offshore  # noqa: E402

# ---------------------------------------------------------------------------
# 离岸设施扩展剔除（方案 3.6.3 步骤1"剔除离岸/海工"的补充）：
# 原 STEAM 关键词（FSO/FPSO/OFFSHORE/PLATFORM/RIG/SBM/...）漏掉的两类条目——
# (a) 名称模式：油田/单点系泊浮筒/船对船过驳区(STS≈Lightering)/港界外水域(OPL)
# (b) 人工核定名单：港口点距所属城市面 >10km 且经核实为海上油气装载设施
#     （证据：port_city_mapping dist_km + L3 仲裁记录；审计表见
#       outputs/excluded_offshore_ports.csv）
# ---------------------------------------------------------------------------
import re  # noqa: E402

OFFSHORE_EXTRA_PATTERN = re.compile(
    r"\b(OIL\s*FIELD|OILFIELD|CALM\s+BUOY|CALM|STS|OPL)\b", re.IGNORECASE)

OFFSHORE_FACILITY_PORTS = {
    # 海上油气装载终端/油田（港口点离岸 >10km，逐一核实）
    "Kizomba C Terminal", "Halul Island", "Al Shaheen Terminal",
    "Odudu Terminal", "Ashtart Terminal", "Moudi Terminal",
    "Wandoo Marine Terminal", "Kumul Marine Terminal", "Qua Iboe",
    "Palanca Terminal", "Ardjuna Marine Terminal", "Fateh Terminal",
    "Pennington", "Takula Terminal", "LOOP Terminal", "Brass Terminal",
    "Bahregan Oil Centre (BOC)", "Cinta", "Bima Terminal",
    "Mubarek Terminal", "Malongo Terminal", "Varandey",
    "Rio Cullen Marine Terminal", "Senipah Terminal",
    # 南极科考站（非商业海港，方案 3.6.7 研究范围外）
    "McMurdo Station",
}


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_coastal_ports(port_xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (港口级表, 运河表)。筛选规则与 shipping_network_main.load_port_coordinates 一致。"""
    df = pd.read_excel(port_xlsx)
    n0 = len(df)

    sea_mask = df["Port_Type"].astype(str).str.strip().str.casefold() == "sea"
    df = df[sea_mask].copy()
    print(f"Port_Type==Sea: {len(df)}/{n0}")

    n1 = len(df)
    df = df[~df.apply(_row_is_offshore, axis=1)].copy()
    print(f"剔除离岸/海工: -{n1 - len(df)} → {len(df)}")

    n2 = len(df)
    df = df[~df["Port.Name"].apply(is_non_port_name)].copy()
    print(f"剔除非港口节点: -{n2 - len(df)} → {len(df)}")

    # 扩展离岸剔除：名称模式 + 人工核定名单 + 运河等待子港（运河逻辑单独处理）
    n3 = len(df)
    extra_mask = (
        df["Port.Name"].astype(str).str.contains(OFFSHORE_EXTRA_PATTERN, na=False)
        | df["Port.Name"].isin(OFFSHORE_FACILITY_PORTS)
        | df["Port.Name"].isin(CANAL_WAITING_SUBPORTS)
    )
    excluded = df[extra_mask][["Port.Name", "Country/Region"]].drop_duplicates()
    df = df[~extra_mask].copy()
    print(f"扩展剔除离岸设施/STS/运河子港: -{n3 - len(df)} 行 "
          f"({excluded['Port.Name'].nunique()} 港) → {len(df)}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    excluded.to_csv(os.path.join(out_dir, "excluded_offshore_ports.csv"),
                    index=False, encoding="utf-8-sig")

    canal_mask = df["Port.Name"].isin(CANAL_PORT_NAMES)
    canals = df[canal_mask].copy()
    df = df[~canal_mask].copy()
    print(f"拆出运河条目: {canals['Port.Name'].nunique()} 个运河, 港口剩 {len(df)} 行")

    for col in ("Lon_port", "Lat_port"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Port.Name", "Lon_port", "Lat_port"])
    df = df[df["Lon_port"].between(-180, 180) & df["Lat_port"].between(-90, 90)]

    # 泊位级 → 港口级：Port.Name 分组；参考点取众数坐标（同名港的 Lon_port 应一致，
    # 个别不一致时取出现最多的一组）；Country/Region 取众数
    def _agg(g: pd.DataFrame) -> pd.Series:
        coord = g.groupby(["Lon_port", "Lat_port"]).size().idxmax()
        cr = g["Country/Region"].mode()
        return pd.Series({
            "Country_Region": cr.iloc[0] if len(cr) else "",
            "Lon_port": coord[0],
            "Lat_port": coord[1],
            "n_berths": len(g),
        })

    ports = df.groupby("Port.Name", sort=True).apply(_agg).reset_index()
    ports = ports.rename(columns={"Port.Name": "Port_Name"})
    print(f"港口级去重: {len(ports)} 个沿海港口")

    # 同名港跨国检查（影响 L1 缓存键与 L2 国家过滤）
    dup_cr = df.groupby("Port.Name")["Country/Region"].nunique()
    multi = dup_cr[dup_cr > 1]
    if len(multi):
        print(f"警告: {len(multi)} 个港口名对应多个 Country/Region（取众数）: "
              f"{list(multi.index[:10])}")
    return ports, canals


def main() -> None:
    cfg = load_config()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    ports, canals = extract_coastal_ports(cfg["paths"]["port_xlsx"])

    out_csv = cfg["paths"]["coastal_ports_csv"]
    ports.to_csv(out_csv, index=False, encoding="utf-8-sig")
    canals.to_csv(os.path.join(out_dir, "canal_entries.csv"), index=False, encoding="utf-8-sig")
    print(f"已写出: {out_csv} ({len(ports)} 港)")


if __name__ == "__main__":
    main()
