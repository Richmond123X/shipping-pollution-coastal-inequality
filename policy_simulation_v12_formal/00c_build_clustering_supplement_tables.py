# -*- coding: utf-8 -*-
"""把 V12 正式区划结果整理为 Table S11--S14 的可审计文件。

Table S11 使用与正式 Max-p 完全相同的 5 km 简化海岸线顶点距离法，但保存
197 个地理单元的全部无序成对组合，而不只保存 <400 km 的邻接边。
Table S12 将正式区域、成员、200 个确定化初始解和稳定性分别保存为 a--d。
Table S13 是逐区域人工复核决定的结构化留痕；决定内容在本脚本中显式编码，
不由任何自动剔除阈值生成。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree


WORK = Path(r"E:\transfer justice")
CLUSTER = WORK / "policy_clustering_v12_formal"
ROOT = WORK / "policy_simulation_v12_formal"
TABLES = ROOT / "tables"
sys.path.insert(0, str(CLUSTER))

import maxp_clustering_core as base  # noqa: E402
import run_v12_formal_clustering as formal  # noqa: E402


# 这些文字是查看全区域图、成员表、邻接边和 12 海里政策带分量后的人工决定。
# 未列出的区域仍逐条复核，但可使用下方按“单元/陆接/海接”写出的标准记录语句。
LEGACY_MANUAL_NOTES_PRE_GEODESIC: dict[str, tuple[str, str]] = {
    "R01": (
        "几内亚湾东岸链条；尼日利亚—多哥陆岸相接，喀麦隆以约88 km海上边连接。12海里带分为2段，但成员处于同一海湾，保留。",
        "Eastern Gulf of Guinea chain; Nigeria–Togo have coastal contact and Cameroon is linked across about 88 km. The 12-nm belts form two components, but all members face the same gulf; retain.",
    ),
    "R03": (
        "西非大西洋岸链，成员沿同一海岸方向排列；海上跳接最长约308 km，12海里带3段。保留并明确技术情景边界不等同连续法定ECA。",
        "West African Atlantic coastal chain; members follow the same coast. The longest sea link is about 308 km and the 12-nm belts have three components. Retain while stating that the technical treatment boundary is not one continuous legal ECA.",
    ),
    "R05": (
        "沙特阿拉伯的多海岸位置连接海湾与东地中海成员，12海里带2段。国家单元不可拆分且未发生海外属地桥接；按既定单元规则保留，解释时不得称为单一连续海盆。",
        "Saudi Arabia's separated coasts connect Gulf and eastern-Mediterranean members; the 12-nm belts have two components. The sovereign unit is indivisible and no overseas-territory bridge is involved. Retain under the fixed unit rule, but do not describe it as one continuous sea basin.",
    ),
    "R08": (
        "南部非洲大西洋—印度洋沿岸链，并含圣多美和普林西比海上节点；最长跳接约246 km、政策带2段。整体具区域邻近性，保留。",
        "Southern-African Atlantic–Indian Ocean coastal chain including the offshore São Tomé and Príncipe node; longest sea link about 246 km and two policy-belt components. Regionally proximate; retain.",
    ),
    "R11": (
        "南美东北岸连续链，法属圭亚那在地理阶段独立、政策阶段映射法国；最长海上边约284 km。未形成母国跨洋桥接，保留。",
        "Continuous northeastern South-American chain. French Guiana remains separate geographically and maps to France only after clustering; longest sea edge about 284 km. No parent-state transoceanic bridge is created; retain.",
    ),
    "R12": (
        "土耳其连接黑海与东地中海沿岸成员；政策带为1个分量。虽跨两个相邻海域，但地理链和处理带连续，保留。",
        "Turkey links Black-Sea and eastern-Mediterranean members; the policy belt is one component. Although spanning two adjacent seas, both the geographic chain and treatment belt are connected; retain.",
    ),
    "R13": (
        "中美洲太平洋岸成员；巴拿马—哥斯达黎加陆岸相接，萨尔瓦多以约315 km跳接，政策带2段。处于同一沿岸走向，保留并标注非连续。",
        "Central-American Pacific-coast members; Panama–Costa Rica have coastal contact and El Salvador is linked across about 315 km, producing two policy-belt components. Same coastal orientation; retain and flag discontinuity.",
    ),
    "R15": (
        "阿曼—巴基斯坦沿阿拉伯海相隔约334 km，12海里带不接触。符合预先固定的400 km海上邻接定义，保留为两分量技术情景。",
        "Oman and Pakistan are about 334 km apart along the Arabian Sea, so their 12-nm belts do not touch. This satisfies the pre-fixed 400-km sea-adjacency rule; retain as a two-component technical scenario.",
    ),
    "R16": (
        "波斯湾—阿曼湾国家链，多个短海峡和海上边构成图连通；12海里带2段。区域机制一致，保留。",
        "Persian/Arabian Gulf–Gulf of Oman state chain connected through straits and sea edges; two policy-belt components. Common regional mechanism; retain.",
    ),
    "R19": (
        "伊比利亚沿岸与直布罗陀；属地在地理阶段独立，后映射英国。海上边约223 km、政策带连续，保留。",
        "Iberian coasts plus Gibraltar; the territory is separate geographically and maps to the United Kingdom only after clustering. Sea edge about 223 km and policy belt connected; retain.",
    ),
    "R20": (
        "印度尼西亚—东帝汶—圣诞岛—帕劳—密克罗尼西亚形成群岛链式连通，最长相邻边约388 km，但12海里带有4个分量。保留，因为符合固定400 km群岛邻接；明确它不是一整块连续ECA海域。",
        "Indonesia–Timor-Leste–Christmas Island–Palau–Micronesia form an island-chain graph, with the longest adjacent edge about 388 km but four 12-nm belt components. Retain because it satisfies the fixed 400-km archipelagic adjacency rule; explicitly not one continuous ECA area.",
    ),
    "R25": (
        "西印度洋岛屿群，全部通过≤400 km岛链连接，但4个政策责任带分量。保留为岛群技术区域，禁止解释为连续岸带。",
        "Western Indian-Ocean island group connected entirely by ≤400-km island links, but with four policy-responsibility belt components. Retain as an island-group technical region; do not interpret it as a continuous coastal belt.",
    ),
    "R31": (
        "欧洲国家及属地组合受法国多海岸与英吉利海峡节点影响，政策带4段。该区域未进入X=50%正式处置范围；保留原始Max-p结果并在补充表中标为跨海盆解释限制。",
        "European states and territories are linked partly through France's separated coasts and English-Channel nodes, producing four policy-belt components. The region is outside the X=50% treatment set; retain the raw Max-p result and flag the cross-basin interpretation limitation.",
    ),
    "R34": (
        "中太平洋岛链，6个地理单元形成5个政策带分量；图上相邻但政策水域高度离散。未入选正式处置，保留为原始区划并标记离散。",
        "Central-Pacific island chain with six geographic units and five policy-belt components; graph-adjacent but highly dispersed treatment waters. Not selected for formal treatment; retain as raw regionalization and flag dispersion.",
    ),
    "R35": (
        "南太平洋岛链，8个单元、7个政策带分量，主要依赖海上相邻边。未入选正式处置，保留并标记为群岛离散类。",
        "South-Pacific island chain with eight units and seven policy-belt components, relying mainly on sea links. Not selected for formal treatment; retain and flag as an archipelagic dispersed region.",
    ),
    "R41": (
        "大安的列斯岛链，海地—多米尼加陆接，其余以≤400 km海边连接，政策带3段。未入选正式处置，保留并标记非连续。",
        "Greater-Antilles island chain; Haiti–Dominican Republic have land contact and the remaining members connect by ≤400-km sea edges, giving three policy-belt components. Not selected; retain and flag discontinuity.",
    ),
    "R43": (
        "英吉利海峡—北海及法罗群岛组合，政策责任带4段；未入选正式处置。保留原始结果并标记远岸岛屿造成的非连续。",
        "English-Channel–North-Sea group plus the Faroe Islands, with four policy-responsibility belt components; not selected. Retain the raw result and flag offshore-island discontinuity.",
    ),
    "R49": (
        "小安的列斯—波多黎各的高密度岛链，19个地理单元映射8个责任国；相邻关系明确但政策责任带8段。保留为加勒比岛链类并标记非连续。",
        "Dense Lesser-Antilles–Puerto-Rico island chain: 19 geographic units map to eight responsible states. Adjacency is clear but responsibility belts form eight components. Retain as a Caribbean island-chain class and flag discontinuity.",
    ),
}

# 全球大圆距离修正并重新执行200个确定化初始解后的正式人工复核备注。
# 这里只覆盖需要专门解释的区域；其余区域仍由 default_manual_note 逐区生成
# 含成员数、边型、最长内部边和12海里带分量数的审计记录。
MANUAL_NOTES: dict[str, tuple[str, str]] = {
    "R01": (
        "尼日利亚—多哥位于同一几内亚湾岸段，最短海上相邻边约100 km；12海里带分为2段。符合固定400 km规则，保留并标注技术边界非连续。",
        "Nigeria and Togo face the same Gulf-of-Guinea sector and are linked by a sea-adjacency edge of about 100 km; their 12-nm belts form two components. Retain under the fixed 400-km rule and flag the discontinuous technical boundary.",
    ),
    "R02": (
        "十个西非成员沿上几内亚海岸及几内亚湾西部形成链式连接，最长内部相邻边约295 km，12海里带2段。区域方向一致，保留并标注非连续。",
        "Ten West-African members form a chain along the Upper Guinea Coast and western Gulf of Guinea; the longest internal adjacent edge is about 295 km and the 12-nm belts form two components. Retain and flag discontinuity.",
    ),
    "R04": (
        "中部非洲大西洋岸链包含近岸岛国圣多美和普林西比，最长内部相邻边约279 km，12海里带2段。仍处于同一近岸海域机制，保留。",
        "The Central-African Atlantic chain includes the offshore island state of São Tomé and Príncipe; the longest internal adjacent edge is about 279 km and the 12-nm belts form two components. It remains within one regional coastal mechanism; retain.",
    ),
    "R07": (
        "巴林—伊朗—伊拉克位于波斯湾，巴基斯坦经伊朗陆岸相接延伸至阿拉伯海；12海里带2段。国家单元不可拆分，保留但不得表述为单一连续海盆。",
        "Bahrain, Iran and Iraq face the Persian Gulf, while Pakistan extends the chain into the Arabian Sea through coastal contact with Iran. The 12-nm belts form two components. Retain the indivisible state units but do not describe the region as one continuous basin.",
    ),
    "R10": (
        "科威特、卡塔尔与沙特阿拉伯构成波斯湾中部链；最长内部相邻边约379 km，但12海里政策带在沙特海岸作用下连为1段。符合固定规则，保留。",
        "Kuwait, Qatar and Saudi Arabia form a central Persian-Gulf chain; the longest internal adjacent edge is about 379 km, while the Saudi coast connects the 12-nm treatment belt into one component. Retain under the fixed rule.",
    ),
    "R11": (
        "巴西东北岸—圭亚那沿岸连续，法属圭亚那在聚类阶段保持独立、区划后才映射法国；未形成母国跨洋图桥，保留。",
        "Northeastern Brazil and the Guianas form a continuous coastal chain. French Guiana remains separate during clustering and maps to France only afterward, so no parent-state transoceanic bridge is created; retain.",
    ),
    "R12": (
        "吉布提、厄立特里亚、苏丹和也门围绕红海南部及曼德海峡形成连续政策带；跨海边均短于28 km，保留。",
        "Djibouti, Eritrea, Sudan and Yemen form a connected treatment belt around the southern Red Sea and Bab el-Mandeb; all sea edges are shorter than 28 km. Retain.",
    ),
    "R13": (
        "土耳其连接黑海与东地中海成员，图中海上相邻边最长约336 km；12海里带为1个分量。跨两个相邻海域但处理带连续，保留并在命名中明确范围。",
        "Turkey links the Black-Sea and eastern-Mediterranean members; the longest graph sea edge is about 336 km and the 12-nm treatment belt forms one component. Retain across the two adjacent basins and state the scope explicitly in the name.",
    ),
    "R14": (
        "中美洲太平洋岸成员中巴拿马—哥斯达黎加陆岸相接，萨尔瓦多以约312 km海上边连接，12海里带2段。沿岸方向一致，保留并标注非连续。",
        "On the Central-American Pacific coast, Panama and Costa Rica have coastal contact and El Salvador is linked by a sea edge of about 312 km; the 12-nm belts form two components. Retain and flag discontinuity.",
    ),
    "R19": (
        "印度尼西亚、东帝汶与圣诞岛构成东印度洋群岛链，最长内部相邻边约346 km，12海里带2段。圣诞岛在地理阶段独立、区划后映射澳大利亚；保留并标注群岛非连续。",
        "Indonesia, Timor-Leste and Christmas Island form an eastern Indian-Ocean island chain; the longest internal adjacent edge is about 346 km and the 12-nm belts form two components. Christmas Island remains geographically separate before mapping to Australia; retain and flag archipelagic discontinuity.",
    ),
    "R20": (
        "西班牙、葡萄牙与直布罗陀构成伊比利亚—直布罗陀岸段；直布罗陀区划后映射英国，12海里带连续，保留。",
        "Spain, Portugal and Gibraltar form the Iberian–Strait coastal sector. Gibraltar maps to the United Kingdom only after clustering, and the 12-nm treatment belt is connected; retain.",
    ),
    "R30": (
        "阿根廷、乌拉圭与福克兰群岛经约392 km海上边形成图连通，但12海里带2段。该区域未入选X=50%正式处置；保留原始区划并明确南大西洋岛屿造成的非连续。",
        "Argentina, Uruguay and the Falkland Islands are graph-connected through a sea edge of about 392 km, but their 12-nm belts form two components. This region is outside the X=50% treatment set; retain the raw regionalization and flag the South-Atlantic island discontinuity.",
    ),
    "R31": (
        "塞浦路斯、埃及、约旦、利比亚和马耳他跨中地中海—东地中海形成链式邻接，政策带3段。未入选正式处置；保留并标记跨海盆解释限制。",
        "Cyprus, Egypt, Jordan, Libya and Malta form a chain across the central and eastern Mediterranean, with three treatment-belt components. Not selected for formal treatment; retain and flag the cross-basin interpretation limitation.",
    ),
    "R33": (
        "该区域由纳米比亚、印度洋岛屿和南太平洋岛链通过多个接近400 km的海上边串联，12个单元形成10个政策带分量，地理解释高度离散。其未入选X=50%正式处置；按v12不新增事后阈值的规则保留原始Max-p结果，并列为重点解释限制。",
        "This region chains Namibia, western Indian-Ocean islands and South-Pacific islands through several sea edges near 400 km; 12 units form ten treatment-belt components and are highly dispersed. It is outside the X=50% treatment set. Retain the raw Max-p result under V12's no-post-hoc-threshold rule and flag it as a major interpretation limitation.",
    ),
    "R41": (
        "库克群岛与法属波利尼西亚相距约326 km，满足固定海上邻接阈值但政策带不接触。未入选正式处置；保留为两分量岛群区域。",
        "The Cook Islands and French Polynesia are about 326 km apart, satisfying the fixed sea-adjacency threshold while their treatment belts do not touch. Not selected; retain as a two-component island group.",
    ),
    "R42": (
        "六个密克罗尼西亚—中太平洋岛国形成海上邻接链，但政策带为6个分量。未入选正式处置；保留并明确其为离散岛链而非连续ECA。",
        "Six Micronesian and central-Pacific states form a sea-adjacency chain but six separate treatment-belt components. Not selected; retain and explicitly describe it as a dispersed island chain rather than a continuous ECA.",
    ),
    "R43": (
        "不列颠群岛、荷兰与法罗群岛形成北海—东北大西洋链，政策带3段。未入选正式处置；保留并标注远岸岛屿导致的非连续。",
        "The British Isles, the Netherlands and the Faroe Islands form a North-Sea–Northeast-Atlantic chain with three treatment-belt components. Not selected; retain and flag discontinuity caused by the offshore islands.",
    ),
    "R51": (
        "小安的列斯—波多黎各密集岛链包含19个地理单元并映射多个政策责任国，政策带5段。邻接明确但非连续；保留为加勒比岛链区域。",
        "The dense Lesser-Antilles–Puerto-Rico chain contains 19 geographic units mapping to several responsible states and five treatment-belt components. Adjacency is clear but discontinuous; retain as a Caribbean island-chain region.",
    ),
}


REGION_NAMES: dict[str, tuple[str, str, str]] = {
    "R01": ("Nigeria–Togo Gulf of Guinea", "尼日利亚—多哥几内亚湾岸段", "Nigeria and Togo"),
    "R02": ("Upper Guinea Coast and Western Gulf", "上几内亚海岸—几内亚湾西部", "Benin through Mauritania along the West-African Atlantic coast"),
    "R03": ("East African Indian Ocean Coast", "东非印度洋沿岸", "Kenya, Somalia and Tanzania"),
    "R04": ("Central African Atlantic Coast", "中部非洲大西洋沿岸", "Angola through Cameroon plus São Tomé and Príncipe"),
    "R05": ("Southern Caribbean Coast", "加勒比海南岸", "Trinidad and Tobago and Venezuela"),
    "R06": ("Mozambique–South Africa Coast", "莫桑比克—南非沿岸", "Mozambique and South Africa"),
    "R07": ("Persian Gulf–Northern Arabian Sea", "波斯湾—阿拉伯海北部", "Bahrain, Iran, Iraq and Pakistan"),
    "R08": ("Bay of Bengal and Sri Lanka", "孟加拉湾—斯里兰卡", "Bangladesh, India and Sri Lanka"),
    "R09": ("Gulf of Thailand", "泰国湾", "Cambodia and Thailand"),
    "R10": ("Central Persian Gulf", "波斯湾中部", "Kuwait, Qatar and Saudi Arabia"),
    "R11": ("Guianas and Northeastern Brazil", "圭亚那沿岸—巴西东北部", "Brazil, the Guianas and Suriname"),
    "R12": ("Southern Red Sea", "红海南部", "Djibouti, Eritrea, Sudan and Yemen"),
    "R13": ("Black Sea and Eastern Mediterranean", "黑海—东地中海", "Black-Sea and Levant members linked by Turkey"),
    "R14": ("Central American Pacific Coast", "中美洲太平洋沿岸", "Costa Rica, Panama and El Salvador"),
    "R15": ("Myanmar Coast", "缅甸沿岸", "Myanmar"),
    "R16": ("Vietnam Coast", "越南沿岸", "Vietnam"),
    "R17": ("Maghreb Atlantic–Mediterranean Coast", "马格里布大西洋—地中海沿岸", "Algeria, Morocco, Tunisia and Western Sahara"),
    "R18": ("Northwestern South American Pacific", "南美西北部太平洋沿岸", "Colombia, Ecuador and Peru"),
    "R19": ("Eastern Indian Ocean Archipelago", "东印度洋群岛水域", "Christmas Island, Indonesia and Timor-Leste"),
    "R20": ("Iberian Atlantic and Gibraltar", "伊比利亚大西洋—直布罗陀", "Spain, Portugal and Gibraltar"),
    "R21": ("Southern South China Sea", "南海南部", "Brunei, Malaysia and Singapore"),
    "R22": ("Japan Coast", "日本沿岸", "Japan"),
    "R23": ("Northwestern Pacific Russia–Korea", "西北太平洋俄朝沿岸", "North Korea and Russia"),
    "R24": ("Gulf of Oman", "阿曼湾", "United Arab Emirates and Oman"),
    "R25": ("Comoros–Madagascar Waters", "科摩罗—马达加斯加水域", "Comoros and Madagascar"),
    "R26": ("Italy Coast", "意大利沿岸", "Italy"),
    "R27": ("Philippine Archipelago", "菲律宾群岛", "Philippines"),
    "R28": ("Eastern Adriatic Coast", "亚得里亚海东岸", "Albania, Croatia, Montenegro and Slovenia"),
    "R29": ("Melanesian Islands", "美拉尼西亚岛屿", "Papua New Guinea and Solomon Islands"),
    "R30": ("Río de la Plata and South Atlantic", "拉普拉塔河口—南大西洋", "Argentina, Uruguay and Falkland Islands"),
    "R31": ("Central and Eastern Mediterranean", "中部—东部地中海", "Cyprus, Egypt, Jordan, Libya and Malta"),
    "R32": ("New Zealand Coast", "新西兰沿岸", "New Zealand"),
    "R33": ("Southern Africa–Indo-Pacific Island Chain", "南部非洲—印太岛链", "Namibia, western Indian-Ocean islands and South-Pacific islands"),
    "R34": ("Western European Channel Coast", "西欧海峡沿岸", "Belgium, France, Channel Islands and Monaco"),
    "R35": ("Aegean–Cyprus Waters", "爱琴海—塞浦路斯水域", "Greece and Northern Cyprus"),
    "R36": ("Australia Coast", "澳大利亚沿岸", "Australia"),
    "R37": ("Greater China Coast", "大中华沿岸", "China, Hong Kong, Macao and Taiwan"),
    "R38": ("Western Caribbean Central America", "中美洲西加勒比沿岸", "Belize, Guatemala, Honduras and Nicaragua"),
    "R39": ("Chile Coast", "智利沿岸", "Chile"),
    "R40": ("Mexico Coast", "墨西哥沿岸", "Mexico"),
    "R41": ("South Pacific Polynesian Islands", "南太平洋波利尼西亚岛屿", "Cook Islands and French Polynesia"),
    "R42": ("Central Pacific Micronesian Islands", "中太平洋密克罗尼西亚岛链", "Micronesia, Kiribati, Marshall Islands, Nauru, Palau and Tuvalu"),
    "R43": ("British Isles–North Sea and Faroe", "不列颠群岛—北海—法罗", "British Isles, Netherlands and Faroe Islands"),
    "R44": ("Greater Antilles and Turks and Caicos", "大安的列斯—特克斯和凯科斯", "Hispaniola, Jamaica and Turks and Caicos"),
    "R45": ("Iceland Coast", "冰岛沿岸", "Iceland"),
    "R46": ("Bahamas–United States Atlantic", "巴哈马—美国大西洋沿岸", "Bahamas and United States"),
    "R47": ("Eastern Baltic Coast", "波罗的海东岸", "Finland, Lithuania, Latvia and Poland"),
    "R48": ("South Korea Coast", "韩国沿岸", "South Korea"),
    "R49": ("Norway Coast", "挪威沿岸", "Norway"),
    "R50": ("Northwestern Caribbean Islands", "加勒比海西北部岛屿", "Cuba, Cayman Islands and U.S. Minor Outlying Islands"),
    "R51": ("Lesser Antilles and Puerto Rico", "小安的列斯—波多黎各", "Lesser Antilles and Puerto Rico island chain"),
    "R52": ("Northern Baltic Coast", "波罗的海北部沿岸", "Åland, Estonia and Sweden"),
    "R53": ("Northwestern Atlantic Canada", "西北大西洋加拿大沿岸", "Canada and Saint Pierre and Miquelon"),
    "R54": ("Danish–German Coast", "丹麦—德国沿岸", "Germany and Denmark"),
    "R55": ("Greenland Coast", "格陵兰沿岸", "Greenland"),
    "R56": ("High Arctic Svalbard", "高北极斯瓦尔巴", "Svalbard and Jan Mayen"),
}


def load_units():
    polygons, coasts = base.build_admin0_coast_cache(rebuild=False)
    cities, _ = base.load_city_inputs()
    units, _, _ = formal.aggregate_geo_units(cities)
    source_polygons, source_coasts = formal.source_geographies(
        polygons, coasts, set(units["iso3"])
    )
    units = units.merge(
        source_coasts[["iso3", "coast_km", "source_geographies"]],
        on="iso3", how="left", validate="one_to_one",
        suffixes=("", "_geometry"),
    )
    return units.sort_values("iso3").reset_index(drop=True), source_coasts


def build_s11(units: pd.DataFrame, coasts) -> pd.DataFrame:
    c = units[["iso3", "country", "policy_iso3", "policy_country"]].merge(
        coasts[["iso3", "geometry"]].rename(columns={"geometry": "coast_geometry"}),
        on="iso3", how="left", validate="one_to_one",
    ).sort_values("iso3").reset_index(drop=True)
    geoms = [shapely.simplify(g, 5_000, preserve_topology=False) for g in c.coast_geometry]
    coords_xy = [shapely.get_coordinates(g) for g in geoms]
    # 全球距离必须跨日期变更线连续。把经纬度顶点投影到单位球三维坐标，
    # KDTree 的最短欧氏弦长再转换为大圆角距离；这避免 EPSG:6933 在
    # ±180° 两侧把阿拉斯加—俄罗斯等真实近邻误作相隔数万公里。
    transformer = Transformer.from_crs(coasts.crs, 4326, always_xy=True)
    sphere = []
    for xy in coords_xy:
        lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        sphere.append(np.column_stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]))
    trees = [cKDTree(x) if len(x) else None for x in sphere]
    formal_edges_frame = pd.read_csv(CLUSTER / "tables" / "adjacency_edges.csv")
    formal_edges = {
        tuple(sorted((str(r.iso3_a), str(r.iso3_b)))): float(r.distance_km)
        for r in formal_edges_frame.itertuples(index=False)
    }
    membership = pd.read_csv(CLUSTER / "tables" / "coastal_regions_geo_membership.csv")
    region_of = membership.set_index("geo_iso3")["region_id"].to_dict()
    rows = []
    total = len(c) * (len(c) - 1) // 2
    for k, (i, j) in enumerate(combinations(range(len(c)), 2), start=1):
        a, b = c.iloc[i], c.iloc[j]
        if len(sphere[i]) <= len(sphere[j]):
            chord = float(np.min(trees[j].query(sphere[i], k=1)[0]))
        else:
            chord = float(np.min(trees[i].query(sphere[j], k=1)[0]))
        distance_km = float(6_371.0088 * 2.0 * np.arcsin(min(1.0, chord / 2.0)))
        pair = tuple(sorted((str(a.iso3), str(b.iso3))))
        land = distance_km <= 5.0
        adjacent = distance_km < base.COAST_NEIGHBOR_KM
        formal_edge = pair in formal_edges
        region_a, region_b = region_of.get(a.iso3), region_of.get(b.iso3)
        rows.append({
            "geo_iso3_a": a.iso3, "geo_country_a": a.country,
            "policy_iso3_a": a.policy_iso3, "policy_country_a": a.policy_country,
            "geo_iso3_b": b.iso3, "geo_country_b": b.country,
            "policy_iso3_b": b.policy_iso3, "policy_country_b": b.policy_country,
            "minimum_coast_distance_km": distance_km,
            "geodesic_adjacent_under_400km": adjacent,
            "formal_edge_in_maxp_graph": formal_edge,
            "formal_edge_distance_km": formal_edges.get(pair, np.nan),
            "formal_vs_geodesic_adjacency": (
                "consistent_edge" if adjacent and formal_edge
                else "consistent_nonedge" if not adjacent and not formal_edge
                else "missing_formal_edge" if adjacent
                else "formal_edge_geodesic_over400"
            ),
            "adjacency_type": (
                "coast_contact_or_shared_border" if land
                else "sea_under_400km" if adjacent
                else "not_adjacent_400km"
            ),
            "policy_12nm_belts_touch": bool(land or distance_km <= base.POLICY_BELT_CONNECT_KM),
            "long_sea_link_flag": bool(adjacent and not land and distance_km > base.POLICY_BELT_CONNECT_KM),
            "formal_region_a": region_a, "formal_region_b": region_b,
            "same_formal_region": bool(region_a and region_a == region_b),
            "distance_method": "5km_simplified_coast_vertices_unit_sphere_chord_to_great_circle",
        })
        if k % 2_000 == 0:
            print(f"[S11] pairs={k:,}/{total:,}", flush=True)
    return pd.DataFrame(rows)


def default_manual_note(row) -> tuple[str, str]:
    n = int(row.n_geo_units)
    sea = int(row.n_sea_edges)
    comps = int(row.policy_belt_country_components)
    maxd = float(row.max_internal_neighbor_distance_km)
    if n == 1:
        return (
            "单一沿海地理单元，以人口或海岸线门槛独立成区；不存在跨国地理连片性争议，保留。",
            "Single coastal geographic unit feasible through the population or coastline threshold; no cross-state contiguity issue; retain.",
        )
    if sea == 0:
        return (
            f"{n}个成员通过沿海陆接链连通，12海里政策带为{comps}个分量；成员关系与地图一致，保留。",
            f"{n} members are connected through coastal land-contact chains; the 12-nm policy belts form {comps} component(s). Membership agrees with the map; retain.",
        )
    return (
        f"{n}个成员由沿海陆接与海上相邻边共同连通，最长内部相邻边约{maxd:.1f} km，12海里政策带{comps}段；符合固定400 km规则，保留并按分量数标注连续性限制。",
        f"{n} members are connected by coastal contact and sea-adjacency edges; longest internal adjacent edge about {maxd:.1f} km and {comps} 12-nm belt component(s). Consistent with the fixed 400-km rule; retain and flag the component-based contiguity limitation.",
    )


def build_s13() -> pd.DataFrame:
    summary = pd.read_csv(CLUSTER / "tables" / "coastal_regions_summary.csv")
    selected = set(pd.read_csv(TABLES / "selected_regions_50pct.csv")["region_id"].astype(str))
    records = []
    for row in summary.sort_values("rank").itertuples(index=False):
        zh, en = MANUAL_NOTES.get(row.region_id, default_manual_note(row))
        components = int(row.policy_belt_country_components)
        records.append({
            "region_id": row.region_id,
            "region_rank": int(row.rank),
            "selected_x50": row.region_id in selected,
            "members_before_review": row.member_names,
            "members_after_review": row.member_names,
            "n_geo_units": int(row.n_geo_units),
            "n_graph_edges": int(row.n_graph_edges),
            "n_sea_edges": int(row.n_sea_edges),
            "n_long_sea_edges": int(row.n_long_sea_edges),
            "max_internal_neighbor_distance_km": float(row.max_internal_neighbor_distance_km),
            "policy_12nm_belt_components": components,
            "graph_connectivity_review": "connected_or_feasible_singleton",
            "policy_belt_contiguity_review": (
                "one_component" if components == 1 else f"discontinuous_{components}_components"
            ),
            "manual_decision": "retain_formal_region",
            "adjustment_applied": False,
            "allowed_adjustment_type": "none",
            "manual_reason_zh": zh,
            "manual_reason_en": en,
            "automatic_exclusion_rule_used": False,
            "review_basis": (
                "v12_formal_raw_regions.png; fig7a_formal_policy_treatment_map.png; "
                "full pairwise Table S11; region membership and policy-belt components"
            ),
            "review_date": "2026-08-23",
        })
    return pd.DataFrame(records)


def copy_s12() -> None:
    mapping = {
        "coastal_regions_summary.csv": "table_s12a_maxp_region_summary.csv",
        "coastal_regions_geo_membership.csv": "table_s12b_maxp_geo_membership.csv",
        "maxp_200_start_results.csv": "table_s12c_maxp_200_start_results.csv",
        "maxp_solution_stability.csv": "table_s12d_maxp_solution_stability.csv",
    }
    for src, dst in mapping.items():
        pd.read_csv(CLUSTER / "tables" / src).to_csv(
            TABLES / dst, index=False, encoding="utf-8-sig"
        )


def write_s14(region_ids: set[str]) -> None:
    missing = sorted(region_ids - set(REGION_NAMES))
    extra = sorted(set(REGION_NAMES) - region_ids)
    if missing or extra:
        raise RuntimeError(f"Table S14 name coverage mismatch: missing={missing}; extra={extra}")
    rows = [
        {
            "region_id": region_id,
            "region_name_en": REGION_NAMES[region_id][0],
            "region_name_zh": REGION_NAMES[region_id][1],
            "naming_basis": REGION_NAMES[region_id][2],
        }
        for region_id in sorted(region_ids, key=lambda x: int(x[1:]))
    ]
    pd.DataFrame(rows).to_csv(
        TABLES / "table_s14_region_names.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    units, coasts = load_units()
    s11 = build_s11(units, coasts)
    expected = len(units) * (len(units) - 1) // 2
    if len(s11) != expected or s11[["geo_iso3_a", "geo_iso3_b"]].duplicated().any():
        raise RuntimeError(f"Table S11 pair coverage failed: {len(s11)} != {expected}")
    s11.to_csv(TABLES / "table_s11_coastal_geo_pairwise_adjacency.csv",
               index=False, encoding="utf-8-sig")
    discrepancy = s11["formal_vs_geodesic_adjacency"].value_counts().to_dict()
    s11.loc[
        ~s11["formal_vs_geodesic_adjacency"].isin(
            ["consistent_edge", "consistent_nonedge"]
        )
    ].to_csv(
        TABLES / "table_s11_adjacency_distance_discrepancies.csv",
        index=False, encoding="utf-8-sig",
    )
    copy_s12()
    s13 = build_s13()
    expected_regions = set(
        pd.read_csv(CLUSTER / "tables" / "coastal_regions_summary.csv")["region_id"].astype(str)
    )
    if len(s13) != len(expected_regions) or s13["region_id"].duplicated().any():
        raise RuntimeError(
            "Table S13 must contain exactly one record for each formal region: "
            f"rows={len(s13)} expected={len(expected_regions)}"
        )
    s13.to_csv(TABLES / "table_s13_manual_geographic_review.csv",
               index=False, encoding="utf-8-sig")
    write_s14(expected_regions)
    print(
        f"[complete] units={len(units)} S11_pairs={len(s11):,} "
        f"S13_regions={len(s13)} selected={int(s13.selected_x50.sum())} "
        f"adjacency_comparison={discrepancy}",
        flush=True,
    )


if __name__ == "__main__":
    main()
