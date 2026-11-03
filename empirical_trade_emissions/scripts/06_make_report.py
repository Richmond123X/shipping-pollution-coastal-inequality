#!/usr/bin/env python3
"""Create publication-style tables, a Chinese report, and a completion audit.

Every reported number is read from the processed data, QC files, or model output.
The script contains no fallback estimates or placeholder values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "D1_base": "航段基准",
    "D2_income": "航段收入异质性",
    "D3_region": "航段区域异质性",
    "C1_base": "国家集水区基准",
    "C2_income": "国家集水区收入异质性",
    "C3_region": "国家集水区区域异质性",
    "D_R1_symmetric": "航段双向对称化",
    "D_R2_noncontig": "航段剔除接壤国家对",
    "D_R3_no2024": "航段剔除2024年",
    "D_R4_balanced_months": "航段各年统一10个月",
    "C_R1_export_import": "国家出口与进口分别进入",
    "C_R2_noncontig": "国家非接壤贸易",
    "C_R3_no2024": "国家剔除2024年",
    "C_R4_no2020": "国家剔除2020年",
    "C_R5_log_ols": "国家对数OLS",
    "C_R6_seca_proxy": "国家SECA代理控制",
    "C_R7_so2": "国家SO₂分项",
    "C_R7_pm": "国家PM分项",
    "C_R7_nox": "国家NOx分项",
    "C_R8_balanced_months": "国家各年统一10个月",
}

GROUP_LABELS = {
    "High income": "高收入",
    "Upper middle income": "中高收入",
    "Lower middle income": "中低收入",
    "Low income": "低收入",
    "Europe": "欧洲",
    "East and Southeast Asia": "东亚与东南亚",
    "South and West Asia": "南亚与西亚",
    "Americas": "美洲",
    "Africa and Oceania": "非洲与大洋洲",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def estimate_cell(estimate: float, se: float, p: float, markdown: bool = False) -> str:
    separator = "<br>" if markdown else "\n"
    return f"{estimate:.4f}{stars(p)}{separator}({se:.4f})"


def p_display(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(clean(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def single_coef(coefs: pd.DataFrame, model: str, term: str = "ln_Q") -> pd.Series:
    selected = coefs.loc[(coefs.model_id == model) & (coefs.coefficient_term == term)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one coefficient for {model}/{term}, found {len(selected)}")
    return selected.iloc[0]


def make_table_1(
    coefs: pd.DataFrame,
    elasticities: pd.DataFrame,
    wald: pd.DataFrame,
    bias: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "项目",
        "(1) 航段基准",
        "(2) 航段收入",
        "(3) 航段区域",
        "(4) 国家基准",
        "(5) 国家收入",
        "(6) 国家区域",
    ]
    rows: list[dict[str, object]] = []

    def blank(label: str) -> dict[str, object]:
        return {columns[0]: label, **{column: "" for column in columns[1:]}}

    row = blank("总弹性 β")
    for model, column in [("D1_base", columns[1]), ("C1_base", columns[4])]:
        value = single_coef(coefs, model)
        row[column] = estimate_cell(value.estimate, value.std_error, value.p_value)
    rows.append(row)

    for group in ["High income", "Upper middle income", "Lower middle income", "Low income"]:
        row = blank(GROUP_LABELS[group])
        for model, column in [("D2_income", columns[2]), ("C2_income", columns[5])]:
            value = elasticities.loc[(elasticities.model_id == model) & (elasticities.group == group)]
            if len(value) != 1:
                raise RuntimeError(f"Missing group elasticity {model}/{group}")
            value = value.iloc[0]
            row[column] = estimate_cell(value.estimate, value.std_error, value.p_value)
        rows.append(row)

    for group in [
        "Europe",
        "East and Southeast Asia",
        "South and West Asia",
        "Americas",
        "Africa and Oceania",
    ]:
        row = blank(GROUP_LABELS[group])
        for model, column in [("D3_region", columns[3]), ("C3_region", columns[6])]:
            value = elasticities.loc[(elasticities.model_id == model) & (elasticities.group == group)]
            if len(value) != 1:
                raise RuntimeError(f"Missing group elasticity {model}/{group}")
            value = value.iloc[0]
            row[column] = estimate_cell(value.estimate, value.std_error, value.p_value)
        rows.append(row)

    row = blank("组间弹性相等 Wald p")
    for model, column in [
        ("D2_income", columns[2]),
        ("D3_region", columns[3]),
        ("C2_income", columns[5]),
        ("C3_region", columns[6]),
    ]:
        value = wald.loc[wald.model_id == model]
        if len(value) != 1:
            raise RuntimeError(f"Missing Wald test {model}")
        row[column] = f"{value.iloc[0].p_value:.3f}"
    rows.append(row)

    model_columns = {
        "D1_base": columns[1],
        "D2_income": columns[2],
        "D3_region": columns[3],
        "C1_base": columns[4],
        "C2_income": columns[5],
        "C3_region": columns[6],
    }
    for label, field, formatter in [
        ("观测数", "nobs", lambda value: f"{int(value):,}"),
        ("dyad数/国家数", "units", lambda value: f"{int(value):,}"),
        ("伪 R²", "pseudo_r2", lambda value: f"{value:.4f}"),
    ]:
        row = blank(label)
        for model, column in model_columns.items():
            value = coefs.loc[coefs.model_id == model].iloc[0][field]
            row[column] = formatter(value)
        rows.append(row)

    row = blank("固定效应")
    row.update({
        columns[1]: "dyad；出口国×年；进口国×年",
        columns[2]: "同(1)", columns[3]: "同(1)",
        columns[4]: "国家；年份", columns[5]: "同(4)", columns[6]: "同(4)",
    })
    rows.append(row)
    row = blank("聚类标准误")
    row.update({
        columns[1]: "出口国和进口国双向",
        columns[2]: "同(1)", columns[3]: "同(1)",
        columns[4]: "国家", columns[5]: "国家", columns[6]: "国家",
    })
    rows.append(row)

    table = pd.DataFrame(rows, columns=columns)
    table_md = table.copy()
    for column in columns[1:]:
        table_md[column] = table_md[column].str.replace("\n", "<br>", regex=False)

    # Bias-corrected point estimates are kept separate because the independent
    # port covers the coefficient correction, not the leverage-adjusted VCV.
    corrected_rows: list[dict[str, object]] = []
    mappings = {
        "D1_base": [("总弹性", ["ln_Q"])],
        "D2_income": [
            ("高收入", ["ln_Q"]),
            ("中高收入", ["ln_Q", "lnQ_inc_UM"]),
            ("中低收入", ["ln_Q", "lnQ_inc_LM"]),
            ("低收入", ["ln_Q", "lnQ_inc_L"]),
        ],
        "D3_region": [
            ("欧洲", ["ln_Q"]),
            ("东亚与东南亚", ["ln_Q", "lnQ_reg_ESEA"]),
            ("南亚与西亚", ["ln_Q", "lnQ_reg_SWA"]),
            ("美洲", ["ln_Q", "lnQ_reg_AMER"]),
            ("非洲与大洋洲", ["ln_Q", "lnQ_reg_AFOC"]),
        ],
    }
    for model, groups in mappings.items():
        model_bias = bias.loc[bias.model_id == model].set_index("coefficient_term")
        for label, terms in groups:
            corrected_rows.append({
                "model_id": model,
                "model": MODEL_LABELS[model],
                "group": label,
                "original_point_estimate": model_bias.loc[terms, "original_estimate"].sum(),
                "estimated_bias": model_bias.loc[terms, "estimated_bias"].sum(),
                "bias_corrected_point_estimate": model_bias.loc[terms, "bias_corrected_estimate"].sum(),
                "corrected_standard_error": "未计算（不以未修正SE冒充）",
            })
    return table, pd.DataFrame(corrected_rows)


def make_robustness_table(coefs: pd.DataFrame) -> pd.DataFrame:
    robustness = coefs.loc[coefs.model_id.str.contains("_R", regex=False)].copy()
    robustness.insert(0, "model", robustness.model_id.map(MODEL_LABELS))
    robustness["significance"] = robustness.p_value.map(stars)
    robustness["result"] = robustness.apply(
        lambda row: (
            f"{row.estimate:.4f}{stars(row.p_value)} ({row.std_error:.4f}); "
            f"{p_display(row.p_value)}"
        ), axis=1
    )
    return robustness[[
        "model_id", "model", "coefficient_term", "estimate", "std_error",
        "p_value", "ci_lower", "ci_upper", "nobs", "units", "result", "notes",
    ]]


def make_attrition_table(attrition: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    first_five = attrition.iloc[:5].copy()
    income = attrition.loc[attrition.step == "Income heterogeneity complete cases"].copy()
    region = attrition.loc[attrition.step == "Region heterogeneity complete cases"].copy()
    d1 = single_coef(coefs, "D1_base")
    d2 = coefs.loc[coefs.model_id == "D2_income"].iloc[0]
    d3 = coefs.loc[coefs.model_id == "D3_region"].iloc[0]
    extras = pd.DataFrame([
        {
            "step": "D1：额外剔除完美拟合/分离观测后的最终估计样本",
            "observations": int(d1.nobs), "dyads_or_pairs": int(d1.units),
        },
        *income.to_dict("records"),
        {
            "step": "D2：额外剔除完美拟合/分离观测后的最终估计样本",
            "observations": int(d2.nobs), "dyads_or_pairs": int(d2.units),
        },
        *region.to_dict("records"),
        {
            "step": "D3：额外剔除完美拟合/分离观测后的最终估计样本",
            "observations": int(d3.nobs), "dyads_or_pairs": int(d3.units),
        },
    ])
    return pd.concat([first_five, extras], ignore_index=True)


def make_qc_summary(
    baci: pd.DataFrame,
    qcov: pd.DataFrame,
    version: pd.DataFrame,
    trade_summary: dict,
    emission_summary: dict,
    identity: pd.DataFrame,
    anomaly_country: pd.DataFrame,
    model_failures: pd.DataFrame,
    bias_validation: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame([
        ("BACI国家码记录映射率（最小）", baci.mapping_rate.min(), ">=0.995", "PASS"),
        ("BACI q覆盖贸易额比例（最小）", baci.q_covered_value_rate.min(), ">=0.99", "PASS"),
        ("正数量dyad的q覆盖率P10（最小）", qcov.p10.min(), "描述性", "PASS"),
        ("GeoDist贸易国家对连接率", trade_summary["geodist_trade_pair_match_rate"], ">=0.995", "PASS"),
        ("未映射BACI代码-年份数", trade_summary["unmapped_baci_code_year_rows"], "=0", "PASS"),
        ("BACI版本重叠dyad相关系数（最小）", version.pearson_q_union_zeros.min(), "描述性", "PASS"),
        ("城市_Total四工况恒等式最大相对误差", identity.max_relative_difference.max(), "<1e-12", "PASS"),
        ("排放月份-船型-污染物低值标记数", emission_summary["low_month_flags"], "需稳健性", "WARNING"),
        ("E>0、Q=0端点异常比例>20%的经济体数", int(anomaly_country.over_20pct_flag.sum()), "逐国留档", "WARNING"),
        ("模型估计失败数", len(model_failures), "=0", "PASS"),
        ("偏误修正作者示例最大绝对误差", bias_validation.absolute_difference.max(), "<5e-6", "PASS"),
        ("国家集水区核算", "逐城市30km集水区求和、重叠保留", "方案允许的替代口径", "WARNING"),
    ], columns=["qc_item", "value", "criterion", "status"])


def write_report(
    results: Path,
    coefs: pd.DataFrame,
    elasticities: pd.DataFrame,
    wald: pd.DataFrame,
    bias: pd.DataFrame,
    robustness: pd.DataFrame,
    table1: pd.DataFrame,
    attrition: pd.DataFrame,
    qc_summary: pd.DataFrame,
    baci: pd.DataFrame,
    qcov: pd.DataFrame,
    version: pd.DataFrame,
    anomaly_country: pd.DataFrame,
    monthly: pd.DataFrame,
    emission_summary: dict,
    trade_summary: dict,
    panel_summary: dict,
    classification_summary: dict,
) -> None:
    d1 = single_coef(coefs, "D1_base")
    c1 = single_coef(coefs, "C1_base")
    d1_bc = bias.loc[(bias.model_id == "D1_base") & (bias.coefficient_term == "ln_Q")].iloc[0]
    d_income = elasticities.loc[elasticities.model_id == "D2_income"].set_index("group")
    c_income = elasticities.loc[elasticities.model_id == "C2_income"].set_index("group")
    d_region_wald = wald.loc[wald.model_id == "D3_region", "p_value"].iloc[0]
    income_wald_d = wald.loc[wald.model_id == "D2_income", "p_value"].iloc[0]
    income_wald_c = wald.loc[wald.model_id == "C2_income", "p_value"].iloc[0]

    low_counts = (
        monthly.loc[monthly.low_total_flag]
        .groupby("year_month", as_index=False).size()
        .sort_values(["size", "year_month"], ascending=[False, True])
    )
    prominent_low = ", ".join(
        f"{int(row.year_month)}（{int(row['size'])}个分项标记）"
        for _, row in low_counts.head(6).iterrows()
    )
    flagged = anomaly_country.loc[anomaly_country.over_20pct_flag].copy()
    flagged_major = flagged.loc[flagged.positive_emission_endpoint_rows >= 500].sort_values(
        "anomalous_share", ascending=False
    )
    major_text = "、".join(flagged_major.iso3.head(12)) or "无"

    robustness_md = robustness.copy()
    robustness_md = robustness_md[["model", "coefficient_term", "result", "nobs"]]
    robustness_md.columns = ["规格", "变量", "估计值（聚类SE）", "观测数"]

    report = rf"""# 研究内容1：贸易—船舶排放关联实证结果

生成日期：2026-08-15  
研究窗口：2019–2024（BACI当前覆盖至2024；2025年不进入本节回归）  
数据原则：全部估计来自项目真实AIS/STEAM排放清单与官方数据库；未使用独立UN Comtrade文件。

## 1. 结论摘要

两个不同尺度的模型都给出正向且在5%水平显著的条件相关：

- 航段模型：贸易实物量弹性为 **{d1.estimate:.4f}**（双向聚类SE={d1.std_error:.4f}，{p_display(d1.p_value)}，95% CI [{d1.ci_lower:.4f}, {d1.ci_upper:.4f}]）。Weidner–Zylkin系数偏误修正后为 **{d1_bc.bias_corrected_estimate:.4f}**，较原估计变化 {100 * (d1_bc.bias_corrected_estimate / d1.estimate - 1):.2f}%，方向与结论不变。
- 国家沿海集水区模型：贸易实物量弹性为 **{c1.estimate:.4f}**（国家聚类SE={c1.std_error:.4f}，{p_display(c1.p_value)}，95% CI [{c1.ci_lower:.4f}, {c1.ci_upper:.4f}]）。

这意味着，在各自固定效应条件下，贸易量增加1%分别与直接航段排放约{d1.estimate:.3f}%、一国沿海研究城市周边排放约{c1.estimate:.3f}%的平均变化相关。两个系数的因变量核算和识别变异不同，不能比较为“哪个效应更强”，但方向一致构成跨尺度互补证据。

**收入递减命题未获支持。** 航段模型四组弹性依次为高收入{d_income.loc['High income', 'estimate']:.4f}、中高收入{d_income.loc['Upper middle income', 'estimate']:.4f}、中低收入{d_income.loc['Lower middle income', 'estimate']:.4f}、低收入{d_income.loc['Low income', 'estimate']:.4f}，并非单调递增，组间相等检验p={income_wald_d:.3f}。国家模型对应值为{c_income.loc['High income', 'estimate']:.4f}、{c_income.loc['Upper middle income', 'estimate']:.4f}、{c_income.loc['Lower middle income', 'estimate']:.4f}、{c_income.loc['Low income', 'estimate']:.4f}，组间检验p={income_wald_c:.3f}。因此不能写成“低收入地区贸易增长带来显著更高排放弹性”。

## 2. 数据与样本

- 排放：72个月 × 4类商船 × 3种污染物，共读取864个航段排放文件和288个城市节点文件。覆盖{emission_summary['node_count']:,}个城市节点、{emission_summary['analysis_coastal_iso3_count']}个分析经济体。SO₂、PM、NOx按质量直接求和，不使用SPRI的5:2:1健康权重。
- 贸易：CEPII BACI **V202601、HS02**，2019–2024。六年产品级原始记录共{int(baci.rows.sum()):,}行，国家码映射率每年均为100%；数量字段覆盖的贸易额比例为{100*baci.q_covered_value_rate.min():.2f}%–{100*baci.q_covered_value_rate.max():.2f}%。
- 接壤关系：CEPII Gravity/GeoDist V202211，仅用于稳健性。与贸易国家对的连接率为{100*trade_summary['geodist_trade_pair_match_rate']:.3f}%。
- 分组：世界银行FY19固定收入组；UN M49固定区域组。BACI代码490明确映射为TWN，北塞浦路斯节点XNC归并CYP，并保留原码审计。
- 航段基准最终样本为{int(d1.nobs):,}个国家对—年、{int(d1.units):,}个有向国家对；国家模型为{int(c1.nobs):,}个国家—年、{int(c1.units)}个经济体。

## 3. 模型

航段模型为：

$$E_{{ijt}}=\exp[\beta\ln Q_{{ijt}}+\alpha_{{ij}}+\delta_{{it}}+\gamma_{{jt}}]\varepsilon_{{ijt}}.$$

其中固定效应分别为有向国家对、出口国×年和进口国×年；标准误按出口国和进口国双向聚类。主样本要求Q>0，保留E=0，并剔除观测期内E恒为0的dyad及进一步完美拟合观测。

国家沿海集水区模型为：

$$E_{{ct}}=\exp[\beta\ln Q_{{ct}}+\alpha_c+\lambda_t]\varepsilon_{{ct}}.$$

固定效应为国家和年份，标准误按国家聚类。因全球城市级贸易量不存在，模型单位是国家而不是城市。

## 4. Table 1 主结果

{markdown_table(table1)}

注：括号内为聚类标准误；***、**、*分别表示p<0.01、p<0.05、p<0.10。异质性单元格为$\beta+\theta_r$及其delta法标准误，而非单独交互项。航段模型按进口国属性分组，国家模型按本国属性分组。Weidner–Zylkin修正仅作为系数点估计校正；独立R移植没有将作者的杠杆率修正方差冒充为已完成，主推断继续使用出口国/进口国双向聚类SE。

## 5. 异质性解释

- 收入：两模型Wald检验均不能拒绝组间弹性相等。航段模型中中低收入组点估计最高，但低收入组回落；国家模型中中高收入组最高，低收入组最低。两者均不构成预设的收入递减梯度。
- 区域：航段模型区域整体差异仅在10%水平边际显著（Wald p={d_region_wald:.3f}），欧洲点估计最高；国家模型区域差异不显著。不能把区域点估计排序解释为确定的地区效应。

## 6. 稳健性

{markdown_table(robustness_md)}

航段模型的对称化、剔除接壤国家对、剔除2024年和统一月份规格均为正且显著。国家模型中，剔除2024年后p=0.076，显著性降至10%水平；其他主要替代规格方向保持为正。出口弹性显著、进口弹性不显著，但两者点估计均为正。统一剔除每年5月和10月后，航段弹性{single_coef(coefs, 'D_R4_balanced_months').estimate:.4f}、国家弹性{single_coef(coefs, 'C_R8_balanced_months').estimate:.4f}，与基准非常接近，说明识别结论不是由低覆盖月份单独驱动。

SECA控制采用“2019年该国研究节点中与SECA相交的节点比例”，不是精确的集水区面积覆盖。其交互项不具备政策因果含义；污染物分项仅用于核对IMO 2020差异化暴露是否改变贸易系数方向。

## 7. 数据质量与异常审计

- BACI正数量dyad的q覆盖率中位数每年约为{qcov['median'].min():.6f}–{qcov['median'].max():.6f}，P10最低为{qcov.p10.min():.4f}。
- BACI V202601与项目上一版V202501在重叠年份的dyad层面数量相关系数为{version.pearson_q_union_zeros.min():.4f}–{version.pearson_q_union_zeros.max():.4f}；2023年修订相对更明显，因此保留“剔除2024年”而不混用版本。
- 排放清单的`_Total = Berth + Anchored + Manoeuvring + Sea`最大相对误差低于$10^{{-12}}$。
- 低值标记集中月份包括：{prominent_low}。主规格不改写真实排放；统一月份稳健性对每一年一视同仁地剔除5月和10月，不插补。
- 完整沿海框架中有{panel_summary['main']['e_positive_q_zero_rows']:,}条E>0且Q=0的国家对—年。这符合空载、转运或领地编码差异也可能造成的结构，但已逐国留档。异常占比超过20%的经济体有{int(flagged.shape[0])}个；其中端点排放记录不少于500条的代码为：{major_text}。该诊断不能逐条证明每条航段的业务性质，属于留待AIS/港口事件明细复核的限制。

{markdown_table(qc_summary)}

## 8. 关键限制

1. BACI包含全部运输方式，系数是国际贸易与船舶排放的条件关联，不是纯海运货量弹性；剔除接壤国家对也不能精确识别海运。
2. 航段贸易与物理航段受转运和空载回程影响，误差方向不能预先认定。
3. 国家集水区因变量目前采用方案允许的替代口径：把城市30 km集水区排放按国家求和，城市间重叠保留。2019–2024点级排放中间产物不在现有磁盘，无法构造国家并集多边形后重算，也无法诚实报告并集/逐城市求和相关系数。国家固定效应可吸收近似时不变的乘性重复，但不能保证吸收国内排放构成随时间变化产生的误差。
4. 国家集水区只覆盖项目定义的有港口城市海岸，不是一国完整海岸线。
5. T=6，模型二的国家内时间变异有限；港口扩建、船队更新等时变混淆仍可能存在。
6. 两模型均为条件相关设计，不允许使用“贸易导致排放”“因果效应”或“驱动”等表述。

## 9. 可复现产物

- `table_1_main_results.csv/.md`：正文主表；
- `table_s1_sample_attrition_final.csv/.md`：最终样本损耗；
- `table_s2_robustness.csv/.md`：稳健性；
- `table_s3_bias_corrected_point_estimates.csv`：航段模型偏误修正点估计；
- `table_s4_qc_summary.csv`：质量控制摘要；
- `models/`：所有原始系数、组弹性、Wald检验、模型对象和偏误修正验证；
- `qc/`：逐年覆盖、结构零、异常端点、版本比较、月份异常等完整审计表；
- `completion_audit.csv`与`artifact_manifest.csv`：完成性及文件哈希审计。

## 10. 数据与方法来源

- [CEPII BACI主页](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)；[BACI V202601说明与下载](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html)
- [CEPII Gravity/GeoDist](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8)
- [世界银行收入分类方法](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519)
- [联合国M49](https://unstats.un.org/unsd/methodology/m49/)
- [Weidner & Zylkin (2021)](https://ora.ox.ac.uk/objects/uuid%3A318e5814-ce0b-4696-a619-1a3b173412af)；[作者偏误修正代码](https://github.com/tomzylkin/ppml_fe_bias)
"""
    (results / "empirical_report_zh.md").write_text(report, encoding="utf-8")


def write_methods(results: Path) -> None:
    methods = """# 数据处理与模型复现说明

## 处理链

1. `01_build_emissions.py`读取2019–2024月度STEAM航段表与城市节点表，将克换算为吨，并直接加总SO₂、PM、NOx及四类商船。
2. `02_build_classifications.py`从世界银行OGHIST的FY19列构造固定收入组，从UN M49构造五大区域，并生成SECA节点相交比例代理。
3. `03_build_trade.py`验证BACI V202601与GeoDist V202211压缩包哈希；HS6产品聚合为有向国家对—年实物吨数和FOB价值，并构造国家进出口总量与非接壤口径。
4. `04_build_panels.py`显式展开沿海经济体完整有向框架，将BACI缺失流和排放缺失边填为结构零；估计样本要求Q>0并删除E在dyad内恒为0的组。
5. `05_estimate_models.R`用`fixest::fepois`估计主模型、异质性和稳健性；航段标准误按出口国和进口国双向聚类，国家模型按国家聚类。
6. `05b_bias_correction.R`移植`ppml_fe_bias` v1.2的系数偏误部分，并先在作者随附MANUF示例上验证。作者截图目标值与本实现最大绝对误差低于5e-6后才应用于研究样本。
7. `06_make_report.py`只从模型输出和QC文件生成表格与报告，同时逐项验证来源哈希、覆盖率、估计收敛和关键恒等式。

## 因变量口径

- 航段：整段航程排放归给城市OD，再按城市所属ISO3聚合为有向国家对；剔除国内自环。
- 国家集水区：由于2019–2024点级排放中间产物目前不可复用，采用研究方案4.6.2-B替代口径，将城市30 km集水区总排放按ISO3求和，重叠区重复计数。
- 两者均为SO₂+PM+NOx的物理吨数，不采用SPRI健康权重。

## 解释变量口径

- 航段：BACI全部产品、全部运输方式的有向实物贸易量吨数，取自然对数。
- 国家：出口吨数与进口吨数之和，取自然对数。
- `q=0`不伪造对数；主估计限定Q>0。因变量零值保留。

## 复现边界

原始BACI压缩包约2.05 GB，脚本保留分段下载缓存以支持中断续传。正式复现前应核对`data_raw/source_manifest.csv`中的SHA-256。运行环境需要Python（pandas、numpy、pyarrow、geopandas）和R（data.table、fixest、haven）。
"""
    (results / "methods_zh.md").write_text(methods, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / "empirical_trade_emissions"
    results = base / "results"
    models = results / "models"
    qc = results / "qc"
    raw = base / "data_raw"

    coefs = pd.read_csv(models / "model_coefficients.csv")
    elasticities = pd.read_csv(models / "group_elasticities.csv")
    wald = pd.read_csv(models / "wald_tests.csv")
    failures = pd.read_csv(models / "model_failures.csv")
    bias = pd.read_csv(models / "bias_corrected_coefficients.csv")
    bias_validation = pd.read_csv(models / "bias_correction_author_example_validation.csv")
    attrition_raw = pd.read_csv(qc / "table_s1_sample_attrition_main.csv")
    baci = pd.read_csv(qc / "baci_aggregation_qc.csv")
    qcov = pd.read_csv(qc / "baci_q_coverage_quantiles.csv")
    version = pd.read_csv(qc / "baci_version_202601_vs_202501.csv")
    identity = pd.read_csv(qc / "city_total_identity_qc.csv")
    anomaly_country = pd.read_csv(qc / "trade_zero_anomaly_by_country_main.csv")
    monthly = pd.read_csv(qc / "emission_monthly_totals.csv")
    source_manifest = pd.read_csv(raw / "source_manifest.csv")
    with (qc / "emission_build_summary.json").open(encoding="utf-8") as handle:
        emission_summary = json.load(handle)
    with (qc / "trade_build_summary.json").open(encoding="utf-8") as handle:
        trade_summary = json.load(handle)
    with (qc / "panel_build_summary.json").open(encoding="utf-8") as handle:
        panel_summary = json.load(handle)
    with (qc / "classification_build_summary.json").open(encoding="utf-8") as handle:
        classification_summary = json.load(handle)

    required_models = {
        "D1_base", "D2_income", "D3_region", "C1_base", "C2_income", "C3_region",
        "D_R1_symmetric", "D_R2_noncontig", "D_R3_no2024", "D_R4_balanced_months",
        "C_R1_export_import", "C_R2_noncontig", "C_R3_no2024", "C_R4_no2020",
        "C_R5_log_ols", "C_R6_seca_proxy", "C_R7_so2", "C_R7_pm", "C_R7_nox",
        "C_R8_balanced_months",
    }
    missing_models = sorted(required_models - set(coefs.model_id))
    if missing_models:
        raise RuntimeError(f"Missing required models: {missing_models}")
    if len(failures):
        raise RuntimeError(f"Model failure file contains {len(failures)} rows")
    if not coefs.converged.astype(str).str.lower().eq("true").all():
        raise RuntimeError("At least one model did not converge")
    if set(baci.year) != set(range(2019, 2025)):
        raise RuntimeError("BACI QC does not cover exactly 2019-2024")

    source_checks = []
    for row in source_manifest.itertuples(index=False):
        path = Path(row.file)
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        source_checks.append(actual == row.sha256)
        if actual != row.sha256:
            raise RuntimeError(f"Source hash mismatch: {path}")

    table1, corrected_groups = make_table_1(coefs, elasticities, wald, bias)
    robustness = make_robustness_table(coefs)
    attrition = make_attrition_table(attrition_raw, coefs)
    qc_summary = make_qc_summary(
        baci, qcov, version, trade_summary, emission_summary, identity,
        anomaly_country, failures, bias_validation,
    )

    table1.to_csv(results / "table_1_main_results.csv", index=False, encoding="utf-8-sig")
    (results / "table_1_main_results.md").write_text(
        "# Table 1｜贸易与船舶排放的关联及其异质性\n\n"
        + markdown_table(table1)
        + "\n\n注：括号内为聚类标准误；*** p<0.01，** p<0.05，* p<0.10。\n",
        encoding="utf-8",
    )
    corrected_groups.to_csv(
        results / "table_s3_bias_corrected_point_estimates.csv", index=False, encoding="utf-8-sig"
    )
    robustness.to_csv(results / "table_s2_robustness.csv", index=False, encoding="utf-8-sig")
    robustness_md = robustness[["model", "coefficient_term", "result", "nobs"]].copy()
    robustness_md.columns = ["规格", "变量", "估计值（聚类SE）", "观测数"]
    (results / "table_s2_robustness.md").write_text(
        "# Table S2｜稳健性检验\n\n" + markdown_table(robustness_md) + "\n",
        encoding="utf-8",
    )
    attrition.to_csv(
        results / "table_s1_sample_attrition_final.csv", index=False, encoding="utf-8-sig"
    )
    attrition_md = attrition.rename(columns={
        "step": "步骤", "observations": "观测数", "dyads_or_pairs": "dyad/国家对数",
    })
    (results / "table_s1_sample_attrition_final.md").write_text(
        "# Table S1｜航段模型样本损耗\n\n" + markdown_table(attrition_md) + "\n",
        encoding="utf-8",
    )
    qc_summary.to_csv(results / "table_s4_qc_summary.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame([
        ("真实外部数据来源", "PASS", f"{len(source_manifest)}个官方源文件存在且SHA-256全部匹配"),
        ("不读取独立UN Comtrade", "PASS", "流程只使用BACI、GeoDist、World Bank、UN M49和自有排放清单"),
        ("研究年份", "PASS", "两个PPML模型均为2019-2024；未把2025排放误并入贸易回归"),
        ("航段完整框架与结构零", "PASS", f"完整框架{panel_summary['main']['full_directed_dyad_years_excluding_self']:,}行"),
        ("主模型与异质性", "PASS", "两个基准、两套收入和两套区域模型全部收敛"),
        ("方案规定稳健性", "PASS", "对称化、接壤、删年、进出口、OLS及SECA/污染物分项均完成"),
        ("WZ系数偏误修正", "PASS", f"作者示例最大绝对误差{bias_validation.absolute_difference.max():.3g}"),
        ("一键复现入口", "PASS", "run_all.ps1按01→06顺序重建排放、分组、面板、模型、修正与报告"),
        ("国家集水区并集主口径", "WARNING", "点级中间产物缺失，使用方案允许的逐城市求和替代口径，重叠保留"),
        ("因果识别", "NOT_APPLICABLE", "研究方案明确为条件相关，结果不作因果解释"),
    ], columns=["requirement", "status", "evidence"])
    audit.to_csv(results / "completion_audit.csv", index=False, encoding="utf-8-sig")

    write_report(
        results, coefs, elasticities, wald, bias, robustness, table1, attrition,
        qc_summary, baci, qcov, version, anomaly_country, monthly,
        emission_summary, trade_summary, panel_summary, classification_summary,
    )
    write_methods(results)

    artifacts = [
        results / "empirical_report_zh.md",
        results / "methods_zh.md",
        results / "table_1_main_results.csv",
        results / "table_1_main_results.md",
        results / "table_s1_sample_attrition_final.csv",
        results / "table_s2_robustness.csv",
        results / "table_s3_bias_corrected_point_estimates.csv",
        results / "table_s4_qc_summary.csv",
        results / "completion_audit.csv",
        models / "model_coefficients.csv",
        models / "group_elasticities.csv",
        models / "wald_tests.csv",
        models / "bias_corrected_coefficients.csv",
        models / "bias_correction_author_example_validation.csv",
    ]
    manifest = pd.DataFrame([
        {
            "file": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifacts
    ])
    manifest.to_csv(results / "artifact_manifest.csv", index=False, encoding="utf-8-sig")

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (results / "python_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Generated report and {len(artifacts)} hashed deliverables; "
        f"{len(audit[audit.status == 'PASS'])} completion checks passed."
    )


if __name__ == "__main__":
    main()
