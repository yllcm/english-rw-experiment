"""
主程序入口 - 爬取 OpenAlex 数据并计算论文各项指标

使用方法:
    python main.py

配置修改:
    编辑 config.py 文件调整搜索关键词、年份范围、论文数量等参数
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import (
    SEARCH_QUERY,
    SEARCH_CONCEPT_FILTER,
    YEAR_FROM,
    YEAR_TO,
    MAX_WORKS,
    OUTPUT_DIR,
    RESULTS_FILE,
    VIZ_FILE,
)
from api_client import OpenAlexClient
from metrics import MetricsCalculator
from collect_controls import collect_controls_batch


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(results: list, filepath: str):
    """
    将结果保存为 CSV 文件（输出所有字段）

    Args:
        results: 计算结果列表
        filepath: 输出文件路径
    """
    processed = []
    for r in results:
        row = {}
        for key, value in r.items():
            if isinstance(value, list):
                row[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                row[key] = "; ".join(f"{k}:{v}" for k, v in value.items())
            else:
                row[key] = value
        processed.append(row)

    df = pd.DataFrame(processed)

    # 按 Rela_Dz 降序排列
    df = df.sort_values("Rela_Dz", ascending=False)

    # 定义友好的列顺序（所有指标按逻辑分组排列）
    column_order = [
        # --- 基本信息 ---
        "work_id", "title", "publication_year",
        # --- 颠覆性指数 ---
        "Rela_Dz", "N_F", "N_R", "both", "neither",
        "cited_by_count", "citation_impact", "num_references", "num_citing_works",
        # --- 基于引用的跨学科性 ---
        "cit_interdisciplinarity", "cit_interdisc_citing_with_disc", "cit_interdisc_total_citing",
        # --- 团队学科多样性 ---
        "discipline_variety", "num_authors", "author_discipline_counts",
        # --- 团队学科相似性 ---
        "discipline_similarity", "ds_num_authors", "pairwise_similarities",
        # --- 团队学科均衡性 ---
        "discipline_balance", "db_num_authors", "sorted_frequencies",
        # --- AI for Science 融合度 ---
        "ai4s_balance", "p_a", "p_b",
        "camp_a_count", "camp_b_count", "total_refs_ai4s",
        # --- AI 技术时效性 ---
        "ai_ref_age", "ai_ref_count", "total_refs",
        "avg_ai_ref_year", "min_ai_ref_year", "max_ai_ref_year",
        "ai_ref_years",
    ]

    existing_columns = [col for col in column_order if col in df.columns]
    extra_columns = [col for col in df.columns if col not in column_order]
    df_out = df[existing_columns + extra_columns]

    df_out.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至: {filepath}")
    print(f"共 {len(df_out)} 条记录，{len(df_out.columns)} 个字段")

    return df


def visualize_results(df: pd.DataFrame, filepath: str):
    """
    可视化结果

    Args:
        df: 结果 DataFrame
        filepath: 图表保存路径
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    has_dv = "discipline_variety" in df.columns and df["discipline_variety"].sum() > 0

    if has_dv:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rela_Dz 分布直方图
    ax1 = axes[0, 0]
    sns.histplot(df["Rela_Dz"], bins=30, kde=True, ax=ax1, color="steelblue")
    ax1.set_title("Rela_Dz 分布", fontsize=12)
    ax1.set_xlabel("Rela_Dz")
    ax1.set_ylabel("论文数量")

    # 2. Rela_Dz vs 被引次数散点图
    ax2 = axes[0, 1]
    sns.scatterplot(
        data=df, x="cited_by_count", y="Rela_Dz",
        alpha=0.6, ax=ax2, color="coral"
    )
    ax2.set_title("Rela_Dz vs 被引次数", fontsize=12)
    ax2.set_xlabel("被引次数 (C)")
    ax2.set_ylabel("Rela_Dz")
    if df["cited_by_count"].max() > 100:
        ax2.set_xscale("log")

    # 3. N_F vs N_R 散点图
    ax3 = axes[0, 2] if has_dv else axes[1, 0]
    sns.scatterplot(
        data=df, x="N_R", y="N_F",
        hue="Rela_Dz", size="Rela_Dz",
        alpha=0.7, ax=ax3, palette="viridis"
    )
    ax3.set_title("N_F vs N_R (颜色表示 Rela_Dz)", fontsize=12)
    ax3.set_xlabel("N_R (仅引参考文献)")
    ax3.set_ylabel("N_F (仅引目标论文)")

    if has_dv:
        ax4 = axes[1, 0]
        sns.histplot(df["discipline_variety"], bins=20, kde=True, ax=ax4, color="green")
        ax4.set_title("Discipline Variety 分布", fontsize=12)
        ax4.set_xlabel("Discipline Variety")
        ax4.set_ylabel("论文数量")

        ax5 = axes[1, 1]
        sns.scatterplot(
            data=df, x="discipline_variety", y="Rela_Dz",
            hue="num_authors", size="cited_by_count",
            alpha=0.7, ax=ax5, palette="coolwarm"
        )
        ax5.set_title("Rela_Dz vs Discipline Variety", fontsize=12)
        ax5.set_xlabel("Discipline Variety")
        ax5.set_ylabel("Rela_Dz")

        ax6 = axes[1, 2]
        top10 = df.head(10)
        short_titles = [
            t[:30] + "..." if len(t) > 30 else t
            for t in top10["title"]
        ]
        ax6.barh(
            range(len(short_titles)),
            top10["Rela_Dz"],
            color=plt.cm.viridis(top10["discipline_variety"] / top10["discipline_variety"].max())
            if has_dv and top10["discipline_variety"].max() > 0
            else "steelblue"
        )
        ax6.set_yticks(range(len(short_titles)))
        ax6.set_yticklabels(short_titles, fontsize=8)
        ax6.set_title("Top 10 颠覆性论文 (颜色=DV)", fontsize=12)
        ax6.set_xlabel("Rela_Dz")
        ax6.invert_yaxis()
    else:
        ax4 = axes[1, 1]
        top10 = df.head(10)
        short_titles = [
            t[:30] + "..." if len(t) > 30 else t
            for t in top10["title"]
        ]
        bars = ax4.barh(
            range(len(short_titles)),
            top10["Rela_Dz"],
            color="steelblue"
        )
        ax4.set_yticks(range(len(short_titles)))
        ax4.set_yticklabels(short_titles, fontsize=8)
        ax4.set_title("Top 10 颠覆性论文", fontsize=12)
        ax4.set_xlabel("Rela_Dz")
        ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {filepath}")


def print_summary(df: pd.DataFrame):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)

    print(f"总论文数: {len(df)}")
    print(f"Rela_Dz 范围: {df['Rela_Dz'].min():.6f} ~ {df['Rela_Dz'].max():.6f}")
    print(f"Rela_Dz 均值: {df['Rela_Dz'].mean():.6f}")
    print(f"Rela_Dz 中位数: {df['Rela_Dz'].median():.6f}")

    if "discipline_variety" in df.columns and df["discipline_variety"].sum() > 0:
        print(f"\nDiscipline Variety 范围: {df['discipline_variety'].min():.4f} ~ "
              f"{df['discipline_variety'].max():.4f}")
        print(f"Discipline Variety 均值: {df['discipline_variety'].mean():.4f}")
        print(f"Discipline Variety 中位数: {df['discipline_variety'].median():.4f}")

    if "discipline_similarity" in df.columns and df["discipline_similarity"].sum() > 0:
        print(f"\nDiscipline Similarity 范围: {df['discipline_similarity'].min():.4f} ~ "
              f"{df['discipline_similarity'].max():.4f}")
        print(f"Discipline Similarity 均值: {df['discipline_similarity'].mean():.4f}")
        print(f"Discipline Similarity 中位数: {df['discipline_similarity'].median():.4f}")

    if "discipline_balance" in df.columns and df["discipline_balance"].sum() > 0:
        print(f"\nDiscipline Balance 范围: {df['discipline_balance'].min():.4f} ~ "
              f"{df['discipline_balance'].max():.4f}")
        print(f"Discipline Balance 均值: {df['discipline_balance'].mean():.4f}")
        print(f"Discipline Balance 中位数: {df['discipline_balance'].median():.4f}")

    if "ai4s_balance" in df.columns and df["ai4s_balance"].sum() > 0:
        print(f"\nAI4S_Balance 范围: {df['ai4s_balance'].min():.4f} ~ "
              f"{df['ai4s_balance'].max():.4f}")
        print(f"AI4S_Balance 均值: {df['ai4s_balance'].mean():.4f}")
        print(f"AI4S_Balance 中位数: {df['ai4s_balance'].median():.4f}")
        print(f"P_A 均值: {df['p_a'].mean():.4f}, P_B 均值: {df['p_b'].mean():.4f}")

    if "ai_ref_age" in df.columns and df["ai_ref_age"].sum() > 0:
        print(f"\nAI_Ref_Age 范围: {df['ai_ref_age'].min():.2f} ~ "
              f"{df['ai_ref_age'].max():.2f} 年")
        print(f"AI_Ref_Age 均值: {df['ai_ref_age'].mean():.2f} 年")
        print(f"AI_Ref_Age 中位数: {df['ai_ref_age'].median():.2f} 年")
        print(f"AI 参考文献占比均值: {df['ai_ref_count'].mean()/df['total_refs'].mean()*100:.1f}%")

    print("\n--- Top 5 颠覆性论文 ---")
    for i, row in df.head(5).iterrows():
        print(f"  {i+1}. {row['title'][:60]}...")
        extras = []
        if "discipline_variety" in row:
            extras.append(f"DV={row['discipline_variety']:.4f}")
        if "discipline_similarity" in row:
            extras.append(f"DS={row['discipline_similarity']:.4f}")
        if "discipline_balance" in row:
            extras.append(f"DB={row['discipline_balance']:.4f}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        print(f"     Rela_Dz={row['Rela_Dz']:.6f}, C={row['cited_by_count']}, "
              f"N_F={row['N_F']}, N_R={row['N_R']}{extra_str}")

    print("\n--- Bottom 5 (最低颠覆性) ---")
    for i, row in df.tail(5).iterrows():
        print(f"  {i+1}. {row['title'][:60]}...")
        extras = []
        if "discipline_variety" in row:
            extras.append(f"DV={row['discipline_variety']:.4f}")
        if "discipline_similarity" in row:
            extras.append(f"DS={row['discipline_similarity']:.4f}")
        if "discipline_balance" in row:
            extras.append(f"DB={row['discipline_balance']:.4f}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        print(f"     Rela_Dz={row['Rela_Dz']:.6f}, C={row['cited_by_count']}, "
              f"N_F={row['N_F']}, N_R={row['N_R']}{extra_str}")


def main():
    """主函数"""
    print("=" * 60)
    print("AI4S 论文指标计算器")
    print("=" * 60)
    print(f"搜索条件: '{SEARCH_QUERY}'")
    print(f"年份范围: {YEAR_FROM} - {YEAR_TO}")
    print(f"最大论文数: {MAX_WORKS}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_output_dir()

    # 第一步：搜索论文
    print("\n[步骤 1/3] 搜索论文...")
    client = OpenAlexClient()
    works = client.search_works(
        query=SEARCH_QUERY,
        year_from=YEAR_FROM,
        year_to=YEAR_TO,
        max_results=MAX_WORKS,
        concept_filter=SEARCH_CONCEPT_FILTER,
    )

    if not works:
        print("未找到任何论文，请检查搜索条件。")
        return

    # 第二步：计算所有指标
    print(f"\n[步骤 2/3] 计算指标 (共 {len(works)} 篇论文)...")
    calculator = MetricsCalculator()
    results = calculator.compute_batch(works)

    if not results:
        print("计算失败，未得到任何结果。")
        return

    # 筛选 AI4S 论文（AI4S_Balance > 0 表示论文同时涉及 AI 和科学学科）
    ai4s_results = [r for r in results if r.get("ai4s_balance", 0) > 0]
    filtered_count = len(results) - len(ai4s_results)
    if filtered_count > 0:
        print(f"\n  已过滤 {filtered_count} 篇纯 AI 论文 (AI4S_Balance = 0)")
    results = ai4s_results

    if not results:
        print("未找到 AI4S 论文（所有论文的 AI4S_Balance 均为 0）。")
        return

    # 第三步：保存和可视化结果（文件名自动添加时间戳，避免覆盖）
    print(f"\n[步骤 3/3] 保存结果和可视化...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3a. 保存指标结果
    csv_filename = RESULTS_FILE.replace(".csv", f"_{timestamp}.csv")
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df = save_results(results, csv_path)

    # 3b. 收集并保存控制变量
    print("\n收集控制变量...")
    controls = collect_controls_batch(works)
    controls_df = pd.DataFrame(controls)
    controls_filename = f"controls_{timestamp}.csv"
    controls_path = os.path.join(OUTPUT_DIR, controls_filename)
    controls_df.to_csv(controls_path, index=False, encoding="utf-8-sig")
    print(f"控制变量已保存至: {controls_path}")
    print(f"共 {len(controls_df)} 条记录，{len(controls_df.columns)} 个字段: "
          f"{', '.join(controls_df.columns)}")

    # 3c. 合并指标 + 控制变量（按 work_id 匹配）
    merged = df.merge(controls_df, on="work_id", how="left", suffixes=("", "_ctrl"))
    merged_filename = f"ai4s_metrics_full_{timestamp}.csv"
    merged_path = os.path.join(OUTPUT_DIR, merged_filename)
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"合并数据已保存至: {merged_path}")
    print(f"共 {len(merged)} 条记录，{len(merged.columns)} 个字段")

    print_summary(df)

    viz_filename = VIZ_FILE.replace(".png", f"_{timestamp}.png")
    viz_path = os.path.join(OUTPUT_DIR, viz_filename)
    visualize_results(df, viz_path)

    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
