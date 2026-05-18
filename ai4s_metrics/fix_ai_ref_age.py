"""
快速修正已有 CSV 文件中的 ai_ref_age 列

错误: ai_ref_age = 当前年份(2026) - avg_ai_ref_year
正确: ai_ref_age = publication_year - avg_ai_ref_year
"""

import os
import pandas as pd
import glob

RESULTS_DIR = "results"

# 查找所有相关 CSV 文件
csv_files = (
    glob.glob(os.path.join(RESULTS_DIR, "ai4s_metrics_results_*.csv")) +
    glob.glob(os.path.join(RESULTS_DIR, "ai4s_metrics_full_*.csv"))
)

if not csv_files:
    print("未找到 CSV 文件")
else:
    for filepath in sorted(csv_files):
        print(f"\n处理: {filepath}")
        df = pd.read_csv(filepath)

        if "ai_ref_age" not in df.columns or "publication_year" not in df.columns or "avg_ai_ref_year" not in df.columns:
            print(f"  跳过: 缺少必要列")
            continue

        # 统计修正前
        before = df["ai_ref_age"].describe()
        print(f"  修正前 ai_ref_age: 均值={before['mean']:.2f}, 范围=[{before['min']:.2f}, {before['max']:.2f}]")

        # 修正: ai_ref_age = publication_year - avg_ai_ref_year
        # 只修正 avg_ai_ref_year > 0 的行（即有 AI 参考文献的论文）
        mask = (df["avg_ai_ref_year"] > 0) & (df["avg_ai_ref_year"] <= df["publication_year"])
        df.loc[mask, "ai_ref_age"] = round(
            df.loc[mask, "publication_year"] - df.loc[mask, "avg_ai_ref_year"], 2
        )
        # 如果 avg_ai_ref_year > publication_year（数据异常），整行剔除
        anomaly_mask = (df["avg_ai_ref_year"] > 0) & (df["avg_ai_ref_year"] > df["publication_year"])
        if anomaly_mask.any():
            print(f"  发现 {anomaly_mask.sum()} 行异常数据（avg_ai_ref_year > publication_year），已剔除")
            df = df[~anomaly_mask]

        # 统计修正后
        after = df["ai_ref_age"].describe()
        print(f"  修正后 ai_ref_age: 均值={after['mean']:.2f}, 范围=[{after['min']:.2f}, {after['max']:.2f}]")
        print(f"  共修正 {mask.sum()} 行")

        # 保存
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  已保存")

print("\n完成!")
