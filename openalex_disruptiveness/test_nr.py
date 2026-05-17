"""
分析 CS >= 0.5 且 非CS <= 0.3 的组合
"""
import csv
import re

csv_path = "results/disruptiveness_results_test.csv"

cs_scores = []
non_cs_max_scores = []
titles = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        camp_a_str = row.get("camp_a_disciplines", "")
        camp_b_str = row.get("camp_b_disciplines", "")
        title = row.get("title", "")[:50]
        
        # 解析 camp_a 中的 CS score
        cs_score = None
        items = re.findall(r'(\w[\w\s-]+?)\(([\d.]+)\)', camp_a_str)
        for name, score in items:
            if name.strip() == "Computer science":
                cs_score = float(score)
        
        # 解析 camp_b 中的最高 score
        non_cs_max = 0
        items = re.findall(r'(\w[\w\s-]+?)\(([\d.]+)\)', camp_b_str)
        for name, score in items:
            s = float(score)
            if s > non_cs_max:
                non_cs_max = s
        
        if cs_score is not None:
            cs_scores.append(cs_score)
            non_cs_max_scores.append(non_cs_max)
            titles.append(title)

total = len(cs_scores)
cs_05 = sum(1 for s in cs_scores if s >= 0.5)
nc_03 = sum(1 for s in non_cs_max_scores if s <= 0.3)
both = sum(1 for cs, nc in zip(cs_scores, non_cs_max_scores) if cs >= 0.5 and nc <= 0.3)

print(f"总论文数: {total}")
print(f"CS >= 0.5: {cs_05} 篇 ({cs_05/total*100:.1f}%)")
print(f"非CS <= 0.3: {nc_03} 篇 ({nc_03/total*100:.1f}%)")
print(f"同时满足: {both} 篇 ({both/total*100:.1f}%)")

# 列出同时满足的论文
print(f"\n同时满足的论文:")
for i, (cs, nc, t) in enumerate(zip(cs_scores, non_cs_max_scores, titles)):
    if cs >= 0.5 and nc <= 0.3:
        print(f"  CS={cs:.3f}, 非CS最高={nc:.3f}: {t}")
