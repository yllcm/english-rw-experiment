"""
抽取一篇论文，查看其参考文献的 CS 得分和非 CS 得分分布
"""
import sys
sys.path.insert(0, "openalex_disruptiveness")

from openalex_client import OpenAlexClient
from config import AI4S_CAMP_A

client = OpenAlexClient()

# 搜索一篇论文
works = client.search_works(
    query="machine learning OR deep learning OR artificial intelligence",
    year_from=2020,
    year_to=2023,
    max_results=1,
    concept_filter="C86803240|C185592680|C121332964|C71924100|C192562407|C33923547|C39432304|C127313418|C15744967|C162324750",
)

if not works:
    print("未找到论文")
    sys.exit(1)

work = works[0]
title = work.get("title", "")[:60]
wid = work.get("id", "")
ref_ids = work.get("referenced_works", [])

print(f"论文: {title}")
print(f"ID: {wid}")
print(f"参考文献总数: {len(ref_ids)}")

# 获取参考文献信息
ref_infos = client._batch_get_works(ref_ids, batch_size=50)

# 统计参考文献的 CS 得分和非 CS 得分
cs_scores = []
non_cs_max_scores = []
level0_non_cs_max_scores = []  # 只统计 level=0 的非 CS 学科

for ref_id, ref_info in ref_infos.items():
    concepts = ref_info.get("concepts", [])
    if not concepts:
        continue
    
    # 找 CS 最高得分
    cs_max = 0
    # 找非 CS 最高得分（所有学科）
    non_cs_max = 0
    # 找非 CS 最高得分（仅 level=0）
    level0_non_cs_max = 0
    
    for c in concepts:
        cid = c.get("id", "")
        score = c.get("score", 0)
        level = c.get("level", -1)
        pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
        
        if pure_cid in AI4S_CAMP_A or cid in AI4S_CAMP_A:
            if score > cs_max:
                cs_max = score
        else:
            if score > non_cs_max:
                non_cs_max = score
            if level == 0 and score > level0_non_cs_max:
                level0_non_cs_max = score
    
    if cs_max > 0:
        cs_scores.append(cs_max)
        non_cs_max_scores.append(non_cs_max)
        level0_non_cs_max_scores.append(level0_non_cs_max)

print(f"\n有 CS 得分的参考文献数: {len(cs_scores)}")

# CS 得分分布
print(f"\nCS 得分分布:")
for threshold in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    count = sum(1 for s in cs_scores if s >= threshold)
    print(f"  >= {threshold}: {count} 篇 ({count/len(cs_scores)*100:.1f}%)")

# 非 CS 得分分布（所有学科）
print(f"\n非 CS 得分分布（所有学科）:")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    count = sum(1 for s in non_cs_max_scores if s <= threshold)
    print(f"  <= {threshold}: {count} 篇 ({count/len(non_cs_max_scores)*100:.1f}%)")

# 非 CS 得分分布（仅 level=0）
print(f"\n非 CS 得分分布（仅 level=0 一级学科）:")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    count = sum(1 for s in level0_non_cs_max_scores if s <= threshold)
    print(f"  <= {threshold}: {count} 篇 ({count/len(level0_non_cs_max_scores)*100:.1f}%)")

# 组合条件对比
print(f"\n组合条件对比:")
# 条件1: CS >= 0.5
# 条件2: 非CS <= 0.3（所有学科）
both_all = sum(1 for cs, nc in zip(cs_scores, non_cs_max_scores) if cs >= 0.5 and nc <= 0.3)
print(f"  CS>=0.5 且 非CS<=0.3（所有学科）: {both_all} 篇 ({both_all/len(cs_scores)*100:.1f}%)")

# 条件1: CS >= 0.5
# 条件2: 非CS <= 0.3（仅 level=0）
both_level0 = sum(1 for cs, nc in zip(cs_scores, level0_non_cs_max_scores) if cs >= 0.5 and nc <= 0.3)
print(f"  CS>=0.5 且 非CS<=0.3（仅 level=0）: {both_level0} 篇 ({both_level0/len(cs_scores)*100:.1f}%)")

# 条件1: CS >= 0.4
# 条件2: 非CS <= 0.3（仅 level=0）
both_level0_04 = sum(1 for cs, nc in zip(cs_scores, level0_non_cs_max_scores) if cs >= 0.4 and nc <= 0.3)
print(f"  CS>=0.4 且 非CS<=0.3（仅 level=0）: {both_level0_04} 篇 ({both_level0_04/len(cs_scores)*100:.1f}%)")

# 条件1: CS >= 0.5
# 条件2: 非CS <= 0.5（仅 level=0）
both_level0_05 = sum(1 for cs, nc in zip(cs_scores, level0_non_cs_max_scores) if cs >= 0.5 and nc <= 0.5)
print(f"  CS>=0.5 且 非CS<=0.5（仅 level=0）: {both_level0_05} 篇 ({both_level0_05/len(cs_scores)*100:.1f}%)")

# 打印一些示例参考文献的 concepts
print(f"\n\n前 5 篇参考文献的 concepts 详情:")
shown = 0
for ref_id, ref_info in ref_infos.items():
    if shown >= 5:
        break
    concepts = ref_info.get("concepts", [])
    year = ref_info.get("publication_year", "?")
    print(f"\n  Ref: {ref_id} ({year})")
    for c in concepts:
        cid = c.get("id", "")
        score = c.get("score", 0)
        name = c.get("display_name", "")
        level = c.get("level", -1)
        pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
        in_camp_a = pure_cid in AI4S_CAMP_A or cid in AI4S_CAMP_A
        marker = " [CAMP_A]" if in_camp_a else ""
        if level == 0 or in_camp_a:
            print(f"    {name}({score:.3f}) level={level}{marker}")
    shown += 1
