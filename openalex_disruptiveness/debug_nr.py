"""
调试：查看论文本身的 concepts（只显示 level=0 的一级学科）
"""
import sys
sys.path.insert(0, "openalex_disruptiveness")

from openalex_client import OpenAlexClient
from config import AI4S_CAMP_A, AI_CAMP_A_SCORE_MIN, AI_NON_CAMP_A_SCORE_MAX

client = OpenAlexClient()

works = client.search_works(
    query="machine learning OR deep learning OR artificial intelligence",
    year_from=2020,
    year_to=2023,
    max_results=5,
    concept_filter="C86803240|C185592680|C121332964|C71924100|C192562407|C33923547|C39432304|C127313418|C15744967|C162324750",
)

for i, work in enumerate(works):
    title = work.get("title", "")[:60]
    concepts = work.get("concepts", [])
    
    print(f"\n{'='*60}")
    print(f"[{i+1}] {title}")
    
    # 只显示 level=0 的一级学科
    print(f"  Level-0 Concepts:")
    has_high_cs = False
    all_non_cs_low = True
    
    for c in concepts:
        level = c.get("level", -1)
        if level != 0:
            continue
        cid = c.get("id", "")
        score = c.get("score", 0)
        name = c.get("display_name", "")
        pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
        in_camp_a = pure_cid in AI4S_CAMP_A or cid in AI4S_CAMP_A
        marker = " [CAMP_A]" if in_camp_a else ""
        
        if in_camp_a:
            if score >= AI_CAMP_A_SCORE_MIN:
                has_high_cs = True
        else:
            if score > AI_NON_CAMP_A_SCORE_MAX:
                all_non_cs_low = False
        
        print(f"    {name}({score:.3f}) level={level}{marker}")
    
    print(f"  条件1(CS>={AI_CAMP_A_SCORE_MIN}): {'✅' if has_high_cs else '❌'}")
    print(f"  条件2(非CS<={AI_NON_CAMP_A_SCORE_MAX}): {'✅' if all_non_cs_low else '❌'}")
    print(f"  _is_camp_a_work = {has_high_cs and all_non_cs_low}")
