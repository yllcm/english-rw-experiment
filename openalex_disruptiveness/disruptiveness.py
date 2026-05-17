"""
颠覆性指数计算模块 - 实现 Rela_Dz 公式

公式:
    Rela_Dz = (8 × N_F²) / (2 × C² + C × N_R)

其中:
    N_F: 仅引用目标论文，但不引用目标论文参考文献的后续论文数量
    N_R: 仅引用目标论文的参考文献，但不引用目标论文本身的后续论文数量
    C:   目标论文的总被引频次
"""

import math
from typing import Dict, List, Optional
from openalex_client import OpenAlexClient
from config import (
    ENABLE_DISCIPLINE_VARIETY,
    ENABLE_DISCIPLINE_SIMILARITY,
    ENABLE_DISCIPLINE_BALANCE,
    ENABLE_AI4S_BALANCE,
    ENABLE_AI_REF_AGE,
    DISCIPLINE_SCORE_THRESHOLD,
)


class DisruptivenessCalculator:
    """颠覆性指数计算器"""

    def __init__(self):
        self.client = OpenAlexClient()

    def calculate_for_work(self, work: dict) -> Optional[Dict]:
        """
        计算单篇论文的颠覆性指数

        Args:
            work: 论文数据字典（必须包含 id, cited_by_count, referenced_works）

        Returns:
            包含计算结果的字典，或 None（计算失败时）
        """
        work_id = work.get("id")
        title = work.get("title", "Unknown Title")
        cited_by_count = work.get("cited_by_count", 0)

        if not work_id:
            print(f"  错误: 论文缺少 ID")
            return None

        # 获取目标论文的参考文献 ID 集合
        ref_ids = set(work.get("referenced_works", []))
        if not ref_ids:
            print(f"  论文 '{title[:50]}...' 没有参考文献，跳过")
            return None

        print(f"\n计算论文: {title[:60]}...")
        print(f"  ID: {work_id}, 被引次数: {cited_by_count}, 参考文献数: {len(ref_ids)}")

        # 获取施引论文
        citing_works = self.client.get_citing_works(work_id, max_results=500)

        if not citing_works:
            print(f"  没有施引论文，无法计算")
            return None

        # 统计 N_F 和 N_R
        n_f = 0  # 仅引用目标论文，不引参考文献
        n_r = 0  # 仅引参考文献，不引目标论文
        both = 0  # 两者都引（不计入公式）
        neither = 0  # 两者都不引（理论上不会出现，因为施引论文至少引了目标论文）

        for citing_work in citing_works:
            citing_refs = set(citing_work.get("referenced_works", []))

            # cites_target: 施引论文的参考文献中是否包含目标论文
            # 注意：通过 cites:{work_id} 过滤得到的施引论文，OpenAlex 保证它们引用了目标论文
            # 但 referenced_works 字段可能不完整，所以这里用 work_id in citing_refs 检查
            cites_target = work_id in citing_refs
            cites_refs = bool(citing_refs & ref_ids)  # 是否引用了目标论文的参考文献

            if cites_target and not cites_refs:
                n_f += 1
            elif cites_target and cites_refs:
                both += 1
            else:
                neither += 1

        # N_R: 仅引用目标论文的参考文献，但不引用目标论文本身的后续论文
        # 这些论文不在 cites:{work_id} 的结果中，需要额外查询
        # 为了效率，只采样查询部分参考文献（最多 20 篇）
        n_r = self.client.get_nr_count(work_id, ref_ids, max_sample=20)

        # 总被引数 C
        c = cited_by_count

        # 计算 Rela_Dz
        # 公式: Rela_Dz = (8 × N_F²) / (2 × C² + C × N_R)
        denominator = 2 * c * c + c * n_r
        if denominator == 0:
            rela_dz = 0.0
        else:
            rela_dz = (8 * n_f * n_f) / denominator

        # Citation Impact = ln(Citations + 1)
        citation_impact = math.log(c + 1)

        result = {
            "work_id": work_id,
            "title": title,
            "publication_year": work.get("publication_year"),
            "cited_by_count": c,
            "citation_impact": round(citation_impact, 4),
            "num_references": len(ref_ids),
            "num_citing_works": len(citing_works),
            "N_F": n_f,
            "N_R": n_r,
            "both": both,
            "neither": neither,
            "Rela_Dz": round(rela_dz, 6),
        }

        # ============ Discipline Variety 计算 ============
        if ENABLE_DISCIPLINE_VARIETY:
            dv_result = self.client.get_work_discipline_variety(
                work, score_threshold=DISCIPLINE_SCORE_THRESHOLD
            )
            result["discipline_variety"] = dv_result["discipline_variety"]
            result["num_authors"] = dv_result["num_authors"]
            result["author_discipline_counts"] = dv_result["author_discipline_counts"]
            # 将作者学科详情转为可读字符串（用于 CSV 输出）
            author_summary = "; ".join([
                f"{a['name']}({a['count']}:{','.join(a['disciplines'])})"
                for a in dv_result["author_disciplines"]
            ])
            result["author_disciplines_detail"] = author_summary
            print(f"  Discipline Variety: {dv_result['discipline_variety']:.4f} "
                  f"(作者数: {dv_result['num_authors']})")
        else:
            result["discipline_variety"] = 0.0
            result["num_authors"] = 0
            result["author_discipline_counts"] = []
            result["author_disciplines_detail"] = ""

        # ============ Discipline Similarity 计算 ============
        if ENABLE_DISCIPLINE_SIMILARITY:
            ds_result = self.client.get_work_discipline_similarity(
                work, score_threshold=DISCIPLINE_SCORE_THRESHOLD
            )
            result["discipline_similarity"] = ds_result["discipline_similarity"]
            result["ds_num_authors"] = ds_result["num_authors"]
            # 将成对相似度转为可读字符串
            pairwise_str = "; ".join([
                f"{p['author_i']}-{p['author_j']}:{p['similarity']}"
                for p in ds_result["pairwise_similarities"]
            ])
            result["pairwise_similarities"] = pairwise_str
            print(f"  Discipline Similarity: {ds_result['discipline_similarity']:.4f} "
                  f"(作者数: {ds_result['num_authors']})")
        else:
            result["discipline_similarity"] = 0.0
            result["ds_num_authors"] = 0
            result["pairwise_similarities"] = ""

        # ============ Discipline Balance 计算 ============
        if ENABLE_DISCIPLINE_BALANCE:
            db_result = self.client.get_work_discipline_balance(
                work, score_threshold=DISCIPLINE_SCORE_THRESHOLD
            )
            result["discipline_balance"] = db_result["discipline_balance"]
            result["db_num_authors"] = db_result["num_authors"]
            # 将排序后的频次转为可读字符串
            freq_str = ", ".join([str(f) for f in db_result["sorted_frequencies"]])
            result["sorted_frequencies"] = freq_str
            print(f"  Discipline Balance: {db_result['discipline_balance']:.4f} "
                  f"(作者数: {db_result['num_authors']}, "
                  f"频次: {freq_str})")
        else:
            result["discipline_balance"] = 0.0
            result["db_num_authors"] = 0
            result["sorted_frequencies"] = ""

        # ============ Citation-based Interdisciplinarity 计算 ============
        cit_interdisc_result = self.client.get_work_cit_interdisciplinarity(citing_works)
        result["cit_interdisciplinarity"] = cit_interdisc_result["cit_interdisciplinarity"]
        result["cit_interdisc_citing_with_disc"] = cit_interdisc_result["num_citing_with_discipline"]
        result["cit_interdisc_total_citing"] = cit_interdisc_result["total_citing"]
        print(f"  Cit Interdisciplinarity: {cit_interdisc_result['cit_interdisciplinarity']:.4f} "
              f"(有学科的施引: {cit_interdisc_result['num_citing_with_discipline']}/"
              f"{cit_interdisc_result['total_citing']})")

        # ============ AI4S_Balance 计算 ============
        if ENABLE_AI4S_BALANCE:
            ai4s_result = self.client.get_work_ai4s_balance(work)
            result["ai4s_balance"] = ai4s_result["ai4s_balance"]
            result["camp_a_count"] = ai4s_result["camp_a_count"]
            result["camp_b_count"] = ai4s_result["camp_b_count"]
            result["total_refs_ai4s"] = ai4s_result["total_refs"]
            result["p_a"] = ai4s_result["p_a"]
            result["p_b"] = ai4s_result["p_b"]
            print(f"  AI4S_Balance: {ai4s_result['ai4s_balance']:.4f} "
                  f"(P_A={ai4s_result['p_a']:.3f}, P_B={ai4s_result['p_b']:.3f}, "
                  f"CampA={ai4s_result['camp_a_count']}, CampB={ai4s_result['camp_b_count']})")
        else:
            result["ai4s_balance"] = 0.0
            result["camp_a_count"] = 0
            result["camp_b_count"] = 0
            result["total_refs_ai4s"] = 0
            result["p_a"] = 0.0
            result["p_b"] = 0.0

        # ============ AI_Ref_Age 计算 ============
        if ENABLE_AI_REF_AGE:
            ai_ref_result = self.client.get_work_ai_ref_age(work)
            result["ai_ref_age"] = ai_ref_result["ai_ref_age"]
            result["ai_ref_count"] = ai_ref_result["ai_ref_count"]
            result["total_refs"] = ai_ref_result["total_refs"]
            result["avg_ai_ref_year"] = ai_ref_result["avg_ai_ref_year"]
            result["min_ai_ref_year"] = ai_ref_result["min_ai_ref_year"]
            result["max_ai_ref_year"] = ai_ref_result["max_ai_ref_year"]
            print(f"  AI_Ref_Age: {ai_ref_result['ai_ref_age']:.2f} 年 "
                  f"(AI 参考文献: {ai_ref_result['ai_ref_count']}/{ai_ref_result['total_refs']}, "
                  f"平均年份: {ai_ref_result['avg_ai_ref_year']})")
        else:
            result["ai_ref_age"] = 0.0
            result["ai_ref_count"] = 0
            result["total_refs"] = 0
            result["avg_ai_ref_year"] = 0.0
            result["min_ai_ref_year"] = 0
            result["max_ai_ref_year"] = 0

        print(f"  结果: N_F={n_f}, N_R={n_r}, C={c}, Rela_Dz={rela_dz:.6f}")

        return result

    def calculate_batch(self, works: list) -> List[Dict]:
        """
        批量计算多篇论文的颠覆性指数

        Args:
            works: 论文数据字典列表

        Returns:
            计算结果字典列表
        """
        results = []
        total = len(works)

        for i, work in enumerate(works, 1):
            print(f"\n[{i}/{total}] 处理中...")
            result = self.calculate_for_work(work)
            if result:
                results.append(result)
            print(f"  进度: {i}/{total}")

        return results
