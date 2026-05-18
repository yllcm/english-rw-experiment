"""
指标计算模块 - 包含所有论文指标的计算逻辑

所有指标平起平坐，每个指标是一个独立方法。
颠覆性指数 (Rela_Dz) 只是众多指标之一。
"""

import math
import random
from datetime import datetime
from typing import Dict, List, Optional

from api_client import OpenAlexClient
from config import (
    ENABLE_DISCIPLINE_VARIETY,
    ENABLE_DISCIPLINE_SIMILARITY,
    ENABLE_DISCIPLINE_BALANCE,
    ENABLE_AI4S_BALANCE,
    ENABLE_AI_REF_AGE,
    DISCIPLINE_SCORE_THRESHOLD,
    AI4S_CAMP_A,
    AI4S_CAMP_B,
    AI_REF_SCORE_THRESHOLD,
    AI_CAMP_A_SCORE_MIN,
    AI_NON_CAMP_A_SCORE_MAX,
    AI_REF_BATCH_SIZE,
)

# ============ 19 个一级学科（OpenAlex level 0 concepts）============
LEVEL0_DISCIPLINES = [
    "Art", "Biology", "Business", "Chemistry", "Computer science",
    "Economics", "Engineering", "Environmental science", "Geography",
    "Geology", "History", "Materials science", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Political science",
    "Psychology", "Sociology",
]

LEVEL0_ID_TO_INDEX = {
    "142362112": 0,   # Art
    "86803240": 1,    # Biology
    "144133560": 2,   # Business
    "185592680": 3,   # Chemistry
    "41008148": 4,    # Computer science
    "162324750": 5,   # Economics
    "127413603": 6,   # Engineering
    "39432304": 7,    # Environmental science
    "205649164": 8,   # Geography
    "127313418": 9,   # Geology
    "95457728": 10,   # History
    "192562407": 11,  # Materials science
    "33923547": 12,   # Mathematics
    "71924100": 13,   # Medicine
    "138885662": 14,  # Philosophy
    "121332964": 15,  # Physics
    "17744445": 16,   # Political science
    "15744967": 17,   # Psychology
    "144024400": 18,  # Sociology
}


class MetricsCalculator:
    """论文指标计算器"""

    def __init__(self):
        self.client = OpenAlexClient()

    # ================================================================
    # 颠覆性指数 (Rela_Dz)
    # 公式: Rela_Dz = (8 × N_F²) / (2 × C² + C × N_R)
    # ================================================================

    def compute_disruptiveness(self, work: dict) -> Optional[Dict]:
        """
        计算单篇论文的颠覆性指数 Rela_Dz

        Args:
            work: 论文数据字典

        Returns:
            包含 Rela_Dz 及相关字段的字典
        """
        work_id = work.get("id")
        title = work.get("title", "Unknown Title")
        cited_by_count = work.get("cited_by_count", 0)

        if not work_id:
            print(f"  错误: 论文缺少 ID")
            return None

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
        n_f = 0
        n_r = 0
        both = 0
        neither = 0

        for citing_work in citing_works:
            citing_refs = set(citing_work.get("referenced_works", []))
            cites_target = work_id in citing_refs
            cites_refs = bool(citing_refs & ref_ids)

            if cites_target and not cites_refs:
                n_f += 1
            elif cites_target and cites_refs:
                both += 1
            else:
                neither += 1

        n_r = self.client.get_nr_count(work_id, ref_ids, max_sample=20)

        c = cited_by_count
        denominator = 2 * c * c + c * n_r
        rela_dz = (8 * n_f * n_f) / denominator if denominator != 0 else 0.0

        citation_impact = math.log(c + 1)

        print(f"  结果: N_F={n_f}, N_R={n_r}, C={c}, Rela_Dz={rela_dz:.6f}")

        return {
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

    # ================================================================
    # Discipline Variety - 团队学科多样性
    # ================================================================

    def compute_discipline_variety(self, work: dict) -> dict:
        """
        计算一篇论文的 Discipline Variety

        Args:
            work: 论文数据字典

        Returns:
            {"discipline_variety": float, "author_disciplines": list, ...}
        """
        authorships = work.get("authorships", [])
        if not authorships:
            return {
                "discipline_variety": 0.0,
                "author_disciplines": [],
                "num_authors": 0,
                "author_discipline_counts": []
            }

        m = len(authorships)
        author_discipline_counts = []
        author_disciplines_detail = []

        for authorship in authorships:
            author_info = authorship.get("author", {})
            author_id = author_info.get("id", "")

            if not author_id:
                author_discipline_counts.append(0)
                author_disciplines_detail.append({
                    "name": authorship.get("raw_author_name", "Unknown"),
                    "disciplines": [],
                    "count": 0
                })
                continue

            author_data = self.client.get_author_info(author_id)
            disciplines = []
            if author_data:
                x_concepts = author_data.get("x_concepts", [])
                disciplines = [
                    c["display_name"]
                    for c in x_concepts
                    if c.get("score", 0) >= DISCIPLINE_SCORE_THRESHOLD
                ]
            d_i = len(disciplines)
            author_discipline_counts.append(d_i)
            author_disciplines_detail.append({
                "name": author_info.get("display_name", "Unknown"),
                "disciplines": disciplines,
                "count": d_i
            })

        total_disciplines = sum(author_discipline_counts)
        variety = total_disciplines / m if m > 0 else 0.0

        return {
            "discipline_variety": round(variety, 4),
            "author_disciplines": author_disciplines_detail,
            "num_authors": m,
            "author_discipline_counts": author_discipline_counts
        }

    # ================================================================
    # Discipline Similarity - 团队学科相似性
    # ================================================================

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _get_author_discipline_vector(self, author_id: str) -> List[int]:
        """
        获取某位作者的 19 维学科背景特征向量（0/1 数组）
        """
        # 利用 api_client 的缓存
        author_data = self.client.get_author_info(author_id)
        if not author_data:
            return [0] * 19

        x_concepts = author_data.get("x_concepts", [])

        vector = [0] * 19
        for c in x_concepts:
            cid = str(c.get("id", ""))
            score = c.get("score", 0)
            if score >= DISCIPLINE_SCORE_THRESHOLD and cid in LEVEL0_ID_TO_INDEX:
                idx = LEVEL0_ID_TO_INDEX[cid]
                vector[idx] = 1

        return vector

    def compute_discipline_similarity(self, work: dict) -> dict:
        """
        计算一篇论文的 Discipline Similarity

        公式:
            Discipline Similarity = sum(s_i) / m
            s_i = sum(cos(d_i, d_j)) / (m - 1)  for j != i

        Args:
            work: 论文数据字典

        Returns:
            {"discipline_similarity": float, ...}
        """
        authorships = work.get("authorships", [])
        if not authorships:
            return {
                "discipline_similarity": 0.0,
                "num_authors": 0,
                "author_vectors": [],
                "pairwise_similarities": []
            }

        m = len(authorships)
        if m <= 1:
            return {
                "discipline_similarity": 0.0,
                "num_authors": m,
                "author_vectors": [],
                "pairwise_similarities": []
            }

        author_vectors = []
        author_names = []
        for authorship in authorships:
            author_info = authorship.get("author", {})
            author_id = author_info.get("id", "")
            name = author_info.get("display_name", "Unknown")
            author_names.append(name)

            if not author_id:
                author_vectors.append([0] * 19)
            else:
                vec = self._get_author_discipline_vector(author_id)
                author_vectors.append(vec)

        pairwise = []
        for i in range(m):
            for j in range(i + 1, m):
                sim = self._cosine_similarity(author_vectors[i], author_vectors[j])
                pairwise.append({
                    "author_i": author_names[i],
                    "author_j": author_names[j],
                    "similarity": round(sim, 4)
                })

        s_values = []
        for i in range(m):
            total_sim = 0.0
            for j in range(m):
                if i != j:
                    total_sim += self._cosine_similarity(
                        author_vectors[i], author_vectors[j]
                    )
            s_i = total_sim / (m - 1)
            s_values.append(round(s_i, 4))

        similarity = sum(s_values) / m

        return {
            "discipline_similarity": round(similarity, 4),
            "num_authors": m,
            "author_names": author_names,
            "author_vectors": author_vectors,
            "s_values": s_values,
            "pairwise_similarities": pairwise
        }

    # ================================================================
    # Discipline Balance - 团队学科均衡性
    # ================================================================

    def compute_discipline_balance(self, work: dict) -> dict:
        """
        计算一篇论文的 Discipline Balance

        公式:
            Discipline Balance = 1 - sum((2i - 20) * x_i) / (19 * sum(x_i))

        Args:
            work: 论文数据字典

        Returns:
            {"discipline_balance": float, ...}
        """
        authorships = work.get("authorships", [])
        if not authorships:
            return {
                "discipline_balance": 0.0,
                "num_authors": 0,
                "discipline_frequencies": [],
                "sorted_frequencies": []
            }

        m = len(authorships)

        freq = [0] * 19
        for authorship in authorships:
            author_info = authorship.get("author", {})
            author_id = author_info.get("id", "")
            if author_id:
                vec = self._get_author_discipline_vector(author_id)
                for k in range(19):
                    freq[k] += vec[k]

        sorted_freq = sorted(freq, reverse=True)

        numerator = 0.0
        for i in range(19):
            weight = 2 * (i + 1) - 20
            numerator += weight * sorted_freq[i]

        total_freq = sum(sorted_freq)
        denominator = 19 * total_freq if total_freq > 0 else 1
        balance = 1.0 - (numerator / denominator)

        return {
            "discipline_balance": round(balance, 4),
            "num_authors": m,
            "discipline_frequencies": freq,
            "sorted_frequencies": sorted_freq,
            "total_discipline_occurrences": total_freq
        }

    # ================================================================
    # Citation-based Interdisciplinarity - 基于引用的跨学科性
    # ================================================================

    def compute_cit_interdisciplinarity(self, citing_works: list) -> dict:
        """
        计算一篇论文的 Citation-based Interdisciplinarity

        公式:
            Cit Interdisciplinarity_i = 1 - sum(q_k^2)

        Args:
            citing_works: 施引论文列表

        Returns:
            {"cit_interdisciplinarity": float, ...}
        """
        if not citing_works:
            return {
                "cit_interdisciplinarity": 0.0,
                "num_citing_with_discipline": 0,
                "total_citing": 0,
                "discipline_distribution": {}
            }

        discipline_counts = {name: 0 for name in LEVEL0_DISCIPLINES}
        total_with_discipline = 0

        for citing in citing_works:
            concepts = citing.get("concepts", [])
            if not concepts:
                continue

            best_score = -1
            best_discipline = None
            for c in concepts:
                level = c.get("level", -1)
                if level != 0:
                    continue
                score = c.get("score", 0)
                name = c.get("display_name", "")
                if score > best_score:
                    best_score = score
                    best_discipline = name

            if best_discipline and best_discipline in discipline_counts:
                discipline_counts[best_discipline] += 1
                total_with_discipline += 1

        if total_with_discipline == 0:
            return {
                "cit_interdisciplinarity": 0.0,
                "num_citing_with_discipline": 0,
                "total_citing": len(citing_works),
                "discipline_distribution": discipline_counts
            }

        sum_sq = 0.0
        distribution = {}
        for name, count in discipline_counts.items():
            q_k = count / total_with_discipline
            distribution[name] = round(q_k, 4)
            sum_sq += q_k * q_k

        cit_interdisciplinarity = 1.0 - sum_sq

        return {
            "cit_interdisciplinarity": round(cit_interdisciplinarity, 4),
            "num_citing_with_discipline": total_with_discipline,
            "total_citing": len(citing_works),
            "discipline_distribution": distribution
        }

    # ================================================================
    # AI4S_Balance - AI for Science 双向融合度
    # ================================================================

    def _is_camp_a_work(self, work_info: dict,
                        camp_a: set = None,
                        score_threshold: float = None) -> bool:
        """判断一篇论文是否属于阵营 A（AI/计算机类）"""
        if camp_a is None:
            camp_a = AI4S_CAMP_A
        if score_threshold is None:
            score_threshold = AI_REF_SCORE_THRESHOLD

        concepts = work_info.get("concepts", [])
        if not concepts:
            return False

        if score_threshold is not None:
            for c in concepts:
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
                if (pure_cid in camp_a or cid in camp_a) and score >= score_threshold:
                    return True
            return False
        else:
            has_high_cs = False
            all_non_cs_low = True

            for c in concepts:
                level = c.get("level", -1)
                if level != 0:
                    continue
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid

                if pure_cid in camp_a or cid in camp_a:
                    if score >= AI_CAMP_A_SCORE_MIN:
                        has_high_cs = True
                else:
                    if score > AI_NON_CAMP_A_SCORE_MAX:
                        all_non_cs_low = False

            return has_high_cs and all_non_cs_low

    def _is_camp_b_work(self, work_info: dict,
                        camp_b: set = None,
                        score_threshold: float = None) -> bool:
        """判断一篇论文是否属于阵营 B（基础科学类）"""
        if camp_b is None:
            camp_b = AI4S_CAMP_B
        if score_threshold is None:
            score_threshold = AI_REF_SCORE_THRESHOLD

        concepts = work_info.get("concepts", [])
        if not concepts:
            return False

        if score_threshold is not None:
            for c in concepts:
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
                if (pure_cid in camp_b or cid in camp_b) and score >= score_threshold:
                    return True
            return False
        else:
            has_high_camp_b = False
            all_non_camp_b_low = True

            for c in concepts:
                level = c.get("level", -1)
                if level != 0:
                    continue
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid

                if pure_cid in camp_b or cid in camp_b:
                    if score >= AI_CAMP_A_SCORE_MIN:
                        has_high_camp_b = True
                else:
                    if score > AI_NON_CAMP_A_SCORE_MAX:
                        all_non_camp_b_low = False

            return has_high_camp_b and all_non_camp_b_low

    def compute_ai4s_balance(self, work: dict) -> dict:
        """
        计算一篇论文的 AI4S_Balance

        公式:
            AI4S_Balance = 1 - |P_A - P_B|
            P_A = N_A / N_total, P_B = N_B / N_total

        Args:
            work: 论文数据字典

        Returns:
            {"ai4s_balance": float, ...}
        """
        ref_ids = work.get("referenced_works", [])
        if not ref_ids:
            return {
                "ai4s_balance": 0.0,
                "camp_a_count": 0,
                "camp_b_count": 0,
                "total_refs": 0,
                "p_a": 0.0,
                "p_b": 0.0,
                "camp_a_refs": [],
                "camp_b_refs": [],
                "neither_refs": []
            }

        ref_infos = self.client.batch_get_works(ref_ids, AI_REF_BATCH_SIZE)

        camp_a_count = 0
        camp_b_count = 0
        camp_a_refs = []
        camp_b_refs = []
        neither_refs = []

        for ref_id, ref_info in ref_infos.items():
            is_a = self._is_camp_a_work(ref_info)
            is_b = self._is_camp_b_work(ref_info)

            if is_a:
                camp_a_count += 1
                camp_a_refs.append(ref_id)
            if is_b:
                camp_b_count += 1
                camp_b_refs.append(ref_id)
            if not is_a and not is_b:
                neither_refs.append(ref_id)

        total_refs = len(ref_ids)
        if total_refs == 0:
            return {
                "ai4s_balance": 0.0,
                "camp_a_count": camp_a_count,
                "camp_b_count": camp_b_count,
                "total_refs": total_refs,
                "p_a": 0.0,
                "p_b": 0.0,
                "camp_a_refs": camp_a_refs,
                "camp_b_refs": camp_b_refs,
                "neither_refs": neither_refs
            }

        p_a = camp_a_count / total_refs
        p_b = camp_b_count / total_refs
        balance = 1.0 - abs(p_a - p_b)

        return {
            "ai4s_balance": round(balance, 4),
            "camp_a_count": camp_a_count,
            "camp_b_count": camp_b_count,
            "total_refs": len(ref_ids),
            "p_a": round(p_a, 4),
            "p_b": round(p_b, 4),
            "camp_a_refs": camp_a_refs,
            "camp_b_refs": camp_b_refs,
            "neither_refs": neither_refs
        }

    # ================================================================
    # AI_Ref_Age - AI 技术时效性
    # ================================================================

    def compute_ai_ref_age(self, work: dict) -> dict:
        """
        计算一篇论文的 AI_Ref_Age（AI 技术时效性）

        公式:
            AI_Ref_Age = 论文发表年份 - 平均(所有阵营A参考文献的发表年份)

        Args:
            work: 论文数据字典

        Returns:
            {"ai_ref_age": float, ...}
        """
        pub_year = work.get("publication_year")
        if not pub_year:
            return {
                "ai_ref_age": 0.0,
                "ai_ref_count": 0,
                "ai_ref_years": [],
                "total_refs": 0,
                "avg_ai_ref_year": 0.0,
                "min_ai_ref_year": 0,
                "max_ai_ref_year": 0
            }

        ref_ids = work.get("referenced_works", [])
        if not ref_ids:
            return {
                "ai_ref_age": 0.0,
                "ai_ref_count": 0,
                "ai_ref_years": [],
                "total_refs": 0,
                "avg_ai_ref_year": 0.0,
                "min_ai_ref_year": 0,
                "max_ai_ref_year": 0
            }

        ref_infos = self.client.batch_get_works(ref_ids, AI_REF_BATCH_SIZE)

        ai_ref_years = []
        for ref_id, ref_info in ref_infos.items():
            if self._is_camp_a_work(ref_info):
                year = ref_info.get("publication_year")
                # 过滤异常年份：参考文献的发表年份不能晚于论文本身的发表年份
                if year and year <= pub_year:
                    ai_ref_years.append(year)

        if not ai_ref_years:
            return {
                "ai_ref_age": 0.0,
                "ai_ref_count": 0,
                "ai_ref_years": [],
                "total_refs": len(ref_ids),
                "avg_ai_ref_year": 0.0,
                "min_ai_ref_year": 0,
                "max_ai_ref_year": 0
            }

        avg_ai_ref_year = sum(ai_ref_years) / len(ai_ref_years)
        ai_ref_age = pub_year - avg_ai_ref_year

        return {
            "ai_ref_age": round(ai_ref_age, 2),
            "ai_ref_count": len(ai_ref_years),
            "total_refs": len(ref_ids),
            "avg_ai_ref_year": round(avg_ai_ref_year, 1),
            "min_ai_ref_year": min(ai_ref_years),
            "max_ai_ref_year": max(ai_ref_years),
            "ai_ref_years": sorted(ai_ref_years)
        }

    # ================================================================
    # 综合计算 - 计算一篇论文的所有指标
    # ================================================================

    def _compute_disruptiveness_inner(self, work_id: str, title: str,
                                       cited_by_count: int, ref_ids: set,
                                       citing_works: list) -> dict:
        """
        颠覆性指数内部计算（供 compute_all_metrics 使用，避免重复获取施引论文）

        Args:
            work_id: 论文 ID
            title: 论文标题
            cited_by_count: 被引次数
            ref_ids: 参考文献 ID 集合
            citing_works: 施引论文列表

        Returns:
            包含 Rela_Dz 及相关字段的字典
        """
        n_f = 0
        n_r = 0
        both = 0
        neither = 0

        for citing_work in citing_works:
            citing_refs = set(citing_work.get("referenced_works", []))
            cites_target = work_id in citing_refs
            cites_refs = bool(citing_refs & ref_ids)

            if cites_target and not cites_refs:
                n_f += 1
            elif cites_target and cites_refs:
                both += 1
            else:
                neither += 1

        n_r = self.client.get_nr_count(work_id, ref_ids, max_sample=20)

        c = cited_by_count
        denominator = 2 * c * c + c * n_r
        rela_dz = (8 * n_f * n_f) / denominator if denominator != 0 else 0.0

        citation_impact = math.log(c + 1)

        print(f"  结果: N_F={n_f}, N_R={n_r}, C={c}, Rela_Dz={rela_dz:.6f}")

        return {
            "work_id": work_id,
            "title": title,
            "publication_year": None,  # 由调用者补充
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

    def compute_all_metrics(self, work: dict) -> Optional[Dict]:
        """
        计算单篇论文的所有指标

        Args:
            work: 论文数据字典

        Returns:
            包含所有指标的字典，或 None（计算失败时）
        """
        work_id = work.get("id")
        title = work.get("title", "Unknown Title")
        cited_by_count = work.get("cited_by_count", 0)

        if not work_id:
            print(f"  错误: 论文缺少 ID")
            return None

        ref_ids = set(work.get("referenced_works", []))
        if not ref_ids:
            print(f"  论文 '{title[:50]}...' 没有参考文献，跳过")
            return None

        print(f"\n计算论文: {title[:60]}...")
        print(f"  ID: {work_id}, 被引次数: {cited_by_count}, 参考文献数: {len(ref_ids)}")

        # 获取施引论文（只需一次，多个指标共用）
        citing_works = self.client.get_citing_works(work_id, max_results=500)

        if not citing_works:
            print(f"  没有施引论文，无法计算")
            return None

        # 1. 颠覆性指数
        result = self._compute_disruptiveness_inner(
            work_id, title, cited_by_count, ref_ids, citing_works
        )
        result["publication_year"] = work.get("publication_year")

        # 2. Discipline Variety
        if ENABLE_DISCIPLINE_VARIETY:
            dv_result = self.compute_discipline_variety(work)
            result["discipline_variety"] = dv_result["discipline_variety"]
            result["num_authors"] = dv_result["num_authors"]
            result["author_discipline_counts"] = dv_result["author_discipline_counts"]
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

        # 3. Discipline Similarity
        if ENABLE_DISCIPLINE_SIMILARITY:
            ds_result = self.compute_discipline_similarity(work)
            result["discipline_similarity"] = ds_result["discipline_similarity"]
            result["ds_num_authors"] = ds_result["num_authors"]
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

        # 4. Discipline Balance
        if ENABLE_DISCIPLINE_BALANCE:
            db_result = self.compute_discipline_balance(work)
            result["discipline_balance"] = db_result["discipline_balance"]
            result["db_num_authors"] = db_result["num_authors"]
            freq_str = ", ".join([str(f) for f in db_result["sorted_frequencies"]])
            result["sorted_frequencies"] = freq_str
            print(f"  Discipline Balance: {db_result['discipline_balance']:.4f} "
                  f"(作者数: {db_result['num_authors']}, "
                  f"频次: {freq_str})")
        else:
            result["discipline_balance"] = 0.0
            result["db_num_authors"] = 0
            result["sorted_frequencies"] = ""

        # 5. Citation-based Interdisciplinarity（复用已获取的 citing_works）
        cit_result = self.compute_cit_interdisciplinarity(citing_works)
        result["cit_interdisciplinarity"] = cit_result["cit_interdisciplinarity"]
        result["cit_interdisc_citing_with_disc"] = cit_result["num_citing_with_discipline"]
        result["cit_interdisc_total_citing"] = cit_result["total_citing"]
        print(f"  Cit Interdisciplinarity: {cit_result['cit_interdisciplinarity']:.4f} "
              f"(有学科的施引: {cit_result['num_citing_with_discipline']}/"
              f"{cit_result['total_citing']})")

        # 6. AI4S_Balance
        if ENABLE_AI4S_BALANCE:
            ai4s_result = self.compute_ai4s_balance(work)
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

        # 7. AI_Ref_Age
        if ENABLE_AI_REF_AGE:
            ai_ref_result = self.compute_ai_ref_age(work)
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

        return result

    def compute_batch(self, works: list) -> List[Dict]:
        """
        批量计算多篇论文的所有指标

        Args:
            works: 论文数据字典列表

        Returns:
            计算结果字典列表
        """
        results = []
        total = len(works)

        for i, work in enumerate(works, 1):
            print(f"\n[{i}/{total}] 处理中...")
            result = self.compute_all_metrics(work)
            if result:
                results.append(result)
            print(f"  进度: {i}/{total}")

        return results
