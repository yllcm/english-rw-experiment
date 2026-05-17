"""
OpenAlex API 客户端 - 负责数据爬取
"""

import time
import math
import random
import requests
from typing import Optional, List

from config import (
    OPENALEX_BASE_URL,
    MAILTO,
    MAX_RETRIES,
    TIMEOUT,
    PER_PAGE,
    RANDOM_SEED,
    RANDOM_SAMPLE_FACTOR,
)

# ============ 19 个一级学科（OpenAlex level 0 concepts）============
# 按字母顺序排列，固定顺序用于构建特征向量
LEVEL0_DISCIPLINES = [
    "Art", "Biology", "Business", "Chemistry", "Computer science",
    "Economics", "Engineering", "Environmental science", "Geography",
    "Geology", "History", "Materials science", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Political science",
    "Psychology", "Sociology",
]

# 一级学科 ID 到索引的映射（用于快速查找）
# 作者 x_concepts 中的 ID 是纯数字（如 41008148），
# 一级学科的 ID 是 C41008148，去掉 C 前缀即可匹配
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


class OpenAlexClient:
    """OpenAlex API 客户端"""

    def __init__(self):
        self.base_url = OPENALEX_BASE_URL
        self.session = requests.Session()
        # 设置请求头，模拟浏览器访问
        self.session.headers.update({
            "User-Agent": "OpenAlexDisruptiveness/1.0 (mailto:{})".format(
                MAILTO if MAILTO else "anonymous@example.com"
            )
        })

    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """
        发送 API 请求，包含重试机制和速率限制

        Args:
            url: API 端点 URL
            params: 查询参数

        Returns:
            API 响应 JSON 或 None（失败时）
        """
        if params is None:
            params = {}

        # 如果配置了邮箱，添加到参数中以获得更高速率限制
        if MAILTO:
            params["mailto"] = MAILTO

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    url, params=params, timeout=TIMEOUT
                )
                # 检查是否被限流
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"  被限流，等待 {retry_after} 秒后重试...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"  请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  请求最终失败: {url}")
                    return None

        return None

    def search_works(self, query: str, year_from: int = None,
                     year_to: int = None, max_results: int = 200,
                     per_page: int = 200,
                     concept_filter: str = None) -> list:
        """
        搜索论文列表（支持随机采样）

        Args:
            query: 搜索关键词
            year_from: 起始年份
            year_to: 结束年份
            max_results: 最大返回结果数
            per_page: 每页数量
            concept_filter: 学科过滤条件（OpenAlex concept ID，用 | 分隔）

        Returns:
            论文列表
        """
        all_works = []
        cursor = "*"  # OpenAlex 使用游标分页

        # 构建过滤条件
        filters = [f"title.search:{query}"]
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if concept_filter:
            filters.append(f"concepts.id:{concept_filter}")

        # 随机采样：搜索更多候选论文，然后从中随机抽取
        # 这样每次跑出的论文都不同
        sample_max = max_results * RANDOM_SAMPLE_FACTOR

        params = {
            "filter": ",".join(filters),
            "per_page": min(per_page, 200),
            "sort": "cited_by_count:desc",
            "cursor": cursor,
            "select": "id,doi,title,authorships,publication_year,cited_by_count,referenced_works,primary_location,concepts"
        }

        # 构建显示用的过滤描述
        concept_desc = ""
        if concept_filter:
            concept_names = {
                "C86803240": "Biology", "C185592680": "Chemistry",
                "C121332964": "Physics", "C71924100": "Medicine",
                "C192562407": "Materials science", "C41008148": "Computer science",
                "C33923547": "Mathematics", "C127413603": "Engineering",
                "C142362112": "Art", "C144133560": "Business",
                "C162324750": "Economics", "C39432304": "Environmental science",
                "C205649164": "Geography", "C127313418": "Geology",
                "C95457728": "History", "C138885662": "Philosophy",
                "C17744445": "Political science", "C15744967": "Psychology",
                "C144024400": "Sociology"
            }
            names = [concept_names.get(cid, cid) for cid in concept_filter.split("|")]
            concept_desc = f" (学科: {', '.join(names)})"

        print(f"正在搜索论文: '{query}'{concept_desc} (年份: {year_from}-{year_to})")
        print(f"  随机采样模式: 搜索 {sample_max} 篇候选 → 随机抽取 {max_results} 篇")

        while cursor and len(all_works) < sample_max:
            params["cursor"] = cursor
            url = f"{self.base_url}/works"

            data = self._make_request(url, params)
            if data is None:
                break

            results = data.get("results", [])
            all_works.extend(results)

            print(f"  已获取 {len(all_works)} 篇论文...")

            # 获取下一页游标
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")

            # 礼貌性延迟，避免触发限流
            time.sleep(0.1)

        # 从候选论文中随机抽取 max_results 篇
        if len(all_works) > max_results:
            # 设置随机种子（如果配置了）
            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)
            all_works = random.sample(all_works, max_results)
            print(f"  从 {len(all_works)} 篇候选中随机抽取 {max_results} 篇")
        else:
            all_works = all_works[:max_results]

        print(f"搜索完成，共获取 {len(all_works)} 篇论文")
        return all_works

    def get_work_by_id(self, work_id: str) -> Optional[dict]:
        """
        根据 OpenAlex ID 获取单篇论文详情

        Args:
            work_id: OpenAlex 论文 ID (例如: W123456789)

        Returns:
            论文详情
        """
        url = f"{self.base_url}/works/{work_id}"
        params = {
            "select": "id,doi,title,publication_year,cited_by_count,referenced_works"
        }
        return self._make_request(url, params)

    def get_citing_works(self, work_id: str, max_results: int = 500) -> list:
        """
        获取引用某篇论文的所有施引论文

        Args:
            work_id: 目标论文的 OpenAlex ID
            max_results: 最大返回结果数

        Returns:
            施引论文列表（包含其参考文献列表）
        """
        citing_works = []
        cursor = "*"

        params = {
            "filter": f"cites:{work_id}",
            "per_page": 200,
            "cursor": cursor,
            "select": "id,title,publication_year,referenced_works,concepts"
        }

        print(f"  正在获取引用论文 (被引: {work_id})...")

        while cursor and len(citing_works) < max_results:
            params["cursor"] = cursor
            url = f"{self.base_url}/works"

            data = self._make_request(url, params)
            if data is None:
                break

            results = data.get("results", [])
            citing_works.extend(results)

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")

            # 礼貌性延迟
            time.sleep(0.1)

        citing_works = citing_works[:max_results]
        print(f"  获取到 {len(citing_works)} 篇施引论文")
        return citing_works

    def get_nr_count(self, target_id: str, ref_ids: set,
                     max_sample: int = 20) -> int:
        """
        计算 N_R: 仅引用目标论文的参考文献，但不引用目标论文本身的后续论文数量

        通过采样查询目标论文的部分参考文献，统计引用这些参考文献但不引用目标论文的论文数。
        为了效率，只采样 max_sample 篇参考文献，并对结果进行放大。

        放大公式:
            N_R = sampled_nr * (total_refs / sampled_refs)

        其中:
            sampled_nr = 从采样参考文献中统计到的 N_R
            total_refs = 参考文献总数
            sampled_refs = 实际采样的参考文献数

        Args:
            target_id: 目标论文的 OpenAlex ID
            ref_ids: 目标论文的参考文献 ID 集合
            max_sample: 最多采样查询的参考文献数量

        Returns:
            N_R 值（放大后的估计值）
        """
        if not ref_ids:
            return 0

        total_refs = len(ref_ids)

        # 采样部分参考文献进行查询
        ref_list = list(ref_ids)
        if len(ref_list) > max_sample:
            ref_list = random.sample(ref_list, max_sample)

        sampled_refs = len(ref_list)

        # 收集所有引用这些参考文献的论文（去重）
        nr_paper_ids = set()

        for ref_id in ref_list:
            pure_ref_id = ref_id.strip("/").split("/")[-1]
            url = f"{self.base_url}/works"
            params = {
                "filter": f"cites:{pure_ref_id}",
                "select": "id,referenced_works",
                "per_page": 200
            }

            data = self._make_request(url, params)
            if data is None:
                continue

            results = data.get("results", [])
            for citing in results:
                citing_id = citing.get("id", "")
                citing_refs = set(citing.get("referenced_works", []))
                # 如果这篇施引论文没有引用目标论文，则计入 N_R
                if target_id not in citing_refs:
                    nr_paper_ids.add(citing_id)

            time.sleep(0.1)

        # 对采样结果进行放大
        sampled_nr = len(nr_paper_ids)
        if sampled_refs < total_refs and sampled_nr > 0:
            n_r = int(round(sampled_nr * total_refs / sampled_refs))
        else:
            n_r = sampled_nr

        return n_r

    def get_work_references(self, work: dict) -> list:
        """
        从论文数据中提取参考文献 ID 列表

        Args:
            work: 论文数据字典

        Returns:
            参考文献 OpenAlex ID 列表
        """
        return work.get("referenced_works", [])

    # ============ 作者学科相关 ============
    _author_cache: dict = {}  # 作者信息缓存，避免重复请求

    def get_author_disciplines(self, author_id: str,
                               score_threshold: float = 0.5) -> list:
        """
        获取某位作者的主学科列表（基于 x_concepts）

        Args:
            author_id: 作者的 OpenAlex ID (例如: https://openalex.org/A5030978914)
            score_threshold: 学科得分阈值，仅保留 score >= 此值的学科

        Returns:
            主学科名称列表
        """
        # 检查缓存
        if author_id in self._author_cache:
            return self._author_cache[author_id]

        # 从 author_id 中提取纯 ID（例如 "A5030978914"）
        pure_id = author_id.strip("/").split("/")[-1]

        url = f"{self.base_url}/authors/{pure_id}"
        params = {
            "select": "id,display_name,x_concepts"
        }

        data = self._make_request(url, params)
        if data is None:
            print(f"  警告: 无法获取作者信息 {author_id}")
            self._author_cache[author_id] = []
            return []

        x_concepts = data.get("x_concepts", [])

        # 筛选得分 >= 阈值的学科
        disciplines = [
            c["display_name"]
            for c in x_concepts
            if c.get("score", 0) >= score_threshold
        ]

        # 存入缓存
        self._author_cache[author_id] = disciplines
        return disciplines

    def get_work_discipline_variety(self, work: dict,
                                    score_threshold: float = 0.5) -> dict:
        """
        计算一篇论文的 Discipline Variety

        Args:
            work: 论文数据字典
            score_threshold: 学科得分阈值

        Returns:
            {"discipline_variety": float, "author_disciplines": list, "num_authors": int}
        """
        authorships = work.get("authorships", [])
        if not authorships:
            return {
                "discipline_variety": 0.0,
                "author_disciplines": [],
                "num_authors": 0,
                "author_discipline_counts": []
            }

        m = len(authorships)  # 作者总人数
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

            disciplines = self.get_author_disciplines(author_id, score_threshold)
            d_i = len(disciplines)
            author_discipline_counts.append(d_i)
            author_disciplines_detail.append({
                "name": author_info.get("display_name", "Unknown"),
                "disciplines": disciplines,
                "count": d_i
            })

        # Discipline Variety = sum(d_i) / m
        total_disciplines = sum(author_discipline_counts)
        variety = total_disciplines / m if m > 0 else 0.0

        return {
            "discipline_variety": round(variety, 4),
            "author_disciplines": author_disciplines_detail,
            "num_authors": m,
            "author_discipline_counts": author_discipline_counts
        }

    # ============ Discipline Similarity 相关 ============

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec_a: 向量 A
            vec_b: 向量 B

        Returns:
            余弦相似度（范围 0~1）
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_author_discipline_vector(self, author_id: str,
                                     score_threshold: float = 0.5) -> List[int]:
        """
        获取某位作者的 19 维学科背景特征向量（0/1 数组）

        Args:
            author_id: 作者的 OpenAlex ID
            score_threshold: 学科得分阈值

        Returns:
            19 维 0/1 向量，表示该作者涉足的一级学科
        """
        # 先获取作者的 x_concepts 数据（利用已有的缓存机制）
        pure_id = author_id.strip("/").split("/")[-1]

        # 检查缓存中是否已有向量
        cache_key = f"vec_{author_id}"
        if cache_key in self._author_cache:
            return self._author_cache[cache_key]

        url = f"{self.base_url}/authors/{pure_id}"
        params = {"select": "id,display_name,x_concepts"}

        data = self._make_request(url, params)
        if data is None:
            return [0] * 19

        x_concepts = data.get("x_concepts", [])

        # 构建 19 维 0/1 向量
        vector = [0] * 19
        for c in x_concepts:
            cid = str(c.get("id", ""))
            score = c.get("score", 0)
            if score >= score_threshold and cid in LEVEL0_ID_TO_INDEX:
                idx = LEVEL0_ID_TO_INDEX[cid]
                vector[idx] = 1

        # 存入缓存
        self._author_cache[cache_key] = vector
        return vector

    def get_work_discipline_similarity(self, work: dict,
                                       score_threshold: float = 0.5) -> dict:
        """
        计算一篇论文的 Discipline Similarity

        公式:
            Discipline Similarity = sum(s_i) / m
            s_i = sum(cos(d_i, d_j)) / (m - 1)  for j != i

        Args:
            work: 论文数据字典
            score_threshold: 学科得分阈值

        Returns:
            {"discipline_similarity": float, "pairwise_similarities": list, ...}
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
            # 只有 1 个作者时，相似度定义为 0
            return {
                "discipline_similarity": 0.0,
                "num_authors": m,
                "author_vectors": [],
                "pairwise_similarities": []
            }

        # 获取所有作者的 19 维向量
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
                vec = self.get_author_discipline_vector(author_id, score_threshold)
                author_vectors.append(vec)

        # 计算所有作者对的余弦相似度
        pairwise = []
        for i in range(m):
            for j in range(i + 1, m):
                sim = self._cosine_similarity(author_vectors[i], author_vectors[j])
                pairwise.append({
                    "author_i": author_names[i],
                    "author_j": author_names[j],
                    "similarity": round(sim, 4)
                })

        # 计算每个作者的 s_i = 与其他所有作者的平均相似度
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

        # Discipline Similarity = sum(s_i) / m
        similarity = sum(s_values) / m

        return {
            "discipline_similarity": round(similarity, 4),
            "num_authors": m,
            "author_names": author_names,
            "author_vectors": author_vectors,
            "s_values": s_values,
            "pairwise_similarities": pairwise
        }

    # ============ Discipline Balance 相关 ============

    def get_work_discipline_balance(self, work: dict,
                                    score_threshold: float = 0.5) -> dict:
        """
        计算一篇论文的 Discipline Balance

        公式:
            Discipline Balance = 1 - sum((2i - 20) * x_i) / (19 * sum(x_i))

        其中 x_i 是排序后第 i 个学科在团队中出现的总频次（i 从 1 到 19）

        Args:
            work: 论文数据字典
            score_threshold: 学科得分阈值

        Returns:
            {"discipline_balance": float, "discipline_frequencies": list, ...}
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

        # 获取所有作者的 19 维向量，按列求和得到频次
        freq = [0] * 19
        for authorship in authorships:
            author_info = authorship.get("author", {})
            author_id = author_info.get("id", "")
            if author_id:
                vec = self.get_author_discipline_vector(author_id, score_threshold)
                for k in range(19):
                    freq[k] += vec[k]

        # 按频次从大到小排序
        sorted_freq = sorted(freq, reverse=True)

        # 计算分子: sum((2i - 20) * x_i)
        numerator = 0.0
        for i in range(19):
            weight = 2 * (i + 1) - 20  # i 从 1 开始，所以用 i+1
            numerator += weight * sorted_freq[i]

        # 计算分母: 19 * sum(x_i)
        total_freq = sum(sorted_freq)
        denominator = 19 * total_freq if total_freq > 0 else 1

        # Discipline Balance = 1 - numerator / denominator
        balance = 1.0 - (numerator / denominator)

        return {
            "discipline_balance": round(balance, 4),
            "num_authors": m,
            "discipline_frequencies": freq,       # 原始频次（按学科索引顺序）
            "sorted_frequencies": sorted_freq,    # 排序后的频次
            "total_discipline_occurrences": total_freq
        }

    # ============ AI4S_Balance 相关 ============

    def _is_camp_b_work(self, work_info: dict,
                        camp_b: set = None,
                        score_threshold: float = None) -> bool:
        """
        判断一篇论文是否属于阵营 B（基础科学类）

        与 _is_camp_a_work 对称，使用相同的双条件逻辑：
        条件1: camp_b 中至少有一个一级学科 score >= AI_CAMP_A_SCORE_MIN
        条件2: 所有非 camp_b 的一级学科 score <= AI_NON_CAMP_A_SCORE_MAX

        Args:
            work_info: 论文信息字典（包含 concepts）
            camp_b: 阵营 B 的 concept ID 集合
            score_threshold: 学科得分阈值，设为 None 则使用双条件模式

        Returns:
            是否属于阵营 B
        """
        from config import (AI4S_CAMP_B, AI_REF_SCORE_THRESHOLD,
                            AI_CAMP_A_SCORE_MIN, AI_NON_CAMP_A_SCORE_MAX)

        if camp_b is None:
            camp_b = AI4S_CAMP_B
        if score_threshold is None:
            score_threshold = AI_REF_SCORE_THRESHOLD

        concepts = work_info.get("concepts", [])
        if not concepts:
            return False

        if score_threshold is not None:
            # 固定阈值模式
            for c in concepts:
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
                if (pure_cid in camp_b or cid in camp_b) and score >= score_threshold:
                    return True
            return False
        else:
            # 双条件模式（只检查 level=0 的一级学科）
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

    def get_work_ai4s_balance(self, work: dict,
                              camp_a: set = None,
                              camp_b: set = None,
                              batch_size: int = 50) -> dict:
        """
        计算一篇论文的 AI4S_Balance（AI for Science 双向融合度）

        公式:
            AI4S_Balance = 1 - |P_A - P_B|

        其中:
            P_A = N_A / N_total  # 阵营 A（AI/计算机）的参考文献比例
            P_B = N_B / N_total  # 阵营 B（基础科学）的参考文献比例
            N_A = 参考文献中属于阵营 A 的数量
            N_B = 参考文献中属于阵营 B 的数量
            N_total = 参考文献总数

        注意:
            P_A 和 P_B 基于参考文献的阵营分类计算，而非论文本身的 concepts。
            一篇参考文献可能同时属于阵营 A 和阵营 B（或都不属于），
            因此 P_A + P_B 不一定等于 1。

        Args:
            work: 论文数据字典
            camp_a: 阵营 A 的 concept ID 集合
            camp_b: 阵营 B 的 concept ID 集合
            batch_size: 批量查询的每批数量

        Returns:
            {"ai4s_balance": float, "camp_a_count": int, "camp_b_count": int, ...}
        """
        from config import AI4S_CAMP_A, AI4S_CAMP_B

        if camp_a is None:
            camp_a = AI4S_CAMP_A
        if camp_b is None:
            camp_b = AI4S_CAMP_B

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

        # 批量获取参考文献信息
        ref_infos = self._batch_get_works(ref_ids, batch_size)

        # 统计每篇参考文献属于哪个阵营
        camp_a_count = 0
        camp_b_count = 0
        camp_a_refs = []
        camp_b_refs = []
        neither_refs = []

        for ref_id, ref_info in ref_infos.items():
            is_a = self._is_camp_a_work(ref_info, camp_a)
            is_b = self._is_camp_b_work(ref_info, camp_b)

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

        # P_A = 参考文献中属于阵营 A 的数量 / 参考文献总数
        # P_B = 参考文献中属于阵营 B 的数量 / 参考文献总数
        # 注意：P_A + P_B 不一定等于 1，因为有些参考文献既不属于 A 也不属于 B
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

    # ============ AI_Ref_Age 相关 ============

    _work_cache: dict = {}  # 论文信息缓存（ID -> publication_year, concepts）

    def _batch_get_works(self, work_ids: List[str],
                         batch_size: int = 50) -> dict:
        """
        批量获取论文的发表年份和 concepts

        Args:
            work_ids: 论文 OpenAlex ID 列表
            batch_size: 每批数量（OpenAlex 限制最多 50）

        Returns:
            {work_id: {"publication_year": int, "concepts": list}}
        """
        result = {}
        uncached_ids = []

        # 先检查缓存
        for wid in work_ids:
            if wid in self._work_cache:
                result[wid] = self._work_cache[wid]
            else:
                uncached_ids.append(wid)

        if not uncached_ids:
            return result

        # 分批查询
        for i in range(0, len(uncached_ids), batch_size):
            batch = uncached_ids[i:i + batch_size]
            # 从 ID 中提取纯 ID（去掉 URL 前缀）
            pure_ids = [wid.strip("/").split("/")[-1] for wid in batch]
            filter_str = "|".join(pure_ids)

            url = f"{self.base_url}/works"
            params = {
                "filter": f"openalex:{filter_str}",
                "select": "id,publication_year,concepts",
                "per_page": batch_size
            }

            data = self._make_request(url, params)
            if data and "results" in data:
                for w in data["results"]:
                    wid = w.get("id", "")
                    info = {
                        "publication_year": w.get("publication_year"),
                        "concepts": w.get("concepts", [])
                    }
                    self._work_cache[wid] = info
                    result[wid] = info

            time.sleep(0.1)  # 礼貌性延迟

        return result

    def _is_camp_a_work(self, work_info: dict,
                        camp_a: set = None,
                        score_threshold: float = None) -> bool:
        """
        判断一篇论文是否属于阵营 A（AI/计算机类）

        有两种判断模式：
        1. 固定阈值模式（score_threshold 不为 None）：
           只要 camp_a 中任一学科的 score >= 阈值即认定为 AI 论文
        2. 双条件模式（score_threshold 为 None，默认）：
           条件1: Computer Science 子领域至少有一个学科 score >= AI_CAMP_A_SCORE_MIN
           条件2: 所有非 Computer Science 领域的学科 score <= AI_NON_CAMP_A_SCORE_MAX
           两个条件同时满足才认定为 AI 论文

        Args:
            work_info: 论文信息字典（包含 concepts）
            camp_a: 阵营 A 的 concept ID 集合
            score_threshold: 学科得分阈值，设为 None 则使用双条件模式

        Returns:
            是否属于阵营 A
        """
        from config import (AI4S_CAMP_A, AI_REF_SCORE_THRESHOLD,
                            AI_CAMP_A_SCORE_MIN, AI_NON_CAMP_A_SCORE_MAX)

        if camp_a is None:
            camp_a = AI4S_CAMP_A
        if score_threshold is None:
            score_threshold = AI_REF_SCORE_THRESHOLD

        concepts = work_info.get("concepts", [])
        if not concepts:
            return False

        if score_threshold is not None:
            # 固定阈值模式：检查 camp_a 中是否有任一学科 score >= 阈值
            for c in concepts:
                cid = c.get("id", "")
                score = c.get("score", 0)
                pure_cid = cid.strip("/").split("/")[-1] if "/" in cid else cid
                if (pure_cid in camp_a or cid in camp_a) and score >= score_threshold:
                    return True
            return False
        else:
            # 双条件模式（只检查 level=0 的一级学科，忽略细分子学科）
            # 条件1: CS 子领域至少有一个学科 score >= AI_CAMP_A_SCORE_MIN
            has_high_cs = False
            # 条件2: 所有非 CS 领域的一级学科 score <= AI_NON_CAMP_A_SCORE_MAX
            all_non_cs_low = True

            for c in concepts:
                level = c.get("level", -1)
                # 只检查 level=0 的一级学科
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

    # ============ Citation-based Interdisciplinarity 相关 ============

    def get_work_cit_interdisciplinarity(self, citing_works: list) -> dict:
        """
        计算一篇论文的 Citation-based Interdisciplinarity（基于引用的跨学科性）

        公式:
            Cit Interdisciplinarity_i = 1 - sum(q_k^2)

        其中:
            q_k = 引用论文 i 的施引文献中，属于学科 k 的比例
            k 遍历 19 个一级学科

        对每篇施引论文，取 score 最高的 level-0 学科作为其"主学科"。
        如果施引论文没有 level-0 学科，则不计入统计。

        Args:
            citing_works: 施引论文列表（每篇必须包含 concepts 字段）

        Returns:
            {"cit_interdisciplinarity": float, "discipline_distribution": dict, ...}
        """
        if not citing_works:
            return {
                "cit_interdisciplinarity": 0.0,
                "num_citing_with_discipline": 0,
                "total_citing": 0,
                "discipline_distribution": {}
            }

        # 统计每个一级学科出现的频次
        discipline_counts = {name: 0 for name in LEVEL0_DISCIPLINES}
        total_with_discipline = 0

        for citing in citing_works:
            concepts = citing.get("concepts", [])
            if not concepts:
                continue

            # 找到 score 最高的 level-0 学科
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

        # 计算 q_k = 学科k的频次 / 有学科的施引论文总数
        # 并计算 sum(q_k^2)
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

    def get_work_ai_ref_age(self, work: dict,
                            camp_a: set = None,
                            batch_size: int = 50) -> dict:
        """
        计算一篇论文的 AI_Ref_Age（AI 技术时效性）

        公式:
            AI_Ref_Age = 2026 - 平均(所有阵营A参考文献的发表年份)

        其中 2026 为当前年份，表示 AI 参考文献距离现在有多久。
        值越大说明引用的 AI 文献越陈旧。

        Args:
            work: 论文数据字典
            camp_a: 阵营 A 的 concept ID 集合
            batch_size: 批量查询的每批数量

        Returns:
            {"ai_ref_age": float, "ai_ref_count": int, "ai_ref_years": list, ...}
        """
        from config import AI4S_CAMP_A
        from datetime import datetime

        if camp_a is None:
            camp_a = AI4S_CAMP_A

        CURRENT_YEAR = datetime.now().year

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

        # 批量获取参考文献信息
        ref_infos = self._batch_get_works(ref_ids, batch_size)

        # 筛选属于阵营 A 的参考文献，收集其发表年份
        ai_ref_years = []
        for ref_id, ref_info in ref_infos.items():
            if self._is_camp_a_work(ref_info, camp_a):
                year = ref_info.get("publication_year")
                if year:
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
        ai_ref_age = CURRENT_YEAR - avg_ai_ref_year

        return {
            "ai_ref_age": round(ai_ref_age, 2),
            "ai_ref_count": len(ai_ref_years),
            "total_refs": len(ref_ids),
            "avg_ai_ref_year": round(avg_ai_ref_year, 1),
            "min_ai_ref_year": min(ai_ref_years),
            "max_ai_ref_year": max(ai_ref_years),
            "ai_ref_years": sorted(ai_ref_years)
        }
