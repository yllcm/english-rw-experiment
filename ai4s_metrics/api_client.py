"""
OpenAlex API 客户端 - 负责数据爬取，不包含任何指标计算逻辑
"""

import time
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


class OpenAlexClient:
    """OpenAlex API 客户端"""

    def __init__(self):
        self.base_url = OPENALEX_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OpenAlexMetrics/1.0 (mailto:{})".format(
                MAILTO if MAILTO else "anonymous@example.com"
            )
        })
        # 缓存
        self._author_cache: dict = {}
        self._work_cache: dict = {}

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

        if MAILTO:
            params["mailto"] = MAILTO

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=TIMEOUT)
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
                    wait_time = 2 ** attempt
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
        cursor = "*"

        filters = [f"title.search:{query}"]
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if concept_filter:
            filters.append(f"concepts.id:{concept_filter}")

        sample_max = max_results * RANDOM_SAMPLE_FACTOR

        params = {
            "filter": ",".join(filters),
            "per_page": min(per_page, 200),
            "sort": "cited_by_count:desc",
            "cursor": cursor,
            "select": "id,doi,title,authorships,publication_year,cited_by_count,referenced_works,primary_location,concepts,open_access"
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

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            time.sleep(0.1)

        # 从候选论文中随机抽取 max_results 篇
        if len(all_works) > max_results:
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
            time.sleep(0.1)

        citing_works = citing_works[:max_results]
        print(f"  获取到 {len(citing_works)} 篇施引论文")
        return citing_works

    def get_nr_count(self, target_id: str, ref_ids: set,
                     max_sample: int = None) -> int:
        """
        计算 N_R: 仅引用目标论文的参考文献，但不引用目标论文本身的后续论文数量

        通过采样查询目标论文的部分参考文献，统计引用这些参考文献但不引用目标论文的论文数。
        为了效率，只采样 max_sample 篇参考文献，并对结果进行放大。

        放大公式:
            N_R = sampled_nr * (total_refs / sampled_refs)

        Args:
            target_id: 目标论文的 OpenAlex ID
            ref_ids: 目标论文的参考文献 ID 集合
            max_sample: 最多采样查询的参考文献数量（默认从 config 读取）

        Returns:
            N_R 值（放大后的估计值）
        """
        if max_sample is None:
            from config import NR_SAMPLE_SIZE
            max_sample = NR_SAMPLE_SIZE
        if not ref_ids:
            return 0

        total_refs = len(ref_ids)

        ref_list = list(ref_ids)
        if len(ref_list) > max_sample:
            ref_list = random.sample(ref_list, max_sample)

        sampled_refs = len(ref_list)

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
                if target_id not in citing_refs:
                    nr_paper_ids.add(citing_id)

            time.sleep(0.1)

        sampled_nr = len(nr_paper_ids)
        # 注意：不再使用放大公式 N_R = sampled_nr * (total_refs / sampled_refs)
        # 因为放大公式会引入 num_ref 与 Rela_Dz 的人为负相关
        # 直接用采样得到的 sampled_nr 作为 N_R（虽然低估绝对值，但不引入偏差）
        n_r = sampled_nr

        return n_r

    def get_author_info(self, author_id: str) -> Optional[dict]:
        """
        获取作者信息（含 x_concepts）

        Args:
            author_id: 作者的 OpenAlex ID

        Returns:
            作者信息字典
        """
        if author_id in self._author_cache:
            return self._author_cache[author_id]

        pure_id = author_id.strip("/").split("/")[-1]
        url = f"{self.base_url}/authors/{pure_id}"
        params = {"select": "id,display_name,x_concepts"}

        data = self._make_request(url, params)
        if data:
            self._author_cache[author_id] = data
        return data

    def batch_get_works(self, work_ids: List[str],
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

        for wid in work_ids:
            if wid in self._work_cache:
                result[wid] = self._work_cache[wid]
            else:
                uncached_ids.append(wid)

        if not uncached_ids:
            return result

        for i in range(0, len(uncached_ids), batch_size):
            batch = uncached_ids[i:i + batch_size]
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

            time.sleep(0.1)

        return result
