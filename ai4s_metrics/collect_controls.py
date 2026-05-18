"""
控制变量收集模块 - 收集论文的控制变量数据

用于后续回归分析，排除混淆因素，观察颠覆性指数的净效应。

控制变量列表:
  - num_authors: 作者数量
  - publication_year: 发表年份
  - num_references: 参考文献数量
  - num_institutions: 作者所属机构数量（去重）
  - has_international_collab: 是否有国际合作（不同国家的机构）
  - journal_impact: 期刊影响力（期刊的总被引次数）
  - open_access: 论文是否开源获取（0=否, 1=是）
"""

from typing import Dict, Optional


def collect_controls(work: dict) -> Dict:
    """
    收集一篇论文的所有控制变量

    Args:
        work: 论文数据字典（来自 OpenAlex API）

    Returns:
        包含所有控制变量的字典
    """
    controls = {}

    # 1. 作者数量
    authorships = work.get("authorships", [])
    controls["num_authors"] = len(authorships)

    # 2. 发表年份
    controls["publication_year"] = work.get("publication_year")

    # 3. 参考文献数量
    refs = work.get("referenced_works", [])
    controls["num_references"] = len(refs) if refs else 0

    # 4. 机构数量（去重）
    institution_ids = set()
    for authorship in authorships:
        institutions = authorship.get("institutions", [])
        for inst in institutions:
            inst_id = inst.get("id")
            if inst_id:
                institution_ids.add(inst_id)
    controls["num_institutions"] = len(institution_ids)

    # 5. 是否有国际合作
    country_codes = set()
    for authorship in authorships:
        institutions = authorship.get("institutions", [])
        for inst in institutions:
            country = inst.get("country_code")
            if country:
                country_codes.add(country)
    controls["has_international_collab"] = 1 if len(country_codes) >= 2 else 0

    # 6. 期刊影响力
    controls["journal_impact"] = _get_journal_impact(work)

    # 7. 是否开源获取
    controls["open_access"] = _get_open_access(work)

    return controls


def _get_journal_impact(work: dict) -> Optional[float]:
    """
    获取论文所在期刊的影响力

    优先使用 OpenAlex source 的 cited_by_count（总被引次数），
    如果没有则返回 None。

    Args:
        work: 论文数据字典

    Returns:
        期刊总被引次数，或 None（无法获取时）
    """
    primary_location = work.get("primary_location")
    if not primary_location:
        return None

    source = primary_location.get("source")
    if not source:
        return None

    # 尝试获取 summary_stats 中的 2yr_mean_citedness（类似影响因子）
    summary_stats = source.get("summary_stats")
    if summary_stats:
        two_year = summary_stats.get("2yr_mean_citedness")
        if two_year is not None:
            return round(two_year, 2)

    # 备选：使用 source 自身的 cited_by_count
    cited_by_count = source.get("cited_by_count")
    if cited_by_count is not None:
        return cited_by_count

    return None


def _get_open_access(work: dict) -> int:
    """
    获取论文是否开源获取

    OpenAlex 的 open_access 字段结构:
    {
        "is_oa": true/false,
        "oa_status": "gold" | "green" | "hybrid" | "bronze" | "closed",
        "oa_url": "..."
    }

    Args:
        work: 论文数据字典

    Returns:
        1 = 开源, 0 = 非开源
    """
    oa = work.get("open_access")
    if oa and isinstance(oa, dict):
        is_oa = oa.get("is_oa", False)
        return 1 if is_oa else 0
    return 0


def collect_controls_batch(works: list) -> list:
    """
    批量收集多篇论文的控制变量

    Args:
        works: 论文数据字典列表

    Returns:
        控制变量字典列表
    """
    results = []
    for i, work in enumerate(works):
        work_id = work.get("id", "unknown")
        title = work.get("title", "Unknown")[:50]

        controls = collect_controls(work)
        controls["work_id"] = work_id
        controls["title"] = title

        results.append(controls)

        if (i + 1) % 50 == 0:
            print(f"  已收集 {i + 1}/{len(works)} 篇论文的控制变量")

    return results
