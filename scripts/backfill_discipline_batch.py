"""
批量补全脚本 - 为 data/raw/ 下所有 CSV 文件添加 primary_discipline 字段

策略：
1. 收集所有 CSV 文件中的唯一 work_id
2. 只对唯一 work_id 调用 API（避免重复请求）
3. 将补全结果写回所有 CSV 文件

用法:
    python scripts/backfill_discipline_batch.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai4s_metrics.config import MAILTO, OPENALEX_BASE_URL, TIMEOUT
from ai4s_metrics.collect_controls import _get_primary_discipline


def collect_all_work_ids(raw_dir: str) -> dict:
    """
    收集所有 CSV 文件中的 work_id

    Args:
        raw_dir: data/raw 目录路径

    Returns:
        {work_id: [file1, file2, ...]} 字典
    """
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    csv_files.sort()

    work_id_to_files = defaultdict(list)
    file_info = []

    for f in csv_files:
        path = os.path.join(raw_dir, f)
        try:
            df = pd.read_csv(path)
            if 'work_id' not in df.columns:
                continue

            # 检查是否已有 primary_discipline
            has_disc = 'primary_discipline' in df.columns
            if has_disc and df['primary_discipline'].notna().all():
                file_info.append((f, len(df), True, True))
                continue

            file_info.append((f, len(df), True, False))

            for wid in df['work_id'].unique():
                if pd.notna(wid) and str(wid).strip():
                    work_id_to_files[wid].append(f)

        except Exception as e:
            print(f"  ERROR reading {f}: {e}")

    return work_id_to_files, file_info


def fetch_discipline(work_id: str, session: requests.Session) -> str:
    """
    从 OpenAlex API 获取论文的主学科

    Args:
        work_id: OpenAlex work ID (如 https://openalex.org/Wxxxxxxxx)
        session: requests Session

    Returns:
        主学科名称，失败返回 "Unknown"
    """
    if "/" in work_id:
        oa_id = work_id.split("/")[-1]
    else:
        oa_id = work_id

    url = f"{OPENALEX_BASE_URL}/works/{oa_id}"
    params = {"mailto": MAILTO}

    try:
        response = session.get(url, params=params, timeout=TIMEOUT)
        if response.status_code == 200:
            work = response.json()
            return _get_primary_discipline(work)
        else:
            print(f"    HTTP {response.status_code} for {oa_id}")
            return "Unknown"
    except Exception as e:
        print(f"    Error for {oa_id}: {e}")
        return "Unknown"


def backfill_all_files(raw_dir: str = "data/raw"):
    """
    批量补全所有 CSV 文件的 primary_discipline 字段

    Args:
        raw_dir: data/raw 目录路径
    """
    print("=" * 60)
    print("批量补全脚本 - 为所有 CSV 添加 primary_discipline")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 收集所有 work_id
    print("\n[步骤 1/4] 扫描所有 CSV 文件...")
    work_id_to_files, file_info = collect_all_work_ids(raw_dir)

    print(f"\nCSV 文件概况:")
    for f, rows, has_wid, has_disc in file_info:
        status = "OK" if has_disc else "NEED UPDATE"
        print(f"  {f:50s}  {rows:>5} rows  {status}")

    # 2. 统计需要补全的 work_id
    print(f"\n[步骤 2/4] 统计需要补全的 work_id...")
    total_unique = len(work_id_to_files)
    print(f"  共 {total_unique} 个唯一 work_id 需要补全")

    if total_unique == 0:
        print("  所有文件已包含 primary_discipline，无需补全")
        return

    # 3. 调用 API 补全
    print(f"\n[步骤 3/4] 调用 OpenAlex API 补全学科信息...")
    print(f"  (共 {total_unique} 个唯一 work_id, 预计耗时 {total_unique * 0.05:.0f}s)")

    session = requests.Session()
    session.headers.update({"User-Agent": f"mailto:{MAILTO}"})

    discipline_cache = {}
    success_count = 0
    error_count = 0

    for i, work_id in enumerate(work_id_to_files.keys()):
        discipline = fetch_discipline(work_id, session)
        discipline_cache[work_id] = discipline

        if discipline != "Unknown":
            success_count += 1
        else:
            error_count += 1

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{total_unique} 个 work_id...")

        # 速率限制
        time.sleep(0.05)

    print(f"\n  API 调用完成: {success_count} 成功, {error_count} 失败")

    # 4. 写回所有 CSV 文件
    print(f"\n[步骤 4/4] 将学科信息写回所有 CSV 文件...")

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    csv_files.sort()

    updated_count = 0
    for f in csv_files:
        path = os.path.join(raw_dir, f)
        try:
            df = pd.read_csv(path)
            if 'work_id' not in df.columns:
                continue

            # 检查是否已有完整的 primary_discipline
            if 'primary_discipline' in df.columns and df['primary_discipline'].notna().all():
                continue

            # 补全
            if 'primary_discipline' not in df.columns:
                df['primary_discipline'] = None

            for idx, row in df.iterrows():
                wid = row.get('work_id', '')
                if pd.notna(wid) and str(wid).strip() and wid in discipline_cache:
                    df.at[idx, 'primary_discipline'] = discipline_cache[wid]

            # 保存
            df.to_csv(path, index=False, encoding="utf-8-sig")
            updated_count += 1
            print(f"  ✓ {f}")

        except Exception as e:
            print(f"  ✗ {f}: {e}")

    # 5. 清理临时文件
    temp_file = os.path.join(os.path.dirname(raw_dir), "..", "check_csv_files.py")
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

    print(f"\n{'=' * 60}")
    print(f"批量补全完成!")
    print(f"  已更新: {updated_count} 个文件")
    print(f"  API 调用: {success_count} 成功, {error_count} 失败")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    backfill_all_files()
