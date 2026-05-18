"""
补全脚本 - 为已有数据添加 primary_discipline 字段

读取已有的 CSV 文件，对每条记录调用 OpenAlex API 获取论文的 concepts，
提取主学科（level=0 中 score 最高的学科），保存为新的 CSV。

用法:
    python scripts/backfill_discipline.py <input_csv> [output_csv]

示例:
    python scripts/backfill_discipline.py data/raw/ai4s_metrics_full_20260517_235902.csv
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai4s_metrics.config import MAILTO, OPENALEX_BASE_URL, TIMEOUT
from ai4s_metrics.collect_controls import _get_primary_discipline


def backfill_discipline(input_file: str, output_file: str = None):
    """
    为已有 CSV 数据补全 primary_discipline 字段

    Args:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径（None 则自动生成）
    """
    print("=" * 60)
    print("补全脚本 - 添加 primary_discipline 字段")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 读取数据
    print(f"\n读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"共 {len(df)} 条记录，{len(df.columns)} 个字段")

    # 检查是否已有 primary_discipline
    if "primary_discipline" in df.columns:
        existing = df["primary_discipline"].notna().sum()
        print(f"已有 {existing}/{len(df)} 条记录包含 primary_discipline")
        if existing == len(df):
            print("所有记录已包含 primary_discipline，无需补全")
            return df

    # 确定输出文件
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_with_discipline{ext}"

    # 初始化 primary_discipline 列
    if "primary_discipline" not in df.columns:
        df["primary_discipline"] = None

    # 创建 session
    session = requests.Session()
    session.headers.update({"User-Agent": f"mailto:{MAILTO}"})

    # 遍历每条记录，补全 primary_discipline
    updated_count = 0
    skipped_count = 0
    error_count = 0

    print(f"\n开始补全 {len(df)} 条记录...")

    for idx, row in df.iterrows():
        work_id = row.get("work_id", "")
        current_value = row.get("primary_discipline")

        # 跳过已有值的记录
        if pd.notna(current_value) and str(current_value).strip():
            skipped_count += 1
            continue

        if not work_id or not isinstance(work_id, str):
            error_count += 1
            continue

        # 从 work_id 提取 OpenAlex ID
        # work_id 格式: https://openalex.org/Wxxxxxxxx
        if "/" in work_id:
            oa_id = work_id.split("/")[-1]
        else:
            oa_id = work_id

        # 调用 API
        url = f"{OPENALEX_BASE_URL}/works/{oa_id}"
        params = {"mailto": MAILTO}

        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 200:
                work = response.json()
                discipline = _get_primary_discipline(work)
                df.at[idx, "primary_discipline"] = discipline
                updated_count += 1

                if updated_count % 20 == 0:
                    print(f"  已补全 {updated_count}/{len(df)} 条...")
            else:
                print(f"  API 错误 (idx={idx}, id={oa_id}): HTTP {response.status_code}")
                error_count += 1

        except Exception as e:
            print(f"  请求失败 (idx={idx}, id={oa_id}): {e}")
            error_count += 1

        # 速率限制：每秒最多 10 次（不带邮箱）或 100 次（带邮箱）
        time.sleep(0.05)

    # 保存结果
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"补全完成!")
    print(f"  已更新: {updated_count} 条")
    print(f"  已跳过: {skipped_count} 条")
    print(f"  错误:   {error_count} 条")
    print(f"  输出:   {output_file}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/backfill_discipline.py <input_csv> [output_csv]")
        print("示例: python scripts/backfill_discipline.py data/raw/ai4s_metrics_full_20260517_235902.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        sys.exit(1)

    backfill_discipline(input_file, output_file)
