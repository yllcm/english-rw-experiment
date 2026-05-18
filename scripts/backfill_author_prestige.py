"""
Backfill author prestige variables (first_author_hindex, last_author_hindex, max_author_hindex)
from OpenAlex Author API.

Strategy:
1. For each paper, get authorships from OpenAlex works API (select=authorships)
2. Collect all unique author IDs
3. Batch query author API for h_index, cited_by_count, works_count
4. Compute per-paper prestige metrics
5. Merge back to the original data

Usage:
    python scripts/backfill_author_prestige.py
"""

import pandas as pd
import requests
import time
import os
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "data/raw/ai4s_metrics_full_merged.csv"
OUTPUT_FILE = "data/raw/ai4s_metrics_full_merged.csv"  # overwrite
BATCH_SIZE = 50  # OpenAlex allows up to 50 works per request
AUTHOR_BATCH_SIZE = 50  # Max authors per batch request
SLEEP_BETWEEN_BATCHES = 1.0  # seconds
SLEEP_BETWEEN_AUTHOR_BATCHES = 0.5  # seconds

USER_AGENT = "mailto:test@test.com"  # TODO: replace with your email


def extract_oaid(work_id):
    """Extract OpenAlex ID from full URL or return as-is."""
    if pd.isna(work_id):
        return None
    work_id = str(work_id)
    if '/' in work_id:
        return work_id.split('/')[-1]
    return work_id


def main():
    # ============================================================
    # Step 1: Load data
    # ============================================================
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} papers")

    df['oaid'] = df['work_id'].apply(extract_oaid)
    valid = df['oaid'].notna()
    print(f"Valid work_ids: {valid.sum()} / {len(df)}")

    # ============================================================
    # Step 2: Get authorships for each paper
    # ============================================================
    print("\n=== Step 2: Fetching authorships from works API ===")

    # Check cache
    cache_file = "data/raw/author_work_cache.csv"
    if os.path.exists(cache_file):
        cache = pd.read_csv(cache_file)
        # cache columns: oaid, first_author_id, last_author_id, all_author_ids
        work_author_map = {}
        for _, row in cache.iterrows():
            work_author_map[row['oaid']] = {
                'first_author_id': row.get('first_author_id'),
                'last_author_id': row.get('last_author_id'),
                'all_author_ids': row.get('all_author_ids', ''),
            }
        print(f"Loaded {len(work_author_map)} cached work-author mappings")
    else:
        work_author_map = {}

    valid_oaids = df.loc[valid, 'oaid'].tolist()
    total = len(valid_oaids)
    to_fetch = [oid for oid in valid_oaids if oid not in work_author_map]
    print(f"Works to fetch: {len(to_fetch)} / {total}")

    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i:i+BATCH_SIZE]
        ids_str = '|'.join(batch)
        url = f'https://api.openalex.org/works?filter=openalex:{ids_str}&select=id,authorships&per_page={BATCH_SIZE}'

        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                results = r.json().get('results', [])
                for work in results:
                    wid = work['id'].split('/')[-1]
                    authorships = work.get('authorships', [])

                    first_author_id = None
                    last_author_id = None
                    all_ids = []

                    for a in authorships:
                        author = a.get('author', {})
                        aid = author.get('id', '').split('/')[-1] if author.get('id') else None
                        if aid:
                            all_ids.append(aid)
                            pos = a.get('author_position', '')
                            if pos == 'first':
                                first_author_id = aid
                            elif pos == 'last':
                                last_author_id = aid

                    work_author_map[wid] = {
                        'first_author_id': first_author_id,
                        'last_author_id': last_author_id,
                        'all_author_ids': '|'.join(all_ids),
                    }
            else:
                print(f"  Batch {i//BATCH_SIZE+1}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE+1}: Error {e}")

        if (i // BATCH_SIZE + 1) % 10 == 0:
            pct = min(i + BATCH_SIZE, len(to_fetch)) / len(to_fetch) * 100
            print(f"  Progress: {min(i+BATCH_SIZE, len(to_fetch))}/{len(to_fetch)} ({pct:.0f}%)")

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save work-author cache
    cache_rows = []
    for oaid, info in work_author_map.items():
        cache_rows.append({
            'oaid': oaid,
            'first_author_id': info['first_author_id'],
            'last_author_id': info['last_author_id'],
            'all_author_ids': info['all_author_ids'],
        })
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_csv(cache_file, index=False)
    print(f"Saved {len(work_author_map)} work-author mappings to cache")

    # ============================================================
    # Step 3: Get h_index for all unique authors
    # ============================================================
    print("\n=== Step 3: Fetching author h_indices from author API ===")

    # Collect all unique author IDs
    all_author_ids = set()
    for info in work_author_map.values():
        if info['first_author_id']:
            all_author_ids.add(info['first_author_id'])
        if info['last_author_id']:
            all_author_ids.add(info['last_author_id'])
        if info['all_author_ids']:
            for aid in info['all_author_ids'].split('|'):
                if aid:
                    all_author_ids.add(aid)

    print(f"Unique authors: {len(all_author_ids)}")

    # Check author cache
    author_cache_file = "data/raw/author_prestige_cache.csv"
    if os.path.exists(author_cache_file):
        author_cache = pd.read_csv(author_cache_file)
        author_info_map = {}
        for _, row in author_cache.iterrows():
            author_info_map[row['author_id']] = {
                'h_index': row.get('h_index'),
                'cited_by_count': row.get('cited_by_count'),
                'works_count': row.get('works_count'),
            }
        print(f"Loaded {len(author_info_map)} cached author infos")
    else:
        author_info_map = {}

    to_fetch_authors = [aid for aid in all_author_ids if aid not in author_info_map]
    print(f"Authors to fetch: {len(to_fetch_authors)} / {len(all_author_ids)}")

    for i in range(0, len(to_fetch_authors), AUTHOR_BATCH_SIZE):
        batch = to_fetch_authors[i:i+AUTHOR_BATCH_SIZE]
        ids_str = '|'.join(batch)
        url = f'https://api.openalex.org/authors?filter=openalex:{ids_str}&select=id,summary_stats,cited_by_count,works_count&per_page={AUTHOR_BATCH_SIZE}'

        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                results = r.json().get('results', [])
                for author in results:
                    aid = author['id'].split('/')[-1]
                    stats = author.get('summary_stats', {})
                    author_info_map[aid] = {
                        'h_index': stats.get('h_index'),
                        'cited_by_count': author.get('cited_by_count'),
                        'works_count': author.get('works_count'),
                    }
            else:
                print(f"  Author batch {i//AUTHOR_BATCH_SIZE+1}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Author batch {i//AUTHOR_BATCH_SIZE+1}: Error {e}")

        if (i // AUTHOR_BATCH_SIZE + 1) % 20 == 0:
            pct = min(i + AUTHOR_BATCH_SIZE, len(to_fetch_authors)) / len(to_fetch_authors) * 100
            print(f"  Author progress: {min(i+AUTHOR_BATCH_SIZE, len(to_fetch_authors))}/{len(to_fetch_authors)} ({pct:.0f}%)")

        time.sleep(SLEEP_BETWEEN_AUTHOR_BATCHES)

    # Save author cache
    author_cache_rows = []
    for aid, info in author_info_map.items():
        author_cache_rows.append({
            'author_id': aid,
            'h_index': info['h_index'],
            'cited_by_count': info['cited_by_count'],
            'works_count': info['works_count'],
        })
    author_cache_df = pd.DataFrame(author_cache_rows)
    author_cache_df.to_csv(author_cache_file, index=False)
    print(f"Saved {len(author_info_map)} author infos to cache")

    # ============================================================
    # Step 4: Compute per-paper prestige metrics
    # ============================================================
    print("\n=== Step 4: Computing prestige metrics ===")

    first_author_hindex = []
    last_author_hindex = []
    max_author_hindex = []
    max_author_cited_by = []

    for _, row in df.iterrows():
        oaid = row.get('oaid')
        if pd.isna(oaid) or oaid not in work_author_map:
            first_author_hindex.append(None)
            last_author_hindex.append(None)
            max_author_hindex.append(None)
            max_author_cited_by.append(None)
            continue

        info = work_author_map[oaid]

        # First author h_index
        fa_id = info['first_author_id']
        if fa_id and fa_id in author_info_map:
            first_author_hindex.append(author_info_map[fa_id]['h_index'])
        else:
            first_author_hindex.append(None)

        # Last author h_index
        la_id = info['last_author_id']
        if la_id and la_id in author_info_map:
            last_author_hindex.append(author_info_map[la_id]['h_index'])
        else:
            last_author_hindex.append(None)

        # Max h_index among all authors
        all_ids = info['all_author_ids'].split('|') if info['all_author_ids'] else []
        max_h = None
        max_c = None
        for aid in all_ids:
            if aid in author_info_map:
                h = author_info_map[aid]['h_index']
                c = author_info_map[aid]['cited_by_count']
                if h is not None and (max_h is None or h > max_h):
                    max_h = h
                if c is not None and (max_c is None or c > max_c):
                    max_c = c
        max_author_hindex.append(max_h)
        max_author_cited_by.append(max_c)

    df['first_author_hindex'] = first_author_hindex
    df['last_author_hindex'] = last_author_hindex
    df['max_author_hindex'] = max_author_hindex
    df['max_author_cited_by'] = max_author_cited_by

    # ============================================================
    # Step 5: Summary and save
    # ============================================================
    print("\n=== Step 5: Summary ===")

    for col in ['first_author_hindex', 'last_author_hindex', 'max_author_hindex', 'max_author_cited_by']:
        filled = df[col].notna().sum()
        print(f"{col}: filled {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")
        if filled > 0:
            print(f"  Mean: {df[col].mean():.1f}, Median: {df[col].median():.1f}, "
                  f"Range: [{df[col].min():.0f}, {df[col].max():.0f}]")

    # Drop temporary column
    df = df.drop(columns=['oaid'])

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
