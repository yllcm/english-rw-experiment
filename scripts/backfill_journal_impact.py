"""
Backfill journal_impact (2yr_mean_citedness) from OpenAlex Source API.

Strategy:
1. For each paper, get source_id from OpenAlex works API (select=primary_location)
2. Deduplicate source_ids, query source API for 2yr_mean_citedness
3. Merge back to the original data

Usage:
    python scripts/backfill_journal_impact.py
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
SLEEP_BETWEEN_BATCHES = 1.0  # seconds, to be polite
SLEEP_BETWEEN_SOURCES = 0.1  # seconds

# ============================================================
# Step 1: Load data and get work_ids
# ============================================================
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} papers")

# Extract OpenAlex work IDs
def extract_oaid(work_id):
    if pd.isna(work_id):
        return None
    work_id = str(work_id)
    if '/' in work_id:
        return work_id.split('/')[-1]
    return work_id

df['oaid'] = df['work_id'].apply(extract_oaid)
valid = df['oaid'].notna()
print(f"Valid work_ids: {valid.sum()} / {len(df)}")

# ============================================================
# Step 2: Get source_id for each paper via works API
# ============================================================
print("\n=== Step 2: Fetching source_ids from works API ===")

# Check if we already have cached source_ids
cache_file = "data/raw/source_id_cache.csv"
if os.path.exists(cache_file):
    cache = pd.read_csv(cache_file)
    source_id_map = dict(zip(cache['oaid'], cache['source_id']))
    print(f"Loaded {len(source_id_map)} cached source_ids")
else:
    source_id_map = {}
    cache = []

valid_oaids = df.loc[valid, 'oaid'].tolist()
total = len(valid_oaids)

for i in range(0, total, BATCH_SIZE):
    batch = valid_oaids[i:i+BATCH_SIZE]
    # Use OpenAlex's pipe-separated filter
    ids_str = '|'.join(batch)
    url = f'https://api.openalex.org/works?filter=openalex:{ids_str}&select=id,primary_location&per_page={BATCH_SIZE}'
    
    try:
        r = requests.get(url)
        if r.status_code == 200:
            results = r.json().get('results', [])
            for work in results:
                wid = work['id']
                pl = work.get('primary_location', {})
                src = pl.get('source', {})
                source_id = src.get('id', None)
                if source_id:
                    source_id_map[wid.split('/')[-1]] = source_id
        else:
            print(f"  Batch {i//BATCH_SIZE+1}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  Batch {i//BATCH_SIZE+1}: Error {e}")
    
    if (i // BATCH_SIZE + 1) % 10 == 0:
        print(f"  Progress: {min(i+BATCH_SIZE, total)}/{total} ({min(i+BATCH_SIZE, total)/total*100:.0f}%)")
    
    time.sleep(SLEEP_BETWEEN_BATCHES)

# Save cache
cache_df = pd.DataFrame([{'oaid': k, 'source_id': v} for k, v in source_id_map.items()])
cache_df.to_csv(cache_file, index=False)
print(f"Saved {len(source_id_map)} source_ids to cache")

# Map source_ids back to df
df['source_id'] = df['oaid'].map(source_id_map)
found = df['source_id'].notna().sum()
print(f"Found source_ids for {found}/{total} papers ({found/total*100:.1f}%)")

# ============================================================
# Step 3: Get 2yr_mean_citedness for each unique source
# ============================================================
print("\n=== Step 3: Fetching 2yr_mean_citedness from source API ===")

unique_sources = df['source_id'].dropna().unique()
print(f"Unique sources: {len(unique_sources)}")

# Check cache
source_cache_file = "data/raw/source_impact_cache.csv"
if os.path.exists(source_cache_file):
    impact_cache = pd.read_csv(source_cache_file)
    impact_map = dict(zip(impact_cache['source_id'], impact_cache['2yr_mean_citedness']))
    print(f"Loaded {len(impact_map)} cached source impacts")
else:
    impact_map = {}

# Find sources not yet cached
to_fetch = [s for s in unique_sources if s not in impact_map]
print(f"Sources to fetch: {len(to_fetch)}")

for i, source_id in enumerate(to_fetch):
    sid = source_id.split('/')[-1]
    url = f'https://api.openalex.org/sources/{sid}&select=id,summary_stats'
    
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            stats = data.get('summary_stats', {})
            impact = stats.get('2yr_mean_citedness', None)
            impact_map[source_id] = impact
        else:
            print(f"  Source {i+1}: HTTP {r.status_code} for {source_id}")
            impact_map[source_id] = None
    except Exception as e:
        print(f"  Source {i+1}: Error {e}")
        impact_map[source_id] = None
    
    if (i + 1) % 50 == 0:
        print(f"  Source progress: {i+1}/{len(to_fetch)}")
    
    time.sleep(SLEEP_BETWEEN_SOURCES)

# Save source cache
source_cache_df = pd.DataFrame([{'source_id': k, '2yr_mean_citedness': v} for k, v in impact_map.items()])
source_cache_df.to_csv(source_cache_file, index=False)
print(f"Saved {len(impact_map)} source impacts to cache")

# ============================================================
# Step 4: Merge back to original data
# ============================================================
print("\n=== Step 4: Merging back ===")

df['journal_impact'] = df['source_id'].map(impact_map)
filled = df['journal_impact'].notna().sum()
print(f"Filled journal_impact for {filled}/{len(df)} papers ({filled/len(df)*100:.1f}%)")

if filled > 0:
    print(f"journal_impact stats:")
    print(f"  Mean: {df['journal_impact'].mean():.2f}")
    print(f"  Median: {df['journal_impact'].median():.2f}")
    print(f"  Range: [{df['journal_impact'].min():.2f}, {df['journal_impact'].max():.2f}]")

# Drop temporary columns
df = df.drop(columns=['oaid', 'source_id'])

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
print("Done!")
