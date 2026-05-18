"""
检查导致 AI_Ref_Age 为负数的参考文献
"""
import requests
import sys

work_id = "W4383186800"

# 获取论文的参考文献列表
url = f"https://api.openalex.org/works/{work_id}"
params = {"select": "id,title,referenced_works"}
r = requests.get(url, params=params).json()
print(f"论文标题: {r.get('title', 'Unknown')}")
ref_ids = r.get("referenced_works", [])
print(f"参考文献总数: {len(ref_ids)}")

# 批量查询参考文献的年份
pure_ids = [rid.split("/")[-1] for rid in ref_ids]
filter_str = "|".join(pure_ids)

url2 = "https://api.openalex.org/works"
params2 = {
    "filter": f"openalex:{filter_str}",
    "select": "id,title,publication_year",
    "per_page": 200
}
r2 = requests.get(url2, params=params2).json()
results = r2.get("results", [])

# 找出年份 >= 2024 的
print("\n年份 >= 2024 的参考文献:")
found = False
for w in results:
    year = w.get("publication_year")
    if year and year >= 2024:
        found = True
        print(f"  {year}: {w['title'][:100]} ({w['id']})")
if not found:
    print("  (无)")
