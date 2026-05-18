"""
配置文件 - AI4S 论文指标计算
"""

# ============ OpenAlex API 配置 ============
# 建议填写你的邮箱，OpenAlex 会给予更高的 API 速率限制（每秒 100 次）
# 如果不填，默认速率为每秒 10 次
MAILTO = "2011785637@qq.com"

# API 基础 URL
OPENALEX_BASE_URL = "https://api.openalex.org"

# 每次 API 请求的最大重试次数
MAX_RETRIES = 3

# 请求超时时间（秒）
TIMEOUT = 30

# ============ 论文搜索条件 ============
# 搜索关键词（可按需修改）
SEARCH_QUERY = "machine learning OR deep learning OR artificial intelligence"

# AI for Science 学科过滤
# 限制论文必须属于以下学科之一（使用 OpenAlex concept ID）
# 自然科学: C86803240=Biology, C185592680=Chemistry, C121332964=Physics,
#           C71924100=Medicine, C192562407=Materials science,
#           C33923547=Mathematics, C39432304=Environmental science,
#           C127313418=Geology
# 人文社科: C15744967=Psychology, C162324750=Economics
# 设为 None 则不进行学科过滤
SEARCH_CONCEPT_FILTER = "C86803240|C185592680|C121332964|C71924100|C192562407|C33923547|C39432304|C127313418|C15744967|C162324750"

# 搜索年份范围（None 表示不限制）
YEAR_FROM = 2023
YEAR_TO = 2026

# 最大论文数量（控制数据规模，建议几百篇）
MAX_WORKS = 200

# 随机种子（用于随机化搜索结果，保证每次跑出的论文不同）
# 设为 None 则每次随机；设为固定整数（如 42）可复现结果
RANDOM_SEED = None

# 随机采样倍数：搜索时获取 MAX_WORKS * RANDOM_SAMPLE_FACTOR 篇候选论文，
# 然后从中随机抽取 MAX_WORKS 篇。倍数越大，随机性越好，但搜索时间越长。
RANDOM_SAMPLE_FACTOR = 3

# 每页返回结果数（OpenAlex 最大 200）
PER_PAGE = 200

# ============ Discipline Variety 配置 ============
# 是否启用 Discipline Variety 计算
ENABLE_DISCIPLINE_VARIETY = True

# 学科得分阈值（仅保留 score >= 此值的学科作为"主学科"）
DISCIPLINE_SCORE_THRESHOLD = 0.5

# ============ Discipline Similarity 配置 ============
# 是否启用 Discipline Similarity 计算
ENABLE_DISCIPLINE_SIMILARITY = True

# ============ Discipline Balance 配置 ============
# 是否启用 Discipline Balance 计算
ENABLE_DISCIPLINE_BALANCE = True

# ============ AI4S_Balance 配置 ============
# 是否启用 AI4S_Balance 计算
ENABLE_AI4S_BALANCE = True

# AI4S 阵营定义
# 阵营 A: AI/计算机类学科（使用 OpenAlex concept ID）
# C41008148 = Computer science, C154945302 = Artificial intelligence,
# C119857082 = Machine learning, C108583219 = Deep learning,
# C124101348 = Data mining, C77277458 = Temporal database,
# C81363708 = Convolutional neural network, C153180895 = Pattern recognition
AI4S_CAMP_A = {
    "C41008148",     # Computer science
    "C154945302",    # Artificial intelligence
    "C119857082",    # Machine learning
    "C108583219",    # Deep learning
    "C124101348",    # Data mining
    "C81363708",     # Convolutional neural network
    "C153180895",    # Pattern recognition
    "C77277458",     # Temporal database
}

# 阵营 B: 基础科学类学科
AI4S_CAMP_B = {
    "C86803240",   # Biology
    "C185592680",  # Chemistry
    "C121332964",  # Physics
    "C71924100",   # Medicine
    "C192562407",  # Materials science
    "C33923547",   # Mathematics
    "C39432304",   # Environmental science
    "C127313418",  # Geology
    "C15744967",   # Psychology
    "C162324750",  # Economics
}

# ============ N_R 采样配置 ============
# N_R 计算时，最多采样多少篇参考文献进行查询
# 采样后按比例放大：N_R = sampled_nr * (total_refs / sampled_refs)
# 值越大越精确，但 API 调用次数越多
NR_SAMPLE_SIZE = 20

# ============ AI_Ref_Age 配置 ============
# 是否启用 AI_Ref_Age 计算
ENABLE_AI_REF_AGE = True

# 批量查询参考文献时，每批的最大数量（OpenAlex 限制每批最多 50 个 ID）
AI_REF_BATCH_SIZE = 50

# AI 参考文献的学科得分阈值（0-1）
# 只有 Computer Science 学科得分 >= 此值的参考文献才被计入 AI 参考文献
# 0.8 = 核心 AI 论文，0.5 = 强相关 AI 论文
# 设为 None 则使用 AI_CAMP_A_SCORE_MIN / AI_NON_CAMP_A_SCORE_MAX 逻辑
AI_REF_SCORE_THRESHOLD = None

# AI 论文判定条件（当 AI_REF_SCORE_THRESHOLD 为 None 时使用）：
# 条件1: Computer Science 子领域至少有一个学科 score >= AI_CAMP_A_SCORE_MIN
# 条件2: 所有非 Computer Science 领域的学科 score <= AI_NON_CAMP_A_SCORE_MAX
# 两个条件同时满足才认定为 AI 论文
AI_CAMP_A_SCORE_MIN = 0.5      # CS 子领域学科最低得分
AI_NON_CAMP_A_SCORE_MAX = 0.3  # 非 CS 领域学科最高允许得分

# ============ 输出配置 ============
# 结果输出目录
OUTPUT_DIR = "results"

# 结果 CSV 文件名
RESULTS_FILE = "ai4s_metrics_results.csv"

# 可视化图表文件名
VIZ_FILE = "ai4s_metrics_distribution.png"
