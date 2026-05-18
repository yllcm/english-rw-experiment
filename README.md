# AI4S Metrics - 论文指标计算工具

从 OpenAlex 爬取论文数据，计算多项论文指标，包括颠覆性指数、团队学科多样性、AI for Science 融合度等。

## 功能

- **颠覆性指数 (Rela_Dz)** — 基于引用网络的论文颠覆性度量
- **团队学科多样性 (Discipline Variety)** — 作者团队的学科背景多样性
- **团队学科相似性 (Discipline Similarity)** — 作者之间的学科背景相似度
- **团队学科均衡性 (Discipline Balance)** — 团队学科分布的均衡程度
- **基于引用的跨学科性 (Cit Interdisciplinarity)** — 施引论文的学科分布广度
- **AI4S_Balance** — AI 与基础科学的双向融合度
- **AI_Ref_Age** — AI 技术参考文献的时效性

## 快速开始

```bash
# 安装依赖
pip install -r ai4s_metrics/requirements.txt

# 运行（默认搜索 200 篇 AI 相关论文）
python -m ai4s_metrics.main
```

## 配置

编辑 `config.py` 调整搜索条件：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SEARCH_QUERY` | 搜索关键词 | `machine learning OR deep learning OR artificial intelligence` |
| `YEAR_FROM` / `YEAR_TO` | 年份范围 | 2020-2023 |
| `MAX_WORKS` | 最大论文数 | 200 |
| `RANDOM_SEED` | 随机种子（None=每次不同） | `None` |

## 输出

结果保存在 `data/raw/` 目录下，每次运行自动生成带时间戳的文件：

- `ai4s_metrics_results_YYYYMMDD_HHMMSS.csv` — 指标数据
- `ai4s_metrics_distribution_YYYYMMDD_HHMMSS.png` — 可视化图表

## 回归分析

对采集的论文数据进行回归建模，分析跨学科指标对研究影响力的预测能力。

### 模型

| 模型 | 因变量 | 估计方法 |
|:----|:-------|:---------|
| 模型1 | Citation Impact（引用影响力） | OLS |
| 模型2 | Citing Interdisc.（施引跨学科性） | OLS |
| 模型3 | Disruptiveness（颠覆性指数） | OLS + **Ordered Logit** |

### 自变量

**跨学科变量（5个）：**
- `discipline_variety` — 团队学科多样性
- `discipline_similarity` — 团队学科相似性
- `discipline_balance` — 团队学科均衡性
- `ai4s_balance` — AI4S 双向融合度
- `ai_ref_age_c` / `ai_ref_age_c_sq` — AI 引用年龄（含平方项，检验 U 型关系）

**控制变量（6个）：**
- `num_authors` — 作者数量
- `publication_year_c` — 发表年份（中心化）
- `num_references` — 参考文献数量
- `num_institutions` — 机构数量
- `has_international_collab` — 国际合作（0/1）
- `open_access` — 开放获取（0/1）

### 使用方法

```bash
# 对采集的指标数据运行回归分析
python scripts/regression_analysis.py <input_csv> <output_dir>

# 示例
python scripts/regression_analysis.py data/combined/ai4s_metrics_combined.csv data/regression/
```

### 输出

回归分析自动生成以下内容：

- `descriptive_stats.csv` — 描述性统计
- `correlation_heatmap.png` — 相关性热力图
- `regression_*.txt` — 各模型的回归结果（含标准化系数）
- `diagnostics_*.txt` — 模型诊断（VIF、Breusch-Pagan 检验）
- `residuals_*.png` — 残差诊断图
- `regression_coefficients.png` — 三模型系数对比图
- `ordered_logit_*.txt` — Ordered Logit 回归结果
- `ordered_logit_*_confusion.png` — 混淆矩阵

## 数据来源

[OpenAlex](https://openalex.org/) — 开放学术图谱
