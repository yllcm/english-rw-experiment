# AI4S Metrics - 论文指标计算工具

从 [OpenAlex](https://openalex.org/) 爬取论文数据，计算多项论文指标，包括颠覆性指数、团队学科多样性、AI for Science 融合度等。

## 功能

- **颠覆性指数 (Rela_Dz)** — 基于引用网络的论文颠覆性度量
- **团队学科多样性 (Discipline Variety)** — 作者团队的学科背景多样性
- **团队学科相似性 (Discipline Similarity)** — 作者之间的学科背景相似度
- **团队学科均衡性 (Discipline Balance)** — 团队学科分布的均衡程度
- **基于引用的跨学科性 (Cit Interdisciplinarity)** — 施引论文的学科分布广度
- **AI4S_Balance** — AI 与基础科学的双向融合度
- **AI_Ref_Age** — AI 技术参考文献的时效性

## 环境要求

- Python 3.8+
- 网络连接（需要访问 OpenAlex API）

## 快速开始

```bash
# 1. 进入项目目录
cd ai4s_metrics

# 2. 安装依赖
pip install -r requirements.txt

# 3. 修改配置（重要！）
#    编辑 config.py，将 MAILTO 改为你自己的邮箱：
#    MAILTO = "your_email@example.com"
#    这样可以获得更高的 API 速率限制（每秒 100 次 vs 默认 10 次）

# 4. 运行（默认搜索 200 篇 AI 相关论文）
python main.py
```

> **注意**：请始终在 `ai4s_metrics/` 目录下运行 `python main.py`，不要在上级目录用 `python -m ai4s_metrics.main` 运行。

## 配置

编辑 `config.py` 调整搜索条件：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MAILTO` | **你的邮箱**（必改，影响 API 速率） | `2011785637@qq.com` |
| `SEARCH_QUERY` | 搜索关键词 | `machine learning OR deep learning OR artificial intelligence` |
| `SEARCH_CONCEPT_FILTER` | 学科过滤（\| 分隔的 concept ID） | 10 个自然科学+社科 |
| `YEAR_FROM` / `YEAR_TO` | 年份范围 | 2017-2025 |
| `MAX_WORKS` | 最大论文数 | 200 |
| `RANDOM_SEED` | 随机种子（`None`=每次不同） | `None` |
| `RANDOM_SAMPLE_FACTOR` | 随机采样倍数 | 3 |

### 配置详解

- **`MAILTO`**：OpenAlex 根据邮箱给予 API 速率限制。填写真实邮箱可获得 **100 次/秒** 的速率，不填只有 **10 次/秒**。**强烈建议修改为你自己的邮箱。**
- **`SEARCH_CONCEPT_FILTER`**：限制论文必须属于指定的学科。设为 `None` 则不限制学科。
- **`RANDOM_SEED`**：设为固定整数（如 `42`）可复现结果；设为 `None` 则每次随机抽取不同论文。
- **`MAX_WORKS`**：建议 50-500 篇。每篇论文需要多次 API 调用（获取施引论文、参考文献信息等），数量越大运行时间越长。

## 输出

结果保存在 `data/raw/` 目录下，每次运行自动生成带时间戳的文件：

| 文件 | 说明 |
|:----|:------|
| `ai4s_metrics_results_YYYYMMDD_HHMMSS.csv` | 论文各项指标数据 |
| `controls_YYYYMMDD_HHMMSS.csv` | 控制变量（作者数、机构数、国际合作等） |
| `ai4s_metrics_full_YYYYMMDD_HHMMSS.csv` | 指标 + 控制变量合并数据（用于回归分析） |
| `ai4s_metrics_distribution_YYYYMMDD_HHMMSS.png` | 可视化图表 |

## 回归分析

对采集的论文数据进行回归建模，分析跨学科指标对研究影响力的预测能力。

### 模型

| 模型 | 因变量 | 估计方法 |
|:----|:-------|:---------|
| 模型1 | Citation Impact（引用影响力） | OLS |
| 模型2 | Citing Interdisc.（施引跨学科性） | OLS |
| 模型3 | Disruptiveness（颠覆性指数） | OLS + **sqrt 变换**（处理右偏分布） |

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
# 在项目根目录（code/）下运行
python scripts/regression_analysis.py <input_csv> <output_dir>

# 示例
python scripts/regression_analysis.py data/raw/ai4s_metrics_full_20260518_133747.csv data/regression/
```

### 输出

回归分析自动生成以下内容：

- `descriptive_stats.csv` — 描述性统计
- `correlation_heatmap.png` — 相关性热力图
- `regression_*.txt` — 各模型的回归结果（含标准化系数、HC3/Cluster 稳健标准误）
- `diagnostics_*.txt` — 模型诊断（VIF、Breusch-Pagan 检验）
- `residuals_*.png` — 残差诊断图
- `regression_coefficients.png` — 三模型系数对比图
- `sqrt_robustness_*.txt` — sqrt 变换稳健性检验

## 常见问题

### Q: 运行报错 `ModuleNotFoundError: No module named 'config'`

**原因**：在错误目录下运行了 `python main.py`。
**解决**：先 `cd ai4s_metrics`，再运行 `python main.py`。

### Q: 运行很慢，卡在"正在获取引用论文"

**原因**：每篇论文需要查询施引论文和参考文献信息，200 篇论文可能需要 10-30 分钟。
**解决**：
- 在 `config.py` 中减小 `MAX_WORKS`（如设为 50）
- 确保 `MAILTO` 已设为你的邮箱（提高 API 速率）
- 网络不稳定时程序会自动重试

### Q: 报错 `requests.exceptions.ConnectionError`

**原因**：网络连接问题，无法访问 OpenAlex API。
**解决**：检查网络连接，或使用代理。

### Q: 结果中很多论文的 Rela_Dz 为 0

**原因**：论文被引次数较低或没有施引论文，无法计算颠覆性指数。
**解决**：这是正常现象，低被引论文的 Rela_Dz 确实接近 0。

### Q: 如何复现相同的结果？

**解决**：在 `config.py` 中将 `RANDOM_SEED` 设为固定整数（如 `42`），每次运行会得到相同的论文样本。

## 数据来源

[OpenAlex](https://openalex.org/) — 开放学术图谱
