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
pip install -r requirements.txt

# 运行（默认搜索 200 篇 AI 相关论文）
python main.py
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

结果保存在 `results/` 目录下，每次运行自动生成带时间戳的文件：

- `ai4s_metrics_results_YYYYMMDD_HHMMSS.csv` — 指标数据
- `ai4s_metrics_distribution_YYYYMMDD_HHMMSS.png` — 可视化图表

## 数据来源

[OpenAlex](https://openalex.org/) — 开放学术图谱
