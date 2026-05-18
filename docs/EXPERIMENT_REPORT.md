# AI4S 跨学科研究实验报告

## 1. 数据概况

| 指标 | 值 |
|------|:---:|
| 样本量 | 192 篇论文 |
| 数据来源 | OpenAlex API |
| 时间范围 | 2020–2023 |

### 描述性统计

| 变量 | 均值 | 标准差 | 最小值 | 最大值 |
|------|:----:|:------:|:-----:|:-----:|
| Citation Impact (log) | 4.23 | 0.78 | 3.33 | 7.23 |
| Citing Interdisciplinarity | 0.51 | 0.17 | 0.00 | 0.83 |
| Disruptiveness (Rela_Dz) | 0.07 | 0.13 | 0.00 | 1.00 |
| Discipline Variety | 3.95 | 0.87 | 0.00 | 5.00 |
| Discipline Similarity | 0.58 | 0.27 | 0.00 | 1.00 |
| Discipline Balance | 1.84 | 0.14 | 1.00 | 1.95 |
| AI4S Balance | 0.67 | 0.22 | 0.05 | 1.00 |
| AI Ref Age (年) | 6.18 | 4.71 | 0.00 | 40.00 |
| Num Authors | 5.46 | 4.40 | 1.00 | 41.00 |
| Num References | 80.63 | 75.06 | 3.00 | 512.00 |
| Num Institutions | 3.53 | 2.99 | 0.00 | 18.00 |
| Intl. Collaboration | 43.2% | — | 0 | 1 |
| Open Access | 70.8% | — | 0 | 1 |


---

## 2. 回归模型处理流程报告

### 2.1 数据准备

**数据来源：** OpenAlex API 采集的 192 篇 AI4S 相关论文

**变量处理：**

| 类别 | 变量名 | 说明 |
|:-----|:-------|:-----|
| **因变量（3个）** | citation_impact | 引用影响力（对数变换） |
| | cit_interdisciplinarity | 施引跨学科性 |
| | Rela_Dz | 颠覆性指数 |
| **核心自变量（6个）** | discipline_variety | 学科多样性 |
| | discipline_similarity | 学科相似度（0-1） |
| | discipline_balance | 学科均衡度 |
| | ai4s_balance | AI4S 平衡度（0-1） |
| | ai_ref_age_c | AI 引用年龄（中心化：原始值 - 均值 6.18） |
| | ai_ref_age_c_sq | 中心化后的平方项（捕捉 U 型关系） |
| **控制变量（6个）** | num_authors | 作者数量 |
| | publication_year_c | 发表年份（中心化：原始值 - 均值 2021.55） |
| | num_references | 参考文献数量 |
| | num_institutions | 机构数量 |
| | has_international_collab | 是否有国际合作（0/1） |
| | open_access | 是否开放获取（0/1） |

**缺失值处理：** 缺失率 < 50% 用中位数填充，> 50% 删除该列

### 2.2 回归模型

建立了 **3 个 OLS 回归模型**，共享同一组自变量：

```
Y_m = β₀ + β₁·discipline_variety + β₂·discipline_similarity 
      + β₃·discipline_balance + β₄·ai4s_balance 
      + β₅·ai_ref_age_c + β₆·ai_ref_age_c_sq
      + β₇·num_authors + β₈·publication_year_c + β₉·num_references
      + β₁₀·num_institutions + β₁₁·has_international_collab 
      + β₁₂·open_access + ε
```

| 模型 | 因变量 | 估计方法 |
|:----|:-------|:---------|
| **模型1** | Citation Impact（引用影响力） | **OLS** |
| **模型2** | Citing Interdisc.（施引跨学科性） | **OLS** |
| **模型3** | Disruptiveness（颠覆性指数） | **OLS + Ordered Logit** |

### 2.3 标准化处理

由于各变量量纲不同（如 num_references 范围 3~512，ai4s_balance 范围 0~1），原始系数无法直接比较效应大小。因此对所有连续变量做 **Z-score 标准化**：

```
x_std = (x - mean) / std
```

标准化后的回归系数（β）表示：**自变量每变化 1 个标准差，因变量变化 β 个标准差**，可以直接比较效应大小。

### 2.4 模型诊断

| 检验 | 方法 | 说明 |
|:-----|:-----|:------|
| **多重共线性** | VIF 方差膨胀因子 | 阈值 VIF > 10 表示严重共线性 |
| **异方差性** | Breusch-Pagan 检验 | p > 0.05 表示无异方差 |
| **残差分布** | Q-Q 图 | 检查正态性假设 |
| **拟合优度** | R², Adj R², AIC | 模型间比较 |

### 2.5 稳健性检验（模型3）

由于 Rela_Dz 分布严重偏态（67.7% 的值 < 0.05），OLS 可能不稳健，因此补充 **Ordered Logit 回归**：

```
P(Y = j) = F(τⱼ - Xβ) - F(τⱼ₋₁ - Xβ)
```

将 Rela_Dz 分为 3 个有序类别：
- 0 = 低颠覆性（Rela_Dz < 0.05）
- 1 = 中颠覆性（0.05 ≤ Rela_Dz < 0.2）
- 2 = 高颠覆性（Rela_Dz ≥ 0.2）

预测准确率：**78.65%**，结果与 OLS 一致。

### 2.6 软件实现

- **语言：** Python 3
- **库：** statsmodels（OLS, OrderedModel, VIF, Breusch-Pagan）
- **可视化：** matplotlib, seaborn
- **输出：** 300 DPI PNG 图表 + 文本格式回归结果

---

## 3. 模型 1：Citation Impact（引用影响力）

**模型公式：** citation_impact ~ 跨学科变量 + 控制变量

### 拟合效果

| 指标 | 值 |
|------|:---:|
| R² | 0.285 |
| Adj R² | 0.237 |
| F-test | p < 0.001 *** |
| AIC | 411.7 |

### 显著变量

| 变量 | 系数 | p值 | 解释 |
|------|:----:|:---:|:----:|
| **ai4s_balance** | +0.507 | **0.042** * | AI4S平衡度越高，引用影响力越高 |
| **num_authors** | +0.031 | **0.031** * | 作者越多，引用越高 |
| **publication_year_c** | -0.136 | **0.004** ** | 越新的论文，引用越低 |
| **num_references** | +0.004 | **<0.001** *** | 参考文献越多，引用越高 |

### 不显著变量
- discipline_variety (p=0.132)
- discipline_similarity (p=0.816)
- discipline_balance (p=0.417)
- ai_ref_age_c (p=0.197)
- ai_ref_age_c_sq (p=0.596)

**结论：** 跨学科多样性对引用影响力无显著影响，只有 AI4S 平衡度有微弱正向效应。

---

## 4. 模型 2：Citing Interdisciplinarity（施引跨学科性）

**模型公式：** cit_interdisciplinarity ~ 跨学科变量 + 控制变量

### 拟合效果

| 指标 | 值 |
|------|:---:|
| R² | **0.408** |
| Adj R² | **0.369** |
| F-test | p < 0.001 *** |
| AIC | -220.7 |

### 显著变量

| 变量 | 系数 | p值 | 解释 |
|------|:----:|:---:|:----:|
| **ai4s_balance** | +0.450 | **<0.001** *** | AI4S平衡度越高，施引跨学科性越高（最强效应） |
| **ai_ref_age_c** | -0.007 | **0.024** * | 在均值处，AI引用年龄增加降低施引跨学科性 |
| **ai_ref_age_c_sq** | +0.001 | **0.002** ** | 存在 U 型关系 |

### U 型关系解释

ai_ref_age 与 cit_interdisciplinarity 呈 **U 型关系**（β₁ = -0.007, β₂ = +0.001）：
- 转折点（谷底）约在 **ai_ref_age ≈ 12.9 年**
- AI 引用年龄 < 12.9 年时，施引跨学科性随引用年龄增加而**下降**
- AI 引用年龄 > 12.9 年时，施引跨学科性随引用年龄增加而**上升**
- 在常见引用范围内（0-15 年），引用最新 AI 文献（0-3 年）的论文施引跨学科性最高（0.56），引用约 12 年 AI 文献的论文最低（0.48）

### 不显著变量
- discipline_variety (p=0.335)
- discipline_similarity (p=0.301)
- discipline_balance (p=0.429)

**结论：** 这是三个模型中表现最好的。AI4S 平衡度是施引跨学科性的最强预测因子，AI 引用年龄呈 U 型关系。

---

## 5. 模型 3：Rela_Dz（颠覆性指数）

### 5.1 OLS 回归

**模型公式：** Rela_Dz ~ 跨学科变量 + 控制变量

#### 拟合效果

| 指标 | 值 |
|------|:---:|
| R² | 0.259 |
| Adj R² | 0.209 |
| F-test | p < 0.001 *** |
| AIC | -271.3 |

#### 显著变量

| 变量 | 系数 | p值 | 解释 |
|------|:----:|:---:|:----:|
| **ai_ref_age_c_sq** | +0.001 | **<0.001** *** | 存在 U 型关系（转折点 ~10.2 年） |
| **num_references** | -0.0004 | **<0.001** *** | 参考文献越少，颠覆性越高 |
| ai_ref_age_c | -0.005 | 0.060 (边缘显著) | — |

#### U 型关系解释

ai_ref_age 与 Rela_Dz 呈 **U 型关系**（β₁ = -0.005, β₂ = +0.001）：
- 转折点（谷底）约在 **ai_ref_age ≈ 10.2 年**
- AI 引用年龄 < 10.2 年时，颠覆性随引用年龄增加而**下降**
- AI 引用年龄 > 10.2 年时，颠覆性随引用年龄增加而**上升**
- 在常见引用范围内（0-15 年），引用最新 AI 文献（0 年）的论文颠覆性最高（0.110），引用约 10 年 AI 文献的论文最低（0.047）

#### 不显著变量
- 所有跨学科核心变量（discipline_variety, similarity, balance, ai4s_balance）均不显著

### 5.2 Ordered Logit 回归（改进模型）

由于 Rela_Dz 分布严重偏态（67.7% 为低颠覆性），使用 Ordered Logit 作为稳健性检验。

#### 类别分布

| 类别 | 数量 | 占比 |
|:----:|:----:|:----:|
| 低颠覆性 (<0.05) | 130 | 67.7% |
| 中颠覆性 (0.05–0.2) | 47 | 24.5% |
| 高颠覆性 (≥0.2) | 15 | 7.8% |

#### 拟合效果

| 指标 | 值 |
|------|:---:|
| Pseudo R² | 0.257 |
| Log-Likelihood | -115.3 |
| AIC | 258.6 |
| **预测准确率** | **78.65%** |

#### 显著变量

| 变量 | 系数 | p值 | 解释 |
|------|:----:|:---:|:----:|
| **num_references** | -0.041 | **<0.001** *** | 参考文献越少，颠覆性越高（唯一显著变量） |

**结论：** 跨学科变量对颠覆性无显著预测能力。颠覆性主要由参考文献数量驱动——参考文献越少的论文越可能具有颠覆性。

---

## 6. 回归结果汇总表（标准化系数 β）

| 变量 | 模型1: Citation Impact | 模型2: Citing Interdisc. | 模型3: Disruptiveness |
|:-----|:---------------------:|:-----------------------:|:--------------------:|
| **核心自变量** | | | |
| Discipline Variety | 0.128 | 0.074 | −0.118 |
| Discipline Similarity | 0.017 | −0.070 | 0.048 |
| Discipline Balance | −0.070 | −0.062 | 0.060 |
| AI4S Balance | **0.140*** | **0.585***** | −0.058 |
| AI Ref Age (centered) | −0.119 | **−0.191*** | −0.177† |
| AI Ref Age Sq (centered) | 0.050 | **0.269**** | **0.413***** |
| **控制变量** | | | |
| Num Authors | **0.173*** | 0.008 | −0.046 |
| Publication Year (centered) | **−0.190**** | −0.018 | −0.109† |
| Num References | **0.405***** | 0.086 | **−0.245***** |
| Num Institutions | −0.016 | −0.020 | 0.152 |
| Intl. Collaboration | −0.060 | 0.074 | −0.150† |
| Open Access | 0.083 | 0.107† | −0.021 |
| **模型拟合指标** | | | |
| R² | 0.285 | 0.408 | 0.259 |
| Adj R² | 0.237 | 0.369 | 0.209 |
| AIC | 411.7 | −220.7 | −271.3 |
| N | 192 | 192 | 192 |

*注: 表中数值为标准化回归系数（β），可直接比较效应大小。† p<0.1, * p<0.05, ** p<0.01, *** p<0.001*

---

## 7. 综合发现总结

| 发现 | 模型1: Citation Impact | 模型2: Citing Interdisc. | 模型3: Disruptiveness |
|:-----|:---------------------:|:-----------------------:|:--------------------:|
| **AI4S Balance** | ✅ 显著 (+) | ✅✅ 高度显著 (+) | ❌ 不显著 |
| **AI Ref Age (U型)** | ❌ 不显著 | ✅ 显著 (U型) | ✅ 显著 (U型) |
| **Discipline Variety** | ❌ 不显著 | ❌ 不显著 | ❌ 不显著 |
| **Discipline Similarity** | ❌ 不显著 | ❌ 不显著 | ❌ 不显著 |
| **Discipline Balance** | ❌ 不显著 | ❌ 不显著 | ❌ 不显著 |
| **Num References** | ✅✅ 高度显著 (+) | ❌ 不显著 | ✅✅ 高度显著 (-) |
| **模型 R²** | 0.285 | **0.408** | 0.259 |

### 核心结论

1. **AI4S Balance 是最重要的跨学科指标**——它显著预测引用影响力和施引跨学科性，但对颠覆性无影响
2. **AI Ref Age 存在 U 型关系**——在施引跨学科性和颠覆性模型中均发现 U 型模式，转折点（谷底）分别在约 12.9 年和 10.2 年
3. **传统跨学科指标（Variety, Similarity, Balance）在所有模型中均不显著**——说明简单的跨学科多样性指标不足以预测研究影响力
4. **颠覆性主要由参考文献数量驱动**——参考文献越少，颠覆性越高（与直觉一致：颠覆性工作通常引用较少）
5. **模型2（施引跨学科性）表现最佳**——R²=0.408，是三个模型中解释力最强的

---

## 8. 图表目录

所有图表位于 `results/regression_v2/实验报告/` 目录下。

| 图表文件 | 用途 | 推荐位置 |
|:---------|:-----|:---------|
| `regression_coefficients.png` | 三个模型的系数对比图 | 主结果图 |
| `correlation_heatmap.png` | 变量相关性矩阵 | 附录 |
| `rela_dz_summary_figure.png` | Rela_Dz 模型 2×2 组合图 | 模型3结果 |
| `rela_dz_predicted_probs.png` | 预测概率曲线 | 模型3讨论 |
| `rela_dz_coefficient_forest.png` | 系数森林图 | 模型3附录 |
| `ordered_logit_Rela_Dz_confusion.png` | 混淆矩阵 | 稳健性检验 |
| `scatter_matrix.png` | 核心变量散点矩阵 | 附录 |
| `nr_sampling_bias_analysis.png` | N_R 采样偏差分析 | 方法学讨论 |

---

*报告生成时间：2026-05-17*
*数据来源：OpenAlex API，192 篇 AI4S 相关论文*
