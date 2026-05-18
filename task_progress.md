# 回归分析任务清单

## 第一阶段：补充控制变量数据
- [x] 创建 `collect_controls.py`（已有，但缺少 open_access）
- [ ] 修改 `collect_controls.py`，加入 `open_access` 字段
- [ ] 运行 `collect_controls.py` 补充控制变量数据，生成新 CSV

## 第二阶段：回归分析
- [ ] 创建 `regression_analysis.py` 主脚本
  - [ ] 数据加载与清洗
  - [ ] 描述性统计（均值、标准差、相关性矩阵）
  - [ ] 模型1：citation_impact 回归
  - [ ] 模型2：cit_interdisciplinarity 回归
  - [ ] 模型3：Rela_Dz 回归
  - [ ] 模型诊断（VIF、异方差检验）
  - [ ] 可视化（回归系数图、散点图、残差诊断图）
- [ ] 运行回归分析，输出结果
- [ ] 验证结果合理性
