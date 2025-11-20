# Step 1（Phase 1）采样简要说明

## 1. 概览 ✅

Step 1 用于生成预热（Phase 1）采样方案。实现里使用所谓“5 步采样法”（改进版），包含：Core-1 战略性重复点、Boundary（边界点）去重、Core-2a/2b（主效应/交互）随机池分配、LHS 全局填充。核心目标是：保证重复点以便估计被试内一致性（ICC），同时覆盖空间以估计主效应/初步筛选交互、并为 Phase 2 提供初始参数（例如 λ）。

---

## 2. Step 1 的“采样方法” — 重点（通俗）🔍

- Core-1（精简说明）: 选 8 个代表性配置（全最小/全最大/中位/交替/前后半/中位扰动），每个被试都重复这些配置以获得一致性（ICC）估计；如果不够再用 MaxiMin 补充。
- 每个 Core-1 的目标不是随机采样，而是挑选能代表极值、中心和互补组合的点。实现中为了把“理想”目标映射到离散的设计库，使用了 Gower 距离（`gower_distance`）来找“最接近”的真实配置。
- 如果 Core-1 定义的点不够，或需要更多多样性，会用 MaxiMin 准则补充（在已选点间选择最远的点以提高覆盖）——即“尽可能选相互差别大的点”。
- 后续步骤还会：
  - 选边界点（单维极值）并去重
  - 用随机/设计化方法生成 Core-2（主效应/交互）的点池
  - 用 LHS（Latin Hypercube Sampling）生成全局填充点并映射到最近的离散配置
- 最终把这些点分摊到被试：每位被试都包含 Core-1 的重复点，并从配置池中分配一定数量的其它点。

---

## 3. 为什么这样做？目的与益处 🎯

- 保证重复（Core-1） → 用来估算被试间差异/被试内一致性（ICC），这对混合效应模型很重要；同一配置跨被试的重复让模型能稳定估计个体差别。
- 覆盖主效应与交互候选（Core-2a/2b） → 初步识别重要因子与交互对，后续 Phase 2 可以专注这些方向。
- 边界点确保对极端值进行测试（检查非线性/边缘效应）
- LHS 使整个设计空间均匀覆盖，帮助发现未预期的模式或作用
- Gower 距离 + MaxiMin + LHS 的组合使得对混合数据类型（数值/分类/布尔）也能合理距离衡量与采样

---

## 4. 能把握到什么？（Step 1 提供的信息）📈

- ICC 和重复测量信息：通过 Core-1 的重复样本可以估计受试者层面的方差，判断是否需要更多被试或增加重复点。
- 主效应估计：Core-2a 与 LHS 覆盖提供了对每个因子主效应的初步估计。
- 交互潜力筛选：通过残差模式（四象限）+ BIC 改进 + 方差解释，Phase 1 能筛选出 top-k 交互对。
- λ（交互权重）初估：Phase 1 会计算主效应与含交互模型的改进量，并据此估计 λ，供 Phase 2 自适应采样使用。
- 设计空间覆盖率：估算独特配置占比、平均每水平样本等指标以判断是否存在覆盖不足。

---

## 5. 简要表格（采样类型、具体是什么、目标、数量）🧾

| 采样类型 | 具体采样哪些 | 目标 | 数量（实现中/建议） |
|---|---:|---|---:|
| Core-1（战略性重复点） | 全最小 / 全最大 / 全中位数 / 奇偶交替 / 前后半分 / 中位扰动 + MaxiMin 补充 | 提供重复点以估算 ICC 和被试内一致性；稳定混合效应模型 | 默认 8 个配置 × 每人（代码中 n_core1_configs=8） |
| Core-2a（主效应池） | D-optimal 或随机选择主效应覆盖点 | 覆盖各因子水平以估计主效应 | 由预算分配（约占剩余预算的40%） → 代码里按比例计算（n_core2a） |
| Core-2b（交互候选） | 交互对探索点（可选） | 初步筛选可能的重要交互对，供 Phase 2 深挖 | 由预算决定（默认 25 个或根据预算按比例） |
| 边界点（Boundary） | 每个因子最小/最大点（去重） | 覆盖极端配置，检查边缘效应／非线性 | 至少 2 * 因子数（去重以后的数量） |
| 全局 LHS | Latin Hypercube sampling 映射到最邻近离散配置（使用 Gower） | 增强整个空间的覆盖均匀性 | 剩余预算按比例分配（代码中 n_lhs） |

注：数量由预算估算模块 `WarmupBudgetEstimator.estimate_budget_requirements` 根据被试数和每人 trials 自动分配。

### 5b. 预算分配的代码规则（简述）

- 总预算 = n_subjects × trials_per_subject
- Core-1 固定为 8 × n_subjects（每人重复）
- 剩余预算 = 总预算 - Core-1
- 若包含交互 (skip_interaction=False): 剩余预算按 40% (Core-2a) / 28% (Core-2b) / 32% (Boundary+LHS) 分配
- 若跳过交互 (skip_interaction=True): 剩余预算按 55% (Core-2a) / 0% (Core-2b) / 45% (Boundary+LHS) 分配
- 边界与 LHS 从“探索 (explore)”预算中继以 40%/60% 分配
- n_core2a 会受到 min_limit 的约束（取决于因子水平数和因子数）；Core-2b 也有最小值（25）

---

## 6. 代码里关键词与位置 🔎

- `WarmupSampler._select_core1_strategic`：实现 Core-1 的语义/策略点选择
- `WarmupSampler._find_closest_config`：用 Gower 距离把理想目标映射到离散配置
- `WarmupSampler._select_maximin_next`：MaxiMin 准则补充点
- `WarmupSampler._select_boundary_configs`：边界点、去重逻辑
- `WarmupSampler._select_lhs_global`：LHS 全局采样映射与去重
- `WarmupBudgetEstimator.estimate_budget_requirements`：预算分配策略（Core-1/2a/2b/边界/LHS 的比重）
- `analyze_phase1.py` / `phase1_analyzer.py`：用 Phase 1 数据来筛选交互对与估计 λ

---

## 7. 进一步建议 & 风险 ⚠️

- 如果设计空间非常大或类目非常多，LHS 映射可能会重复很多配置，建议增加 LHS 的生成数或增加 Maximin 补充。
- 如果被试数量 < 5，ICC 估计不稳定，需谨慎解释 λ 与交互筛选结果。
- Core-1 固定点如果过多会占用预算（降低覆盖），过少则 ICC 估计不稳，建议保持 6-10 个重复点。

---

## 8. 结论（简洁）

Step 1 的采样方法以“保证可重复点”（Core-1）为核心，结合边界、主效应池、交互初筛和全局 LHS 填充，确保既能估计个体差异（ICC），又能在 Phase 2 之前筛出重要交互、估计 λ 并得到可用的覆盖性指标。

---

## 附：5 被试 × 每人 25 次 — 算法例子（数值说明）

示例基于仓库中 `data/only_independences/...6vars...1200combinations.csv`（6 因子，最大水平 5）用法：

- 总预算 = 5 × 25 = 125
- Core-1 = 8 × 5 = 40（固定重复点）
- 剩余预算 = 125 - 40 = 85

(skip_interaction=True 时的分配)

- Core-2a ≈ max(int(85 × 55%) = 46, n_core2a_min = 26) → 46
- Core-2b = 0
- 探索预算 = 85 - 46 = 39
  - 边界 = max(2 × 因子数 = 12, int(39×40%) = 15) → 15
  - LHS = 39 - 15 = 24

- 汇总：Core-1:40；Core-2a:46；Core-2b:0；Boundary:15；LHS:24；总计 125
- 独特配置数 ≈ 8 + 46 + 0 + 15 + 24 = 93 → 独特配置占比 ≈ 93/125 = 74.4%（满足建议 ≥70%）
- pool_size = Core-2a + Boundary + LHS = 46 + 15 + 24 = 85
- pool_size_per_subject = 85 // 5 = 17
- 每人样本 = Core-1 (8) + pool_per_subject (17) = 25（刚好匹配 trials_per_subject）

此示例演示：算法通过“核心重复 + 池化 + 均匀分配”保证重复测量、覆盖空间并达到预算目标。
