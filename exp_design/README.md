# 实验设计文档索引

## 📚 文档结构

本文件夹包含模块化的实验设计文档，方便AI快速理解和人工查阅。

---

## 🎯 快速导航

### 1. 快速理解（必读）

**文件**：`00_CORE_IDEAS.md`（~2页）

**适用场景**：
- ✅ 第一次了解实验设计
- ✅ 向AI简要说明需求
- ✅ 快速回顾核心思路

**包含内容**：
- 实验目标和背景
- 两阶段分离原则（采样 vs 分析）
- 三阶段流程（预热、主动学习、验证）
- 关键决策总结
- 检查清单

---

### 2. 预热方案详解（实施前必读）

**文件**：`01_WARMUP_STRATEGY.md`（~8页）

**适用场景**：
- ✅ 准备开始预热阶段数据收集
- ✅ 理解为什么选择策略A（Space-filling）
- ✅ 学习如何进行Meta-learning分析

**包含内容**：
- 预热目标和配置（8人×20次）
- Maximin LHS生成代码
- 数据收集流程
- Meta-learning分析步骤
  - 拟合初步混合模型
  - 筛选交互对
  - 确定采集函数超参数
- 质量评估指标
- 预热报告模板

---

### 3. 完整实验设计（参考手册）

**文件**：`../EXPERIMENT_DESIGN.md`（~15页）

**适用场景**：
- ✅ 需要查阅完整的技术细节
- ✅ 理解主动学习阶段配置
- ✅ 查看数据分析代码示例
- ✅ 理论支撑和文献引用

**包含内容**：
- 所有阶段的详细说明
- 采集函数参数配置
- 混合效应模型代码
- APA格式报告示例
- 常见问题处理
- 完整的检查清单

---

## 🔄 文档使用流程

### 阶段1：设计阶段

```
1. 阅读 00_CORE_IDEAS.md
   → 理解整体思路

2. 阅读 01_WARMUP_STRATEGY.md
   → 学习预热方案

3. 需要细节时查阅 ../EXPERIMENT_DESIGN.md
   → 完整技术参考
```

### 阶段2：实施阶段

```
预热前：
  → 01_WARMUP_STRATEGY.md（"检查清单"章节）

预热后：
  → 01_WARMUP_STRATEGY.md（"Meta-learning分析"章节）

主动学习：
  → ../EXPERIMENT_DESIGN.md（"阶段2"章节）

数据分析：
  → ../EXPERIMENT_DESIGN.md（"最终数据分析"章节）
```

### 阶段3：向AI提问

**简短提问**（让AI快速理解）：
```
请阅读 exp_design/00_CORE_IDEAS.md，
然后帮我 [具体问题]
```

**详细提问**（需要技术细节）：
```
请阅读 exp_design/01_WARMUP_STRATEGY.md 中的"Meta-learning分析"部分，
然后帮我 [具体问题]
```

**完整参考**：
```
请阅读 EXPERIMENT_DESIGN.md，
然后帮我 [复杂问题]
```

---

## 📋 文档对比

| 文档 | 长度 | 用途 | 阅读时间 |
|------|------|------|---------|
| `00_CORE_IDEAS.md` | 2页 | 快速理解 | 5分钟 |
| `01_WARMUP_STRATEGY.md` | 8页 | 预热实施 | 15分钟 |
| `../EXPERIMENT_DESIGN.md` | 15页 | 完整参考 | 30分钟 |

---

## 🔗 其他相关文件

位于上级目录 `../`：

- **策略对比分析**：`strategy_matrix_critique.md`
  - 为什么选择策略A（Space-filling）？
  - 策略B的问题在哪里？
  - 详细的理论分析

- **预热代码示例**：`warmup_strategy_example.py`
  - Python实现的预热设计生成
  - 策略A、策略C的代码对比
  - 覆盖度计算示例

- **采集函数源码**：`eur_anova_pair.py`
  - EURAnovaPairAcqf完整实现
  - 动态权重机制（λ_t、γ_t）
  - 详细的docstring和注释

---

## ⚡ 常见场景索引

### 场景1：我是第一次接触这个实验设计

**推荐阅读顺序**：
1. `00_CORE_IDEAS.md`（必读）
2. `01_WARMUP_STRATEGY.md`（如需实施）
3. `../EXPERIMENT_DESIGN.md`（需要时查阅）

---

### 场景2：我要开始收集预热数据

**推荐阅读**：
- `01_WARMUP_STRATEGY.md`（完整阅读）
- 重点关注：
  - "预热配置"章节 → 参数设置
  - "生成代码"章节 → LHS实现
  - "数据收集"章节 → 流程和记录格式
  - "检查清单"章节 → 实施前准备

---

### 场景3：预热完成，需要分析数据

**推荐阅读**：
- `01_WARMUP_STRATEGY.md` 的"Meta-learning分析"章节
- 按步骤执行：
  1. 拟合初步混合模型
  2. 筛选交互对
  3. 确定超参数
  4. 生成Meta-learning报告

---

### 场景4：我要向AI解释我的需求

**快速提问**（AI第一次接触）：
```
Context: 我正在设计一个心理学实验，
请先阅读 exp_design/00_CORE_IDEAS.md 了解背景。

问题：[你的具体问题]
```

**详细提问**（AI已理解背景）：
```
Context: 我正在实施预热阶段，
请参考 exp_design/01_WARMUP_STRATEGY.md。

问题：[具体技术问题]
```

---

### 场景5：我忘记了某个参数的含义

**快速查找**：

| 参数 | 文档位置 | 章节 |
|------|---------|------|
| `lambda_max` | `00_CORE_IDEAS.md` | "关键决策"表格 |
| `tau_n_max` | `../EXPERIMENT_DESIGN.md` | "关键参数配置参考" |
| `r_t` | `../eur_anova_pair.py` | 第952-956行注释 |
| 交互对筛选 | `01_WARMUP_STRATEGY.md` | "步骤2：筛选交互对" |

---

## 💡 提示

**文档太长看不完？**
- 只看带 ✅ 标记的章节
- 使用 Ctrl+F 搜索关键词
- 查看文档开头的"目录"（如果有）

**不确定该读哪个文档？**
- 永远从 `00_CORE_IDEAS.md` 开始
- 需要代码时查 `01_WARMUP_STRATEGY.md`
- 需要理论时查 `../EXPERIMENT_DESIGN.md`

**向AI提问时不知道提供哪个文档？**
- 第一次提问：`00_CORE_IDEAS.md`
- 技术细节：`01_WARMUP_STRATEGY.md` 或 `../EXPERIMENT_DESIGN.md`
- 策略争议：`../strategy_matrix_critique.md`

---

## 📧 反馈

如果文档有不清楚的地方，可以：
1. 查看 `../EXPERIMENT_DESIGN.md` 的"常见问题"章节
2. 阅读 `../strategy_matrix_critique.md` 了解设计决策
3. 查看 `../eur_anova_pair.py` 的详细注释

---

**最后更新**：2025-11-05
