# 文档快速索引

## 📖 按阅读顺序

1. **00_CORE_IDEAS.md** - 核心思路（2页，必读）
2. **01_WARMUP_STRATEGY.md** - 预热方案（8页，实施前必读）
3. **99_FULL_REFERENCE.md** - 完整参考（15页，需要时查阅）
4. **README.md** - 文档使用指南

---

## ⚡ 快速跳转

### 我想快速理解整体设计
→ `00_CORE_IDEAS.md`

### 我要开始收集预热数据
→ `01_WARMUP_STRATEGY.md` + "检查清单"章节

### 我要分析预热数据（Meta-learning）
→ `01_WARMUP_STRATEGY.md` + "Meta-learning分析"章节

### 我要配置采集函数
→ `99_FULL_REFERENCE.md` + "关键参数配置参考"章节

### 我要拟合最终的混合模型
→ `99_FULL_REFERENCE.md` + "最终数据分析"章节

### 我要向AI解释我的需求
→ 提供 `00_CORE_IDEAS.md`

---

## 🔍 按关键词查找

| 关键词 | 文档 | 章节/行号 |
|--------|------|----------|
| **Space-filling** | 01_WARMUP_STRATEGY.md | "策略A：Maximin LHS" |
| **λ_t（交互权重）** | 00_CORE_IDEAS.md | "关键决策"表格 |
| **r_t（参数收敛率）** | 99_FULL_REFERENCE.md | "关键参数配置" |
| **交互对筛选** | 01_WARMUP_STRATEGY.md | "步骤2：筛选交互对" |
| **混合效应模型** | 99_FULL_REFERENCE.md | "混合效应模型配置" |
| **8人×20次** | 01_WARMUP_STRATEGY.md | "为什么是8人×20次？" |
| **前5次随机** | 00_CORE_IDEAS.md | "阶段2：主动学习" |
| **样本预算** | 00_CORE_IDEAS.md | "样本预算"表格 |

---

## 📂 文件说明

```
exp_design/
├── INDEX.md                  ← 你在这里
├── README.md                 ← 详细的使用指南
├── 00_CORE_IDEAS.md          ← 核心思路（2页）
├── 01_WARMUP_STRATEGY.md     ← 预热方案（8页）
└── 99_FULL_REFERENCE.md      ← 完整参考（15页）
```

---

## 💡 使用建议

**第一次阅读**：
1. 00_CORE_IDEAS.md（5分钟）
2. 如需实施 → 01_WARMUP_STRATEGY.md（15分钟）
3. 如需细节 → 99_FULL_REFERENCE.md（30分钟）

**实施阶段**：
- 预热前 → 01_WARMUP_STRATEGY.md（检查清单）
- 预热后 → 01_WARMUP_STRATEGY.md（Meta-learning）
- 主动学习 → 99_FULL_REFERENCE.md（阶段2）
- 数据分析 → 99_FULL_REFERENCE.md（分析章节）

**向AI提问**：
- 快速理解 → 提供 00_CORE_IDEAS.md
- 技术细节 → 提供 01_WARMUP_STRATEGY.md 或 99_FULL_REFERENCE.md
