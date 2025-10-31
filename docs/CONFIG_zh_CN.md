# dynamic_eur_acquisition 配置说明（中文）

本文件说明如何在 AEPsych 的 INI 配置中使用 PoolBasedGenerator 搭配 EUR 采集函数（V4/V5），包括字段含义、默认值与示例（连续回归与序数结果两种场景）。

V5 核心特性：对序数模型（OrdinalGPModel + OrdinalLikelihood）安全，信息项在 torch 后验上计算；覆盖项使用 numpy Gower 距离。V4 在序数模型上可能产生矩阵尺寸不匹配的警告，不建议与 OrdinalGPModel 搭配。

---

## 适用范围

- Pool-based 主动学习（固定候选池），生成器：`PoolBasedGenerator`。
- 采集函数：
  - `EURAcqfV5`（推荐，序数安全）。
  - `EURAcqfV4`（旧版，适用于回归 GP；不推荐用于序数模型）。
- 模型：
  - 连续回归：`GPRegressionModel`。
  - 序数：`OrdinalGPModel` + `OrdinalLikelihood`（指定 `n_levels`）。

---

## INI 结构总览

典型的 INI 至少包含以下段落：

- `[common]`：策略名、维度边界、结果类型等全局设置。
- `[opt]`：优化策略（模型、预算、生成器等）。
- `[PoolBasedGenerator]`：生成器及采集函数参数（含 EUR 的参数）。
- `[<ModelSection>]`：模型具体设置（如 `GPRegressionModel` 或 `OrdinalGPModel`/`OrdinalLikelihood`）。

> 提示：采集函数（`acqf`）的参数与生成器参数放在同一个 `[PoolBasedGenerator]` 段中传入。

---

## 字段一览（重点在 `[PoolBasedGenerator]`）

- 通用：
  - `acqf`：采集函数名。
    - 可选：`EURAcqfV5`（推荐）、`EURAcqfV4`（仅回归）。
  - `allow_resampling`（默认 `false`）：是否允许从池中重复采样已选过的点。
  - `shuffle`（默认 `true`）：初始化时是否打乱池顺序（仅无模型时有用）。
  - `pool_points`（必填）：候选池（二维数组，形如 `[[...],[...],...]`）。

- EUR 公共参数（V4/V5 兼容键，但生效逻辑不同）：
  - `gamma`（默认 `0.3`）：覆盖项权重，实际目标为 `info + gamma * coverage`（内部做标准化后再相加）。
  - `coverage_method`（默认 `min_distance`）：覆盖计算方法。当前实现支持 `min_distance`（最小 Gower 距离），也可按需扩展。
  - `variable_types_list`（可选）：变量类型列表，用于 Gower 距离区分范畴/整数变量。例如：`[categorical, categorical, integer, categorical]`。未提供时，V5 会从 AEPsych 参数变换中尽力推断（categorical/round）。
  - `debug_components`（默认 `false`）：调试用，导出 info/cov 组件（仅内部缓存）。

- V5（序数安全）专用/常用参数：
  - `use_entropy`（默认 `true`）：是否在信息项中加入基于 cutpoints 的类别熵（阈值附近更不确定）。
  - `use_cut_sensitivity`（默认 `true`）：是否在信息项中加入“距最近 cutpoint 的敏感度”项。

- V4/回归 ΔVar 相关的兼容参数（V5 当前默认不启用 ΔVar 信息路径，因此这些参数对 V5 不生效，仅保留兼容）：
  - `lambda_min`（默认 `0.2`）、`lambda_max`（默认 `2.0`）：交互项权重的动态范围。
  - `tau_1`（默认 `0.5`）、`tau_2`（默认 `0.1`）：基于参数方差比率的分段阈值，用于调节 `lambda`。
  - `interaction_terms`（可选，示例：`(0,1),(0,2)`）：指定二阶交互项（回归 ΔVar 路径使用）。
  - `noise_variance`（默认 `1.0`）、`prior_variance`（默认 `1.0`）：线性 ΔVar 计算的噪声/先验方差（回归 ΔVar 使用）。

> 结论：V5 目前默认通过 torch 后验（均值/方差）构造信息，不走回归 ΔVar 通道，因此 `lambda_*`/`tau_*`/`interaction_terms` 等键被接受但不会影响计算。若后续需要对回归任务恢复 ΔVar 逻辑，可在 V5 内对 GPRegressionModel 分支开启该路径。

---

## `[common]` 常见字段

- `strategy_names = [opt]`
- `outcome_types = [continuous]`
  - 注：即便做序数实验，服务器层面通常仍以数值型记录结果，真正的序数建模由模型与 likelihood 决定（见下方模型段）。
- `stimuli_per_trial = 1`
- `parnames = [<p1>, <p2>, ...]`
- `lb = [...]`/`ub = [...]`：各维的标准化边界（通常在 `[0,1]`）。

## `[opt]` 常见字段

- `model`：`GPRegressionModel` 或 `OrdinalGPModel`。
- `min_asks`/`max_asks`：预算。
- `refit_every`：每多少步重拟合一次（例如 `1`）。
- `generator = PoolBasedGenerator`：启用池式生成器。

## 模型段落示例

- 连续回归：
  - `[GPRegressionModel]`（使用默认 mean/covar/lk）。
- 序数：
  - `[OrdinalGPModel]` + `[OrdinalLikelihood]`（至少需要 `n_levels`）。

---

## 配置示例

### 1）连续回归 + EUR_V5（建议用于对照/连续任务）

```ini
[common]
strategy_names = [opt]
outcome_types = [continuous]
stimuli_per_trial = 1
parnames = [x1, x2, x3]
lb = [0.0, 0.0, 0.0]
ub = [1.0, 1.0, 1.0]

[opt]
model = GPRegressionModel
min_asks = 50
max_asks = 50
refit_every = 1
generator = PoolBasedGenerator

[PoolBasedGenerator]
acqf = EURAcqfV5
allow_resampling = false
shuffle = true
pool_points = [[0.0,0.0,0.0]]  # 运行时注入真实池
# V5 常用参数
gamma = 0.3
coverage_method = min_distance
use_entropy = true
use_cut_sensitivity = true
# Gower 变量类型（可选）
variable_types_list = [categorical, integer, categorical]
# 兼容键（V5 当前默认不生效）
lambda_min = 0.2
lambda_max = 2.0
tau_1 = 0.5
tau_2 = 0.1

[GPRegressionModel]
# 默认设置
```

### 2）序数 + EUR_V5（推荐）

```ini
[common]
strategy_names = [opt]
outcome_types = [continuous]
stimuli_per_trial = 1
parnames = [color, layout, font_size, animation]
lb = [0.0, 0.0, 0.0, 0.0]
ub = [1.0, 1.0, 1.0, 1.0]

[opt]
model = OrdinalGPModel
min_asks = 50
max_asks = 50
refit_every = 1
generator = PoolBasedGenerator

[PoolBasedGenerator]
acqf = EURAcqfV5
allow_resampling = false
shuffle = true
pool_points = [[0.0,0.0,0.0,0.0]]
# V5 信息/覆盖参数
gamma = 0.3
coverage_method = min_distance
use_entropy = true
use_cut_sensitivity = true
variable_types_list = [categorical, categorical, integer, categorical]

[OrdinalLikelihood]
n_levels = 5

[OrdinalGPModel]
# 其它核/均值配置按需设置
```

> 注意：V5 当前仅支持 `q=1`（一次选择一个点）。

---

## V4 vs V5 兼容性要点

- V4（`EURAcqfV4`）：
  - 线性 ΔVar 信息路径基于 numpy；对 GPRegressionModel 友好。
  - 与 OrdinalGPModel 搭配时可能触发矩阵尺寸不匹配的告警/失败，原因是 numpy 线性路径与序数模型的内部形状不一致。
- V5（`EURAcqfV5`）：
  - 统一使用 torch 后验（均值/方差）构造信息项；对 Ordinal/回归均稳健。
  - 覆盖仍用 numpy Gower，但仅用于“覆盖”分量。
  - 接受 V4 的大多数键，但 ΔVar 相关键对 V5 当前默认不生效。

---

## 常见问题（FAQ）

- Q：我定义了 `interaction_terms`，为什么对 V5 没效果？
  - A：`interaction_terms` 只在“回归 ΔVar 信息路径”中生效。V5 默认以 torch 后验信息替代 ΔVar，因此不生效。若需要，我们可以仅对 `GPRegressionModel` 分支开启 ΔVar 路径。
- Q：`variable_types_list` 有什么用？
  - A：用于 Gower 距离区分范畴/整数变量，从而更合理地衡量“覆盖”。未提供时，V5 会从参数变换中尽量推断。
- Q：为什么我用 V4 在序数模型上看到很多 `matmul` 尺寸警告？
  - A：V4 的 numpy ΔVar 路径与序数模型的内部形状不兼容。建议改用 V5。

---

## 结论与建议

- 回归/连续任务：V4 与 V5 都可用；如需 `interaction_terms` 等 ΔVar 控制，可等我们为 V5 恢复回归分支的 ΔVar 路径；目前推荐直接用 V5（更稳健）。
- 序数任务：请使用 V5（`EURAcqfV5`）并搭配 `OrdinalGPModel` + `OrdinalLikelihood`（指定 `n_levels`）。
- 覆盖权重 `gamma` 和 `coverage_method` 是最常用的可调参数；`use_entropy`/`use_cut_sensitivity` 对序数阈值附近的探索尤为关键。
