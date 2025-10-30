# Test Suite

This directory contains all tests and examples for the Variance Reduction with Coverage Acquisition Function.

## Directory Structure

### `unit_tests/` - Unit Tests

Fast, focused tests for individual components.

- **`simple_test.py`**: Quick validation test (~30s)
  - Tests basic functionality
  - Good for quick verification after changes
  
- **`test_acquisition_function.py`**: Comprehensive unit tests
  - Tests all components in isolation
  - Includes edge cases and error handling

**Run unit tests:**

```bash
cd test/unit_tests
pixi run python simple_test.py
pixi run python test_acquisition_function.py
```

### `integration_tests/` - Integration Tests

Tests that validate end-to-end workflows.

- **`complete_test.py`**: Full functionality test
  - Tests all components working together
  - Validates Gower distance, GP variance, acquisition function
  - Tests mixed variables, config loading
  - Runtime: ~1 minute
  
- **`simulation_experiment.py`**: Simulation with synthetic data
  - 30 iterations of active learning
  - Tests dynamic weighting adaptation
  - Generates visualization
  - Runtime: ~2 minutes
  
- **`end_to_end_experiment.py`**: Complete active learning workflow
  - Follows standard AEPsych workflow
  - Initialization → Optimization → Evaluation
  - Uses standard config format
  - Generates comprehensive plots
  - Runtime: ~3 minutes

**Run integration tests:**

```bash
cd test/integration_tests
pixi run python complete_test.py
pixi run python simulation_experiment.py
pixi run python end_to_end_experiment.py
```

### `examples/` - Usage Examples

Practical examples showing different use cases.

- **`example_usage.py`**: 5 usage scenarios
  1. Basic usage with default parameters
  2. Custom parameters
  3. Loading from config file
  4. With interaction effects
  5. Mixed continuous/categorical variables

**Run examples:**

```bash
cd test/examples
pixi run python example_usage.py
```

## Test Coverage

### Unit Tests (`unit_tests/`)

- ✓ Module imports
- ✓ Gower distance computation
- ✓ GP variance calculation
- ✓ Acquisition function evaluation
- ✓ Config file loading
- ✓ Mixed variable handling
- ✓ Edge cases and error handling

### Integration Tests (`integration_tests/`)

- ✓ Full active learning workflow
- ✓ Dynamic weighting adaptation
- ✓ Variance reduction tracking
- ✓ Spatial coverage
- ✓ Model fitting and prediction
- ✓ Visualization generation
- ✓ Config integration

## Expected Outputs

### `complete_test.py`

```
✓ 测试 1/7: 模块导入
✓ 测试 2/7: Gower距离计算
✓ 测试 3/7: GP方差计算
✓ 测试 4/7: 基本采集函数
✓ 测试 5/7: 带交互项的采集函数
✓ 测试 6/7: 配置文件加载
✓ 测试 7/7: 混合变量类型

所有测试通过! (7/7)
```

### `simulation_experiment.py`

```
30 iterations completed
Final dataset: 45 samples
Test MSE: 0.040163
Generated files:
- simulation_results.png
- simulation_results_data.npz
- simulation_results_history.npz
```

### `end_to_end_experiment.py`

```
Complete end-to-end experiment:
1. Initialization: 15 samples (Sobol)
2. Optimization: 30 iterations
3. Final evaluation: R² = 0.968

Generated files:
- end_to_end_results.png (9-panel visualization)
- end_to_end_results.npz (results data)
```

## Quick Start

**Run all tests sequentially:**

```bash
# From project root
cd extensions/dynamic_eur_acquisition

# Unit tests
pixi run python test/unit_tests/simple_test.py
pixi run python test/unit_tests/test_acquisition_function.py

# Integration tests
pixi run python test/integration_tests/complete_test.py
pixi run python test/integration_tests/simulation_experiment.py
pixi run python test/integration_tests/end_to_end_experiment.py

# Examples
pixi run python test/examples/example_usage.py
```

**Quick validation (recommended after changes):**

```bash
pixi run python test/unit_tests/simple_test.py
pixi run python test/integration_tests/complete_test.py
```

## Test Data

All tests use synthetic data with known properties:

- True function: y = 2x₁ + 3x₂ + 1.5x₃ + 1.5x₁x₂ + 2x₂x₃ + noise
- 3 continuous parameters: x₁, x₂, x₃ ∈ [0, 1]
- Known interactions: (x₁, x₂) and (x₂, x₃)
- Gaussian noise: σ = 0.3

## Troubleshooting

**Import errors:**

- Make sure you're running from project root or test directory
- Acquisition function supports both relative and absolute imports

**Config file errors:**

- Check encoding (must be UTF-8)
- Verify section names match class name
- Ensure numeric values have no extra spaces

**Matplotlib warnings:**

- Chinese character warnings are cosmetic, ignore them
- Figures still save correctly as PNG files

**Pixi not found:**

- Ensure pixi is installed and in PATH
- Alternative: use `python test/...` directly with your environment

## Performance Benchmarks

| Test | Runtime | Samples | Iterations |
|------|---------|---------|------------|
| simple_test.py | ~30s | 45 | 30 |
| test_acquisition_function.py | ~45s | Various | - |
| complete_test.py | ~60s | Various | 7 tests |
| simulation_experiment.py | ~120s | 45 | 30 |
| end_to_end_experiment.py | ~180s | 45 | 30 + eval |

*Benchmarks on Intel i7, 16GB RAM*

## Additional Resources

- **API Documentation**: `../docs/README.md`
- **Quick Start Guide**: `../docs/QUICKSTART.md`
- **Implementation Details**: `../docs/IMPLEMENTATION_SUMMARY.md`
- **Configuration Templates**: `../configs/`
