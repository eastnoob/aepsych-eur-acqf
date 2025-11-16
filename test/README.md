# Test Suite for EURAnovaPairAcqf

This directory contains all test scripts for the dynamic EUR acquisition function.

## Directory Structure

```
test/
├── unit_tests/                 # Unit tests for individual components
│   └── [Component-specific tests]
│
├── integration_tests/          # Integration tests for complete workflows
│   └── core_functionality_test_20251105.py
│
└── examples/                   # Example usage scripts
    └── [Usage examples]
```

## Test Categories

### Unit Tests (`unit_tests/`)
Tests for individual components and functions:
- `gower_distance` module tests
- `gp_variance` module tests
- Individual methods of `EURAnovaPairAcqf`

### Integration Tests (`integration_tests/`)
End-to-end tests for complete workflows:
- **core_functionality_test_20251105.py**: Comprehensive test covering:
  - Module imports
  - GP model creation
  - Acquisition function initialization
  - Forward pass computation
  - Debug mode
  - Mixed variable types
  - Gower distance calculation
  - GP variance calculator

### Examples (`examples/`)
Usage examples and demonstration scripts:
- Basic usage examples
- Advanced configuration examples
- Real-world use cases

## Running Tests

### Run Integration Tests
```bash
pixi run python test/integration_tests/core_functionality_test_20251105.py
```

### Run All Tests
```bash
# Add a test runner script here when available
```

## Naming Convention

All test files should follow this naming pattern:
```
<functionality_description>_<YYYYMMDD>.py
```

Example:
- `core_functionality_test_20251105.py` - Core functionality test created on 2025-11-05
- `categorical_variables_test_20251106.py` - Categorical variables test created on 2025-11-06

## Test Requirements

Tests assume the following environment:
- PyTorch
- BoTorch
- GPyTorch
- NumPy
- AEPsych (parent package)

All tests should be run from the `dynamic_eur_acquisition` directory.
