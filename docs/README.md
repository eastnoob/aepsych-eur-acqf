# Dynamic EUR Acquisition Function

## Overview

The Dynamic EUR (Expected Uncertainty Reduction) Acquisition Function is an active learning tool designed for efficient experimental design in the AEPsych framework. It combines information gain with spatial coverage to intelligently select the most valuable experimental points.

## Features

- **Information Gain (EUR)**: Reduces uncertainty in model parameters
  - Main effects: Direct impact of individual features
  - Interaction effects: Combined impact of feature pairs
  
- **Spatial Coverage**: Ensures exploration of the design space
  - Uses Gower distance for mixed-type variables
  - Supports continuous and categorical variables
  
- **Dynamic Weighting**: Adapts exploration/exploitation balance
  - Automatically adjusts interaction term importance
  - Based on variance reduction progress

## Mathematical Formulation

### Acquisition Score

The total acquisition score for a candidate point $x$ is:

$$\alpha(x; D_t) = \alpha_{\text{info}}(x; D_t) + \alpha_{\text{cov}}(x; D_t)$$

### Information Gain Component

$$\alpha_{\text{info}}(x; D_t) = \frac{1}{|\mathcal{J}|} \sum_{j \in \mathcal{J}} \Delta \text{Var}_{GP}[\theta_j | D_t \cup \{x\}] + \lambda_t(r_t) \cdot \frac{1}{|\mathcal{I}|} \sum_{(j, k) \in \mathcal{I}} \Delta \text{Var}_{GP}[\theta_{jk} | D_t \cup \{x\}]$$

where:

- $\theta_j$ represents the main effect of feature $j$
- $\theta_{jk}$ represents the interaction effect between features $j$ and $k$
- $\Delta \text{Var}_{GP}$ is the variance reduction from adding point $x$

### Spatial Coverage Component

$$\alpha_{\text{cov}}(x; D_t) = \gamma \cdot \text{COV}(x; D_t)$$

where $\text{COV}(x; D_t)$ is the minimum Gower distance from $x$ to existing samples.

### Dynamic Interaction Weight

The interaction weight $\lambda_t$ adapts based on variance reduction progress:

$$\lambda_t(r_t) = \begin{cases}
\lambda_{\min}, & r_t > \tau_1 \\
\lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \cdot \frac{\tau_1 - r_t}{\tau_1 - \tau_2}, & \tau_2 \leq r_t \leq \tau_1 \\
\lambda_{\max}, & r_t < \tau_2
\end{cases}$$

where:
- $r_t$ is the relative variance: $r_t = \frac{1}{|\mathcal{J}|} \sum_{j \in \mathcal{J}} \frac{\text{Var}_{GP}[\theta_j | D_t]}{\text{Var}_{GP}[\theta_j | D_0]}$
- $\tau_1, \tau_2$ are threshold parameters

## Installation

The module is self-contained within the `extensions/dynamic_eur_acquisition` folder. No additional installation is required beyond standard scientific Python packages:

- numpy
- scipy

## Quick Start

### Basic Usage

```python
import numpy as np
from dynamic_eur_acquisition import DynamicEURAcquisitionFunction

# Generate some training data
X_train = np.random.rand(30, 3)
y_train = X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2]

# Initialize acquisition function
acq_fn = DynamicEURAcquisitionFunction()

# Fit on training data
acq_fn.fit(X_train, y_train)

# Evaluate candidate points
X_candidates = np.random.rand(100, 3)
scores = acq_fn(X_candidates)

# Select best point
best_idx = np.argmax(scores)
next_point = X_candidates[best_idx]
```

### With Interaction Terms

```python
# Specify interaction terms (feature pairs)
acq_fn = DynamicEURAcquisitionFunction(
    interaction_terms=[(0, 1), (1, 2)],
    lambda_min=0.5,
    lambda_max=3.0
)

acq_fn.fit(X_train, y_train)
scores = acq_fn(X_candidates)
```

### Load from Configuration File

```python
# Create config.ini with your parameters
acq_fn = DynamicEURAcquisitionFunction(config_ini_path='config.ini')
acq_fn.fit(X_train, y_train)
```

### Active Learning Loop

```python
# Initialize with small dataset
X_train = np.random.rand(10, 3)
y_train = true_function(X_train)

acq_fn = DynamicEURAcquisitionFunction()

# Iteratively add points
for iteration in range(20):
    # Fit model
    acq_fn.fit(X_train, y_train)

    # Generate and evaluate candidates
    X_candidates = np.random.rand(200, 3)
    next_X, _ = acq_fn.select_next(X_candidates, n_select=1)

    # Query true function
    next_y = true_function(next_X)

    # Add to training set
    X_train = np.vstack([X_train, next_X])
    y_train = np.concatenate([y_train, next_y])
```

## Configuration

### Parameters

- **lambda_min** (float, default=0.2): Minimum weight for interaction effects
- **lambda_max** (float, default=2.0): Maximum weight for interaction effects
- **tau_1** (float, default=0.5): Upper threshold for relative variance
- **tau_2** (float, default=0.1): Lower threshold for relative variance
- **gamma** (float, default=0.3): Weight for spatial coverage term
- **interaction_terms** (list, optional): List of interaction term pairs, e.g., [(0,1), (2,3)]
- **noise_variance** (float, default=1.0): GP observation noise variance
- **prior_variance** (float, default=1.0): Prior variance for parameters
- **coverage_method** (str, default='min_distance'): Method for coverage computation

### Configuration File Format

```ini
[AcquisitionFunction]
lambda_min = 0.2
lambda_max = 2.0
tau_1 = 0.5
tau_2 = 0.1
gamma = 0.3
interaction_terms = (0,1);(1,2);(2,3)
noise_variance = 1.0
prior_variance = 1.0
coverage_method = min_distance
```

## API Reference

### DynamicEURAcquisitionFunction

Main acquisition function class.

#### Methods

- **`__init__(...)`**: Initialize with parameters or config file
- **`fit(X, y, variable_types=None)`**: Fit on training data
- **`__call__(X_candidates, return_components=False)`**: Evaluate candidates
- **`select_next(X_candidates, n_select=1)`**: Select top n points
- **`get_current_lambda()`**: Get current interaction weight
- **`get_variance_reduction_ratio()`**: Get current variance ratio

### Utility Functions

#### gower_distance

```python
gower_distance(x1, x2, variable_types=None, ranges=None)
```

Calculate Gower distance between two points.

#### compute_coverage

```python
compute_coverage(x, X_sampled, variable_types=None, ranges=None, method='min_distance')
```

Compute spatial coverage score for a candidate point.

#### GPVarianceCalculator

```python
gp = GPVarianceCalculator(noise_variance=1.0, prior_variance=1.0)
gp.fit(X, y, interaction_indices=[(0,1)])
main_var, inter_var = gp.compute_variance_reduction(X_new)
```

Calculate GP posterior variance for main and interaction effects.

## Examples

See `example_usage.py` for comprehensive examples including:
1. Basic usage with default parameters
2. Using interaction terms
3. Loading from configuration file
4. Active learning loop
5. Mixed continuous and categorical variables

Run examples:
```bash
python example_usage.py
```

## Testing

Run unit tests:
```bash
python -m unittest test.test_acquisition_function
```

Or from the test directory:
```bash
cd test
python test_acquisition_function.py
```

## Advanced Usage

### Mixed Variable Types

```python
# Define variable types
variable_types = {
    0: 'continuous',
    1: 'continuous',
    2: 'categorical',
    3: 'categorical'
}

acq_fn = DynamicEURAcquisitionFunction(variable_types=variable_types)
acq_fn.fit(X_train, y_train, variable_types=variable_types)
```

### Custom Interaction Terms

```python
# Specify only certain interactions
# E.g., for 5 features, only consider (0,1), (2,3), and (3,4)
interactions = [(0, 1), (2, 3), (3, 4)]

acq_fn = DynamicEURAcquisitionFunction(
    interaction_terms=interactions,
    lambda_min=0.3,
    lambda_max=2.5
)
```

### Monitoring Progress

```python
acq_fn.fit(X_train, y_train)

# Check current state
lambda_t = acq_fn.get_current_lambda()
r_t = acq_fn.get_variance_reduction_ratio()

print(f"Current interaction weight: {lambda_t:.3f}")
print(f"Variance reduction ratio: {r_t:.3f}")

# Get score components
total, info, cov = acq_fn(X_candidates, return_components=True)
print(f"Info scores: min={info.min():.3f}, max={info.max():.3f}")
print(f"Coverage scores: min={cov.min():.3f}, max={cov.max():.3f}")
```

## Tips and Best Practices

1. **Start with defaults**: The default parameters work well for most cases
2. **Adjust gamma**: Increase `gamma` (e.g., 0.5-0.7) for more exploration
3. **Interaction terms**: Only include if you expect significant interactions
4. **Sample size**: Works best with at least 20-30 initial samples
5. **Candidates**: Use 100-500 candidate points for good coverage
6. **Normalization**: Features should be roughly on the same scale

## Troubleshooting

### Common Issues

**Q: Acquisition scores are all very similar**
- Try increasing `gamma` for more exploration
- Check if you need more initial samples

**Q: Lambda stays at minimum/maximum**
- Adjust `tau_1` and `tau_2` thresholds
- Check variance reduction progress with `get_variance_reduction_ratio()`

**Q: Numerical instability**
- Reduce `prior_variance` or increase `noise_variance`
- Normalize input features

## Citation

If you use this acquisition function in your research, please cite:

```bibtex
@misc{dynamic_eur_acquisition,
  title={Dynamic EUR Acquisition Function for Active Learning},
  author={Your Name},
  year={2025},
  howpublished={GitHub}
}
```

## License

This module is provided as an extension to the AEPsych framework. See the main AEPsych license for details.

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.
