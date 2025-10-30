"""
Complete End-to-End Active Learning Experiment

This script demonstrates a full active learning experiment using:
1. Standard AEPsych configuration format
2. VarianceReductionWithCoverageAcqf acquisition function
3. Complete workflow: initialization -> model-based acquisition -> evaluation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import acquisition function
try:
    from acquisition_function import VarianceReductionWithCoverageAcqf
except ImportError:
    from .acquisition_function import VarianceReductionWithCoverageAcqf

try:
    from gp_variance import GPVarianceCalculator
except ImportError:
    from .gp_variance import GPVarianceCalculator


def true_function(X: np.ndarray) -> np.ndarray:
    """
    Ground truth function for testing.
    
    This function has:
    - Main effects from all three features
    - Interaction effects between x1-x2 and x2-x3
    - Some noise
    
    y = 2*x1 + 3*x2 + 1.5*x3 + 1.5*x1*x2 + 2*x2*x3 + noise
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    
    # Main effects
    y = 2.0 * x1 + 3.0 * x2 + 1.5 * x3
    
    # Interaction effects
    y += 1.5 * x1 * x2  # x1-x2 interaction
    y += 2.0 * x2 * x3  # x2-x3 interaction
    
    # Add noise
    noise = np.random.normal(0, 0.3, size=X.shape[0])
    y += noise
    
    return y


def sobol_sample(n_samples: int, n_dims: int, bounds: np.ndarray) -> np.ndarray:
    """
    Generate quasi-random Sobol samples.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_dims : int
        Number of dimensions
    bounds : np.ndarray
        Bounds for each dimension, shape (n_dims, 2)
    
    Returns
    -------
    np.ndarray
        Sobol samples, shape (n_samples, n_dims)
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dims, scramble=True)
        samples = sampler.random(n_samples)
    except ImportError:
        # Fallback to random sampling
        samples = np.random.rand(n_samples, n_dims)
    
    # Scale to bounds
    for i in range(n_dims):
        samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])
    
    return samples


def run_complete_experiment(
    n_init: int = 15,
    n_opt: int = 30,
    n_candidates: int = 200,
    config_path: str = "full_experiment_config.ini",
    seed: int = 42
):
    """
    Run a complete active learning experiment.
    
    Workflow:
    1. Initialization: Collect n_init samples using Sobol sampling
    2. Optimization: Iteratively select n_opt samples using acquisition function
    3. Evaluation: Assess final model performance on test set
    
    Parameters
    ----------
    n_init : int
        Number of initial random samples
    n_opt : int
        Number of optimization iterations
    n_candidates : int
        Number of candidate points per iteration
    config_path : str
        Path to configuration file
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    print("=" * 80)
    print("完整端到端主动学习实�?)
    print("=" * 80)
    
    # Parameter bounds
    n_dims = 3
    bounds = np.array([[0.0, 1.0]] * n_dims)
    
    # Create acquisition function from config
    print(f"\n1. 从配置文件初始化: {config_path}")
    config_path_full = Path(__file__).parent / config_path
    if not config_path_full.exists():
        print(f"   警告: 配置文件不存�?使用默认参数")
        config_path_full = None
    
    acq_fn = VarianceReductionWithCoverageAcqf(
        config_ini_path=config_path_full,
        interaction_terms=[(0, 1), (1, 2)],  # x1-x2, x2-x3 interactions
    )
    print(f"   - lambda_min={acq_fn.lambda_min}, lambda_max={acq_fn.lambda_max}")
    print(f"   - tau_1={acq_fn.tau_1}, tau_2={acq_fn.tau_2}")
    print(f"   - gamma={acq_fn.gamma}")
    print(f"   - 交互�? {acq_fn.interaction_terms}")
    
    # Phase 1: Initialization with Sobol sampling
    print(f"\n2. 初始化阶�? 使用Sobol采样收集 {n_init} 个样�?)
    X_train = sobol_sample(n_init, n_dims, bounds)
    y_train = true_function(X_train)
    print(f"   �?初始数据�? {X_train.shape[0]} 样本")
    
    # Fit initial model
    acq_fn.fit(X_train, y_train)
    print(f"   �?初始模型拟合完成")
    print(f"   - λ_t = {acq_fn.get_current_lambda():.3f}")
    print(f"   - r_t = {acq_fn.get_variance_reduction_ratio():.3f}")
    
    # Phase 2: Optimization with acquisition function
    print(f"\n3. 优化阶段: 使用采集函数迭代 {n_opt} �?)
    
    # Track progress
    history = {
        'n_samples': [len(X_train)],
        'lambda_t': [acq_fn.get_current_lambda()],
        'r_t': [acq_fn.get_variance_reduction_ratio()],
        'best_acq_score': [],
        'mean_acq_score': [],
        'main_variances': [acq_fn.gp_calculator.get_parameter_variance()[1:n_dims+1].copy()],
        'inter_variances': []
    }
    
    for iteration in range(n_opt):
        # Generate candidate points
        X_candidates = np.random.rand(n_candidates, n_dims)
        for i in range(n_dims):
            X_candidates[:, i] = bounds[i, 0] + X_candidates[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Evaluate acquisition function
        acq_scores, info_scores, cov_scores = acq_fn(X_candidates, return_components=True)
        
        # Select best point
        best_idx = np.argmax(acq_scores)
        x_next = X_candidates[best_idx:best_idx+1]
        y_next = true_function(x_next)
        
        # Add to training set
        X_train = np.vstack([X_train, x_next])
        y_train = np.hstack([y_train, y_next])
        
        # Refit model
        acq_fn.fit(X_train, y_train)
        
        # Track metrics
        lambda_t = acq_fn.get_current_lambda()
        r_t = acq_fn.get_variance_reduction_ratio()
        
        history['n_samples'].append(len(X_train))
        history['lambda_t'].append(lambda_t)
        history['r_t'].append(r_t)
        history['best_acq_score'].append(np.max(acq_scores))
        history['mean_acq_score'].append(np.mean(acq_scores))
        history['main_variances'].append(acq_fn.gp_calculator.get_parameter_variance()[1:n_dims+1].copy())
        
        # Print progress
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"   迭代 {iteration+1:3d}: "
                  f"样本={len(X_train):3d}, "
                  f"λ_t={lambda_t:.3f}, "
                  f"r_t={r_t:.3f}, "
                  f"最佳得�?{np.max(acq_scores):.4f}, "
                  f"信息={info_scores[best_idx]:.4f}, "
                  f"覆盖={cov_scores[best_idx]:.4f}")
    
    print(f"   �?优化完成! 最终数据集: {len(X_train)} 样本")
    
    # Phase 3: Final evaluation
    print(f"\n4. 最终评�?)
    
    # Generate test set
    n_test = 500
    X_test = np.random.rand(n_test, n_dims)
    for i in range(n_dims):
        X_test[:, i] = bounds[i, 0] + X_test[:, i] * (bounds[i, 1] - bounds[i, 0])
    y_test = true_function(X_test)
    
    # Predict on test set using GP
    gp = acq_fn.gp_calculator
    y_pred = gp.predict(X_test)
    
    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"   测试集性能 (n={n_test}):")
    print(f"   - MSE  = {mse:.6f}")
    print(f"   - MAE  = {mae:.6f}")
    print(f"   - R²   = {r2:.6f}")
    
    # Print parameter variances
    param_var = gp.get_parameter_variance()
    print(f"\n   最终参数方�?")
    print(f"   - 截距: {param_var[0]:.6f}")
    for i in range(n_dims):
        print(f"   - x{i+1} (主效�?: {param_var[i+1]:.6f}")
    
    if len(acq_fn.interaction_terms) > 0:
        print(f"   交互效应:")
        inter_offset = 1 + n_dims
        for idx, (i, j) in enumerate(acq_fn.interaction_terms):
            print(f"   - x{i+1} × x{j+1}: {param_var[inter_offset + idx]:.6f}")
    
    # Visualization
    print(f"\n5. 生成可视�?)
    visualize_results(history, X_train, y_train, X_test, y_test, y_pred, acq_fn)
    
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)
    
    return X_train, y_train, history, acq_fn


def visualize_results(history, X_train, y_train, X_test, y_test, y_pred, acq_fn):
    """Create comprehensive visualization of experiment results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Lambda and r_t over time
    ax1 = plt.subplot(3, 3, 1)
    iterations = np.arange(len(history['lambda_t']))
    ax1.plot(iterations, history['lambda_t'], 'b-', linewidth=2, label='λ_t')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('λ_t (交互权重)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(iterations, history['r_t'], 'r-', linewidth=2, label='r_t')
    ax1_twin.set_ylabel('r_t (相对方差)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.axhline(y=acq_fn.tau_1, color='r', linestyle='--', alpha=0.5, label=f'τ�?{acq_fn.tau_1}')
    ax1_twin.axhline(y=acq_fn.tau_2, color='r', linestyle='--', alpha=0.5, label=f'τ�?{acq_fn.tau_2}')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('动态权重调�?)
    
    # 2. Acquisition scores
    ax2 = plt.subplot(3, 3, 2)
    if len(history['best_acq_score']) > 0:
        opt_iters = np.arange(1, len(history['best_acq_score']) + 1)
        ax2.plot(opt_iters, history['best_acq_score'], 'g-', linewidth=2, label='最佳分�?)
        ax2.plot(opt_iters, history['mean_acq_score'], 'b--', linewidth=2, label='平均分数')
        ax2.set_xlabel('优化迭代')
        ax2.set_ylabel('采集函数分数')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('采集函数分数演化')
    
    # 3. Main effect variances
    ax3 = plt.subplot(3, 3, 3)
    main_vars = np.array(history['main_variances'])
    for i in range(main_vars.shape[1]):
        ax3.plot(iterations, main_vars[:, i], linewidth=2, label=f'x{i+1}')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('参数方差')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('主效应方差减�?)
    
    # 4. Training data distribution (x1 vs x2)
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='k')
    ax4.set_xlabel('x�?)
    ax4.set_ylabel('x�?)
    ax4.set_title('训练数据分布 (x�?x�?')
    plt.colorbar(scatter, ax=ax4, label='y')
    ax4.grid(True, alpha=0.3)
    
    # 5. Training data distribution (x2 vs x3)
    ax5 = plt.subplot(3, 3, 5)
    scatter = ax5.scatter(X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis',
                         s=50, alpha=0.6, edgecolors='k')
    ax5.set_xlabel('x�?)
    ax5.set_ylabel('x�?)
    ax5.set_title('训练数据分布 (x�?x�?')
    plt.colorbar(scatter, ax=ax5, label='y')
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction vs True
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(y_test, y_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax6.set_xlabel('真实�?)
    ax6.set_ylabel('预测�?)
    ax6.set_title('预测准确�?)
    ax6.grid(True, alpha=0.3)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    ax6.text(0.05, 0.95, f'MSE={mse:.4f}\nR²={r2:.4f}', 
             transform=ax6.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Residuals
    ax7 = plt.subplot(3, 3, 7)
    residuals = y_test - y_pred
    ax7.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax7.set_xlabel('预测�?)
    ax7.set_ylabel('残差')
    ax7.set_title('残差分析')
    ax7.grid(True, alpha=0.3)
    
    # 8. Sample size growth
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(iterations, history['n_samples'], 'b-', linewidth=2)
    ax8.set_xlabel('迭代次数')
    ax8.set_ylabel('样本�?)
    ax8.grid(True, alpha=0.3)
    ax8.set_title('数据集增�?)
    
    # 9. Error histogram
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('预测误差')
    ax9.set_ylabel('频数')
    ax9.set_title('误差分布')
    ax9.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / 'end_to_end_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   �?可视化已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Run complete experiment
    X_train, y_train, history, acq_fn = run_complete_experiment(
        n_init=15,      # Initial samples
        n_opt=30,       # Optimization iterations
        n_candidates=200,  # Candidates per iteration
        seed=42
    )
    
    # Save results
    output_path = Path(__file__).parent / 'end_to_end_results.npz'
    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        lambda_t=np.array(history['lambda_t']),
        r_t=np.array(history['r_t']),
        n_samples=np.array(history['n_samples']),
        best_acq_score=np.array(history['best_acq_score']),
        mean_acq_score=np.array(history['mean_acq_score']),
    )
    print(f"\n结果已保�? {output_path}")
