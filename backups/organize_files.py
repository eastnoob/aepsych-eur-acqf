#!/usr/bin/env python3
"""Organize files into proper directories"""
import os
import shutil
from pathlib import Path
from loguru import logger

base_dir = Path(".")

# File organization mapping: (source_files, destination_folder)
moves = [
    # Source code -> src/
    (
        [
            "eur_anova_pair.py",
            "eur_anova_multi.py",
            "gower_distance.py",
            "gp_variance.py",
            "multiscale_learned_implementation.py",
        ],
        "src",
    ),
    # Tests -> test/
    (["test_aepsych_integration.py", "test_import.py", "test_multi_order.py"], "test"),
    # Docs -> docs/
    (
        [
            "FINAL_SUMMARY.md",
            "QUICK_REFERENCE.md",
            "MULTISCALE_LEARNED_PERTURBATION.md",
            "INTUITIVE_EXPLANATION.md",
            "COMPLETION_REPORT_MULTISCALE_LEARNED.md",
            "AEPSYCH_INTEGRATION.md",
            "CODE_REVIEW_FIXES.md",
            "EXPERIMENT_DESIGN.md",
            "REFACTORING_SUMMARY.md",
            "HANDOVER_HYBRID_PERTURBATION.md",
            "CHANGELOG.md",
            "CONTRIBUTORS.md",
            "USAGE_MULTI_ORDER.md",
        ],
        "docs",
    ),
    # Config -> configs/
    (["recommended_config.ini"], "configs"),
    # Backups -> backups/
    (
        ["eur_anova_pair.py.bak", "eur_anova_multi.py.bak", "gower_distance.py.bak"],
        "backups",
    ),
    # Examples -> examples/
    (["demo_comparison.py", "single_vs_multiscale.png"], "examples"),
]

logger.info("开始整理文件...\n")

for files, dest_folder in moves:
    dest_path = base_dir / dest_folder
    dest_path.mkdir(exist_ok=True)

    for file in files:
        src = base_dir / file
        if src.exists():
            dst = dest_path / file
            shutil.move(str(src), str(dst))
            logger.info(f"✓ 移动: {file} -> {dest_folder}/")
        else:
            logger.info(f"⊘ 跳过: {file} (不存在)")

# Remove duplicate/unnecessary folders
logger.info("\n清理重复文件夹...\n")

# Remove empty tests/ if it exists and test/ has content
if (base_dir / "tests").exists() and (base_dir / "test").exists():
    shutil.rmtree(base_dir / "tests")
    logger.info("✓ 删除: tests/ (已合并到 test/)")

# Remove empty config/ if configs/ exists
if (base_dir / "config").exists() and (base_dir / "configs").exists():
    shutil.rmtree(base_dir / "config")
    logger.info("✓ 删除: config/ (已合并到 configs/)")

# Remove unused tests/ and src/ created earlier if empty
temp_dirs = ["tests", "config"]
for temp_dir in temp_dirs:
    path = base_dir / temp_dir
    if path.exists() and not any(path.iterdir()):
        shutil.rmtree(path)
        logger.info(f"✓ 删除: {temp_dir}/ (空文件夹)")

logger.info("\n✅ 文件整理完成！")
logger.info("\n最终结构:")
logger.info("  src/              - 源代码")
logger.info("  test/             - 单元测试")
logger.info("  docs/             - 文档")
logger.info("  configs/          - 配置文件")
logger.info("  examples/         - 示例和演示")
logger.info("  backups/          - 备份文件")
logger.info("  archive/          - 归档文件")
logger.info("  modules/          - 其他模块")
logger.info("  Introduction/     - 介绍文档")
logger.info("  exp_design/       - 实验设计")
