import os
import shutil

os.chdir("f:\\Github\\aepsych-source\\extensions\\dynamic_eur_acquisition")

# 源代码移到 src/
for f in [
    "eur_anova_pair.py",
    "eur_anova_multi.py",
    "gower_distance.py",
    "gp_variance.py",
    "multiscale_learned_implementation.py",
]:
    if os.path.exists(f):
        shutil.move(f, f"src/{f}")
        print(f"✓ {f} -> src/")

# 测试移到 test/
for f in ["test_aepsych_integration.py", "test_import.py", "test_multi_order.py"]:
    if os.path.exists(f):
        shutil.move(f, f"test/{f}")
        print(f"✓ {f} -> test/")

# 文档移到 docs/
docs = [
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
]
for f in docs:
    if os.path.exists(f):
        shutil.move(f, f"docs/{f}")
        print(f"✓ {f} -> docs/")

# 配置移到 configs/
if os.path.exists("recommended_config.ini"):
    shutil.move("recommended_config.ini", "configs/recommended_config.ini")
    print(f"✓ recommended_config.ini -> configs/")

# 备份移到 backups/
for f in ["eur_anova_pair.py.bak", "eur_anova_multi.py.bak", "gower_distance.py.bak"]:
    if os.path.exists(f):
        shutil.move(f, f"backups/{f}")
        print(f"✓ {f} -> backups/")

# 示例移到 examples/
for f in ["demo_comparison.py", "single_vs_multiscale.png"]:
    if os.path.exists(f):
        shutil.move(f, f"examples/{f}")
        print(f"✓ {f} -> examples/")

# 删除重复文件夹
for folder in ["tests", "config"]:
    if os.path.exists(folder) and not os.listdir(folder):
        shutil.rmtree(folder)
        print(f"✓ 删除空文件夹: {folder}/")

print("\n✅ 文件整理完成！")
