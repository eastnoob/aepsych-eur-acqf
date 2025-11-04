"""简单的导入和基本功能测试"""

import sys

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")

# 测试1：导入
print("\n" + "=" * 60)
print("测试1：导入模块")
print("=" * 60)

try:
    from eur_anova_pair import EURAnovaPairAcqf

    print("✅ 成功导入 EURAnovaPairAcqf")
    print(f"   类名: {EURAnovaPairAcqf.__name__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# 测试2：检查类属性
print("\n" + "=" * 60)
print("测试2：检查类文档")
print("=" * 60)

if EURAnovaPairAcqf.__doc__:
    doc_lines = EURAnovaPairAcqf.__doc__.split("\n")[:5]
    print("✅ 类文档存在:")
    for line in doc_lines:
        print(f"   {line}")
else:
    print("⚠️  类文档不存在")

# 测试3：检查__init__签名
print("\n" + "=" * 60)
print("测试3：检查__init__参数")
print("=" * 60)

import inspect

sig = inspect.signature(EURAnovaPairAcqf.__init__)
params = list(sig.parameters.keys())

print(f"✅ __init__ 参数（前10个）:")
for p in params[:10]:
    print(f"   - {p}")

# 检查关键新参数
if "total_budget" in params:
    print("✅ 新参数 'total_budget' 存在")
else:
    print("❌ 新参数 'total_budget' 不存在")

print("\n" + "=" * 60)
print("基本验证完成")
print("=" * 60)
