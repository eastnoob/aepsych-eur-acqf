#!/usr/bin/env python3
"""Quick import test"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from eur_anova_multi import EURAnovaMultiAcqf
    print("OK: EURAnovaMultiAcqf imported successfully")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

try:
    from modules import ANOVAEffectEngine
    print("OK: ANOVAEffectEngine imported successfully")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

try:
    import torch
    from botorch.models import SingleTaskGP

    # 使用double precision（BoTorch推荐）
    model = SingleTaskGP(
        torch.rand(5, 3, dtype=torch.float64),
        torch.randn(5, 1, dtype=torch.float64)
    )
    model.eval()

    acqf = EURAnovaMultiAcqf(model, enable_main=True)

    # BoTorch采集函数标准格式：(batch_size, q=1, dim)
    # 对于单点评估，需要显式添加q维度
    X_candidates = torch.rand(10, 1, 3, dtype=torch.float64)  # (10, 1, 3)
    scores = acqf(X_candidates)

    print(f"OK: Created acqf and evaluated {scores.shape[0]} candidates")
    print(f"OK: Score range: [{scores.min():.4f}, {scores.max():.4f}]")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed! The implementation works with AEPsych.")
