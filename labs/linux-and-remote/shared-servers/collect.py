"""2.5 실습: array job의 결과(runs/seed*/result.json)를 모아 요약한다. 빠진 seed가 있으면 알린다.

실행: python3 collect.py runs
"""
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED = [0, 1, 2, 3, 4]

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
results = {}
for seed in EXPECTED:
    f = root / f"seed{seed}" / "result.json"
    if f.exists():
        results[seed] = json.loads(f.read_text())["mse"]

missing = [s for s in EXPECTED if s not in results]
for seed, mse in sorted(results.items()):
    print(f"seed {seed}: mse = {mse:.4f}")
if missing:
    print(f"빠진 seed: {missing} — 해당 task의 로그와 sacct 상태를 확인한다")
    sys.exit(1)
mse = np.array(list(results.values()))
print(f"mse 평균 {mse.mean():.4f} ± 표준편차 {mse.std(ddof=1):.4f} (n = {len(mse)})")
