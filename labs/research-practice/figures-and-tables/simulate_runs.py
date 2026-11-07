"""23.9 실습용 결과 파일(results/runs.csv)을 만든다. 실제 연구에서는 실험 스크립트가 이 파일을 쓴다.

실행: uv run --project labs python labs/research-practice/figures-and-tables/simulate_runs.py
"""
import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "runs.csv"
rng = np.random.default_rng(0)
true_rmse_k = {"A": {0: 0.62, 25: 0.41, 45: 0.47}, "B": {0: 0.66, 25: 0.37, 45: 0.43}}

rows = []
for seed in range(5):
    seed_fx = rng.normal(0, 0.03)                       # seed마다 두 방법 공통 효과
    for method, by_temp in true_rmse_k.items():
        for temp_c, mu in by_temp.items():
            rmse_k = mu + seed_fx + rng.normal(0, 0.01)
            rows.append({"method": method, "seed": seed, "temp_c": temp_c, "rmse_k": f"{rmse_k:.6f}"})

OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "seed", "temp_c", "rmse_k"])
    w.writeheader()
    w.writerows(rows)
print(f"{OUT.name}: {len(rows)}행")
