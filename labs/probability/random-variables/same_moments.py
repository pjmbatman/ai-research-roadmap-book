"""11.1 실습: 평균 0, 분산 1로 맞춘 다섯 분포의 모양 차이.

실행: uv run --project labs python labs/probability/random-variables/same_moments.py
그림: labs/probability/random-variables/outputs/same_moments.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 1_000_000
rng = np.random.default_rng(0)
s = np.sqrt(1 - 0.9**2)                                    # 쌍봉 성분의 표준편차
samples = {                                                # 모두 (N,), 평균 0, 분산 1
    "normal": rng.normal(0.0, 1.0, N),
    "uniform": rng.uniform(-np.sqrt(3), np.sqrt(3), N),
    "Laplace": rng.laplace(0.0, 1 / np.sqrt(2), N),
    "exp - 1": rng.exponential(1.0, N) - 1.0,
    "bimodal": np.where(rng.random(N) < 0.5, -0.9, 0.9) + s * rng.normal(size=N),
}

print(f"samples {N:,}, seed 0")
print(f"{'dist':<10}{'mean':>7}{'var':>7}{'median':>8}{'P(|X|<0.25)':>13}{'P(X>2)':>9}{'P(|X|>3)':>10}")
for name, x in samples.items():
    print(f"{name:<10}{x.mean():7.3f}{x.var():7.3f}{np.median(x):8.3f}"
          f"{np.mean(np.abs(x) < 0.25):13.3f}{np.mean(x > 2):9.4f}{np.mean(np.abs(x) > 3):10.4f}")

out = Path(__file__).parent / "outputs"
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
grid = np.linspace(-4, 4, 161)
for name, x in samples.items():
    axes[0].hist(x, bins=grid, density=True, histtype="step", label=name)
    axes[1].plot(np.sort(x)[::1000], np.linspace(0, 1, N)[::1000], label=name)
axes[0].set(title="density (histogram)", xlabel="x")
axes[1].set(title="CDF", xlabel="x", xlim=(-4, 4))
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "same_moments.png", dpi=120)
