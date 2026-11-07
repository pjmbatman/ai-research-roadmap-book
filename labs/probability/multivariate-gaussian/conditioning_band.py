"""11.4 실습: 2차원 가우시안의 조건부분포를 공식과 샘플링(띠 안의 표본)으로 구해 비교한다.

띠 폭 h를 바꾸며 샘플링 추정이 공식에서 어떻게 벗어나는지 본다.
실행: uv run --project labs python labs/probability/multivariate-gaussian/conditioning_band.py
그림: labs/probability/multivariate-gaussian/outputs/conditioning_band.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S1, S2, RHO = 1.0, 1.5, 0.8
Sigma = np.array([[S1**2, RHO * S1 * S2], [RHO * S1 * S2, S2**2]])   # (2, 2), 평균 0
N = 200_000
rng = np.random.default_rng(0)
X = rng.standard_normal((N, 2)) @ np.linalg.cholesky(Sigma).T        # (N, 2)

slope = Sigma[1, 0] / Sigma[0, 0]
var_c = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]
print(f"N = {N:,}, seed 0, sigma1 = {S1}, sigma2 = {S2}, rho = {RHO}")
print(f"formula: E[x2|x1] = {slope:.2f} x1,  Var[x2|x1] = {var_c:.3f} (x1과 무관)")
print(f"{'x1':>5} {'h':>5} {'n':>7} {'mean':>8} {'(formula)':>10} {'var':>7} {'var+slope^2 h^2/3':>18}")
for x1 in (0.0, 1.2, 2.5):
    for h in (0.02, 0.1, 0.5):
        sel = np.abs(X[:, 0] - x1) < h
        y = X[sel, 1]
        print(f"{x1:5.1f} {h:5.2f} {sel.sum():7d} {y.mean():8.3f} {slope * x1:10.3f} {y.var():7.3f} "
              f"{var_c + slope**2 * h**2 / 3:18.3f}")

out = Path(__file__).parent / "outputs"
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(X[:5000, 0], X[:5000, 1], ".", ms=1, alpha=0.4)
g = np.linspace(-3.5, 3.5, 50)
ax.plot(g, slope * g, "k-", label="E[x2 | x1]")
ax.plot(g, slope * g + 2 * np.sqrt(var_c), "k--", lw=0.8, label="+-2 sd")
ax.plot(g, slope * g - 2 * np.sqrt(var_c), "k--", lw=0.8)
ax.set(xlabel="x1", ylabel="x2", xlim=(-3.5, 3.5))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "conditioning_band.png", dpi=120)
