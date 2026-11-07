"""10.7 실습: 비싼 시뮬레이터를 베이지안 최적화와 random search로 최적화해 평가 횟수 대비 성능을 비교한다.

시뮬레이터: 연속 반응 A -> B -> C에서 중간 생성물 B의 수율을 반응 온도 T [K]와 반응 시간 t [h]의 함수로 계산한다.
  k_i(T) = A_i exp(-Ea_i / (R T))           (Arrhenius 식)
  y_B(T, t) = k1 / (k2 - k1) (exp(-k1 t) - exp(-k2 t))
측정에는 표준편차 0.01의 잡음이 더해진다. 탐색 범위는 T in [300, 400] K, t in [0.1, 10] h.

베이지안 최적화: scikit-learn GaussianProcessRegressor(Matern 5/2 + 잡음 커널, 입력은 [0, 1]^2로 변환),
획득함수 expected improvement(EI), 후보 점 2000개 중 EI 최대인 점을 고른다. 처음 5점은 무작위.

실행: uv run --project labs python labs/optimization/special-optimization/bo_vs_random.py
"""
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

OUT = Path(__file__).parent / "outputs"
R_J_PER_MOL_K = 8.314
LO, HI = np.array([300.0, 0.1]), np.array([400.0, 10.0])      # (T [K], t [h])의 하한·상한
NOISE = 0.01


def yield_b(temp_k, time_h):
    k1 = 4.0e6 * np.exp(-50e3 / (R_J_PER_MOL_K * temp_k))      # [1/h]
    k2 = 2.0e9 * np.exp(-75e3 / (R_J_PER_MOL_K * temp_k))      # [1/h]
    return k1 / (k2 - k1) * (np.exp(-k1 * time_h) - np.exp(-k2 * time_h))


def to_phys(u):                                                # [0, 1]^2 -> (T, t)
    return LO + u * (HI - LO)


def f_true(u):
    p = to_phys(np.atleast_2d(u))
    return yield_b(p[:, 0], p[:, 1])


GRID = np.stack(np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401)), -1).reshape(-1, 2)
F_GRID = f_true(GRID)
F_STAR = F_GRID.max()
U_STAR = GRID[np.argmax(F_GRID)]


def measure(u, rng):
    return f_true(u) + NOISE * rng.normal(size=len(np.atleast_2d(u)))


def expected_improvement(mu, sd, best, xi=0.01):              # 최대화 문제의 EI
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def run(method, seed, budget=30, n_init=5):
    rng = np.random.default_rng(seed)
    U = rng.random((n_init, 2))
    y = measure(U, rng)
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1))
    while len(U) < budget:
        if method == "BO":
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(U, y)
            cand = rng.random((2000, 2))
            mu, sd = gp.predict(cand, return_std=True)
            u_next = cand[np.argmax(expected_improvement(mu, sd, y.max()))]
        else:
            u_next = rng.random(2)
        U = np.vstack([U, u_next])
        y = np.append(y, measure(u_next, rng))
    # 평가 n번까지 관측값이 가장 좋은 점을 추천하고, 그 점의 참 수율로 regret을 잰다
    rec = np.array([np.argmax(y[:n]) for n in range(1, budget + 1)])
    return F_STAR - f_true(U[rec]), U[rec[-1]]


def main():
    print(f"참 최적: T = {to_phys(U_STAR)[0]:.1f} K, t = {to_phys(U_STAR)[1]:.2f} h, B 수율 {F_STAR:.4f}")
    print(f"잡음 표준편차 {NOISE}, 시드 20개, 처음 5점 무작위\n")
    checkpoints = [5, 10, 15, 20, 30]
    curves = {}
    print("방법     regret 중앙값 [25%, 75%] (평가 횟수별)")
    print("         " + "".join(f"{n:>25d}회" for n in checkpoints))
    for method in ["BO", "random"]:
        R, recs = [], []
        for s in range(20):
            r, u = run(method, s)
            R.append(r)
            recs.append(to_phys(u))
        R = np.array(R)                                        # (20, budget)
        curves[method] = R
        cells = [f"{np.median(R[:, n - 1]):.4f} [{np.quantile(R[:, n - 1], 0.25):.4f}, {np.quantile(R[:, n - 1], 0.75):.4f}]"
                 for n in checkpoints]
        print(f"{method:8s} " + "".join(f"{c:>26s}" for c in cells))
        recs = np.array(recs)
        print(f"         30회 뒤 추천 T {np.median(recs[:, 0]):.1f} K (사분위 범위 {np.quantile(recs[:, 0], 0.25):.1f}–"
              f"{np.quantile(recs[:, 0], 0.75):.1f}), t {np.median(recs[:, 1]):.2f} h")
    for target in [0.01, 0.005]:
        line = []
        for method, R in curves.items():
            hit = [int(np.argmax(r < target)) + 1 if (r < target).any() else None for r in R]
            ok = [h for h in hit if h is not None]
            line.append(f"{method} {len(ok):2d}/20 (도달한 시드의 중앙값 {np.median(ok):.0f}회)" if ok else f"{method} 0/20")
        print(f"regret < {target}: " + ",  ".join(line))
    plot(curves)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    n = np.arange(1, curves["BO"].shape[1] + 1)
    for method, R in curves.items():
        ax.semilogy(n, np.median(R, 0) + 1e-6, label=f"{method} (median)")
        ax.fill_between(n, np.quantile(R, 0.25, 0) + 1e-6, np.quantile(R, 0.75, 0) + 1e-6, alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("regret of recommended point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "regret.png", dpi=120)


if __name__ == "__main__":
    main()
