"""13.1 실습: 사전분포에 따라 사후분포가 어떻게 달라지는지 두 예제로 확인한다.

1. 동전 던지기: 참 θ* = 0.7. 위젯(bayesian-update)과 같은 세 사전분포
   균등 Beta(1, 1), 약한 믿음 Beta(4, 4), 강한 편향 Beta(16, 48)로 관측 수에 따른 사후를 비교한다.
2. 센서 보정: 온도 센서의 bias b [°C]를 기준 항온조에서 반복 측정해 추정한다.
   측정 y_i = b + ε_i, ε_i ~ N(0, σ²), σ = 0.3 °C (알려져 있다고 둔다), 참 bias b* = 0.4 °C.
   사전분포 세 가지: 균등(사실상 사전 지식 없음), 사양서 N(0, 0.25²), 과신한 사양서 N(0, 0.05²).
   사후분포는 격자에서 계산한다(닫힌 형태는 13.2장).

그림은 outputs/prior_sensitivity.png에 저장한다.

실행: uv run --project labs python labs/bayesian-inference/bayesian-thinking/prior_sensitivity.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).parent / "outputs"
THETA_TRUE = 0.7
COIN_PRIORS = {"균등 Beta(1,1)": (1, 1), "약한 믿음 Beta(4,4)": (4, 4), "강한 편향 Beta(16,48)": (16, 48)}
SIGMA_C, BIAS_TRUE_C = 0.3, 0.4
SENSOR_PRIORS = {"균등": None, "사양서 N(0,0.25²)": 0.25, "과신 N(0,0.05²)": 0.05}


def coin(rng):
    tosses = (rng.random(100) < THETA_TRUE).astype(int)          # (100,)
    print(f"[동전] 참 θ* = {THETA_TRUE}, 관측 수 n별 사후 평균과 95% 신용구간")
    print(f"{'n':>4} {'앞면':>4}   " + "   ".join(f"{name:>22s}" for name in COIN_PRIORS))
    rows = {}
    for n in (0, 5, 20, 100):
        h = int(tosses[:n].sum())
        cells = []
        for name, (a0, b0) in COIN_PRIORS.items():
            a, b = a0 + h, b0 + n - h
            lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
            cells.append(f"{a / (a + b):.3f} [{lo:.3f}, {hi:.3f}]")
            rows.setdefault(name, []).append((n, a, b))
        print(f"{n:4d} {h:4d}   " + "   ".join(f"{c:>22s}" for c in cells))
    return rows


def sensor(rng):
    y = BIAS_TRUE_C + SIGMA_C * rng.normal(size=30)              # 측정 30번, (30,)
    b = np.linspace(-1.5, 1.5, 30001)                            # bias 격자 [°C], (G,)
    db = b[1] - b[0]
    print(f"\n[센서 보정] 참 bias {BIAS_TRUE_C} °C, 잡음 σ = {SIGMA_C} °C")
    print(f"{'n':>3} {'MLE(표본평균)':>14}   " + "   ".join(f"{name:>24s}" for name in SENSOR_PRIORS))
    posts = {}
    for n in (1, 3, 10, 30):
        loglik = stats.norm.logpdf(y[:n, None], b, SIGMA_C).sum(axis=0)          # (G,)
        cells = []
        for name, tau in SENSOR_PRIORS.items():
            logprior = 0.0 if tau is None else stats.norm.logpdf(b, 0.0, tau)
            lp = loglik + logprior
            p = np.exp(lp - lp.max())
            p /= p.sum() * db                                                     # p(b | y), (G,)
            cdf = np.cumsum(p) * db
            mean = np.sum(b * p) * db
            lo, hi = b[np.searchsorted(cdf, [0.025, 0.975])]
            cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            posts[(name, n)] = p
        print(f"{n:3d} {y[:n].mean():+14.3f}   " + "   ".join(f"{c:>24s}" for c in cells))
    return b, posts


def main():
    rng = np.random.default_rng(0)
    rows = coin(rng)
    b, posts = sensor(rng)

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    t = np.linspace(0, 1, 501)
    for j, n in enumerate((0, 5, 20, 100)):
        ax = axes[0, j]
        for name, r in rows.items():
            _, a, bb = next(x for x in r if x[0] == n)
            ax.plot(t, stats.beta.pdf(t, a, bb), label=name)
        ax.axvline(THETA_TRUE, color="k", lw=0.8, ls=":")
        ax.set_title(f"coin, n = {n}")
        ax.set_xlabel("theta")
    for j, n in enumerate((1, 3, 10, 30)):
        ax = axes[1, j]
        for name in SENSOR_PRIORS:
            ax.plot(b, posts[(name, n)], label=name)
        ax.axvline(BIAS_TRUE_C, color="k", lw=0.8, ls=":")
        ax.set_xlim(-0.6, 1.2)
        ax.set_title(f"sensor bias, n = {n}")
        ax.set_xlabel("bias [degC]")
    fig.tight_layout()
    fig.savefig(OUT / "prior_sensitivity.png", dpi=120)
    print(f"\n그림 저장: {OUT / 'prior_sensitivity.png'}")


if __name__ == "__main__":
    main()
