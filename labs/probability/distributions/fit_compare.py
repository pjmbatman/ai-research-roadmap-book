"""11.2 실습: 고장 간격과 측정 잡음에 여러 분포를 적합하고 적합도를 비교한다.

데이터는 생성 과정을 알고 있는 합성 데이터다(참 분포를 알아야 비교 결과를 채점할 수 있다).
실행: uv run --project labs python labs/probability/distributions/fit_compare.py
그림: labs/probability/distributions/outputs/fit_compare.png (QQ plot)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)


class Mixture2:
    """평균 0을 공유하지 않는 일반 2성분 가우시안 혼합(EM 적합). scipy 분포처럼 logpdf·cdf·sf·ppf를 제공한다."""

    def __init__(self, x):
        gm = GaussianMixture(2, random_state=0).fit(x[:, None])
        self.w, self.m = gm.weights_, gm.means_[:, 0]
        self.s = np.sqrt(gm.covariances_[:, 0, 0])

    def logpdf(self, x):
        return np.log(sum(w * stats.norm.pdf(x, m, s) for w, m, s in zip(self.w, self.m, self.s)))

    def cdf(self, x):
        return sum(w * stats.norm.cdf(x, m, s) for w, m, s in zip(self.w, self.m, self.s))

    def sf(self, x):
        return 1 - self.cdf(x)

    def ppf(self, p):
        return np.array([optimize.brentq(lambda v: self.cdf(v) - q, -50, 50) for q in np.atleast_1d(p)])


def report(title, x, cands, tail):
    """후보 분포마다 최대우도 적합 -> logL, AIC, KS 통계량, 꼬리 확률을 출력한다."""
    lo, hi = tail
    print(f"\n[{title}]  n = {x.size}")
    print(f"{'model':<10}{'k':>3}{'logL':>10}{'AIC':>10}{'KS D':>8}{f'P(X<{lo})':>12}{f'P(X>{hi})':>12}")
    fits = {}
    for name, cand in cands.items():
        if isinstance(cand, tuple):
            dist, kw = cand
            params = dist.fit(x, **kw)
            d, k = dist(*params), len(params) - len(kw)   # k = 고정하지 않은 모수의 수
        else:
            d, k = cand(x), 5                             # 혼합: 가중치 1 + 평균 2 + 분산 2
        ll = d.logpdf(x).sum()
        D = stats.kstest(x, d.cdf).statistic
        print(f"{name:<10}{k:3d}{ll:10.1f}{2 * k - 2 * ll:10.1f}{D:8.3f}{d.cdf(lo):12.4f}{d.sf(hi):12.4f}")
        fits[name] = d
    print(f"{'sample':<10}{'':3}{'':>10}{'':>10}{'':>8}{np.mean(x < lo):12.4f}{np.mean(x > hi):12.4f}")
    return fits


# 1) 고장 간격(시간): 마모가 쌓여 고장 나는 부품 -> 참 분포 Gamma(shape 2, scale 500 h)
life_h = rng.gamma(2.0, 500.0, size=300)
fits_life = report("failure intervals, true Gamma(2, scale 500 h)", life_h, {
    "expon":   (stats.expon, {"floc": 0}),
    "gamma":   (stats.gamma, {"floc": 0}),
    "lognorm": (stats.lognorm, {"floc": 0}),
    "Weibull": (stats.weibull_min, {"floc": 0}),
}, tail=(100, 3000))

# 2) 측정 잡음: 95%는 N(0, 0.1^2), 5%는 N(0, 1^2)인 이상치가 섞인 잡음
n = 2000
noise = np.where(rng.random(n) < 0.95, rng.normal(0, 0.1, n), rng.normal(0, 1.0, n))
fits_noise = report("measurement noise, true 0.95 N(0, 0.1^2) + 0.05 N(0, 1^2)", noise, {
    "normal":    (stats.norm, {}),
    "Laplace":   (stats.laplace, {}),
    "Student-t": (stats.t, {}),
    "mixture2":  Mixture2,
}, tail=(-0.5, 0.5))

out = Path(__file__).parent / "outputs"
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, x, fits in ((axes[0], life_h, fits_life), (axes[1], noise, fits_noise)):
    p = (np.arange(1, x.size + 1) - 0.5) / x.size
    xs = np.sort(x)
    for name, d in fits.items():
        ax.plot(d.ppf(p), xs, ".", ms=3, label=name)
    ax.plot([xs[0], xs[-1]], [xs[0], xs[-1]], "k-", lw=0.8)
    ax.set(xlabel="fitted quantile", ylabel="sample quantile")
    ax.legend(fontsize=8)
axes[0].set_title("failure intervals (h)")
axes[1].set_title("measurement noise")
fig.tight_layout()
fig.savefig(out / "fit_compare.png", dpi=120)
