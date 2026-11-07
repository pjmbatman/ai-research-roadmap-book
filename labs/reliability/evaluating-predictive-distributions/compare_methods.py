"""21.2 실습: 네 가지 불확실성 방법을 NLL·CRPS·coverage·폭·조건별 coverage로 비교한다.

실행: uv run --project labs python labs/reliability/evaluating-predictive-distributions/compare_methods.py

데이터: x ~ U(0, 3), y = sin(2x) + ε, ε ~ N(0, (0.05 + 0.2 x)²). 학습 300개.
평가: 학습 범위 안(x ∈ [0, 3]) 20,000개와 학습 범위 밖(x ∈ [3, 3.5]) 20,000개.
방법(모두 평균은 5차 다항식):
  등분산       잔차 표준편차 하나
  이분산       log 잔차 제곱을 2차 다항식으로 회귀(21.1장의 방법)
  등분산 앙상블 부트스트랩 20개의 혼합
  이분산 앙상블 부트스트랩 20개의 혼합
혼합분포의 NLL은 밀도를, CRPS와 구간은 혼합에서 뽑은 표본 200개를 쓴다.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
sd_true = lambda x: 0.05 + 0.2 * x
f_true = lambda x: np.sin(2 * x)


def sample(n, lo, hi):
    x = rng.uniform(lo, hi, n)
    return x, f_true(x) + rng.normal(size=n) * sd_true(x)


def fit(x, y, hetero):
    c = np.polyfit(x, y, 5)
    r2 = (y - np.polyval(c, x)) ** 2
    if hetero:
        v = np.polyfit(x, np.log(r2) + 1.27, 2)
        return lambda t: (np.polyval(c, t), np.exp(0.5 * np.polyval(v, t)))
    s = np.sqrt(r2.mean())
    return lambda t: (np.polyval(c, t), np.full_like(t, s))


def members(x, y, hetero, M):
    if M == 1:
        return [fit(x, y, hetero)]
    return [fit(x[i], y[i], hetero) for i in rng.integers(0, len(x), size=(M, len(x)))]


def crps_samples(s, y):                          # s: (S, N), y: (N,)
    t1 = np.abs(s - y).mean(0)
    ss = np.sort(s, axis=0)                      # E|X - X'| = (2 / S²) Σ_i (2i - S - 1) x_(i)
    S = s.shape[0]
    w = (2 * np.arange(1, S + 1) - S - 1)[:, None]
    return t1 - (w * ss).sum(0) / S**2


def evaluate(ms, x, y, z90=norm.ppf(0.95)):
    mus, sds = map(np.array, zip(*[m(x) for m in ms]))           # (M, N) 둘 다
    dens = norm.pdf(y, mus, sds).mean(0)
    nll = -np.log(np.maximum(dens, 1e-300)).mean()
    k = rng.integers(0, len(ms), size=(200, len(x)))             # 혼합에서 표본 200개
    smp = mus[k, np.arange(len(x))] + sds[k, np.arange(len(x))] * rng.normal(size=k.shape)
    lo, hi = np.quantile(smp, [0.05, 0.95], axis=0)
    inside = (lo < y) & (y < hi)
    thirds = [inside[(x >= a) & (x < b)].mean() for a, b in ((0, 1), (1, 2), (2, 3))]
    return nll, crps_samples(smp, y).mean(), inside.mean(), (hi - lo).mean(), thirds


x_tr, y_tr = sample(300, 0, 3)
x_in, y_in = sample(20000, 0, 3)
x_out, y_out = sample(20000, 3, 3.5)
methods = {"등분산": (False, 1), "이분산": (True, 1), "등분산 앙상블": (False, 20), "이분산 앙상블": (True, 20)}

print("학습 범위 안 (x ∈ [0, 3])                             90% 포함률 (x 구간별)")
print(f"{'방법':12s}  NLL     CRPS   포함    폭     [0,1)  [1,2)  [2,3)")
fitted = {}
for name, (het, M) in methods.items():
    fitted[name] = members(x_tr, y_tr, het, M)
    nll, crps, cov, width, th = evaluate(fitted[name], x_in, y_in)
    print(f"{name:12s} {nll:6.3f}  {crps:.3f}  {cov:.3f}  {width:.3f}  " + "  ".join(f"{t:.3f}" for t in th))

print("\n학습 범위 밖 (x ∈ [3, 3.5])")
print(f"{'방법':12s}  NLL     CRPS   포함    폭")
for name in methods:
    nll, crps, cov, width, _ = evaluate(fitted[name], x_out, y_out)
    print(f"{name:12s} {nll:6.3f}  {crps:.3f}  {cov:.3f}  {width:.3f}")
