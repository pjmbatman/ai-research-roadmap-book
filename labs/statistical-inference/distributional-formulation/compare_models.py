"""12.3 실습: 같은 데이터에 세 가지 모델을 학습하고 '80% 예측 구간'의 실제 포함률(coverage)을 구간별로 비교한다.

- MSE 회귀 + 등분산 가정: 평균 ± 1.2816 * (잔차 표준편차)
- 이분산 가우시안 회귀: 평균 ± 1.2816 * sigma(x), sigma(x) = a x + d (참 모델과 같은 꼴)
- 분위수 회귀: tau = 0.1, 0.9 두 직선

데이터: y = 2x + eps, eps ~ N(0, (0.1 + 0.5x)^2), x ~ U(0, 1)
실행: uv run --project labs python labs/statistical-inference/distributional-formulation/compare_models.py
"""
import numpy as np

Z80 = 1.2816  # 표준정규분포의 90% 분위수: 평균 ± Z80 * sigma가 80% 구간
EPS = 1e-3    # sigma의 하한 (양수 보장)


def make_data(n, rng):
    x = rng.uniform(0, 1, size=n)
    y = 2.0 * x + rng.normal(size=n) * (0.1 + 0.5 * x)
    return x, y


def fit_mse(x, y):
    w, c = np.polyfit(x, y, 1)
    s = np.std(y - (w * x + c))
    return lambda x: (w * x + c - Z80 * s, w * x + c + Z80 * s)


def fit_hetero(x, y, steps=20000, lr=0.05):
    """mu = w x + c, sigma = max(a x + d, EPS)를 NLL 경사하강으로 학습한다."""
    w, c = np.polyfit(x, y, 1)
    a, d = 0.0, float(np.std(y - (w * x + c)))
    for _ in range(steps):
        mu = w * x + c
        raw = a * x + d
        s = np.maximum(raw, EPS)
        active = raw > EPS                           # 하한에 걸린 샘플은 sigma 기울기 0
        d_mu = -(y - mu) / s**2
        d_s = (1.0 / s - (y - mu) ** 2 / s**3) * active
        w, c = w - lr * np.mean(d_mu * x), c - lr * np.mean(d_mu)
        a, d = a - lr * np.mean(d_s * x), d - lr * np.mean(d_s)
    sigma = lambda x: np.maximum(a * x + d, EPS)
    print(f"이분산 모델: mu = {w:.3f} x {c:+.3f},  sigma = {a:.3f} x {d:+.3f}   (참값: 2x, 0.5x + 0.1)")
    return lambda x: ((w * x + c) - Z80 * sigma(x), (w * x + c) + Z80 * sigma(x))


def fit_quantile(x, y, tau, steps=40000, lr=0.5):
    """q(x) = a x + b를 pinball loss의 부분 기울기 하강으로 학습한다."""
    a, b = np.polyfit(x, y, 1)
    for t in range(steps):
        u = y - (a * x + b)
        g = -(tau - (u < 0))                         # d rho / d q
        step = lr / np.sqrt(t + 1)
        a, b = a - step * np.mean(g * x), b - step * np.mean(g)
    return lambda x: a * x + b


def coverage_by_bin(x, y, lo, hi, bins=(0, 1 / 3, 2 / 3, 1)):
    out = []
    for l, r in zip(bins[:-1], bins[1:]):
        m = (x >= l) & (x < r)
        out.append(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
    return out


def main():
    rng = np.random.default_rng(0)
    x_tr, y_tr = make_data(4000, rng)
    x_te, y_te = make_data(20000, rng)

    q_lo, q_hi = fit_quantile(x_tr, y_tr, 0.1), fit_quantile(x_tr, y_tr, 0.9)
    models = {
        "MSE+등분산": fit_mse(x_tr, y_tr),
        "이분산 가우시안": fit_hetero(x_tr, y_tr),
        "분위수 회귀": lambda x: (q_lo(x), q_hi(x)),
    }
    print()
    print("80% 예측 구간의 실제 포함률 (목표 0.80)")
    print("x<1/3  1/3~2/3  x>2/3   전체   모델")
    for name, model in models.items():
        lo, hi = model(x_te)
        cov = coverage_by_bin(x_te, y_te, lo, hi)
        total = np.mean((y_te >= lo) & (y_te <= hi))
        print(f"{cov[0]:.3f}  {cov[1]:.3f}    {cov[2]:.3f}   {total:.3f}  {name}")


if __name__ == "__main__":
    main()
