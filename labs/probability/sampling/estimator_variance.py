"""11.6 실습: 같은 기댓값을 여러 Monte Carlo 추정량으로 구해 분산을 비교한다.

목표량: I = E[exp(X)], X ~ N(0, 1). 참값 exp(1/2) (로그정규분포의 평균).
모든 방법은 추정 한 번에 함수 exp를 N번 계산하도록 예산을 맞춘다. 같은 실험을 R번 반복해
추정값의 평균·표준편차와 naive 대비 분산비(클수록 좋음)를 보고한다.

실행: uv run --project labs python labs/probability/sampling/estimator_variance.py
"""
import numpy as np
from scipy import stats

SEED, N, R = 0, 1_000, 2_000
EXACT = np.exp(0.5)


def naive(rng):
    return np.exp(rng.normal(size=(R, N))).mean(axis=1), None


def antithetic(rng):
    x = rng.normal(size=(R, N // 2))                     # 쌍 N/2개 = 함수 계산 N번
    return (0.5 * (np.exp(x) + np.exp(-x))).mean(axis=1), None


def control_variate(rng):
    x = rng.normal(size=(R, N))
    f = np.exp(x)
    g = x + x**2 / 2                                     # E[g] = 1/2, exp의 2차 Taylor 근사
    gc = g - g.mean(axis=1, keepdims=True)
    c = (gc * (f - f.mean(axis=1, keepdims=True))).sum(axis=1) / (gc**2).sum(axis=1)   # 추정마다 최적 계수
    return (f - c[:, None] * (g - 0.5)).mean(axis=1), None


def stratified(rng):
    u = (np.arange(N) + rng.uniform(size=(R, N))) / N    # 층 N개에서 하나씩
    return np.exp(stats.norm.ppf(u)).mean(axis=1), None


def importance(loc, scale):
    def run(rng):
        z = rng.normal(loc, scale, size=(R, N))
        w = np.exp(stats.norm.logpdf(z) - stats.norm.logpdf(z, loc, scale))   # p/q
        ess = w.sum(axis=1) ** 2 / (w**2).sum(axis=1)
        return (w * np.exp(z)).mean(axis=1), ess / N
    return run


METHODS = [
    ("naive", naive),
    ("antithetic", antithetic),
    ("control variate (x + x²/2)", control_variate),
    ("stratified (층 N개)", stratified),
    ("IS q=N(0.5, 1)", importance(0.5, 1.0)),
    ("IS q=N(1, 1) (최적)", importance(1.0, 1.0)),
    ("IS q=N(0, 0.6²) (좁음)", importance(0.0, 0.6)),
]


def main():
    rng = np.random.default_rng(SEED)
    print(f"I = E[exp X], X ~ N(0,1), 참값 {EXACT:.5f}   (추정 한 번에 함수 계산 {N}번, 반복 {R}번, 시드 {SEED})")
    print("평균      표준편차   분산비     ESS/N   방법")
    base = None
    for name, method in METHODS:
        est, ess = method(rng)
        var = est.var()
        base = var if base is None else base
        ratio = f"{base / var:8.2f}" if var > 1e-20 else "       ∞"
        ess_s = f"{np.median(ess):.3f}" if ess is not None else "  -  "
        print(f"{est.mean():.5f}  {est.std():.2e}  {ratio}   {ess_s}   {name}")

    # 좁은 제안분포: 분산이 무한하다. 반복 추정값의 분포가 어떻게 생겼는지 본다.
    est, _ = importance(0.0, 0.6)(rng)
    q = np.quantile(est, [0.05, 0.5, 0.95])
    print()
    print(f"좁은 제안분포 추정값: 5% {q[0]:.3f}, 중앙값 {q[1]:.3f}, 95% {q[2]:.3f}, 최댓값 {est.max():.3f}, "
          f"참값보다 작은 비율 {np.mean(est < EXACT):.2f}")


if __name__ == "__main__":
    main()
