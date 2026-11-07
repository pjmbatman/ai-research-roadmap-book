"""26.4 실습: 대리모델로 찾은 최적 설계를 고충실도 해석으로 재검증한다.

실행: uv run --project labs python labs/physics-and-engineering/design-optimization/surrogate_design.py
- 설계: 막대의 열전도도 분포를 제어점 3개(x = 0, 0.5, 1)의 로그값으로 매개변수화하고 재료 예산 sum(k) h = 1로 정규화
- 목적: 단열된 끝의 최고 온도 J = T(L) (요소 200개 유한요소 = 고충실도)
- 대리모델: GP(표본 8 / 16 / 32개, 라틴 초입방체), 저충실도 해석(요소 4개)
- 기준: L-BFGS-B(수치 기울기, 다중 시작)로 고충실도 문제를 직접 최적화한 값
- 이론 하한: 어떤 분포도 연속 문제의 최적값 4/9보다 낮은 J를 낼 수 없다(본문 셀 2)
"""
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

BOUNDS = np.array([[-2.0, 2.0]] * 3)          # 제어점 로그 전도도의 범위


def peak_temp(z, n=200):
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    k = np.exp(np.interp(x, [0.0, 0.5, 1.0], z))
    k *= 1.0 / (k.sum() * h)                  # 재료 예산 1
    main = np.append(k[:-1] + k[1:], k[-1]) / h
    K = np.diag(main) - np.diag(k[1:] / h, 1) - np.diag(k[1:] / h, -1)
    f = np.full(n, h)
    f[-1] /= 2
    return np.linalg.solve(K, f)[-1]


def multistart(fun, rng, n_starts=20):
    best = None
    for z0 in rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_starts, 3)):
        r = minimize(fun, z0, method="L-BFGS-B", bounds=BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    rng = np.random.default_rng(0)
    ref = multistart(peak_temp, rng, n_starts=10)
    print(f"고충실도 직접 최적화(기준): J = {ref.fun:.5f}  z = {np.round(ref.x, 2)}")
    print(f"균일 설계: J = {peak_temp(np.zeros(3)):.5f}")
    print(f"{'대리모델':22s} {'예측 J':>8s} {'재검증 J':>9s} {'예측 오차':>9s} {'기준 대비 손실':>12s}  예측 < 이론 하한 4/9")
    for m in [8, 16, 32]:
        Z = qmc.scale(qmc.LatinHypercube(d=3, seed=m).random(m), BOUNDS[:, 0], BOUNDS[:, 1])
        y = np.array([peak_temp(z) for z in Z])
        kernel = ConstantKernel(1.0) * RBF([1.0] * 3) + WhiteKernel(1e-6, (1e-10, 1e-2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                                          random_state=0).fit(Z, y)
        r = multistart(lambda z: gp.predict(z[None])[0], rng)
        report(f"GP, 표본 {m}개", r.fun, peak_temp(r.x), ref.fun)
    r = multistart(lambda z: peak_temp(z, n=4), rng)
    report("저충실도 해석(요소 4개)", r.fun, peak_temp(r.x), ref.fun)


def report(name, pred, true, best):
    flag = "예" if pred < 4 / 9 else "아니오"
    print(f"{name:22s} {pred:8.5f} {true:9.5f} {pred - true:+9.5f} {true - best:+12.5f}  {flag}")


if __name__ == "__main__":
    main()
