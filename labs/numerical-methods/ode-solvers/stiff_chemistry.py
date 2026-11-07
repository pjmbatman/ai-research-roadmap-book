"""9.6 실습: stiff한 화학 반응 ODE(Robertson 문제)를 explicit RK와 implicit 방법으로 풀어 step 수·시간·정확도를 비교하고,
tolerance를 바꾸어 결과가 수렴하는지 확인한다.

실행: uv run --project labs python labs/numerical-methods/ode-solvers/stiff_chemistry.py
(벽시계 시간은 기계마다 다르다. step 수와 f 호출 수는 같은 SciPy 버전에서 재현된다.)
"""
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp


def robertson(t, y):
    """A → B (0.04), B + B → C + B (3e7), B + C → A + C (1e4). y = 세 화학종의 농도, shape (3,)"""
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3,
            0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
            3e7 * y2**2]


def jac(t, y):
    """해석적 Jacobian, shape (3, 3). implicit 방법에 주면 유한차분 Jacobian 계산을 생략한다."""
    y1, y2, y3 = y
    return [[-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]]


Y0 = [1.0, 0.0, 0.0]


def reference(T):
    """매우 엄격한 tolerance의 Radau 해. 이것도 근사이므로, 아래에서 두 tolerance의 차이로 그 오차를 확인한다."""
    return solve_ivp(robertson, (0, T), Y0, method="Radau", jac=jac, rtol=1e-12, atol=[1e-14, 1e-18, 1e-14]).y[:, -1]


def run(method, T, **kw):
    t0 = time.perf_counter()
    sol = solve_ivp(robertson, (0, T), Y0, method=method, **kw)
    return sol, time.perf_counter() - t0


def report(T, methods, ref, **kw):
    print(f"\n[T = {T:g}]  기준해 y = {np.array2string(ref, precision=6)}")
    print(f"{'method':<7}{'성공':>6}{'도달 t':>10}{'step':>7}{'f 호출':>8}{'Jac':>5}{'시간[s]':>9}{'최대 상대오차':>14}")
    for m in methods:
        extra = {"jac": jac} if m in ("BDF", "Radau", "LSODA") else {}
        sol, sec = run(m, T, **kw, **extra)
        err = np.max(np.abs(sol.y[:, -1] - ref) / np.abs(ref)) if sol.success else np.nan
        print(f"{m:<7}{sol.success!s:>6}{sol.t[-1]:10.3g}{sol.t.size - 1:7d}{sol.nfev:8d}{sol.njev:5d}{sec:9.3f}{err:14.1e}")


def main():
    # explicit 방법(특히 DOP853)이 시도했다가 버리는 step에서 overflow 경고가 난다. 결과에는 영향이 없어 숨긴다.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # 1) 짧은 구간: explicit 방법도 끝까지 가지만 step 수가 훨씬 많다.
    report(1.0, ["RK23", "RK45", "DOP853", "BDF", "Radau", "LSODA"], reference(1.0))
    # 2) 긴 구간: 기본 tolerance의 explicit 방법은 실패하거나 매우 많은 step을 쓴다.
    report(40.0, ["RK45", "BDF", "Radau", "LSODA"], reference(40.0))
    report(1e5, ["BDF", "Radau", "LSODA"], reference(1e5))

    # 3) tolerance 수렴 확인: 같은 solver에서 tolerance를 10배씩 줄여 결과의 변화를 본다.
    print("\n[tolerance 수렴 확인: BDF, T = 40, atol = rtol × 1e-6]")
    prev = None
    for rtol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        sol = solve_ivp(robertson, (0, 40), Y0, method="BDF", jac=jac, rtol=rtol, atol=rtol * 1e-6)
        y = sol.y[:, -1]
        change = "" if prev is None else f"  직전 대비 최대 상대 변화 {np.max(np.abs(y - prev) / np.abs(y)):.1e}"
        print(f"rtol {rtol:.0e}  y2(40) = {y[1]:.8e}  step {sol.t.size - 1:4d}{change}")
        prev = y

    # 4) 기준해 자체의 검증: 서로 다른 solver·tolerance의 기준해가 일치하는가.
    r1 = reference(40.0)
    r2 = solve_ivp(robertson, (0, 40), Y0, method="BDF", jac=jac, rtol=1e-11, atol=[1e-13, 1e-17, 1e-13]).y[:, -1]
    print(f"\n기준해 확인: Radau(1e-12)와 BDF(1e-11)의 최대 상대 차이 {np.max(np.abs(r1 - r2) / np.abs(r1)):.1e}")


if __name__ == "__main__":
    main()
