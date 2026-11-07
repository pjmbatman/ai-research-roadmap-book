"""30.5 실습: 운전 조건(C-rate, 온도)에 대한 배터리 방전 용량의 대리모델.

고충실도 모델: 단일 입자 확산 모델(구 안의 Fick 확산, 유한체적 30층, 암시적 Euler)에
Arrhenius 온도 의존 직렬 저항을 붙인 가상의 셀. 출력은 3.3 V 차단까지의 방전 용량 비율이다.
대리모델: (1) 확산을 무시한 저충실도 물리 모델, (2) 데이터만 쓴 GP, (3) 저충실도 모델 + GP 잔차 보정(하이브리드).
평가: 학습 범위 안·밖의 정확도, 물리적 일관성(C-rate가 커지면 용량이 줄어야 한다, 용량 비율은 0–1), 속도.

모든 파라미터는 설명용 값이다(특정 셀의 측정값이 아니다).

실행: uv run --project labs python labs/battery-ai/parameter-estimation/surrogate.py
"""
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

Q_AH, V_CUT_V, N_SHELL, DT_S = 3.0, 3.3, 30, 5.0
R_GAS, T_REF_K = 8.314, 298.15


def ocv_v(theta):
    """입자 표면 농도(0–1)의 개방 전압 곡선(설명용)."""
    theta = np.clip(theta, 1e-6, 1 - 1e-6)
    return 3.5 + 0.55 * theta + 0.04 * np.log((theta + 0.02) / (1.02 - theta))


def arrhenius(value_ref, ea_j, temp_c):
    return value_ref * np.exp(ea_j / R_GAS * (1 / (temp_c + 273.15) - 1 / T_REF_K))


def high_fidelity(c_rate, temp_c):
    """3.3 V 차단까지 방전한 용량 비율. 구 좌표 Fick 확산, 무차원 반지름 r ∈ [0, 1]."""
    i_a = c_rate * Q_AH
    tau_s = 1 / arrhenius(1 / 1800.0, 35e3, temp_c)      # 확산 시상수 R_p^2 / D (25 °C에서 1800 s)
    r0_ohm = arrhenius(0.020, 30e3, temp_c)
    edges = np.linspace(0, 1, N_SHELL + 1)              # (N+1,) 층 경계
    vol = (edges[1:] ** 3 - edges[:-1] ** 3) / 3        # (N,) 층 부피 (4π 생략)
    area = edges[1:-1] ** 2                             # (N-1,) 내부 경계 면적
    flux_coef = area / (edges[1] - edges[0]) / tau_s    # 인접 층 사이 확산 계수
    L = np.zeros((N_SHELL, N_SHELL))
    for k, g in enumerate(flux_coef):                   # 층 k와 k+1 사이
        L[k, k] -= g / vol[k]; L[k, k + 1] += g / vol[k]
        L[k + 1, k + 1] -= g / vol[k + 1]; L[k + 1, k] += g / vol[k + 1]
    lu = lu_factor(np.eye(N_SHELL) - DT_S * L)          # 암시적 Euler
    src = np.zeros(N_SHELL)
    src[-1] = -(i_a / (3600 * Q_AH)) * vol.sum() / vol[-1]   # 표면 층에서 빠져나가는 리튬(평균 농도 감소율 = I / Q)
    theta = np.ones(N_SHELL)
    t_s = 0.0
    while t_s < 4 * 3600 / c_rate:
        theta_new = lu_solve(lu, theta + DT_S * src)
        v = ocv_v(1.5 * theta_new[-1] - 0.5 * theta_new[-2]) - i_a * r0_ohm   # 표면 농도는 외삽
        if v < V_CUT_V or theta_new[-1] <= 0:
            break
        theta, t_s = theta_new, t_s + DT_S
    return i_a * t_s / 3600 / Q_AH


def low_fidelity(X):
    """확산을 무시한 저충실도 물리 모델: OCV(1 - q) - I R0(T) = 3.3 V가 되는 용량 비율 q.
    R0(T)의 Arrhenius 파라미터는 펄스 시험으로 추정했다고 본다(이 장의 본문). X: (N, 2) → (N,)"""
    out = []
    for c_rate, temp_c in X:
        drop_v = c_rate * Q_AH * arrhenius(0.020, 30e3, temp_c)
        g = lambda q: ocv_v(1 - q) - drop_v - V_CUT_V
        out.append(0.0 if g(0.0) < 0 else (1.0 if g(1 - 1e-6) > 0 else brentq(g, 0.0, 1 - 1e-6)))
    return np.array(out)


def fit_gp(X, y):
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(1e-6, (1e-10, 1e-2))
    mu, sd = X.mean(0), X.std(0)
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit((X - mu) / sd, y)
    return lambda Xq: gp.predict((Xq - mu) / sd)


def main():
    rng = np.random.default_rng(0)
    # 학습 범위: 0.5–2 C, 10–40 °C에서 고충실도 실행 30번
    X_train = np.column_stack([rng.uniform(0.5, 2.0, 30), rng.uniform(10, 40, 30)])
    t0 = time.perf_counter()
    y_train = np.array([high_fidelity(c, T) for c, T in X_train])
    t_hf = (time.perf_counter() - t0) / len(X_train)

    tests = {
        "범위 안 (0.5–2 C, 10–40 °C)": np.column_stack([rng.uniform(0.5, 2.0, 40), rng.uniform(10, 40, 40)]),
        "고율 외삽 (2–3 C, 10–40 °C)": np.column_stack([rng.uniform(2.0, 3.0, 40), rng.uniform(10, 40, 40)]),
        "저온 외삽 (0.5–2 C, -10–10 °C)": np.column_stack([rng.uniform(0.5, 2.0, 40), rng.uniform(-10, 10, 40)]),
    }
    y_test = {k: np.array([high_fidelity(c, T) for c, T in X]) for k, X in tests.items()}

    print(f"고충실도 모델 학습 실행 30번, 용량 비율 범위 {y_train.min():.3f}–{y_train.max():.3f}")
    gp_only = fit_gp(X_train, y_train)
    gp_resid = fit_gp(X_train, y_train - low_fidelity(X_train))
    models = {
        "저충실도 물리 모델만 (확산 무시)": low_fidelity,
        "데이터만: GP(C-rate, T)": gp_only,
        "하이브리드: 저충실도 + GP 잔차 보정": lambda X: low_fidelity(X) + gp_resid(X),
    }
    for name, model in models.items():
        print(f"\n{name}")
        for k, X in tests.items():
            err = model(X) - y_test[k]
            print(f"  {k:28s} RMSE {np.sqrt(np.mean(err**2)):.4f}  최대 오차 {np.max(np.abs(err)):.4f}")
        # 물리적 일관성: 온도마다 C-rate를 늘리면 용량이 줄어야 한다(0.25–3 C 촘촘한 격자)
        c_grid = np.linspace(0.25, 3.0, 56)
        viol, out_of_range = 0, 0
        for T in np.linspace(-10, 40, 11):
            pred = model(np.column_stack([c_grid, np.full_like(c_grid, T)]))
            viol += int(np.sum(np.diff(pred) > 1e-4))
            out_of_range += int(np.sum((pred < 0) | (pred > 1)))
        print(f"  단조성 위반(C-rate 증가에 용량 증가) {viol} / {10 * 55}구간, 0–1 범위 이탈 {out_of_range} / {11 * 56}점")
        Xq = np.column_stack([rng.uniform(0.5, 2.0, 2000), rng.uniform(10, 40, 2000)])
        t0 = time.perf_counter()
        model(Xq)
        t_sur = (time.perf_counter() - t0) / len(Xq)
        print(f"  평가 1회 시간: 고충실도 {t_hf * 1e3:.1f} ms, 대리모델 {t_sur * 1e6:.1f} µs (약 {t_hf / t_sur:,.0f}배)")


if __name__ == "__main__":
    main()
