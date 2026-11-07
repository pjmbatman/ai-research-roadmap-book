"""18.3 실습: 같은 모델을 one-step(teacher forcing)과 free rollout으로 평가하고 오차 누적을 분석한다.

계: 감쇠 진자 θ'' = -(g/ℓ) sin θ - c θ',  g/ℓ = 9.81 s⁻², c = 0.05 s⁻¹, 관측 간격 0.05 s.
학습: 초기 각 0.5, 1.0, 1.5 rad 궤적(60 s), 관측 잡음 표준편차 0.001 rad. θ_{t+1} = φ(θ_t, θ_{t-1})·w 를 최소제곱으로.
모델 세 가지(특징 φ만 다름):
  - 선형 AR(2)  : [θ_t, θ_{t-1}]
  - 3차 AR(2)   : 위에 3차 단항식 4개
  - sin AR(2)   : [θ_t, θ_{t-1}, sin θ_t]  (운동 방정식의 형태를 아는 모델)
평가: 학습 범위 안(1.2 rad)과 밖(2.0 rad)의 초기 각에서
  - one-step RMSE: 참값 과거로 한 step 예측
  - rollout: 처음 두 값만 주고 자기 예측을 이어 붙인 궤적의 시간 구간별 RMSE, 오차가 처음 0.1 rad를 넘는 시각

실행: uv run --project labs python labs/architectures/recurrent-and-ssm/rollout_eval.py
"""
import numpy as np

DT_S, N_STEPS = 0.05, 1200


def pendulum_traj(th0, n_steps=N_STEPS, dt_s=DT_S, sub=10):
    f = lambda s: np.stack([s[1], -9.81 * np.sin(s[0]) - 0.05 * s[1]])
    s, h, out = np.stack([np.asarray(th0, float), np.zeros(len(th0))]), dt_s / sub, []
    for _ in range(n_steps):
        out.append(s[0].copy())
        for _ in range(sub):
            k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(out).T


FEATS = {
    "선형 AR(2)": lambda a, b: np.c_[a, b],
    "3차 AR(2)": lambda a, b: np.c_[a, b, a ** 3, b ** 3, a ** 2 * b, a * b ** 2],
    "sin AR(2)": lambda a, b: np.c_[a, b, np.sin(a)],
}


def fit(phi, train):
    X = np.vstack([phi(s[1:-1], s[:-2]) for s in train])
    y = np.concatenate([s[2:] for s in train])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def rollout(phi, w, first_two, n):
    roll = list(first_two)
    for _ in range(n - 2):
        roll.append((phi(np.array([roll[-1]]), np.array([roll[-2]])) @ w).item())
    return np.array(roll)


def main():
    rng = np.random.default_rng(0)
    train = pendulum_traj([0.5, 1.0, 1.5]) + 0.001 * rng.normal(size=(3, N_STEPS))
    tests = {"범위 안 1.2 rad": 1.2, "범위 밖 2.0 rad": 2.0}
    horizons_s = [1, 5, 20, 60]
    for tname, th0 in tests.items():
        test = pendulum_traj([th0])[0]
        print(f"[{tname}]")
        print("모델          one-step RMSE   rollout RMSE " + "  ".join(f"~{h:2d} s" for h in horizons_s) + "   0.1 rad 넘는 시각")
        for name, phi in FEATS.items():
            w = fit(phi, train)
            one = np.sqrt(np.mean((phi(test[1:-1], test[:-2]) @ w - test[2:]) ** 2))
            err = np.abs(rollout(phi, w, test[:2], len(test)) - test)
            rms = [np.sqrt(np.mean(err[: int(h / DT_S)] ** 2)) for h in horizons_s]
            over = np.flatnonzero(err > 0.1)
            t_fail = f"{over[0] * DT_S:5.2f} s" if len(over) else "  없음"
            print(f"{name:12s}  {one:13.1e}   {'':12s}" + "  ".join(f"{r:5.3f}" for r in rms) + f"   {t_fail}")
        print()


if __name__ == "__main__":
    main()
