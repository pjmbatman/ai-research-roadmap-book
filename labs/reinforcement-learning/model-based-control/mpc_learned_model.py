"""20.3 실습: 학습된 동역학 모델로 MPC를 돌리고 실제 모델을 쓴 MPC와 비교한다.

문제: 감쇠 진자를 각도 0에서 목표 각도 0.8 rad로 옮겨 유지한다.
제약: 토크 |u| <= 8 [1/s^2], 각속도 |w| <= 0.6 [rad/s] (MPC에서는 벌점으로 처리).
모델: (1) 실제 비선형 모델, (2) 선형 회귀 모델 [th, w, u, 1],
      (3) 물리 특징을 넣은 회귀 모델 [th, w, u, sin th, 1].
각 모델로 샘플링 기반 MPC(교차 엔트로피 방법, warm start)를 하고, 실제 시스템에서 비용·최종 오차·제약 위반을 잰다. seed 5개.

실행: uv run --project labs python labs/reinforcement-learning/model-based-control/mpc_learned_model.py
"""
import numpy as np

DT_S, G_MS2, LEN_M, DAMP = 0.1, 9.81, 1.0, 0.1
U_MAX, W_MAX, TARGET = 8.0, 0.6, 0.8


def pend(X, U):
    th, w = X[..., 0], X[..., 1]
    w2 = w + DT_S * (-G_MS2 / LEN_M * np.sin(th) - DAMP * w + U[..., 0])
    return np.stack([th + DT_S * w2, w2], axis=-1)


def fit(features, rng, n=500, noise=0.01):
    X = rng.uniform([-1.5, -2.0], [1.5, 2.0], size=(n, 2))
    U = rng.uniform(-U_MAX, U_MAX, size=(n, 1))
    Y = pend(X, U) + rng.normal(0, noise, size=(n, 2))
    Theta, *_ = np.linalg.lstsq(features(X, U), Y, rcond=None)
    return lambda X_, U_: features(X_, U_) @ Theta


def lin_feat(X, U):
    return np.hstack([X, U, np.ones((len(X), 1))])


def phys_feat(X, U):
    return np.hstack([X, U, np.sin(X[:, :1]), np.ones((len(X), 1))])


def stage_cost(X, U):
    return (X[:, 0] - TARGET) ** 2 + 0.1 * X[:, 1] ** 2 + 0.001 * U[:, 0] ** 2


def plan_cost(x, U, model):
    X, J = np.tile(x, (len(U), 1)), np.zeros(len(U))
    for h in range(U.shape[1]):
        J += stage_cost(X, U[:, h]) + 100.0 * np.maximum(np.abs(X[:, 1]) - W_MAX, 0.0)
        X = model(X, U[:, h])
    return J


def mpc_action(x, model, rng, mean, H=15, N=300, n_elite=30, n_iter=4):
    """교차 엔트로피 방법(CEM): 좋은 입력열 주변으로 표집 분포를 좁혀 가는 샘플링 MPC."""
    std = np.full((H, 1), U_MAX / 2)
    for _ in range(n_iter):
        U = np.clip(mean + std * rng.normal(size=(N, H, 1)), -U_MAX, U_MAX)
        elite = U[np.argsort(plan_cost(x, U, model))[:n_elite]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
    return mean


def run(model, rng, T=80, H=15):
    x, cost, viol, errs, mean = np.zeros(2), 0.0, 0, [], np.zeros((H, 1))
    for t in range(T):
        plan = mpc_action(x, model, rng, mean, H=H)
        u = plan[0]
        mean = np.vstack([plan[1:], plan[-1:]])          # 다음 step의 초기 추측(warm start)
        cost += stage_cost(x[None], u[None])[0]
        viol += int(abs(x[1]) > W_MAX + 1e-9)
        if t >= T - 20:
            errs.append(abs(x[0] - TARGET))
        x = pend(x, u)
    return cost, np.mean(errs), viol


def main():
    print(f"{'모델':22s} {'비용 평균':>9s} {'마지막 20 step 각도 오차 [rad]':>26s} {'속도 제약 위반 step':>18s}")
    for name, make in [("실제 모델", lambda rng: pend),
                       ("선형 회귀 모델", lambda rng: fit(lin_feat, rng)),
                       ("sin 특징 회귀 모델", lambda rng: fit(phys_feat, rng))]:
        res = np.array([run(make(np.random.default_rng(100 + s)), np.random.default_rng(s)) for s in range(5)])
        print(f"{name:22s} {res[:, 0].mean():9.3f} {res[:, 1].mean():26.3f} {res[:, 2].mean():18.1f}")


if __name__ == "__main__":
    main()
