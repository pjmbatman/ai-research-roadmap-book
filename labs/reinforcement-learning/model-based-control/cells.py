"""20.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reinforcement-learning/model-based-control/cells.py
"""
# %% cell 1 — 이중 적분기의 LQR: Riccati 반복
import numpy as np
from scipy.linalg import solve_discrete_are

dt_s = 0.1                                   # 시간 간격 [s]
A = np.array([[1.0, dt_s], [0.0, 1.0]])      # 상태 x = (위치 [m], 속도 [m/s]), (2, 2)
B = np.array([[0.5 * dt_s**2], [dt_s]])      # 입력 u = 가속도 [m/s^2], (2, 1)
Qc, Rc = np.diag([1.0, 0.1]), np.array([[0.1]])   # 상태 비용, 입력 비용
P = Qc.copy()                                # 가치함수 V(x) = x^T P x, 지평 끝에서 출발
for k in range(500):                         # Riccati 반복 = 이차 가치함수에 대한 DP
    K = np.linalg.solve(Rc + B.T @ P @ B, B.T @ P @ A)    # (1, 2), u = -K x
    P_new = Qc + A.T @ P @ (A - B @ K)
    if np.abs(P_new - P).max() < 1e-12:
        break
    P = P_new
print(f"Riccati {k + 1}번 반복, K = {K.round(3)}")
print("scipy DARE와 일치:", np.allclose(P, solve_discrete_are(A, B, Qc, Rc)))
print("폐루프 고윳값 크기:", np.abs(np.linalg.eigvals(A - B @ K)).round(3))
def rollout(policy, x0, T=60):               # 실제 시스템에서 비용과 궤적을 잰다
    x, cost, xs, us = x0.copy(), 0.0, [], []
    for t in range(T):
        u = policy(x, t)
        cost += x @ Qc @ x + u @ Rc @ u
        xs.append(x); us.append(u)
        x = A @ x + B @ u
    return cost, np.array(xs), np.array(us)
x0 = np.array([1.0, 0.0])                    # 1 m 떨어진 곳에서 정지 상태로 출발
cost, xs, us = rollout(lambda x, t: -K @ x, x0)
print(f"LQR 비용 {cost:.3f}  (예측 x0^T P x0 = {x0 @ P @ x0:.3f}),  최대 |u| = {np.abs(us).max():.2f} m/s^2,"
      f"  최대 |v| = {np.abs(xs[:, 1]).max():.2f} m/s")

# %% cell 2 — 제약이 있는 MPC: 랜덤 슈팅과 receding horizon
U_MAX, V_MAX = 1.0, 0.5                      # 입력 한계 [m/s^2], 속도 한계 [m/s]
rng = np.random.default_rng(0)

def mpc(x, t, model=lambda X, U: X @ A.T + U @ B.T, H=15, N=1000):
    U = rng.uniform(-U_MAX, U_MAX, size=(N, H, 1))         # 후보 입력열 N개, (N, H, 1)
    X, J = np.tile(x, (N, 1)), np.zeros(N)                 # (N, 2), (N,)
    for h in range(H):                       # 모델로 H step 앞까지 굴려 비용을 센다
        J += np.einsum("ni,ij,nj->n", X, Qc, X) + Rc[0, 0] * U[:, h, 0] ** 2
        J += 1e3 * np.maximum(np.abs(X[:, 1]) - V_MAX, 0.0)   # 속도 제약 위반 벌점
        X = model(X, U[:, h])
    return U[np.argmin(J), 0]                # 가장 좋은 입력열의 첫 입력만 쓴다

lqr_clip = lambda x, t: np.clip(-K @ x, -U_MAX, U_MAX)     # 비교: LQR을 한계에서 자른 것
for name, pol in [("LQR(자르기)", lqr_clip), ("MPC(랜덤 슈팅)", mpc)]:
    c, xs, us = rollout(pol, x0)
    print(f"{name:14s} 비용 {c:6.3f}  최대 |u| {np.abs(us).max():.2f}  최대 |v| {np.abs(xs[:, 1]).max():.2f}"
          f"  최종 |x| {np.abs(xs[-1, 0]):.3f} m")

# %% cell 3 — 학습된 동역학 모델과 오차의 누적
g_ms2, len_m, damp = 9.81, 1.0, 0.1          # 진자: 중력 가속도, 길이, 감쇠
def pend(X, U):                              # 실제 시스템(비선형), X = (각도 [rad], 각속도 [rad/s])
    th, w = X[..., 0], X[..., 1]
    w2 = w + dt_s * (-g_ms2 / len_m * np.sin(th) - damp * w + U[..., 0])
    return np.stack([th + dt_s * w2, w2], axis=-1)

X_data = rng.uniform([-1.5, -2.0], [1.5, 2.0], size=(500, 2))   # 무작위 상태 500개
U_data = rng.uniform(-2.0, 2.0, size=(500, 1))                  # 무작위 토크 [1/s^2]
Y = pend(X_data, U_data) + rng.normal(0, 0.01, size=(500, 2))   # 측정 잡음
Z = np.hstack([X_data, U_data, np.ones((500, 1))])              # (500, 4) 선형 모델 특징
Theta, *_ = np.linalg.lstsq(Z, Y, rcond=None)                    # (4, 2) 최소제곱 적합
learned = lambda X, U: np.hstack([X, U, np.ones((len(X), 1))]) @ Theta

X0 = rng.uniform([-1.2, -1.0], [1.2, 1.0], size=(200, 2))   # 시험 초기 상태 200개, 입력 0
x_true, x_mod, zero, worst = X0, X0, np.zeros((200, 1)), np.zeros(200)
for k in range(1, 41):                       # 모델을 자기 예측 위에서 굴린다(open-loop rollout)
    x_true, x_mod = pend(x_true, zero), learned(x_mod, zero)
    worst = np.maximum(worst, np.abs(x_mod[:, 0] - x_true[:, 0]))   # 지평 k까지의 최대 오차
    if k in (1, 5, 10, 20, 40):
        print(f"지평 {k:2d} step 안의 최대 각도 오차(200개 평균) {worst.mean():.3f} rad")

# %% cell 4 — 미분가능 시뮬레이션: 시간을 거슬러 기울기 전파
def grad_open_loop(A_, U, x0_, Qm, r):       # J = sum_t x_t^T Q x_t + r u_t^2,  x_{t+1} = A x_t + B u_t
    T, xs = len(U), [x0_]
    for t in range(T):
        xs.append(A_ @ xs[-1] + B[:, 0] * U[t])
    lam, g = 2 * Qm @ xs[T], np.zeros(T)     # 수반(adjoint) 변수 lam_t = dJ/dx_t
    for t in reversed(range(T)):
        g[t] = 2 * r * U[t] + B[:, 0] @ lam  # dJ/du_t
        lam = 2 * Qm @ xs[t] + A_.T @ lam
    return g, sum(x @ Qm @ x for x in xs) + r * np.sum(U**2)

U = np.zeros(40)
g, J = grad_open_loop(A, U, x0, Qc, 0.1)
eps = 1e-6; e = np.zeros(40); e[7] = eps
num = (grad_open_loop(A, U + e, x0, Qc, 0.1)[1] - J) / eps
print(f"수반법 dJ/du_7 = {g[7]:.4f}, 유한차분 = {num:.4f}")
for _ in range(300):                         # 기울기 하강으로 개루프 입력열 최적화
    g, J = grad_open_loop(A, U, x0, Qc, 0.1)
    U -= 0.02 * g
Pt = Qc.copy()
for _ in range(40):                          # 같은 40 step 문제의 정확한 최적값(유한 지평 Riccati)
    Pt = Qc + A.T @ Pt @ A - A.T @ Pt @ B @ np.linalg.solve(Rc + B.T @ Pt @ B, B.T @ Pt @ A)
print(f"기울기 하강 300번 뒤 비용 {J:.4f}, 유한 지평 Riccati 최적값 {x0 @ Pt @ x0:.4f}")
A_unst = np.array([[1.0, dt_s], [dt_s * 9.81, 1.0]])   # 뒤집힌 진자의 선형화: 불안정
for T in (10, 40, 80):
    g, _ = grad_open_loop(A_unst, np.zeros(T), x0, Qc, 0.1)
    print(f"불안정 시스템, 지평 {T:2d}: |dJ/du_0| = {abs(g[0]):.2e}")
