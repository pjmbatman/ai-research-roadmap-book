"""20.3 퀴즈(content/20-reinforcement-learning/03-model-based-control/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

QUIZ = Path(__file__).parents[3] / "content/20-reinforcement-learning/03-model-based-control/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_lqr_gain_scalar():
    P = solve_discrete_are(np.eye(1), np.eye(1), np.eye(1), np.eye(1))[0, 0]
    assert np.isclose(P, (1 + np.sqrt(5)) / 2)
    K = P / (1 + P)
    assert abs(K - Q["lqr-gain-scalar"]["answer"]) <= Q["lqr-gain-scalar"]["tolerance"]


def test_lqr_is_dp():
    # Riccati 반복(가치 반복)의 고정점이 DARE 해와 같고, 정책은 선형 u = -K x
    A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.005], [0.1]])
    Qc, R = np.eye(2), np.eye(1)
    P = Qc.copy()
    for _ in range(5000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Qc + A.T @ P @ (A - B @ K)
    assert np.allclose(P, solve_discrete_are(A, B, Qc, R))
    assert correct("lqr-is-dp") == [0]


def test_receding_horizon_corrects_model_error():
    # 모형 x' = x + u, 실제 x' = 1.2 x + u. 개루프 계획(한 번 계획 후 실행)과 매 step 재계획 비교.
    def plan(x):                             # 모형에서 원점으로 한 번에: u = -x
        return -x
    x_open, x_mpc = 1.0, 1.0
    u_seq = [plan(x_open)] + [0.0] * 9       # 모형상 첫 step 뒤 원점이므로 나머지 0
    for t in range(10):
        x_open = 1.2 * x_open + u_seq[t]
        x_mpc = 1.2 * x_mpc + plan(x_mpc)
    assert abs(x_mpc) < abs(x_open)
    assert correct("receding-horizon") == [0]


def test_model_error_compounding():
    # cell 3과 같은 설정으로 1 step 오차보다 40 step 오차가 크다
    dt, g, damp = 0.1, 9.81, 0.1
    def pend(X, U):
        th, w = X[..., 0], X[..., 1]
        w2 = w + dt * (-g * np.sin(th) - damp * w + U[..., 0])
        return np.stack([th + dt * w2, w2], axis=-1)
    rng = np.random.default_rng(0)
    X = rng.uniform([-1.5, -2.0], [1.5, 2.0], size=(500, 2))
    U = rng.uniform(-2.0, 2.0, size=(500, 1))
    Z = np.hstack([X, U, np.ones((500, 1))])
    Th, *_ = np.linalg.lstsq(Z, pend(X, U), rcond=None)
    model = lambda X_, U_: np.hstack([X_, U_, np.ones((len(X_), 1))]) @ Th
    X0, zero = rng.uniform([-1.2, -1.0], [1.2, 1.0], size=(200, 2)), np.zeros((200, 1))
    xt, xm, errs = X0, X0, []
    for _ in range(40):
        xt, xm = pend(xt, zero), model(xm, zero)
        errs.append(np.abs(xt[:, 0] - xm[:, 0]).mean())
    assert max(errs) > 20 * errs[0]
    assert correct("model-error-compounding") == [1]


def test_mpc_vs_rl():
    assert correct("mpc-vs-rl") == [0, 1]


def test_diffsim_exploding():
    A = np.array([[1.0, 0.1], [0.981, 1.0]])  # 뒤집힌 진자 선형화
    assert np.abs(np.linalg.eigvals(A)).max() > 1
    norms = [np.linalg.norm(np.linalg.matrix_power(A.T, T)) for T in (10, 40, 80)]
    assert norms[2] / norms[1] > 1e3 and norms[1] / norms[0] > 1e3    # 지평에 대해 지수적
    assert correct("diffsim-exploding") == [1]
