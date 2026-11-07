"""18.3 퀴즈(content/18-architectures/03-recurrent-and-ssm/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.linalg import expm

QUIZ = Path(__file__).parents[3] / "content/18-architectures/03-recurrent-and-ssm/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_vanishing_factor():
    eps = 1e-6
    xs = rng.normal(size=50)
    def h50(h0):
        h = h0
        for x in xs:
            h = 0.9 * h + x
        return h
    grad = (h50(eps) - h50(0.0)) / eps
    q = Q["vanishing-factor"]
    assert abs(grad - q["answer"]) <= q["tolerance"]
    assert abs(0.9 ** 50 - q["answer"]) <= q["tolerance"]


def test_lstm_additive_path():
    f = 0.97
    assert abs(f ** 60 - 0.16) < 0.005                      # 본문·해설의 수치
    assert correct("lstm-additive-path") == [0]


def test_lstm_param_count():
    D, H = 16, 32
    per_gate = H * D + H * H + 2 * H                         # 입력 가중치, hidden 가중치, bias 두 벌
    assert 4 * per_gate == Q["lstm-param-count"]["answer"] == 6400
    assert 3 * per_gate == 4800 and per_gate == 1600
    assert 4 * (H * D + H * H + H) == 6272


def test_ssm_kernel_formula():
    N, L = 3, 50
    A = np.diag([-0.2, -0.5, -1.0]) + 0.3 * rng.normal(size=(N, N))
    B, C = rng.normal(size=(N, 1)), rng.normal(size=(1, N))
    M = expm(np.block([[A, B], [np.zeros((1, N + 1))]]) * 0.1)
    Ab, Bb = M[:N, :N], M[:N, N:]
    u = rng.normal(size=L)
    x, y = np.zeros((N, 1)), []
    for k in range(L):
        x = Ab @ x + Bb * u[k]
        y.append((C @ x).item())
    K = np.array([(C @ np.linalg.matrix_power(Ab, j) @ Bb).item() for j in range(L)])
    assert np.allclose(y, np.convolve(u, K)[:L])
    K_wrong = np.array([(C @ Ab @ np.linalg.matrix_power(Bb @ np.ones((1, N)), j) @ Bb).item() if j else 0 for j in range(L)])
    assert not np.allclose(y, np.convolve(u, K_wrong)[:L])
    assert correct("ssm-kernel-formula") == [0]


def test_selective_ssm_scan():
    c = rng.normal(size=60)
    delta = np.log1p(np.exp(2 * c))
    def resp(j):
        x, out = 0.0, []
        for k in range(60):
            x = np.exp(-delta[k]) * x + delta[k] * (1.0 if k == j else 0.0)
            out.append(x)
        return np.array(out[j:j + 4])
    assert not np.allclose(resp(10), resp(40))              # 시점에 따라 응답이 다름: 하나의 kernel 없음
    # 선형 재귀 x_k = a_k x_{k-1} + b_k 의 결합 연산 (a1,b1)∘(a2,b2) = (a1 a2, a2 b1 + b2)
    a, b = np.exp(-delta), delta * rng.normal(size=60)
    seq = 0.0
    for ak, bk in zip(a, b):
        seq = ak * seq + bk
    def combine(p, q):
        return (p[0] * q[0], q[0] * p[1] + q[1])
    pairs = list(zip(a, b))
    while len(pairs) > 1:                                    # 트리 모양으로 묶어도 결과가 같다
        pairs = [combine(pairs[i], pairs[i + 1]) if i + 1 < len(pairs) else pairs[i] for i in range(0, len(pairs), 2)]
    assert np.isclose(pairs[0][1], seq)
    assert correct("selective-ssm-scan") == [0]


def test_one_step_vs_rollout():
    # 본문과 같은 설정의 축소판: one-step 오차는 작아도 rollout 오차는 훨씬 크다
    def traj(th0, n=600, dt=0.05, sub=10):
        f = lambda s: np.array([s[1], -9.81 * np.sin(s[0]) - 0.05 * s[1]])
        s, h, out = np.array([th0, 0.0]), dt / sub, []
        for _ in range(n):
            out.append(s[0])
            for _ in range(sub):
                k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
                s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.array(out)
    phi = lambda a, b: np.c_[a, b, a ** 3, b ** 3, a ** 2 * b, a * b ** 2]
    tr = [traj(a) + 0.001 * rng.normal(size=600) for a in (0.5, 1.0, 1.5)]
    w = np.linalg.lstsq(np.vstack([phi(s[1:-1], s[:-2]) for s in tr]), np.concatenate([s[2:] for s in tr]), rcond=None)[0]
    te = traj(1.2)
    one = np.sqrt(np.mean((phi(te[1:-1], te[:-2]) @ w - te[2:]) ** 2))
    roll = list(te[:2])
    for _ in range(598):
        roll.append((phi(np.array([roll[-1]]), np.array([roll[-2]])) @ w).item())
    assert one < 1e-3 and np.abs(np.array(roll) - te).max() > 30 * one
    assert correct("one-step-vs-rollout") == [0]


def test_irregular_sampling():
    # 연속시간 모델을 실제 간격으로 이산화하면 간격을 나눠 두 번 간 것과 한 번에 간 것이 같다
    A = np.array([[-0.1, 1.0], [-1.0, -0.1]])
    assert np.allclose(expm(A * 0.7) @ expm(A * 0.3), expm(A * 1.0))
    assert correct("irregular-sampling") == [0, 1]
