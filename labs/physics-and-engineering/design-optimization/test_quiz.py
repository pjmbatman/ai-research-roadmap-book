"""26.4 퀴즈(content/26-physics-and-engineering/04-design-optimization/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/26-physics-and-engineering/04-design-optimization/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


class CountingSolver:
    def __init__(self):
        self.n = 0

    def __call__(self, A, b):
        self.n += 1
        return np.linalg.solve(A, b)


def system(n):
    A = rng.normal(size=(n, n)) / np.sqrt(n) + 3 * np.eye(n)
    dA = [np.outer(rng.normal(size=n), rng.normal(size=n)) * 0.1 for _ in range(4)]   # ∂K/∂p_i (설계 변수 4개)
    return A, dA, rng.normal(size=n)


def test_adjoint_cost():
    n_outputs = 3                                     # 목적 1 + 제약 2
    A, dA, f = system(30)
    C = rng.normal(size=(n_outputs, 30))
    solve = CountingSolver()
    u = solve(A, f)
    grads = []
    for c in C:
        lam = solve(A.T, c)
        grads.append([-lam @ (d @ u) for d in dA])
    assert solve.n == Q["adjoint-cost"]["answer"] == 1 + n_outputs
    eps = 1e-6                                        # 유한차분으로 기울기 확인
    for i, d in enumerate(dA):
        fd = (C @ np.linalg.solve(A + eps * d, f) - C @ u) / eps
        assert np.allclose(fd, np.array(grads)[:, i], rtol=1e-4, atol=1e-6)


def test_fd_central_count():
    n = 100
    calls = 0

    def J(p):
        nonlocal calls
        calls += 1
        return np.sum(p**2)

    p, eps = rng.normal(size=n), 1e-5
    g = np.array([(J(p + eps * e) - J(p - eps * e)) / (2 * eps) for e in np.eye(n)])
    assert calls == Q["fd-central-count"]["answer"]
    assert np.allclose(g, 2 * p, atol=1e-6)


def test_adjoint_not_always():
    n_vars, n_outputs = 3, 500
    forward_cost = 1 + n_vars                         # 상태 + 변수당 민감도
    adjoint_cost = 1 + n_outputs
    assert forward_cost < adjoint_cost
    assert correct("adjoint-not-always") == [0]


def test_beam_pareto_on_bounds():
    # 같은 질량(b h 일정)에서 처짐은 h가 클수록 작다: 높이 상한 쪽으로 간다
    area = 0.001
    h = np.linspace(0.02, 0.10, 9)
    b = area / h
    defl = 1 / (b * h**3)
    assert np.all(np.diff(defl) < 0)
    assert correct("pareto-on-bounds") == [0]


def test_choice_keys():
    assert correct("fake-optimum") == [0]
    assert correct("generative-design-report") == [0, 1]
