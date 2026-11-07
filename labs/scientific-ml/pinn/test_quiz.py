"""24.2 퀴즈(content/24-scientific-ml/02-pinn/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/02-pinn/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

f = lambda x: np.pi**2 * np.sin(np.pi * x) + 0.2 * (4 * np.pi) ** 2 * np.sin(4 * np.pi * x)
u_true = lambda x: np.sin(np.pi * x) + 0.2 * np.sin(4 * np.pi * x)
rng = np.random.default_rng(0)
W, B = rng.normal(0, 8, 40), rng.uniform(-8, 8, 40)             # 본문 셀 2와 같은 random feature
phi = lambda x: np.tanh(np.outer(x, W) + B)


def phi_xx(x):
    t = np.tanh(np.outer(x, W) + B)
    return -2 * t * (1 - t**2) * W**2


def system(xc, lam):
    A = np.vstack([-phi_xx(xc) / np.sqrt(len(xc)), np.sqrt(lam / 2) * phi(np.array([0.0, 1.0]))])
    return A, np.concatenate([f(xc) / np.sqrt(len(xc)), np.zeros(2)])


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_pinn_loss_terms():
    # 잔차만으로는 해가 정해지지 않는다: 0과 정확해 모두 잔차 0
    x, t = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    for u_t, u_xx in [(0 * x, 0 * x), (-0.1 * np.pi**2 * np.exp(-0.1 * np.pi**2 * t) * np.sin(np.pi * x),
                                          -np.pi**2 * np.exp(-0.1 * np.pi**2 * t) * np.sin(np.pi * x))]:
        assert np.allclose(u_t - 0.1 * u_xx, 0)
    assert correct("pinn-loss-terms") == [0, 1, 2]


def test_hard_constraint_ansatz():
    N = lambda x: np.cos(3 * x) + 2.0                                 # 아무 함수
    u = lambda x: x * (1 - x) * N(x)
    assert u(0.0) == 0 and u(1.0) == 0
    assert correct("hard-constraint-ansatz") == [0]


def test_residual_frequency_ratio():
    x = np.linspace(0, 1, 100001)
    e = 0.01 * np.sin(10 * np.pi * x)
    r = 0.01 * (10 * np.pi) ** 2 * np.sin(10 * np.pi * x)
    ratio = np.sqrt(np.mean(r**2)) / np.sqrt(np.mean(e**2))
    q = Q["residual-frequency-ratio"]
    assert abs(ratio - q["answer"]) <= q["tolerance"]


def test_zero_residual_wrong_solution():
    x = np.linspace(0, 1, 201)
    u = u_true(x) + 0.5 * x
    u_xx = -f(x) + 0.0                                                # 0.5x의 2계 도함수는 0
    assert np.all(-u_xx - f(x) == 0) and u[-1] == 0.5 + u_true(1.0)
    assert correct("zero-residual-wrong-solution") == [1]


def test_collocation_overfit():
    xt = np.linspace(0, 1, 1001)
    c8 = np.linalg.lstsq(*system(np.linspace(0, 1, 8), 1.0), rcond=None)[0]
    c100 = np.linalg.lstsq(*system(np.linspace(0, 1, 100), 1.0), rcond=None)[0]
    xc = np.linspace(0, 1, 8)
    assert np.sqrt(np.mean((-phi_xx(xc) @ c8 - f(xc)) ** 2)) < 1e-8
    assert np.sqrt(np.mean((-phi_xx(xt) @ c8 - f(xt)) ** 2)) > 1
    assert np.abs(phi(xt) @ c8 - u_true(xt)).max() > 1
    assert np.abs(phi(xt) @ c100 - u_true(xt)).max() < 1e-5             # 표현력은 충분하다
    assert correct("collocation-overfit") == [0]


def test_bc_weight_imbalance():
    xt, xc = np.linspace(0, 1, 1001), np.linspace(0, 1, 64)
    errs, res = {}, {}
    for lam in [1.0, 1e2, 1e4]:
        A, rhs = system(xc, lam)
        lr = 1.0 / np.linalg.eigvalsh(A.T @ A).max()
        c = np.zeros(40)
        for _ in range(20000):
            c -= lr * A.T @ (A @ c - rhs)
        errs[lam] = np.abs(phi(xt) @ c - u_true(xt)).max()
        res[lam] = np.sqrt(np.mean((-phi_xx(xc) @ c - f(xc)) ** 2))
    assert errs[1e2] < errs[1.0] < errs[1e4]
    assert res[1e4] > 3 * res[1e2]
    assert correct("bc-weight-imbalance") == [0]


def test_pinn_vs_solver():
    lam1 = (4 / 0.02**2) * np.sin(np.pi * 0.02 / 2) ** 2
    fd_err = abs(np.exp(-0.1 * (lam1 - np.pi**2)) - 1)
    assert 2.5e-4 < fd_err < 3.5e-4                                   # 본문 셀 5의 비교값
    assert correct("pinn-vs-solver") == [1]
