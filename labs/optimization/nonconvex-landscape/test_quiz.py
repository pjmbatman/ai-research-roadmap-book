"""10.6 퀴즈(content/10-optimization/06-nonconvex-landscape/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/06-nonconvex-landscape/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_saddle_plateau():
    # 안장점 근처: 정체 구간의 최소 gradient가 작고, 결국 벗어나 손실이 크게 내려간다
    grad = lambda p: np.array([p[0], -p[1] + p[1] ** 3])
    f = lambda p: p[0] ** 2 / 2 - p[1] ** 2 / 2 + p[1] ** 4 / 4
    p, gmin = np.array([1.0, 1e-9]), np.inf
    for _ in range(400):
        p = p - 0.1 * grad(p)
        gmin = min(gmin, np.linalg.norm(grad(p)))
    assert gmin < 1e-4 and f(p) < -0.24                     # 최솟값 -1/4 근처까지 내려감
    assert correct("saddle-plateau") == [1]


def test_saddle_escape_steps():
    k_lin = np.log(0.5 / 1e-6) / np.log(1.1)
    assert abs(k_lin - Q["saddle-escape-steps"]["answer"]) <= 0.5
    y, k = 1e-6, 0
    while abs(y) < 0.5:                                      # 4차 항을 포함한 실제 반복
        y, k = y - 0.1 * (-y + y**3), k + 1
    assert abs(k - Q["saddle-escape-steps"]["answer"]) <= Q["saddle-escape-steps"]["tolerance"]


def test_sharpness_reparametrization():
    x = np.linspace(-1, 1, 41)
    y = 2.0 * np.maximum(0, x)
    L = lambda w: 0.5 * np.mean((w[1] * np.maximum(0, w[0] * x) - y) ** 2)
    def top(w):                                              # 해에서의 정확한 Hessian(잔차 0): J^T J / N
        J = np.stack([w[1] * x * (x > 0), np.maximum(0, w[0] * x)], axis=1)
        return np.linalg.eigvalsh(J.T @ J / len(x))[-1]
    a, b = np.array([1.0, 2.0]), np.array([10.0, 0.2])
    assert L(a) < 1e-20 and L(b) < 1e-20
    assert top(b) > 10 * top(a)
    assert correct("sharpness-reparametrization") == [0]


def test_sam_objective():
    L1 = lambda w: 0.05 * w**2 + 1 - np.exp(-((w + 1) / 0.15) ** 2) - 0.8 * np.exp(-((w - 1.5) / 0.8) ** 2)
    grid = np.linspace(-3, 4, 1401)
    plain = L1(grid)
    sam = np.array([L1(np.linspace(w - 0.1, w + 0.1, 41)).max() for w in grid])
    assert abs(grid[np.argmin(plain)] + 1) < 0.05            # 원래 최소: 날카로운 골
    assert abs(grid[np.argmin(sam)] - 1.44) < 0.05           # SAM 최소: 평평한 골
    assert correct("sam-objective") == [0]


def test_edge_of_stability_threshold():
    assert 2 / 0.01 == Q["edge-of-stability-threshold"]["answer"]
    grad = lambda p: (p[0] * p[1] - 1) * np.array([p[1], p[0]])
    for lr in [0.15, 0.18, 0.2]:
        p = np.array([0.2, 4.0])
        for _ in range(5000):
            p = p - lr * grad(p)
        assert abs(p[0] * p[1] - 1) < 1e-8
        assert p[0] ** 2 + p[1] ** 2 < 2 / lr                # 최솟점의 sharpness = u^2 + v^2


def test_permutation_barrier():
    rng = np.random.default_rng(0)
    H, x = 8, np.linspace(-2, 2, 64)
    y = np.sin(3 * x)
    def loss(th):
        a = np.tanh(np.outer(x, th[:H]) + th[H:2 * H])
        return 0.5 * np.mean((a @ th[2 * H:3 * H] + th[3 * H] - y) ** 2)
    th = rng.normal(size=3 * H + 1)
    perm = rng.permutation(H)
    thp = th[np.concatenate([perm, H + perm, 2 * H + perm, [3 * H]])]
    assert np.isclose(loss(th), loss(thp))                    # 같은 함수
    mid = loss(0.5 * (th + thp))
    assert not np.isclose(mid, loss(th))                      # 중간점은 다른 함수
    assert correct("permutation-barrier") == [0]


def test_slice_interpretation():
    # 음의 곡률 방향 하나를 무작위 2차원 단면이 거의 보지 못한다
    rng = np.random.default_rng(0)
    n = 10000
    lam = np.ones(n)
    lam[0] = -5.0                                            # 첫 좌표축만 음의 곡률
    d1, d2 = rng.normal(size=n), rng.normal(size=n)
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    curv = [d @ (lam * d) for d in (d1, d2)]                  # 단면 방향의 곡률
    assert min(curv) > 0.9                                    # 단면은 볼록해 보인다
    assert correct("slice-interpretation") == [0, 1]
