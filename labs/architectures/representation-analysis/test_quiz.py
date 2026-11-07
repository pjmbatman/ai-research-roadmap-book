"""18.8 퀴즈(content/18-architectures/08-representation-analysis/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/08-representation-analysis/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def _data(seed=0, N=1200, T=64):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    omega = rng.uniform(2 * np.pi * 3, 2 * np.pi * 8, N)
    gamma = rng.uniform(0.5, 4.0, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    X = np.exp(-gamma[:, None] * t) * np.cos(omega[:, None] * t + phi[:, None]) + 0.1 * rng.normal(size=(N, T))
    return X, omega, rng


def _probe_r2(Z, y, tr, te, lam=1e-2):
    Zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-8)
    w = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * len(tr) * np.eye(Z.shape[1]), Zs[tr].T @ (y[tr] - y[tr].mean()))
    return 1 - ((y[te] - Zs[te] @ w - y[tr].mean()) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()


def _cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))


def test_probe_random_baseline():
    X, omega, rng = _data()
    tr, te = np.arange(900), np.arange(900, 1200)
    H = np.maximum(0, X @ rng.normal(size=(64, 256)) / 8)            # 학습하지 않은 무작위 층
    assert _probe_r2(X, omega, tr, te) < 0.1                         # 입력에서는 선형으로 안 읽힘
    assert _probe_r2(H, omega, tr, te) > 0.85                        # 무작위 층에서는 높게 읽힘
    assert correct("probe-random-baseline") == [0]


def test_probe_not_usage():
    # 정보가 있어도 출력이 쓰지 않는 예: 출력 가중치가 그 방향과 직교
    rng = np.random.default_rng(1)
    lab = rng.integers(2, size=500)
    Z = np.column_stack([lab + 0.01 * rng.normal(size=500), rng.normal(size=500)])
    w_out = np.array([0.0, 1.0])                                     # 출력은 둘째 성분만 쓴다
    assert ((Z[:, 0] > 0.5) == lab).mean() > 0.99                    # probe로는 완벽히 읽힘
    Z_ablate = Z.copy(); Z_ablate[:, 0] = 0                          # 그 방향을 제거하는 개입
    assert np.allclose(Z @ w_out, Z_ablate @ w_out)                  # 출력 불변: 의존하지 않음
    assert correct("probe-not-usage") == [0]


def test_cka_invariances():
    rng = np.random.default_rng(2)
    A = np.maximum(0, rng.normal(size=(800, 64)))
    Q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    M = rng.normal(size=(64, 64))
    assert np.isclose(_cka(A, A @ Q), 1.0) and np.isclose(_cka(A, 3 * A), 1.0)
    assert not np.isclose(_cka(A, A @ M), 1.0, atol=1e-3)
    a, b = rng.normal(size=(800, 64)), rng.normal(size=(800, 64))
    big = 10 * rng.normal(size=(800, 1))
    assert _cka(a, b) > 0.01                                         # 무관해도 0이 아님
    assert _cka(np.hstack([a, big]), np.hstack([b, big])) > 0.9      # 공통 큰 분산 방향이 지배
    assert correct("cka-invariances") == [0, 1]


def test_nn_unscaled_feature():
    X, omega, rng = _data(seed=3)
    tr, te = np.arange(900), np.arange(900, 1200)
    temp = rng.normal(298_000, 2_000, size=(1200, 1))

    def nn_err(E):
        d2 = ((E[te][:, None] - E[tr][None]) ** 2).sum(-1)
        return np.abs(omega[te] - omega[tr][d2.argmin(1)]).mean()

    E = np.hstack([X, temp])
    Es = (E - E[tr].mean(0)) / E[tr].std(0)
    assert nn_err(E) > 3 * nn_err(X)                                 # 큰 척도의 열이 거리를 지배
    assert nn_err(Es) < 1.5 * nn_err(X)                              # 표준화하면 회복
    assert correct("nn-unscaled-feature") == [0]


def test_pca_explained_variance():
    ev = np.array([6.0, 2.0, 1.0, 1.0])
    q = Q["pca-explained-variance"]
    assert abs(ev[:2].sum() / ev.sum() - q["answer"]) <= q["tolerance"]


def test_probe_capacity():
    assert correct("probe-capacity") == [0]
