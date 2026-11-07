"""18.7 퀴즈(content/18-architectures/07-time-series-deep-learning/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/07-time-series-deep-learning/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def _series(seed=0):
    rng = np.random.default_rng(seed)
    n = 24 * 120
    t = np.arange(n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.97 * noise[i - 1] + 0.4 * rng.normal()
    return 10 + 0.003 * t + 5 * np.sin(2 * np.pi * (t - 9) / 24) + noise, rng


def _windows(x, L, H):
    s = np.arange(len(x) - L - H + 1)
    return np.stack([x[i:i + L] for i in s]), np.stack([x[i + L:i + L + H] for i in s]), s


def _ridge(X, Y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ W


def test_patch_count():
    L, P, S = 336, 16, 8
    x = np.arange(L)
    n = len([s for s in range(0, L - P + 1, S)])                       # 실제로 잘라 센 수
    assert n == (L - P) // S + 1 == Q["patch-count"]["answer"] == 41


def test_known_future_covariates():
    assert correct("known-future-covariates") == [0, 1, 2]


def test_random_split_windows():
    x, rng = _series()
    L, H = 48, 24
    X, Y, s = _windows(x, L, H)
    idx = rng.permutation(len(X)); tr, te = idx[:int(0.8 * len(X))], idx[int(0.8 * len(X)):]
    covered = np.zeros(len(x), bool)
    for i in s[tr]:
        covered[i:i + L + H] = True
    frac = np.mean([covered[i + L:i + L + H].all() for i in s[te]])
    assert frac > 0.99                                                 # test 정답 구간이 거의 다 노출
    assert correct("random-split-windows") == [0]


def test_mase_interpretation():
    # MASE = 1.25면 test MAE가 척도의 1.25배
    scale, mae = 0.8, 1.0
    assert np.isclose(mae / scale, 1.25) and mae > scale
    assert correct("mase-interpretation") == [0]


def test_recursive_vs_direct():
    x, _ = _series()
    L, H, cut = 48, 24, 24 * 96
    X, Y, s = _windows(x, L, H)
    tr, te = s + L + H <= cut, s + L >= cut
    Ls = 6
    a, c, Ytr, Yte = X[tr][:, -Ls:], X[te][:, -Ls:], Y[tr], Y[te]
    one = _ridge(a, Ytr[:, :1])
    w, rec = c.copy(), []
    for _ in range(H):
        y = one(w); rec.append(y[:, 0]); w = np.hstack([w[:, 1:], y])
    rec = np.abs(np.stack(rec, 1) - Yte).mean(0)
    direct = np.abs(_ridge(a, Ytr)(c) - Yte).mean(0)
    assert np.isclose(rec[0], direct[0], atol=1e-6)                    # h=1은 같다
    assert rec[11] > 2 * direct[11]                                    # h=12에서 recursive가 크게 나쁘다
    assert correct("recursive-vs-direct") == [0]


def test_normalization_leak():
    x, _ = _series()
    cut = 24 * 96
    assert x[:cut].mean() < x.mean()                                   # 추세 때문에 전체 평균에 미래가 섞인다
    assert correct("normalization-leak") == [0]
