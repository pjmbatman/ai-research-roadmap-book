"""19.7 퀴즈(content/19-generative-and-foundation-models/07-evaluating-generative-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/07-evaluating-generative-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
MODES = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def real(n, std=0.4, use=4):
    return MODES[rng.integers(use, size=n)] + std * rng.normal(size=(n, 2))


def dists(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def pr(real_x, fake_x, k=3):
    r_real = np.sort(dists(real_x, real_x), 1)[:, k]
    r_fake = np.sort(dists(fake_x, fake_x), 1)[:, k]
    return (dists(fake_x, real_x) <= r_real).any(1).mean(), (dists(real_x, fake_x) <= r_fake).any(1).mean()


def frechet(a, b):
    Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return np.sum((a.mean(0) - b.mean(0)) ** 2) + np.trace(Sa + Sb - 2 * sqrtm(Sa @ Sb).real)


def test_precision_recall_mode_drop():
    test = real(500)
    p_drop, r_drop = pr(test, real(500, use=3))
    p_blur, r_blur = pr(test, real(500, std=0.8))
    assert p_drop > 0.95 and r_drop < 0.8                     # 모드 누락: precision 높고 recall 낮음
    assert p_blur < 0.8 and r_blur > 0.95                     # 흐림: 반대
    assert correct("precision-recall-failure") == [1]


def test_likelihood_mixture_bound():
    bound = np.log(100)
    d = 8
    x = rng.normal(size=(5000, d))
    lp = multivariate_normal(np.zeros(d), np.eye(d)).logpdf(x)
    lj = multivariate_normal(np.zeros(d), 25 * np.eye(d)).logpdf(x)
    lq = np.logaddexp(np.log(0.01) + lp, np.log(0.99) + lj)
    assert np.all(lp - lq <= bound + 1e-9)                    # 표본마다 log 100 이하로만 줄어든다
    assert abs(bound - Q["likelihood-mixture-bound"]["answer"]) <= Q["likelihood-mixture-bound"]["tolerance"]


def test_mmd_unbiased_can_be_negative():
    def mmd2(x, y, bw=1.0):
        k = lambda a, b: np.exp(-dists(a, b) ** 2 / (2 * bw**2))
        m = len(x)
        return (k(x, x).sum() - m) / (m * (m - 1)) + (k(y, y).sum() - m) / (m * (m - 1)) - 2 * k(x, y).mean()
    vals = [mmd2(rng.normal(size=(30, 2)), rng.normal(size=(30, 2))) for _ in range(20)]
    assert min(vals) < 0                                      # p 값과 MMD² 크기는 다른 양
    assert correct("mmd-power") == [1]


def test_fid_sample_size_and_memorization():
    W = rng.normal(size=(2, 32))
    feat = lambda x: np.maximum(0, x @ W + 0.5)
    small = np.mean([frechet(feat(real(50)), feat(real(50))) for _ in range(5)])
    big = np.mean([frechet(feat(real(500)), feat(real(500))) for _ in range(5)])
    assert small > 3 * big                                    # 표본 수에 따른 편향
    train = real(500)
    copy = train + 0.02 * rng.normal(size=(500, 2))
    assert frechet(feat(copy), feat(train)) < 0.01            # 학습 자료 기준이면 복사가 최고점
    assert correct("fid-memorization") == [0, 1]


def test_shuffle_keeps_marginals_breaks_autocorr():
    t = np.linspace(0, 1, 64)
    om = 2 * np.pi * rng.uniform(3, 8, 1000)
    X = np.exp(-rng.uniform(0.5, 4, 1000)[:, None] * t) * np.cos(om[:, None] * t + rng.uniform(0, 2 * np.pi, 1000)[:, None])
    Xs = np.stack([c[rng.permutation(1000)] for c in X.T], 1)
    assert np.allclose(np.sort(Xs, 0), np.sort(X, 0))         # 시점별 분포는 정확히 같다
    lag1 = lambda Z: np.mean(np.sum(Z[:, 1:] * Z[:, :-1], 1) / np.sum(Z**2, 1))
    assert lag1(X) > 0.7 and abs(lag1(Xs)) < 0.1
    assert correct("marginal-stats-shuffle") == [1]


def test_tstr_design():
    assert correct("tstr-baseline") == [0, 1]
