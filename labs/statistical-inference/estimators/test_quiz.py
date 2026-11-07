"""12.1 퀴즈(content/12-statistical-inference/01-estimators/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/01-estimators/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_estimator_vs_estimate():
    xbar = rng.normal(2.0, 0.5, size=(5000, 25)).mean(axis=1)      # 추정량은 반복마다 다른 값
    assert xbar.std() > 0.05
    assert correct("estimator-vs-estimate") == [0]


def test_what_shrinks_with_n():
    for n in (10, 1000):
        X = rng.normal(0.0, 2.0, size=(2000, n))
        assert abs(X.std(axis=1, ddof=1).mean() - 2.0) < 0.15          # s → σ
        assert abs(X.mean(axis=1).std() - 2.0 / np.sqrt(n)) < 0.05      # SE = σ/√n
    assert correct("what-shrinks-with-n") == [0, 1]


def test_mse_from_bias_variance():
    assert math.isclose(Q["mse-from-bias-variance"]["answer"], 0.3**2 + 0.16)


def test_unbiased_not_lowest_mse():
    n = 5
    X = rng.normal(size=(200000, n))
    ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    mse = {d: np.mean((ss / d - 1.0) ** 2) for d in (n - 1, n, n + 1)}
    assert mse[n + 1] < mse[n] < mse[n - 1]
    assert correct("unbiased-not-lowest-mse") == [1]


def test_unbiased_but_inconsistent():
    for n in (10, 1000):
        X1 = rng.normal(1.0, 1.0, size=(20000, n))[:, 0]
        assert abs(X1.mean() - 1.0) < 0.03 and abs(X1.std() - 1.0) < 0.03
    assert correct("unbiased-but-inconsistent") == [0]


def test_median_sample_size():
    X = rng.normal(size=(20000, 401))
    ratio = np.median(X, axis=1).var() / X.mean(axis=1).var()    # 점근적으로 π/2
    assert abs(ratio - math.pi / 2) < 0.06
    assert Q["median-sample-size"]["answer"] == round(100 * math.pi / 2)


def test_poisson_crlb():
    lam, n = 4.0, 100
    xbar = rng.poisson(lam, size=(50000, n)).mean(axis=1)
    assert abs(xbar.std() - math.sqrt(lam / n)) < 0.005           # 표본 평균이 하한을 달성한다
    assert math.isclose(Q["poisson-crlb"]["answer"], math.sqrt(lam / n))
