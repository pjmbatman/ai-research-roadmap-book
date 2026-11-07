"""21.1 퀴즈(content/21-reliability/01-uncertainty-quantification/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/21-reliability/01-uncertainty-quantification/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_more_data_reduces_epistemic_only():
    def ensemble(n):
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.3, n)                         # 잡음 sd 0.3 고정
        mus, sds = [], []
        for _ in range(30):
            i = rng.integers(0, n, n)
            c = np.polyfit(x[i], y[i], 1)
            mus.append(np.polyval(c, 0.5))
            sds.append(np.std(y[i] - np.polyval(c, x[i])))
        return np.std(mus), np.mean(sds)
    epi_small, alea_small = ensemble(50)
    epi_big, alea_big = ensemble(500)
    assert epi_big < 0.5 * epi_small                               # epistemic은 줄어든다
    assert abs(alea_big - 0.3) < 0.03 and abs(alea_small - 0.3) < 0.08   # aleatoric은 0.3 근처에 머문다
    assert correct("more-data-reduces-which") == [0]


def test_total_variance():
    mu, var = np.array([1.0, 3.0]), np.array([2.0, 4.0])
    total = var.mean() + mu.var()
    comp = rng.integers(0, 2, 400_000)
    samples = mu[comp] + np.sqrt(var[comp]) * rng.normal(size=comp.size)
    assert abs(samples.var() - total) < 0.05
    assert Q["total-variance-numeric"]["answer"] == total == 4


def test_logistic_far_point_saturates():
    w, b = np.array([1.5, 0.2]), -0.1                              # 임의의 선형 경계
    p = lambda q: 1 / (1 + np.exp(-(w @ q + b)))
    assert p(np.array([20.0, 30.0])) > 0.999
    assert correct("softmax-confidence") == [0]


def test_ensemble_diversity_source():
    assert correct("ensemble-diversity-source") == [0, 1]


def test_heteroscedastic_mean_gradient_scales_with_inverse_variance():
    y, mu = 1.0, 0.0
    grad = lambda s2: -(y - mu) / s2                               # d NLL / d mu
    assert np.isclose(grad(4.0) / grad(1.0), 0.25)
    # 잔차가 큰 점에서 NLL을 최소화하는 σ²는 잔차 제곱(=1)으로 커진다
    s2 = np.linspace(0.05, 3, 2000)
    nll = 0.5 * np.log(s2) + (y - mu) ** 2 / (2 * s2)
    assert abs(s2[np.argmin(nll)] - 1.0) < 0.01
    assert correct("heteroscedastic-gradient") == [0]
