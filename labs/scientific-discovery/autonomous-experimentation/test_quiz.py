"""25.4 퀴즈(content/25-scientific-discovery/04-autonomous-experimentation/quiz.toml) 정답과 본문 수치를 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import norm

QUIZ = Path(__file__).parents[3] / "content/25-scientific-discovery/04-autonomous-experimentation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_constraint_probability():
    val = 0.20 * norm.cdf((4.0 - 3.0) / 0.5)
    q = Q["constraint-probability"]
    assert abs(val - q["answer"]) <= q["tolerance"]
    assert np.isclose(norm.cdf(2), 0.977, atol=5e-4)
    assert np.isclose(norm.cdf((4.0 - 3.9) / 0.5), 0.58, atol=5e-3)   # 해설의 예


def test_max_of_noisy_measurements_is_biased_up():
    rng = np.random.default_rng(0)
    best = (0.8 + 0.02 * rng.normal(size=(20000, 10))).max(1)          # 같은 조건을 10번 잰 최댓값
    assert best.mean() > 0.8 + 0.02                                     # 위로 치우친다
    assert correct("max-measurement-bias") == [0]


def test_kriging_believer_spreads_batch():
    # GP 사후 분산은 관측값과 무관하다: 가상 측정을 넣으면 그 근처의 분산이 줄어든다
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None]) ** 2 / 0.1**2)
    X, xs = np.array([0.0, 1.0]), np.linspace(0, 1, 101)
    def post_sd(X):
        L = np.linalg.cholesky(k(X, X) + 1e-4 * np.eye(len(X)))
        v = np.linalg.solve(L, k(X, xs))
        return np.sqrt(np.clip(1 - (v**2).sum(0), 0, None))
    sd0 = post_sd(X)
    j = int(np.argmax(sd0))
    sd1 = post_sd(np.append(X, xs[j]))
    assert sd1[j] < 0.05 * sd0[j]                                       # 고른 점의 불확실성이 사라진다
    assert abs(xs[int(np.argmax(sd1))] - xs[j]) > 0.1                   # 다음 선택은 다른 곳
    assert correct("kriging-believer") == [0]


def test_choice_keys():
    assert correct("saturation-feedback") == [0]
    assert correct("drift-detection") == [0]
    assert correct("loop-record") == [0, 1, 2]
