"""26.2 퀴즈(content/26-physics-and-engineering/02-surrogate-modeling/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/26-physics-and-engineering/02-surrogate-modeling/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_pod_energy_truncation():
    s = np.array([6.0, 3.0, 2.0, 1.0])
    assert np.isclose(100 * np.sum(s[:2] ** 2) / np.sum(s**2), Q["pod-energy-truncation"]["answer"])


def test_pod_linear_invariants():
    M, N = 12, 40
    X = np.abs(rng.normal(size=(M, N))) + 0.1                   # u >= 0
    X[:, 0] = 1.0                                               # 경계값 1
    X[:, 1:] *= (5.0 - 1.0) / X[:, 1:].sum(1, keepdims=True)    # 질량(성분 합) 5
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    pred = mean + 50 * rng.normal(size=4) @ Vt[:4]              # 학습 범위 밖의 큰 계수
    assert np.isclose(pred.sum(), 5.0)                          # 질량 보존
    assert np.isclose(pred[0], 1.0)                             # 경계값 보존
    assert pred.min() < 0                                       # 양수성은 깨진다
    assert pred.max() > X.max()                                 # 최댓값 조건도 깨진다
    assert correct("pod-linear-invariants") == [0, 1]


def test_fair_speedup():
    assert np.isclose(1000 / 50, 20)
    assert correct("fair-speedup") == [0]


def test_surrogate_break_even():
    fixed_min = 200 * 10 + 5 * 60
    assert fixed_min / 10 == Q["surrogate-break-even"]["answer"]


def test_choice_keys():
    assert correct("surrogate-error-decomposition") == [0]
    assert correct("physics-check-not-sufficient") == [0]
