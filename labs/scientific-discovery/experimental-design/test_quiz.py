"""25.3 퀴즈(content/25-scientific-discovery/03-experimental-design/quiz.toml) 정답과 본문 수치를 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/25-scientific-discovery/03-experimental-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_eig_scalar_repeats():
    Phi = np.ones((3, 1))                                       # y_i = θ + ε, 3번
    eig = 0.5 * np.linalg.slogdet(np.eye(3) + Phi @ Phi.T / 1.0**2)[1]
    post_var = 1 / (1 + 3)
    assert np.isclose(eig, 0.5 * np.log(1 / post_var))
    assert abs(eig - Q["eig-scalar-repeats"]["answer"]) <= Q["eig-scalar-repeats"]["tolerance"]


def test_eig_decay_timing_extremes():
    rng = np.random.default_rng(0)
    k = np.exp(rng.normal(0.0, 0.5, 5000))                      # 사전 표본 [1/s]
    spread = {t: np.std(np.exp(-k * t)) for t in (0.01, 1.0, 20.0)}
    assert spread[1.0] > 10 * spread[0.01]                      # 이른 시각: 예측이 거의 같다
    assert spread[1.0] > 10 * spread[20.0]                      # 늦은 시각: 모두 0 근처
    assert correct("eig-decay-timing") == [0]


def test_d_optimal_quadratic():
    def logdet(x):
        X = np.stack([np.ones_like(x), x, x**2], 1)
        return np.linalg.slogdet(X.T @ X)[1]
    best = logdet(np.array([-1, -1, 0, 0, 1, 1.0]))
    assert np.isclose(best, np.log(32))                         # 본문 3.466
    assert best > logdet(np.linspace(-1, 1, 6))
    rng = np.random.default_rng(1)
    for _ in range(2000):                                       # 무작위 설계는 더 좋을 수 없다
        assert logdet(rng.uniform(-1, 1, 6)) <= best + 1e-9
    assert correct("d-optimal-replicates") == [0]


def test_linear_eig_designs_in_text():
    def eig(x):
        P = np.stack([np.ones_like(x), x], 1)
        return 0.5 * np.linalg.slogdet(np.eye(len(x)) + P @ P.T / 0.25)[1]
    assert np.isclose(eig(np.zeros(4)), 0.5 * np.log(17))       # 한 점에 4번: 1.417
    assert eig(np.array([-1, -1, 1, 1.0])) > eig(np.array([-1, -1 / 3, 1 / 3, 1]))


def test_choice_keys():
    assert correct("uncertainty-sampling-trap") == [0]
    assert correct("fair-comparison") == [0, 1, 2]
    assert correct("al-vs-bo") == [0]
