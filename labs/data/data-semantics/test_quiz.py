"""15.1 퀴즈(content/15-data/01-data-semantics/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/15-data/01-data-semantics/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_unit_of_independence():
    cell_id = np.repeat(np.arange(8), 30)
    assert len(cell_id) == 240
    assert len(np.unique(cell_id)) == Q["unit-of-independence"]["answer"] == 8


def test_pseudoreplication_se_underestimates():
    rng = np.random.default_rng(1)
    G, m, sg, se = 8, 30, 0.05, 0.005
    naive, truth = [], []
    for _ in range(400):
        y = (rng.normal(0, sg, G)[:, None] + rng.normal(0, se, (G, m))).ravel()
        naive.append(y.std(ddof=1) / np.sqrt(G * m))
        truth.append(y.mean())
    assert np.mean(naive) < 0.5 * np.std(truth)        # 행 기반 표준오차가 실제 변동보다 훨씬 작다
    assert correct("pseudoreplication-se") == [0]


def test_computed_label_bias():
    rng = np.random.default_rng(2)
    biases = []
    for _ in range(50):
        meas = 2.0 + 0.02 + rng.normal(0, 0.05, 3600)
        biases.append(meas.sum() / 3600 - 2.0)
    assert abs(np.mean(biases) - 0.02) < 0.001         # 잡음은 상쇄되고 offset은 0.02 Ah로 남는다
    assert correct("computed-label-bias") == [0]


def test_label_noise_floor():
    rng = np.random.default_rng(3)
    eta = rng.normal(0, 0.05, 200_000)
    rmse = np.sqrt(np.mean(eta**2))
    q = Q["label-noise-floor"]
    assert abs(rmse - q["answer"]) <= q["tolerance"]


def test_data_dictionary_fields():
    assert correct("data-dictionary-fields") == [0, 1, 2]


def test_raw_is_immutable():
    assert correct("raw-is-immutable") == [0]
