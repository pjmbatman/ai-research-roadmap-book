"""25.5 퀴즈(content/25-scientific-discovery/05-scientific-foundation-models/quiz.toml) 정답을 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/25-scientific-discovery/05-scientific-foundation-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_label_efficiency():
    raw = {10: 0.722, 30: 0.870, 100: 0.913}                 # 본문 셀 1의 출력
    target = 0.874
    n = min(raw, key=lambda k: abs(raw[k] - target))
    q = Q["label-efficiency"]
    assert n == q["answer"] == 30
    s = q["slider"]
    assert s["min"] <= q["answer"] <= s["max"] and (q["answer"] - s["min"]) % s["step"] == 0


def test_split_leakage_toy():
    # 형제 시료가 학습에 있으면 1-최근접 이웃이 형제를 베껴 점수가 오른다
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(40, 4))
    X = np.repeat(centers, 5, 0) + 0.05 * rng.normal(size=(200, 4))
    fam = np.repeat(np.arange(40), 5)
    y = rng.normal(size=40)[fam]
    def nn_r2(test):
        d = np.linalg.norm(X[test][:, None] - X[~test][None], axis=-1)
        pred = y[~test][d.argmin(1)]
        return 1 - np.mean((pred - y[test]) ** 2) / np.var(y[test])
    random_test = rng.random(200) < 0.25
    family_test = np.isin(fam, np.arange(10))
    assert nn_r2(random_test) > 0.9 > nn_r2(family_test)
    assert correct("split-leakage") == [0]


def test_choice_keys():
    assert correct("transfer-small-data") == [0]
    assert correct("pretraining-overlap") == [0, 1, 2]
    assert correct("agent-verification") == [0]
    assert correct("dual-use-scope") == [0]
