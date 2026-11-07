"""12.5 퀴즈(content/12-statistical-inference/05-comparing-models/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/05-comparing-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_sd_vs_se():
    assert abs(0.011 / math.sqrt(10) - 0.0035) < 1e-4
    assert correct("sd-vs-se") == [0]


def test_se_from_sd():
    assert math.isclose(Q["se-from-sd"]["answer"], 0.012 / math.sqrt(9))


def test_single_seed_claim():
    assert correct("single-seed-claim") == [1]


def test_why_pairing():
    common = rng.normal(0, 1, 5000)                       # 샘플의 난이도: 두 모델에 공통
    ea, eb = common + rng.normal(0, 0.5, 5000), common - 0.05 + rng.normal(0, 0.5, 5000)
    unpaired = np.sqrt(ea.var(ddof=1) / 5000 + eb.var(ddof=1) / 5000)
    paired = (eb - ea).std(ddof=1) / np.sqrt(5000)
    assert np.corrcoef(ea, eb)[0, 1] > 0.7 and paired < unpaired / 2
    assert correct("why-pairing") == [0]


def test_repeated_cv_se():
    # 같은 데이터셋에서 나온 상관된 점수: 공통 성분 + 반복마다의 작은 변동
    for reps in (10, 100):
        dataset_effect = rng.normal(0, 0.04, size=(2000, 1))
        s = 0.4 + dataset_effect + rng.normal(0, 0.01, size=(2000, reps))
        naive = (s.std(axis=1, ddof=1) / np.sqrt(reps)).mean()
        assert naive < 0.004 and s.mean(axis=1).std() > 0.035
    assert correct("repeated-cv-se") == [0]


def test_critical_difference():
    cd = 2.343 * math.sqrt(3 * 4 / (6 * 12))
    assert abs(cd - Q["critical-difference"]["answer"]) <= Q["critical-difference"]["tolerance"]


def test_five_seed_sign_test():
    p = 2 * 0.5**5
    sim = np.mean(np.isin((rng.random((200000, 5)) < 0.5).sum(axis=1), [0, 5]))
    assert abs(sim - p) < 0.003
    assert math.isclose(Q["five-seed-sign-test"]["answer"], p)
