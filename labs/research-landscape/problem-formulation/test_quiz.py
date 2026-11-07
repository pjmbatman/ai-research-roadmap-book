"""1.3 퀴즈(content/01-research-landscape/03-problem-formulation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/01-research-landscape/03-problem-formulation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_noise_floor_leak():
    q = Q["noise-floor-leak"]
    floor = 0.3 * np.sqrt(2 / np.pi)
    assert abs(floor - q["answer"]) <= q["tolerance"]
    eps = rng.normal(0, 0.3, size=200_000)                   # 참 온도를 정확히 맞혀도 남는 오차
    assert abs(np.mean(np.abs(eps)) - q["answer"]) <= q["tolerance"]
    assert 0.222 < floor                                       # 본문의 "중심 평균" MAE는 하한보다 작다


def test_leaky_window_beats_floor_in_cells():
    """본문 cell 1과 같은 데이터에서 중심 평균 입력이 하한 아래로 내려가는지 확인한다."""
    r = np.random.default_rng(0)
    T = 3000
    temp_c = 25.0 + np.cumsum(r.normal(0, 0.1, size=T))
    obs_c = temp_c + r.normal(0, 0.3, size=T)
    centered = np.convolve(obs_c, np.ones(5) / 5, mode="valid")  # t-2..t+2 평균, t = 2..T-3
    target = obs_c[3:T - 1]                                    # y_{t+1}, t = 2..T-3
    assert len(centered) == len(target)
    a, b = np.polyfit(centered[:2000], target[:2000], 1)
    mae = np.mean(np.abs(a * centered[2000:] + b - target[2000:]))
    assert mae < 0.3 * np.sqrt(2 / np.pi)


def test_single_choice_answers():
    assert correct("variable-roles") == [1]
    assert correct("problem-type") == [1]
    assert correct("control-vs-optimization") == [1]
    assert correct("split-for-new-cells") == [1]
    assert correct("inference-time-info") == [0, 1]
