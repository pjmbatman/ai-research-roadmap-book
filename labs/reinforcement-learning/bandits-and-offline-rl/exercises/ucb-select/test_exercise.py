import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ucb_select = load().ucb_select


def test_untried_arm_first():
    counts = np.array([3, 0, 2])
    sums = np.array([3.0, 0.0, 2.0])
    assert ucb_select(counts, sums, t=5) == 1, "한 번도 당기지 않은 팔이 있으면 그 팔을 먼저 고릅니다"


def test_quiz_example():
    counts = np.array([50, 5])
    sums = np.array([0.6 * 50, 0.5 * 5])
    assert ucb_select(counts, sums, t=55) == 1, "퀴즈의 예: 시행이 적은 B(지표 약 1.77)가 A(약 1.00)보다 큽니다"


def test_exploitation_when_bonus_small():
    counts = np.array([10_000, 10_000])
    sums = np.array([0.40 * 10_000, 0.45 * 10_000])
    assert ucb_select(counts, sums, t=20_000) == 1, "시행이 충분하면 평균이 큰 팔을 고릅니다"


def test_c_scales_bonus():
    counts = np.array([100, 4])
    sums = np.array([70.0, 1.0])                   # 평균 0.70, 0.25
    assert ucb_select(counts, sums, t=104, c=0.1) == 0, "c가 작으면 보너스가 작아 평균이 큰 팔을 고릅니다"
    assert ucb_select(counts, sums, t=104, c=2.0) == 1, "c가 크면 덜 해 본 팔을 고릅니다"


def test_returns_python_int_and_runs_bandit():
    p = np.array([0.2, 0.5, 0.8])
    rng = np.random.default_rng(0)
    counts, sums = np.zeros(3, dtype=int), np.zeros(3)
    for t in range(1, 1501):
        a = ucb_select(counts, sums, t)
        assert isinstance(a, (int, np.integer)), "팔 번호(정수)를 반환하세요"
        counts[a] += 1
        sums[a] += rng.random() < p[a]
    assert counts[2] > 0.8 * counts.sum(), "1500번 뒤에는 대부분 가장 좋은 팔을 골라야 합니다"
    assert counts.min() >= 5, "UCB는 나쁜 팔도 가끔 다시 시도합니다"
