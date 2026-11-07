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


successive_halving = load().successive_halving


class Recorder:
    """evaluate 호출을 기록하는 학습 곡선: 최종 손실 final[i]로 같은 속도로 내려간다"""

    def __init__(self, final, speed=None):
        self.final, self.speed = np.asarray(final, dtype=float), speed
        self.calls = []

    def __call__(self, i, budget):
        self.calls.append((i, budget))
        k = 0.5 if self.speed is None else self.speed[i]
        return self.final[i] + (1 - self.final[i]) * np.exp(-k * budget)


def test_budget_27_configs():
    rng = np.random.default_rng(0)
    ev = Recorder(rng.uniform(0.1, 0.9, 27))
    _, total = successive_halving(ev, 27, min_budget=1, eta=3)
    assert total == 108, f"27×1 + 9×3 + 3×9 + 1×27 = 108이어야 합니다. 현재 {total}"
    budgets = sorted({b for _, b in ev.calls})
    assert budgets == [1, 3, 9, 27], f"단계별 예산은 1, 3, 9, 27이어야 합니다. 현재 {budgets}"
    assert len(ev.calls) == 27 + 9 + 3 + 1, "각 단계에서 살아남은 후보만 평가해야 합니다"


def test_finds_best_when_ranking_is_early():
    rng = np.random.default_rng(1)
    final = rng.uniform(0.1, 0.9, 27)
    best, _ = successive_halving(Recorder(final), 27, min_budget=1, eta=3)
    assert best == int(np.argmin(final)), "모든 곡선이 같은 속도로 내려가면 첫 단계의 순위가 최종 순위와 같으므로 최선을 찾아야 합니다"


def test_non_power_of_eta():
    ev = Recorder(np.linspace(0.9, 0.1, 10))      # 후보 9가 최선
    best, total = successive_halving(ev, 10, min_budget=1, eta=3)
    assert best == 9
    assert total == 10 * 1 + 3 * 3 + 1 * 9, "10개 → 3개 → 1개: 10 + 9 + 9 = 28"


def test_tie_breaks_by_index():
    ev = Recorder([0.5, 0.2, 0.2, 0.9])
    best, _ = successive_halving(ev, 4, min_budget=1, eta=2)
    assert best == 1, "손실이 같으면 번호가 작은 후보를 남깁니다"


def test_late_bloomer_can_be_missed():
    final = np.array([0.30, 0.31, 0.32, 0.10])      # 후보 3이 최종적으로 가장 좋지만
    speed = np.array([1.0, 1.0, 1.0, 0.05])         # 아주 느리게 내려간다
    best, _ = successive_halving(Recorder(final, speed), 4, min_budget=1, eta=2)
    assert best != 3, "느리게 좋아지는 후보는 첫 단계에서 탈락한다: 조기 중단의 대가"
