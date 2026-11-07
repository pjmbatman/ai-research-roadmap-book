import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def test_update_counts():
    a, b = m.update(2, 2, [1, 0, 1, 1, 0, 1, 0])
    assert (a, b) == pytest.approx((6, 5)), "앞면 수를 a에, 뒷면 수를 b에 더합니다"


def test_update_no_data_returns_prior():
    assert m.update(3.5, 1.5, []) == pytest.approx((3.5, 1.5)), "관측이 없으면 사후분포는 사전분포와 같습니다"


def test_sequential_equals_batch():
    rng = np.random.default_rng(0)
    tosses = (rng.random(40) < 0.3).astype(int)
    a, b = 1.0, 1.0
    for t in tosses:
        a, b = m.update(a, b, [t])                                # 오늘의 사후가 내일의 사전
    assert (a, b) == pytest.approx(m.update(1.0, 1.0, tosses)), "하나씩 갱신한 결과가 한꺼번에 갱신한 결과와 달라요"


@pytest.mark.parametrize("a,b,mm", [(9, 15, 20), (0.5, 0.5, 7), (30, 70, 50)])
def test_predictive_is_distribution_with_right_mean(a, b, mm):
    p = np.asarray(m.predictive_pmf(a, b, mm))
    assert p.shape == (mm + 1,), f"길이가 m+1 = {mm + 1}이어야 합니다. 현재 {p.shape}"
    assert np.all(p >= 0) and p.sum() == pytest.approx(1.0), "확률은 0 이상이고 합이 1이어야 합니다"
    assert np.sum(np.arange(mm + 1) * p) == pytest.approx(mm * a / (a + b)), "평균은 m · a/(a+b)여야 합니다"
    assert p == pytest.approx(stats.betabinom.pmf(np.arange(mm + 1), mm, a, b)), "Beta-Binomial 확률과 다릅니다"


def test_one_toss_predictive():
    p = np.asarray(m.predictive_pmf(9, 15, 1))
    assert p == pytest.approx([15 / 24, 9 / 24]), "m = 1이면 [b/(a+b), a/(a+b)]입니다"


def test_wider_than_plugin():
    a, b, mm = 9, 15, 20
    j = np.arange(mm + 1)
    p = np.asarray(m.predictive_pmf(a, b, mm))
    q = stats.binom.pmf(j, mm, a / (a + b))                       # θ를 사후평균으로 고정
    var = lambda r: np.sum(j**2 * r) - np.sum(j * r) ** 2
    assert var(p) > 1.3 * var(q), "사후예측분포는 θ의 불확실성까지 반영하므로 plug-in 이항분포보다 넓어야 합니다"


def test_large_counts_are_stable():
    p = np.asarray(m.predictive_pmf(400, 600, 300))
    assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0), "큰 수에서 overflow가 납니다. 로그 공간에서 계산해 보세요"
