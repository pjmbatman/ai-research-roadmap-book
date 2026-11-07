import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def test_iid_chains_near_one():
    rng = np.random.default_rng(0)
    r = m.split_rhat(rng.normal(size=(4, 2000)))
    assert r == pytest.approx(1.0, abs=0.01), "같은 분포의 독립 체인은 R-hat이 1에 가까워야 합니다"


def test_hand_computed_value():
    chains = [[0.0, 1.0, 2.0, 3.0]]                    # 나누면 [0, 1], [2, 3]
    # W = 0.5, B = 2 * var([0.5, 2.5], ddof=1) = 4, R = sqrt((0.5 * 0.5 + 4 / 2) / 0.5)
    assert m.split_rhat(chains) == pytest.approx(np.sqrt(4.5)), "분산은 분모 n - 1(ddof=1)로 계산합니다"


def test_separated_chains_detected():
    rng = np.random.default_rng(1)
    chains = rng.normal(size=(4, 1000)) + np.array([[0.0], [0.0], [0.0], [3.0]])
    assert m.split_rhat(chains) > 1.1, "한 체인만 다른 곳에 있어도 R-hat이 커야 합니다"


def test_single_chain_with_trend():
    rng = np.random.default_rng(2)
    chain = np.linspace(0, 3, 1000) + rng.normal(size=1000)
    assert m.split_rhat(chain[None, :]) > 1.1, "체인이 하나라도 앞뒤 절반이 다르면 split R-hat이 커야 합니다"


def test_odd_length_and_list_input():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 1001))
    assert m.split_rhat(x.tolist()) == pytest.approx(m.split_rhat(x[:, :1000])), "홀수 길이면 마지막 표본을 버립니다"


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        m.split_rhat(np.zeros(10))
    with pytest.raises(ValueError):
        m.split_rhat(np.zeros((2, 3)))
