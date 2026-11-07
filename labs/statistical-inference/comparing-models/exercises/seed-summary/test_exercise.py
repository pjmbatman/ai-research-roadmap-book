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


def test_summarize_hand_example():
    s = m.summarize([0.40, 0.42, 0.44, 0.46])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.43)
    assert s["sd"] == pytest.approx(np.sqrt(0.002 / 3)), "표본 표준편차는 n-1로 나눕니다"
    assert s["se"] == pytest.approx(s["sd"] / 2), "표준오차는 표준편차를 √n으로 나눈 값입니다"


def test_se_shrinks_but_sd_does_not():
    rng = np.random.default_rng(0)
    small = m.summarize(rng.normal(0.4, 0.01, size=5))
    large = m.summarize(rng.normal(0.4, 0.01, size=500))
    assert large["sd"] == pytest.approx(0.01, rel=0.15) and small["sd"] > 0.003, "sd는 실행 사이의 퍼짐(약 0.01)을 추정합니다"
    assert large["se"] < small["se"] / 3,"se는 실행 횟수가 늘면 줄어듭니다"


def test_paired_summary_hand_example():
    a = [0.43, 0.41, 0.42, 0.40, 0.41]
    b = [0.40, 0.40, 0.41, 0.41, 0.39]
    r = m.paired_summary(a, b, t_crit=2.776)
    d = np.array(b) - np.array(a)
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(5))
    assert r["lo"] == pytest.approx(d.mean() - 2.776 * r["se"]) and r["hi"] == pytest.approx(d.mean() + 2.776 * r["se"])
    assert r["wins_b"] == 4


def test_pairing_cancels_shared_variation():
    rng = np.random.default_rng(1)
    split_effect = rng.normal(0, 0.05, size=10)          # seed·분할마다 두 방법에 공통인 변동
    a = 0.42 + split_effect + rng.normal(0, 0.002, size=10)
    b = 0.41 + split_effect + rng.normal(0, 0.002, size=10)
    r = m.paired_summary(a, b, t_crit=2.262)
    unpaired_se = np.sqrt(m.summarize(a)["se"] ** 2 + m.summarize(b)["se"] ** 2)
    assert r["se"] < unpaired_se / 5, "공통 변동은 짝지은 차이에서 상쇄됩니다"
    assert r["hi"] < 0 and r["wins_b"] == 10
