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


find_spikes = load().find_spikes


def test_example_from_problem():
    norms = [1.0, 1.1, 0.9, 1.0, 1.2, 1.0, 9.0, 1.1, np.nan, 1.0]
    assert list(find_spikes(norms)) == [6, 8]


def test_quiet_log_has_no_spikes():
    rng = np.random.default_rng(0)
    norms = np.exp(rng.normal(0, 0.2, size=500))            # 평소의 잡음
    assert list(find_spikes(norms)) == []


def test_early_large_values_are_not_flagged():
    norms = [50.0, 20.0, 8.0, 3.0, 2.0, 1.5, 1.2, 1.0, 1.1, 1.0]   # 학습 초반에 줄어드는 gradient
    assert list(find_spikes(norms)) == [], "이력이 min_history보다 짧을 때는 spike 판정을 하지 않습니다"


def test_spikes_do_not_raise_the_baseline():
    norms = [1.0] * 10 + [8.0, 8.0, 8.0, 8.0] + [1.0] * 5
    assert list(find_spikes(norms, factor=5.0)) == [10, 11, 12, 13], \
        "이상으로 판정된 값을 이력에 넣으면 연속된 spike가 기준을 끌어올려 놓칩니다"


def test_window_uses_recent_history():
    norms = [10.0] * 30 + [1.0] * 30 + [6.0]
    assert list(find_spikes(norms, window=20)) == [60], "기준은 최근 window개의 정상 값입니다"
    assert list(find_spikes(norms, window=100)) == [], "window가 길면 오래된 큰 값이 중앙값을 끌어올립니다"


def test_nonfinite_always_flagged():
    assert list(find_spikes([np.inf, 1.0, -np.inf, np.nan])) == [0, 2, 3]


def test_median_not_mean():
    norms = [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 5.5]
    # step 5(4.0)는 5배를 넘지 않아 정상 이력에 들어간다. step 8의 기준은 중앙값 1.0이므로 5.5는 spike다.
    # 평균(1.375)을 기준으로 쓰면 5.5 < 6.875라 놓친다.
    assert list(find_spikes(norms)) == [8], "기준은 평균이 아니라 중앙값입니다"
