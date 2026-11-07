import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EarlyStopping = load().EarlyStopping


def run(es, values):
    """멈춘 epoch(1부터)를 반환한다. 끝까지 멈추지 않으면 None."""
    for epoch, v in enumerate(values, start=1):
        if es.update(epoch, v):
            return epoch
    return None


LOSSES = [1.0, 0.8, 0.85, 0.79, 0.80, 0.81]


def test_example_from_problem():
    es = EarlyStopping(patience=2)
    assert run(es, LOSSES) == 6
    assert es.best_epoch == 4 and es.best == pytest.approx(0.79)


def test_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.02)
    assert run(es, LOSSES) == 4, "0.79는 0.8보다 min_delta(0.02) 이상 좋아지지 않았으므로 개선이 아닙니다"
    assert es.best_epoch == 2


def test_tie_is_not_improvement():
    es = EarlyStopping(patience=2)
    assert run(es, [0.5, 0.5, 0.5]) == 3
    assert es.best_epoch == 1, "동점은 개선이 아닙니다. best_epoch는 처음 그 값이 나온 epoch입니다"


def test_wait_resets_after_improvement():
    es = EarlyStopping(patience=2)
    assert run(es, [1.0, 1.1, 0.9, 1.0, 0.8, 0.85, 0.7]) is None, "개선이 있으면 기다린 횟수가 0으로 돌아가야 합니다"
    assert es.best_epoch == 7


def test_max_mode():
    es = EarlyStopping(patience=3, mode="max")
    assert run(es, [0.6, 0.7, 0.72, 0.71, 0.70, 0.72, 0.69]) == 6
    assert es.best_epoch == 3 and es.best == pytest.approx(0.72)


def test_nan_stops_immediately():
    es = EarlyStopping(patience=10)
    assert run(es, [1.0, 0.9, float("nan"), 0.1]) == 3, "NaN은 발산이므로 patience와 무관하게 즉시 멈춥니다"
    assert es.best == pytest.approx(0.9) and es.best_epoch == 2


def test_inf_stops_immediately():
    es = EarlyStopping(patience=10, mode="max")
    assert run(es, [0.5, float("inf")]) == 2
    assert es.best == pytest.approx(0.5)


def test_first_value_is_improvement():
    es = EarlyStopping(patience=1, min_delta=100.0)
    assert es.update(1, 3.0) is False
    assert es.best_epoch == 1 and es.best == pytest.approx(3.0)
