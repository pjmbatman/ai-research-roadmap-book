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


mod = load()
training_memory, max_batch = mod.training_memory, mod.max_batch


def test_worked_example():
    m = training_memory([2, 3, 1], 4, "fp32-adam")
    assert m["params"] == 52, f"파라미터는 13개 × 4 byte = 52 byte입니다. 현재 {m['params']}"
    assert m["grads"] == 52
    assert m["optimizer"] == 104, "Adam은 파라미터마다 m, v 두 개(각 4 byte)를 가집니다"
    assert m["activations"] == 80, "활성값은 batch × (마지막 층을 뺀 폭의 합) × 4 byte입니다"
    assert m["total"] == 288


def test_values_are_int():
    m = training_memory([8, 16, 4], 3, "bf16-adam")
    assert all(isinstance(v, int) for v in m.values()), "byte 수는 int로 반환합니다"


def test_bias_counted():
    m = training_memory([1, 1], 1, "fp32-sgd")
    assert m["params"] == 8, "Linear(1, 1)의 파라미터는 가중치 1 + bias 1 = 2개입니다"


@pytest.mark.parametrize("recipe", ["fp32-sgd", "fp32-adam", "bf16-adam"])
def test_static_is_batch_independent(recipe):
    w = [64, 256, 256, 10]
    a, b = training_memory(w, 1, recipe), training_memory(w, 100, recipe)
    for k in ["params", "grads", "optimizer"]:
        assert a[k] == b[k], f"{k}는 batch와 무관해야 합니다"
    assert b["activations"] == 100 * a["activations"], "활성값은 batch에 정확히 비례해야 합니다"


def test_bf16_adam_same_static_as_fp32_adam():
    w = [64, 256, 256, 10]
    a, b = training_memory(w, 0, "fp32-adam"), training_memory(w, 0, "bf16-adam")
    assert a["total"] == b["total"], "bf16 가중치 + fp32 사본 + Adam도 파라미터당 16 byte입니다"


def test_max_batch_boundary():
    assert max_batch([2, 3, 1], 288, "fp32-adam") == 4
    assert max_batch([2, 3, 1], 287, "fp32-adam") == 3, "예산을 1 byte라도 넘으면 안 됩니다"


def test_max_batch_does_not_fit():
    assert max_batch([2, 3, 1], 208, "fp32-adam") == 0, "정적 메모리만으로 예산이 차면 0입니다"
    assert max_batch([2, 3, 1], 100, "fp32-adam") == 0


def test_max_batch_is_maximal():
    w, budget = [128, 512, 512, 10], 20_000_000
    b = max_batch(w, budget, "fp32-adam")
    assert training_memory(w, b, "fp32-adam")["total"] <= budget
    assert training_memory(w, b + 1, "fp32-adam")["total"] > budget
