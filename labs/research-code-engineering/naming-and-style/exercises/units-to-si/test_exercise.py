import copy
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


to_si = load().to_si


def test_example():
    out = to_si({"voltage_mv": 3700.0, "time_min": 30.0, "temp_c": 25.0})
    assert set(out) == {"voltage_v", "time_s", "temp_k"}, f"이름이 SI 접미사로 바뀌어야 합니다. 현재 {sorted(out)}"
    assert np.isclose(out["voltage_v"], 3.7)
    assert np.isclose(out["time_s"], 1800.0)
    assert np.isclose(out["temp_k"], 298.15), "섭씨 → 켈빈은 273.15를 더합니다"


@pytest.mark.parametrize("name,value,new_name,expected", [
    ("v_bus_kv", 1.2, "v_bus_v", 1200.0),
    ("current_ma", 500.0, "current_a", 0.5),
    ("dt_ms", 250.0, "dt_s", 0.25),
    ("rest_h", 2.0, "rest_s", 7200.0),
    ("capacity_mah", 3000.0, "capacity_ah", 3.0),
    ("current_a", 1.5, "current_a", 1.5),
    ("ambient_k", 300.0, "ambient_k", 300.0),
])
def test_each_unit(name, value, new_name, expected):
    out = to_si({name: value})
    assert list(out) == [new_name], f"{name} → {new_name}이어야 합니다. 현재 {list(out)}"
    assert np.isclose(out[new_name], expected), f"{name}={value} → {expected}이어야 합니다. 현재 {out[new_name]}"


def test_arrays_keep_shape():
    voltage_mv = np.array([[4150.0, 3900.0], [3700.0, 3500.0]])   # (2, 2)
    out = to_si({"voltage_mv": voltage_mv})
    assert np.shape(out["voltage_v"]) == (2, 2), "배열의 shape은 그대로여야 합니다"
    assert np.allclose(out["voltage_v"], voltage_mv / 1000)


def test_celsius_offset_not_scale():
    out = to_si({"t_c": np.array([0.0, 100.0])})
    assert np.allclose(out["t_k"], [273.15, 373.15]), "0 °C는 273.15 K입니다. 곱셈이 아니라 더하기입니다"


def test_passthrough():
    out = to_si({"cell_id": 7, "cycle_index": 120, "current_ma": 100.0}, passthrough=("cell_id", "cycle_index"))
    assert out["cell_id"] == 7 and out["cycle_index"] == 120
    assert np.isclose(out["current_a"], 0.1)


@pytest.mark.parametrize("name", ["voltage", "voltage_mvolt", "temp_f", "cell_id"])
def test_unknown_suffix_raises(name):
    with pytest.raises(ValueError):
        to_si({name: 1.0})


def test_collision_raises():
    with pytest.raises(ValueError):
        to_si({"voltage_mv": 3700.0, "voltage_v": 3.7})


def test_input_not_mutated():
    record = {"voltage_mv": np.array([3700.0]), "time_min": 1.0}
    before = copy.deepcopy(record)
    to_si(record)
    assert list(record) == list(before) and np.array_equal(record["voltage_mv"], before["voltage_mv"]), "입력 dict를 바꾸면 안 됩니다"
