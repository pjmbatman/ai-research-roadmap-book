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


mod = load()

MODEL = {"0.weight": (8, 4), "0.bias": (8,), "1.weight": (8,), "1.bias": (8,),
         "1.running_mean": (8,), "1.running_var": (8,), "2.weight": (1, 8), "2.bias": (1,)}


def fake_sd(shapes, prefix=""):
    return {prefix + k: np.zeros(s) for k, s in shapes.items()}


def test_exact_match():
    d = mod.diff_state_dict(MODEL, fake_sd(MODEL))
    assert d == {"missing": [], "unexpected": [], "shape_mismatch": []}, "같은 이름·shape이면 세 목록이 모두 비어야 합니다"


def test_example_from_problem():
    d = mod.diff_state_dict({"fc.weight": (2, 3), "fc.bias": (2,)},
                            {"fc.weight": np.zeros((3, 3)), "head.bias": np.zeros(1)})
    assert d == {"missing": ["fc.bias"], "unexpected": ["head.bias"], "shape_mismatch": ["fc.weight"]}


def test_prefix_makes_everything_mismatch():
    d = mod.diff_state_dict(MODEL, fake_sd(MODEL, "module."))
    assert d["missing"] == sorted(MODEL), "접두사가 붙으면 모든 키가 missing입니다"
    assert d["unexpected"] == sorted("module." + k for k in MODEL)
    assert d["shape_mismatch"] == []


def test_strip_prefix_fixes_it():
    sd = fake_sd(MODEL, "module.")
    fixed = mod.strip_prefix(sd, "module.")
    assert set(fixed) == set(MODEL)
    assert mod.diff_state_dict(MODEL, fixed) == {"missing": [], "unexpected": [], "shape_mismatch": []}
    assert all(k.startswith("module.") for k in sd), "원본 dict를 바꾸면 안 됩니다"


def test_strip_prefix_only_at_start():
    sd = {"model.encoder.weight": 1, "encoder.model.bias": 2, "head": 3}
    out = mod.strip_prefix(sd, "model.")
    assert out == {"encoder.weight": 1, "encoder.model.bias": 2, "head": 3}, \
        "접두사는 키의 맨 앞에 있을 때만 떼어 냅니다"


def test_values_are_kept():
    W = np.arange(6.0).reshape(2, 3)
    out = mod.strip_prefix({"module.w": W}, "module.")
    assert out["w"] is W or np.array_equal(out["w"], W)


def test_shape_mismatch_after_resizing_head():
    sd = fake_sd(MODEL)
    sd["2.weight"], sd["2.bias"] = np.zeros((10, 8)), np.zeros(10)   # 출력 클래스 수가 다른 head
    d = mod.diff_state_dict(MODEL, sd)
    assert d == {"missing": [], "unexpected": [], "shape_mismatch": ["2.bias", "2.weight"]}, \
        "shape_mismatch 목록은 정렬되어야 합니다"
