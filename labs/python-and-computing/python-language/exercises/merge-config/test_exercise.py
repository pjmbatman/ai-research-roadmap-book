import copy
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_config = load().merge_config


def make_inputs():
    base = {"model": {"width": 64, "depth": 2, "act": {"name": "relu"}}, "tags": ["baseline"], "lr": 1e-3}
    overrides = {"model": {"depth": 4, "act": {"slope": 0.1}}, "seed": 1, "extra": {"notes": ["v2"]}}
    return base, overrides


def test_values():
    base, overrides = make_inputs()
    out = merge_config(base, overrides)
    assert out == {
        "model": {"width": 64, "depth": 4, "act": {"name": "relu", "slope": 0.1}},
        "tags": ["baseline"],
        "lr": 1e-3,
        "seed": 1,
        "extra": {"notes": ["v2"]},
    }, f"병합 결과가 다릅니다: {out}"


def test_inputs_unchanged():
    base, overrides = make_inputs()
    base_before, overrides_before = copy.deepcopy(base), copy.deepcopy(overrides)
    merge_config(base, overrides)
    assert base == base_before, "base가 바뀌었습니다. 중첩 dict를 제자리에서 고치고 있지 않은지 확인하세요"
    assert overrides == overrides_before, "overrides가 바뀌었습니다"


def test_result_does_not_share_objects_with_base():
    base, overrides = make_inputs()
    out = merge_config(base, overrides)
    out["tags"].append("changed")
    out["model"]["width"] = 0
    out["model"]["act"]["name"] = "gelu"
    assert base["tags"] == ["baseline"], "결과의 리스트가 base의 리스트와 같은 객체입니다(얕은 복사)"
    assert base["model"]["width"] == 64, "결과의 중첩 dict가 base와 같은 객체입니다"
    assert base["model"]["act"]["name"] == "relu", "더 깊은 중첩 dict도 공유하면 안 됩니다"


def test_result_does_not_share_objects_with_overrides():
    base, overrides = make_inputs()
    out = merge_config(base, overrides)
    out["extra"]["notes"].append("changed")
    assert overrides["extra"]["notes"] == ["v2"], "overrides에서 가져온 값도 복사해야 합니다"


def test_non_dict_override_replaces():
    out = merge_config({"model": {"width": 64}}, {"model": "resnet"})
    assert out == {"model": "resnet"}
    out = merge_config({"layers": [1, 2]}, {"layers": {"n": 3}})
    assert out == {"layers": {"n": 3}}


def test_calls_are_independent():
    base = {"tags": []}
    first = merge_config(base, {})
    first["tags"].append("a")
    second = merge_config(base, {})
    assert second == {"tags": []}, "앞선 호출의 결과가 다음 호출에 새어 들어갑니다"
