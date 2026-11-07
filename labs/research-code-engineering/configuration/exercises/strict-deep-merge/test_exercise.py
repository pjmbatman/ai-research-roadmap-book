import copy
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


deep_merge = load().deep_merge


def make_base():
    return {
        "data": {"name": "cycles", "splits": {"train": 0.8, "val": 0.1}},
        "model": {"width": 64, "depth": 3, "dropout": 0.1},
        "train": {"lr": 1e-3, "seed": 0, "milestones": [30, 60]},
    }


def test_nested_merge_keeps_siblings():
    out = deep_merge(make_base(), {"model": {"width": 128}})
    assert out["model"] == {"width": 128, "depth": 3, "dropout": 0.1}, "덮어쓰지 않은 같은 섹션의 키가 남아 있어야 합니다"
    assert out["train"] == make_base()["train"], "건드리지 않은 섹션은 그대로여야 합니다"


def test_two_levels_deep():
    out = deep_merge(make_base(), {"data": {"splits": {"val": 0.15}}})
    assert out["data"]["splits"] == {"train": 0.8, "val": 0.15}
    assert out["data"]["name"] == "cycles"


def test_list_is_replaced_not_extended():
    out = deep_merge(make_base(), {"train": {"milestones": [50]}})
    assert out["train"]["milestones"] == [50], "리스트는 이어 붙이지 않고 통째로 대체합니다"


def test_sequential_layers():
    cfg = make_base()
    for layer in [{"model": {"width": 128}}, {"train": {"lr": 3e-4}}, {"model": {"width": 256}}]:
        cfg = deep_merge(cfg, layer)
    assert cfg["model"]["width"] == 256 and cfg["train"]["lr"] == 3e-4, "뒤에 오는 층이 우선합니다"


@pytest.mark.parametrize("override,path", [
    ({"train": {"learning_rate": 3e-4}}, "train.learning_rate"),
    ({"optim": {"lr": 3e-4}}, "optim"),
    ({"data": {"splits": {"tset": 0.1}}}, "data.splits.tset"),
])
def test_unknown_key_raises_with_path(override, path):
    with pytest.raises(KeyError) as info:
        deep_merge(make_base(), override)
    assert path in str(info.value), f"오류 메시지에 점 경로 {path!r}가 있어야 합니다. 현재 {info.value}"


@pytest.mark.parametrize("override,path", [
    ({"model": 5}, "model"),
    ({"train": {"lr": {"value": 1e-3}}}, "train.lr"),
])
def test_dict_value_mismatch_raises(override, path):
    with pytest.raises(TypeError) as info:
        deep_merge(make_base(), override)
    assert path in str(info.value), f"오류 메시지에 점 경로 {path!r}가 있어야 합니다. 현재 {info.value}"


def test_inputs_not_mutated():
    base, override = make_base(), {"model": {"width": 128}, "train": {"milestones": [10]}}
    base_before, override_before = copy.deepcopy(base), copy.deepcopy(override)
    deep_merge(base, override)
    assert base == base_before, "base를 바꾸면 안 됩니다"
    assert override == override_before, "override를 바꾸면 안 됩니다"


def test_result_shares_nothing_with_inputs():
    base, override = make_base(), {"train": {"milestones": [10, 20]}}
    out = deep_merge(base, override)
    out["data"]["splits"]["train"] = 0.5
    out["train"]["milestones"].append(99)
    assert base["data"]["splits"]["train"] == 0.8, "반환값을 고쳤더니 base가 바뀌었습니다(중첩 dict 공유)"
    assert override["train"]["milestones"] == [10, 20], "반환값을 고쳤더니 override가 바뀌었습니다(list 공유)"
