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


resolve_config = load().resolve_config


def make_defaults():
    return {
        "data": {"name": "sine", "noise": 0.1},
        "model": {"name": "poly", "degree": 5},
        "train": {"lr": 1e-3, "epochs": 50, "seed": 0},
        "eval": {"metrics": ["rmse"]},
    }


def test_example_from_problem():
    cfg, src = resolve_config(
        {"model": {"width": 64}, "train": {"lr": 1e-3, "seed": 0}},
        {"train": {"lr": 3e-4}},
        {"command": "train", "train.lr": None, "train.seed": 3},
    )
    assert cfg == {"model": {"width": 64}, "train": {"lr": 3e-4, "seed": 3}}
    assert src == {"model.width": "default", "train.lr": "file", "train.seed": "cli"}


def test_priority_default_file_cli():
    cfg, src = resolve_config(make_defaults(), {"train": {"lr": 3e-4, "epochs": 100}}, {"train.epochs": 20})
    assert cfg["train"] == {"lr": 3e-4, "epochs": 20, "seed": 0}, "기본값 < 설정 파일 < 명령행 순서로 우선해야 합니다"
    assert (src["train.lr"], src["train.epochs"], src["train.seed"]) == ("file", "cli", "default")


def test_none_does_not_override_file():
    cfg, src = resolve_config(make_defaults(), {"train": {"lr": 3e-4}}, {"train.lr": None})
    assert cfg["train"]["lr"] == 3e-4, "명령행에서 주지 않은 옵션(None)이 설정 파일 값을 덮어쓰면 안 됩니다"
    assert src["train.lr"] == "file"


def test_non_dotted_keys_are_ignored():
    cfg, _ = resolve_config(make_defaults(), {}, {"command": "train", "config": "exp.toml", "dry_run": True})
    assert cfg == make_defaults(), "점이 없는 CLI 전용 인자는 설정에 들어가면 안 됩니다"


def test_sources_cover_every_leaf():
    _, src = resolve_config(make_defaults(), {}, {})
    expected = {"data.name", "data.noise", "model.name", "model.degree", "train.lr", "train.epochs", "train.seed", "eval.metrics"}
    assert set(src) == expected, f"모든 말단 키의 출처가 있어야 합니다. 빠진 키 {expected - set(src)}"
    assert set(src.values()) == {"default"}


def test_list_replaced_from_file():
    cfg, src = resolve_config(make_defaults(), {"eval": {"metrics": ["mae", "r2"]}}, {})
    assert cfg["eval"]["metrics"] == ["mae", "r2"] and src["eval.metrics"] == "file"


@pytest.mark.parametrize("file_cfg,cli_args,path", [
    ({"train": {"learning_rate": 3e-4}}, {}, "train.learning_rate"),
    ({"optim": {"lr": 3e-4}}, {}, "optim"),
    ({}, {"train.learning_rate": 3e-4}, "train.learning_rate"),
])
def test_unknown_keys_raise(file_cfg, cli_args, path):
    with pytest.raises(KeyError) as info:
        resolve_config(make_defaults(), file_cfg, cli_args)
    assert path in str(info.value), f"오류 메시지에 점 경로 {path!r}가 있어야 합니다. 현재 {info.value}"


def test_section_value_mismatch_raises():
    with pytest.raises(TypeError):
        resolve_config(make_defaults(), {"model": "poly"}, {})
    with pytest.raises(TypeError):
        resolve_config(make_defaults(), {"train": {"lr": {"value": 0.1}}}, {})


def test_inputs_not_mutated():
    defaults, file_cfg = make_defaults(), {"eval": {"metrics": ["mae"]}, "train": {"lr": 3e-4}}
    cli_args = {"train.seed": 7}
    before = copy.deepcopy((defaults, file_cfg, cli_args))
    cfg, _ = resolve_config(defaults, file_cfg, cli_args)
    cfg["eval"]["metrics"].append("x")
    assert (defaults, file_cfg, cli_args) == before, "입력(또는 입력과 공유된 리스트)이 바뀌었습니다"
