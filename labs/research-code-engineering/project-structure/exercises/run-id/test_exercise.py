import copy
import hashlib
import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_run_id = load().make_run_id
T0 = datetime(2026, 11, 7, 14, 3, 9)
BASE = {
    "data": {"name": "cycles", "split_seed": 0},
    "model": {"width": 64, "depth": 3},
    "train": {"lr": 0.001, "seed": 0},
    "paths": {"output_dir": "outputs", "data_root": "data/processed"},
    "logging": {"level": "INFO"},
}


def reference_hash(cfg):
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def test_format_and_timestamp():
    rid = make_run_id(BASE, T0)
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", rid), f"형식이 'YYYYmmdd-HHMMSS-xxxxxxxx'가 아닙니다: {rid!r}"
    assert rid.startswith("20261107-140309-"), f"시각 부분이 다릅니다: {rid!r}"


def test_hash_matches_spec():
    rid = make_run_id(BASE, T0)
    assert rid.endswith(reference_hash(BASE)), "json.dumps(sort_keys=True, separators=(',', ':'))의 SHA-256 앞 8자리여야 합니다"


def test_key_order_does_not_matter():
    shuffled = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(BASE.items()))}
    assert make_run_id(shuffled, T0) == make_run_id(BASE, T0), "키 순서만 다른 설정은 같은 run ID여야 합니다"


def test_result_affecting_change_changes_hash():
    other = copy.deepcopy(BASE)
    other["train"]["seed"] = 1
    assert make_run_id(other, T0) != make_run_id(BASE, T0), "seed가 다르면 hash도 달라야 합니다"


def test_ignored_keys_do_not_change_hash():
    ignore = ("paths.output_dir", "logging")
    other = copy.deepcopy(BASE)
    other["paths"]["output_dir"] = "/scratch/me/outputs"
    other["logging"]["level"] = "DEBUG"
    assert make_run_id(other, T0, ignore) == make_run_id(BASE, T0, ignore), "ignore에 적힌 값은 hash에 영향을 주면 안 됩니다"
    other["paths"]["data_root"] = "data/v2"
    assert make_run_id(other, T0, ignore) != make_run_id(BASE, T0, ignore), "ignore에 없는 paths.data_root는 hash에 반영돼야 합니다"


def test_missing_ignore_path_is_fine():
    assert make_run_id(BASE, T0, ("no.such.key", "train.nope")) == make_run_id(BASE, T0)


def test_config_is_not_mutated():
    before = copy.deepcopy(BASE)
    make_run_id(BASE, T0, ("paths.output_dir", "logging"))
    assert BASE == before, "인자로 받은 config를 바꾸면 안 됩니다"
