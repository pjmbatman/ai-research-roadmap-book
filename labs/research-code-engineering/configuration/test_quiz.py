"""6.3 퀴즈(content/06-research-code-engineering/03-configuration/quiz.toml) 정답을 계산으로 검증한다."""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/03-configuration/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_cwd_relative_path(tmp_path):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "data" / "train.csv").write_text("x\n1\n")
    old = os.getcwd()
    try:
        os.chdir(root)
        assert open("data/train.csv").read().startswith("x")        # root에서는 동작
        os.chdir(root / "notebooks")
        with pytest.raises(FileNotFoundError):
            open("data/train.csv")                                    # 작업 디렉터리 기준
    finally:
        os.chdir(old)
    assert correct("cwd-relative-path") == [1]


def test_shallow_merge():
    defaults = {"model": {"width": 64, "depth": 3}, "train": {"lr": 0.001}}
    exp = {"model": {"width": 128}}
    assert {**defaults, **exp}["model"] == {"width": 128}
    assert correct("shallow-merge") == [0]


def test_dataclass_no_type_check():
    @dataclass
    class TrainConfig:
        seed: int = 0

    cfg = TrainConfig(seed="0")
    assert cfg.seed == "0" and isinstance(cfg.seed, str)
    assert correct("dataclass-no-type-check") == [2]


def test_other_answers():
    assert correct("string-lr-remedy") == [1]
    assert correct("resolved-config-timing") == [1]
    assert correct("restore-requirements") == [0, 1, 2]
