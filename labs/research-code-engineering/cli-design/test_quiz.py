"""6.4 퀴즈(content/06-research-code-engineering/04-cli-design/quiz.toml) 정답을 argparse로 직접 검증한다."""
import argparse
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/04-cli-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def test_abbrev_prefix():
    p = Parser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr-schedule", default="cosine")
    args = p.parse_args(["--lr", "3e-4"])
    assert args.lr_schedule == "3e-4" and not hasattr(args, "lr")
    strict = Parser(allow_abbrev=False)
    strict.add_argument("--lr-schedule", default="cosine")
    with pytest.raises(ValueError):
        strict.parse_args(["--lr", "3e-4"])
    assert correct("abbrev-prefix") == [1]


def test_parse_known_args_drops_typos():
    p = Parser()
    p.add_argument("--lr", type=float, default=1e-3)
    args, unknown = p.parse_known_args(["--learning-rate", "3e-4"])
    assert args.lr == 1e-3 and unknown == ["--learning-rate", "3e-4"]
    assert correct("parse-known-args-risk") == [0]


def test_parser_default_overrides_file():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    final = {**{"lr": 1e-3}, **{"lr": 3e-4}, **vars(p.parse_args([]))}
    assert final["lr"] == 1e-3
    assert correct("parser-default-overrides-file") == [1]


def test_keep_precedence():
    defaults, file_cfg = {"lr": 1e-3}, {"lr": 3e-4}
    none_p = argparse.ArgumentParser()
    none_p.add_argument("--lr", type=float, default=None)
    given = {k: v for k, v in vars(none_p.parse_args([])).items() if v is not None}
    assert {**defaults, **file_cfg, **given}["lr"] == 3e-4                       # 선택지 0
    sup = argparse.ArgumentParser()
    sup.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    assert vars(sup.parse_args([])) == {}                                         # 선택지 1
    assert {**defaults, **file_cfg, **vars(sup.parse_args(["--lr", "0.2"]))}["lr"] == 0.2
    same = argparse.ArgumentParser()
    same.add_argument("--lr", type=float, default=1e-3)
    assert {**defaults, **file_cfg, **vars(same.parse_args([]))}["lr"] == 1e-3   # 선택지 2: 파일 값을 잃음
    assert {**defaults, **{"lr": 0.2}, **file_cfg}["lr"] == 3e-4                  # 선택지 3: 명령행이 짐
    assert correct("keep-precedence") == [0, 1]


def test_subcommand_option_position():
    cli = Parser(prog="soh")
    cli.add_argument("--seed", type=int, default=0)
    sub = cli.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("config")
    assert cli.parse_args(["--seed", "1", "train", "configs/base.toml"]).seed == 1
    with pytest.raises(ValueError):
        cli.parse_args(["train", "configs/base.toml", "--seed", "1"])
    assert correct("subcommand-option-position") == [0]


def test_console_script_entry():
    assert correct("console-script-entry") == [0]
