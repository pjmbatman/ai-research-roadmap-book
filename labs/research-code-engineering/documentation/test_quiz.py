"""6.7 퀴즈(content/06-research-code-engineering/07-documentation/quiz.toml) 정답을 실행으로 검증한다."""
import argparse
import doctest
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/07-documentation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_doctest_numpy_repr():
    assert np.mean([1.0, 2.0]) == 1.5                       # 값은 정확히 1.5
    assert int(np.__version__.split(".")[0]) >= 2
    assert repr(np.mean([1.0, 2.0])) == "np.float64(1.5)"   # 출력 모양이 다르다
    doc = ">>> np.mean([1.0, 2.0])\n1.5\n"
    test = doctest.DocTestParser().get_doctest(doc, {"np": np}, "ex", None, 0)
    runner = doctest.DocTestRunner()
    assert runner.run(test, out=lambda s: None).failed == 1
    doc_fixed = ">>> print(np.mean([1.0, 2.0]))\n1.5\n"
    test = doctest.DocTestParser().get_doctest(doc_fixed, {"np": np}, "ex", None, 0)
    assert doctest.DocTestRunner().run(test, out=lambda s: None).failed == 0
    assert correct("doctest-numpy-repr") == [0]


def test_readme_command_drift_is_detectable():
    p = argparse.ArgumentParser(prog="train.py")
    p.add_argument("--learning-rate", type=float)
    with pytest.raises(SystemExit):
        p.parse_args(["--lr", "3e-4"])                      # unrecognized arguments: --lr
    assert correct("readme-command-drift") == [0]


def test_changelog_impact_size():
    assert abs(0.58 / 0.6 - 1) == pytest.approx(0.0333, abs=1e-3)   # 약 3%
    assert correct("changelog-result-impact") == [0]


def test_other_keys():
    assert correct("readme-essentials") == [0, 1, 2]
    assert correct("docstring-research-essentials") == [0]
    assert correct("adr-purpose") == [0]
