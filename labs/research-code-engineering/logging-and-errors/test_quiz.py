"""6.5 퀴즈(content/06-research-code-engineering/05-logging-and-errors/quiz.toml) 정답을 실행으로 검증한다."""
import io
import logging
import math
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/05-logging-and-errors/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_logger_handler_levels():
    logger = logging.getLogger("quiz.levels")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console, file_ = io.StringIO(), io.StringIO()
    for stream, level in [(console, logging.WARNING), (file_, logging.DEBUG)]:
        h = logging.StreamHandler(stream)
        h.setLevel(level)
        logger.addHandler(h)
    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    assert file_.getvalue().splitlines() == ["i", "w", "e"]
    assert console.getvalue().splitlines() == ["w", "e"]
    assert Q["logger-handler-levels"]["answer"] == 3


def test_nan_equality_never_fires():
    loss = float("nan")
    assert not (loss == float("nan"))
    assert math.isnan(loss) and not math.isfinite(loss)
    assert float("inf") == float("inf")
    assert correct("nan-equality") == [0]


def test_assert_removed_under_O():
    code = "assert False, 'gone'"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode != 0
    assert subprocess.run([sys.executable, "-O", "-c", code], capture_output=True).returncode == 0
    assert correct("assert-for-validation") == [0]


def test_repeated_warning_shown_once_by_default():
    def bad_batch():
        return np.log(np.array([0.0]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(50):
            bad_batch()
    assert len(caught) == 1
    assert correct("repeated-warning-once") == [0]


def test_bare_except_swallows_keyboard_interrupt():
    swallowed = False
    try:
        raise KeyboardInterrupt
    except:  # noqa: E722 — 퀴즈 피드백의 주장을 확인한다
        swallowed = True
    assert swallowed
    assert issubclass(KeyboardInterrupt, BaseException) and not issubclass(KeyboardInterrupt, Exception)
    assert correct("silent-except") == [0]
