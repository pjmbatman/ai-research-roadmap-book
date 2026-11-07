"""2.4 퀴즈(content/02-linux-and-remote/04-execution-topology/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/02-linux-and-remote/04-execution-topology/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_transfer_time_hours():
    hours = 1e12 * 8 / 1e9 / 3600
    q = Q["transfer-time-hours"]
    assert hours == pytest.approx(2.222, abs=1e-3)
    assert abs(hours - q["answer"]) <= q["tolerance"]
    # bits/bytes를 혼동한 값(약 0.28 h)은 허용 범위 밖이어야 한다
    assert abs(hours / 8 - q["answer"]) > q["tolerance"]


def test_session_survives():
    # (선택지, 세션을 유지하는 것)
    holders = ["gpu1 tmux", "scheduler", "ssh 연결", "노트북"]
    survives = [i for i, h in enumerate(holders) if h not in ("ssh 연결", "노트북")]
    assert correct("session-survives") == survives


def test_single_choice_answers():
    for qid in ["edited-wrong-copy", "remote-control-vs-ssh", "sensitive-data-location"]:
        assert correct(qid) == [0], qid
