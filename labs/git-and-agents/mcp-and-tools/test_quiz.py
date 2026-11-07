"""4.5 퀴즈(content/04-git-and-agents/05-mcp-and-tools/quiz.toml) 중 계산으로 확인할 수 있는 정답을 검증한다."""
import sqlite3
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/04-git-and-agents/05-mcp-and-tools/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_clients_per_server():
    servers = ["filesystem", "experiments-db", "literature"]
    clients = {s: f"client-{s}" for s in servers}                  # 서버마다 client 하나 (1:1)
    assert len(clients) == Q["clients-per-server"]["answer"]


def test_readonly_enforcement(tmp_path):
    db = tmp_path / "e.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (x)")
    con.execute("INSERT INTO runs VALUES (1)")
    con.commit()
    con.close()
    sneaky = "/* select */ DELETE FROM runs"
    assert "select" in sneaky.lower()                              # 문자열 검사는 통과한다
    ro = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute(sneaky)                                         # 읽기 전용 연결은 거부한다
    assert ro.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert correct("readonly-enforcement") == [0]


def test_non_computational_answers():
    assert correct("model-controlled-primitive") == [0]
    assert correct("stdio-credentials") == [0]
    assert correct("least-privilege-token") == [0]
    assert correct("tool-output-instructions") == [0]
