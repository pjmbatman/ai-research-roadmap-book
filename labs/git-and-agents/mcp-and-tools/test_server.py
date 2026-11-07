"""4.5 실습 서버(sqlite_server.py)가 명세의 메시지 형식대로 응답하고, 쓰기를 거부하는지 확인한다."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_db, client_demo = load("make_db"), load("client_demo")


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "experiments.db"
    make_db.make(db)
    c = client_demo.StdioClient(db)
    yield c, db
    c.close()


def init(c, version="2025-11-25"):
    r = c.send("initialize", {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    c.send("notifications/initialized", notify=True)        # 알림에는 응답이 없어야 한다
    return r


def test_initialize_and_version_negotiation(client):
    c, _ = client
    r = init(c)
    assert r["jsonrpc"] == "2.0" and r["id"] == 1
    assert r["result"]["protocolVersion"] == "2025-11-25"
    assert "tools" in r["result"]["capabilities"]
    assert c.send("ping") == {"jsonrpc": "2.0", "id": 2, "result": {}}   # 알림 뒤 다음 응답이 바로 ping의 것


def test_unknown_version_gets_server_latest(tmp_path):
    db = tmp_path / "e.db"
    make_db.make(db)
    c = client_demo.StdioClient(db)
    assert init(c, "1999-01-01")["result"]["protocolVersion"] == "2025-11-25"
    c.close()


def test_tools_list_has_schemas(client):
    c, _ = client
    init(c)
    tools = {t["name"]: t for t in c.send("tools/list")["result"]["tools"]}
    assert set(tools) == {"list_tables", "query"}
    assert tools["query"]["inputSchema"]["required"] == ["sql"]
    assert all(t["inputSchema"]["type"] == "object" for t in tools.values())


def test_query_returns_rows(client):
    c, _ = client
    init(c)
    r = c.send("tools/call", {"name": "query", "arguments": {"sql": "SELECT COUNT(*) FROM runs WHERE temp_k > 300"}})
    assert r["result"]["isError"] is False
    assert client_demo.tool_text(r)["rows"] == [[2]]


@pytest.mark.parametrize("sql", ["DELETE FROM runs", "UPDATE runs SET temp_k = 0", "DROP TABLE runs",
                                 "INSERT INTO runs VALUES ('x', 'y', 1, 1, '')"])
def test_writes_are_rejected_and_data_unchanged(client, sql):
    c, db = client
    init(c)
    r = c.send("tools/call", {"name": "query", "arguments": {"sql": sql}})
    assert r["result"]["isError"] is True                          # 도구 실행 오류(프로토콜 오류 아님)
    assert "readonly" in client_demo.tool_text(r)["error"]
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 5


def test_injection_text_is_returned_as_data(client):
    c, db = client
    init(c)
    r = c.send("tools/call", {"name": "query", "arguments": {"sql": "SELECT notes FROM runs WHERE run_id = 'r005'"}})
    assert "ignore your previous instructions" in client_demo.tool_text(r)["rows"][0][0]
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 5


def test_protocol_errors(client):
    c, _ = client
    init(c)
    assert c.send("tools/call", {"name": "drop_table", "arguments": {}})["error"]["code"] == -32602
    assert c.send("resources/list")["error"]["code"] == -32601    # 선언하지 않은 기능


def test_bad_argument_is_tool_error(client):
    c, _ = client
    init(c)
    r = c.send("tools/call", {"name": "query", "arguments": {}})
    assert r["result"]["isError"] is True
