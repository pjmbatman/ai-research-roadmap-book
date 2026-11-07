"""4.5 실습: 실험 기록 SQLite 데이터베이스를 **읽기 전용**으로 여는 최소 MCP 서버 (stdio, 표준 라이브러리만).

MCP 명세(집필 시점 2025-11-25 판)의 일부만 구현한 학습용 서버다: initialize, ping, tools/list, tools/call.
실제 서비스에는 공식 SDK를 쓰는 편이 낫다. 이 파일의 목적은 메시지가 실제로 어떻게 오가는지 보이는 것이다.

    python sqlite_server.py --db experiments.db

stdout에는 MCP 메시지(JSON 한 줄)만 쓴다. 로그는 stderr로 보낸다.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18"]
MAX_ROWS = 200

TOOLS = [
    {
        "name": "list_tables",
        "description": "데이터베이스의 테이블 이름과 열 이름을 반환한다.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
    {
        "name": "query",
        "description": f"읽기 전용 SQL(SELECT) 하나를 실행해 최대 {MAX_ROWS}행을 JSON으로 반환한다. 데이터를 바꾸는 문은 실패한다.",
        "inputSchema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "실행할 SELECT 문 하나"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def open_readonly(db_path):
    # mode=ro: SQLite 수준에서 쓰기를 거부한다. 서버 코드에 버그가 있어도 데이터는 바뀌지 않는다.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def text_result(obj, is_error=False):
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False)}], "isError": is_error}


def call_tool(conn, name, args):
    if name == "list_tables":
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return text_result({t: [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')")] for t in names})
    if name == "query":
        sql = args.get("sql")
        if not isinstance(sql, str):
            return text_result({"error": "인자 sql(문자열)이 필요하다"}, is_error=True)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as e:                       # 도구 실행 오류: 모델이 읽고 고칠 수 있게 결과로 돌려준다
            return text_result({"error": f"{type(e).__name__}: {e}"}, is_error=True)
        return text_result({"columns": cols, "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS})
    return None


def handle(conn, msg):
    """요청 하나를 처리해 응답(dict)을 반환한다. 알림(id 없음)에는 None."""
    method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
    if mid is None:                                      # notifications/initialized 등
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "lab-sqlite-readonly", "version": "0.1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = call_tool(conn, params.get("name"), params.get("arguments") or {})
        if result is None:                               # 프로토콜 오류: 없는 도구
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {params.get('name')}"}}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    conn = open_readonly(p.parse_args().db)
    log("lab-sqlite-readonly: 시작")
    for line in sys.stdin:                               # 메시지는 줄바꿈으로 구분된다
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(conn, msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
