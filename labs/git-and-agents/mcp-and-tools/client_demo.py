"""4.5 실습: MCP client가 하는 일을 손으로 해 본다. sqlite_server.py를 subprocess로 띄우고 stdio로 메시지를 주고받는다.

    python client_demo.py [experiments.db]

에이전트 도구(host)는 이 일을 자동으로 하고, 도구 결과를 언어 모델에 넘긴다.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


class StdioClient:
    def __init__(self, db_path):
        self.proc = subprocess.Popen([sys.executable, str(HERE / "sqlite_server.py"), "--db", str(db_path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8")
        self.next_id = 1

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"], self.next_id = self.next_id, self.next_id + 1
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return None if notify else json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def tool_text(reply):
    """tools/call 결과의 첫 text 내용을 파싱한다."""
    return json.loads(reply["result"]["content"][0]["text"])


def main(db_path):
    c = StdioClient(db_path)
    init = c.send("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                 "clientInfo": {"name": "demo-client", "version": "0.1"}})
    print("initialize →", init["result"]["protocolVersion"], init["result"]["capabilities"])
    c.send("notifications/initialized", notify=True)
    print("tools/list →", [t["name"] for t in c.send("tools/list")["result"]["tools"]])

    r = c.send("tools/call", {"name": "query", "arguments": {
        "sql": "SELECT material, AVG(conductivity_s_per_m) FROM runs GROUP BY material ORDER BY material"}})
    print("query →", tool_text(r)["rows"])

    r = c.send("tools/call", {"name": "query", "arguments": {"sql": "DELETE FROM runs"}})
    print("DELETE → isError =", r["result"]["isError"], "|", tool_text(r)["error"])

    r = c.send("tools/call", {"name": "query", "arguments": {"sql": "SELECT notes FROM runs WHERE run_id = 'r005'"}})
    print("notes(r005) →", tool_text(r)["rows"][0][0][:60] + "...")

    print("없는 도구 →", c.send("tools/call", {"name": "drop_table", "arguments": {}})["error"])
    print("남은 행 수 →", tool_text(c.send("tools/call", {"name": "query", "arguments": {
        "sql": "SELECT COUNT(*) FROM runs"}}))["rows"][0][0])
    c.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        sys.path.insert(0, str(HERE))
        from make_db import make
        with tempfile.TemporaryDirectory() as d:
            make(Path(d) / "experiments.db")
            main(Path(d) / "experiments.db")
