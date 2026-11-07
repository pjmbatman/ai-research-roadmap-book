"""2.3 퀴즈(content/02-linux-and-remote/03-ssh/quiz.toml) 정답을 계산으로 검증한다."""
import importlib.util
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/02-linux-and-remote/03-ssh/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

_spec = importlib.util.spec_from_file_location("ssh_cfg_solution", HERE / "exercises/ssh-config-lookup/solution.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resolve = _mod.resolve


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_private_key_permissions():
    assert 0o644 & 0o077 != 0            # 그룹·기타 읽기가 열려 있다
    assert 0o600 & 0o077 == 0            # 해결 후
    assert correct("private-key-permissions") == [0]


def test_ssh_config_order():
    text = "Host *\n    User kim\nHost gpu1\n    HostName 10.0.3.21\n    User lab\n"
    r = resolve(text, "gpu1")
    assert r["user"] == "kim" and r["hostname"] == "10.0.3.21"
    assert correct("ssh-config-order") == [0]


def test_local_forward_port():
    spec = "8890:localhost:8888"
    local_port, dest_host, dest_port = spec.split(":")
    assert int(local_port) == Q["local-forward-port"]["answer"]
    assert (dest_host, dest_port) == ("localhost", "8888")


def test_single_choice_answers():
    for qid in ["agent-forwarding-risk", "local-forward-tensorboard", "remote-extension-location", "host-key-changed"]:
        assert correct(qid) == [0], qid
