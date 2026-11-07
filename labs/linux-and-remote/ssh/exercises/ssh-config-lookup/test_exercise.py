import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resolve = load().resolve

CONFIG = """
# 연구실 서버
Host bastion
    HostName gate.example.org
    User kim

Host gpu?  gpu-old
    HostName %h.internal
    User lab
    ProxyJump bastion
    IdentityFile ~/.ssh/id_lab

Host * !bastion
    ServerAliveInterval 60

Host *
    User kim
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
"""


def test_problem_example():
    text = "Host gpu*\n    User lab\nHost *\n    User kim\n    ServerAliveInterval 60\n"
    assert resolve(text, "gpu1") == {"user": "lab", "serveraliveinterval": "60"}


def test_first_value_wins():
    r = resolve(CONFIG, "gpu1")
    assert r["user"] == "lab", "gpu? 블록이 Host *보다 먼저 나오므로 User는 lab입니다"
    assert r["serveraliveinterval"] == "60", "Host * !bastion 블록의 60이 먼저 얻어집니다"
    assert r["proxyjump"] == "bastion"


def test_keys_are_lowercase_values_kept():
    r = resolve(CONFIG, "gpu1")
    assert "hostname" in r and "HostName" not in r, "키는 소문자로 반환합니다"
    assert r["hostname"] == "%h.internal", "값은 그대로 둡니다(%h 확장은 하지 않음)"


def test_identityfile_accumulates():
    r = resolve(CONFIG, "gpu1")
    assert r["identityfile"] == ["~/.ssh/id_lab", "~/.ssh/id_ed25519"], "IdentityFile은 맞는 값을 모두 순서대로 모읍니다"


def test_negated_pattern_excludes_block():
    r = resolve(CONFIG, "bastion")
    assert r["serveraliveinterval"] == "30", "bastion에는 'Host * !bastion' 블록이 적용되지 않습니다"
    assert r["hostname"] == "gate.example.org"
    assert "proxyjump" not in r


def test_question_mark_matches_one_char():
    r = resolve(CONFIG, "gpu12")
    assert r["user"] == "kim", "gpu?는 한 글자만 맞습니다. gpu12에는 Host *만 적용됩니다"
    assert "proxyjump" not in r


def test_multiple_patterns_on_host_line():
    assert resolve(CONFIG, "gpu-old")["user"] == "lab", "Host 줄의 패턴 중 하나라도 맞으면 적용됩니다"


def test_global_options_and_equals_syntax():
    text = "User=global\nHost a\n  User a_user\n  Port=2222\n"
    assert resolve(text, "a") == {"user": "global", "port": "2222"}, "첫 Host 줄 이전 설정이 먼저 얻어지고, 키=값 문법도 읽습니다"
    assert resolve(text, "b") == {"user": "global"}


def test_case_insensitive_keys():
    text = "Host x\n  USER first\n  user second\n"
    assert resolve(text, "x") == {"user": "first"}, "USER와 user는 같은 키입니다"
