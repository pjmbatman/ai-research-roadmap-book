"""2.1 퀴즈(content/02-linux-and-remote/01-linux-shell/quiz.toml) 정답을 계산과 실제 bash 실행으로 검증한다."""
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/02-linux-and-remote/01-linux-shell/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash 없음")


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run_bash(script, cwd=None):
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, cwd=cwd)


def test_chmod_octal_value():
    s = "-rw-r-----"[1:]
    value = "".join(str(4 * (s[i] == "r") + 2 * (s[i + 1] == "w") + (s[i + 2] == "x")) for i in range(0, 9, 3))
    assert int(value) == Q["chmod-octal-value"]["answer"] == 640


@needs_bash
def test_set_e_pitfalls():
    lines = ["false", "false | tee /dev/null", "x=$(false)", 'false || echo "계속"']
    stops = []
    for i, line in enumerate(lines):
        r = run_bash(f"set -e\n{line}\necho REACHED")
        if "REACHED" not in r.stdout:
            stops.append(i)
    assert correct("set-e-pitfalls") == stops == [0, 2]


@needs_bash
def test_redirect_order(tmp_path):
    cmd = "echo out; echo err >&2"
    variants = [f"({cmd}) > log 2>&1", f"({cmd}) 2>&1 > log", f"({cmd}) > log", f"({cmd}) | tee log"]
    both = []
    for i, v in enumerate(variants):
        run_bash(v, cwd=tmp_path)
        text = (tmp_path / "log").read_text()
        if "out" in text and "err" in text:
            both.append(i)
        os.remove(tmp_path / "log")
    assert correct("redirect-order") == both == [0]


@needs_bash
def test_symlink_vs_hardlink(tmp_path):
    r = run_bash("echo data > data.h5; ln -s data.h5 sym.h5; ln data.h5 hard.h5; rm data.h5;"
                 " cat hard.h5; cat sym.h5 || echo BROKEN", cwd=tmp_path)
    assert r.stdout.split() == ["data", "BROKEN"]
    assert correct("symlink-vs-hardlink") == [1]


def test_path_first_match():
    path = "/home/kim/.local/bin:/usr/bin:/opt/conda/envs/proj/bin".split(":")
    has = {"/home/kim/.local/bin", "/opt/conda/envs/proj/bin"}
    first = next(d for d in path if d in has)
    assert first == "/home/kim/.local/bin"
    assert correct("path-first-match") == [0]
