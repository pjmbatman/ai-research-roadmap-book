"""2.2 퀴즈(content/02-linux-and-remote/02-processes-and-monitoring/quiz.toml) 정답을 실제 프로세스 실험으로 검증한다."""
import os
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/02-linux-and-remote/02-processes-and-monitoring/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
BASH = shutil.which("bash")

CHILD = r'''
import os, signal, sys, time
def on_term(signum, frame):
    os.write(1, b"saved\n")  # print는 신호가 print("start") 도중에 오면 재진입 오류가 난다
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
try:
    print("start", flush=True)
    time.sleep(30)
finally:
    print("finally", flush=True)
'''


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run_child(sig):
    p = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True)
    p.stdout.readline()
    p.send_signal(sig)
    out, _ = p.communicate(timeout=10)
    return out.split(), p.returncode


def test_sigterm_before_sigkill():
    out, code = run_child(signal.SIGTERM)
    assert out == ["saved", "finally"] and code == 0       # handler와 finally가 실행된다
    out, code = run_child(signal.SIGKILL)
    assert out == [] and code == -signal.SIGKILL            # 아무것도 실행되지 않는다
    assert correct("sigterm-before-sigkill") == [0]


@pytest.mark.skipif(BASH is None, reason="bash 없음")
def test_exit_code_137():
    r = subprocess.run([BASH, "-c", "sleep 30 & p=$!; kill -9 $p; wait $p; echo $?"], capture_output=True, text=True)
    assert int(r.stdout.strip()) == Q["exit-code-137"]["answer"] == 128 + signal.SIGKILL


@pytest.mark.skipif(shutil.which("lsof") is None, reason="lsof 없음")
def test_deleted_but_open_file_holds_space(tmp_path):
    path = tmp_path / "debug.log"
    with open(path, "w") as f:
        f.write("x" * 1_000_000)
        f.flush()
        os.remove(path)
        out = subprocess.run(["lsof", "-a", "+L1", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        assert "debug.log (deleted)" in out
    assert correct("disk-full-deleted-open") == [0]


def test_gpu_util_meaning():
    # 작은 커널이 쉬지 않고 이어지면: 커널 실행 시간 비율 100%, SM 활용 10%
    busy, sm_frac = [1] * 100, 0.1
    util = sum(1 for b in busy if b) / len(busy)
    assert util == 1.0 and util * sm_frac == pytest.approx(0.1)
    assert correct("gpu-util-meaning") == [0]


def test_ghost_gpu_process_choices():
    assert correct("ghost-gpu-process") == [0, 1, 2]
    assert correct("nohup-vs-tmux") == [0]
