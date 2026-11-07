"""2.2 실습: 공유 서버를 위험하게 하지 않는 범위에서 세 가지 장애를 일부러 만들고 흔적을 확인한다.

실행: python3 labs/linux-and-remote/processes-and-monitoring/drills.py
- 메모리: 자식 프로세스의 주소 공간을 2 GiB로 제한(RLIMIT_AS)하고 4 GiB를 요청한다. 실제 OOM killer는 건드리지 않는다.
- 끊긴 세션: 접속이 끊길 때 터미널의 프로세스가 받는 SIGHUP을 직접 보낸다. 일반 프로세스와 nohup 프로세스를 비교한다.
- 디스크: 큰 파일을 연 채로 지워, '지웠는데 공간이 돌아오지 않는' 상황을 lsof +L1로 찾는다.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

GIB = 1024**3


def drill_memory():
    child = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({2 * GIB}, {2 * GIB}))\n"
        f"buf = bytearray({4 * GIB})\n"
    )
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    print(f"[메모리] 종료 코드 {r.returncode}, stderr 마지막 줄: {r.stderr.strip().splitlines()[-1]}")


def drill_hangup():
    plain = subprocess.Popen(["sleep", "60"])
    nohup = subprocess.Popen(["nohup", "sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)                              # nohup이 SIGHUP 무시를 설정하고 sleep을 실행할 시간
    for p in (plain, nohup):
        p.send_signal(signal.SIGHUP)
    time.sleep(0.5)
    for name, p in (("sleep", plain), ("nohup sleep", nohup)):
        state = "살아 있음" if p.poll() is None else f"종료(returncode {p.returncode})"
        print(f"[끊긴 세션] SIGHUP 후 {name:11s}: {state}")
        p.kill()
        p.wait()


def drill_deleted_open_file():
    if shutil.which("lsof") is None:
        print("[디스크] lsof가 없어 건너뜀")
        return
    d = tempfile.mkdtemp()
    path = os.path.join(d, "debug.log")
    with open(path, "wb") as f:
        f.write(b"x" * 50 * 1024**2)            # 50 MiB
        f.flush()
        os.remove(path)
        out = subprocess.run(["lsof", "-a", "+L1", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if "debug.log" in l)
        size_mib = int(line.split()[6]) / 1024**2
        print(f"[디스크] 지웠지만 열려 있는 파일: {os.path.basename(line.split()[-2])} {line.split()[-1]}, {size_mib:.0f} MiB")
    os.rmdir(d)


if __name__ == "__main__":
    drill_memory()
    drill_hangup()
    drill_deleted_open_file()
