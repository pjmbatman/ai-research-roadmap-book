"""3.6 실습: 고장 난 환경 세 가지를 임시 디렉터리에 만들고, 증상과 진단 근거를 출력한다.

실행: python3 labs/environments/environment-diagnosis/broken_envs.py
      (Linux, gcc, 네트워크(PyPI)가 필요하다. 시나리오마다 수십 초 걸릴 수 있다)

출력을 읽고 시나리오마다 '증상 → 가설 → 근거 → 원인 → 해결'을 진단 보고서로 쓴다.
스크립트가 마지막에 보여 주는 '해결 확인'은 보고서를 쓴 뒤에 맞춰 본다.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, env=None, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, cwd=cwd)
    return p.returncode, (p.stdout + p.stderr).strip()


def last_line(text):
    return text.splitlines()[-1] if text else ""


def scenario_pip_mismatch(root):
    print("== 시나리오 1: 설치했는데 import가 안 된다")
    a, b = root / "proj-a", root / "proj-b"
    for env in (a, b):
        run([sys.executable, "-m", "venv", str(env)])
    run([str(b / "bin/pip"), "install", "-q", "six"])                  # 사용자는 A에 설치했다고 믿는다
    code, out = run([str(a / "bin/python"), "-c", "import six"])
    print("  증상:", last_line(out))
    _, exe = run([str(a / "bin/python"), "-c", "import sys; print(sys.prefix)"])
    _, pipv = run([str(b / "bin/pip"), "--version"])
    print("  근거: 실행한 python의 sys.prefix =", exe.replace(str(root), "<tmp>"))
    print("  근거: 사용한 pip --version      =", pipv.split(" from ")[1].replace(str(root), "<tmp>"))
    code, _ = run([str(a / "bin/python"), "-m", "pip", "install", "-q", "six"])
    code, _ = run([str(a / "bin/python"), "-c", "import six"])
    print("  해결 확인: python -m pip으로 설치한 뒤 import 종료 코드 =", code)


def scenario_shared_library(root):
    print("== 시나리오 2: 공유 라이브러리를 찾지 못한다")
    lib = root / "vendor" / "lib"
    lib.mkdir(parents=True)
    (root / "toy.c").write_text("int toy_add(int a, int b) { return a + b; }\n")
    run(["gcc", "-shared", "-fPIC", "-o", str(lib / "libtoy.so.1"), str(root / "toy.c")])
    script = "import ctypes; print('toy_add(2, 3) =', ctypes.CDLL('libtoy.so.1').toy_add(2, 3))"
    code, out = run([sys.executable, "-c", script], env={"LD_LIBRARY_PATH": ""})
    print("  증상:", last_line(out))
    _, cache = run(["sh", "-c", "ldconfig -p | grep -c libtoy || true"])
    print("  근거: 시스템 라이브러리 cache(ldconfig -p)에서 libtoy 항목 수 =", cache)
    print("  근거: 파일은 있다 ->", sorted(p.name for p in lib.iterdir()))
    code, out = run([sys.executable, "-c", script], env={"LD_LIBRARY_PATH": str(lib)})
    print("  해결 확인: LD_LIBRARY_PATH에 경로를 넣고 실행 ->", out)


def scenario_version_conflict(root):
    print("== 시나리오 3: 설치는 성공했는데 의존성 제약이 깨져 있다")
    env = root / "proj-c"
    run([sys.executable, "-m", "venv", str(env)])
    pip = str(env / "bin/pip")
    run([pip, "install", "-q", "pandas==3.1.1"])
    run([pip, "install", "-q", "--no-deps", "python-dateutil==2.8.2"])   # 누군가 '급한 불'을 껐다
    code, out = run([pip, "check"])
    print("  증상/근거: pip check 종료 코드", code, "->", last_line(out))
    run([pip, "install", "-q", "python-dateutil>=2.9.0"])
    code, out = run([pip, "check"])
    print("  해결 확인: 제약에 맞게 다시 설치한 뒤 pip check ->", last_line(out))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (scenario_pip_mismatch, scenario_shared_library, scenario_version_conflict):
            d = Path(tmp) / fn.__name__
            d.mkdir()
            fn(d)
