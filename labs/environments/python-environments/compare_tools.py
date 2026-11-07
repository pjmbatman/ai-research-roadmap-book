"""3.3 실습: 같은 프로젝트(numpy + pandas)를 venv+pip, uv, conda로 만들고 비교한다.

실행: python3 labs/environments/python-environments/compare_tools.py
      (네트워크가 필요하다. 각 도구의 캐시를 임시 디렉터리로 돌려 '처음 설치'에 가깝게 잰다)

재는 것
- 설치 시간: 환경 생성 + 패키지 설치에 걸린 벽시계 시간(네트워크 속도에 크게 좌우된다)
- 디스크: 환경 디렉터리 크기
- 재현: 기록 파일(requirements.txt 또는 uv.lock)만으로 새 디렉터리에 다시 만든 환경이 같은 패키지 목록을 갖는가
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PACKAGES = ["numpy", "pandas"]


def sh(cmd, cwd, env=None):
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def out(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True,
                          capture_output=True, text=True).stdout


def du_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()) / 1e6


def venv_pip(root):
    a, b = root / "a", root / "b"
    a.mkdir(), b.mkdir()
    t0 = time.perf_counter()
    sh([sys.executable, "-m", "venv", ".venv"], a)
    sh([".venv/bin/pip", "install", "--no-cache-dir", *PACKAGES], a)
    elapsed = time.perf_counter() - t0
    freeze = out([".venv/bin/pip", "freeze"], a)
    (b / "requirements.txt").write_text(freeze)                    # 기록 파일
    sh([sys.executable, "-m", "venv", ".venv"], b)
    sh([".venv/bin/pip", "install", "--no-cache-dir", "-r", "requirements.txt"], b)
    same = freeze == out([".venv/bin/pip", "freeze"], b)
    return elapsed, du_mb(a / ".venv"), len(freeze.splitlines()), same


def uv(root):
    if shutil.which("uv") is None:
        return None
    a, b = root / "a", root / "b"
    a.mkdir(), b.mkdir()
    env = {"UV_CACHE_DIR": str(root / "uv-cache")}
    t0 = time.perf_counter()
    sh(["uv", "init", "--bare", "--no-workspace", "--python", sys.executable], a, env)
    sh(["uv", "add", *PACKAGES], a, env)                         # .venv와 uv.lock을 만든다
    elapsed = time.perf_counter() - t0
    for f in ("pyproject.toml", "uv.lock"):                       # 기록 파일
        shutil.copy(a / f, b / f)
    sh(["uv", "sync", "--locked"], b, {"UV_CACHE_DIR": str(root / "uv-cache-2")})
    fa = out(["uv", "pip", "freeze"], a, env)
    same = fa == out(["uv", "pip", "freeze"], b, env)
    return elapsed, du_mb(a / ".venv"), len(fa.splitlines()), same


def conda(root):
    exe = shutil.which("mamba") or shutil.which("conda")
    if exe is None:
        return None
    env_dir = root / "env"
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    t0 = time.perf_counter()
    sh([exe, "create", "-y", "-p", str(env_dir), f"python={pyver}", *PACKAGES], root,
       {"CONDA_PKGS_DIRS": str(root / "pkgs")})
    elapsed = time.perf_counter() - t0
    n = len([line for line in out([exe, "list", "-p", str(env_dir)], root).splitlines() if not line.startswith("#")])
    return elapsed, du_mb(env_dir), n, None                       # 재현 비교는 conda-lock 등으로 따로


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}, 패키지 {PACKAGES}\n")
    print(f"{'도구':10s} {'설치 시간':>9s} {'디스크':>9s} {'패키지 수':>8s}  재현")
    for name, fn in [("venv+pip", venv_pip), ("uv", uv), ("conda", conda)]:
        with tempfile.TemporaryDirectory() as tmp:
            r = fn(Path(tmp))
        if r is None:
            print(f"{name:10s} 설치되어 있지 않아 건너뜀")
            continue
        t, mb, n, same = r
        print(f"{name:10s} {t:8.1f}s {mb:7.0f}MB {n:8d}  {'-' if same is None else '같음' if same else '다름'}")
