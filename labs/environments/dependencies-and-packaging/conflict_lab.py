"""3.4 실습: 일부러 충돌하는 의존성을 만들고, resolver 메시지를 읽고, 고친 뒤 lockfile로 재현한다.

실행: python3 labs/environments/dependencies-and-packaging/conflict_lab.py
      (uv와 네트워크가 필요하다. 모든 작업은 임시 디렉터리에서 한다)
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYPROJECT = """\
[project]
name = "conflict-lab"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
dependencies = [{deps}]

[tool.uv]
environments = ["sys_platform == 'linux'"]   # 이 실습은 Linux에서만 풀리면 된다
"""


def uv(*args, cwd):
    p = subprocess.run(["uv", *args], cwd=cwd, capture_output=True, text=True)
    err = "\n".join(line for line in p.stderr.splitlines() if "UV_NATIVE_TLS" not in line)
    return p.returncode, p.stdout, err


def write(proj, deps):
    (proj / "pyproject.toml").write_text(PYPROJECT.format(deps=", ".join(f'"{d}"' for d in deps)))


def step(title):
    print(f"\n== {title}")


if __name__ == "__main__":
    if shutil.which("uv") is None:
        sys.exit("uv가 필요하다")
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        proj.mkdir()

        step("1. 충돌: pandas==3.1.1 과 numpy<2")
        write(proj, ["pandas==3.1.1", "numpy<2"])
        code, _, err = uv("lock", cwd=proj)
        print(f"uv lock 종료 코드 {code}")
        print(err)

        step("2. 해결: 메시지가 지목한 numpy 제약을 완화한다")
        write(proj, ["pandas==3.1.1", "numpy>=2"])
        code, _, err = uv("lock", cwd=proj)
        print(f"uv lock 종료 코드 {code}")
        _, tree, _ = uv("tree", "--frozen", cwd=proj)
        print(tree.rstrip())

        step("3. uv.lock에 기록된 numpy")
        lock = (proj / "uv.lock").read_text()
        entry = re.search(r'\[\[package\]\]\nname = "numpy"\n.*?(?=\n\[\[package\]\]|\Z)', lock, re.S).group(0)
        lines = entry.splitlines()
        print("\n".join(lines[:4]))
        n_wheels = sum('.whl"' in line for line in lines)
        print(f"... (이 패키지에 대해 기록된 wheel {n_wheels}개, 각 wheel의 sha256 해시 포함)")

        step("4. 재현: pyproject.toml과 uv.lock만 복사해 새 디렉터리에서 sync --locked")
        clone = Path(tmp) / "clone"
        clone.mkdir()
        for f in ("pyproject.toml", "uv.lock"):
            shutil.copy(proj / f, clone / f)
        code, _, _ = uv("sync", "--locked", cwd=clone)
        uv("sync", "--locked", cwd=proj)
        _, a, _ = uv("pip", "freeze", cwd=proj)
        _, b, _ = uv("pip", "freeze", cwd=clone)
        print(f"sync 종료 코드 {code}, 설치 목록 동일: {a == b}")
        print(a.rstrip())
