"""5.2 실습: '노트북(저장소 루트)에서는 되는데 스크립트에서는 import가 안 되는' 상황을 재현하고 src layout으로 해결한다.

실행: uv run --project labs python labs/python-and-computing/modules-and-imports/reproduce_layout.py
임시 디렉터리에 두 프로젝트(flat layout, src layout)를 만들고, 여러 위치·방식으로 실행한 결과를 표로 출력한다.
src layout 단계는 uv로 임시 가상환경을 만들어 editable 설치를 하므로 uv와 패키지 저장소 접근이 필요하다.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG_INIT = '"""열전도 연구 패키지(데모)."""\n'
PKG_PHYSICS = "def diffusivity(k_w_mk, rho_kg_m3, cp_j_kgk):\n    return k_w_mk / (rho_kg_m3 * cp_j_kgk)\n"
TRAIN = (
    "from mypkg.physics import diffusivity\n"
    "def main():\n"
    "    print(f'alpha = {diffusivity(401.0, 8960.0, 385.0):.3e} m2/s')\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)
PYPROJECT = """[project]
name = "mypkg"
version = "0.1.0"
requires-python = ">=3.11"

[project.scripts]
mypkg-train = "mypkg.train:main"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str], cwd: Path, python: str = sys.executable) -> str:
    """명령을 실행하고 결과를 한 줄로 요약한다(성공이면 stdout, 실패면 마지막 오류 줄)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    cmd = [python if c == "python" else c for c in cmd]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if p.returncode == 0:
        return "성공  " + p.stdout.strip().splitlines()[-1]
    return "실패  " + p.stderr.strip().splitlines()[-1]


def flat_layout(base: Path) -> None:
    proj = base / "flat"
    write(proj / "mypkg/__init__.py", PKG_INIT)
    write(proj / "mypkg/physics.py", PKG_PHYSICS)
    write(proj / "scripts/train.py", TRAIN)
    print("[flat layout] mypkg/가 저장소 루트에 있다")
    cases = [
        ("루트에서 python -c 'import mypkg' (노트북과 같은 상황)", ["python", "-c", "import mypkg; print('import 성공')"], proj),
        ("루트에서 python scripts/train.py", ["python", "scripts/train.py"], proj),
        ("루트에서 python -m scripts.train", ["python", "-m", "scripts.train"], proj),
        ("scripts/에서 python train.py", ["python", "train.py"], proj / "scripts"),
    ]
    for label, cmd, cwd in cases:
        print(f"  {label:<48} {run(cmd, cwd)}")


def src_layout(base: Path) -> None:
    proj = base / "srcproj"
    write(proj / "pyproject.toml", PYPROJECT)
    write(proj / "src/mypkg/__init__.py", PKG_INIT)
    write(proj / "src/mypkg/physics.py", PKG_PHYSICS)
    write(proj / "src/mypkg/train.py", TRAIN)
    print("[src layout] src/mypkg/ + pyproject.toml")
    print(f"  {'설치 전, 루트에서 import mypkg':<48} {run(['python', '-c', 'import mypkg'], proj)}")
    uv = shutil.which("uv")
    if uv is None:
        print("  uv가 없어 설치 단계를 건너뛴다")
        return
    venv = base / "venv"
    subprocess.run([uv, "venv", "-q", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run([uv, "pip", "install", "-q", "--python", py, "-e", str(proj)], check=True)
    other = base / "elsewhere"
    other.mkdir()
    print(f"  {'설치 후, 다른 디렉터리에서 python -m mypkg.train':<48} {run(['python', '-m', 'mypkg.train'], other, py)}")
    print(f"  {'설치 후, 다른 디렉터리에서 mypkg-train':<48} {run([str(venv / 'bin' / 'mypkg-train')], other, py)}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        flat_layout(base)
        src_layout(base)


if __name__ == "__main__":
    main()
