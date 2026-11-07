"""3.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/python-environments/cells.py
"""
# %% cell 1 — import는 sys.path의 앞에서부터 찾는다
sys_path = ["<project>",
            "/home/me/.local/lib/python3.12/site-packages",   # user site (pip install --user)
            "/usr/lib/python3/dist-packages"]                  # 시스템 패키지(apt)
installed = {
    "/home/me/.local/lib/python3.12/site-packages": {"numpy": "1.26.4"},
    "/usr/lib/python3/dist-packages": {"numpy": "2.1.0", "scipy": "1.13.0"},
}

def resolve(pkg):
    for d in sys_path:
        if pkg in installed.get(d, {}):
            return installed[d][pkg], d
    return None, None

for pkg in ["numpy", "scipy", "torch"]:
    version, where = resolve(pkg)
    print(f"import {pkg:5s} ->", f"{version}  {where}" if version else "ModuleNotFoundError")

# %% cell 2 — 가상환경은 디렉터리 하나다
import tempfile
import venv
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    env_dir = Path(tmp) / ".venv"
    venv.create(env_dir, with_pip=False, symlinks=True)       # python -m venv --without-pip
    cfg = (env_dir / "pyvenv.cfg").read_text()
    for line in cfg.splitlines():
        if line.split(" = ")[0] in ("home", "include-system-site-packages", "version"):
            print("pyvenv.cfg |", line)
    print("bin/       |", sorted(p.name for p in (env_dir / "bin").iterdir()))
    py = env_dir / "bin" / "python"
    print("python은   |", "심볼릭 링크 ->", py.resolve().parent if py.is_symlink() else "복사본")

# %% cell 3 — 지금 실행 중인 python은 어느 환경인가
import site
import sys
from pathlib import Path

in_venv = sys.prefix != sys.base_prefix
print("가상환경 안인가   :", in_venv)
print("환경 디렉터리 이름 :", Path(sys.prefix).name)
print("base 인터프리터    :", sys.base_prefix)
print("user site 사용     :", site.ENABLE_USER_SITE)
for p in sys.path[1:]:
    label = p.replace(sys.prefix, "<env>") if in_venv else p
    print("  sys.path:", label.replace(sys.base_prefix, "<base>"))
