"""5.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/python-and-computing/modules-and-imports/cells.py
브라우저(Pyodide)에서도 돌도록 임시 디렉터리에 모듈 파일을 만들어 import한다.
"""
# %% cell 1 — import는 한 번만 실행되고 sys.modules에 캐시된다
import importlib
import sys
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp())          # 데모용 임시 "프로젝트" 디렉터리
sys.path.insert(0, str(root))            # 데모를 위해서만 직접 조작한다
(root / "demo_consts.py").write_text(
    'print("demo_consts 실행됨")\n'
    'G_M_S2 = 9.81\n'
)

import demo_consts
import demo_consts                       # 두 번째 import는 캐시에서 꺼낸다
print("캐시에 있는가:", "demo_consts" in sys.modules)
print("같은 객체인가:", sys.modules["demo_consts"] is demo_consts)
importlib.reload(demo_consts)            # 명시적으로 다시 실행
print("값:", demo_consts.G_M_S2)

# %% cell 2 — from import는 import 시점의 값에 새 이름을 붙인다
from demo_consts import G_M_S2

demo_consts.G_M_S2 = 1.62                # 다른 모듈이 모듈 속성을 바꿨다고 하자
print("모듈 속성      :", demo_consts.G_M_S2)
print("from으로 받은 이름:", G_M_S2)

# %% cell 3 — sys.path 앞쪽의 같은 이름이 표준 라이브러리를 가린다
from importlib.machinery import PathFinder

(root / "statistics.py").write_text("# 우연히 표준 라이브러리와 같은 이름\n")
importlib.invalidate_caches()                        # 실행 중에 새로 만든 파일을 찾게 한다
spec = PathFinder.find_spec("statistics", sys.path)  # sys.path를 앞에서부터 찾는다
print("statistics를 찾은 곳이 데모 디렉터리인가:", Path(spec.origin).parent == root)
(root / "statistics.py").unlink()

# %% cell 4 — __name__과 __main__ 보호
import runpy

(root / "demo_tool.py").write_text(
    'def main():\n'
    '    print("main 실행, __name__ =", __name__)\n'
    'print("모듈 본문 실행, __name__ =", __name__)\n'
    'if __name__ == "__main__":\n'
    '    main()\n'
)
importlib.invalidate_caches()

import demo_tool                                   # 라이브러리로 import
print("---")
runpy.run_module("demo_tool", run_name="__main__")  # python -m demo_tool과 같은 방식

# %% cell 5 — 순환 import
(root / "cyc_a.py").write_text("from cyc_b import helper\nVALUE = 1\n")
(root / "cyc_b.py").write_text("from cyc_a import VALUE\ndef helper():\n    return VALUE\n")
importlib.invalidate_caches()
try:
    import cyc_a
except ImportError as e:
    print("ImportError:", str(e).split(" (")[0])
