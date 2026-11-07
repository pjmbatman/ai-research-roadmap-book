---
id: python-and-computing/modules-and-imports
title: "모듈·패키지·import 시스템"
title_en: "Modules, Packages, and Imports"
tier: core
status: draft
prerequisites: [python-and-computing/python-language]
---

# 모듈·패키지·import 시스템

"노트북에서는 되는데 스크립트로 돌리면 `ModuleNotFoundError`가 난다." "서버에서는 되는데 내 노트북에서는 다른 함수가 불린다." 연구 코드가 한 파일을 넘어서는 순간 이런 문제가 생긴다. 많은 사람이 `sys.path.append("..")` 한 줄로 넘어가지만, 그 줄은 실행 위치가 바뀌면 다시 깨지고, 왜 깨졌는지 아무도 설명하지 못하는 코드를 남긴다.

이 장은 `import` 한 줄이 실제로 무엇을 하는지, 즉 어디에서 파일을 찾고, 언제 실행하고, 결과를 어디에 저장하는지를 정확히 설명한다. 그 위에서 어디서 실행해도 같은 방식으로 import되는 패키지 구조(src layout)와, 라이브러리 코드와 실행 진입점을 분리하는 방법을 다룬다. 환경과 의존성 관리는 3.3–3.4장, 프로젝트 전체 구조는 6.1장에서 이어진다.

## 학습 목표

- import가 동작하는 방식을 설명한다
- src layout 패키지를 구성한다
- 실행 진입점을 설계한다

## import의 동작

**모듈**(module)은 Python 코드를 담은 파일 하나(`physics.py`)이고, import하면 **모듈 객체**가 된다. 모듈 안의 함수·클래스·변수는 그 객체의 속성이다. `import physics`는 다음 세 단계로 동작한다.

1. **캐시 확인.** `sys.modules`(이름 → 모듈 객체의 dict)에 `"physics"`가 있으면 그 객체를 꺼내 바로 4단계로 간다.
2. **찾기.** 없으면 `sys.path`에 있는 디렉터리를 **앞에서부터 차례로** 뒤져 `physics.py`나 `physics/` 패키지를 찾는다. 처음 찾은 것을 쓰고 더는 찾지 않는다.
3. **실행.** 새 모듈 객체를 만들어 `sys.modules`에 **먼저 등록한 뒤**, 파일의 최상위 코드를 위에서 아래로 실행한다. 최상위의 `def`, `class`, 할당이 모듈의 속성이 된다.
4. **이름 붙이기.** 현재 scope에 이름 `physics`를 만들어 모듈 객체에 붙인다.

5.1장의 언어로 말하면, import는 "모듈 객체를 (필요하면 만들고) 이름표를 붙이는 일"이다. 그 결과 모듈의 최상위 코드는 **프로세스당 한 번** 실행된다.

```python {exec="browser" session="imports"}
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
```

```output
demo_consts 실행됨
캐시에 있는가: True
같은 객체인가: True
demo_consts 실행됨
값: 9.81
```

`import`를 두 번 했지만 "실행됨"은 `reload` 전까지 한 번만 출력되었다. 노트북에서 모듈 파일을 고친 뒤 셀을 다시 실행해도 바뀐 내용이 반영되지 않는 이유가 이것이다. `import`는 캐시에서 옛 모듈을 꺼낼 뿐이다. `importlib.reload`는 같은 모듈 객체 안에서 코드를 다시 실행하지만, 이미 다른 곳에서 `from ... import`로 가져간 이름이나 옛 클래스로 만든 인스턴스는 바꾸지 못한다. 확실한 방법은 커널을 재시작하는 것이다.

::quiz{id="import-runs-once"}

**`import m`과 `from m import x`.** 앞의 형태는 모듈 객체에 이름을 붙이고, 뒤의 형태는 모듈을 import한 뒤 **그 순간** `m.x`가 가리키는 객체에 새 이름 `x`를 붙인다. 이후 누군가 `m.x`를 다른 객체로 바꿔도, 이미 붙인 이름 `x`는 따라가지 않는다.

```python {exec="browser" session="imports"}
from demo_consts import G_M_S2

demo_consts.G_M_S2 = 1.62                # 다른 모듈이 모듈 속성을 바꿨다고 하자
print("모듈 속성      :", demo_consts.G_M_S2)
print("from으로 받은 이름:", G_M_S2)
```

```output
모듈 속성      : 1.62
from으로 받은 이름: 9.81
```

모듈을 전역 설정 저장소로 쓰고 실행 중에 값을 바꾸는 코드가 위험한 이유다. 어떤 모듈은 바뀐 값을, 어떤 모듈은 옛 값을 보게 된다. 설정은 인자로 명시적으로 넘기는 것이 원칙이다(6.3장).

::quiz{id="from-import-binding"}

**`sys.path`는 어떻게 정해지는가.** Python이 시작할 때 `sys.path`는 대략 다음 순서로 채워진다.

1. **실행 방식에 따라 정해지는 첫 항목.** `python script.py`이면 `script.py`가 있는 **디렉터리**, `python -m mod`이면 **현재 작업 디렉터리**, `python -c ...`와 대화형 인터프리터이면 빈 문자열(현재 작업 디렉터리를 뜻한다). Jupyter 커널은 보통 노트북이 있는 디렉터리를 작업 디렉터리로 시작한다.
2. 환경 변수 `PYTHONPATH`에 적힌 디렉터리.
3. 표준 라이브러리 디렉터리.
4. 현재 환경의 `site-packages`. `pip install`로 설치한 패키지가 여기에 있다.

첫 항목이 실행 방식마다 다르다는 점이 "노트북에서는 되는데 스크립트에서는 안 되는" 문제의 거의 전부다. 그리고 첫 항목이 표준 라이브러리**보다 앞에** 있기 때문에, 프로젝트 디렉터리에 표준 라이브러리와 같은 이름의 파일이 있으면 그 파일이 이긴다.

```python {exec="browser" session="imports"}
from importlib.machinery import PathFinder

(root / "statistics.py").write_text("# 우연히 표준 라이브러리와 같은 이름\n")
importlib.invalidate_caches()                        # 실행 중에 새로 만든 파일을 찾게 한다
spec = PathFinder.find_spec("statistics", sys.path)  # sys.path를 앞에서부터 찾는다
print("statistics를 찾은 곳이 데모 디렉터리인가:", Path(spec.origin).parent == root)
(root / "statistics.py").unlink()
```

```output
statistics를 찾은 곳이 데모 디렉터리인가: True
```

실험 스크립트 옆에 `random.py`, `statistics.py`, `copy.py`, `test.py`, `numpy.py` 같은 파일을 만들면, 그 디렉터리에서 실행하는 모든 코드(그리고 그 코드가 import하는 라이브러리)가 표준 모듈 대신 그 파일을 가져간다. 오류는 엉뚱한 곳(`AttributeError: module 'random' has no attribute 'seed'`)에서 난다. 이미 `sys.modules`에 올라간 모듈은 캐시에서 꺼내므로 가려지지 않는다는 점이 증상을 더 헷갈리게 만든다.

**절대 import와 상대 import.** `import heatlab.models.cnn`처럼 최상위 패키지부터 쓰는 것이 절대 import, `from .physics import diffusivity`처럼 점으로 시작하는 것이 상대 import다. 상대 import의 점은 **파일 시스템 경로가 아니라 패키지 계층**을 가리킨다. 점 하나는 현재 모듈이 속한 패키지(`__package__`), 점 둘은 그 위 패키지다. 그래서 상대 import는 모듈이 패키지의 일부로 import되었을 때만 동작한다. 연습문제에서 이 변환 규칙을 직접 구현한다.

## 패키지 구성

**패키지**(package)는 모듈을 담은 디렉터리다. `heatlab/models/cnn.py`가 있으면 `import heatlab.models.cnn`은 `heatlab`, `heatlab.models`, `heatlab.models.cnn`을 차례로 import한다. 각 디렉터리의 `__init__.py`는 그 패키지가 import될 때 실행되는 모듈 본문이다. `__init__.py`가 없는 디렉터리도 namespace package로 import될 수 있지만, 이는 여러 배포판이 한 이름 공간을 나누어 쓰는 특수한 경우를 위한 기능이다. 연구 패키지에서는 `__init__.py`를 두어 패키지 경계를 분명히 한다.

`__init__.py`에는 짧은 것만 둔다. 하위 모듈의 핵심 이름을 다시 내보내는 정도(`from .physics import diffusivity`)는 사용자 편의에 좋지만, 무거운 라이브러리를 import하거나 파일을 읽는 코드를 넣으면 패키지의 어떤 부분을 import하든 그 비용을 치른다.

**flat layout과 src layout.** 패키지 디렉터리를 저장소 어디에 두는가에 두 가지 관례가 있다.

```text
flat layout                    src layout
heat-project/                  heat-project/
├── pyproject.toml             ├── pyproject.toml
├── heatlab/                   ├── src/
│   ├── __init__.py            │   └── heatlab/
│   └── physics.py             │       ├── __init__.py
├── scripts/train.py           │       └── physics.py
└── tests/test_physics.py      ├── scripts/train.py
                               └── tests/test_physics.py
```

flat layout에서는 저장소 루트에서 실행한 Python이 현재 디렉터리 덕분에 `heatlab`을 **우연히** 찾는다. 루트에서 연 노트북과 `pytest`는 되고, `scripts/`에서 실행한 스크립트나 다른 디렉터리의 코드는 안 된다. src layout은 패키지를 `src/` 아래로 내려 이 우연을 없앤다. 루트에서도 `import heatlab`이 실패하므로 **설치해야만** 쓸 수 있고, 일단 설치하면 어디에서 실행하든 같은 방식(`site-packages`를 통해)으로 찾는다. 테스트도 저장소의 작업 파일이 아니라 설치된 패키지를 검사하게 되어, `pyproject.toml`에 빠뜨린 하위 패키지나 데이터 파일이 테스트에서 드러난다.

설치는 `pyproject.toml`에 패키지 이름과 build backend를 적고, 개발 중에는 **editable 설치**(`pip install -e .` 또는 `uv pip install -e .`)를 한다. editable 설치는 `site-packages`에 소스 디렉터리를 가리키는 연결만 만들기 때문에, 소스를 고치면 다시 설치하지 않아도 바로 반영된다(실행 중인 프로세스에서는 앞의 캐시 규칙이 그대로 적용된다). `pyproject.toml`의 작성법과 의존성 선언은 3.4장에서 다룬다.

:::note[분야 예시]
기후 자료 분석 그룹에서 한 사람이 `regrid.py`를 `analysis/` 아래에 두고 노트북에서 `from analysis.regrid import ...`로 쓰다가, 같은 코드를 클러스터의 배치 작업(2.5장)으로 옮기면 작업이 시작되는 디렉터리가 달라 import가 실패한다. 경로를 하드코딩해 고치면 다른 사람의 계정에서 다시 깨진다. 공용 코드를 src layout 패키지로 만들어 각자의 환경에 editable 설치하면, 노트북·배치 작업·테스트가 모두 같은 import 문으로 같은 코드를 쓴다.
:::

## 실행 방식

같은 파일도 **import될 때**와 **직접 실행될 때** 다르게 동작할 수 있다. 그 차이를 만드는 것이 모듈 속성 `__name__`이다. import된 모듈의 `__name__`은 모듈 이름(`"heatlab.train"`)이고, 직접 실행된 모듈의 `__name__`은 `"__main__"`이다.

```python {exec="browser" session="imports"}
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
```

```output
모듈 본문 실행, __name__ = demo_tool
---
모듈 본문 실행, __name__ = __main__
main 실행, __name__ = __main__
```

import할 때는 모듈 본문만 실행되고 `main()`은 실행되지 않았다. `if __name__ == "__main__":` 보호가 그 차이를 만든다. 보호 없이 모듈 최상위에 실험 실행 코드를 두면, 다른 모듈이 그 파일의 함수 하나를 import하려 할 때 실험 전체가 돌아간다. `multiprocessing`처럼 자식 프로세스가 주 모듈을 다시 import하는 경우에도 이 보호가 필요하다(5.5장).

**`python script.py`와 `python -m package.module`.** 두 실행 방식은 세 가지가 다르다.

| | `python heatlab/train.py` | `python -m heatlab.train` |
|---|---|---|
| 파일을 찾는 방법 | 주어진 파일 경로 | `sys.path`에서 모듈 이름으로 |
| `sys.path[0]` | 파일이 있는 디렉터리(`heatlab/`) | 현재 작업 디렉터리 |
| `__package__` | 비어 있음 → 상대 import 불가 | `"heatlab"` → 상대 import 가능 |

파일 경로 실행에서 `sys.path[0]`이 파일의 디렉터리라는 점 때문에, 패키지를 설치하지 않았다면 `heatlab/train.py` 안의 `import heatlab`은 실패하고 대신 `heatlab/` 안의 다른 모듈이 최상위 모듈처럼 import된다. 같은 파일이 `heatlab.physics`와 `physics`라는 두 이름으로 두 번 import되어, 두 개의 서로 다른 모듈 객체(그리고 두 벌의 전역 상태)가 생기는 미묘한 버그도 여기서 나온다. 패키지 안의 모듈은 `python -m`으로 실행한다. 패키지 디렉터리에 `__main__.py`를 두면 `python -m heatlab`으로 패키지 자체를 실행할 수 있다.

::quiz{id="script-sys-path"}

## 진입점

연구 코드가 커지면 "무엇을 실행하는가"와 "무엇을 재사용하는가"를 분리해야 한다. 원칙은 다음과 같다.

- **라이브러리 코드**(패키지 안의 모듈)는 import했을 때 아무것도 실행하지 않는다. 함수와 클래스만 정의한다. 인자 파싱, 파일 경로, 로깅 설정 같은 "실행 환경"에 대한 결정을 하지 않는다.
- **진입점**(entry point)은 인자를 받아 라이브러리 함수를 조립해 호출하는 얇은 층이다. `main(argv: list[str] | None = None) -> int`처럼 인자를 받고 종료 코드를 돌려주는 함수로 만들면, 명령행에서도 테스트에서도 같은 함수를 부를 수 있다.

진입점을 실행하는 방법은 세 가지다.

1. `python -m heatlab.train` — 설치된 패키지의 모듈을 실행한다. 별도 설정이 필요 없다.
2. **console script.** `pyproject.toml`의 `[project.scripts]`에 `heat-train = "heatlab.train:main"`처럼 등록하면, 설치할 때 환경의 `bin/`(Windows는 `Scripts\`)에 `heat-train`이라는 실행 파일이 만들어진다. 이 실행 파일은 `heatlab.train`을 import해 `main()`을 부른다. 사용자는 패키지 구조를 몰라도 명령 이름만 알면 된다.
3. `scripts/` 디렉터리의 스크립트 — 패키지를 **설치한 환경에서** 실행하면 된다. 스크립트 자체는 패키지 함수를 import해 부르는 몇 줄이어야 한다.

```python
# src/heatlab/train.py — 라이브러리 함수와 진입점을 분리한 예
import argparse

from heatlab.physics import diffusivity

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-w-mk", type=float, required=True)
    args = parser.parse_args(argv)                 # argv=None이면 sys.argv[1:]
    print(diffusivity(args.k_w_mk, 8960.0, 385.0))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

테스트에서는 `main(["--k-w-mk", "401"])`처럼 부르면 된다. CLI 설계의 나머지(인자 이름, 설정 파일과의 관계, 종료 코드)는 6.4장에서 다룬다.

## 순환 import와 import 비용

**순환 import.** 모듈 `a`가 `b`를 import하고 `b`가 다시 `a`를 import하면 무슨 일이 생기는가. import 3단계에서 모듈 객체는 **실행 전에** `sys.modules`에 등록된다. 그래서 `b`가 `a`를 import하는 시점에는 반쯤 실행된(partially initialized) `a`가 캐시에서 나온다. `b`가 `a`의 아직 정의되지 않은 이름을 꺼내려 하면 실패한다.

```python {exec="browser" session="imports"}
(root / "cyc_a.py").write_text("from cyc_b import helper\nVALUE = 1\n")
(root / "cyc_b.py").write_text("from cyc_a import VALUE\ndef helper():\n    return VALUE\n")
importlib.invalidate_caches()
try:
    import cyc_a
except ImportError as e:
    print("ImportError:", str(e).split(" (")[0])
```

```output
ImportError: cannot import name 'VALUE' from partially initialized module 'cyc_a'
```

오류 메시지가 원인을 정확히 말한다. `cyc_a`는 첫 줄에서 `cyc_b`를 import하느라 아직 `VALUE = 1`까지 실행되지 않았다. 순환을 푸는 방법은 다음 순서로 고려한다.

1. **구조를 고친다.** 두 모듈이 서로를 필요로 한다는 것은 대개 둘이 공유하는 부분이 제3의 모듈로 분리되어야 한다는 신호다. 공통 상수·자료형을 `heatlab/types.py` 같은 아래층 모듈로 옮기면 의존이 한 방향이 된다.
2. **모듈째 import한다.** `from cyc_a import VALUE` 대신 `import cyc_a`로 두고 함수 안에서 `cyc_a.VALUE`로 읽으면, 이름을 꺼내는 시점이 import 시점에서 호출 시점으로 미뤄진다.
3. **함수 안에서 import한다.** 최후의 수단이다. 의존 관계가 코드 곳곳에 숨는다.

**import 비용.** import는 모듈 본문을 실행하므로 공짜가 아니다. PyTorch, TensorFlow, 큰 SciPy 하위 패키지처럼 무거운 라이브러리는 import에만 수백 밀리초에서 몇 초가 걸릴 수 있다. `--help`만 출력하는 CLI나 짧은 테스트가 매번 그 비용을 치르면 개발 속도가 떨어진다. 어디서 시간이 드는지는 `python -X importtime -c "import heatlab"`으로 측정한다. 모듈별 import 시간이 마이크로초 단위로 stderr에 출력된다.

무거운 의존성을 일부 기능에서만 쓴다면 그 기능의 함수 안에서 import한다(지연 import). 두 번째 호출부터는 `sys.modules` 캐시 덕분에 비용이 거의 없다. 다만 지연 import는 "설치되지 않은 의존성"의 오류를 그 기능을 처음 쓸 때까지 미루므로, 선택적 의존성(예: 시각화 전용 라이브러리)에만 쓰고 핵심 의존성은 모듈 머리에 둔다.

:::note[분야 예시]
단백질 구조 분석 패키지가 `__init__.py`에서 구조 예측 모델의 가중치를 내려받는 함수를 import 시점에 실행하면, 서열 파일을 읽는 작은 유틸리티 하나를 쓰려는 사용자도 매번 네트워크 접근과 수 초의 대기를 겪는다. 클러스터의 계산 노드처럼 외부 네트워크가 막힌 곳에서는 import 자체가 실패한다. 무거운 자원은 그것이 필요한 함수가 처음 호출될 때 불러온다.
:::

## 실습

"노트북에서는 되는데 스크립트에서는 import가 안 되는" 상황을 재현하고 src layout으로 해결한다. 코드는 `labs/python-and-computing/modules-and-imports/reproduce_layout.py`다. 임시 디렉터리에 같은 내용의 패키지를 flat layout과 src layout으로 만들고, 여러 위치와 방식으로 실행한 결과를 표로 출력한다. src layout 단계는 uv로 임시 가상환경을 만들어 editable 설치를 하므로 uv와 패키지 저장소 접근이 필요하다. 집필 시점의 실행 결과는 다음과 같다.

```output
[flat layout] mypkg/가 저장소 루트에 있다
  루트에서 python -c 'import mypkg' (노트북과 같은 상황)       성공  import 성공
  루트에서 python scripts/train.py                     실패  ModuleNotFoundError: No module named 'mypkg'
  루트에서 python -m scripts.train                     성공  alpha = 1.162e-04 m2/s
  scripts/에서 python train.py                       실패  ModuleNotFoundError: No module named 'mypkg'
[src layout] src/mypkg/ + pyproject.toml
  설치 전, 루트에서 import mypkg                          실패  ModuleNotFoundError: No module named 'mypkg'
  설치 후, 다른 디렉터리에서 python -m mypkg.train            성공  alpha = 1.162e-04 m2/s
  설치 후, 다른 디렉터리에서 mypkg-train                      성공  alpha = 1.162e-04 m2/s
```

결과를 다음 순서로 읽는다.

- **flat layout의 성공과 실패는 모두 `sys.path[0]`으로 설명된다.** 루트에서의 `-c`와 `-m`은 현재 디렉터리를 넣어 성공하고, 파일 경로 실행은 `scripts/`를 넣어 실패한다. 각 줄에 대해 `sys.path[0]`이 무엇인지 먼저 예측하고 결과와 맞춰 본다.
- **src layout은 설치 전에 일부러 실패한다.** "우연히 되는" 경로를 없앤 것이다.
- **설치 후에는 실행 위치와 무관하게 성공한다.** console script `mypkg-train`은 `pyproject.toml`의 `[project.scripts]`에서 만들어졌다.

이어서 자신의 연구 코드 하나를 같은 방식으로 옮긴다. `src/<패키지>/`로 옮기고, `pyproject.toml`을 쓰고, editable 설치한 뒤, 코드 안의 `sys.path` 조작을 모두 지우고도 노트북·스크립트·테스트가 그대로 도는지 확인한다.

마지막으로 상대 import의 규칙을 연습문제로 확인한다. 테스트가 바로 채점한다.

::exercise{id="resolve-relative-import"}

## 실패 사례

- **`sys.path`를 코드 안에서 조작하는 것.** `sys.path.append("..")`나 `sys.path.insert(0, os.path.dirname(__file__) + "/../src")`는 실행 위치나 파일 배치가 바뀌면 깨지고, 같은 모듈이 두 이름으로 import되는 버그를 만들 수 있다. 정적 분석기와 편집기도 그 경로를 알지 못한다. 패키지를 설치하고, 실행은 `python -m`이나 console script로 한다.
- **표준 라이브러리나 유명 패키지와 같은 이름의 파일을 만드는 것.** `random.py`, `statistics.py`, `copy.py`, `numpy.py`, `torch.py`가 스크립트 옆에 있으면 그 디렉터리에서 실행하는 모든 코드가 원래 모듈 대신 그 파일을 가져간다. 이상한 `AttributeError`가 나면 `import m; print(m.__file__)`로 실제로 어떤 파일이 import되었는지부터 확인한다.
- **노트북에서 모듈을 고치고 셀만 다시 실행하는 것.** `sys.modules` 캐시 때문에 옛 코드가 계속 실행된다. 수정한 결과로 그린 그림이 사실은 옛 코드의 결과일 수 있다. 커널을 재시작하고 처음부터 다시 실행한 결과만 믿는다.
- **모듈 최상위에 실험 실행 코드를 두는 것.** 다른 곳에서 import하는 순간 실험이 돌아가고, 병렬 처리에서는 자식 프로세스가 같은 코드를 다시 실행한다. 실행 코드는 `main()`에 넣고 `if __name__ == "__main__":`으로 보호한다.

## 통과 기준

- `src/` 아래에 패키지를 둔 연구 프로젝트를 만들고 editable 설치한 뒤, 저장소 루트·다른 디렉터리·노트북에서 같은 import 문으로 import되고, console script나 `python -m`으로 실행된다. 코드에 `sys.path` 조작이 없다.
- `reproduce_layout.py`의 flat layout 네 줄 각각에 대해 `sys.path[0]`이 무엇인지와 성공·실패를 실행 전에 예측하고, 결과와 일치한다.
- 연습문제 `resolve-relative-import`의 테스트를 통과한다.

## 참고 문헌

- Python Software Foundation. *The Python Language Reference* — The import system(모듈 캐시 `sys.modules`, finder와 loader, 패키지, 상대 import). Python 공식 문서.
- Python Software Foundation. *The Python Standard Library* — `sys.path` 초기화, `runpy`, `importlib`. Python 공식 문서.
- Python Packaging Authority. *Python Packaging User Guide* — src layout과 flat layout의 비교, `pyproject.toml` 작성법. packaging.python.org.
