---
id: environments/python-environments
title: "Python 환경 관리: venv·pip·uv·conda"
title_en: "Python Environments: venv, pip, uv, conda"
tier: core
status: draft
prerequisites: [environments/software-stack]
---

# Python 환경 관리: venv·pip·uv·conda

한 서버에서 두 프로젝트를 진행한다고 하자. 하나는 3년 전 논문의 코드를 재현해야 해서 오래된 NumPy가 필요하고, 다른 하나는 최신 PyTorch가 필요하다. 두 프로젝트가 같은 Python 환경을 쓰면, 한쪽을 위해 패키지를 바꾸는 순간 다른 쪽이 깨진다. 더 나쁜 경우는 깨진 줄도 모르고 다른 버전으로 실험을 계속하는 것이다.

이 장은 **프로젝트마다 격리된 Python 환경**을 만들고 다루는 법을 익힌다. venv·pip·uv·conda가 각각 어느 범위를 관리하는지 비교하고, "지금 실행되는 python은 어느 환경의 것인가"를 확인하는 습관을 들이고, 연구실이 따를 표준 도구를 정한다. 명령과 버전은 모두 집필 시점의 예시이며, 각 도구의 공식 문서로 확인한다.

## 학습 목표

- 각 도구의 역할과 범위를 비교한다
- 프로젝트별 격리 환경을 만든다
- 연구실 표준 도구를 정한다

## 왜 격리 환경인가

Linux 서버에서 Python 패키지가 놓일 수 있는 곳은 크게 세 군데다.

| 위치 | 예 | 누가 관리하나 |
|---|---|---|
| 시스템 Python | `/usr/bin/python3`와 `/usr/lib/python3/dist-packages` | 배포판 패키지 관리자(apt). OS 도구가 이 Python에 기댄다 |
| 사용자 site-packages | `~/.local/lib/python3.X/site-packages` | `pip install --user`. 그 사용자의 모든 Python 실행에 섞여 들어간다 |
| 프로젝트 환경 | 프로젝트의 `.venv/` 또는 conda 환경 | 프로젝트별로 만들고 지운다 |

`import numpy`를 실행하면 Python은 `sys.path`의 디렉터리를 **앞에서부터** 보고 처음 찾은 `numpy`를 불러온다. 사용자 site-packages는 시스템 site-packages보다 앞에 놓이므로, `pip install --user`로 설치한 패키지는 시스템 패키지를 가린다. 가상의 예로 확인한다.

```python {exec="browser" session="pyenv"}
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
```

```output
import numpy -> 1.26.4  /home/me/.local/lib/python3.12/site-packages
import scipy -> 1.13.0  /usr/lib/python3/dist-packages
import torch -> ModuleNotFoundError
```

NumPy는 사용자가 설치한 1.26.4가, SciPy는 apt가 설치한 1.13.0이 쓰인다. 서로 다른 관리자가 따로 설치한, 함께 시험된 적 없는 조합이다. 어느 버전이 쓰이는지는 디렉터리 순서가 정하고, 그 순서는 사용자의 셸 설정이나 실행 방법에 따라 달라진다. 실험 결과가 "어느 날부터" 달라지는 전형적인 경로다.

**격리 환경**(isolated environment)은 이 문제를 구조로 막는다. 프로젝트마다 Python 인터프리터와 **독립된 site-packages 디렉터리**를 묶은 환경을 하나씩 두고, 그 프로젝트의 패키지는 그 안에만 설치한다. 격리 환경의 핵심 성질은 세 가지다.

- **독립성.** 한 환경의 설치·업그레이드가 다른 환경에 영향을 주지 않는다.
- **명시성.** 무엇이 설치되었는지 환경 안에서 완전히 알 수 있다. 사용자 site-packages나 시스템 패키지가 섞여 들지 않는다.
- **폐기 가능성.** 환경이 망가지면 지우고 다시 만든다. 다시 만드는 데 필요한 정보(의존성 목록과 lockfile, 3.4장)는 환경 밖, 프로젝트 저장소에 둔다.

최근 배포판은 시스템 Python을 보호하는 장치도 둔다. Ubuntu 24.04의 시스템 Python에서 `pip install`을 실행하면 `error: externally-managed-environment`로 거부된다. 시스템 Python을 배포판 패키지 관리자의 소유로 표시하는 표준(PEP 668)에 따른 동작이다. 이 오류는 우회할 대상이 아니라 "가상환경을 만들라"는 안내로 읽는다.

:::note[분야 예시]
생명정보학 분석 파이프라인은 여러 도구를 이어 붙인다. 정렬 도구, 변이 검출 도구, 통계 분석 스크립트가 서로 다른 버전의 같은 라이브러리를 요구하는 일이 흔하다. 그래서 파이프라인 관리 도구들은 단계마다 별도 환경(conda 환경이나 컨테이너)을 지정하는 기능을 둔다. 격리의 단위를 "프로젝트"에서 "파이프라인의 한 단계"로 더 잘게 나눈 것이다.
:::

## 도구별 역할

Python 환경 도구는 관리하는 범위가 서로 다르다. 같은 질문 네 개로 비교한다. Python 인터프리터 자체를 설치하는가, 격리 환경을 만드는가, 의존성을 해석하고 lockfile로 고정하는가, Python 밖의 바이너리(컴파일러, C 라이브러리, CUDA runtime 등)를 패키지로 관리하는가.

| 도구 | 인터프리터 설치 | 격리 환경 | 의존성 해석·lockfile | Python 밖의 바이너리 |
|---|---|---|---|---|
| venv + pip | 아니오(있는 것을 쓴다) | venv | pip이 해석. lockfile은 없고 `requirements.txt`로 대신한다 | wheel에 든 것만 |
| uv | 예(`uv python install`) | 예(`.venv`) | 예(`uv.lock`) | wheel에 든 것만 |
| conda / mamba | 예(환경마다 Python 포함) | 예(conda 환경) | 예. lockfile은 별도 도구(conda-lock 등) | 예(채널의 패키지로) |
| pyenv | 예(소스에서 빌드) | 아니오(플러그인으로 보완) | 아니오 | 아니오 |

**venv + pip.** 표준 라이브러리의 `venv` 모듈이 가상환경을 만들고, pip이 그 안에 패키지를 설치한다. 별도 설치 없이 어디서나 쓸 수 있는 최소 구성이다. Ubuntu에서는 `python3-venv` 패키지가 있어야 `python3 -m venv`가 동작한다. 가상환경이 실제로 무엇인지 보면 이 도구의 범위가 분명해진다. 로컬 Python에서 임시 디렉터리에 만들어 본다.

```python {exec="python"}
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
```

```output
pyvenv.cfg | home = /usr/local/bin
pyvenv.cfg | include-system-site-packages = false
pyvenv.cfg | version = 3.13.18
bin/       | ['Activate.ps1', 'activate', 'activate.csh', 'activate.fish', 'python', 'python3', 'python3.13']
python은   | 심볼릭 링크 -> /usr/local/bin
```

가상환경은 **디렉터리 하나**다. 그 안의 `python`은 기존 인터프리터(`home`)를 가리키는 링크이고, `pyvenv.cfg`가 "이 디렉터리를 prefix로 삼고, 시스템 site-packages는 보지 않는다"(`include-system-site-packages = false`)고 알려 준다. 패키지는 이 디렉터리의 `lib/python3.X/site-packages`에 설치된다. 인터프리터는 새로 설치하지 않으므로, Python 3.12 환경이 필요하면 3.12 인터프리터가 먼저 있어야 한다. 그리고 활성화 스크립트와 설치된 명령의 첫 줄에 절대 경로가 박혀 있어, 디렉터리를 다른 경로로 옮기면 동작하지 않는다.

**uv.** 인터프리터 설치, 가상환경 생성, 의존성 해석, lockfile, 명령 실행을 하나의 도구로 묶는다. Rust로 작성되어 빠르고, `uv pip install`처럼 pip과 비슷한 명령도 제공한다. 프로젝트의 `pyproject.toml`에 의존성을 선언하면 uv가 해를 풀어 `uv.lock`에 정확한 버전을 기록하고, `.venv`를 그 기록과 일치시킨다(3.4장). PyPI의 wheel로 해결되는 의존성이라면, PyTorch를 포함해 대부분의 ML 프로젝트를 이 도구 하나로 다룰 수 있다.

**conda / mamba.** conda는 Python 패키지 관리자가 아니라 **언어와 무관한 패키지 관리자**다. 환경마다 자기 Python 인터프리터를 가지며, 채널(conda-forge 등)에서 C/C++/Fortran 라이브러리, 컴파일러, MPI, CUDA runtime까지 패키지로 설치한다. 그래서 PyPI wheel로 구하기 어려운 바이너리 의존성이 있을 때 강하다. mamba는 같은 패키지 형식을 쓰는 더 빠른 구현이다. 배포판을 고를 때는 기본 채널의 이용 약관을 확인한다. conda-forge를 기본 채널로 쓰는 Miniforge 배포판이 연구실에서 흔히 쓰인다.

**pyenv 등 Python 버전 관리자.** 여러 버전의 인터프리터를 설치하고 디렉터리별로 고르게 해 준다(`.python-version` 파일). 패키지와 환경은 다른 도구(venv 등)가 맡는다. uv와 conda가 인터프리터 관리까지 하므로, 이들을 쓰면 따로 필요하지 않은 경우가 많다.

::quiz{id="interpreter-managers"}

## 일상 작업

도구별로 환경을 만들고, 쓰고, 지우는 명령은 다음과 같다(집필 시점 예시).

```bash
# venv + pip
python3 -m venv .venv              # 만들기 (프로젝트 루트에 .venv)
source .venv/bin/activate          # 활성화
python -m pip install numpy        # 설치: 'python -m pip'으로 실행할 python을 명시
deactivate                         # 비활성화
rm -rf .venv                       # 삭제

# uv
uv init                            # pyproject.toml 만들기
uv add numpy                       # 의존성 추가: pyproject.toml, uv.lock, .venv를 함께 갱신
uv run python train.py             # .venv의 python으로 실행 (활성화 불필요)
rm -rf .venv && uv sync            # 지우고 lockfile대로 다시 만들기

# conda (conda-forge 채널)
conda create -n proj -c conda-forge python=3.12 numpy
conda activate proj
conda deactivate
conda env remove -n proj
```

**활성화가 하는 일.** `source .venv/bin/activate`는 셸의 `PATH` 맨 앞에 `.venv/bin`을 붙이고 `VIRTUAL_ENV` 변수를 설정할 뿐이다. 그래서 활성화하지 않고 `.venv/bin/python train.py`처럼 환경의 python을 직접 실행해도 같은 환경에서 실행된다. 배치 작업 스크립트에서는 이 방식이 더 확실하다. 활성화 상태는 셸마다 따로이므로, 새 터미널이나 `tmux` 창, 작업 스케줄러가 띄운 셸은 활성화되지 않은 상태로 시작한다.

**어떤 python이 실행되는가.** 환경 문제를 진단할 때 가장 먼저 확인할 것이다.

```bash
which python                       # PATH에서 처음 찾은 python
python -c "import sys; print(sys.executable)"
pip --version                      # 이 pip이 속한 python의 경로도 함께 보여 준다
```

`which python`과 `pip --version`이 서로 다른 환경을 가리키면, pip이 설치한 패키지를 python이 보지 못한다. `python -m pip`처럼 실행할 python을 명시하면 두 명령이 어긋날 수 없다. 코드 안에서는 `sys.prefix`(현재 환경의 루트)와 `sys.base_prefix`(바탕 인터프리터의 루트)를 비교해 가상환경 안인지 판단한다. 이 책의 실습 환경(`uv run --project labs`)에서 실행한 결과다.

```python {exec="python"}
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
```

```output
가상환경 안인가   : True
환경 디렉터리 이름 : .venv
base 인터프리터    : /usr/local
user site 사용     : False
  sys.path: <base>/lib/python313.zip
  sys.path: <base>/lib/python3.13
  sys.path: <base>/lib/python3.13/lib-dynload
  sys.path: <env>/lib/python3.13/site-packages
```

표준 라이브러리는 바탕 인터프리터(`<base>`)에서, 서드파티 패키지는 환경(`<env>`)의 site-packages에서만 온다. 사용자 site-packages는 꺼져 있다(`user site 사용: False`). 앞 절의 가상 예처럼 `~/.local`의 패키지가 끼어들 길이 없다는 뜻이다. 이 네 줄이 격리 환경이 제대로 동작한다는 증거다.

::quiz{id="pip-python-mismatch"}

## 도구 혼용의 문제

도구 하나로 끝나지 않는 경우가 있다. 혼용 자체가 금지는 아니지만, **두 관리자가 같은 디렉터리를 서로의 기록을 모른 채 관리**한다는 점을 알고 규칙을 지켜야 한다.

**conda 환경 안의 pip.** conda로 설치할 수 없는 패키지를 pip으로 넣는 일은 흔하다. 문제는 conda가 pip이 설치한 패키지의 의존성을 고려하지 않고 해를 푼다는 것이다. 나중에 `conda install`이나 `conda update`가 pip 패키지가 기대하던 라이브러리 버전을 바꿔도 conda는 알지 못한다. 그래서 흔히 권하는 순서는 다음과 같다.

1. conda로 설치할 수 있는 것을 먼저 모두 설치한다.
2. pip은 마지막에 쓴다.
3. 그 뒤에 conda 패키지가 더 필요하면 환경을 처음부터 다시 만든다.

::quiz{id="conda-pip-mixing"}

**사용자 site-packages의 누수.** `pip install --user`로 `~/.local`에 설치한 패키지는 그 사용자의 여러 Python 실행에 섞여 들어갈 수 있다. venv는 기본적으로 사용자 site-packages를 끄지만(위 출력), 모든 환경 도구가 그렇지는 않다. 격리되어 있다고 믿은 환경에서 `~/.local`의 오래된 패키지가 import되는 사고가 여기서 나온다. 의심되면 `python -m site`로 사용자 site가 켜져 있는지와 그 경로를 확인하고, 환경변수 `PYTHONNOUSERSITE=1`로 끌 수 있다. 근본 해결책은 `pip install --user`를 쓰지 않는 것이다.

**시스템 pip.** 가상환경을 활성화하지 않은 채 `pip install`을 실행하면 시스템 Python이나 사용자 site-packages로 들어간다. 앞 절의 `externally-managed-environment` 오류가 이를 막아 주는 배포판도 있지만, 직접 설치한 Python에는 이런 보호가 없다.

**셸 설정이 고른 python.** conda를 설치하면 셸 시작 파일(`~/.bashrc`)에 초기화 코드가 들어가고, 새 셸마다 base 환경이 자동으로 활성화될 수 있다. 그러면 `PATH`의 맨 앞에 conda의 python이 놓여, uv나 venv로 만든 환경을 쓰려던 명령이 엉뚱한 python으로 실행된다. 자동 활성화를 끌지 정하고, 진단할 때는 늘 `which python`부터 확인한다.

## 연구실 표준 정하기

도구마다 장단점이 있지만, 연구실 안에서 도구가 사람 수만큼 다르면 서로의 환경을 재생성할 수 없다. 표준은 **기본 도구 하나와 예외 규칙**으로 정한다. 다음은 한 가지 예다.

- **기본 도구: uv.** 프로젝트 루트에 `pyproject.toml`과 `uv.lock`을 커밋하고, 환경은 프로젝트 루트의 `.venv`에 둔다. 실행은 `uv run`으로 한다. PyPI wheel로 해결되는 의존성(PyTorch 포함)은 모두 이 경로로 관리한다.
- **예외 1: conda(conda-forge).** PyPI wheel로 구하기 어려운 바이너리 의존성(특정 C/Fortran 과학 라이브러리, MPI를 포함한 빌드 등)이 핵심일 때 쓴다. `environment.yml`과 lockfile을 커밋하고, pip 혼용 규칙을 지킨다.
- **예외 2: 컨테이너.** 시스템 패키지까지 고정해야 하거나 클러스터에서 실행할 때 3.5장의 방법을 쓴다.
- **금지 사항.** 시스템 Python과 conda base 환경에 프로젝트 패키지를 설치하지 않는다. `pip install --user`를 쓰지 않는다.
- **공유 서버 운영.** 패키지 캐시가 홈 디렉터리 용량 제한을 넘지 않도록 큰 디스크로 옮긴다(uv는 `UV_CACHE_DIR`, conda는 패키지 디렉터리 설정).

어떤 도구를 기본으로 할지보다 중요한 것은 **새 구성원이 문서 한 장을 보고 다른 사람의 환경을 그대로 다시 만들 수 있는가**다. 표준을 정했다면 실제로 그 시험을 해 본다.

:::note[분야 예시]
지구과학·기후 연구에서 많이 쓰는 지리공간 라이브러리와 기상 데이터 형식(netCDF, GRIB 등)의 라이브러리는 C 라이브러리에 크게 기대는 경우가 많다. 이런 스택은 conda-forge에서 한 번에 설치하는 쪽이 수월한 경우가 많아, 이 분야 연구실은 conda를 기본 도구로 정하기도 한다. 표준은 도구의 유행이 아니라 연구실이 실제로 쓰는 의존성을 보고 정한다.
:::

## 실습

같은 프로젝트(NumPy + pandas)를 venv+pip, uv, conda로 각각 구성하고 설치 시간·디스크 사용량·재현성을 비교한다. 코드는 `labs/environments/python-environments/compare_tools.py`다. 도구마다 다음을 한다.

1. 임시 디렉터리에 환경을 만들고 두 패키지를 설치한다. 캐시는 임시 디렉터리로 돌려 처음 설치에 가깝게 잰다.
2. 기록 파일을 만든다(venv+pip은 `pip freeze` 결과의 `requirements.txt`, uv는 `pyproject.toml`과 `uv.lock`).
3. 기록 파일만 새 디렉터리로 복사해 환경을 다시 만들고, 설치된 패키지 목록이 같은지 비교한다.

집필 환경에서 실행한 결과는 다음과 같다(conda는 설치되어 있지 않아 측정하지 못했다).

```output
Python 3.13.18, 패키지 ['numpy', 'pandas']

도구             설치 시간       디스크    패키지 수  재현
venv+pip       12.9s     157MB        4  같음
uv              2.5s     101MB        4  같음
conda      설치되어 있지 않아 건너뜀
```

결과를 읽을 때 주의할 점이 있다.

- **설치 시간은 네트워크에 크게 좌우된다.** 같은 측정을 다른 시간에 하면 숫자가 달라진다. 여러 번 재고, 캐시가 있을 때와 없을 때를 나누어 본다.
- **디스크 차이는 내용물 차이다.** 집필 환경에서 확인해 보니 venv+pip 환경에는 pip 자체가 들어 있고, pip이 설치하면서 만든 `.pyc` 바이트코드 파일이 2,296개 있었다. uv 환경에는 `.pyc`가 하나도 없었다. uv는 기본 설정에서 바이트코드를 미리 만들지 않기 때문이다(첫 import 때 만들어진다). 같은 기준으로 비교하려면 이런 차이를 함께 적는다.
- **"재현: 같음"의 범위.** 같은 날, 같은 머신, 같은 Python에서 다시 만든 결과다. `requirements.txt`는 pip freeze로 정확한 버전을 적었기 때문에 같았을 뿐, 플랫폼이 다른 머신이나 Python 버전이 다른 환경에서는 보장되지 않는다. 이 차이는 3.4장의 lockfile에서 다룬다.

자신의 서버에서 conda(또는 mamba)까지 포함해 실행하고, 세 도구를 표로 비교한 뒤 연구실 표준 문서 초안을 쓴다. 그다음 다른 사람이 그 문서만 보고 환경을 재생성하게 한다.

이어서 연습문제를 푼다. `sys.path` 순서와 설치 위치에서 가려진(shadowed) 패키지를 찾아내는 함수를 작성한다.

::exercise{id="shadowed-packages"}

## 실패 사례

- **활성화하지 않은 환경에 패키지를 설치하는 것.** 새 터미널을 열고 활성화를 잊은 채 `pip install`을 실행하면 시스템 Python이나 사용자 site-packages에 설치된다. 프로젝트 환경에서는 여전히 `ModuleNotFoundError`가 나고, 엉뚱한 곳에 설치된 패키지는 다른 프로젝트에 섞여 든다. `python -m pip`과 `uv add`·`uv run`처럼 환경을 명시하는 명령을 쓴다.
- **base 환경에 모든 것을 설치하는 것.** 처음에는 편하지만, 프로젝트가 늘수록 버전 충돌이 쌓이고, 어느 패키지가 어느 프로젝트에 필요했는지 알 수 없게 된다. conda 자체가 기대는 환경이 망가지면 복구도 어렵다.
- **가상환경 디렉터리를 복사하거나 옮겨서 쓰는 것.** 가상환경 안의 스크립트에는 만들 때의 절대 경로가 박혀 있다. 다른 경로나 다른 머신으로 옮긴 `.venv`는 일부만 동작하다가 이상한 곳에서 실패한다. 옮기는 것은 환경이 아니라 기록 파일이고, 환경은 그 자리에서 다시 만든다.
- **`pip install --user`로 급한 불을 끄는 것.** 그 순간에는 문제가 풀리지만, 그 패키지가 이후 다른 환경에 섞여 드는 원인이 된다.

## 통과 기준

- 연구실 표준 도구로 새 프로젝트 환경을 만들고, 기록 파일을 커밋한 뒤, 다른 사람이 그 기록만으로 같은 패키지 목록의 환경을 재생성한다.
- `which python`, `sys.executable`, `sys.prefix`, `pip --version`으로 실행 중인 python과 설치 대상 환경을 확인하고 설명한다.
- 연습문제 `shadowed-packages`의 테스트를 통과한다.
