---
id: environments/dependencies-and-packaging
title: "의존성과 패키징"
title_en: "Dependencies and Packaging"
tier: core
status: draft
prerequisites: [environments/python-environments]
---

# 의존성과 패키징

`pip install pandas` 한 줄은 pandas 하나가 아니라 여러 패키지를 설치한다. pandas가 NumPy를 요구하고, 날짜 처리 라이브러리를 요구하고, 그 라이브러리가 또 다른 것을 요구하기 때문이다. 오늘 설치한 환경과 6개월 뒤에 같은 명령으로 설치한 환경은, 그 사이 어느 패키지든 새 버전을 냈다면 서로 다르다. 논문의 결과를 재현하려는 사람에게 "`pip install pandas`를 실행하세요"는 충분한 설명이 아니다.

이 장은 의존성이 어떻게 얽히는지, 무엇을 선언하고 무엇을 고정해야 하는지, 충돌이 났을 때 resolver의 메시지를 어떻게 읽는지를 다룬다. 목표는 `pyproject.toml`과 lockfile 두 파일만으로 새 머신에서 같은 환경을 다시 만드는 것이다. 도구는 이 책의 실습 환경과 같은 uv를 기준으로 쓰고, pip·conda의 대응 방법을 함께 적는다. 명령과 버전은 집필 시점의 예시다.

## 학습 목표

- 직접·전이 의존성과 버전 제약을 이해한다
- pyproject와 lockfile로 환경을 고정한다
- 의존성 충돌을 해결한다

## 의존성의 구조

**직접 의존성**(direct dependency)은 프로젝트가 직접 import해서 쓰고 그래서 직접 선언하는 패키지다. **전이 의존성**(transitive dependency)은 직접 의존성이 다시 요구하는 패키지다. 둘을 이으면 **의존성 그래프**가 된다. 같은 패키지가 여러 경로로 요구될 수 있고, 그때는 모든 경로의 **버전 제약**을 동시에 만족해야 한다. 가상의 분석 프로젝트로 그래프를 따라가 본다.

```python {exec="browser" session="deps"}
meta = {   # 가상의 메타데이터: 패키지 -> {의존 패키지: 버전 제약}
    "my-analysis":     {"pandas": ">=2.2", "scipy": ">=1.11", "matplotlib": ">=3.8"},
    "pandas":          {"numpy": ">=1.26", "python-dateutil": ">=2.8"},
    "scipy":           {"numpy": ">=1.23,<2.6"},
    "matplotlib":      {"numpy": ">=1.23", "pillow": ">=8"},
    "python-dateutil": {"six": ">=1.5"},
}

direct = set(meta["my-analysis"])
seen, stack, on_numpy = set(), list(direct), []
while stack:                                   # 의존성 그래프를 따라 내려간다
    pkg = stack.pop()
    if pkg in seen:
        continue
    seen.add(pkg)
    for dep, spec in meta.get(pkg, {}).items():
        stack.append(dep)
        if dep == "numpy":
            on_numpy.append(f"{pkg}: numpy{spec}")

print("직접 의존성:", sorted(direct))
print("전이 의존성:", sorted(seen - direct))
print("numpy에 걸린 제약:", sorted(on_numpy))
```

```output
직접 의존성: ['matplotlib', 'pandas', 'scipy']
전이 의존성: ['numpy', 'pillow', 'python-dateutil', 'six']
numpy에 걸린 제약: ['matplotlib: numpy>=1.23', 'pandas: numpy>=1.26', 'scipy: numpy>=1.23,<2.6']
```

프로젝트는 NumPy를 선언하지 않았지만 NumPy에는 세 개의 제약이 걸린다. 설치될 NumPy는 이 셋의 교집합 `>=1.26, <2.6` 안에 있어야 한다. 직접 선언한 세 패키지보다 전이 의존성이 더 많다는 점도 눈여겨본다. 실제 ML 프로젝트에서는 직접 의존성 10여 개가 전이 의존성 100개 이상을 끌어오는 일이 흔하다.

::quiz{id="transitive-count"}

**버전 제약.** Python 패키지의 버전 제약은 표준화된 문법을 따른다.

| 제약 | 의미 | 예 |
|---|---|---|
| `==` | 정확히 이 버전 | `numpy==2.5.4` |
| `>=`, `<` 등 | 범위 | `numpy>=1.23,<2.6` |
| `!=` | 이 버전만 제외(알려진 버그가 있는 버전 등) | `foo!=1.2.0` |
| `==X.*` | 앞자리가 일치 | `numpy==1.26.*` |
| `~=` | 호환 릴리스(compatible release) | `~=1.4.2`는 `>=1.4.2, ==1.4.*` |

버전 번호의 의미는 패키지마다의 약속에 달려 있다. **semantic versioning**(MAJOR.MINOR.PATCH)을 따르는 패키지는 하위 호환이 깨지는 변경에 MAJOR를, 기능 추가에 MINOR를, 버그 수정에 PATCH를 올린다. 이 약속이 지켜진다면 `>=1.4, <2`나 `~=1.4`가 "호환되는 범위"가 된다. 그러나 과학 Python 생태계의 많은 패키지는 이 약속을 엄격히 따르지 않는다. 자체적인 deprecation 정책에 따라 MINOR 버전에서도 동작을 바꾸거나 기능을 제거하고, `tzdata 2026.6`처럼 날짜를 버전으로 쓰는 패키지(calendar versioning)도 있다. 그래서 상한을 어디에 둘지는 버전 번호만 보고 정할 수 없고, 해당 패키지의 릴리스 정책과 변경 기록을 보고 정한다.

## pyproject.toml

`pyproject.toml`은 Python 프로젝트의 메타데이터와 의존성을 선언하는 표준 파일이다. 주요 표(table)는 다음과 같다.

```toml
[project]                                  # 프로젝트 메타데이터 (표준)
name = "my-analysis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [                           # 직접 의존성과 허용 범위
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.11",
]

[project.optional-dependencies]            # 사용자가 골라 설치하는 기능 (extras)
plot = ["matplotlib>=3.8"]                 # pip install "my-analysis[plot]"

[dependency-groups]                        # 배포되지 않는 개발용 그룹 (표준)
dev = ["pytest>=8", "ruff"]

[build-system]                             # 이 프로젝트를 패키지로 빌드하는 도구
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- **`dependencies`**: 프로젝트가 동작하는 데 필요한 직접 의존성이다. 정확한 버전이 아니라 **동작한다고 확인한 범위**를 쓴다. 하한은 실제로 쓰는 기능이 들어온 버전, 상한은 알려진 비호환이 있을 때만 건다.
- **`optional-dependencies`**(extras): 패키지를 설치하는 사람이 고르는 선택 기능이다. 배포된 패키지 메타데이터에 포함된다.
- **`dependency-groups`**: 테스트·린트·문서 빌드처럼 개발할 때만 필요한 의존성이다. 배포 메타데이터에 포함되지 않는다. uv에서는 `uv add --dev pytest`가 `dev` 그룹에 추가하고, `uv sync`가 기본으로 함께 설치한다.
- **`build-system`**: 이 프로젝트를 wheel로 빌드할 도구(빌드 백엔드)를 지정한다. 라이브러리로 배포하지 않는 분석 프로젝트에는 없어도 된다.

이 책의 실습 환경 `labs/pyproject.toml`이 실제 예다. `dependencies`에 NumPy와 matplotlib을, `[dependency-groups]`의 `dev`에 pytest를 두고, 패키지로 빌드하지 않는 프로젝트이므로 `[tool.uv]`에 `package = false`를 적었다. `[tool.<도구 이름>]` 표는 각 도구가 자기 설정을 두는 곳이다.

## lockfile과 재현

`dependencies`의 범위 선언만으로는 재현이 안 된다. `numpy>=1.26`은 오늘은 2.5.4로, 내년에는 다른 버전으로 풀린다. 그래서 resolver가 푼 **결과**를 따로 기록한다. 이것이 **lockfile**이다. lockfile에는 전이 의존성까지 모든 패키지의 정확한 버전과, 내려받을 파일의 해시가 들어간다.

| 파일 | 만드는 방법 | 특징 |
|---|---|---|
| `requirements.txt` | `pip freeze`, 또는 `uv pip compile requirements.in` | 한 환경(플랫폼·Python 버전)에 대한 결과. 해시는 선택 |
| `uv.lock` | `uv lock` (또는 `uv add`·`uv sync`가 자동으로) | 여러 플랫폼·Python 버전을 함께 푼 **universal** lockfile. 해시 포함 |
| conda-lock 파일 | `conda-lock` 도구 | conda 환경을 플랫폼별로 고정 |

`pip freeze`는 "지금 이 환경에 설치된 것"을 적을 뿐, 그것이 어떤 선언에서 나왔는지 모른다. 반면 `uv.lock`은 `pyproject.toml`의 선언에서 나온 결과임을 알고, 선언이 바뀌면 다시 풀어야 한다는 것도 안다. 표준 lockfile 형식(`pylock.toml`)도 정해져 일부 도구가 지원하기 시작했다(집필 시점).

uv에서 lock과 환경 동기화는 두 단계다(집필 시점 예시).

```bash
uv lock                 # pyproject.toml을 풀어 uv.lock을 만들거나 갱신한다
uv sync                 # .venv를 uv.lock과 정확히 일치시킨다 (없는 것은 설치, 남는 것은 제거)
uv sync --locked        # uv.lock이 pyproject.toml과 맞지 않으면 갱신하지 않고 실패한다 (CI·새 머신용)
uv lock --upgrade-package numpy   # 의도적으로 numpy만 새 버전으로 다시 푼다
uv export --format requirements-txt > requirements.txt   # pip 사용자를 위한 내보내기
```

`pyproject.toml`과 `uv.lock`을 **둘 다 커밋**한다. 선언은 의도를, lockfile은 그 의도를 특정 시점에 푼 결과를 기록한다. 새 머신에서는 `uv sync --locked` 한 줄로 기록된 환경을 다시 만든다. `--locked`가 중요하다. 이 옵션 없이 `uv sync`를 실행하면 선언과 lockfile이 어긋났을 때 lockfile을 자동으로 다시 풀어, "기록된 그대로"가 아니게 될 수 있다.

lockfile이 보장하는 범위도 알아 둔다. lockfile은 **어떤 버전의 어떤 파일을 설치할지**를 고정한다. 그 파일을 실행할 수 있는지(드라이버, glibc, CPU 아키텍처)와, 그 플랫폼용 미리 빌드된 파일이 있는지는 다음 절의 문제다.

:::note[분야 예시]
계산화학이나 재료 시뮬레이션 논문의 부록에 "NumPy 1.x, SciPy 1.x에서 실행했다"라고만 적혀 있으면, 몇 년 뒤 재현하려는 사람은 수십 개 버전 조합을 시험해야 한다. 수치 결과가 라이브러리 버전에 따라 미세하게 달라질 수 있는 분야일수록, 코드 저장소에 lockfile을 함께 공개하는 것이 결과를 검증 가능하게 만드는 가장 값싼 방법이다.
:::

## 배포 형식

Python 패키지는 두 가지 형식으로 배포된다.

- **sdist**(source distribution, `.tar.gz`): 소스 코드 묶음이다. 설치하려면 빌드해야 하고, C/C++/Fortran/CUDA 확장이 있으면 컴파일러가 필요하다.
- **wheel**(`.whl`): 미리 빌드된 설치 파일이다. 내려받아 풀기만 하면 된다. 컴파일된 확장이 든 wheel은 특정 Python 버전·ABI·플랫폼용이다.

wheel의 파일 이름은 `이름-버전-python태그-abi태그-platform태그.whl` 형식이고, 패키지 관리자는 이 태그로 현재 환경에 맞는 파일을 고른다. 순수 Python 패키지는 `py3-none-any`(모든 Python 3, 모든 플랫폼)다. Linux의 platform tag `manylinux_2_28_x86_64`는 "glibc 2.28 이상인 x86_64 Linux"를 뜻하고, `musllinux`는 glibc 대신 musl을 쓰는 배포판(Alpine 등)용이다. 집필 시점 PyPI에 올라 있는 실제 파일 이름으로, CPython 3.13을 쓰는 x86_64 Linux의 두 glibc 버전에서 무엇이 설치되는지 판정한다.

```python {exec="browser" session="deps"}
import re

files = ["numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
         "numpy-2.5.4-cp313-cp313-musllinux_1_2_x86_64.whl",
         "numpy-2.5.4-cp313-cp313-win_amd64.whl",
         "torch-2.15.0-cp313-cp313-manylinux_2_28_x86_64.whl",
         "numpy-2.5.4.tar.gz"]

def check(fn, py, glibc, arch="x86_64"):
    if not fn.endswith(".whl"):
        return "sdist: 소스에서 빌드"
    name, version, pytag, abi, plats = fn[:-4].split("-")     # build tag가 없는 이름
    for plat in plats.split("."):
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", plat)
        if pytag == py and m and (int(m[1]), int(m[2])) <= glibc and m[3] == arch:
            return f"설치 가능 ({plat})"
    return "해당 없음"

for glibc in [(2, 39), (2, 17)]:
    print(f"== Linux x86_64, CPython 3.13, glibc {glibc[0]}.{glibc[1]}")
    for fn in files:
        print(f"  {fn:72s} {check(fn, 'cp313', glibc)}")
```

```output
== Linux x86_64, CPython 3.13, glibc 2.39
  numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  설치 가능 (manylinux_2_27_x86_64)
  numpy-2.5.4-cp313-cp313-musllinux_1_2_x86_64.whl                         해당 없음
  numpy-2.5.4-cp313-cp313-win_amd64.whl                                    해당 없음
  torch-2.15.0-cp313-cp313-manylinux_2_28_x86_64.whl                       설치 가능 (manylinux_2_28_x86_64)
  numpy-2.5.4.tar.gz                                                       sdist: 소스에서 빌드
== Linux x86_64, CPython 3.13, glibc 2.17
  numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  해당 없음
  numpy-2.5.4-cp313-cp313-musllinux_1_2_x86_64.whl                         해당 없음
  numpy-2.5.4-cp313-cp313-win_amd64.whl                                    해당 없음
  torch-2.15.0-cp313-cp313-manylinux_2_28_x86_64.whl                       해당 없음
  numpy-2.5.4.tar.gz                                                       sdist: 소스에서 빌드
```

glibc 2.39 시스템(Ubuntu 24.04)에서는 wheel이 바로 설치된다. 오래된 glibc 2.17 시스템(오래된 클러스터에서 볼 수 있다)에서는 맞는 wheel이 없어, NumPy는 소스 빌드로 넘어가고 PyTorch는 설치할 수 없다. 3.1장에서 본 glibc 하위 호환 규칙이 파일 이름에 그대로 적혀 있는 것이다.

**소스 빌드가 일어나는 경우**는 다음과 같다.

- 새 Python 버전에 오래된 패키지 버전을 설치할 때(그 버전이 나올 때 새 Python용 wheel이 없었다)
- 흔하지 않은 플랫폼(오래된 glibc, 일부 aarch64 환경 등)
- 패키지가 애초에 sdist만 배포할 때, 또는 설치 옵션으로 wheel을 금지했을 때

집필 환경(Python 3.13)에서 `numpy<2`와 `pandas<3`을 함께 요구했더니 정확히 첫 번째 경우가 되었다. NumPy 1.x의 마지막 버전인 1.26.4는 Python 3.12까지의 wheel만 PyPI에 올라 있어, uv가 sdist를 내려받아 C 코드를 컴파일했다. 출력의 일부는 다음과 같다.

```output
   Building numpy==1.26.4
      Built numpy==1.26.4
Prepared 3 packages in 4m 46s
```

몇 초면 끝날 설치가 5분 가까이 걸렸고, 컴파일러가 없는 머신이었다면 실패했을 것이다. 설치가 갑자기 느려지거나 컴파일 오류가 나면, 먼저 "어느 패키지가 왜 wheel이 아닌 sdist로 설치되는가"를 확인한다.

::quiz{id="unexpected-source-build"}

## PyTorch 설치의 특수성

PyTorch는 같은 버전이라도 CUDA 버전마다, 그리고 CPU 전용으로 여러 **빌드 변형**이 있다. 이 변형들은 PyPI 하나에 다 올라가지 않는다.

- **PyPI의 기본 wheel.** 집필 시점 PyPI의 `torch` 2.15.0 Linux wheel은 메타데이터에 `cuda-toolkit[...]==13.2.2`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` 같은 의존성을 선언한다. 즉 PyPI에서 그대로 설치하면 CUDA 13 계열 빌드와 그 runtime 라이브러리 패키지들이 함께 설치된다(3.1장). 기본값이 어느 CUDA인지는 버전마다 바뀌므로, 설치 전에 `uv tree`나 패키지 메타데이터로 확인한다.
- **PyTorch 자체 패키지 index.** 다른 CUDA 버전이나 CPU 전용 빌드는 `https://download.pytorch.org/whl/<변형>` 형식의 별도 index에서 받는다(변형 이름 예: `cpu`, `cu128`). 어떤 변형이 있는지는 PyTorch 공식 설치 페이지에서 확인한다.

uv 프로젝트에서는 특정 패키지만 별도 index에서 받도록 `pyproject.toml`에 적는다. 다음은 uv 문서의 PyTorch 안내를 따른 형식이다(집필 시점 예시. index 이름과 URL은 공식 페이지에서 확인한다).

```toml
[project]
dependencies = ["torch>=2.8"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true            # 아래 sources에서 지정한 패키지만 이 index에서 찾는다

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

`explicit = true`가 핵심이다. 이 index는 `torch`에만 쓰이고, 나머지 패키지는 PyPI에서 온다. 이 설정이 lockfile에 기록되므로 다른 사람도 같은 변형을 받는다.

pip에서 흔히 쓰는 `--extra-index-url`은 이와 다르다. pip은 여러 index를 우선순위 없이 모두 후보로 보고 **가장 높은 버전**을 고른다. 그래서 의도와 다른 index의 파일이 선택되거나, 같은 이름의 악성 패키지가 다른 index에 올라오는 공격(dependency confusion)에 노출될 수 있다. pip에서 특정 변형을 확실히 받으려면 `--index-url`로 index를 하나만 지정해 PyTorch 관련 패키지만 따로 설치하는 방법이 안전하다.

설치 후에는 늘 `python -c "import torch; print(torch.__version__, torch.version.cuda)"`로 무엇이 설치되었는지 확인한다. 별도 index의 빌드는 버전 문자열에 `+cu128` 같은 꼬리표(local version label)가 붙어 변형을 구분할 수 있다.

## 설치와 개발 모드

분석 코드가 커지면 노트북과 스크립트에 흩어진 함수들을 하나의 패키지로 묶게 된다. 이때 두 가지 관례가 도움이 된다.

**src layout.** 패키지 코드를 저장소 루트가 아니라 `src/` 아래에 둔다.

```text
my-analysis/
├── pyproject.toml
├── uv.lock
├── src/
│   └── my_analysis/
│       ├── __init__.py
│       └── features.py
├── scripts/train.py
└── tests/test_features.py
```

루트에 `my_analysis/`를 두면, 저장소 루트에서 실행한 Python은 설치 여부와 상관없이 현재 디렉터리의 코드를 import한다. 그래서 "설치하지 않아도 되는" 것처럼 보이다가, 다른 디렉터리나 다른 머신에서 실행하면 실패한다. `src/` 아래에 두면 **설치된 패키지로만** import되므로, 테스트가 사용자가 실제로 쓰게 될 형태를 검사한다.

**editable install.** 개발 중에는 코드를 고칠 때마다 다시 설치하지 않도록 **editable 모드**로 설치한다. 환경의 site-packages에 소스 디렉터리를 가리키는 연결만 두는 방식이라, 소스를 고치면 바로 반영된다. pip에서는 `pip install -e .`이고, uv에서는 패키지로 빌드되는 프로젝트(`build-system`이 있는 프로젝트)라면 `uv sync`가 프로젝트 자신을 editable로 설치한다. 단, C/CUDA 확장이 있는 패키지는 확장 코드를 고치면 다시 빌드해야 한다.

**개발 의존성.** pytest·린터·문서 도구는 `[dependency-groups]`에 두어, 사용자에게 배포되는 의존성과 섞지 않는다. 배포용 이미지나 서버에서는 `uv sync --no-dev`로 개발 그룹을 빼고 설치할 수 있다.

## 충돌 해결

모든 제약을 동시에 만족하는 버전 조합이 없으면 resolver는 실패한다. resolver는 대체로 "높은 버전부터 시도하고, 막히면 되돌아가(backtracking) 다른 버전을 시도"하는 탐색을 한다. 이 원리를 작은 가상 index로 흉내 낸다.

```python {exec="browser" session="deps"}
def V(s):
    return tuple(int(p) for p in s.split("."))

def ok(v, spec):                                 # 이 예에서 쓰는 >=, ==, < 만 지원
    for c in spec.split(","):
        op = c[:2] if c[:2] in (">=", "==") else c[0]
        a, b = V(v), V(c[len(op):])
        if not {">=": a >= b, "==": a == b, "<": a < b}[op]:
            return False
    return True

available = {"pandas": ["3.1.1", "2.2.3"], "scipy": ["1.16.0", "1.13.1"], "numpy": ["2.5.4", "1.26.4"]}
deps = {("pandas", "3.1.1"): {"numpy": ">=2.0.2"}, ("pandas", "2.2.3"): {"numpy": ">=1.26"},
        ("scipy", "1.16.0"): {"numpy": ">=2.0"}, ("scipy", "1.13.1"): {"numpy": ">=1.22,<2.3"}}

def resolve(reqs, pinned):
    for name, spec in reqs:
        if name in pinned:
            if not ok(pinned[name], spec):
                return None                      # 이미 고른 버전과 충돌: 되돌아간다
            continue
        for v in available[name]:                # 높은 버전부터
            if ok(v, spec):
                found = resolve(reqs + list(deps.get((name, v), {}).items()), {**pinned, name: v})
                if found:
                    return found
        return None
    return pinned

print(resolve([("pandas", ">=2.2"), ("scipy", ">=1.11")], {}))
print(resolve([("pandas", ">=2.2"), ("scipy", ">=1.11"), ("numpy", "<2")], {}))
print(resolve([("pandas", ">=3"), ("numpy", "<2")], {}))
```

```output
{'pandas': '3.1.1', 'scipy': '1.16.0', 'numpy': '2.5.4'}
{'pandas': '2.2.3', 'scipy': '1.13.1', 'numpy': '1.26.4'}
None
```

제약이 없으면 모두 최신이 된다. `numpy<2`를 추가하면 resolver는 pandas와 SciPy를 NumPy 1.x를 지원하는 이전 버전으로 **내려서** 해를 찾는다. 사용자가 직접 요구하지 않은 패키지의 버전이 바뀐다는 점에 주의한다. `pandas>=3`과 `numpy<2`처럼 정말 양립할 수 없으면 해가 없다(`None`). 실제 resolver(uv의 resolver는 PubGrub이라는 알고리즘을 쓴다)는 이때 **왜 해가 없는지**를 증명하는 메시지를 낸다. 실습에서 실제로 얻은 메시지는 다음과 같다.

```output
  × No solution found when resolving dependencies for split (markers:
  │ sys_platform == 'linux'):
  ╰─▶ Because pandas>=3.1.1 depends on numpy>=2.0.2 and your project depends
      on numpy<2, we can conclude that your project and pandas>=3.1.1 are
      incompatible.
      And because your project depends on pandas==3.1.1, we can conclude that
      your project's requirements are unsatisfiable.
```

**resolver 오류 읽기.** 이 메시지는 추론의 사슬이다. "A가 B를 요구하고 프로젝트가 C를 요구하므로 A와 프로젝트는 양립 불가", "그런데 프로젝트가 A를 요구하므로 전체가 불가능". 사슬에서 **프로젝트가 직접 건 제약**(여기서는 `numpy<2`와 `pandas==3.1.1`)을 찾으면, 바꿀 수 있는 손잡이가 보인다. 첫 줄의 `split (markers: ...)`는 uv가 여러 플랫폼을 함께 풀다가 어느 플랫폼에서 실패했는지를 알려 준다.

::quiz{id="read-resolver-error"}

**해결 순서.**

1. **누가 무엇을 요구하는지 확인한다.** `uv tree --invert --package numpy`처럼 역방향 트리로 어떤 패키지가 NumPy에 제약을 거는지 본다. pip 환경에서는 `pip check`가 설치된 패키지 사이의 불일치를 보고한다.
2. **자기 제약부터 의심한다.** 충돌의 대부분은 프로젝트가 과하게 건 제약에서 나온다. "예전에 문제가 있어서" 걸어 둔 상한(`numpy<2`)이 지금도 필요한지 확인하고, 필요 없으면 완화한다.
3. **상대 패키지의 버전을 옮긴다.** 제약이 꼭 필요하다면, 그 제약과 양립하는 상대 패키지 버전(예: NumPy 1.x를 지원하는 pandas)으로 옮긴다. 그 버전에 현재 Python용 wheel이 있는지도 확인한다(앞 절의 소스 빌드).
4. **최소 재현을 만든다.** 해결이 어려우면 빈 프로젝트에 충돌하는 두 패키지만 넣어 같은 오류가 나는지 본다. 원인이 좁혀지고, 패키지 관리자에게 질문하거나 이슈를 올릴 때도 이것이 필요하다.
5. **강제 덮어쓰기는 마지막 수단이다.** uv의 `override-dependencies` 같은 설정으로 패키지가 선언한 제약을 무시하게 할 수 있지만, 그 패키지가 실제로 그 버전에서 동작하는지는 아무도 보장하지 않는다. 쓴다면 이유와 검증 결과를 함께 기록한다.

## 실습

의도적으로 충돌하는 의존성 조합을 만들고, resolver 메시지를 해석해 해결한 뒤 lockfile로 재현한다. 코드는 `labs/environments/dependencies-and-packaging/conflict_lab.py`이고, 임시 디렉터리에서 uv로 다음을 한다.

1. `pandas==3.1.1`과 `numpy<2`를 함께 선언하고 `uv lock`을 실행한다. 이 실습은 Linux에서만 풀리면 되도록 `[tool.uv]`의 `environments`를 제한했다.
2. 메시지가 지목한 NumPy 제약을 `numpy>=2`로 완화하고 다시 lock한 뒤 의존성 트리를 본다.
3. `uv.lock`에 NumPy가 어떻게 기록되었는지 본다.
4. `pyproject.toml`과 `uv.lock`만 새 디렉터리로 복사해 `uv sync --locked`로 재현하고, 설치 목록을 비교한다.

집필 환경에서의 실행 결과다(1단계의 오류 메시지는 앞 절에 실었다).

```output
== 2. 해결: 메시지가 지목한 numpy 제약을 완화한다
uv lock 종료 코드 0
conflict-lab v0.1.0
├── numpy v2.5.4
└── pandas v3.1.1
    ├── numpy v2.5.4
    └── python-dateutil v4.0.0
        └── tzdata v2026.6

== 3. uv.lock에 기록된 numpy
[[package]]
name = "numpy"
version = "2.5.4"
source = { registry = "https://pypi.org/simple" }
... (이 패키지에 대해 기록된 wheel 4개, 각 wheel의 sha256 해시 포함)

== 4. 재현: pyproject.toml과 uv.lock만 복사해 새 디렉터리에서 sync --locked
sync 종료 코드 0, 설치 목록 동일: True
numpy==2.5.4
pandas==3.1.1
python-dateutil==4.0.0
tzdata==2026.6
```

직접 선언한 것은 두 패키지지만 설치된 것은 네 패키지다. lockfile에는 NumPy 하나에 대해서도 Linux의 여러 아키텍처·C 라이브러리용 wheel 4개가 해시와 함께 기록되어, 같은 lockfile로 x86_64와 aarch64 머신 모두에서 같은 버전을 설치할 수 있다. 자신의 연구 프로젝트에 같은 과정을 적용하고, 해결한 결과의 `pyproject.toml`과 `uv.lock`을 커밋한다. 커밋 메시지에는 어떤 충돌을 어떤 근거로 어떻게 풀었는지 적는다.

이어서 연습문제를 푼다. resolver의 가장 작은 부품인 버전 제약 검사기를 직접 구현한다.

::exercise{id="version-specifier"}

## 실패 사례

- **`pip freeze` 결과를 그대로 의존성 선언으로 쓰는 것.** 직접·전이 의존성이 섞이고 모든 버전이 고정되어, 무엇이 왜 필요한지 알 수 없게 된다. 다른 OS나 Python 버전에서는 그 정확한 버전의 wheel이 없어 설치가 실패하거나 소스 빌드로 넘어간다. 선언은 직접 의존성과 범위로, 고정은 lockfile로 나눈다.
- **버전 제약 없이 최신 버전을 설치해 결과가 달라지는 것.** 의존성을 이름만 적고 lockfile을 커밋하지 않으면, 같은 코드가 설치 시점에 따라 다른 라이브러리 버전으로 실행된다. 기본값이 바뀐 함수 하나가 수치 결과를 조용히 바꿀 수 있다.
- **충돌을 피하려고 모든 의존성을 `==`로 고정하는 것.** 당장은 풀리지만, 다음 패키지를 추가할 때 resolver가 움직일 여지가 없어 더 큰 충돌이 난다. 보안 수정도 받을 수 없다.
- **`--extra-index-url`로 PyTorch index를 추가하는 것.** pip은 index 사이에 우선순위를 두지 않으므로 의도와 다른 index의 파일이 선택될 수 있다.
- **lockfile만 커밋하고 `pyproject.toml`의 선언을 관리하지 않는 것.** 다음에 의존성을 갱신할 때 무엇을 기준으로 다시 풀어야 할지 알 수 없다.

## 통과 기준

- 자신의 프로젝트를 `pyproject.toml`(직접 의존성과 범위, 개발 의존성 그룹)과 lockfile로 관리하고, 새 머신(또는 새 디렉터리)에서 `uv sync --locked`(또는 사용하는 도구의 동등한 명령)만으로 같은 패키지 목록의 환경을 재생성한다.
- resolver 오류 메시지에서 충돌하는 제약과 그 출처를 찾아 설명하고, 해결 근거를 커밋 메시지에 기록한다.
- 연습문제 `version-specifier`의 테스트를 통과한다.
