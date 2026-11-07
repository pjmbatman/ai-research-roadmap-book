---
id: environments/environment-diagnosis
title: "환경 진단·재현·유지보수"
title_en: "Diagnosing and Maintaining Environments"
tier: core
status: draft
prerequisites: [environments/docker]
---

# 환경 진단·재현·유지보수

환경 문제는 반드시 생긴다. 드라이버가 업데이트되고, 패키지가 새 버전을 내고, 새 GPU가 들어오고, 동료가 "급한 불"을 끄려고 패키지 하나를 손으로 설치한다. 문제가 생겼을 때 흔한 반응은 검색 결과의 명령을 하나씩 실행해 보거나, 환경을 지우고 다시 만드는 것이다. 운이 좋으면 해결되지만 원인은 모른 채로 남고, 같은 문제가 다음 달에 다시 생긴다.

이 장은 앞의 다섯 장에서 배운 계층 구조를 **진단 절차**로 바꾼다. 증상으로 실패한 층을 가리고, 근거를 수집해 가설을 확인하고, 다른 사람이 재현할 수 있는 최소 예제로 줄인다. 그리고 문제가 생기기 전에 할 일, 즉 변경 전 상태를 기록하고 업그레이드와 rollback 정책을 세우는 법을 다룬다. 명령은 집필 시점의 예시다.

## 학습 목표

- 환경 문제를 체계적으로 진단한다
- 환경 보고서와 최소 재현 예제를 만든다
- 업그레이드와 rollback 정책을 세운다

## 증상별 진단

진단의 기본 단위는 **증상 → 가설 → 근거 → 원인 → 해결**이다. 증상(오류 메시지)에서 실패했을 가능성이 높은 층을 가설로 세우고, 그 층의 상태를 확인하는 명령으로 근거를 모으고, 근거가 가설을 지지할 때만 해결책을 적용한다. 해결책을 적용한 뒤에는 증상이 사라졌는지 확인한다. 근거 없이 해결책부터 시도하면, 우연히 문제가 사라져도 원인은 모른다.

오류 메시지의 대부분은 어느 층에서 실패했는지 직접 말해 준다(3.1장의 계층 표). 메시지 패턴으로 층을 가리는 규칙을 코드로 쓰면 다음과 같다.

```python {exec="browser" session="diag"}
import re

rules = [  # (메시지 패턴, 실패한 층, 먼저 확인할 것)
    (r"couldn't communicate with the NVIDIA driver", "드라이버", "nvidia-smi, dkms status, uname -r"),
    (r"Driver/library version mismatch", "드라이버", "재부팅 여부, /proc/driver/nvidia/version"),
    (r"no kernel image is available", "바이너리와 GPU 아키텍처", "torch.cuda.get_arch_list()"),
    (r"cannot open shared object file", "동적 링커", "ldd, LD_LIBRARY_PATH"),
    (r"undefined symbol", "ABI·빌드 버전", "빌드한 torch 버전, C++ ABI"),
    (r"No module named", "Python 환경", "sys.executable, pip --version"),
    (r"CUDA out of memory", "자원(GPU 메모리)", "nvidia-smi 사용량, batch 크기"),
    (r"Segmentation fault", "네이티브 확장 코드", "python -X faulthandler, 최소 재현"),
]

def triage(msg):
    for pattern, layer, check in rules:
        if re.search(pattern, msg):
            return layer, check
    return "미분류", "전체 traceback과 환경 보고서"

for msg in ["RuntimeError: CUDA error: no kernel image is available for execution on the device",
            "OSError: libcudart.so.12: cannot open shared object file: No such file or directory",
            "ModuleNotFoundError: No module named 'torch'",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "Segmentation fault (core dumped)"]:
    layer, check = triage(msg)
    print(f"{msg[:48]:48s} -> [{layer}] {check}")
```

```output
RuntimeError: CUDA error: no kernel image is ava -> [바이너리와 GPU 아키텍처] torch.cuda.get_arch_list()
OSError: libcudart.so.12: cannot open shared obj -> [동적 링커] ldd, LD_LIBRARY_PATH
ModuleNotFoundError: No module named 'torch'     -> [Python 환경] sys.executable, pip --version
torch.OutOfMemoryError: CUDA out of memory. Trie -> [자원(GPU 메모리)] nvidia-smi 사용량, batch 크기
Segmentation fault (core dumped)                 -> [네이티브 확장 코드] python -X faulthandler, 최소 재현
```

이 표는 출발점일 뿐이다. 증상별로 확인할 것을 조금 더 자세히 정리한다.

**GPU 인식 실패**(`torch.cuda.is_available()`가 `False`). 위에서부터 아래로 확인한다.

1. CPU 전용 빌드인가: `torch.version.cuda`가 `None`이면 GPU를 쓸 수 없는 빌드다(3.4장의 설치 경로 확인).
2. 이 프로세스에 GPU가 보이는가: `CUDA_VISIBLE_DEVICES`가 빈 값이거나 엉뚱한 번호인지, 컨테이너라면 `--gpus`로 실행했는지(3.5장).
3. 드라이버가 동작하는가: `nvidia-smi`가 성공하는지, 드라이버가 PyTorch의 CUDA를 지원하는지(3.1장).

**CUDA error.** CUDA 연산은 비동기로 실행되므로, 오류가 실제로 난 곳보다 나중의 엉뚱한 줄에서 보고될 수 있다. `CUDA_LAUNCH_BLOCKING=1`을 설정하고 다시 실행하면 연산을 동기적으로 실행해 실제 위치에서 오류가 난다(느려지므로 진단할 때만 쓴다). `no kernel image is available`처럼 메시지가 원인을 직접 말하는 경우도 있고, `illegal memory access`처럼 코드의 버그(잘못된 인덱스 등)인 경우도 있다. 하드웨어 오류가 의심되면 kernel 로그(`sudo dmesg`)에서 NVIDIA 드라이버가 남긴 Xid 메시지를 찾는다.

**missing shared library**(`cannot open shared object file`). 동적 링커가 `.so` 파일을 찾지 못했다. 어느 파일이 해석되지 않는지 `ldd`로 확인한다(집필 시점 예시).

```bash
# 어떤 공유 라이브러리가 '=> not found'인가
ldd /path/to/failing_extension.so | grep "not found"
# 동적 링커가 어디를 뒤지는지 자세히 본다
LD_DEBUG=libs python -c "import failing_module" 2>&1 | grep -i "trying file" | head
# 시스템 라이브러리 cache에 있는가
ldconfig -p | grep libcudart
```

원인은 대개 셋 중 하나다. 그 라이브러리를 제공하는 패키지가 이 환경에 없다(다른 환경에 설치했다), `LD_LIBRARY_PATH`가 다른 버전을 가리키거나 비어 있다, 시스템 toolkit과 환경 안의 라이브러리가 섞였다(3.2장).

**import 오류.** `ModuleNotFoundError`는 3.3장의 "어떤 python이 실행되는가"부터 확인한다. 모듈은 찾았는데 다른 버전이 import된다면 `python -c "import numpy; print(numpy.__file__, numpy.__version__)"`로 실제 파일 위치를 본다. 가려진(shadowed) 패키지가 흔한 원인이다.

**버전 충돌.** 설치는 성공했는데 실행 중에 `AttributeError`나 이상한 동작이 나면, 설치된 패키지들이 서로의 의존성 제약을 만족하는지 `pip check`(또는 `uv pip check`)로 확인한다. 누군가 `--no-deps`나 강제 재설치로 제약을 깨뜨렸을 수 있다.

**Segmentation fault.** Python 코드가 아니라 C/C++/CUDA로 작성된 확장 모듈 안에서 프로세스가 죽은 것이다. traceback이 남지 않으므로 `python -X faulthandler script.py`로 실행해 죽는 순간의 Python 쪽 호출 위치를 출력하게 한다. ABI 불일치나 서로 다른 버전의 같은 라이브러리가 한 프로세스에 함께 로드된 경우가 흔하다.

::quiz{id="no-kernel-image"}

## 환경 정보 수집

진단에도, 다른 사람에게 도움을 청할 때도 **환경 보고서**가 필요하다. 보고서는 손으로 옮겨 적지 않고 명령의 출력을 그대로 붙인다. 손으로 쓰면 "CUDA 12"처럼 어느 층의 어느 값인지 모호해진다.

| 층 | 수집 명령(집필 시점 예시) |
|---|---|
| 하드웨어 | `nvidia-smi -q`(GPU 상세), `lscpu`, `free -h`, `df -h` |
| OS·kernel | `cat /etc/os-release`, `uname -r`, `ldd --version` |
| 드라이버 | `nvidia-smi`, `cat /proc/driver/nvidia/version` |
| Python 환경 | `which python`, `python -VV`, `pip freeze` 또는 `uv pip freeze` |
| PyTorch 스택 | `python -m torch.utils.collect_env` |
| 컨테이너 | image 이름과 digest, `docker run` 명령 전체 |

PyTorch의 `python -m torch.utils.collect_env`는 PyTorch·CUDA 빌드 버전, OS, 컴파일러, Python, GPU 모델, 드라이버, cuDNN, 관련 패키지 버전을 한 번에 출력하는 공식 스크립트로, PyTorch에 버그를 보고할 때 붙이도록 안내되는 정보이기도 하다. NVIDIA 드라이버 문제는 드라이버와 함께 설치되는 `nvidia-bug-report.sh`가 더 자세한 로그를 모은다.

실험 코드 안에서도 보고서를 만들어 결과와 함께 저장해 두면, 몇 달 뒤 "이 결과는 어떤 환경에서 나왔나"에 답할 수 있다. 표준 라이브러리만으로 최소한의 보고서를 만든다. 로컬 환경을 조사하므로 로컬 Python에서 실행하며, 출력은 이 책의 실습 환경에서 얻은 것이다.

```python {exec="python"}
import json
import platform
import sys
from importlib import metadata

def env_report(packages):
    r = {"python": platform.python_version(), "platform": platform.platform(),
         "in_venv": sys.prefix != sys.base_prefix, "packages": {}}
    for name in packages:
        try:
            r["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            r["packages"][name] = None                 # 없는 것도 기록한다
    return r

print(json.dumps(env_report(["numpy", "matplotlib", "torch"]), indent=2))
```

```output
{
  "python": "3.13.18",
  "platform": "Linux-6.18.44-fc-v310-x86_64-with-glibc2.39",
  "in_venv": true,
  "packages": {
    "numpy": "2.5.4",
    "matplotlib": "3.11.2",
    "torch": null
  }
}
```

`"torch": null`처럼 **없다는 사실도 기록**한다. 보고서에서 항목이 빠져 있으면 "확인하지 않았다"인지 "없었다"인지 구분할 수 없다. 3.1장의 `collect_versions.py`가 시스템 층까지 확장한 버전이다. 학습 스크립트가 시작할 때 이런 보고서를 결과 디렉터리에 JSON으로 저장하게 해 두면, 실험 기록과 환경 기록이 함께 남는다.

:::note[분야 예시]
기상·기후 모델 출력이나 분자동역학 궤적처럼 같은 입력에서도 부동소수점 연산 순서에 따라 결과가 미세하게 달라질 수 있는 계산에서는, 결과 파일에 환경 보고서(라이브러리 버전, BLAS 구현, GPU 모델, 스레드 수)를 메타데이터로 함께 저장하는 관행이 유용하다. 나중에 두 실행의 결과가 다를 때, 차이가 모델의 변화 때문인지 환경의 변화 때문인지를 가르는 첫 번째 근거가 된다.
:::

## 최소 재현 예제

다른 사람(라이브러리 개발자, 동료, 미래의 자신)이 문제를 확인하려면 **최소 재현 예제**(minimal reproducible example)가 필요하다. 조건은 셋이다.

- **최소**: 문제와 무관한 코드·데이터·의존성이 없다.
- **완결**: 복사해 실행하면 그대로 돈다. 사내 경로나 비공개 데이터에 기대지 않는다.
- **재현**: 실행하면 실제로 같은 오류가 난다. 직접 실행해 확인했다.

만드는 방법은 "오류가 유지되는 한 덜어 낸다"이다.

1. 현재 상태(코드 커밋, 환경 보고서, lockfile)를 기록한다.
2. 데이터 로딩을 같은 shape·dtype·값 범위의 합성 데이터로 바꾼다(예: 실제 이미지 대신 같은 크기의 난수 텐서). 오류가 유지되는지 확인한다.
3. 모델·학습 루프·로깅·설정 코드를 하나씩 지운다. 지울 때마다 오류가 유지되는지 확인한다. **오류가 사라지면 방금 지운 것이 원인과 관련 있다**는 정보이므로 되돌리고 기록한다.
4. 의존성을 줄인다. 새 가상환경에 꼭 필요한 패키지만 설치해 재현되는지 본다. 재현되지 않으면 환경 차이가 원인이다. 아래 '변경 관리' 절의 보고서 비교로 넘어간다.
5. 결과물: 코드 몇십 줄, 환경 보고서, 실행 명령, 기대한 동작과 실제 동작.

이 과정에서 원인이 저절로 드러나는 경우가 많다. 최소 재현 예제를 만드는 것은 질문을 위한 준비이자, 그 자체로 가장 강력한 진단 방법이다.

::quiz{id="minimal-reproduction"}

## 변경 관리

환경 문제의 상당수는 **무엇이 바뀌었는가**를 알면 바로 풀린다. 그래서 변경 관리의 핵심은 변경 전에 상태를 기록하는 것이다. 변경 전후의 환경 보고서가 있으면 차이를 기계적으로 뽑을 수 있다. 가상의 기록으로 확인한다.

```python {exec="browser" session="diag"}
before = {"kernel": "6.8.0-45-generic", "driver": "550.54.15", "driver_max_cuda": "12.4",
          "torch": "2.5.1+cu124", "numpy": "2.1.3", "smoke_test": "pass"}
after = {"kernel": "6.8.0-48-generic", "driver": "550.54.15", "driver_max_cuda": "12.4",
         "torch": "2.5.1+cu124", "numpy": "2.2.0", "smoke_test": "fail"}

for key in before:
    if before[key] != after[key]:
        print(f"{key:12s} {before[key]:>18s} -> {after[key]}")
```

```output
kernel         6.8.0-45-generic -> 6.8.0-48-generic
numpy                     2.1.3 -> 2.2.0
smoke_test                 pass -> fail
```

smoke test를 깨뜨린 후보가 kernel과 NumPy 둘로 좁혀졌다. 드라이버는 그대로이므로 kernel 변경이 드라이버를 깨뜨린 경우(3.2장)는 아니라는 것도 알 수 있다. 다음 단계는 NumPy만 이전 버전으로 되돌려 smoke test를 돌려 보는 것이다. 한 번에 하나만 바꾸어야 원인을 가를 수 있다.

**이분 탐색(bisect).** 바뀐 것이 많으면 하나씩 시험하는 대신 반씩 나눈다. lockfile이 git으로 관리되고 있다면, 통과하던 커밋과 실패하는 커밋 사이에서 `git bisect`로 lockfile의 어느 변경이 문제를 일으켰는지 찾을 수 있다. 후보가 $N$개면 시험은 약 $\log_2 N$번이면 된다. 각 단계에서 그 커밋의 lockfile로 환경을 `uv sync --locked`로 맞추고 smoke test를 돌리는 스크립트를 `git bisect run`에 넘기면 자동으로 진행된다.

::quiz{id="bisect-steps"}

**업그레이드 정책.** 공유 환경(서버의 드라이버, 연구실 표준 image, 공동 프로젝트의 lockfile)에 대한 규칙을 미리 정한다.

- **주기**: 정기 업그레이드 시점을 정한다(예: 학기 사이, 마감 직후). 마감 직전에는 보안 수정 외의 업그레이드를 하지 않는다.
- **한 번에 한 층**: 드라이버와 PyTorch와 CUDA toolkit을 한꺼번에 올리지 않는다. 문제가 생기면 후보가 셋이 된다.
- **기록**: 변경 전후의 환경 보고서, 실행한 명령, 이유를 운영 일지(3.2장)나 커밋 메시지에 남긴다.
- **검증**: 업그레이드 직후 smoke test를 돌리고, 진행 중인 프로젝트의 대표 실험 하나를 재실행해 이전 결과와 비교한다.

**rollback 절차.** rollback은 "이전 상태로 정확히 돌아가는 것"이므로, 이전 상태가 기록되어 있어야만 가능하다.

| 층 | 이전 상태의 기록 | 되돌리는 방법 |
|---|---|---|
| Python 패키지 | git의 이전 `uv.lock` | 이전 커밋의 lockfile로 `uv sync --locked` |
| 컨테이너 | 이전 image의 digest | 그 digest로 실행 |
| 드라이버·toolkit | 운영 일지의 패키지 버전, `apt-mark` hold 목록 | 이전 버전 패키지 재설치, 필요하면 이전 kernel로 부팅 |

rollback 절차는 실제로 한 번 해 봐야 믿을 수 있다. 업그레이드 계획에 "rollback을 시험해 본다"를 포함한다.

## 검증

환경을 만들거나 바꾼 뒤에는 **smoke test**로 확인한다. smoke test는 짧은 시간(수십 초~몇 분) 안에, 환경의 모든 층이 동작하는지 끝에서 끝까지(end-to-end) 확인하는 테스트다. 연구 프로젝트의 smoke test에는 대체로 다음이 들어간다.

- 핵심 패키지 import와 버전 확인(환경 보고서 출력)
- GPU 프로젝트라면 GPU에서 작은 연산과 역전파 한 번
- 작은 합성 데이터로 학습 몇 step과 checkpoint 저장·불러오기
- 고정된 시드의 작은 계산 결과를 기대값과 **허용 오차 안에서** 비교(3.5장의 `smoke_test.py`)

새 사용자를 위한 최종 검증은 **새 환경에서 설치부터 smoke test까지**다. 설치 문서만 보고, 아무것도 없는 머신(또는 새 컨테이너, 새 계정)에서 시작해 smoke test 통과까지 간다. 이 과정에서 문서에 빠진 단계, 개인 설정 파일(`~/.bashrc`, `~/.local`)에 숨어 있던 의존성, 기록되지 않은 시스템 패키지가 드러난다. 문서를 쓴 사람이 아닌 다른 사람이 해 보는 것이 가장 좋다.

## 실습

고장 난 환경 세 가지를 진단하고, 각각 원인·근거·해결을 기록한 진단 보고서를 작성한다. `labs/environments/environment-diagnosis/broken_envs.py`가 임시 디렉터리에 세 가지 고장을 만들고, 증상과 진단 근거를 출력한다(Linux, gcc, PyPI 접속이 필요하다).

1. **설치했는데 import가 안 된다**: 가상환경 두 개를 만들고, 한 환경의 pip으로 설치한 뒤 다른 환경의 python으로 import한다.
2. **공유 라이브러리를 찾지 못한다**: 작은 C 공유 라이브러리를 gcc로 빌드해 표준 검색 경로 밖에 두고, Python에서 `ctypes`로 불러온다.
3. **설치는 성공했는데 의존성 제약이 깨져 있다**: pandas를 설치한 뒤, 누군가 `--no-deps`로 pandas가 요구하는 것보다 낮은 버전의 python-dateutil을 덮어 설치한다.

집필 환경에서의 실행 결과다(`<tmp>`는 시나리오별 임시 디렉터리).

```output
== 시나리오 1: 설치했는데 import가 안 된다
  증상: ModuleNotFoundError: No module named 'six'
  근거: 실행한 python의 sys.prefix = <tmp>/proj-a
  근거: 사용한 pip --version      = <tmp>/proj-b/lib/python3.13/site-packages/pip (python 3.13)
  해결 확인: python -m pip으로 설치한 뒤 import 종료 코드 = 0
== 시나리오 2: 공유 라이브러리를 찾지 못한다
  증상: OSError: libtoy.so.1: cannot open shared object file: No such file or directory
  근거: 시스템 라이브러리 cache(ldconfig -p)에서 libtoy 항목 수 = 0
  근거: 파일은 있다 -> ['libtoy.so.1']
  해결 확인: LD_LIBRARY_PATH에 경로를 넣고 실행 -> toy_add(2, 3) = 5
== 시나리오 3: 설치는 성공했는데 의존성 제약이 깨져 있다
  증상/근거: pip check 종료 코드 1 -> pandas 3.1.1 has requirement python-dateutil>=2.9.0, but you have python-dateutil 2.8.2.
  해결 확인: 제약에 맞게 다시 설치한 뒤 pip check -> No broken requirements found.
```

시나리오마다 다음 형식으로 보고서를 쓴다. 출력의 "해결 확인"은 보고서를 쓴 뒤에 맞춰 본다.

```text
## 시나리오 N
- 증상: (오류 메시지 그대로)
- 가설: (실패한 층과 그 이유)
- 근거: (실행한 명령과 출력)
- 원인: (근거가 지지하는 원인)
- 해결: (적용한 조치)와 확인: (증상이 사라졌음을 보인 명령과 출력)
- 재발 방지: (이 문제가 다시 생기지 않게 바꿀 규칙)
```

시나리오 2의 "해결"을 `LD_LIBRARY_PATH`로 끝내지 말고, 재발 방지까지 생각한다. 매번 환경변수를 설정해야 하는 해결은 다른 셸, 다른 사용자, 배치 작업에서 다시 깨진다. 라이브러리를 패키지에 포함시키거나, 실행 스크립트에서 경로를 명시하는 방법을 비교한다.

이어서 연습문제를 푼다. 업그레이드 전후의 freeze 목록을 비교해 추가·삭제·변경된 패키지를 찾는 함수를 작성한다.

::exercise{id="diff-requirements"}

## 실패 사례

- **원인을 모른 채 재설치를 반복하는 것.** 환경을 지우고 다시 만들면 문제가 사라질 수 있지만, 원인이 환경 밖(드라이버, 셸 설정, `~/.local`, `LD_LIBRARY_PATH`)에 있으면 그대로 다시 생긴다. 사라졌더라도 무엇이 바뀌어서 사라졌는지 모르므로 다음 번에 쓸 지식이 남지 않는다. 재설치 전에 현재 상태의 보고서를 저장하고, 재설치 후의 보고서와 비교한다.
- **문제 보고에 환경 정보를 빠뜨리는 것.** "PyTorch에서 CUDA 오류가 납니다"만으로는 아무도 도울 수 없다. 첫 답장이 "환경 정보를 주세요"가 되어 하루가 지나간다. 전체 traceback, 환경 보고서, 최소 재현 코드를 처음부터 붙인다.
- **여러 층을 한꺼번에 업그레이드하는 것.** 드라이버, CUDA, PyTorch, 다른 패키지를 같은 날 올린 뒤 결과가 달라지면 원인 후보가 너무 많다.
- **오류 메시지의 마지막 줄만 보는 것.** traceback의 마지막 줄은 결과이고, 원인은 위쪽(처음 실패한 import, 처음 경고)에 있는 경우가 많다. 전체를 읽고, 가장 먼저 나온 오류부터 본다.
- **"내 컴퓨터에서는 된다"로 끝내는 것.** 자신의 환경에서 된다는 것은 그 환경에서만 된다는 증거다. 두 환경의 보고서를 비교하는 것이 다음 단계다.

## 통과 기준

- 고장 난 환경 세 가지에 대해 증상·가설·근거·원인·해결·재발 방지를 기록한 진단 보고서를 작성한다.
- 설치 문서와 lockfile 또는 image로, 문서를 쓰지 않은 새 사용자가 새 환경에서 smoke test를 통과한다.
- 연습문제 `diff-requirements`의 테스트를 통과한다.
