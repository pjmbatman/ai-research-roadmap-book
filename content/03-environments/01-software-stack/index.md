---
id: environments/software-stack
title: "AI 소프트웨어 스택의 계층"
title_en: "Layers of the AI Software Stack"
tier: core
status: draft
prerequisites: [linux-and-remote/linux-shell]
---

# AI 소프트웨어 스택의 계층

`import torch` 한 줄이 GPU에서 행렬을 곱하기까지, 그 아래에는 여러 층의 소프트웨어가 겹쳐 있다. Python 인터프리터, PyTorch, cuDNN 같은 가속 라이브러리, CUDA runtime, NVIDIA 드라이버, 운영체제 kernel, 그리고 하드웨어다. 각 층은 서로 다른 사람이 서로 다른 방법으로 설치하고, 버전도 따로 매겨진다. "이 서버의 CUDA 버전이 뭐예요?"라는 질문에 세 사람이 세 가지 다른 숫자로 답하는 일이 흔한 이유다.

이 장은 스택을 계층으로 나누어 보는 눈을 기른다. 오류 메시지는 대개 맨 위층(Python)에서 보이지만, 원인은 아래층 어딘가에 있다. 어느 층의 무슨 버전이 어느 층의 무엇과 맞아야 하는지 알면, 설치 문제를 추측과 재설치가 아니라 진단으로 풀 수 있다.

## 학습 목표

- 하드웨어부터 프레임워크까지의 계층을 설명한다
- CUDA 관련 버전 정보를 구분해 읽는다

## 계층 구조

GPU에서 PyTorch를 실행하는 서버의 스택을 아래에서 위로 쓰면 다음과 같다.

| 층 | 구성 요소 | 주로 누가·어떻게 설치하나 | 확인 방법 |
|---|---|---|---|
| 하드웨어 | CPU 아키텍처(x86_64, aarch64), GPU 모델과 compute capability | 구매 | `uname -m`, `nvidia-smi` |
| OS·kernel | 배포판(Ubuntu 등), Linux kernel, C 라이브러리(glibc) | 관리자 | `cat /etc/os-release`, `uname -r`, `ldd --version` |
| NVIDIA 드라이버 | kernel 모듈(`nvidia.ko`)과 사용자 공간 라이브러리(`libcuda.so`) | 관리자(root) | `nvidia-smi`, `/proc/driver/nvidia/version` |
| CUDA runtime·toolkit | runtime 라이브러리(`libcudart`), 컴파일러 `nvcc`, 헤더 | 관리자 또는 패키지 관리자 | `nvcc --version` |
| 가속 라이브러리 | cuBLAS(선형대수), cuDNN(신경망 연산), NCCL(GPU 간 통신) | 대개 PyTorch와 함께 | `torch.backends.cudnn.version()` |
| Python | 인터프리터와 표준 라이브러리 | 사용자(3.3장) | `python --version` |
| 프레임워크 | PyTorch와 그 위의 라이브러리 | 사용자(3.3장, 3.4장) | `torch.__version__` |

각 층은 **바로 아래층이 제공하는 인터페이스**에 기대어 동작한다. PyTorch는 cuDNN과 CUDA runtime의 함수를 부르고, CUDA runtime은 드라이버의 사용자 공간 라이브러리 `libcuda.so`를 부르며, 그 라이브러리는 kernel 모듈을 통해 GPU에 명령을 보낸다. 그래서 위층이 요구하는 최소 버전을 아래층이 만족하지 못하면 위층이 실패한다. 반대 방향은 대체로 괜찮다. 새 드라이버는 오래된 CUDA runtime으로 만든 프로그램도 실행한다.

두 가지 하드웨어 속성은 위층에서 자주 문제를 일으키므로 따로 기억한다.

- **CPU 아키텍처.** x86_64용으로 컴파일된 바이너리는 aarch64(ARM) 서버에서 실행되지 않는다. 같은 PyTorch 버전이라도 아키텍처마다 다른 패키지를 받는다.
- **GPU compute capability.** GPU 아키텍처 세대를 나타내는 번호다(예: A100은 8.0, H100은 9.0). PyTorch 바이너리는 지정된 compute capability 목록에 대해 미리 컴파일된 GPU 코드(kernel)를 담고 있다. 목록에 없는 새 GPU에서 오래된 바이너리를 쓰면 `no kernel image is available for execution on the device` 오류가 난다. 바이너리가 지원하는 목록은 `torch.cuda.get_arch_list()`로 확인한다.

**같은 GPU 문제, 다른 층.** 오류 메시지를 보면 대개 어느 층에서 실패했는지 알 수 있다.

| 메시지(일부) | 실패한 층 | 먼저 확인할 것 |
|---|---|---|
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | 드라이버(kernel 모듈) | 드라이버 설치·kernel 업데이트 여부(3.2장) |
| `libcudart.so.12: cannot open shared object file` | CUDA runtime 라이브러리를 찾는 동적 링커 | 라이브러리 경로, 환경 혼용(3.6장) |
| `CUDA error: no kernel image is available for execution on the device` | 프레임워크 바이너리와 GPU 아키텍처 | `torch.cuda.get_arch_list()`와 GPU의 compute capability |
| `undefined symbol: ...` | C++ ABI 또는 빌드 버전 불일치 | 아래 "컴파일러와 ABI" |
| `ModuleNotFoundError: No module named 'torch'` | Python 환경 | 어떤 python이 실행되었는지(3.3장) |

::quiz{id="stack-order"}

층마다 **관리 주체**가 다르다는 점도 실무에서 중요하다. 드라이버는 root 권한으로 시스템 전체에 하나만 설치되고, 바꾸면 서버의 모든 사용자에게 영향을 준다. 반대로 Python 환경과 PyTorch는 사용자가 프로젝트마다 따로 둘 수 있다. 그래서 공유 서버에서는 "드라이버는 넉넉히 새것으로, 나머지는 각자의 환경에서"가 기본 전략이 된다.

:::note[분야 예시]
기후·해양 모델을 돌리는 HPC 클러스터에서는 같은 스택이 **모듈 시스템**(Environment Modules, Lmod)으로 관리되는 경우가 많다. `module load cuda` 같은 명령이 `PATH`와 `LD_LIBRARY_PATH`를 바꿔 특정 toolkit을 선택한다. 계층 구조는 같고, 어느 층을 누가 고르는지만 다르다. 드라이버는 여전히 클러스터 관리자가 노드마다 설치하고, 사용자는 그 위의 층을 고른다.
:::

## 버전 정보 읽기

GPU 서버에서 "CUDA 버전"이라 불리는 숫자는 적어도 세 가지다.

| 출처 | 의미 |
|---|---|
| `nvidia-smi` 머리 줄의 `CUDA Version` | 설치된 **드라이버가 지원하는 최대** CUDA 버전. toolkit이 없어도 표시된다 |
| `nvcc --version`의 `release` | 시스템에 설치된 CUDA **toolkit**(컴파일러)의 버전. toolkit이 없으면 명령 자체가 없다 |
| `torch.version.cuda` | PyTorch 바이너리가 **빌드될 때 쓴** CUDA 버전. CPU 전용 빌드이면 `None` |

예시 출력에서 세 값을 읽어 본다(드라이버 550.54.15, toolkit 12.1은 형식을 보이기 위한 예다).

```python {exec="browser" session="stack"}
import re

smi = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |"
nvcc = "Cuda compilation tools, release 12.1, V12.1.105"
torch_cuda = "12.1"                      # torch.version.cuda가 돌려주는 값의 예

driver = re.search(r"Driver Version:\s*([\d.]+)", smi).group(1)
driver_cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi).group(1)
toolkit = re.search(r"release\s+([\d.]+)", nvcc).group(1)

print("드라이버 버전                :", driver)
print("드라이버가 지원하는 최대 CUDA :", driver_cuda, " <- nvidia-smi")
print("설치된 CUDA toolkit          :", toolkit, " <- nvcc --version")
print("PyTorch가 빌드된 CUDA        :", torch_cuda, " <- torch.version.cuda")
```

```output
드라이버 버전                : 550.54.15
드라이버가 지원하는 최대 CUDA : 12.4  <- nvidia-smi
설치된 CUDA toolkit          : 12.1  <- nvcc --version
PyTorch가 빌드된 CUDA        : 12.1  <- torch.version.cuda
```

::quiz{id="nvidia-smi-cuda-version"}

세 숫자가 서로 달라도 이상한 것이 아니다. 판단에 필요한 관계는 하나다. **PyTorch가 쓰는 CUDA runtime의 버전을 드라이버가 지원하는가.** 드라이버와 CUDA 사이의 호환 규칙은 세 가지로 정리된다.

- **하위 호환(backward compatibility).** 드라이버는 자기가 지원하는 최대 CUDA 버전 이하로 빌드된 프로그램을 실행한다. 드라이버가 12.4를 보고하면 11.8, 12.1, 12.4로 빌드된 프로그램은 모두 실행된다. 가장 기본적이고 안전한 규칙이다.
- **minor version compatibility.** CUDA 11부터는 같은 major 계열(예: 12.x) 안에서, 그 계열의 최소 드라이버 이상이면 더 높은 minor 버전으로 빌드된 프로그램도 대부분 실행된다. 집필 시점 NVIDIA 공식 표에서 CUDA 12.x 계열의 Linux 최소 드라이버는 525.60.13이다. 다만 새 드라이버가 있어야 하는 기능은 쓸 수 없고, PTX 코드를 실행 시점에 컴파일(JIT)하는 경로도 제한된다. 기본 규칙이 아니라 예외 규칙으로 다룬다.
- **forward compatibility.** 오래된 드라이버에서 새 major의 CUDA를 쓰기 위한 별도 호환 패키지가 있다. 지원 대상이 주로 데이터센터용 GPU로 한정되므로, 쓸 수 있는지는 GPU 모델과 공식 문서로 확인한다.

이 규칙을 그대로 코드로 옮기면 다음과 같다. 버전은 문자열이 아니라 정수 tuple로 비교한다. 문자열로 비교하면 `"12.10" < "12.4"`가 참이 되기 때문이다.

```python {exec="browser" session="stack"}
def ver(s):
    return tuple(int(p) for p in s.split("."))   # "12.4" -> (12, 4)

def judge(driver_cuda, runtime_cuda):
    d, r = ver(driver_cuda), ver(runtime_cuda)
    if r <= d:
        return "실행 가능"
    if r[0] == d[0]:
        return "같은 major: minor version compatibility (일부 기능 제한)"
    return "실행 불가: 드라이버 업그레이드 필요"

for runtime in ["11.8", "12.1", "12.4", "12.6", "13.0"]:
    print(f"드라이버 최대 CUDA 12.4, runtime {runtime:>4} -> {judge('12.4', runtime)}")
```

```output
드라이버 최대 CUDA 12.4, runtime 11.8 -> 실행 가능
드라이버 최대 CUDA 12.4, runtime 12.1 -> 실행 가능
드라이버 최대 CUDA 12.4, runtime 12.4 -> 실행 가능
드라이버 최대 CUDA 12.4, runtime 12.6 -> 같은 major: minor version compatibility (일부 기능 제한)
드라이버 최대 CUDA 12.4, runtime 13.0 -> 실행 불가: 드라이버 업그레이드 필요
```

::quiz{id="runtime-count"}

이 판정에 toolkit 버전(`nvcc`)이 등장하지 않는다는 점이 핵심이다. toolkit은 코드를 **컴파일할 때** 쓰이고, 이미 컴파일된 PyTorch를 **실행할 때**는 쓰이지 않는다. 연구에서의 의미는 분명하다. 공유 서버의 드라이버를 충분히 새 버전으로 유지하면, 사용자마다 다른 CUDA 버전의 PyTorch를 각자의 환경에서 쓸 수 있다. 버전을 맞춰야 하는 대상은 "서버의 CUDA"가 아니라 "드라이버가 지원하는 범위"다.

## PyTorch 바이너리와 CUDA

pip이나 uv로 설치하는 PyTorch는 미리 컴파일된 바이너리 패키지(wheel, 3.4장)다. CUDA용 wheel은 실행에 필요한 CUDA runtime, cuBLAS, cuDNN, NCCL 같은 라이브러리를 함께 가져온다. 집필 시점의 Linux용 wheel은 이 라이브러리들을 `nvidia-`로 시작하는 별도 Python 패키지로 의존성에 걸어 두는 방식을 쓴다. 정확한 구성은 버전마다 바뀌므로 자신의 환경에서 직접 확인한다(집필 시점 예시).

```bash
# PyTorch와 함께 설치된 NVIDIA 라이브러리 패키지 확인
pip list 2>/dev/null | grep -i nvidia
uv tree --package torch          # uv 프로젝트라면 의존성 트리로 확인

# PyTorch가 보는 CUDA·cuDNN·GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.backends.cudnn.version(), torch.cuda.get_arch_list())"
```

그래서 PyTorch로 모델을 학습하고 추론하는 대부분의 작업에서 시스템에 필요한 것은 **호환되는 드라이버뿐**이다. 시스템 CUDA toolkit이 필요한 경우는 따로 있다.

- **custom CUDA extension을 빌드할 때.** 직접 쓴 C++/CUDA 코드를 `torch.utils.cpp_extension`으로 컴파일하거나, 저장소의 `setup.py`가 CUDA 코드를 빌드하는 연구 코드를 설치할 때 `nvcc`가 필요하다. PyTorch의 빌드 도구는 `CUDA_HOME` 환경변수나 `PATH`에서 toolkit을 찾는다.
- **소스에서만 배포되는 패키지를 설치할 때.** 플랫폼에 맞는 wheel이 없으면 패키지 관리자가 소스 배포본(sdist)을 받아 그 자리에서 컴파일한다(3.4장). CUDA 코드가 들어 있으면 toolkit이 필요하다.
- **CUDA 프로그램 자체를 개발할 때.**

이때 toolkit의 CUDA major 버전은 PyTorch의 `torch.version.cuda`와 맞춰야 한다. PyTorch의 extension 빌드 도구는 두 버전을 비교해, major가 다르면 빌드를 멈추고 오류를 낸다. 컴파일할 대상 GPU 아키텍처는 `TORCH_CUDA_ARCH_LIST` 환경변수로 지정할 수 있다.

::quiz{id="torch-wheel-needs-toolkit"}

:::note[분야 예시]
분자·재료 그래프를 다루는 그래프 신경망 라이브러리 중에는 희소 연산을 custom CUDA 연산으로 구현한 선택 확장 패키지를 두는 경우가 있다. 이런 패키지는 PyTorch 버전과 CUDA 버전의 조합마다 미리 빌드한 wheel을 따로 배포하는 일이 많다. 맞는 조합의 wheel이 없으면 설치 도구가 소스 빌드로 넘어가고, 그때 toolkit과 컴파일러가 없으면 설치가 실패한다. 설치 문서에 "PyTorch와 CUDA 버전을 먼저 확인하라"는 단계가 있는 이유다.
:::

conda로 설치하는 PyTorch도 원리는 같다. CUDA runtime 라이브러리가 conda 패키지로 함께 설치되고, 시스템에는 드라이버만 있으면 된다. 어느 경로로 설치하든 드라이버만은 패키지 관리자가 가져올 수 없다. 드라이버는 kernel 모듈을 포함하기 때문이다.

## 컴파일러와 ABI

custom extension을 빌드하거나 소스에서 패키지를 설치하면, 스택의 또 다른 축인 **컴파일러와 ABI**가 문제를 일으킨다.

**ABI**(Application Binary Interface)는 이미 컴파일된 코드끼리 맞물리는 규칙이다. 함수 이름이 심볼로 바뀌는 방식(name mangling), 자료형의 메모리 배치, 함수 호출 규약이 여기에 속한다. 소스 코드 수준의 인터페이스(API)가 같아도 ABI가 다르면 컴파일된 두 조각을 연결할 수 없다. 이때 나타나는 대표 오류가 import 시점의 `undefined symbol`이다.

Linux에서 흔히 만나는 ABI 문제는 세 가지다.

- **C++ 표준 라이브러리의 이중 ABI.** GCC의 libstdc++는 `std::string` 등의 내부 구현이 다른 두 ABI를 가지며, 매크로 `_GLIBCXX_USE_CXX11_ABI`(0 또는 1)로 고른다. 두 ABI는 심볼 이름이 다르다. extension은 PyTorch가 빌드된 것과 같은 설정으로 컴파일해야 하고, PyTorch 쪽 설정은 `torch._C._GLIBCXX_USE_CXX11_ABI`로 확인한다. `torch.utils.cpp_extension`은 이 값을 자동으로 맞춰 준다. 직접 CMake로 빌드할 때 어긋나기 쉽다.
- **nvcc가 지원하는 host 컴파일러 범위.** `nvcc`는 CPU 쪽 코드를 시스템의 gcc로 컴파일하는데, CUDA 버전마다 지원하는 gcc 버전의 상한이 있다. OS를 업그레이드해 gcc가 새 버전이 되면 `unsupported GNU version` 오류로 빌드가 멈춘다. 지원 범위는 해당 CUDA 버전의 설치 가이드에 나와 있고, 지원되는 gcc를 따로 설치해 `nvcc -ccbin`이나 `CC`·`CXX` 환경변수로 지정해 해결한다.
- **glibc 버전.** 새 glibc에서 빌드한 바이너리는 오래된 glibc 시스템에서 `version 'GLIBC_2.xx' not found` 오류로 실행되지 않는다. 반대 방향은 대체로 동작한다. 그래서 배포용 wheel은 오래된 glibc 기준으로 빌드되고, 그 최소 요구치를 파일 이름의 platform tag에 적는다(3.4장).

자신의 컴퓨터에서 계층별 버전을 모으는 최소 코드는 다음과 같다. 외부 명령을 실행하므로 브라우저가 아니라 로컬 Python에서 실행한다. 출력은 집필에 쓴 환경(GPU가 없는 Linux 컨테이너)의 결과다.

```python {exec="python"}
import platform
import shutil
import subprocess

def first_line(cmd, last=False):
    if shutil.which(cmd[0]) is None:
        return f"{cmd[0]} 없음"
    lines = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
    return (lines[-1] if last else lines[0]) if lines else "(출력 없음)"

print("CPU 아키텍처 :", platform.machine())
print("kernel       :", platform.release())
print("C 라이브러리  :", " ".join(platform.libc_ver()))
print("드라이버      :", first_line(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]))
print("toolkit      :", first_line(["nvcc", "--version"], last=True))
print("gcc          :", first_line(["gcc", "--version"]))
print("Python       :", platform.python_version())
try:
    import torch
    print("PyTorch      :", torch.__version__, "/ CUDA", torch.version.cuda)
except ImportError:
    print("PyTorch      : 설치되지 않음")
```

```output
CPU 아키텍처 : x86_64
kernel       : 6.18.44-fc-v310
C 라이브러리  : glibc 2.39
드라이버      : nvidia-smi 없음
toolkit      : nvcc 없음
gcc          : gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Python       : 3.13.18
PyTorch      : 설치되지 않음
```

없는 층을 "없음"으로 기록하는 것도 정보다. 이 출력만 보고도 이 컴퓨터에서는 GPU 학습이 불가능하고(드라이버 없음), CUDA extension을 빌드할 수 없다(toolkit 없음)는 것을 알 수 있다. GPU 서버에서 실행하면 같은 코드가 GPU 모델과 드라이버 버전을 채운다.

:::caution[함정]
`ldd --version`과 `platform.libc_ver()`는 **현재 시스템**의 glibc를 보여 준다. 컨테이너 안에서 실행하면 호스트가 아니라 컨테이너 이미지의 glibc가 나온다. 반대로 드라이버의 kernel 모듈은 컨테이너 안에서도 호스트의 것이다(3.5장). 어느 층이 어디에서 오는지 구분하지 않으면 보고서가 사실과 달라진다.
:::

## 실습

자신이 쓰는 서버에서 각 계층의 버전을 수집하고 호환성을 판정한다. 코드는 `labs/environments/software-stack/collect_versions.py`다. 표준 라이브러리만 쓰고, 없는 도구는 `-`로 기록한 뒤 계속 진행한다. 수집하는 항목은 CPU 아키텍처, OS, kernel, glibc, 드라이버(kernel 모듈과 사용자 공간 각각), 드라이버의 최대 CUDA, GPU 이름과 compute capability, `nvcc`, gcc, Python, 그리고 PyTorch가 있으면 PyTorch·CUDA·cuDNN·C++ ABI·지원 아키텍처 목록이다. 수집 뒤에는 다음을 판정한다.

- kernel 모듈과 사용자 공간 드라이버의 버전이 같은가(드라이버를 업그레이드하고 재부팅하지 않으면 다를 수 있다)
- PyTorch의 CUDA가 드라이버가 지원하는 범위에 드는가
- `nvcc`가 있다면 그 major가 PyTorch의 CUDA major와 같은가

집필 환경(GPU 없음, PyTorch 없음)의 실행 결과는 다음과 같다.

```output
cpu_arch               x86_64
os                     Ubuntu 24.04.5 LTS
kernel                 6.18.44-fc-v310
glibc                  glibc 2.39
driver_kernel_module   -
driver                 -
driver_max_cuda        -
gpus                   -
toolkit_nvcc           -
gcc                    13.3.0
python                 3.13.18
torch                  -

[정보] CUDA toolkit(nvcc) 없음: 학습·추론에는 보통 필요 없고, extension 빌드에만 필요
```

GPU 서버에서 PyTorch가 설치된 환경으로 실행하고, 결과를 다음 세 질문으로 해석한다. ① 이 서버에서 CUDA 12.x용 PyTorch를 쓸 수 있는가, 13.x는 어떤가. ② custom CUDA extension을 빌드하려면 무엇을 더 설치해야 하는가. ③ 결과 중 컨테이너 안과 밖에서 달라질 항목은 무엇인가.

이어서 연습문제를 푼다. 세 가지 CUDA 버전을 출력에서 읽고, toolkit 없이도 판정이 되는지, 문자열 비교의 함정을 피하는지 테스트가 채점한다.

::exercise{id="read-cuda-versions"}

## 실패 사례

- **`nvidia-smi`의 `CUDA Version`을 설치된 toolkit 버전으로 오해하는 것.** 이 숫자는 드라이버가 지원하는 최대치다. "서버에 CUDA 12.4가 설치되어 있다"고 보고했는데 `nvcc`가 없거나 11.8인 일이 흔하다. 보고할 때는 세 값을 출처와 함께 따로 적는다.
- **PyTorch를 쓰기 위해 CUDA toolkit부터 무조건 설치하는 것.** wheel이 runtime을 가져오므로 학습·추론에는 필요 없다. 불필요한 toolkit은 `PATH`와 `LD_LIBRARY_PATH`에 또 하나의 CUDA를 더해, 어느 라이브러리가 로드되는지 헷갈리게 만든다. toolkit은 extension을 빌드할 때 필요한 버전으로 설치한다.
- **버전을 문자열로 비교하는 것.** `"12.10" < "12.4"`, `"525.105.17" < "525.60.13"`은 문자열 비교에서 모두 참이다. 스크립트에서 버전을 비교할 때는 정수 tuple로 바꾼다.
- **새 GPU에 오래된 PyTorch를 쓰는 것.** 바이너리에 그 GPU의 compute capability용 kernel이 없으면 `no kernel image is available` 오류가 난다. 드라이버를 아무리 바꿔도 해결되지 않는다. `torch.cuda.get_arch_list()`를 먼저 확인한다.

## 통과 기준

- 임의의 서버에서 `collect_versions.py`로 계층별 버전을 수집하고, PyTorch의 CUDA가 드라이버 범위에 드는지와 extension 빌드 가능 여부를 근거와 함께 판정한다.
- `nvidia-smi`, `nvcc --version`, `torch.version.cuda`가 보여 주는 세 값의 의미를 구분해 설명한다.
- 연습문제 `read-cuda-versions`의 테스트를 통과한다.
