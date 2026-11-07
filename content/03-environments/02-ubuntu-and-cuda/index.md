---
id: environments/ubuntu-and-cuda
title: "Ubuntu·드라이버·CUDA 설치와 버전 전략"
title_en: "Ubuntu, Drivers, and CUDA Installation"
tier: core
status: draft
prerequisites: [environments/software-stack]
---

# Ubuntu·드라이버·CUDA 설치와 버전 전략

GPU 서버의 아래쪽 세 층, 즉 OS·드라이버·CUDA toolkit은 한 번 설치하면 서버의 모든 사용자가 몇 년 동안 그 위에서 일한다. 이 층의 선택이 잘못되면 위쪽 층에서 아무리 환경을 잘 관리해도 소용이 없다. 드라이버가 오래되면 새 PyTorch를 쓸 수 없고, OS 지원이 끝나면 보안 업데이트가 끊기며, 기록 없이 업그레이드한 드라이버는 어제까지 돌던 실험을 오늘 멈추게 한다.

이 장은 설치 명령을 외우는 장이 아니다. 설치 명령과 버전 번호는 몇 달이면 낡는다. 대신 **무엇을 기준으로 버전을 고르고, 어떤 공식 문서로 호환성을 확인하고, 무엇을 기록해 두는지**를 다룬다. 이 장의 명령은 모두 집필 시점(Ubuntu 기준)의 예시이며, 실행 전에 공식 문서로 확인하는 것을 전제로 한다.

## 학습 목표

- OS와 드라이버 버전을 선택하는 기준을 안다
- 공식 호환성 문서를 확인하는 절차를 따른다
- 여러 CUDA 버전을 공존시킨다

## OS 버전 선택

연구용 GPU 서버의 OS로는 Ubuntu가 흔히 쓰인다. NVIDIA 드라이버와 CUDA의 지원 목록에 늘 포함되고, 연구 코드의 설치 문서도 대개 Ubuntu를 기준으로 쓰이기 때문이다. 다른 배포판(Rocky Linux, Debian 등)도 원리는 같다.

Ubuntu는 두 종류의 릴리스를 낸다.

- **LTS**(Long Term Support): 2년마다(짝수 해 4월) 나오는 릴리스. 버전 이름은 `연도.월`이다(예: 22.04, 24.04). 표준 보안 지원 기간이 5년이다.
- **중간 릴리스**: 그 사이 6개월마다 나오며 지원 기간이 9개월 정도로 짧다.

공용 서버는 LTS를 쓴다. 중간 릴리스는 금방 지원이 끝나고, CUDA의 지원 OS 목록에 늦게 오르거나 오르지 않을 수 있다. 어떤 LTS를 고를지는 **프로젝트 기간과 지원 기간의 겹침**으로 판단한다. 2026년 11월에 시작해 2029년 말까지 쓸 서버를 생각한다. 표준 지원 종료를 "발표 연월 + 5년"으로 근사해 계산한다(정확한 종료일은 Ubuntu 공식 릴리스 주기 페이지에서 확인한다).

```python {exec="browser" session="cuda"}
from datetime import date

lts_release = {"20.04": date(2020, 4, 1), "22.04": date(2022, 4, 1), "24.04": date(2024, 4, 1)}
project_start, project_end = date(2026, 11, 1), date(2029, 12, 31)

for name, released in lts_release.items():
    eol = released.replace(year=released.year + 5)      # 표준 지원 약 5년(근사)
    if eol <= project_start:
        status = "이미 표준 지원 종료"
    elif eol < project_end:
        months = (eol.year - project_start.year) * 12 + eol.month - project_start.month
        status = f"프로젝트 시작 후 {months}개월에 종료"
    else:
        status = "프로젝트 기간 전체를 덮음"
    print(f"Ubuntu {name}: 표준 지원 ~{eol:%Y-%m}  -> {status}")
```

```output
Ubuntu 20.04: 표준 지원 ~2025-04  -> 이미 표준 지원 종료
Ubuntu 22.04: 표준 지원 ~2027-04  -> 프로젝트 시작 후 5개월에 종료
Ubuntu 24.04: 표준 지원 ~2029-04  -> 프로젝트 시작 후 29개월에 종료
```

세 LTS 중 어느 것도 프로젝트 기간 전체를 덮지 못한다. 이런 계산을 하면 선택지가 분명해진다. 표에 없는 더 새로운 LTS가 이미 나왔고 쓰려는 CUDA가 그것을 지원한다면 그것을 고르고, 그렇지 않다면 24.04를 고르되 **기간 중 OS 업그레이드 시점을 미리 계획에 넣는다.** 확장 보안 지원 같은 유료·추가 지원 프로그램을 쓸 수도 있지만, 그 경우에도 새 CUDA가 그 OS를 계속 지원하는지는 따로 확인해야 한다.

연구실 표준을 정할 때는 OS 하나만이 아니라 **조합**을 정한다. "Ubuntu LTS 버전, 드라이버 branch, 기본 CUDA toolkit 버전, 업그레이드 주기" 네 가지를 한 문서에 적고, 서버마다 다르게 가지 않게 한다. 같은 연구실의 서버들이 같은 조합이면, 한 서버에서 검증한 환경을 다른 서버로 그대로 옮길 수 있다.

## 드라이버 설치

드라이버는 스택에서 유일하게 **kernel에 들어가는** 층이다(3.1장). 그래서 설치 방식, Secure Boot, kernel 업데이트가 모두 드라이버와 얽힌다.

**설치 방식 세 가지.**

| 방식 | 장점 | 단점 |
|---|---|---|
| 배포판 패키지(Ubuntu 저장소) | 패키지 관리자가 kernel 업데이트와 의존성을 관리한다. Ubuntu의 `ubuntu-drivers` 도구가 권장 드라이버를 고른다 | 최신 드라이버가 늦게 들어온다 |
| NVIDIA 저장소(apt 저장소 추가) | 최신 드라이버와 CUDA toolkit을 같은 저장소에서 받는다 | 메타패키지 선택을 잘못하면 드라이버가 의도치 않게 바뀐다 |
| runfile(`.run` 설치 파일) | 배포판과 무관하게 특정 버전을 설치한다 | 패키지 관리자 밖에서 설치되어, 업데이트·제거·kernel 변경 시 직접 관리해야 한다 |

공용 서버에는 패키지 관리자가 추적하는 방식(앞의 두 가지)을 권한다. 누가 무엇을 설치했는지 `dpkg`와 apt 기록에 남고, 고정(hold)과 rollback도 패키지 단위로 할 수 있다. runfile은 패키지 관리자 바깥에 파일을 흩어 두어, 나중에 패키지 방식과 섞이면 충돌한다.

배포판 패키지 방식의 확인 절차는 다음과 같다(집필 시점 예시. 명령과 패키지 이름은 Ubuntu 문서에서 확인한다).

```bash
# 이 GPU에 맞는 드라이버 후보와 권장 항목 보기
sudo ubuntu-drivers list
# 설치 전 상태 기록 (아래 '설치 기록' 절)
uname -r; dpkg -l | grep -E 'nvidia|cuda' > before-dpkg.txt
# 권장 드라이버 설치 후 재부팅
sudo ubuntu-drivers install
sudo reboot
# 확인
nvidia-smi
cat /proc/driver/nvidia/version
```

**Secure Boot와 kernel 모듈.** UEFI Secure Boot가 켜진 시스템은 서명되지 않은 kernel 모듈을 거부한다. 드라이버 모듈을 설치 시점에 빌드하는 방식이면, 설치 과정에서 MOK(Machine Owner Key)를 등록하라는 안내와 비밀번호 설정이 나오고, 재부팅할 때 펌웨어 화면에서 그 키를 등록해야 모듈이 로드된다. 원격 서버에서 이 화면을 놓치면 드라이버가 설치되었는데도 `nvidia-smi`가 실패한다. 상태는 `mokutil --sb-state`로 확인한다. 서버를 처음 설치할 때 Secure Boot 정책(끌지, 키를 등록할지)을 정해 기록해 둔다.

**kernel 업데이트 후 드라이버가 깨지는 이유.** kernel 모듈은 특정 kernel 버전에 맞춰 빌드된다. 배포판 패키지 방식은 보통 DKMS(Dynamic Kernel Module Support)가 새 kernel이 설치될 때 모듈을 다시 빌드한다. 다음 경우에 이 과정이 실패한다.

- 새 kernel의 헤더 패키지(`linux-headers-...`)가 설치되지 않았다.
- 새 kernel이 드라이버 버전보다 새로워 모듈 빌드가 실패한다.
- runfile로 설치한 드라이버라 DKMS에 등록되어 있지 않다.

그러면 재부팅 후 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`가 나온다. 복구 순서는 다음과 같다.

1. `uname -r`로 현재 kernel을, `dkms status`로 모듈이 어느 kernel에 빌드되었는지 확인한다.
2. 헤더가 없으면 설치하고 모듈을 다시 빌드한다.
3. 급하면 부트 메뉴(GRUB)에서 이전 kernel로 부팅해 서비스를 살린 뒤 원인을 고친다.

반대 상황도 있다. 자동 업데이트가 드라이버 패키지를 올렸는데 재부팅하지 않으면, 메모리의 kernel 모듈은 옛 버전이고 사용자 공간 라이브러리는 새 버전이라 `Failed to initialize NVML: Driver/library version mismatch`가 나온다. 재부팅하면 해결된다. 공유 서버에서는 드라이버 패키지를 `apt-mark hold`로 고정해, 계획한 시점에만 업그레이드한다.

::quiz{id="nvml-mismatch"}

:::note[분야 예시]
천문·입자물리 실험의 데이터 처리 서버는 관측·실험 기간 내내 같은 환경을 유지해야 한다. 관측 도중 드라이버가 바뀌어 처리 결과가 달라지면, 그 기간의 데이터를 다시 처리해야 할 수도 있다. 이런 서버에서는 드라이버·kernel 패키지를 고정하고, 업그레이드는 관측 사이 정비 기간에만, 검증 절차를 거쳐 수행하는 운영 규칙을 둔다.
:::

## CUDA toolkit 설치

3.1장에서 본 것처럼 PyTorch로 학습·추론만 한다면 시스템 CUDA toolkit은 필요 없다. toolkit이 필요한 것은 custom CUDA extension을 빌드하거나, CUDA 코드를 소스에서 컴파일하는 패키지를 설치하거나, CUDA 프로그램을 직접 개발할 때다. 필요 없는 toolkit을 설치하지 않는 것이 첫 번째 원칙이다.

필요하다면 **버전별 디렉터리에 설치해 공존**시킨다. NVIDIA 저장소의 패키지와 runfile은 toolkit을 `/usr/local/cuda-12.4`처럼 버전이 붙은 디렉터리에 설치하고, `/usr/local/cuda`를 그중 하나를 가리키는 심볼릭 링크로 둔다. 드라이버를 건드리지 않으려면 드라이버까지 끌어오는 메타패키지(`cuda` 등) 대신 버전이 붙은 toolkit 패키지(예: `cuda-toolkit-12-4`)를 고른다(패키지 이름은 집필 시점 예시).

어느 toolkit을 쓸지는 환경변수가 정한다.

| 변수 | 쓰는 쪽 | 역할 |
|---|---|---|
| `PATH` | 셸 | 어느 `nvcc`가 실행되는가 |
| `LD_LIBRARY_PATH` | 동적 링커 | 실행 시 어느 디렉터리의 `.so` 라이브러리를 먼저 찾는가 |
| `CUDA_HOME` | PyTorch extension 빌드 도구 등 | 헤더와 라이브러리를 찾을 toolkit의 루트 |

`PATH`와 `LD_LIBRARY_PATH`는 모두 **앞에서부터 찾아 처음 찾은 것**을 쓴다. 여러 CUDA가 공존하는 서버에서 "어느 것이 쓰이는가"는 이 규칙 하나로 결정된다. 심볼릭 링크 `/usr/local/cuda`가 무엇을 가리키는지도 함께 따져야 한다.

```python {exec="browser" session="cuda"}
files = {"/usr/local/cuda-12.1/bin/nvcc", "/usr/local/cuda-12.4/bin/nvcc", "/usr/bin/gcc"}
links = {"/usr/local/cuda": "/usr/local/cuda-12.4"}    # 심볼릭 링크

def realpath(p):                                       # readlink -f 흉내
    for src, dst in links.items():
        if p.startswith(src + "/"):
            return dst + p[len(src):]
    return p

def which(name, path_var):
    for d in path_var.split(":"):                      # 앞에서부터, 처음 찾은 것
        candidate = f"{d.rstrip('/')}/{name}"
        if realpath(candidate) in files:
            return candidate
    return None

for path_var in ["/usr/local/cuda/bin:/usr/bin",
                 "/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/usr/bin",
                 "/usr/bin"]:
    found = which("nvcc", path_var)
    print(f"PATH={path_var}\n  -> {found}  (실제 파일: {realpath(found) if found else '-'})")
```

```output
PATH=/usr/local/cuda/bin:/usr/bin
  -> /usr/local/cuda/bin/nvcc  (실제 파일: /usr/local/cuda-12.4/bin/nvcc)
PATH=/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/usr/bin
  -> /usr/local/cuda-12.1/bin/nvcc  (실제 파일: /usr/local/cuda-12.1/bin/nvcc)
PATH=/usr/bin
  -> None  (실제 파일: -)
```

::quiz{id="which-nvcc-runs"}

실제 서버에서 같은 확인은 `which nvcc`와 `readlink -f "$(which nvcc)"`로 한다. 버전을 고르는 방법은 세 가지가 흔하다.

- **프로젝트 스크립트에서 명시.** 빌드 스크립트나 셸 설정에서 `CUDA_HOME`과 `PATH`를 한 번에 바꾼다. 무엇을 썼는지 스크립트에 남는다.

  ```bash
  # 이 셸에서만 CUDA 12.1을 쓴다 (집필 시점 예시 경로)
  export CUDA_HOME=/usr/local/cuda-12.1
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  which nvcc && nvcc --version | tail -n 1
  ```

- **모듈 시스템.** HPC 클러스터처럼 Environment Modules·Lmod가 있으면 `module load`가 위 변수를 바꾼다.
- **conda 환경 안의 toolkit.** conda 채널의 CUDA 컴파일러 패키지를 환경마다 설치하면, 시스템 toolkit 없이 환경별로 다른 `nvcc`를 쓸 수 있다(3.3장).

`/usr/local/cuda` 링크를 바꾸는 방법은 서버 전체 사용자에게 영향을 주므로 공용 서버에서는 피한다. 그리고 `LD_LIBRARY_PATH`는 꼭 필요할 때만 쓴다. `~/.bashrc`에 시스템 toolkit의 `lib64`를 넣어 두면, pip으로 설치한 PyTorch가 자기 라이브러리 대신 시스템 라이브러리를 집어 오는 일이 생길 수 있다. 바이너리에 실행 경로(RPATH·RUNPATH)가 박혀 있는 경우 검색 순서가 더 복잡해지므로, 실제로 무엇이 로드되었는지는 `ldd`로 확인한다(3.6장).

## 호환성 확인 절차

버전 번호를 외우는 대신, 매번 같은 순서로 **공식 문서의 표**를 확인한다. 위층의 요구 조건에서 시작해 아래로 내려간다.

1. **프레임워크가 제공하는 CUDA 빌드를 확인한다.** 쓰려는 PyTorch 버전의 공식 설치 페이지에서 어떤 CUDA 버전용 빌드가 제공되는지 본다. 이것이 필요한 CUDA runtime 버전을 정한다.
2. **그 CUDA에 필요한 최소 드라이버를 확인한다.** CUDA toolkit release notes에는 toolkit 버전마다 필요한 최소 드라이버 버전을 정리한 표가 있다.
3. **GPU가 그 드라이버 branch에서 지원되는지 확인한다.** NVIDIA 드라이버 다운로드 페이지나 드라이버 release notes의 지원 GPU 목록을 본다. 아주 오래된 GPU는 새 드라이버 branch에서 지원이 끝나고, 아주 새로운 GPU는 오래된 branch에서 지원되지 않는다.
4. **OS·kernel·컴파일러가 지원되는지 확인한다.** CUDA 설치 가이드(Linux)에 지원 배포판, kernel, gcc 버전의 표가 있다. extension을 빌드할 계획이면 gcc 상한이 특히 중요하다(3.1장).
5. **확인한 문서와 날짜를 기록한다.** 같은 질문을 1년 뒤에 다시 받을 때, 무엇을 근거로 정했는지가 남아 있어야 한다.

2단계에서 두 버전을 비교할 때 흔한 실수가 문자열 비교다. 집필 시점 CUDA toolkit release notes의 표에서 세 항목을 가져와(CUDA 12.0 GA는 525.60.13 이상, 12.6 Update 3은 560.35.05 이상, 12.8 Update 1은 570.124.06 이상), 설치된 드라이버 525.105.17과 비교한다.

```python {exec="browser" session="cuda"}
min_driver = {"12.0 GA": "525.60.13", "12.6 Update 3": "560.35.05", "12.8 Update 1": "570.124.06"}
installed = "525.105.17"

def vtuple(s):
    return tuple(int(p) for p in s.split("."))   # "525.105.17" -> (525, 105, 17)

for cuda, need in min_driver.items():
    ok_str = installed >= need                       # 잘못된 방법
    ok_num = vtuple(installed) >= vtuple(need)      # 올바른 방법
    print(f"CUDA {cuda:13s} 최소 {need:>10s}  문자열 비교: {ok_str!s:5s}  숫자 비교: {ok_num}")
```

```output
CUDA 12.0 GA       최소  525.60.13  문자열 비교: False  숫자 비교: True
CUDA 12.6 Update 3 최소  560.35.05  문자열 비교: False  숫자 비교: False
CUDA 12.8 Update 1 최소 570.124.06  문자열 비교: False  숫자 비교: False
```

첫 줄에서 두 방법의 결론이 갈린다. 525.105.17은 525.60.13보다 새 버전이지만, 문자열 비교는 `'1' < '6'`이라 거꾸로 판단한다. 설치 스크립트나 점검 스크립트가 이렇게 쓰여 있으면, 멀쩡한 서버에서 드라이버 업그레이드를 요구하거나 그 반대가 된다.

::quiz{id="driver-version-compare"}

이 표에서 하나 더 읽을 것이 있다. toolkit 표의 최소 드라이버는 **그 toolkit에 포함된 모든 기능**을 쓰기 위한 값이다. 3.1장의 minor version compatibility 덕분에 같은 major 안에서는 더 낮은 드라이버로도 대부분 실행되지만, 그것은 예외 규칙이다. 연구실 표준을 정할 때는 기본 규칙(표의 최소 드라이버 이상)을 만족하도록 고른다.

## 설치 기록

공용 서버의 드라이버와 toolkit은 **운영 일지**(server log)에 남긴다. 몇 달 뒤 "언제부터 이 오류가 났지?"라는 질문에 답할 수 있는 유일한 방법이다. 한 항목에 들어갈 내용은 다음과 같다.

| 항목 | 예 |
|---|---|
| 날짜·작업자 | 2026-11-07, 담당자 이름 |
| 목적 | CUDA 12.8용 PyTorch를 쓰기 위해 드라이버 업그레이드 |
| 근거 | 확인한 공식 문서와 확인 날짜 |
| 변경 전 상태 | `uname -r`, 드라이버 버전, `dpkg -l` 중 nvidia·cuda 패키지 목록, hold 목록 |
| 실행한 명령 | 복사해 붙인 명령 그대로 |
| 변경 후 상태와 검증 | `nvidia-smi`, 3.1장의 수집 스크립트 출력, smoke test 결과 |
| rollback 방법 | 이전 패키지 버전, 이전 kernel로 부팅하는 법 |

변경 전후 상태는 손으로 옮겨 적지 말고 명령 출력을 파일로 남긴다(집필 시점 예시).

```bash
LOG=/srv/admin/log/$(date +%F)-driver-upgrade   # 연구실이 정한 일지 위치
mkdir -p "$LOG"
uname -r                                    > "$LOG/before-kernel.txt"
nvidia-smi                                  > "$LOG/before-nvidia-smi.txt" 2>&1
dpkg -l | grep -E 'nvidia|cuda'             > "$LOG/before-dpkg.txt"
apt-mark showhold                           > "$LOG/before-hold.txt"
# ... 업그레이드 ...
# apt 자체의 기록도 함께 남는다: /var/log/apt/history.log
```

일지는 공유 위치나 연구실 저장소에 Markdown으로 쌓는다. 이 기록이 있어야 3.6장의 변경 관리와 rollback이 가능하다.

:::tip[연구 팁]
드라이버 업그레이드는 "서버의 모든 사용자가 동시에 실험을 멈추는 일"이다. 미리 공지하고, 진행 중인 job이 없는 시간에 하고, 업그레이드 직후 각 사용자가 자신의 smoke test를 돌려 확인하는 시간을 둔다.
:::

## 실습

테스트용 VM이나 컨테이너에서 연구실 표준 조합을 공식 문서로 검증하고 설치 절차서를 작성한다. 드라이버 설치 자체는 GPU가 연결된 VM이나 실제 서버에서만 확인할 수 있지만, 조합의 정합성 검사와 toolkit 공존 설정은 GPU 없이도 연습할 수 있다.

1. `labs/environments/ubuntu-and-cuda/combo.example.toml`을 복사해 연구실의 조합(OS, GPU, 드라이버, toolkit, gcc, PyTorch의 CUDA)을 채운다. 항목마다 근거 문서(`source`)와 확인 날짜(`checked`)를 적는다.
2. `check_combo.py`로 조합을 검사한다. 이 스크립트는 기록 안의 값끼리 맞는지(드라이버 ≥ toolkit의 최소 드라이버, PyTorch CUDA ≤ 드라이버 최대 CUDA, toolkit과 PyTorch의 CUDA major 일치, gcc 상한, OS 지원 기간)와 근거가 비어 있지 않은지를 본다. 값이 공식 문서와 같은지는 사람이 확인한다.
3. 통과한 조합으로 설치·복구 절차서를 Markdown으로 쓴다. 설치 순서, 확인 명령, kernel 업데이트 후 복구, 드라이버 rollback을 포함한다.

예시 파일을 그대로 검사한 결과는 다음과 같다.

```output
[확인 필요] [framework] 근거 없음: source와 checked(확인 날짜)를 채운다
[실패] OS 표준 지원(2029-04)이 프로젝트 종료(2029-12)까지 이어지는가
[통과] 드라이버 570.124.06 >= toolkit 12.8의 최소 드라이버 570.124.06
[통과] PyTorch CUDA 12.8 <= 드라이버 최대 CUDA 12.8
[통과] toolkit 12.8과 PyTorch CUDA 12.8의 major가 같은가(extension 빌드)
[확인 필요] nvcc가 지원하는 gcc 상한이 비어 있다: CUDA 설치 가이드에서 확인한다

통과 3 · 실패 1 · 확인 필요 2
```

"실패"는 OS 지원이 프로젝트보다 먼저 끝난다는 뜻이고, 절차서에 OS 업그레이드 계획을 넣거나 다른 LTS를 고르라는 신호다. "확인 필요"는 아직 공식 문서로 확인하지 않은 값이다. 모든 항목이 통과하거나 근거 있는 예외로 정리될 때까지 절차서를 확정하지 않는다.

이어서 연습문제를 푼다. `PATH`·`LD_LIBRARY_PATH` 규칙으로 실제로 실행·로드될 파일을 찾는 함수를 작성한다.

::exercise{id="path-lookup"}

## 실패 사례

- **블로그의 오래된 설치 명령을 그대로 실행하는 것.** 저장소 주소, 서명 키 설치 방식, 패키지 이름은 몇 년 사이에 바뀐다. 오래된 명령은 실패하면 다행이고, 성공하면 의도하지 않은 드라이버나 저장소를 시스템에 남긴다. 명령은 공식 설치 가이드의 현재 판에서 가져오고, 가져온 날짜를 기록한다.
- **운영 중인 공유 서버에서 드라이버를 기록 없이 업그레이드하는 것.** 실행 중인 학습이 죽거나, 재부팅 전까지 `Driver/library version mismatch` 상태가 되거나, 다른 사용자의 오래된 환경이 동작하지 않게 된다. 무엇이 바뀌었는지 기록이 없으면 원인을 찾는 데 며칠이 걸린다.
- **PyTorch 하나 쓰려고 `cuda` 메타패키지를 설치하는 것.** toolkit이 필요 없었는데 드라이버까지 바뀐다.
- **`~/.bashrc`에 특정 CUDA의 `LD_LIBRARY_PATH`를 영구히 넣는 것.** 그 셸에서 실행하는 모든 프로그램의 라이브러리 검색이 바뀐다. 다른 버전의 CUDA를 쓰는 환경에서 원인 모를 오류가 난다.
- **드라이버 버전을 문자열로 비교하는 점검 스크립트.** 525.105.17을 525.60.13보다 낮다고 판단한다.

## 통과 기준

- 연구실 표준 OS·드라이버·CUDA 조합을 정하고, 각 값의 근거 문서와 확인 날짜를 기록한 `combo.toml`이 `check_combo.py`에서 "실패" 없이 통과한다(남은 "확인 필요"는 이유를 적는다).
- 설치 절차와 복구 절차(kernel 업데이트 후 드라이버 복구, 드라이버 rollback)를 문서로 작성한다.
- 연습문제 `path-lookup`의 테스트를 통과한다.
