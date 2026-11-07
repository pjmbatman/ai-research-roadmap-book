---
id: linux-and-remote/processes-and-monitoring
title: "프로세스·세션·자원 모니터링"
title_en: "Processes, Sessions, and Resource Monitoring"
tier: core
status: draft
prerequisites: [linux-and-remote/linux-shell]
---

# 프로세스·세션·자원 모니터링

밤새 돌린 학습이 아침에 죽어 있다. 로그 마지막 줄은 평범한 epoch 기록이고 traceback도 없다. 이럴 때 "다시 돌려 보자"밖에 할 수 없다면, 같은 일이 다시 일어날 때까지 기다리는 것과 같다. 학습이 죽거나 느려지는 원인은 대부분 코드가 아니라 **자원**에 있다. 메모리가 바닥났거나, 디스크가 찼거나, 접속이 끊기며 프로세스가 함께 죽었거나, GPU가 데이터를 기다리며 놀고 있었던 것이다.

이 장은 실행 중인 프로그램을 운영체제가 어떻게 보는지(프로세스와 시그널)에서 출발해, CPU·메모리·디스크·GPU를 관찰하는 도구, 접속이 끊겨도 살아남는 작업 세션, 그리고 흔한 장애의 흔적을 읽는 법을 다룬다. 명령과 옵션은 일반적인 Linux 서버와 NVIDIA 드라이버를 가정한 **집필 시점 예시**다. 정확한 옵션은 각 서버에서 `man <명령>`, `<명령> --help`로 확인한다.

## 학습 목표

- 프로세스와 시그널을 다룬다
- CPU·메모리·디스크·GPU 자원을 모니터링한다
- 끊기지 않는 작업 세션을 운영한다

## 프로세스와 시그널

**프로세스.** 실행 중인 프로그램 하나가 프로세스(process)다. 각 프로세스는 고유 번호 PID, 자신을 만든 부모의 번호 PPID, 소유 사용자, 상태(실행 중 R, 대기 S, 디스크 I/O 대기 D, 종료됐지만 회수되지 않은 Z 등)를 갖는다. `python train.py --num-workers 8`처럼 DataLoader worker를 쓰는 학습은 프로세스 하나가 아니라 부모 하나와 자식 여럿으로 이루어진 **프로세스 나무**다. 이 점이 뒤의 "유령 프로세스"를 이해하는 열쇠다.

**foreground와 background.** 셸에서 명령을 그냥 실행하면 foreground에서 돌아 끝날 때까지 프롬프트가 돌아오지 않는다. 끝에 `&`를 붙이면 background에서 돈다. 실행 중인 foreground 작업은 `Ctrl-Z`로 멈춘 뒤 `bg`로 background에 보내고, `jobs`로 목록을 보고, `fg`로 다시 불러온다. 다만 background 작업도 그 셸(터미널)에 묶여 있다는 점은 같다.

**시그널.** 시그널(signal)은 운영체제가 프로세스에 보내는 짧은 통지다. 연구에서 알아야 할 것은 다음 넷이다(번호는 `kill -l`로 확인한다).

| 시그널 | 번호 | 언제 오는가 | 프로그램이 잡을 수 있나 | 기본 동작 |
|---|---|---|---|---|
| SIGHUP | 1 | 터미널(SSH 연결)이 끊길 때 | 예 | 종료 |
| SIGINT | 2 | `Ctrl-C` | 예 (Python은 `KeyboardInterrupt`로) | 종료 |
| SIGTERM | 15 | `kill <PID>`의 기본값, scheduler의 작업 취소·선점 | 예 | 종료 |
| SIGKILL | 9 | `kill -9 <PID>`, 커널의 OOM killer | **아니오** | 즉시 종료 |

핵심은 마지막 열이다. SIGTERM은 "정리하고 끝내라"는 요청이라, 프로그램이 handler를 달아 checkpoint를 저장할 수 있다. SIGKILL은 커널이 프로세스를 그 자리에서 없애므로 어떤 정리도 일어나지 않는다. 집필 시점에 Python 3.13으로 확인한 동작은 다음과 같다.

- handler 없이 SIGINT를 받으면 `KeyboardInterrupt`가 발생해 `finally` 블록이 실행된다.
- handler 없이 SIGTERM을 받으면 Python은 그대로 종료되고 `finally`도 실행되지 않는다.
- `signal.signal(signal.SIGTERM, handler)`로 handler를 달면, SIGTERM에 handler와 `finally`가 모두 실행된다.
- SIGKILL에는 handler가 있어도 아무것도 실행되지 않는다.

```python
# 집필 시점 예시: SIGTERM을 받으면 checkpoint를 저장하고 끝내는 학습 루프의 뼈대
import signal, sys

stop = False
def on_term(signum, frame):
    global stop
    stop = True                      # 바로 죽지 않고 "다음 step이 끝나면 멈춤" 표시만 한다

signal.signal(signal.SIGTERM, on_term)
for step in range(start_step, total_steps):
    train_one_step()
    if stop:
        save_checkpoint(step)
        sys.exit(0)
```

handler 안에서 바로 저장하지 않고 표시만 하는 이유는, step 도중의 어중간한 상태를 저장하지 않기 위해서다. 이 패턴은 2.5장의 scheduler 선점(preemption)에서 다시 쓴다.

**멈추는 순서.** 그래서 프로세스를 멈출 때는 `kill <PID>`(SIGTERM)를 먼저 보내고, 정해 둔 시간이 지나도 끝나지 않을 때만 `kill -9`를 쓴다. 셸에서 시그널로 죽은 명령의 종료 상태는 128 + 시그널 번호로 보고된다. 집필 시점에 bash에서 `sleep`에 SIGKILL을 보내면 137, SIGTERM을 보내면 143이 나왔다. traceback 없이 137만 남은 학습은 SIGKILL을 받은 것이고, 보낸 쪽이 사람이 아니라면 대개 OOM killer다.

::quiz{id="sigterm-before-sigkill"}

**nohup.** `nohup 명령 &`은 그 명령이 SIGHUP을 무시하게 해서, 터미널이 끊겨도 살아남게 한다. stdout이 터미널이면 출력은 `nohup.out`으로 간다. 그러나 다시 그 터미널 화면으로 돌아갈 방법은 없다. 이 차이는 뒤의 "작업 세션 유지"에서 tmux와 비교한다.

## 자원 모니터링 도구

학습이 느리거나 죽을 때 확인할 자원은 넷이다. CPU, 메모리(RAM), 디스크(용량과 속도), GPU. 각각을 보는 도구는 다음과 같다.

| 자원 | 명령 (집필 시점 예시) | 무엇을 보는가 |
|---|---|---|
| 프로세스 목록 | `ps -eo pid,ppid,user,stat,rss,etime,comm --sort=-rss \| head` | 메모리(RSS)를 많이 쓰는 순, 경과 시간, 상태 |
| CPU·메모리 실시간 | `top`, `htop`, `btop` (설치되어 있다면) | 프로세스별 %CPU, %MEM. `top` 머리의 `wa`는 I/O 대기 비율 |
| 메모리 전체 | `free -h` | `available`이 실제로 새 프로그램이 쓸 수 있는 양에 가깝다 |
| 디스크 용량 | `df -h`, `df -i` | 파일시스템별 사용률, inode(파일 개수) 사용률 |
| 디렉터리 크기 | `du -sh * \| sort -h` | 어디가 공간을 먹는가 |
| 디스크 I/O | `iostat -x 1` (sysstat 패키지) | 장치별 사용률, 대기 시간 |
| 열린 파일 | `lsof -p <PID>`, `lsof +L1` | 프로세스가 연 파일, 지워졌지만 열려 있는 파일 |

`free`의 `free` 열이 작다고 놀랄 필요는 없다. Linux는 남는 메모리를 파일 캐시(buff/cache)로 쓰고, 필요하면 돌려준다. 메모리 부족을 판단할 기준은 `available`이다. `df -i`는 작은 파일 수백만 개로 이루어진 데이터셋을 풀 때 용량은 남았는데 inode가 바닥나 `No space left on device`가 나는 경우를 잡는다.

**출력을 파싱하는 습관.** 이 도구들의 출력은 사람이 보기 위한 것이지만, 대부분 열 형식을 고를 수 있어 그대로 표로 읽을 수 있다. 아래는 `ps -eo ... --sort=-rss` 형식을 흉내 낸 가상의 출력에서 사용자별 메모리 합계, zombie, 부모가 1인 프로세스를 찾는 예다.

```python {exec="browser" session="monitor"}
import io

import pandas as pd

# ps -eo pid,ppid,user,stat,rss,etime,comm --sort=-rss 형식을 흉내 낸 가상의 출력 (RSS 단위 KiB)
ps_txt = """  PID  PPID USER STAT     RSS     ELAPSED COMMAND
 4121     1 kim  Sl   9437184  2-03:11:05 python
 5310  5001 lee  Rl   7340032    05:12:40 python
 4188  4121 kim  S    3145728  2-03:10:58 python
 4190     1 kim  S    3145728  6-21:40:02 python
 6001  5990 park Z          0       00:10 python"""
ps = pd.read_csv(io.StringIO(ps_txt), sep=r"\s+")
ps["rss_gib"] = ps["RSS"] / 1024**2                     # KiB → GiB
print(ps.groupby("USER")["rss_gib"].sum().round(1).to_string())
print("zombie(Z):", ps.loc[ps["STAT"].str.startswith("Z"), "PID"].tolist())
print("PPID = 1:", ps.loc[ps["PPID"] == 1, ["PID", "ELAPSED"]].values.tolist())
```

```output
USER
kim     15.0
lee      7.0
park     0.0
zombie(Z): [6001]
PPID = 1: [[4121, '2-03:11:05'], [4190, '6-21:40:02']]
```

부모가 1(init, 또는 그 역할을 맡은 프로세스)인 프로세스는 원래 부모가 먼저 끝나 "입양된" 것이다. 4121은 nohup으로 띄우고 셸을 닫은 정상 학습일 수 있고, 4190은 6일 전에 죽은 실험이 남긴 worker일 수 있다. PPID만으로는 구분되지 않는다. 경과 시간(ELAPSED)과 `ps -o cmd -p <PID>`의 전체 명령, 그리고 그 프로세스가 지금도 무언가를 하고 있는지(%CPU, 로그 갱신 시각)를 함께 본다.

## GPU 모니터링

**nvidia-smi 출력 읽기.** NVIDIA GPU가 있는 서버에서 `nvidia-smi`를 실행하면 GPU마다 다음이 표시된다.

- **GPU-Util**: 사용률(%). 뒤에서 정의를 정확히 본다.
- **Memory-Usage**: 사용 중인 VRAM / 전체 VRAM (MiB).
- **Temp**: 온도(°C). 냉각이 부족하면 클럭을 낮춰(throttling) 느려진다.
- **Pwr:Usage/Cap**: 현재 전력 / 전력 상한(W). 연산이 많을수록 상한에 가까워진다.
- **Processes**: GPU를 쓰는 프로세스의 PID와 각자의 메모리 사용량.

사람이 읽는 표 대신 필요한 값만 CSV로 뽑을 수 있다. 사용할 수 있는 필드 이름은 `nvidia-smi --help-query-gpu`로 확인한다.

```bash
# 집필 시점 예시
nvidia-smi                                             # 표 형식 한 번
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
           --format=csv,noheader,nounits               # 기계가 읽기 좋은 CSV
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # GPU를 쓰는 프로세스
nvidia-smi dmon                                        # 1초마다 한 줄씩 사용률·전력·온도 등
watch -n 1 nvidia-smi                                  # 1초마다 화면 갱신
```

`nvtop`, `gpustat`은 여러 GPU를 한눈에 보여 주는 제3자 도구다. 서버에 없으면 사용자 환경에 설치할 수 있다(설치 방법은 각 프로젝트의 문서로 확인한다).

```python {exec="browser" session="monitor"}
# nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw
#            --format=csv,noheader,nounits 형식을 흉내 낸 가상의 출력
gpu_csv = """0, 98, 30512, 40960, 71, 245.3
1, 0, 38200, 40960, 38, 61.2
2, 35, 12004, 40960, 55, 120.8
3, 0, 4, 40960, 30, 52.0"""
cols = ["index", "util_pct", "mem_used_mib", "mem_total_mib", "temp_c", "power_w"]
gpu = pd.read_csv(io.StringIO(gpu_csv), names=cols, skipinitialspace=True)
gpu["mem_free_mib"] = gpu["mem_total_mib"] - gpu["mem_used_mib"]
gpu["idle_holding"] = (gpu["util_pct"] < 5) & (gpu["mem_used_mib"] > 1024)   # 메모리만 잡고 노는 GPU
print(gpu[["index", "util_pct", "mem_free_mib", "temp_c", "idle_holding"]].to_string(index=False))
```

```output
 index  util_pct  mem_free_mib  temp_c  idle_holding
     0        98         10448      71         False
     1         0          2760      38          True
     2        35         28956      55         False
     3         0         40956      30         False
```

GPU 1은 메모리를 38 GB 가까이 잡고 있지만 사용률이 0이다. 순간 표본 하나로는 "잠깐 쉬는 중"일 수도 있으니, 몇 분 동안 기록해 평균을 본 뒤 판단한다(연습문제 `gpu-log-summary`). 계속 그렇다면 누군가 노트북 커널을 열어 둔 채 떠났거나, 죽은 실험의 프로세스가 남아 있는 것이다.

**사용률(utilization)이 재는 것.** nvidia-smi의 GPU 사용률은 표본 구간 동안 **커널이 하나라도 실행 중이던 시간의 비율**이다(`nvidia-smi --help-query-gpu`의 `utilization.gpu` 설명으로 확인한다). 그 커널이 GPU의 연산 유닛(SM)을 얼마나 채웠는지는 재지 않는다. 아래는 이 차이를 1 ms 간격의 단순한 시간선으로 흉내 낸 모델이다. 실제 GPU의 측정이 아니라 정의의 차이를 보이기 위한 계산이다.

```python {exec="browser" session="monitor"}
import numpy as np

def simulate(load_ms, kernel_ms, sm_frac, steps=200):
    """한 step = 데이터 대기(load_ms, GPU idle) + 커널 실행(kernel_ms, SM의 sm_frac만 사용). 1 ms 간격."""
    step = np.r_[np.zeros(load_ms), np.full(kernel_ms, sm_frac)]     # (load_ms + kernel_ms,)
    timeline = np.tile(step, steps)                                   # (T,)
    util = np.mean(timeline > 0)          # 커널이 실행 중이던 시간의 비율(nvidia-smi 사용률의 정의)
    work = np.mean(timeline)              # 실제로 SM을 쓴 비율(이 모델에서만 알 수 있는 값)
    return util, work

cases = {"데이터 대기형": (60, 40, 0.9), "작은 커널형": (0, 100, 0.1), "건강한 학습": (2, 98, 0.9)}
for name, args in cases.items():
    util, work = simulate(*args)
    print(f"{name:7s}  사용률 {util:5.0%}   실제 SM 활용 {work:5.0%}")
```

```output
데이터 대기형  사용률   40%   실제 SM 활용   36%
작은 커널형   사용률  100%   실제 SM 활용   10%
건강한 학습   사용률   98%   실제 SM 활용   88%
```

"작은 커널형"은 사용률 100%를 보이지만 GPU의 10%만 쓴다. 너무 작은 batch, 작은 모델, Python 루프에서 텐서 연산을 하나씩 부르는 코드가 이런 모양이다. 반대로 사용률이 낮고 들쭉날쭉하면 GPU가 데이터를 기다린다는 강한 신호다. 그래서 사용률은 **낮을 때 더 많은 것을 말해 주는 지표**다. 높을 때 실제 효율은 step당 시간, samples/s, 그리고 profiler([Profiling과 학습 효율화](../../22-experiment-operations/04-profiling-and-efficiency/index.md))로 확인한다.

::quiz{id="gpu-util-meaning"}

:::note[분야 예시]
기상·기후 분야의 격자 데이터나 현미경 영상처럼 샘플 하나가 큰 데이터를 학습할 때는, GPU 계산보다 디스크에서 읽고 압축을 풀고 자르는 데 시간이 더 걸리기 쉽다. 이때 nvidia-smi 사용률은 30–50% 사이를 오르내리고, `top`에서는 DataLoader worker들의 %CPU가 100% 가까이 붙어 있거나 `wa`(I/O 대기)가 높다. GPU를 더 좋은 것으로 바꿔도 빨라지지 않는다. 데이터를 미리 전처리해 저장하거나, 빠른 로컬 SSD로 옮기거나(2.5장), worker 수를 조정해야 한다.
:::

## 작업 세션 유지

SSH 접속은 언제든 끊긴다. 노트북 덮개를 닫거나, 와이파이가 바뀌거나, 방화벽이 유휴 연결을 정리한다. 접속이 끊기면 그 터미널에서 실행 중이던 프로세스는 SIGHUP을 받고, 대부분 함께 죽는다. 해결책은 프로세스를 **터미널이 아닌 곳**에 붙여 두는 것이다.

**tmux.** tmux는 서버에서 계속 살아 있는 터미널 관리자다. 내 SSH 접속은 tmux 세션을 "보는 창"일 뿐이라, 접속이 끊겨도 세션과 그 안의 프로세스는 그대로이고 다시 접속해 붙으면(attach) 화면이 그대로 남아 있다. tmux는 세 단계로 구성된다.

- **session**: 독립된 작업 공간 하나. 실험이나 프로젝트마다 하나.
- **window**: 세션 안의 탭.
- **pane**: window 안을 나눈 칸. 왼쪽에 학습, 오른쪽에 `nvidia-smi`를 띄우는 식이다.

```bash
# 집필 시점 예시 (tmux 3.4). 단축키는 기본 접두키 Ctrl-b를 누른 뒤 다음 키를 누른다.
tmux new -s exp42            # 이름 있는 세션 만들기
#   Ctrl-b d                 # 세션에서 빠져나오기(detach). 세션은 계속 돈다
tmux ls                      # 세션 목록
tmux attach -t exp42         # 다시 붙기
#   Ctrl-b c / n / p         # window 새로 만들기 / 다음 / 이전
#   Ctrl-b %  /  Ctrl-b "    # pane 좌우 / 상하로 나누기
#   Ctrl-b [                 # 스크롤 모드(q로 나옴)
tmux kill-session -t exp42   # 세션 끝내기
```

집필 시점에 `tmux new-session -d -s train '...'`으로 세션을 띄우고 빠져나온 뒤 `tmux ls`를 실행하면 `train: 1 windows (created ...)`가 나오고, `tmux capture-pane -p -t train`으로 그 안의 출력을 그대로 읽을 수 있었다. **screen**도 같은 역할을 하는 오래된 도구다(`screen -S name`, `Ctrl-a d`, `screen -r name`). 서버에 tmux가 없으면 screen이 있는지 확인한다.

**nohup과 tmux의 차이.**

| | `nohup cmd &` | tmux 안에서 `cmd` |
|---|---|---|
| 접속이 끊겨도 산다 | 예 (SIGHUP 무시) | 예 (애초에 터미널에 묶이지 않음) |
| 다시 그 화면에 붙기 | 불가 | `tmux attach` |
| 대화형 입력(pdb, 확인 프롬프트) | 불가 | 가능 |
| 서버 재부팅 뒤 | 사라짐 | 사라짐 |

둘 다 재부팅이나 OOM에서는 프로세스를 지켜 주지 못한다. 장기 실험은 checkpoint와 재시작 절차가 따로 필요하다(2.5장).

::quiz{id="nohup-vs-tmux"}

**로그 파일로 출력 남기기.** tmux 화면의 스크롤 기록은 길이 제한이 있고 세션과 함께 사라진다. 학습 출력은 tmux 안에서도 항상 파일로 남긴다. 2.1장의 리다이렉션 규칙대로 stderr까지 함께 남겨야 traceback이 보존된다.

```bash
# 집필 시점 예시: tmux 세션 안에서
mkdir -p logs
python train.py --config cfg.yaml 2>&1 | tee "logs/$(date +%Y%m%d_%H%M%S).log"
# 다른 pane이나 다른 접속에서 따라 보기
tail -f logs/20260302_100000.log
```

Python의 stdout은 터미널이 아닌 곳(파이프·파일)으로 갈 때 버퍼링되어 로그가 늦게 나타날 수 있다. `python -u`나 환경변수 `PYTHONUNBUFFERED=1`, 또는 `print(..., flush=True)`로 바로바로 기록되게 한다.

## 장애 진단

증상에서 원인으로 가는 지도다. 진단은 언제나 **흔적을 먼저 모으고** 그다음 다시 돌린다.

| 증상 | 의심 | 확인 (집필 시점 예시) |
|---|---|---|
| traceback 없이 죽음, 종료 상태 137 | OOM killer | `dmesg -T \| grep -i -E "out of memory\|killed process"` 또는 `journalctl -k` (권한이 없으면 관리자에게 요청) |
| `CUDA out of memory` traceback | GPU 메모리 부족 | batch 크기, 같은 GPU의 다른 프로세스(`nvidia-smi`) |
| `No space left on device` | 디스크 용량 또는 inode | `df -h`, `df -i`, `du -sh * \| sort -h`, `lsof +L1` |
| `ps`에 `Z`/`<defunct>` 상태가 쌓임 | zombie process | 부모 PID(`ps -o ppid= -p <PID>`)를 찾아 부모를 정리 |
| 학습이 점점 느려짐, `wa` 높음 | I/O 병목 | `iostat -x 1`, 데이터 위치(네트워크 저장소인가) |
| 학습을 멈췄는데 VRAM이 안 풀림 | 남은 자식 프로세스 | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, `ps -o pid,ppid,user,etime,cmd -p <PID>` |

**OOM killer.** 시스템 메모리(RAM)가 바닥나면 Linux 커널은 프로세스 하나를 골라 SIGKILL로 없앤다. 대개 메모리를 가장 많이 쓰는 프로세스, 즉 내 학습이다. SIGKILL이라 Python은 아무것도 남기지 못하고, 커널 로그에만 흔적이 남는다(문구는 커널 버전에 따라 조금 다르다). 원인은 흔히 데이터 전체를 RAM에 올리는 코드, worker마다 데이터셋 사본을 만드는 DataLoader, 누적되는 리스트(loss 텐서를 그래프째 모으는 등)다. 프로세스의 메모리 사용량(RSS)을 시간에 따라 기록해 두면 꾸준히 늘다 끊기는 모양으로 바로 드러난다.

**zombie process.** 자식 프로세스가 끝났는데 부모가 그 종료 상태를 회수(`wait`)하지 않으면, 자식은 메모리는 거의 쓰지 않지만 PID 표에 `Z` 상태로 남는다. zombie는 이미 죽었으므로 `kill`해도 사라지지 않는다. 부모가 회수하거나, 부모가 끝나 init이 회수해야 없어진다. 집필 시점에 Python에서 `os.fork()`한 자식을 바로 끝내고 부모가 기다리지 않게 했더니, `ps -o pid,ppid,stat,comm`에 `Z` 상태가 보였다.

**디스크 가득 참.** 공유 서버의 디스크가 차면 나만이 아니라 모두의 학습이 checkpoint 저장에서 실패한다. `du`로 큰 디렉터리를 찾되, `df`의 사용량과 `du`의 합계가 크게 다르면 **지워졌지만 아직 열려 있는 파일**을 의심한다. 파일을 지워도 그 파일을 연 프로세스가 살아 있는 동안 공간은 반환되지 않는다. `lsof +L1`이 이런 파일을 `(deleted)` 표시와 함께 보여 준다.

::quiz{id="disk-full-deleted-open"}

**I/O 병목.** 학습이 처음에는 빠르다가 느려지거나, 다른 사람이 큰 파일을 복사할 때마다 느려진다면 저장소가 병목이다. 네트워크 저장소(NFS 등)에 작은 파일 수십만 개를 두고 매 epoch 읽는 구성이 대표적이다. `iostat -x 1`에서 장치 사용률(`%util`)과 대기 시간이 높게 붙어 있는지 본다.

**GPU 메모리를 점유한 유령 프로세스.** `Ctrl-C`로 학습을 멈췄는데 VRAM이 풀리지 않는 경우, 부모는 죽었지만 DataLoader worker나 `multiprocessing` 자식이 남아 CUDA context를 붙잡고 있는 경우가 많다. PID를 확인하고, 내 것이면 SIGTERM으로 정리한다. 컨테이너 안에서는 PID 공간이 분리되어 nvidia-smi가 프로세스 목록을 비워 보여 줄 수 있으니, 그때는 컨테이너 밖에서 확인하거나 관리자에게 묻는다.

::quiz{id="ghost-gpu-process"}

:::note[분야 예시]
분자 동역학 시뮬레이션 결과로 그래프 신경망을 학습하는 경우, 궤적 파일 수천 개를 한 번에 메모리에 올리는 전처리가 RAM을 다 써 OOM killer에게 죽는 일이 흔하다. 노트북에서는 작은 부분집합으로 잘 돌았던 코드라 원인을 코드에서 찾기 쉽지만, 종료 상태 137과 커널 로그가 바로 답을 준다. 파일을 순서대로 읽어 결과만 모으는 방식(스트리밍)으로 바꾸면 해결된다.
:::

## 실습

공유 서버를 위험하게 하지 않는 범위에서 세 가지 장애를 일부러 만들고, 그 흔적을 도구로 확인한다. 예시 스크립트는 `labs/linux-and-remote/processes-and-monitoring/drills.py`다.

- **메모리 부족**: 자식 프로세스의 주소 공간을 2 GiB로 제한하고(`resource.setrlimit`) 4 GiB를 요청한다. 실제 OOM killer를 부르면 같은 서버의 다른 사람이 피해를 볼 수 있으므로, 제한을 건 자식 안에서만 실패하게 한다.
- **끊긴 세션**: 접속이 끊길 때 오는 SIGHUP을 직접 보내, 일반 프로세스와 `nohup` 프로세스를 비교한다.
- **디스크**: 50 MiB 파일을 연 채로 지우고 `lsof +L1`로 찾는다.

집필 시점의 실행 결과는 다음과 같다.

```text
[메모리] 종료 코드 1, stderr 마지막 줄: MemoryError
[끊긴 세션] SIGHUP 후 sleep      : 종료(returncode -1)
[끊긴 세션] SIGHUP 후 nohup sleep: 살아 있음
[디스크] 지웠지만 열려 있는 파일: debug.log (deleted), 50 MiB
```

메모리 실험에서는 Python이 `MemoryError` traceback을 남겼다. 프로세스 한도에 걸린 것이라 Python이 예외를 낼 기회가 있었기 때문이다. 시스템 전체 메모리가 바닥나 OOM killer가 개입하면 traceback 없이 137로 끝난다는 점과 대비해 기록한다. (`returncode -1`은 Python의 `subprocess`가 "시그널 1로 종료"를 표시하는 방식이다.)

이어서 다음을 직접 해 본다.

1. tmux 세션에서 짧은 학습(또는 `python -c "import time; [print(i, flush=True) or time.sleep(1) for i in range(600)]"`)을 띄우고, SSH 창을 강제로 닫은 뒤 다시 접속해 `tmux attach`로 돌아온다. 같은 일을 tmux 밖에서 하고 결과를 비교한다.
2. 학습 중에 다른 pane에서 `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1 > gpu.log`로 GPU 상태를 기록하고, 다음 연습문제의 함수로 요약한다.
3. DataLoader의 `num_workers`를 0과 8로 바꿔 가며 같은 학습을 돌리고, GPU 사용률 평균과 step당 시간이 어떻게 달라지는지 표로 만든다.

::exercise{id="gpu-log-summary"}

## 실패 사례

- **SSH 세션에서 직접 학습을 실행해 연결과 함께 종료되는 것.** 몇 시간 뒤 접속이 끊기며 SIGHUP으로 죽는다. 로그도 터미널에만 있었다면 어디까지 진행됐는지조차 모른다. 처음부터 tmux 안에서, 출력을 파일로 남기며 실행한다.
- **nvidia-smi의 사용률만 보고 병목을 단정하는 것.** 사용률 100%를 "GPU를 다 쓰고 있다"로 읽어 더 큰 GPU를 요청하거나, 사용률이 낮은 순간 표본 하나로 "데이터 병목"을 단정한다. 사용률은 커널 실행 시간의 비율일 뿐이다. 일정 시간의 기록, step당 시간, CPU·I/O 지표를 함께 보고 판단한다.
- **처음부터 `kill -9`를 쓰는 것.** checkpoint를 저장할 기회를 없애고, 쓰던 파일을 반쯤 쓰인 채 남긴다.
- **zombie를 kill로 지우려 하는 것.** 이미 죽은 프로세스라 시그널이 소용없다. 부모를 찾는다.
- **`du`만 보고 "공간이 남았다"고 판단하는 것.** 지워졌지만 열린 파일, 다른 사용자의 파일, inode 고갈은 `du`에 보이지 않는다.

## 통과 기준

- 학습이 느리거나 죽은 사례 하나를 골라, 종료 상태·로그·`ps`/`free`/`df`/`nvidia-smi` 기록을 근거로 원인을 자원 관점에서 설명하는 짧은 보고서(증상 → 확인한 흔적 → 원인 → 조치)를 쓴다.
- tmux 세션에서 실행한 학습이 SSH 연결을 끊었다 다시 붙은 뒤에도 계속 돌고 있음을 보이고, 그 출력이 로그 파일에도 남아 있음을 확인한다.
- 연습문제 `gpu-log-summary`의 테스트를 통과한다.
