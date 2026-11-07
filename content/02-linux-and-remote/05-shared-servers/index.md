---
id: linux-and-remote/shared-servers
title: "공유 서버와 클러스터 사용"
title_en: "Shared Servers and Clusters"
tier: core
status: draft
prerequisites: [linux-and-remote/processes-and-monitoring]
---

# 공유 서버와 클러스터 사용

연구실의 GPU 서버, 학교의 HPC 클러스터, 국가 슈퍼컴퓨터는 모두 **여러 사람이 함께 쓰는** 자원이다. 개인 노트북에서는 아무 GPU에나 올리고, 아무 디렉터리에나 쓰고, 끝난 프로세스를 내버려 두어도 나만 손해를 본다. 공유 자원에서는 같은 습관이 남의 학습을 죽이고, 디스크를 채워 모두의 checkpoint 저장을 실패시키고, 동료가 마감 전날 GPU를 못 쓰게 만든다.

이 장은 공유 자원을 쓰는 두 가지 방식을 다룬다. scheduler 없이 사람끼리 규칙으로 나눠 쓰는 연구실 서버와, job scheduler(이 장에서는 Slurm)가 자원을 배분하는 클러스터다. 그리고 어느 쪽이든 피할 수 없는 두 문제, 즉 **데이터를 어느 저장소에 둘 것인가**와 **며칠짜리 실험이 중간에 끊겨도 이어 가는 법**을 다룬다. 명령과 옵션은 **집필 시점 예시**다. 클러스터마다 partition 이름, GPU 요청 문법, 저장소 경로, 사용 규칙이 다르므로 반드시 소속 기관의 사용자 문서와 `man sbatch`로 확인한다.

## 학습 목표

- 공유 자원 사용 규칙을 지킨다
- job scheduler로 실험을 제출한다
- 저장소 계층을 구분한다

## 공유 서버 예절

**왜 필요한가.** scheduler가 없는 서버에서는 운영체제가 GPU를 나눠 주지 않는다. 누구든 아무 GPU에나 프로세스를 올릴 수 있고, 두 학습이 한 GPU에 올라가면 둘 다 느려지거나 메모리 부족으로 함께 죽는다. 사람이 scheduler 역할을 해야 한다.

**GPU 할당: CUDA_VISIBLE_DEVICES.** 환경변수 `CUDA_VISIBLE_DEVICES`는 CUDA 프로그램에 **보이는** GPU를 제한한다. `CUDA_VISIBLE_DEVICES=2,3 python train.py`로 실행하면 그 프로세스는 GPU 2와 3만 보고, 그것들을 각각 `cuda:0`, `cuda:1`로 부른다. 그래서 코드에는 `cuda:0`부터 쓰고, 어느 물리 GPU를 쓸지는 실행할 때 정한다. 이 환경변수 없이 실행하면 PyTorch 같은 프레임워크는 기본적으로 `cuda:0`, 즉 서버의 첫 GPU에 올라간다. 대개 그 GPU는 이미 누군가 쓰고 있다.

한 가지 함정이 있다. CUDA가 GPU에 번호를 매기는 기본 순서는 nvidia-smi의 순서(PCI bus 순)와 다를 수 있다. GPU 모델이 섞인 서버에서는 특히 그렇다. `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 함께 설정하면 두 번호가 일치한다. 정확한 동작은 CUDA 문서의 환경변수 항목으로 확인한다.

```bash
# 집필 시점 예시: 쓰기 전에 확인하고, 쓸 GPU만 보이게 한다
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2,3 python train.py
```

::quiz{id="cuda-visible-renumber"}

**비어 있는 GPU 고르기.** 2.2장에서 본 nvidia-smi 출력으로 "프로세스가 없고 메모리가 충분히 빈" GPU를 고른다. 아래는 가상의 출력에서 그 과정을 표로 한 것이다.

```python {exec="browser" session="shared"}
import io

import pandas as pd

# nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits (가상의 출력)
gpu_csv = "0, 30512, 40960\n1, 38200, 40960\n2, 4, 40960\n3, 4, 40960"
# nvidia-smi --query-compute-apps=pid,used_memory 를 ps로 소유자와 GPU 번호까지 붙였다고 가정한 표
procs = pd.DataFrame({"gpu": [0, 1], "pid": [5310, 4190], "user": ["lee", "kim"], "mem_mib": [30500, 38190]})

gpu = pd.read_csv(io.StringIO(gpu_csv), names=["index", "used_mib", "total_mib"], skipinitialspace=True)
gpu["free_mib"] = gpu["total_mib"] - gpu["used_mib"]
gpu["n_procs"] = gpu["index"].map(procs["gpu"].value_counts()).fillna(0).astype(int)
free = gpu[(gpu["n_procs"] == 0) & (gpu["free_mib"] >= 20000)]
print(gpu.to_string(index=False))
print("CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in free["index"]))
```

```output
 index  used_mib  total_mib  free_mib  n_procs
     0     30512      40960     10448        1
     1     38200      40960      2760        1
     2         4      40960     40956        0
     3         4      40960     40956        0
CUDA_VISIBLE_DEVICES=2,3
```

실제 nvidia-smi의 프로세스 목록은 GPU 번호가 아니라 uuid로 GPU를 가리키므로, 두 출력을 uuid로 이어야 한다. 이 과정을 연습문제 `gpu-allocation-plan`에서 직접 구현한다. 그리고 숫자가 "비어 있다"고 말해도, 연구실에 예약 표나 채팅 채널이 있다면 그것이 우선이다.

**사용 시간·우선순위 규칙.** 많은 연구실은 다음과 같은 규칙을 정해 둔다(내용은 연구실마다 다르다).

- GPU 사용 전 공지(채팅 채널, 예약 표)와 예상 종료 시각.
- 한 사람이 동시에 쓸 수 있는 GPU 수의 상한.
- 마감이 임박한 사람의 우선권, 장기 학습에 쓸 수 있는 시간대.
- 디버깅·대화형 작업용 GPU와 장기 학습용 GPU의 구분.

규칙이 없다면 만드는 것을 제안한다. 규칙이 없는 서버는 결국 먼저 올린 사람이 독점한다.

**종료하지 않은 프로세스.** 공유 서버를 가장 흔하게 막는 것은 악의가 아니라 잊어버린 프로세스다. 며칠 전 연 Jupyter 커널이 GPU 메모리 20 GB를 잡고 있거나, `Ctrl-C`로 멈춘 학습의 DataLoader worker가 남아 있다(2.2장). 하루를 마칠 때 `nvidia-smi`와 `ps -u $USER -o pid,etime,cmd`로 내 것을 확인하고 정리한다. 남의 프로세스로 보이는 것은 직접 종료하지 않고 소유자에게 묻는다.

::quiz{id="shared-etiquette"}

## Job scheduler

**왜 필요한가.** 사용자가 수백 명이고 노드가 수십 대인 클러스터에서는 사람끼리의 규칙으로 자원을 나눌 수 없다. **job scheduler**는 사용자가 "이 자원을 이만큼, 이 시간 동안" 요청한 작업(job)을 대기열(queue)에 받아, 자원이 나는 대로 노드에 배정한다. 배정된 job은 요청한 CPU·메모리·GPU만 쓰도록 제한되고, 요청한 시간이 지나면 종료된다. 대학과 연구소의 HPC 클러스터에서 널리 쓰이는 scheduler가 Slurm이다(PBS/Torque, LSF, HTCondor, Kubernetes 기반 시스템 등을 쓰는 곳도 있다).

**로그인 노드와 계산 노드.** 클러스터에 SSH로 접속하면 **로그인 노드**에 도착한다. 로그인 노드는 모두가 함께 쓰는 입구라, 코드 편집·작업 제출·작은 파일 정리에만 쓴다. 학습은 물론 무거운 전처리, 큰 압축 해제, 오래 걸리는 컴파일도 계산 노드에서 한다. 많은 클러스터가 로그인 노드에서 일정 이상의 자원을 쓰는 프로세스를 자동으로 종료한다. 원격 IDE(2.3장)의 서버 프로세스도 로그인 노드의 자원을 쓰므로 기관 규칙을 확인한다.

::quiz{id="login-node-compute"}

**기본 명령.** Slurm의 핵심 명령은 넷이다.

| 명령 | 하는 일 (집필 시점 예시) |
|---|---|
| `sbatch job.sh` | 배치 스크립트를 제출한다. job 번호가 출력된다 |
| `srun ...` | 자원을 받아 명령 하나를 실행한다. `srun --pty bash`처럼 대화형 셸을 받는 데도 쓴다(기관에 따라 `salloc`을 권한다) |
| `squeue -u $USER` | 내 job의 상태(대기 PD, 실행 R 등)를 본다 |
| `scancel <JOBID>` | job을 취소한다. array job의 한 task는 `<JOBID>_<번호>` |

끝난 job의 상태·소요 시간·최대 메모리는 `sacct -j <JOBID>`로 본다(어떤 필드를 볼 수 있는지는 `sacct --helpformat`으로 확인한다). 사용할 수 있는 partition(노드 묶음)과 그 상태는 `sinfo`로 본다.

**자원 요청.** 배치 스크립트 맨 위의 `#SBATCH` 줄이 자원 요청이다. 셸에게는 주석이지만 `sbatch`가 읽는다. 첫 실행 명령보다 앞에 있어야 한다.

```bash
#!/bin/bash
# 집필 시점 예시: GPU 1개 학습 job
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
python train.py --config cfg.yaml
```

| 요청 | 의미 | 너무 작으면 | 너무 크면 |
|---|---|---|---|
| `--gres=gpu:1` (또는 `--gpus=1`) | GPU 개수 | 학습 불가 | 대기 시간 증가, 자원 낭비 |
| `--cpus-per-task` | CPU 코어 수 | DataLoader 병목 | 대기 시간 증가 |
| `--mem` | 노드 메모리 | 메모리 초과로 종료 | 대기 시간 증가 |
| `--time` | 실행 시간 상한 | 시간 초과로 종료 | 대기 시간 증가(짧은 job이 먼저 끼워지는 backfill을 못 받음) |

요청은 **실제 사용량보다 조금 크게**가 원칙이다. 작은 시험 job을 먼저 돌려 `sacct`로 실제 메모리와 시간을 확인한 뒤 본 실험의 요청을 정한다. GPU를 요청하면 Slurm이 보통 그 job의 `CUDA_VISIBLE_DEVICES`를 배정된 GPU로 설정해 준다. 스크립트 안에서 덮어쓰지 않는다.

**array job.** seed만 다른 실험 다섯 개, 하이퍼파라미터 조합 스무 개처럼 같은 스크립트를 인자만 바꿔 여러 번 실행할 때는 **array job**으로 한 번에 제출한다. `#SBATCH --array=0-4`는 task 다섯 개를 만들고, 각 task에서 환경변수 `SLURM_ARRAY_TASK_ID`가 0–4 중 하나가 된다. 스크립트가 이 번호로 seed나 설정 파일을 고른다. `--array=0-99%10`처럼 `%`를 붙이면 동시에 실행되는 task 수를 제한한다. 로그 파일 이름에는 `%A`(array 전체의 job 번호)와 `%a`(task 번호)를 넣어 task마다 따로 남긴다.

```python {exec="browser" session="shared"}
import numpy as np

seeds = [0, 1, 2, 3, 4]                       # --array=0-4 의 각 SLURM_ARRAY_TASK_ID가 고르는 seed
def run_task(task_id):                        # 실제로는 train.py가 하는 일(가상의 정확도)
    rng = np.random.default_rng(seeds[task_id])
    return 0.90 + 0.02 * rng.standard_normal()

results = {f"seed{seeds[t]}": run_task(t) for t in range(5)}
acc = np.array(list(results.values()))        # (5,)
print({k: round(v, 4) for k, v in results.items()})
print(f"정확도 평균 {acc.mean():.4f} ± 표준편차 {acc.std(ddof=1):.4f} (n = {len(acc)})")
```

```output
{'seed0': 0.9025, 'seed1': 0.9069, 'seed2': 0.9038, 'seed3': 0.9408, 'seed4': 0.887}
정확도 평균 0.9082 ± 표준편차 0.0198 (n = 5)
```

seed 하나의 결과(예: seed3의 0.9408)만 보고했다면 성능을 과대평가했을 것이다. 여러 seed를 돌려 평균과 퍼짐을 함께 보고하는 이유이고, array job은 이것을 싸게 만든다. 결과를 모을 때는 **빠진 task가 없는지** 먼저 확인한다. 실패한 task를 조용히 빼고 평균을 내면, 잘 된 seed만 고른 것과 같은 편향이 생긴다.

::quiz{id="array-concurrency"}

:::note[분야 예시]
기후 모델 앙상블이나 분자 동역학의 독립 궤적처럼, 초기 조건만 다른 시뮬레이션 수십 개를 돌리는 일은 과학 계산에서 흔하다. array job의 task 번호로 초기 조건 파일을 고르고, task마다 출력 디렉터리를 나누면, 실패한 구성원만 골라 `--array=7,12`처럼 다시 제출할 수 있다.
:::

## 저장소 계층

**왜 필요한가.** 클러스터와 연구실 서버에는 성격이 다른 저장소가 여러 개 있다. 어디에 무엇을 두느냐가 학습 속도(I/O), 결과의 생존(백업), 다른 사람과의 공존(quota)을 결정한다.

| 계층 | 전형적인 성격 | 두는 것 | 두지 않는 것 |
|---|---|---|---|
| home (`~`) | 작은 quota, 백업되는 경우가 많음, 모든 노드에서 보임 | 코드, 설정 파일, 작은 환경 | 데이터셋, checkpoint |
| scratch | 크고 빠름, **백업 없음**, 오래된 파일 자동 삭제(purge) 정책이 흔함 | 학습 중 checkpoint, 전처리된 데이터, 캐시 | 유일한 최종 결과 |
| 프로젝트·공유 NAS | 중간 크기, 백업되는 경우가 많음, 그룹이 공유 | 원본 데이터, 최종 결과, 논문용 산출물 | 매 step 읽는 수많은 작은 파일 |
| 계산 노드 로컬 SSD | 가장 빠름, job이 끝나면 비워짐, 그 노드에서만 보임 | job 동안 반복해서 읽는 데이터 사본 | job 이후에도 필요한 것 |

위 표는 전형적인 구성이고, 실제 크기·백업 여부·삭제 정책은 기관 문서가 정한다. 내 사용량은 `df -h <경로>`, `du -sh <경로>`, `quota -s`로 확인하며, 병렬 파일시스템을 쓰는 클러스터는 전용 명령(예: Lustre의 `lfs quota`)을 안내한다.

**quota.** quota를 넘으면 쓰기가 실패한다. 학습 도중 checkpoint 저장이 `Disk quota exceeded`로 실패하면, 그 시점까지의 학습이 날아갈 수 있다. 장기 실험 전에 예상 checkpoint 크기 × 보관 개수가 quota 안에 드는지 계산한다. 오래된 checkpoint를 최근 몇 개만 남기고 지우는 정책을 학습 코드에 넣는다.

**로컬 SSD로 옮기기(staging).** 공유 NAS에 작은 파일 수십만 개로 된 데이터셋을 두고 매 epoch 읽으면 모두의 I/O가 느려진다(2.2장). job 시작 시 한 덩어리(tar)로 묶인 데이터를 노드의 로컬 SSD로 복사해 풀고, 학습은 로컬에서 읽고, job이 끝나면 비운다. 로컬 SSD의 경로는 클러스터마다 다르다(`$TMPDIR`, `/local`, `/tmp` 등). 기관 문서를 확인한다.

```bash
# 집필 시점 예시: job 스크립트 안에서 데이터를 로컬 SSD로 옮긴다
LOCAL="${TMPDIR:-/tmp}/$SLURM_JOB_ID"
mkdir -p "$LOCAL"
tar -xf /project/lab/data/train_shards.tar -C "$LOCAL"
python train.py --data "$LOCAL" --ckpt-dir /scratch/$USER/exp42
rm -rf "$LOCAL"
```

::quiz{id="scratch-not-backed-up"}

## 장기 실험 운영

**왜 필요한가.** 며칠짜리 학습은 거의 반드시 한 번은 끊긴다. 시간 상한에 걸리고, 노드가 고장 나고, 우선순위가 높은 job에게 자리를 내주고(**선점**, preemption), 공유 서버가 재부팅된다. 끊길 때마다 처음부터 다시 시작한다면 며칠짜리 실험은 끝나지 않는다.

**checkpoint와 재시작.** 해법은 주기적으로 학습 상태를 저장하고, 시작할 때 저장된 상태가 있으면 거기서 이어 가는 것이다. 저장할 상태는 모델 파라미터만이 아니다. optimizer 상태, learning rate scheduler, 현재 step·epoch, 난수 생성기 상태까지 저장해야 "끊기지 않은 것과 같은" 학습이 된다. 저장은 임시 파일에 쓴 뒤 이름을 바꾸는(`os.replace`) 방식으로 한다. 저장 도중 죽어도 이전 checkpoint가 온전히 남는다.

**선점 신호 받기.** Slurm은 job을 취소하거나 선점할 때 보통 SIGTERM을 먼저 보내고, 설정된 유예 시간이 지나면 SIGKILL을 보낸다. 시간 상한에 대해서는 `#SBATCH --signal=B:USR1@120`처럼 "상한 120초 전에 배치 셸(B)에 SIGUSR1을 보내라"고 요청할 수 있다. 2.2장의 패턴대로 학습 코드가 이 시그널을 받으면 현재 step을 마치고 저장한 뒤 끝내면 된다. 선점된 job을 자동으로 대기열에 되돌리는 `--requeue` 옵션의 동작은 클러스터 설정에 따라 다르니 문서로 확인한다.

**checkpoint 주기.** 자주 저장할수록 끊겼을 때 잃는 작업은 줄지만, 저장 자체에 시간이 든다(대형 모델은 checkpoint 하나가 수십 GB다). 저장 한 번의 비용을 $c$, 선점 사이의 평균 시간을 $M$, checkpoint 주기를 $\tau$라 하자. 단위 시간당 저장 비용은 $c/\tau$이고, 선점이 무작위로 일어나면 한 번 끊길 때 평균 $\tau/2$만큼의 작업을 잃으므로 단위 시간당 손실은 대략 $\tau/(2M)$이다. 둘의 합

$$
\frac{c}{\tau} + \frac{\tau}{2M}
$$

을 $\tau$로 미분해 0으로 놓으면 1차 근사 최적 주기

$$
\tau^* \approx \sqrt{2cM}
$$

을 얻는다. 이 근사는 $c$가 $M$보다 훨씬 작을 때 맞는다. 아래 Monte Carlo는 10시간 분량의 학습, 저장 비용 0.05시간(3분), 평균 8시간마다 선점, 재시작 비용 0.1시간이라는 가정에서 주기별 평균 총 소요 시간을 계산한다. 근사식은 $\sqrt{2 \times 0.05 \times 8} \approx 0.89$시간을 준다.

```python {exec="browser" session="shared"}
def wall_time_h(interval_h, work_h=10.0, ckpt_h=0.05, mtbf_h=8.0, restart_h=0.1, rng=None):
    done, t = 0.0, 0.0
    while done < work_h:
        seg = min(interval_h, work_h - done)              # 다음 checkpoint까지 할 일
        fail_in = rng.exponential(mtbf_h)                 # 다음 선점까지 남은 시간
        if fail_in >= seg + ckpt_h:                       # checkpoint까지 무사히 마침
            t, done = t + seg + ckpt_h, done + seg
        else:                                             # 선점: 마지막 checkpoint 이후 작업을 잃음
            t += fail_in + restart_h
    return t

rng = np.random.default_rng(0)
for interval_h in [0.25, 0.5, 0.9, 2.0, 4.0, 10.0]:
    runs = [wall_time_h(interval_h, rng=rng) for _ in range(2000)]
    print(f"주기 {interval_h:5.2f} h  평균 총 소요 {np.mean(runs):6.2f} h")
```

```output
주기  0.25 h  평균 총 소요  12.39 h
주기  0.50 h  평균 총 소요  11.55 h
주기  0.90 h  평균 총 소요  11.38 h
주기  2.00 h  평균 총 소요  11.88 h
주기  4.00 h  평균 총 소요  13.07 h
주기 10.00 h  평균 총 소요  20.23 h
```

근사 최적 주기 근처(0.9시간)에서 총 소요 시간이 가장 짧고, checkpoint 없이 한 번에 돌리면(주기 10시간) 두 배 가까이 걸린다. 최적점 근처에서는 곡선이 평평하므로 주기를 정확히 맞출 필요는 없다. 중요한 것은 **checkpoint가 있느냐 없느냐**와 자릿수다.

::widget{name="preemption-checkpoint"}

::quiz{id="checkpoint-interval-minutes"}

:::note[분야 예시]
기상 예보 모델처럼 정해진 시각까지 결과를 내야 하는 운영 계산은 선점이 없는 전용 자원을 쓰는 경우가 많다. 반면 연구용 대규모 학습은 값싼 선점형 자원(클라우드의 spot·preemptible 인스턴스, 클러스터의 낮은 우선순위 partition)을 쓰고 checkpoint로 끊김을 견디는 쪽이 비용 면에서 유리할 수 있다. 같은 계산이라도 마감과 예산에 따라 자원 선택이 달라진다.
:::

## 실습

Slurm array job으로 seed가 다른 실험 다섯 개를 제출하고 결과를 모은다. 파일은 `labs/linux-and-remote/shared-servers/`에 있다.

- `array_job.sh`: `--array=0-4%2`, GPU 1개, 30분 상한, task별 로그, 상한 120초 전 SIGUSR1, 배치 셸이 받은 시그널을 학습 프로세스에 전달.
- `train_seed.py`: `SLURM_ARRAY_TASK_ID`로 seed를 고르고, 100 step마다 원자적으로 checkpoint를 저장하고, 시작할 때 checkpoint가 있으면 이어 가고, SIGTERM·SIGUSR1을 받으면 현재 step을 마치고 저장한 뒤 종료한다.
- `collect.py`: `runs/seed*/result.json`을 모아 평균과 표준편차를 내고, 빠진 seed가 있으면 0이 아닌 코드로 끝난다.

Slurm이 없는 환경에서도 `SLURM_ARRAY_TASK_ID`를 직접 주면 같은 코드를 돌릴 수 있다. 집필 시점에 task 0–4를 차례로 실행한 뒤 `collect.py`를 실행한 결과는 다음과 같다.

```text
[seed 0] 완료 mse=0.2516
[seed 1] 완료 mse=0.2610
[seed 2] 완료 mse=0.2590
[seed 3] 완료 mse=0.2417
[seed 4] 완료 mse=0.2385
seed 0: mse = 0.2516
seed 1: mse = 0.2610
seed 2: mse = 0.2590
seed 3: mse = 0.2417
seed 4: mse = 0.2385
mse 평균 0.2504 ± 표준편차 0.0100 (n = 5)
```

선점도 흉내 낼 수 있다. `array_job.sh`를 step이 느려지게(`--step-sleep-s 0.005`) 실행하고 1.5초 뒤 배치 셸에 SIGUSR1을 보낸 다음, 같은 명령을 다시 실행했다.

```text
--- 1차 실행 (1.5초 뒤 SIGUSR1) ---
host=vm job=local_2 gpus=none
[seed 2] 종료 요청: step 261에서 저장하고 종료
train_seed.py exit=0
--- 2차 실행 (재제출) ---
host=vm job=local_2 gpus=none
[seed 2] checkpoint step 261에서 재시작
[seed 2] 완료 mse=0.2590
train_seed.py exit=0
```

시그널이 도착한 step은 실행할 때마다 다르다. 끊겼다 이어 간 seed 2의 최종 결과(0.2590)가 한 번에 끝낸 결과와 같다는 점을 확인한다. 이 장난감 학습은 무작위성이 데이터 생성에만 있어서 같아진다. 실제 학습에서는 optimizer와 난수 생성기 상태까지 저장해야 같아진다.

실제 클러스터에서는 다음을 한다.

1. 기관 문서에 맞춰 `array_job.sh`의 partition, GPU 요청 문법, 시간 상한을 고치고 `mkdir -p logs` 후 `sbatch array_job.sh`로 제출한다.
2. `squeue -u $USER`로 동시에 최대 2개만 실행되는지 보고, 끝난 뒤 `sacct -j <JOBID>`로 task별 상태와 소요 시간, 최대 메모리를 기록한다.
3. 한 task를 `scancel <JOBID>_<번호>`로 취소한 뒤 `collect.py`가 빠진 seed를 알리는지 확인하고, 그 task만 다시 제출해 checkpoint에서 이어 가는지 본다.
4. 결과 디렉터리를 scratch에 두었다면, 최종 결과를 백업되는 저장소로 복사한다.

scheduler가 없는 연구실 서버에서 쓸 GPU 선택은 다음 연습문제로 익힌다.

::exercise{id="gpu-allocation-plan"}

## 실패 사례

- **다른 사람의 GPU에 프로세스를 올리는 것.** `CUDA_VISIBLE_DEVICES` 없이 실행해 GPU 0에 올라가거나, 메모리가 조금 남은 GPU에 끼어든다. 남의 학습이 메모리 부족으로 죽고, 내 학습도 느려진다. 실행 전에 확인하고, 쓸 GPU만 보이게 한다.
- **백업되지 않는 scratch에 유일한 결과를 두는 것.** 몇 주 뒤 purge 정책이나 파일시스템 장애로 사라진다. 실험이 끝나면 최종 산출물을 백업되는 저장소로 옮기는 것까지가 실험이다.
- **로그인 노드에서 계산을 돌리는 것.** 모두의 접속을 느리게 하고, 관리자에게 프로세스가 강제 종료된다.
- **자원을 과하게 요청하는 것.** "넉넉하게" GPU 4개, 메모리 500 GB, 7일을 요청하면 대기열에서 하염없이 기다리고, 배정되어도 대부분을 놀린다. 시험 job으로 실제 사용량을 재고 요청한다.
- **checkpoint 없이 며칠짜리 job을 제출하는 것.** 시간 상한이나 선점 한 번에 모든 것이 날아간다. checkpoint는 있지만 재시작 코드를 시험해 보지 않은 경우도 같다. 일부러 끊고 이어 가는 시험을 한 번은 한다.
- **실패한 task를 조용히 빼고 결과를 요약하는 것.** 남은 seed만으로 낸 평균은 편향된다.

## 통과 기준

- 연구실(또는 기관) 공유 자원 규칙을 확인해 GPU 할당·사용 시간·저장소 위치에 대해 설명하고, 실행 전에 비어 있는 GPU를 확인해 `CUDA_VISIBLE_DEVICES`로 지정한다.
- Slurm array job으로 seed가 다른 실험 다섯 개를 제출하고, `sacct`로 상태를 확인한 뒤 결과를 모아 평균과 표준편차를 보고한다. 한 task를 중간에 끊었다가 checkpoint에서 이어 간다.
- 연습문제 `gpu-allocation-plan`의 테스트를 통과한다.
