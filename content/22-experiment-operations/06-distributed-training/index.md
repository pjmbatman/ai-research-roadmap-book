---
id: experiment-operations/distributed-training
title: "다중 GPU와 분산 학습"
title_en: "Multi-GPU and Distributed Training"
tier: advanced
status: draft
prerequisites: [experiment-operations/memory-management, linux-and-remote/shared-servers]
---

# 다중 GPU와 분산 학습

GPU 한 대로 학습이 너무 오래 걸리거나 모델이 한 대에 들어가지 않으면 여러 GPU를 쓴다. 그런데 GPU를 8대로 늘렸다고 학습이 8배 빨라지는 것은 아니고, 같은 코드가 같은 결과를 내는 것도 아니다. batch가 GPU 수만큼 커지고, 그에 맞춰 학습률과 step 수가 바뀌고, 매 step GPU끼리 gradient를 주고받는 통신 비용이 생긴다. metric을 rank 하나에서만 계산하면 데이터의 일부만 본 숫자를 보고하게 된다.

이 장은 분산 학습을 **같은 수학을 여러 장치에 나누는 일**로 다룬다. 먼저 병렬화 방식을 구분하고, 가장 흔한 데이터 병렬(DDP)이 왜 단일 GPU 학습과 같은 gradient를 내는지 NumPy와 실제 PyTorch로 확인한다. 그다음 데이터 분배, 학습 설정의 변화, 분산 환경의 metric과 checkpoint, 통신 비용을 차례로 본다. 이 환경에는 GPU가 없으므로 PyTorch 코드는 CPU의 gloo backend로 실행하고, 다중 GPU·NCCL에서만 가능한 것은 실행하지 않고 개념과 집필 시점 예시로 쓴다. 메모리 예산은 [22.5장](../05-memory-management/index.md), 공유 서버와 job scheduler는 [2.5장](../../02-linux-and-remote/05-shared-servers/index.md)에서 다뤘다.

## 학습 목표

- 병렬화 방식을 구분한다
- DDP로 학습을 확장한다
- 분산 환경의 metric과 checkpoint를 다룬다

## 병렬화 방식

**왜 필요한가.** 여러 GPU를 쓰는 이유는 두 가지다. 같은 학습을 **더 빨리** 하려는 것(처리량)과, 한 GPU에 **들어가지 않는** 모델을 학습하려는 것(메모리)이다. 어느 이유냐에 따라 나누는 대상이 다르다.

**데이터 병렬(data parallelism).** 모든 GPU가 모델 전체를 복제해 들고, batch를 나누어 각자 forward·backward를 한 뒤, gradient를 평균해 똑같이 갱신한다. GPU마다 다른 데이터를 보지만 파라미터는 항상 같다. 평균은 **all-reduce**라는 집합 통신(collective communication)으로 한다. all-reduce는 모든 참여자의 배열을 더해 그 합을 모두에게 돌려주는 연산이다.

왜 이것이 단일 GPU 학습과 같은가. 손실이 표본 평균 $L = \frac{1}{B}\sum_{i=1}^{B} \ell_i$이고, batch를 $W$개 GPU에 $B/W$개씩 똑같이 나누면

$$
\nabla L = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell_i = \frac{1}{W}\sum_{r=1}^{W} \underbrace{\frac{W}{B}\sum_{i \in \mathcal{D}_r} \nabla \ell_i}_{\text{GPU } r\text{의 평균 gradient}}
$$

이다. 여기서 $\mathcal{D}_r$은 GPU $r$이 맡은 표본의 집합이다. 각 GPU가 자기 몫의 평균 gradient를 구하고 그 $W$개를 평균하면 전체 batch의 gradient와 같다. 단, 이 등식은 **모든 GPU의 표본 수가 같을 때** 성립한다.

```python {exec="browser" session="dist"}
import numpy as np

rng = np.random.default_rng(0)
N, D, W = 64, 3, 4                                 # 표본 수, 특성 수, worker 수
X = rng.normal(size=(N, D))                        # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=N)   # (N,)
w = np.zeros(D)                                    # 모든 worker가 같은 파라미터를 들고 있다

def grad(Xb, yb):                                  # 평균 MSE의 gradient, (D,)
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

g_full = grad(X, y)                                # 한 장치가 batch 64 전체를 본다
local = [grad(Xs, ys) for Xs, ys in zip(np.split(X, W), np.split(y, W))]   # worker마다 16개
g_allreduce = np.sum(local, axis=0) / W            # all-reduce(합) 뒤 W로 나눈다
print("worker별 gradient가 서로 다른가:", not np.allclose(local[0], local[1]))
print("all-reduce 평균 == 전체 batch gradient:", np.allclose(g_allreduce, g_full))

sizes = [40, 8, 8, 8]                              # 표본이 불균등하게 나뉘면
parts = np.split(np.arange(N), np.cumsum(sizes)[:-1])
g_uneven = np.mean([grad(X[p], y[p]) for p in parts], axis=0)
print("불균등 분할에서 단순 평균 == 전체 batch gradient:", np.allclose(g_uneven, g_full))
```

```output
worker별 gradient가 서로 다른가: True
all-reduce 평균 == 전체 batch gradient: True
불균등 분할에서 단순 평균 == 전체 batch gradient: False
```

worker마다 gradient는 다르지만 평균은 전체 batch의 gradient와 같다. 표본을 40·8·8·8개로 불균등하게 나누면 단순 평균은 8개짜리 worker에 과도한 가중치를 주어 전체 gradient와 달라진다. 분산 데이터 분배(아래 "분산 데이터")가 rank마다 표본 수를 똑같이 맞추는 이유다.

::quiz{id="allreduce-equal-shards"}

**텐서 병렬(tensor parallelism).** 한 층의 행렬 자체를 여러 GPU에 나눈다. $\mathbf{Y} = \mathbf{X}\mathbf{W}$에서 $\mathbf{W}$를 열 방향으로 $[\mathbf{W}_1, \mathbf{W}_2]$로 나누면 $\mathbf{Y} = [\mathbf{X}\mathbf{W}_1, \mathbf{X}\mathbf{W}_2]$이므로, 두 GPU가 각자 절반의 열을 계산하고 결과를 이어 붙인다. 행 방향으로 나누면 $\mathbf{Y} = \mathbf{X}_1\mathbf{W}_1 + \mathbf{X}_2\mathbf{W}_2$가 되어 부분 결과를 더하는 all-reduce가 필요하다. 층 하나마다 통신이 일어나므로 GPU 사이 연결이 매우 빨라야 하고, 보통 한 서버(node) 안의 GPU끼리만 쓴다.

**파이프라인 병렬(pipeline parallelism).** 층을 순서대로 묶어 단계(stage)별로 다른 GPU에 둔다. GPU 1이 앞쪽 층, GPU 2가 뒤쪽 층을 맡는 식이다. 그대로 하면 GPU 2가 계산하는 동안 GPU 1은 논다. 그래서 batch를 micro-batch $m$개로 나누어 흘려보낸다. 단계가 $p$개이면 처음 $p-1$개 시간 칸 동안 파이프라인을 채우고 마지막 $p-1$개 칸 동안 비우므로, 노는 시간(bubble)의 비율은 대략

$$
\frac{p - 1}{m + p - 1}
$$

이다. $p = 4$, $m = 4$이면 $3/7 \approx 43\%$, $m = 32$이면 $3/35 \approx 9\%$다. micro-batch가 많을수록 bubble이 줄지만, 그만큼 활성값을 동시에 더 들고 있어야 한다.

**분할(sharding): ZeRO와 FSDP.** 데이터 병렬의 낭비는 모든 GPU가 파라미터·gradient·optimizer 상태를 **똑같이 복제**한다는 점이다. 22.5장에서 본 것처럼 fp32 + Adam은 파라미터당 16 byte인데, GPU 8대가 같은 16 byte를 각자 들고 있다. ZeRO(Zero Redundancy Optimizer)는 이 복제를 단계적으로 없앤다. 1단계는 optimizer 상태를, 2단계는 gradient까지, 3단계는 파라미터까지 GPU 수로 나누어 각 GPU가 자기 몫만 들게 한다. PyTorch의 FSDP(Fully Sharded Data Parallel)는 3단계에 해당하는 방식이다. 데이터를 나누어 처리한다는 점에서는 여전히 데이터 병렬이다.

```python {exec="browser" session="dist"}
P, W = 7e9, 8                                      # 파라미터 수, GPU 수 (fp32 + Adam: 4 + 4 + 8 byte)
schemes = {                                        # GPU 하나가 들고 있는 byte/파라미터: (파라미터, gradient, optimizer 상태)
    "DDP (모두 복제)":                   (4, 4, 8),
    "ZeRO-1 (optimizer 상태 분할)":      (4, 4, 8 / W),
    "ZeRO-2 (+ gradient 분할)":          (4, 4 / W, 8 / W),
    "ZeRO-3 / FSDP (+ 파라미터 분할)":   (4 / W, 4 / W, 8 / W),
}
for name, b in schemes.items():
    print(f"{name:26s} {sum(b):5.2f} byte/파라미터 → GPU당 {P * sum(b) / 2**30:6.1f} GiB")
```

```output
DDP (모두 복제)                16.00 byte/파라미터 → GPU당  104.3 GiB
ZeRO-1 (optimizer 상태 분할)    9.00 byte/파라미터 → GPU당   58.7 GiB
ZeRO-2 (+ gradient 분할)      5.50 byte/파라미터 → GPU당   35.9 GiB
ZeRO-3 / FSDP (+ 파라미터 분할)   2.00 byte/파라미터 → GPU당   13.0 GiB
```

70억 파라미터 모델은 DDP로는 GPU당 104 GiB가 필요하지만, 8대에 완전히 나누면 13 GiB다(활성값 제외). 대가는 통신이다. 파라미터를 나누어 두었으므로 각 층을 계산하기 직전에 다른 GPU에서 그 층의 파라미터를 모아 오는 all-gather가 필요하고, gradient는 all-reduce 대신 각자 자기 몫만 받는 reduce-scatter로 모은다. 그래서 FSDP는 층을 계산하는 동안 **그 층 하나 분량의 전체 파라미터**를 잠깐 들고 있어야 하고, 통신량은 DDP보다 많다.

| 방식 | 나누는 것 | 주로 푸는 문제 | 통신 |
|---|---|---|---|
| 데이터 병렬(DDP) | 데이터 | 처리량 | step마다 gradient all-reduce |
| ZeRO·FSDP | 데이터 + 상태(파라미터·gradient·optimizer) | 메모리 + 처리량 | 층마다 all-gather, gradient reduce-scatter |
| 텐서 병렬 | 층 안의 행렬 | 한 층이 너무 큼 | 층마다 all-reduce·all-gather (빠른 연결 필요) |
| 파이프라인 병렬 | 층의 순서 | 모델 전체가 너무 큼 | 단계 경계의 활성값 전달, bubble |

큰 모델 학습은 이들을 조합한다. 연구실 규모에서 첫 선택은 거의 항상 이렇다. 모델이 GPU 하나에 들어가면 DDP, 들어가지 않으면 FSDP를 먼저 시도한다. 텐서·파이프라인 병렬은 그것으로도 안 될 때 검토한다.

::quiz{id="which-parallelism"}

## DDP

**정의.** 분산 학습의 각 프로세스는 다음 번호로 자기 위치를 안다.

- **world size**: 전체 프로세스 수. 보통 GPU 수와 같다.
- **rank**: 0부터 world size − 1까지의 전역 번호. rank 0이 로그·checkpoint 같은 "대표" 역할을 맡는 관례가 있다.
- **local rank**: 한 서버(node) 안에서의 번호. 이 프로세스가 쓸 GPU 번호로 쓴다.

PyTorch의 `DistributedDataParallel`(DDP)은 다음 일을 자동으로 한다. 생성할 때 rank 0의 파라미터를 모든 rank에 방송(broadcast)해 시작점을 맞추고, backward 중에 gradient를 버킷(bucket) 단위로 all-reduce해 평균한다. 버킷 하나의 gradient가 준비되는 대로 통신을 시작하므로, 뒤쪽 층의 통신이 앞쪽 층의 backward 계산과 겹친다(overlap). 버킷 크기는 `bucket_cap_mb`로 정하고, 집필 시점 기본값은 25 MiB다.

실제 DDP가 단일 프로세스와 같은 gradient를 내는지 CPU에서 확인한다. gloo backend로 프로세스 2개를 띄우고, 각 프로세스가 8개 표본 중 절반을 보게 한다.

```python {exec="python"}
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

def worker(rank, world, init_file, X, y, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = DDP(nn.Linear(3, 1))                   # 생성할 때 rank 0의 파라미터를 모든 rank에 방송한다
    xs, ys = X.chunk(world)[rank], y.chunk(world)[rank]          # rank마다 다른 절반, (4, 3)
    nn.functional.mse_loss(model(xs), ys).backward()             # backward 중에 gradient를 all-reduce(평균)
    out.put((rank, model.module.weight.grad.flatten().tolist()))
    dist.destroy_process_group()

torch.manual_seed(1)
Xt, yt = torch.randn(8, 3), torch.randn(8, 1)      # (8, 3), (8, 1)
ctx = mp.get_context("fork")
out, init_file = ctx.SimpleQueue(), tempfile.mktemp()    # 두 프로세스가 만나는 rendezvous 파일 경로
procs = [ctx.Process(target=worker, args=(r, 2, init_file, Xt, yt, out)) for r in range(2)]
for p in procs: p.start()
grads = dict(out.get() for _ in procs)
for p in procs: p.join()

torch.manual_seed(0)
ref = nn.Linear(3, 1)                              # 한 프로세스가 8개 전체를 본다
nn.functional.mse_loss(ref(Xt), yt).backward()
print("rank 0 gradient:", [round(v, 4) for v in grads[0]])
print("rank 1 gradient:", [round(v, 4) for v in grads[1]])
print("단일 프로세스 :", [round(v, 4) for v in ref.weight.grad.flatten().tolist()])
```

```output
rank 0 gradient: [0.7545, 4.0162, -3.0082]
rank 1 gradient: [0.7545, 4.0162, -3.0082]
단일 프로세스 : [0.7545, 4.0162, -3.0082]
```

두 rank는 서로 다른 절반을 보고 backward를 했지만, DDP가 all-reduce로 평균한 뒤에는 같은 gradient를 갖고, 그 값은 8개 전체를 본 단일 프로세스의 gradient와 같다. 이 셀은 노트북 셀 안에서 돌리기 위해 `fork`로 프로세스를 띄웠다. 실제 학습 스크립트는 아래처럼 `torchrun`으로 띄운다.

**torchrun.** `torchrun`은 프로세스를 원하는 수만큼 띄우고, 각 프로세스에 `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` 같은 환경 변수를 넣어 준다. 스크립트는 `dist.init_process_group(backend)`만 부르면 이 변수를 읽어 서로를 찾는다. GPU에서는 NVIDIA GPU 간 통신 라이브러리인 NCCL backend를, CPU에서는 gloo를 쓴다.

```bash
# 집필 시점 예시: 서버 한 대의 GPU 4개
torchrun --standalone --nproc-per-node=4 train.py --config configs/base.yaml

# 집필 시점 예시: 서버 2대 × GPU 4개 (각 서버에서 실행, 주소와 포트는 클러스터에 맞춘다)
torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=node01:29500 \
    --rdzv-id=exp42 train.py --config configs/base.yaml
```

옵션 이름은 버전에 따라 밑줄(`--nproc_per_node`)과 하이픈 표기가 함께 받아들여지기도 하므로 `torchrun --help`로 확인한다. Slurm 같은 job scheduler에서는 GPU 수와 node 수를 job 스크립트에서 요청하고, 그 안에서 `torchrun`을 실행한다(2.5장).

단일 GPU 학습 스크립트를 DDP로 바꿀 때 고칠 곳은 많지 않다.

1. 시작할 때 `dist.init_process_group(...)`, 끝날 때 `dist.destroy_process_group()`.
2. `LOCAL_RANK`로 GPU를 고르고(`torch.cuda.set_device`), 모델을 그 GPU에 올린 뒤 `DDP(model, device_ids=[local_rank])`로 감싼다.
3. DataLoader에 `DistributedSampler`를 쓰고, 매 epoch `sampler.set_epoch(epoch)`를 부른다(다음 절).
4. 로그·checkpoint 저장은 rank 0에서만, metric은 모든 rank에서 모아서(아래 "metric과 checkpoint").

gradient accumulation을 DDP와 함께 쓰면, 누적하는 중간 micro-batch에서도 매번 all-reduce가 일어나 통신이 낭비된다. DDP의 `no_sync()` context manager 안에서 중간 micro-batch의 backward를 하면 마지막 micro-batch에서만 통신한다.

## 분산 데이터

**왜 필요한가.** 모든 rank가 같은 DataLoader를 쓰면 모든 GPU가 같은 표본을 보고, GPU를 늘려도 새로운 데이터를 보지 못한다. 각 rank는 서로 겹치지 않는 몫을 봐야 한다.

**DistributedSampler.** PyTorch의 `DistributedSampler`는 표본 번호를 rank 수로 나누어 각 rank에 겹치지 않는 몫을 준다. 동작은 다음과 같다.

1. 표본 번호 $0, \dots, N-1$을 섞는다. 이때 난수 시드는 `seed + epoch`이고, 모든 rank가 같은 시드를 쓰므로 **같은 순서**를 얻는다.
2. 길이가 world size $W$의 배수가 되도록, 기본값(`drop_last=False`)에서는 앞쪽 표본을 반복해 덧붙인다.
3. rank $r$은 $r, r + W, r + 2W, \dots$번째를 가져간다.

```python {exec="python"}
from torch.utils.data.distributed import DistributedSampler

data = list(range(10))                             # 표본 10개, rank 4개
for epoch in [0, 1]:
    parts = []
    for r in range(4):
        s = DistributedSampler(data, num_replicas=4, rank=r, shuffle=True, seed=0)
        s.set_epoch(epoch)                         # 빠뜨리면 매 epoch 같은 순서
        parts.append(list(s))
    print(f"epoch {epoch}: {parts}")
flat = sum(parts, [])
print(f"전체 {len(flat)}개, 고유 {len(set(flat))}개 → 중복 {len(flat) - len(set(flat))}개")
```

```output
epoch 0: [[4, 3, 6], [1, 9, 2], [7, 0, 4], [5, 8, 1]]
epoch 1: [[5, 0, 7], [6, 8, 4], [1, 9, 5], [2, 3, 6]]
전체 12개, 고유 10개 → 중복 2개
```

세 가지를 읽는다.

- **rank마다 표본 수가 같다(3개).** 그래서 앞 절의 "평균의 평균 = 전체 평균"이 성립한다.
- **10개를 4로 나누어떨어지게 하려고 2개가 중복되었다.** 학습에서는 영향이 작지만, 평가 metric에서는 중복된 표본이 두 번 세어진다(아래 "metric과 checkpoint").
- **`set_epoch`가 epoch마다 순서를 바꾼다.** `set_epoch`를 부르지 않으면 epoch이 바뀌어도 시드가 그대로라, 매 epoch 같은 순서로 학습한다. 오류는 나지 않는다.

::quiz{id="set-epoch"}

## 학습 설정의 변화

**왜 필요한가.** GPU를 늘리면서 rank당 batch를 그대로 두면 유효 batch가 GPU 수에 비례해 커진다. 그러면 epoch당 step 수가 줄고, 같은 학습률은 다른 학습을 만든다. GPU 수는 하드웨어 설정이지만, 그 결과로 바뀌는 batch·학습률·step 수는 **최적화 설정**이다.

**정의.** rank당 batch $b$, GPU 수 $W$, gradient accumulation $k$이면 유효 batch는 $B = bWk$이고, 표본 $N$개의 epoch당 step 수는 $\lceil N/B \rceil$이다. batch를 $k$배 키울 때 학습률도 $k$배 키우는 **선형 스케일링 규칙**은 momentum SGD에서 warm-up과 함께 쓰여 큰 batch까지 작은 batch와 비슷한 정확도를 냈다는 보고가 있다(Goyal 외, 2017; 10.4장). 학습률을 $\sqrt{k}$배로 키우는 제곱근 규칙도 쓰인다. 어느 규칙도 모든 optimizer와 모든 batch 크기에서 성립하지는 않는다.

```python {exec="browser" session="dist"}
N, per_gpu, base_lr, epochs = 50_000, 32, 0.1, 90  # 기준: GPU 1개, batch 32에서 튜닝한 학습률 0.1
print(" GPU  유효 batch  step/epoch  총 step  학습률(선형)  학습률(제곱근)")
for g in [1, 2, 4, 8, 32]:
    B = per_gpu * g
    steps = int(np.ceil(N / B))
    print(f"{g:4d} {B:10d} {steps:11d} {steps * epochs:8d} {base_lr * g:12.2f} {base_lr * np.sqrt(g):14.3f}")
```

```output
 GPU  유효 batch  step/epoch  총 step  학습률(선형)  학습률(제곱근)
   1         32        1563   140670         0.10          0.100
   2         64         782    70380         0.20          0.141
   4        128         391    35190         0.40          0.200
   8        256         196    17640         0.80          0.283
  32       1024          49     4410         3.20          0.566
```

GPU 32대에서 같은 90 epoch는 4410 step에 불과하다. 처음 1000 step 동안 warm-up하도록 step 단위로 적어 둔 설정은 GPU 1대에서는 학습의 0.7%이지만 32대에서는 23%가 된다. 학습률 3.2는 대부분의 모델에서 발산할 크기다. GPU 수를 바꾸면 다음 중 하나를 고른다.

- **같은 학습을 원하면** 유효 batch를 고정한다. rank당 batch를 $B/W$로 줄이거나, GPU가 적을 때 accumulation으로 채운다. 학습률·스케줄은 그대로 둔다. 실습에서 이렇게 하면 프로세스 수와 무관하게 같은 결과가 나오는 것을 확인한다.
- **유효 batch를 키우려면** 학습률·warm-up·스케줄을 step이 아니라 표본 수나 epoch 기준으로 다시 정하고, 짧은 실행으로 다시 튜닝한다. 결과가 단일 GPU 실험과 비교 가능하다고 가정하지 않는다.

어느 쪽이든 유효 batch, GPU 수, 누적 횟수, 학습률을 실험 기록(22.1장)에 함께 남긴다.

::quiz{id="effective-batch"}

:::note[분야 예시]
기후 재분석 자료로 기상 예측 모델을 학습하는 경우 한 표본이 커서 GPU 하나의 batch가 몇 개에 불과하다. GPU를 늘리면 유효 batch가 수십으로 늘어나는데, 이때 BatchNorm을 쓰는 모델은 통계를 GPU당 몇 개의 표본에서 계산하게 된다. 유효 batch가 커져도 정규화 통계는 rank당 batch에서 계산되므로, rank 사이에서 통계를 모으는 동기화 BatchNorm(`torch.nn.SyncBatchNorm`)을 쓰거나 batch 크기와 무관한 정규화(LayerNorm, GroupNorm)를 고려한다.
:::

## metric과 checkpoint

**분산 집계.** 평가도 분산하면 각 rank는 검증 데이터의 일부에 대한 metric만 안다. rank 0의 값만 보고하면 데이터의 $1/W$로 계산한 숫자를 보고하는 셈이다. 올바른 집계는 **합과 개수를 모아 나누는 것**이다. 정확도라면 rank마다 (맞힌 수, 표본 수)를 구하고 `dist.all_reduce`로 더한 뒤 나눈다. rank별 평균을 단순 평균하면 rank마다 표본 수가 다를 때 틀린다(앞 절의 불균등 분할과 같은 이유).

```python {exec="browser" session="dist"}
rng = np.random.default_rng(1)
N, W = 1001, 4                                     # 검증 표본 1001개, rank 4개
is_correct = rng.random(N) < 0.8                   # 표본별 정답 여부, (N,)
total = int(np.ceil(N / W)) * W                    # DistributedSampler는 W의 배수로 맞춘다
idx = np.concatenate([np.arange(N), np.arange(total - N)])   # 앞쪽 표본을 반복해 채운다(shuffle 없음)
shards = [idx[r::W] for r in range(W)]
print("rank별 정확도:", [f"{is_correct[s].mean():.4f}" for s in shards])
print(f"rank 0만 보고        {is_correct[shards[0]].mean():.4f}")
print(f"전 rank 합산(중복 포함) {np.mean(np.concatenate([is_correct[s] for s in shards])):.4f}  ({total}개)")
print(f"실제 정확도           {is_correct.mean():.4f}  ({N}개)")
```

```output
rank별 정확도: ['0.7928', '0.7928', '0.7968', '0.7530']
rank 0만 보고        0.7928
전 rank 합산(중복 포함) 0.7839  (1004개)
실제 정확도           0.7842  (1001개)
```

rank 0만 보면 0.7928로, 실제 0.7842보다 0.009 높다. rank 사이의 차이(0.7530–0.7968)는 모델의 차이가 아니라 표본 250개의 표집 변동이다. 모든 rank를 합산하면 실제 값에 가까워지지만, DistributedSampler가 채운 중복 3개 때문에 정확히 같지는 않다. 논문 표에 넣을 최종 평가는 중복을 없앤다. 방법은 평가만 단일 프로세스로 하거나, 표본 번호와 함께 예측을 모아(`dist.all_gather_object` 등) 중복을 제거하거나, 마지막 batch를 따로 처리하는 것이다.

::quiz{id="rank0-metric"}

**rank 0 저장.** DDP에서는 모든 rank의 파라미터가 같으므로 checkpoint는 rank 0 하나만 저장하면 된다. 모든 rank가 같은 경로에 동시에 쓰면 파일이 깨질 수 있다. 저장할 것은 `model.module.state_dict()`다. DDP로 감싼 `model.state_dict()`의 키에는 `module.` 접두사가 붙어 단일 GPU 모델에 바로 불러올 수 없다. 저장이 끝나기 전에 다른 rank가 그 파일을 읽지 않도록 `dist.barrier()`로 기다린다. checkpoint에 들어갈 내용(optimizer·scheduler·난수 상태, step)은 17.6장의 규칙을 그대로 따르고, 여기에 world size와 유효 batch를 함께 기록한다.

```python
# 집필 시점 예시 (DDP): rank 0만 저장하고 모두 기다린다
if dist.get_rank() == 0:
    torch.save({"model": model.module.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "world_size": dist.get_world_size()}, ckpt_path)
dist.barrier()
```

**sharded checkpoint.** FSDP에서는 각 rank가 파라미터와 optimizer 상태의 일부만 들고 있다. rank 0 하나가 전체를 모으면 그 rank의 메모리가 넘칠 수 있으므로, 각 rank가 자기 몫을 동시에 저장하는 **sharded checkpoint**를 쓴다. 집필 시점 PyTorch에서는 `torch.distributed.checkpoint`(DCP)의 `save`·`load`와 `torch.distributed.checkpoint.state_dict`의 `get_state_dict`·`set_state_dict`가 이 일을 한다. sharded checkpoint는 저장할 때와 다른 GPU 수로 불러올 수 있도록 설계되어 있지만, 실제로 그렇게 불러온 뒤 같은 손실이 나오는지 확인한다.

## 통신 병목

**왜 필요한가.** 분산 학습의 속도 향상은 계산을 나눈 만큼에서 통신에 쓴 시간을 뺀 것이다. 통신이 계산보다 오래 걸리면 GPU를 늘릴수록 오히려 느려질 수 있다.

**ring all-reduce의 통신량.** GPU를 고리로 잇고 배열을 $W$개 청크로 나누어, 청크를 이웃에게 넘기며 더하는(reduce-scatter) $W-1$단계와 완성된 청크를 돌리는(all-gather) $W-1$단계로 all-reduce를 할 수 있다. 그러면 GPU 하나가 보내는 양은 배열 크기 $S$에 대해

$$
2\,\frac{W - 1}{W}\,S
$$

로, GPU 수가 늘어도 $2S$를 넘지 않는다. 통신 시간의 하한은 이 양을 연결 대역폭으로 나눈 값이다. 연습문제에서 이 알고리즘을 직접 구현한다. 실제 통신 라이브러리(NCCL 등)는 연결 구조에 따라 ring 외의 알고리즘도 고른다.

```python {exec="browser" session="dist"}
P = 350e6                                          # 파라미터 수
grad_bytes = P * 4                                 # fp32 gradient 한 벌, byte
links_gbps = {"연결 A (가정 10 GB/s)": 10e9, "연결 B (가정 100 GB/s)": 100e9}
compute_s = 0.30                                   # GPU당 forward+backward 시간 (가정)
for W in [2, 8, 64]:
    sent = 2 * (W - 1) / W * grad_bytes            # GPU 하나가 보내는(=받는) byte
    for name, bw in links_gbps.items():
        t = sent / bw
        print(f"W={W:2d} {name}: 보내는 양 {sent / 1e9:5.2f} GB, 통신 {t * 1e3:5.0f} ms = 계산의 {t / compute_s:4.0%}")
```

```output
W= 2 연결 A (가정 10 GB/s): 보내는 양  1.40 GB, 통신   140 ms = 계산의  47%
W= 2 연결 B (가정 100 GB/s): 보내는 양  1.40 GB, 통신    14 ms = 계산의   5%
W= 8 연결 A (가정 10 GB/s): 보내는 양  2.45 GB, 통신   245 ms = 계산의  82%
W= 8 연결 B (가정 100 GB/s): 보내는 양  2.45 GB, 통신    24 ms = 계산의   8%
W=64 연결 A (가정 10 GB/s): 보내는 양  2.76 GB, 통신   276 ms = 계산의  92%
W=64 연결 B (가정 100 GB/s): 보내는 양  2.76 GB, 통신    28 ms = 계산의   9%
```

이 표의 대역폭과 계산 시간은 자릿수를 보이기 위한 가정이다. 실제 값은 장비의 사양서와 측정으로 정한다. 읽을 것은 두 가지다. 첫째, GPU를 2대에서 64대로 늘려도 GPU당 보내는 양은 1.40 GB에서 2.76 GB로 2배 안쪽에서 멈춘다. ring all-reduce의 성질이다. 둘째, 같은 모델에서도 연결이 10배 느리면 통신이 계산의 5–9%에서 47–92%로 커진다. DDP는 통신을 backward와 겹치므로 실제로 늘어나는 시간은 겹치지 못한 부분이지만, 통신이 계산보다 길면 겹쳐도 숨길 수 없다. 대역폭은 같은 서버 안의 GPU 사이(전용 GPU 연결이 있는지, PCIe만 있는지)와 서버 사이(네트워크)에서 크게 다르다. `nvidia-smi topo -m`(집필 시점 예시)은 서버 안 GPU들의 연결 구조를 보여 준다.

::widget{name="ring-allreduce"}

**병목을 줄이는 방법.** gradient의 dtype을 bf16으로 줄여 통신량을 반으로 줄이거나, accumulation과 `no_sync()`로 통신 횟수를 줄이거나, 같은 일을 더 적은 GPU·더 빠른 연결 안에서 하도록 배치한다. 어느 방법이든 처리량(samples/s)을 GPU 수별로 재서 확인한다. 이상적인 확장이면 GPU를 $W$배로 늘릴 때 처리량도 $W$배가 되고, 실제 처리량을 이상적 처리량으로 나눈 값을 **scaling efficiency**라 부른다.

**디버깅.** 분산 학습의 오류는 대개 멈춤(hang)으로 나타난다. 한 rank가 collective 호출에 들어갔는데 다른 rank가 다른 분기를 타서 그 호출에 오지 않으면, 모두가 기다리다 timeout까지 멈춘다. rank 0에서만 실행되는 `if` 문 안에 `dist.barrier()`나 `all_reduce`를 넣은 코드가 전형적인 원인이다. 집필 시점 PyTorch에서 도움이 되는 환경 변수는 다음과 같다. 이름과 의미는 버전마다 바뀌므로 그 버전의 문서를 확인한다.

```bash
# 집필 시점 예시
export TORCH_DISTRIBUTED_DEBUG=DETAIL   # collective 호출의 불일치를 검사하고 자세히 기록
export NCCL_DEBUG=INFO                  # NCCL의 초기화·연결 정보를 출력 (GPU)
export NCCL_SOCKET_IFNAME=eth0          # 노드 간 통신에 쓸 네트워크 인터페이스 지정 (클러스터에 맞춘다)
```

:::note[분야 예시]
입자물리 검출기 시뮬레이션이나 단백질 구조 데이터처럼 표본마다 크기가 크게 다른 데이터는 rank마다 batch의 계산량이 달라진다. all-reduce는 가장 느린 rank를 기다리므로, 한 rank에 큰 표본이 몰린 step에서는 나머지 GPU가 논다. 크기가 비슷한 표본끼리 batch를 만들거나(bucketing), rank 사이의 step 시간 분포를 기록해 이 대기 시간을 확인한다.
:::

## 실습

단일 GPU 학습을 DDP로 바꾸고, 프로세스 수에 따른 학습 결과를 비교한다. 코드는 `labs/experiment-operations/distributed-training/ddp_train.py`다. `DistributedSampler`와 `set_epoch`, DDP, 손실의 분산 집계를 갖춘 작은 MLP 회귀 학습이다. 전역 batch 64를 고정하고 rank당 batch를 64/world size로 나누는 것이 기본이고, `--per-rank-batch 64`를 주면 rank당 batch를 고정한다. 이 환경에는 GPU가 없으므로 CPU의 gloo backend로 `torchrun --standalone --nproc-per-node=1`과 `=2`로 실행했다. 집필 시점의 실행 결과는 다음과 같다.

```output
world 1  rank당 batch 64  유효 batch 64
world 1  epoch 1  step   8  학습 손실 0.435448
world 1  epoch 5  step  40  학습 손실 0.079778
world 1  가중치 합 -5.030210  노름 3.565958
world 2  rank당 batch 32  유효 batch 64
world 2  epoch 1  step   8  학습 손실 0.435448
world 2  epoch 5  step  40  학습 손실 0.079778
world 2  가중치 합 -5.030210  노름 3.565958
world 2  rank당 batch 64  유효 batch 128
world 2  epoch 1  step   4  학습 손실 0.538765
world 2  epoch 5  step  20  학습 손실 0.134671
world 2  가중치 합 -4.898803  노름 3.533322
```

(epoch 2–4 줄은 생략했다.) 결과를 다음 순서로 읽는다.

1. **유효 batch를 고정하면 프로세스 수와 무관하게 같은 학습이다.** world 1과 world 2의 손실과 가중치가 소수점 여섯째 자리까지 같다. DistributedSampler는 모든 rank가 같은 순서를 공유하고 rank $r$이 $r, r+W, \dots$번째를 가져가므로, world 2의 한 step에서 두 rank가 보는 표본 32개씩을 합치면 world 1의 한 step이 보는 64개와 같다. 그리고 DDP의 평균 gradient는 그 64개의 gradient와 같다.
2. **rank당 batch를 고정하면 다른 학습이다.** 유효 batch가 128로 커져 epoch당 step이 8에서 4로 줄었고, 같은 학습률·같은 epoch 수에서 손실이 0.0798 대신 0.1347에 멈췄다. "GPU를 늘렸더니 성능이 떨어졌다"는 보고의 상당수가 이 경우다.
3. **처리량은 GPU에서 잰다.** CPU 프로세스 2개의 시간은 실제 다중 GPU의 처리량을 대표하지 않는다. GPU가 있다면 `--backend nccl`로 1·2·4 GPU에서 samples/s와 step당 시간을 재고 scaling efficiency를 계산한다(22.4장의 측정 원칙: warm-up 뒤, `torch.cuda.synchronize()` 포함).

이어서 ring all-reduce를 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="ring-all-reduce"}

## 실패 사례

- **각 rank의 metric을 집계하지 않고 rank 0 값만 보고하는 것.** rank 0은 검증 데이터의 $1/W$만 본다. 위 예에서는 0.009만큼 높게 보고되었고, 표본이 적거나 rank 사이의 데이터 분포가 다르면 차이는 더 커진다. (맞힌 수, 표본 수)를 all-reduce로 합한 뒤 나누고, 최종 평가는 중복 없이 한다.
- **GPU 수를 바꾸면서 유효 batch와 학습률을 함께 바꾸지 않는 것.** rank당 batch를 그대로 두고 GPU만 늘리면 유효 batch가 커지고 step 수가 준다. 단일 GPU에서 튜닝한 학습률·warm-up(step 단위)이 맞지 않게 된다.
- **`set_epoch`를 부르지 않는 것.** 매 epoch 같은 순서로 학습해도 오류가 나지 않는다. 학습 곡선만으로는 알아차리기 어렵다.
- **rank 0에서만 실행되는 분기 안에서 collective를 호출하는 것.** 다른 rank가 그 호출에 오지 않아 학습이 멈춘다. collective(`all_reduce`, `barrier`, `broadcast`)는 모든 rank가 같은 순서로 호출해야 한다.
- **모든 rank가 같은 경로에 checkpoint를 쓰는 것.** 동시에 쓰면 파일이 깨질 수 있다. DDP는 rank 0만, FSDP는 sharded checkpoint로 저장한다.

## 통과 기준

- GPU 수가 $W$배가 될 때 달라지는 유효 batch·학습률·epoch당 step 수·GPU당 통신량을 식으로 계산하고, 같은 학습을 유지하려면 무엇을 바꿔야 하는지 설명한다.
- `ddp_train.py`를 프로세스 1개와 2개로 실행해 유효 batch를 고정하면 같은 결과가 나오는 것과 rank당 batch를 고정하면 결과가 달라지는 것을 재현하고, 그 이유를 DistributedSampler의 분배 규칙으로 설명한다.
- 분산 평가에서 (맞힌 수, 표본 수)를 모든 rank에서 모아 metric을 계산하고, rank 0만의 값과 비교해 보고한다.
- 연습문제 `ring-all-reduce`의 테스트를 통과한다.

## 참고 문헌

- Goyal 외 (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677. — 선형 스케일링 규칙과 warm-up
- PyTorch 공식 문서 — `torch.distributed`(collective 통신, backend), `DistributedDataParallel`, `torchrun`, `DistributedSampler`, FSDP, `torch.distributed.checkpoint`. pytorch.org.
