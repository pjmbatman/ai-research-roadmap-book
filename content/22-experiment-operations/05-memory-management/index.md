---
id: experiment-operations/memory-management
title: "GPU 메모리 관리"
title_en: "GPU Memory Management"
tier: core
status: draft
prerequisites: [experiment-operations/profiling-and-efficiency]
---

# GPU 메모리 관리

학습 스크립트를 처음 GPU에 올리면 가장 먼저 만나는 오류는 대개 `CUDA out of memory`다. 이때 많은 사람이 batch 크기를 절반으로 줄여 다시 돌린다. 학습은 돌아가지만, 그사이 유효 batch 크기와 학습률의 관계가 바뀌었고, 무엇이 메모리를 차지했는지는 여전히 모른다. 다음 모델, 다음 GPU에서 같은 일을 반복한다.

이 장은 GPU 메모리를 **byte 단위의 예산**으로 다룬다. 학습 중 메모리를 차지하는 것들을 항목별로 세는 식을 세우고, PyTorch로 실제 값을 측정해 식과 비교한다. 그다음 activation checkpointing, gradient accumulation, mixed precision, offloading, 8-bit optimizer가 각각 **어느 항목을** 줄이는지, 대가로 무엇을 내주는지 정리한다. 목표는 주어진 GPU에서 돌릴 수 있는 가장 큰 설정을 추측이 아니라 계산과 측정으로 정하는 것이다. GPU와 VRAM의 구조는 [5.5장](../../05-python-and-computing/05-computer-architecture/index.md), fp16·bf16의 비트 구조는 [9.1장](../../09-numerical-methods/01-floating-point/index.md), 학습 루프의 각 줄은 [17.5장](../../17-deep-learning-pytorch/05-training-loop/index.md)에서 다뤘다.

## 학습 목표

- 학습 중 메모리 사용을 계산한다
- 메모리 절약 기법을 사용한다

## 메모리 회계

**왜 필요한가.** GPU는 VRAM에 있는 데이터만 계산할 수 있고(5.5장), VRAM은 GPU마다 고정된 크기다. 학습 한 step 동안 무엇이 얼마나 VRAM을 차지하는지 모르면, OOM이 났을 때 무엇을 줄여야 할지 고를 수 없다.

**정의.** 학습 중 GPU 메모리를 차지하는 것은 다섯 가지로 나눌 수 있다.

| 항목 | 내용 | 크기를 정하는 것 |
|---|---|---|
| 파라미터 | 모델 가중치 | 파라미터 수 × dtype byte |
| gradient | 파라미터마다 하나씩 생기는 `.grad` | 파라미터 수 × dtype byte |
| optimizer 상태 | momentum, Adam의 $\mathbf{m}$·$\mathbf{v}$ 등 | 파라미터 수 × 상태 개수 × byte |
| 활성값(activation) | forward가 backward를 위해 저장하는 중간 텐서 | batch 크기 × 모델 구조 × byte |
| 임시 버퍼 | 연산 중 잠깐 쓰는 작업 공간, 단편화로 비는 공간 | 연산 구현·allocator |

앞의 세 항목은 batch 크기와 무관한 **정적 메모리**다. 파라미터 수를 $P$, 파라미터·gradient·optimizer 상태의 원소당 byte를 $b_p, b_g, b_s$라 하면

$$
M_{\text{static}} = P\,(b_p + b_g + b_s)
$$

이다. 활성값은 batch 크기 $B$에 비례한다. 샘플 하나가 저장하게 만드는 byte를 $a$라 하면 $M_{\text{act}} \approx B\,a$이고, 학습이 돌아가려면 대략

$$
M_{\text{static}} + B\,a + M_{\text{buffer}} \le M_{\text{GPU}}
$$

여야 한다. $M_{\text{buffer}}$는 임시 버퍼와 CUDA context처럼 식으로 세기 어려운 몫이라, 측정해서 여유로 남겨 둔다.

**직관.** Adam으로 fp32 학습을 하면 파라미터 하나가 4 byte(값) + 4 byte(gradient) + 8 byte($m$과 $v$)로 **16 byte**를 차지한다. 모델 파일 크기의 네 배다. 70억 파라미터 모델은 가중치만 fp32로 약 26 GiB이지만, Adam 학습에는 활성값을 빼고도 약 104 GiB가 필요하다. "모델이 GPU에 들어간다"와 "모델을 GPU에서 학습할 수 있다"는 전혀 다른 말이다.

```python {exec="browser" session="mem"}
import numpy as np

def mlp_params(widths):
    return sum(i * o + o for i, o in zip(widths[:-1], widths[1:]))   # 가중치 + bias

recipes = {                      # 파라미터 하나당 byte: (파라미터, gradient, optimizer 상태)
    "fp32 + SGD":              (4, 4, 0),
    "fp32 + SGD(momentum)":    (4, 4, 4),
    "fp32 + Adam":             (4, 4, 8),   # m, v
    "autocast(bf16) + Adam":   (4, 4, 8),   # 파라미터·gradient는 fp32 그대로
    "bf16 + fp32 사본 + Adam":  (2, 2, 12),  # 사본 4 + m 4 + v 4
}
n = mlp_params([1024, 4096, 4096, 4096, 1024])
print(f"파라미터 {n:,}개")
for name, (p, g, s) in recipes.items():
    print(f"{name:22s} {p + g + s:2d} byte/파라미터 → {n * (p + g + s) / 2**30:5.2f} GiB")
print(f"70억 파라미터 × 16 byte = {7e9 * 16 / 2**30:.0f} GiB")
```

```output
파라미터 41,956,352개
fp32 + SGD              8 byte/파라미터 →  0.31 GiB
fp32 + SGD(momentum)   12 byte/파라미터 →  0.47 GiB
fp32 + Adam            16 byte/파라미터 →  0.63 GiB
autocast(bf16) + Adam  16 byte/파라미터 →  0.63 GiB
bf16 + fp32 사본 + Adam  16 byte/파라미터 →  0.63 GiB
70억 파라미터 × 16 byte = 104 GiB
```

표에서 세 가지를 읽는다.

- **optimizer 선택이 정적 메모리를 정한다.** 같은 모델이라도 SGD는 8 byte, Adam은 16 byte다.
- **PyTorch의 autocast는 정적 메모리를 줄이지 않는다.** autocast는 파라미터를 fp32로 둔 채 연산할 때만 bf16으로 바꾼다. 그래서 파라미터·gradient·optimizer 상태는 fp32 그대로다. autocast가 줄이는 것은 활성값과 연산 시간이다(아래 "절약 기법").
- **가중치를 bf16으로 들고 있어도 16 byte가 될 수 있다.** 대규모 혼합 정밀도 학습은 갱신의 정밀도를 지키려고 fp32 사본(master weights)을 따로 둔다(Micikevicius 외, 2018; 9.1장). bf16 가중치 2 + bf16 gradient 2 + fp32 사본 4 + $m$ 4 + $v$ 4 = 16 byte다. 정밀도를 낮췄는데 정적 메모리가 그대로인 이유다.

::quiz{id="adam-bytes-per-param"}

**활성값.** backward는 forward의 중간값이 필요하다. 예를 들어 $\mathbf{y} = \mathbf{W}\mathbf{x}$의 가중치 gradient는 $\partial L/\partial \mathbf{W} = (\partial L/\partial \mathbf{y})\,\mathbf{x}^\top$이므로 입력 $\mathbf{x}$를 backward 때까지 들고 있어야 한다(8.4장). 그래서 각 층은 forward에서 필요한 텐서를 저장해 두고, 이 텐서들이 활성값 메모리가 된다. 크기는 batch 크기, 층의 폭, 시퀀스 길이에 비례한다. 큰 모델에서는 정적 메모리가, 큰 batch·긴 시퀀스·고해상도 입력에서는 활성값이 메모리를 지배한다.

:::note[분야 예시]
3차원 격자 위의 기후·유체 시뮬레이션을 학습하는 신경망은 파라미터가 적어도 입력 하나가 $(C, D, H, W)$ 크기의 큰 텐서라 활성값이 메모리를 지배한다. 반대로 단백질·분자 데이터의 큰 사전학습 모델을 미세조정할 때는 정적 메모리가 먼저 문제다. 같은 "OOM"이라도 줄여야 할 항목이 다르므로, 먼저 어느 항목이 큰지 계산한다.
:::

## 측정

**왜 필요한가.** 식은 모델의 구조에서 나오고, 실제 메모리는 구현에서 나온다. 어떤 층이 무엇을 저장하는지, allocator가 얼마를 예비로 잡는지는 측정해야 안다. 계산과 측정이 다르면 그 차이가 곧 알아야 할 정보다.

**CPU에서 측정할 수 있는 것.** 파라미터·gradient·optimizer 상태는 텐서의 `numel() * element_size()`로 정확히 셀 수 있다. backward를 위해 저장되는 텐서는 `torch.autograd.graph.saved_tensors_hooks`로 가로챌 수 있다. 아래 셀은 ReLU MLP에서 두 가지를 세고, 활성값을 식과 비교한다.

```python {exec="python"}
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
widths = [256, 1024, 1024, 1024, 256]
layers = [m for i, o in zip(widths[:-1], widths[1:]) for m in (nn.Linear(i, o), nn.ReLU())]
model = nn.Sequential(*layers[:-1])                       # 마지막 ReLU는 뺀다
param_ptrs = {p.data_ptr() for p in model.parameters()}

def saved_bytes(fn, x):
    """forward가 backward를 위해 저장하는 텐서(파라미터 제외)의 byte 수. 같은 텐서는 한 번만 센다."""
    seen = {}
    def pack(t):
        if t.data_ptr() not in param_ptrs:
            seen[t.data_ptr()] = t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = fn(x).sum()
    loss.backward()
    return sum(seen.values())

x = torch.randn(64, widths[0])                            # (B, D)
ckpt = lambda h: checkpoint(model[4:], checkpoint(model[:4], h, use_reentrant=False), use_reentrant=False)
print("샘플당 저장 byte: 그대로", saved_bytes(model, x) // 64, "/ checkpointing", saved_bytes(ckpt, x) // 64)
print("식 4 × (Linear 입력 폭의 합) =", 4 * sum(widths[:-1]))
opt = torch.optim.Adam(model.parameters())
opt.step()
nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
print("파라미터", nbytes(model.parameters()), " gradient", nbytes(p.grad for p in model.parameters()),
      " Adam 상태", nbytes(v for s in opt.state.values() for v in s.values()))
```

```output
샘플당 저장 byte: 그대로 13312 / checkpointing 5120
식 4 × (Linear 입력 폭의 합) = 13312
파라미터 10499072  gradient 10499072  Adam 상태 20998176
```

출력을 하나씩 읽는다.

- **활성값 식이 측정과 정확히 같다.** 각 Linear는 입력을 저장하고, ReLU는 출력을 저장한다. 그런데 ReLU의 출력은 다음 Linear의 입력과 **같은 텐서**라 메모리는 한 번만 쓴다. 그래서 샘플당 활성값은 $4 \times (256 + 1024 \times 3) = 13312$ byte다. 같은 텐서를 두 번 세면 식이 실제보다 커진다.
- **checkpointing을 쓰면 저장하는 것이 구간의 입력뿐이다.** 모델을 두 구간으로 나누어 checkpointing을 걸자 샘플당 $4 \times (256 + 1024) = 5120$ byte, 즉 두 구간의 입력만 남았다(다음 절).
- **Adam 상태는 정확히 파라미터의 두 배에 32 byte가 더해진다.** $m$과 $v$가 파라미터와 같은 크기이고, 파라미터 텐서 8개마다 step 카운터(4 byte 텐서)가 하나씩 있다. 그리고 이 상태는 **첫 `step()`에서 만들어진다.** 첫 optimizer step에서 OOM이 나는 이유다.

**GPU에서 측정하기.** CUDA가 있으면 PyTorch의 메모리 통계를 쓴다. 이 환경에는 GPU가 없으므로 아래는 실행 결과 없이 쓰는 **집필 시점 예시**다. 함수 이름과 인자는 쓰는 PyTorch 버전의 문서(CUDA semantics, `torch.cuda` 메모리 관리)에서 확인한다.

```python
# 집필 시점 예시 (CUDA 필요): 한 step의 최대 할당량을 잰다
torch.cuda.reset_peak_memory_stats()
loss = loss_fn(model(xb), yb)
loss.backward()
opt.step()
opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print("현재 할당", torch.cuda.memory_allocated() / 2**30, "GiB")
print("최대 할당", torch.cuda.max_memory_allocated() / 2**30, "GiB")
print("예약(캐시 포함)", torch.cuda.memory_reserved() / 2**30, "GiB")
print(torch.cuda.memory_summary(abbreviated=True))
```

세 숫자는 서로 다르다.

- **allocated**: 지금 살아 있는 텐서가 차지한 byte. 식으로 계산한 값과 비교할 대상이다.
- **max allocated**: 마지막 `reset_peak_memory_stats()` 이후의 최댓값. OOM은 평균이 아니라 **최댓값**에서 나므로 이 값을 본다. 대개 backward 중이나 첫 `opt.step()`에서 최대가 된다.
- **reserved**: PyTorch의 caching allocator가 GPU 드라이버에게서 받아 두고 재사용하는 메모리. 텐서를 해제해도 드라이버에 돌려주지 않고 캐시로 들고 있으므로 allocated보다 크다.

`nvidia-smi`(2.2장)가 보여 주는 프로세스 메모리는 reserved에 CUDA context 등이 더해진 값이라 이보다 더 크다. 그래서 `nvidia-smi`의 숫자로 "텐서가 이만큼 쓴다"고 판단하면 안 된다. `torch.cuda.empty_cache()`는 캐시 가운데 쓰지 않는 블록을 드라이버에 돌려줄 뿐 살아 있는 텐서를 줄이지 않으므로, 같은 프로세스의 OOM을 고치지 못한다.

::quiz{id="nvidia-smi-vs-allocated"}

**memory snapshot.** 최댓값만으로 "무엇이" 메모리를 차지했는지 모를 때는 할당·해제 기록을 남긴다. 집필 시점 PyTorch에는 할당마다 호출 스택을 기록하고 파일로 내보내는 기능이 있고, 내보낸 파일을 pytorch.org/memory_viz의 뷰어에 올리면 시간에 따른 메모리 사용과 각 블록을 만든 코드 줄을 볼 수 있다. 이름 앞의 밑줄이 뜻하듯 공개 API로 고정된 것이 아니므로, 쓰기 전에 그 버전의 문서를 확인한다.

```python
# 집필 시점 예시 (CUDA 필요): 몇 step의 할당 기록을 파일로 남긴다
torch.cuda.memory._record_memory_history(max_entries=100_000)
for _ in range(3):
    train_step()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)    # 기록을 끈다
```

**연구에서의 의미.** 측정은 같은 step에서, warm-up 뒤에, 같은 입력 크기로 한다(22.4장의 측정 원칙과 같다). 첫 step은 optimizer 상태 생성, cuDNN의 알고리즘 탐색, 캐시 확보가 겹쳐 다른 step과 다르다. 입력 크기가 batch마다 바뀌는 데이터(가변 길이 시퀀스, 크기가 다른 그래프)는 **가장 큰 입력**에서 최댓값을 재야 한다. 평균 크기에서 잰 값은 학습 도중 가끔 오는 큰 batch에서 OOM을 막지 못한다.

## 절약 기법

메모리 절약 기법은 각각 다른 항목을 줄이고, 다른 대가를 치른다. 한눈에 보면 다음과 같다.

| 기법 | 줄이는 항목 | 대가 | 수학적으로 같은 학습인가 |
|---|---|---|---|
| activation checkpointing | 활성값 | forward 재계산(계산 시간 증가) | 같다(재계산 순서에 따른 반올림 차이 정도) |
| gradient accumulation | 활성값(micro-batch에 비례) | step당 시간 증가, BatchNorm 통계 변화 | 손실이 표본 평균이고 BatchNorm이 없으면 같다 |
| mixed precision(autocast) | 활성값, 연산 시간 | 수치 정밀도 | 근사적으로 같다(검증 필요) |
| offloading | 정적 메모리(옵티마이저 상태 등) | CPU–GPU 전송 시간 | 같다 |

**activation checkpointing.** 왜 필요한가: 깊은 모델에서는 모든 층의 활성값을 backward까지 들고 있어야 한다. 정의: 일부 지점(checkpoint)의 활성값만 저장하고, 나머지는 backward 때 해당 구간의 forward를 **다시 계산**해 만든다. gradient checkpointing이라고도 한다. 직관: 계산 시간을 메모리와 바꾸는 거래다.

$n$개의 같은 크기 층을 길이 $k$인 구간으로 나누면, forward 동안 구간 경계 $\lceil n/k \rceil$개만 저장하고, backward에서 한 구간을 재계산하는 동안 그 구간의 $k$개를 더 들고 있다. 그래서 최대 저장량은 층 하나의 활성값을 단위로

$$
S(k) \approx \frac{n}{k} + k
$$

이고, $k = \sqrt{n}$ 근처에서 최소 $2\sqrt{n}$이 된다. 재계산하는 양은 forward 약 한 번이다. backward가 forward의 약 두 배 계산이라고 보면, step 시간은 최대 약 33% 늘어난다.

```python {exec="browser" session="mem"}
n_layers = 48                                    # 같은 크기의 층 48개
print(" 구간 길이 k   저장하는 층 수   추가 계산(fwd+bwd 대비)")
for k in [1, 2, 4, 6, 8, 16, 48]:
    if k == 1:
        stored, extra = n_layers, 0.0            # checkpointing 없음
    else:
        stored = int(np.ceil(n_layers / k)) + k  # 구간 경계 + backward 때 다시 만든 한 구간
        extra = (1 - 1 / k) / 3                  # 다시 하는 forward ≈ (1 - 1/k)회, backward ≈ forward 2회
    print(f"{k:8d} {stored:14d} {extra:16.0%}")
```

```output
 구간 길이 k   저장하는 층 수   추가 계산(fwd+bwd 대비)
       1             48               0%
       2             26              17%
       4             16              25%
       6             14              28%
       8             14              29%
      16             19              31%
      48             49              33%
```

48층에서 구간 길이 6–8이면 저장량이 48에서 14로 3분의 1 아래로 준다. 구간을 너무 길게 잡으면(48) 경계는 하나뿐이지만 backward 때 구간 전체를 다시 만들어야 해서 오히려 절약이 없다. 이 표는 모든 층이 같은 크기라는 단순화이고, 실제로는 활성값이 큰 층(attention, 큰 합성곱) 주변을 구간으로 묶는 것이 효과적이다. PyTorch에서는 `torch.utils.checkpoint.checkpoint`로 구간(보통 transformer block 하나)을 감싼다. 앞의 측정 셀에서 본 것처럼, checkpointing은 저장하는 텐서만 바꾸고 계산 결과는 바꾸지 않는다.

::widget{name="checkpoint-tradeoff"}

::quiz{id="checkpoint-segment"}

**gradient accumulation.** micro-batch $b$로 $k$번 forward·backward를 하고 gradient를 모아 한 번 갱신하면 유효 batch는 $bk$다(10.4장, 17.5장). 메모리 관점에서 중요한 것은 **활성값이 micro-batch에 비례**하고 정적 메모리는 그대로라는 점이다. 그래서 정적 메모리가 GPU를 거의 다 차지한 경우에는 accumulation으로 해결되지 않는다. 대가는 step당 시간이다. 작은 micro-batch는 GPU의 연산 강도를 낮춰(5.5장) 같은 표본 수를 처리하는 데 더 오래 걸릴 수 있다. BatchNorm이 있는 모델은 통계를 micro-batch에서 계산하므로 큰 batch와 같은 학습이 아니다.

예산에서 micro-batch와 누적 횟수를 정하는 과정을 계산해 본다. GPU 크기와 샘플당 활성값은 가정한 값이다. 실제로는 앞 절처럼 측정해서 넣는다.

```python {exec="browser" session="mem"}
gpu_gib = 24.0                                   # 가상의 GPU
static_gib = 350e6 * 16 / 2**30                  # 3.5억 파라미터, fp32 + Adam
reserve_gib = 2.0                                # CUDA context·단편화 여유(가정)
act_per_sample_gib = {"fp32": 0.40, "autocast bf16": 0.22, "bf16 + checkpointing": 0.06}  # 측정했다고 가정
target = 256                                     # 목표 유효 batch
print(f"정적 메모리 {static_gib:.2f} GiB, 활성값에 쓸 수 있는 메모리 {gpu_gib - static_gib - reserve_gib:.2f} GiB")
for name, a in act_per_sample_gib.items():
    max_b = int((gpu_gib - static_gib - reserve_gib) // a)
    micro = max(d for d in range(1, max_b + 1) if target % d == 0)   # target의 약수 중 가장 큰 것
    print(f"{name:22s} 최대 {max_b:3d} → micro-batch {micro:3d} × 누적 {target // micro:2d}")
```

```output
정적 메모리 5.22 GiB, 활성값에 쓸 수 있는 메모리 16.78 GiB
fp32                   최대  41 → micro-batch  32 × 누적  8
autocast bf16          최대  76 → micro-batch  64 × 누적  4
bf16 + checkpointing   최대 279 → micro-batch 256 × 누적  1
```

세 설정 모두 유효 batch는 256으로 같다. 그래서 학습률과 스케줄을 그대로 둘 수 있다. 차이는 step당 시간과 수치 정밀도다. micro-batch를 목표 batch의 약수로 고른 것은 누적 횟수를 정수로 맞추기 위해서다.

**mixed precision.** 9.1장에서 본 것처럼 bf16은 fp32와 같은 지수 범위를 가지면서 2 byte다. PyTorch의 `torch.autocast`는 행렬 곱 같은 연산을 bf16(또는 fp16)으로 실행하고, 그 연산이 backward를 위해 저장하는 텐서도 bf16이 된다. 그래서 **활성값이 대략 절반**으로 준다. 앞의 표에서 본 것처럼 파라미터·gradient·optimizer 상태는 fp32 그대로다. 실습에서 측정하겠지만, autocast 아래에서는 연산에 쓰려고 만든 bf16 가중치 사본도 backward까지 저장된다. batch가 작아 활성값이 적은 모델에서는 이 사본 때문에 저장량이 오히려 늘 수 있다. 메모리가 줄었는지는 식이 아니라 측정으로 확인한다. fp16을 쓰면 gradient underflow를 막는 loss scaling이 필요하다(9.1장, 17.5장).

::quiz{id="autocast-saves-what"}

**offloading.** GPU에 다 들어가지 않는 것을 CPU RAM에 두고 필요할 때만 옮기는 방법이다. 가장 흔한 대상은 optimizer 상태다. optimizer 상태는 step에 한 번만 쓰이므로, CPU에 두고 갱신도 CPU에서 하면 GPU 메모리에서 파라미터당 8 byte(Adam)를 덜 수 있다. 대가는 PCIe 전송 시간(5.5장)과 CPU 계산 시간이다. 집필 시점 PyTorch의 FSDP(22.6장)는 파라미터와 gradient를 CPU로 내리는 옵션(`CPUOffload`, `CPUOffloadPolicy`)을 제공한다. offloading은 GPU를 기다리게 만들기 쉬우므로, 켠 뒤에는 step 시간을 함께 잰다(22.4장).

## 큰 모델 다루기

모델이 커서 정적 메모리 자체가 GPU를 넘으면, 활성값을 줄이는 기법으로는 해결되지 않는다. 남은 선택은 정적 메모리의 원소당 byte를 줄이거나, 여러 GPU에 나누어 담는 것(22.6장의 FSDP·ZeRO)이다. 여기서는 앞의 것을 다룬다.

**8-bit optimizer.** Adam의 $\mathbf{m}$과 $\mathbf{v}$는 파라미터당 8 byte로 정적 메모리의 절반이다. 이 상태를 8비트 정수로 저장하고, 갱신할 때만 실수로 복원하면 2 byte로 줄일 수 있다. 정수로 저장하려면 값의 범위를 정하는 **scale**이 필요하다. 가장 단순한 방법은 absmax 양자화로, 묶음 안에서 절댓값의 최댓값을 127에 맞춘다.

$$
s = \frac{\max_i \lvert x_i \rvert}{127}, \qquad q_i = \operatorname{round}(x_i / s) \in \{-127, \dots, 127\}, \qquad \hat{x}_i = s\,q_i.
$$

문제는 큰 값(outlier) 하나가 묶음 전체의 scale을 키운다는 점이다. 그러면 나머지 작은 값들은 $s/2$보다 작아져 모두 0으로 반올림된다. 그래서 텐서를 작은 **블록**으로 나누고 블록마다 scale을 따로 두는 블록별(block-wise) 양자화를 쓴다. 블록이 작을수록 outlier의 영향이 그 블록에 갇히지만, scale을 저장하는 비용이 는다.

```python {exec="browser" session="mem"}
rng = np.random.default_rng(0)
m = rng.normal(scale=1e-3, size=4096).astype(np.float32)   # Adam 1차 moment 흉내, (4096,)
m[::512] *= 100                                            # 드문 큰 값(outlier) 8개

def absmax_int8(x, block):
    xb = x.reshape(-1, block)                              # (블록 수, block)
    scale = np.abs(xb).max(axis=1, keepdims=True) / 127    # 블록마다 fp32 scale 하나
    q = np.round(xb / scale).astype(np.int8)               # -127..127
    return (q * scale).reshape(x.shape), 1 + 4 / block     # 복원값, 원소당 byte

for block in [4096, 256, 64]:
    deq, bpe = absmax_int8(m, block)
    rel = np.linalg.norm(deq - m) / np.linalg.norm(m)
    print(f"블록 {block:4d}: 원소당 {bpe:.3f} byte  상대오차 {rel:.4f}  0이 된 원소 {np.mean(deq == 0):.1%}")
```

```output
블록 4096: 원소당 1.001 byte  상대오차 0.1142  0이 된 원소 60.7%
블록  256: 원소당 1.016 byte  상대오차 0.0353  0이 된 원소 10.5%
블록   64: 원소당 1.062 byte  상대오차 0.0169  0이 된 원소 3.1%
```

텐서 전체에 scale 하나를 쓰면(블록 4096) 원소의 60%가 0이 된다. outlier 8개가 scale을 100배로 키웠기 때문이다. 블록을 64로 줄이면 scale 저장 비용은 원소당 0.06 byte에 불과하지만 0이 되는 원소는 3%로 준다. 실제 8-bit optimizer 구현은 이보다 정교한 양자화(값의 분포에 맞춘 비균일 눈금 등)를 쓰고, 일부 층은 32비트로 남긴다. 쓰기 전에 그 구현의 문서와, 같은 설정의 32비트 학습과 비교한 결과를 확인한다.

**양자화된 가중치.** 같은 생각을 가중치에 적용하면 추론 메모리가 준다(22.7장에서 int8 양자화를 직접 구현한다). 학습에서도 사전학습 모델의 가중치를 양자화해 고정하고, 그 위에 작은 adapter 파라미터만 fp32·bf16으로 학습하는 미세조정 방법이 널리 쓰인다. 이때 gradient와 optimizer 상태는 adapter에만 생기므로 정적 메모리가 크게 준다. 그러나 양자화된 가중치는 원래 가중치와 다르므로, 같은 데이터에서 전체 정밀도 모델과 성능을 비교해 차이를 보고한다.

:::note[분야 예시]
재료·화학 분야의 큰 사전학습 모델을 연구실 GPU 한 대로 미세조정하는 경우, 정적 메모리가 첫 장벽이다. 이때의 순서는 (1) Adam의 16 byte/파라미터로 정적 메모리를 계산해 GPU에 들어가는지 보고, (2) 들어가지 않으면 adapter 미세조정이나 8-bit optimizer처럼 정적 메모리를 줄이는 방법을 고르고, (3) 그다음에 활성값을 checkpointing과 bf16으로 줄여 batch를 정하는 것이다. 순서를 바꾸어 batch부터 줄이면 정적 메모리 문제를 풀지 못한 채 시간만 쓴다.
:::

## OOM 대응

OOM이 나면 다음 순서로 원인을 좁힌다. 앞 단계일수록 학습 결과를 바꾸지 않는다.

1. **어디서 나는지 확인한다.** 오류의 스택에서 forward·backward·`opt.step()` 중 어디인지 본다. 첫 `opt.step()`이면 optimizer 상태(정적 메모리), backward 중이면 활성값과 gradient가 최대인 시점이다. 학습이 한참 돌다가 나면 아래 2번이나 가변 크기 입력을 의심한다. 집필 시점 PyTorch의 OOM 오류 메시지는 할당하려던 크기와 이미 할당·예약된 양을 함께 알려 준다.
2. **새는 메모리를 찾는다.** 매 step `max_memory_allocated`가 조금씩 늘면 무언가가 계산 그래프를 붙잡고 있다. 흔한 원인은 `total_loss += loss`처럼 텐서를 누적해 그래프 전체를 살려 두는 것(`loss.item()`이나 `loss.detach()`로 누적한다), 평가를 `torch.no_grad()` 없이 하는 것, 출력 텐서를 리스트에 모아 두는 것이다(17.5장).
3. **단편화를 확인한다.** reserved가 allocated보다 훨씬 큰데 OOM이 나면, 빈 공간이 잘게 흩어져 큰 블록을 못 잡는 단편화일 수 있다. 입력 크기가 batch마다 바뀌면 생기기 쉽다. 집필 시점 PyTorch는 환경 변수로 allocator 설정을 바꿀 수 있다(예: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 최근 버전에서는 `PYTORCH_ALLOC_CONF`라는 이름도 쓴다). 변수 이름과 옵션은 버전마다 바뀌어 왔으므로 그 버전의 CUDA semantics 문서에서 확인한다.
4. **결과를 바꾸지 않는 기법부터 쓴다.** activation checkpointing, optimizer 상태 offloading처럼 수학적으로 같은 학습을 주는 기법을 먼저 적용한다.
5. **결과를 바꿀 수 있는 기법은 기록과 함께 쓴다.** bf16 autocast, gradient accumulation(BatchNorm이 있으면), 8-bit optimizer, 시퀀스 길이·해상도 축소는 학습을 바꿀 수 있다. 적용했다면 설정 파일과 실험 기록(22.1장)에 남기고, 바뀌기 전 설정과 짧은 학습으로 비교한다.
6. **batch를 줄여야 한다면 유효 batch를 지킨다.** micro-batch를 줄인 만큼 누적 횟수를 늘리면 유효 batch·학습률·스케줄의 step 수가 그대로다. 유효 batch까지 줄였다면 학습률과 스케줄을 다시 정해야 한다(10.4장).

공유 서버에서는 OOM이 내 코드가 아니라 같은 GPU를 쓰는 다른 프로세스 때문일 수도 있다. 실행 전에 `nvidia-smi`로 GPU가 비어 있는지 확인하고, 쓸 GPU만 보이게 한다(2.5장).

::quiz{id="oom-batch-halving"}

## 실습

모델의 메모리 사용량을 손으로 계산하고 측정값과 비교한 뒤, 절약 기법별 효과를 측정한다. 코드는 `labs/experiment-operations/memory-management/memory_lab.py`다. 층 폭 `[512, 2048, 2048, 2048, 2048, 512]`인 ReLU MLP에서 정적 메모리와 활성값을 손으로 계산하고, backward용 저장 텐서를 활성값과 가중치 사본으로 나누어 측정한다. CUDA가 있으면 설정마다 `torch.cuda.max_memory_allocated()`도 함께 출력한다. 집필 시점 CPU에서의 실행 결과는 다음과 같다.

```output
장치 cpu, 층 폭 [512, 2048, 2048, 2048, 2048, 512], batch 128, 파라미터 14,688,768개
손 계산: 파라미터+gradient+Adam   224.1 MiB, 활성값  4.25 MiB
그대로                    저장된 활성값  4.50 MiB  가중치 사본   0.0 MiB
batch 절반               저장된 활성값  2.25 MiB  가중치 사본   0.0 MiB
checkpointing(2구간)     저장된 활성값  1.50 MiB  가중치 사본   0.0 MiB
autocast bf16          저장된 활성값  2.38 MiB  가중치 사본  26.0 MiB
```

결과를 다음 순서로 읽는다.

1. **계산과 측정의 차이를 설명한다.** 손 계산 4.25 MiB와 측정 4.50 MiB의 차이 0.25 MiB는 손실 계산 `out.pow(2)`가 출력 $(128, 512)$를 fp32로 하나 더 저장한 몫이다($128 \times 512 \times 4$ byte = 0.25 MiB). 식에 넣지 않은 연산도 메모리를 쓴다.
2. **batch 절반은 활성값을 정확히 절반으로 줄인다.** 정적 메모리 224 MiB는 그대로다. 이 모델에서는 정적 메모리가 활성값보다 50배 크므로, batch를 줄여 OOM을 피하려는 시도는 거의 효과가 없다.
3. **checkpointing은 구간 입력만 남긴다.** 두 구간의 입력 $(512 + 2048) \times 128 \times 4$ byte = 1.25 MiB에 손실의 0.25 MiB를 더한 1.50 MiB다.
4. **autocast는 활성값을 거의 절반으로 줄였지만, bf16 가중치 사본 26 MiB를 새로 저장했다.** 첫 층은 입력이 gradient를 요구하지 않아 가중치가 필요 없으므로, 나머지 세 층($2048^2$)과 마지막 층($2048 \times 512$)의 bf16 사본이다. 이 작은 batch에서는 autocast가 저장량을 오히려 늘렸다. batch나 시퀀스가 커서 활성값이 가중치보다 훨씬 클 때 autocast의 메모리 이점이 나타난다.

GPU가 있다면 같은 스크립트로 최대 할당량을 함께 보고, 위 네 설정의 차이가 최대 할당량에서도 같은 크기로 나타나는지 확인한다. 차이가 다르면 그 차이를 만든 항목(임시 버퍼, allocator 캐시)을 memory snapshot으로 찾는다.

이어서 연습문제로 메모리 예산 계산기를 만든다. 테스트가 바로 채점한다.

::exercise{id="memory-budget"}

## 실패 사례

- **batch 크기만 줄여 OOM을 해결하고 학습 설정이 바뀐 것을 놓치는 것.** batch를 절반으로 줄이면 유효 batch가 절반이 되고, epoch당 step 수가 두 배가 되며, gradient 잡음이 커진다. 학습률·warm-up·스케줄은 예전 batch에 맞춰져 있다. 결과가 달라지면 "메모리를 줄였더니 성능이 떨어졌다"가 아니라 "다른 학습을 했다"는 뜻이다. 누적 횟수를 늘려 유효 batch를 지키거나, 학습 설정을 다시 정하고 기록한다.
- **`nvidia-smi`의 숫자로 텐서 메모리를 판단하는 것.** `nvidia-smi`는 allocator 캐시와 CUDA context를 포함한다. 텐서가 쓰는 양은 `memory_allocated`, 최댓값은 `max_memory_allocated`로 잰다.
- **mixed precision이 모든 메모리를 절반으로 줄인다고 기대하는 것.** autocast는 정적 메모리를 줄이지 않고, 가중치 사본을 저장하기도 한다. 정적 메모리가 지배하는 큰 모델에서는 효과가 작다.
- **평균 크기의 입력으로 메모리를 재는 것.** 가변 길이 데이터에서는 가장 긴 시퀀스, 가장 큰 그래프가 들어온 step에서 OOM이 난다. 최대 입력으로 측정하거나, 입력 크기의 상한을 정해 둔다.
- **step마다 손실 텐서를 누적해 그래프를 붙잡는 것.** 메모리가 step마다 조금씩 늘다가 수천 step 뒤에 OOM이 난다. 기록용 누적에는 `.item()`이나 `.detach()`를 쓴다.

## 통과 기준

- 주어진 모델·optimizer·정밀도에 대해 정적 메모리와 샘플당 활성값을 byte로 계산하고, `memory_lab.py` 방식으로 측정해 차이를 항목별로 설명한다.
- 주어진 GPU 메모리에서 학습 가능한 최대 micro-batch와, 목표 유효 batch를 맞추는 누적 횟수를 계산과 측정으로 결정한다.
- checkpointing·accumulation·autocast·offloading이 각각 어느 메모리 항목을 줄이고 무엇을 대가로 내는지 설명한다.
- 연습문제 `memory-budget`의 테스트를 통과한다.

## 참고 문헌

- Micikevicius 외 (2018). Mixed precision training. *ICLR 2018*. — 16비트 학습에서 fp32 가중치 사본과 loss scaling
- PyTorch 공식 문서 — CUDA semantics(caching allocator, 메모리 통계, allocator 설정), `torch.utils.checkpoint`, `torch.autograd.graph.saved_tensors_hooks`, memory snapshot. pytorch.org.
