---
id: experiment-operations/profiling-and-efficiency
title: "Profiling과 학습 효율화"
title_en: "Profiling and Training Efficiency"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/training-loop, python-and-computing/computer-architecture]
---

# Profiling과 학습 효율화

학습이 느리면 실험 수가 줄고, 실험 수가 줄면 연구가 느려진다. 그런데 "느리다"를 고치려는 시도의 상당수는 엉뚱한 곳을 고친다. GPU를 더 좋은 것으로 바꿨는데 데이터 로딩이 병목이라 그대로이거나, 모델을 열심히 최적화했는데 실제로는 step마다 checkpoint 디렉터리를 훑는 코드가 시간을 먹고 있었던 식이다. 또 빨라졌다고 기뻐한 변경이 사실은 정확도를 조용히 깎은 경우도 흔하다.

이 장은 학습 성능을 **정확히 재고**, **병목을 찾아**, **고친 뒤 결과가 같은지 확인하는** 순서를 다룬다. 5.4장에서 본 Python 수준의 profiling과 5.5장의 병목 분류·roofline 모형을 학습 루프에 적용하는 단계다. 메모리 부족과 메모리 사용량 관리는 22.5장에서, 여러 GPU로의 확장은 22.6장에서 다룬다.

## 학습 목표

- 학습 성능을 정확히 측정한다
- 병목을 찾아 개선한다

## 측정

**왜 필요한가.** 측정이 틀리면 그 뒤의 모든 판단이 틀린다. GPU 코드는 비동기로 돌아 엉뚱한 줄이 느려 보이고(5.5장), 첫 step은 초기화 비용 때문에 수십 배 느리고, 공유 서버에서는 다른 사람의 job 때문에 같은 코드의 시간이 실행마다 흔들린다.

**무엇을 재는가.** 학습 성능을 말하는 세 가지 양은 다르다.

| 양 | 정의 | 언제 쓰나 |
|---|---|---|
| wall-clock 시간 | 시작부터 끝까지 실제로 흐른 시간 | 실험 하나가 얼마나 걸리는가, 예산 계획 |
| 처리량(throughput) | 단위 시간에 처리한 샘플(또는 token) 수 | 학습 설정끼리의 속도 비교 |
| latency | 입력 하나(또는 batch 하나)를 처리하는 데 걸린 시간 | 추론, 대화형 응용(22.7장) |

처리량을 높이는 변경(batch 크기 키우기)이 latency를 늘리는 경우가 많으므로, 무엇을 개선하려는지 먼저 정한다. 그리고 학습에서 최종적으로 중요한 것은 처리량이 아니라 **목표 성능에 도달하는 시간**(time-to-accuracy)이다. batch 크기처럼 최적화 자체를 바꾸는 변경은 처리량만으로 평가할 수 없다.

**정확히 재는 규칙.**

1. **warm-up을 뺀다.** 첫 몇 step에는 메모리 할당, cuDNN 알고리즘 선택, `torch.compile`의 컴파일, 파일 캐시 채우기 같은 일회성 비용이 섞인다. 정상 상태를 잴 때는 빼고, 일회성 비용은 따로 기록한다.
2. **여러 번 재서 중앙값을 쓴다.** 한 번의 측정은 잡음이 크다. 공유 서버에서는 특히 그렇다. 반복 측정의 중앙값과 퍼짐을 함께 본다.
3. **GPU는 동기화한 뒤 잰다.** 측정 구간 앞뒤에서 `torch.cuda.synchronize()`를 부르거나 CUDA event로 GPU 쪽 시간을 잰다. 동기화 없이 재면 kernel을 대기열에 넣는 시간만 잰다(5.5장).
4. **같은 조건에서 비교한다.** 같은 기계, 같은 입력 크기, 같은 스레드 수, 비슷한 부하. 다른 날 다른 서버에서 잰 두 숫자는 비교하지 않는다.
5. **결과가 같은지도 확인한다.** 빨라진 코드가 같은 답을 내는지 먼저 본다.

이 규칙을 브라우저에서 확인한다. 시간은 기계마다 다르므로 비율의 판정만 출력한다.

```python {exec="browser" session="prof"}
import time

import numpy as np

def bench(fn, repeat=20):
    """fn을 repeat번 실행해 각 실행 시간(초)의 배열을 돌려준다"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)                                    # (repeat,)

_table = {}
def lookup(i):                                                # 첫 호출 때만 큰 표를 만든다(지연 초기화)
    if "sin" not in _table:
        _table["sin"] = np.sin(np.arange(2_000_000) * 1e-3)
    return _table["sin"][i]

t = bench(lambda: lookup(123))
print("첫 호출이 나머지의 중앙값보다 10배 이상 느린가:", t[0] > 10 * np.median(t[1:]))

x = np.random.default_rng(0).normal(size=100_000)            # (N,)
def loop_version():
    return sum(v * v for v in x) / len(x)
def vector_version():
    return float(np.mean(x * x))
print("결과가 같은가(상대 오차 < 1e-12):", abs(loop_version() - vector_version()) < 1e-12 * vector_version())
ratio = np.median(bench(loop_version, 5)) / np.median(bench(vector_version, 5))
print("벡터화가 10배 이상 빠른가:", ratio > 10)
```

```output
첫 호출이 나머지의 중앙값보다 10배 이상 느린가: True
결과가 같은가(상대 오차 < 1e-12): True
벡터화가 10배 이상 빠른가: True
```

`lookup`의 첫 호출은 큰 표를 만드느라 이후 호출보다 훨씬 느리다. 이 시간을 평균에 섞으면 정상 상태의 속도를 크게 과소평가한다. 벡터화한 버전은 결과가 같으면서(상대 오차 $10^{-12}$ 미만) 10배 넘게 빠르다. Python 반복문은 원소마다 인터프리터를 거치고, NumPy는 C 반복문 하나로 처리하기 때문이다(5.3장).

::quiz{id="warmup-first-step"}

GPU에서 같은 일을 할 때의 측정 틀은 다음과 같다(CUDA GPU가 있는 환경의 집필 시점 예시).

```python
# 집필 시점 예시: GPU 학습 step의 정상 상태 시간
for _ in range(5):                                  # warm-up
    train_step(next(batches))
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    train_step(next(batches))
end.record()
torch.cuda.synchronize()
ms_per_step = start.elapsed_time(end) / 50
print(f"{ms_per_step:.1f} ms/step, {batch_size * 1000 / ms_per_step:.0f} samples/s")
```

::quiz{id="sync-timing"}

## profiler

**왜 필요한가.** step 시간이 얼마인지 알아도 그 시간이 어디에 쓰이는지는 모른다. profiler는 시간을 함수·연산·구간별로 나눠 보여 준다. 병목은 대개 예상하지 못한 곳에 있으므로(5.4장), 추측하기 전에 profiler를 먼저 돌린다.

**층에 따라 다른 도구.**

| 도구 | 보는 것 | 언제 |
|---|---|---|
| `cProfile` | Python 함수별 호출 횟수·시간 | 데이터 로딩·전처리·학습 루프의 Python 코드(5.4장) |
| PyTorch profiler | PyTorch 연산(`aten::*`)별 CPU·GPU 시간, 사용자 구간, 메모리 | 모델 forward·backward·optimizer의 구성 |
| Nsight Systems | CPU 스레드, CUDA API 호출, GPU kernel, 메모리 복사를 하나의 시간축에 | GPU가 언제 놀고 있는가, CPU–GPU 겹침 |

먼저 Python 수준에서 데이터 준비와 계산 중 어디에 시간이 가는지 본다. 샘플이 텍스트(JSON)로 저장되어 있어 샘플마다 Python에서 파싱하는 로더다.

```python {exec="browser" session="prof"}
import cProfile
import json
import pstats

rng = np.random.default_rng(0)
records = [json.dumps(rng.normal(size=200).round(4).tolist()) for _ in range(256)]   # 디스크의 텍스트 샘플
W = rng.normal(size=(200, 32))                                # (D, H)

def decode(rec):
    return np.array(json.loads(rec))                          # (D,)

def load_batch():
    return np.stack([decode(r) for r in records])             # (B, D): 샘플마다 Python에서 파싱

def compute(xb):
    return np.tanh(xb @ W).mean()                             # (B, D) @ (D, H)

compute(load_batch())                                         # warm-up: 첫 호출의 초기화 비용 제외
prof = cProfile.Profile()
prof.enable()
for _ in range(5):
    compute(load_batch())
prof.disable()
st = pstats.Stats(prof).stats                                 # {(파일, 줄, 함수): (원시 호출, 호출, tottime, cumtime, ...)}
calls = {fn[2]: s[1] for fn, s in st.items() if fn[2] in ("load_batch", "decode", "compute")}
cum = {fn[2]: s[3] for fn, s in st.items() if fn[2] in ("load_batch", "compute")}
print("호출 횟수:", dict(sorted(calls.items())))
print("데이터 준비가 load+compute 시간의 80% 이상인가:", cum["load_batch"] > 0.8 * (cum["load_batch"] + cum["compute"]))
```

```output
호출 횟수: {'compute': 5, 'decode': 1280, 'load_batch': 5}
데이터 준비가 load+compute 시간의 80% 이상인가: True
```

호출 횟수부터 읽는다. batch 5개 × 샘플 256개 = 1280번의 `decode`는 기대와 맞다. 시간은 데이터 준비가 80% 이상을 차지한다. 이 로더에서 계산을 아무리 빠르게 해도 step 시간은 거의 줄지 않는다. 고칠 곳은 저장 형식이다. 샘플을 미리 숫자 배열로 변환해 두면(예: `.npy` shard, 15.2장) 파싱 비용이 사라진다. 계산 직전의 `compute(load_batch())` 한 번은 warm-up이다. 첫 행렬 곱의 초기화 비용이 profile에 섞이지 않게 한다.

**PyTorch profiler.** 모델 안을 볼 때는 PyTorch profiler를 쓴다. `record_function`으로 구간에 이름을 붙이면, 그 구간 안의 연산들이 이름 아래로 모인다. 로컬 PyTorch가 필요하므로 브라우저가 아닌 로컬 Python에서 실행하며, 출력은 이 책의 실습 환경(CPU)에서 얻은 것이다.

```python {exec="python" session="prof-torch"}
import torch
from torch.profiler import ProfilerActivity, profile, record_function

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.1)

with profile(activities=[ProfilerActivity.CPU]) as prof:     # GPU가 있으면 ProfilerActivity.CUDA를 더한다
    for step in range(5):
        with record_function("load_batch"):
            xb, yb = torch.randn(256, 64), torch.randint(0, 10, (256,))   # (B, D), (B,)
        with record_function("train_step"):
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

ev = {e.key: e for e in prof.key_averages()}
print({k: ev[k].count for k in ["load_batch", "train_step", "aten::addmm", "aten::mm", "aten::relu"]})
# 표 전체: print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# trace 파일: prof.export_chrome_trace("trace.json") → Perfetto 등에서 시간축으로 본다
```

```output
{'load_batch': 5, 'train_step': 5, 'aten::addmm': 10, 'aten::mm': 15, 'aten::relu': 5}
```

호출 횟수가 구조를 확인해 준다. step 5번에 선형층이 두 개이므로 forward의 `aten::addmm`이 10번, backward에서 가중치·입력 gradient를 계산하는 `aten::mm`이 15번(첫 층은 입력 gradient가 필요 없어 층당 1–2번) 나온다. 실제 진단에서는 주석 처리한 `table()`로 연산별 시간을 정렬해 보고, `export_chrome_trace`로 저장한 trace를 시간축 viewer(Perfetto, `chrome://tracing` 등)로 연다. 긴 학습에서는 profiler의 `schedule` 인자로 몇 step만 골라 기록한다. 모든 step을 기록하면 trace 파일이 감당하기 어려울 만큼 커지고, profiler 자체가 학습을 느리게 한다.

**trace 읽기.** 시간축에서 찾을 패턴은 다음과 같다(Nsight Systems: `nsys profile -o report python train.py`로 기록하고 GUI로 연다, 집필 시점 예시).

- **GPU 줄의 빈칸.** kernel 사이가 비어 있으면 GPU가 CPU를 기다린다. 그 시간에 CPU 줄에서 무엇을 하는지 본다. 데이터 로딩, Python 오버헤드, 동기화(`.item()`) 중 하나인 경우가 많다.
- **아주 짧은 kernel이 빽빽하게.** 원소별 연산이 작은 kernel 수백 개로 쪼개져 있으면 kernel launch 비용이 계산보다 크다. 연산 합치기(fusion)나 `torch.compile`의 대상이다.
- **매 step의 큰 memcpy.** CPU→GPU 복사가 계산과 겹치지 않고 순서대로 일어난다. pinned memory와 `non_blocking=True`의 대상이다.

## 데이터 공급

**왜 필요한가.** GPU 학습에서 가장 흔한 병목은 GPU가 아니라 GPU에 데이터를 넣는 쪽이다. GPU 사용률이 낮고 들쭉날쭉하면 먼저 데이터 공급을 의심한다(5.5장의 병목 표).

**데이터 대기 비율.** 학습 루프에서 "다음 batch를 받기까지 기다린 시간"과 "계산 시간"을 따로 재면, 한 step에서 데이터를 기다린 비율을 얻는다. 확인 실험도 간단하다. 같은 batch 하나를 GPU에 올려 두고 반복해서 학습시켰을 때 step 시간이 크게 줄면, 차이만큼이 데이터 공급 비용이다.

**겹치기의 상한.** 데이터 준비 시간이 $t_d$, 계산 시간이 $t_c$인 step이 순서대로 일어나면 step 시간은 $t_d + t_c$다. 다음 batch를 계산하는 동안 미리 준비하면(prefetch) step 시간은

$$
t_{\text{step}} = \max(t_d, t_c)
$$

까지 줄 수 있다. 데이터가 더 느리면($t_d > t_c$) 아무리 겹쳐도 $t_d$ 아래로 내려가지 않으므로, 그때는 데이터 준비 자체를 빠르게 해야 한다. 실습의 연습문제에서 step 기록으로 이 상한을 계산한다.

**조치.** PyTorch DataLoader 기준으로 정리한다(옵션 이름은 집필 시점 예시, 17.4장).

| 조치 | 효과 | 주의 |
|---|---|---|
| `num_workers` 늘리기 | 여러 프로세스가 batch를 동시에 준비 | CPU 코어·메모리 한도. worker마다 seed(22.2장) |
| `prefetch_factor` | worker마다 미리 준비해 둘 batch 수 | 메모리 사용 증가 |
| `persistent_workers=True` | epoch마다 worker를 새로 띄우는 비용 제거 | worker 안의 상태가 epoch을 넘어 유지됨 |
| `pin_memory=True` + `.to(device, non_blocking=True)` | 페이지 고정 메모리에서 비동기 복사, 계산과 겹침 | 고정 메모리는 시스템 메모리를 묶어 둔다 |
| 전처리 미리 하기 | 매 epoch 반복되는 무거운 변환 제거 | 무작위 증강은 미리 할 수 없다 |
| 저장 형식 바꾸기 | 작은 파일 수십만 개 → 큰 shard, 텍스트 → 이진 배열 | 15.2장. 공유 파일 시스템은 파일 수에 약하다(2.5장) |
| 로컬 SSD로 옮기기 | 네트워크 파일 시스템의 지연 제거 | job 시작 시 복사 시간(2.5장) |

::quiz{id="low-util-diagnosis"}

:::note[분야 예시]
위성 영상이나 기후 재분석 자료로 학습할 때는 원자료가 NetCDF·GeoTIFF 같은 형식의 큰 파일이고, 학습 샘플은 그 안에서 잘라 낸 작은 패치다. 매 step 원본 파일을 열어 좌표로 잘라 내고 정규화하면 CPU가 GPU를 따라가지 못한다. 패치를 미리 잘라 정규화한 뒤 chunk 단위로 읽기 좋은 형식(예: Zarr, 또는 큰 `.npy` shard)으로 저장하고, 학습 루프는 연속된 chunk를 읽게 바꾸면 데이터 대기 비율이 크게 줄어드는 경우가 많다. 바꾸기 전과 후의 데이터 대기 비율을 같은 방법으로 재서 효과를 확인한다.
:::

## 연산 효율

**왜 필요한가.** 데이터 공급이 충분해지면 다음 병목은 계산이다. 계산 자체를 줄이거나, 같은 계산을 하드웨어가 더 잘하는 형태로 바꿔 빠르게 한다.

**벡터화와 batch.** 연산 한 번에는 고정 비용이 붙는다. Python 함수 호출, NumPy 배열 생성, GPU kernel launch가 그것이다. 샘플 하나씩 처리하면 이 고정 비용이 샘플마다 붙고, batch로 묶으면 batch마다 한 번만 붙는다. 같은 작은 신경망을 batch 크기만 바꿔 돌린다.

```python {exec="browser" session="prof"}
W1 = rng.normal(size=(4, 8))                                  # (D, H)
W2 = rng.normal(size=(8, 2))                                  # (H, C)

def forward(xb):
    return np.maximum(xb @ W1, 0) @ W2                        # (B, D) → (B, C)

data = rng.normal(size=(4096, 4))                             # (N, D)
per_sample = {}
for bs in [1, 16, 256]:
    def epoch():
        for i in range(0, 512, bs):                           # 샘플 512개를 bs씩
            forward(data[i:i + bs])
    per_sample[bs] = np.median(bench(epoch, 5)) / 512         # 샘플 하나당 초
print("batch 16이 batch 1보다 샘플당 5배 이상 빠른가:", per_sample[1] > 5 * per_sample[16])
print("batch 256이 batch 1보다 샘플당 20배 이상 빠른가:", per_sample[1] > 20 * per_sample[256])
print("결과는 batch 크기와 무관한가:", np.allclose(forward(data[:256]), np.vstack([forward(r[None]) for r in data[:256]])))
```

```output
batch 16이 batch 1보다 샘플당 5배 이상 빠른가: True
batch 256이 batch 1보다 샘플당 20배 이상 빠른가: True
결과는 batch 크기와 무관한가: True
```

batch가 커질수록 샘플당 시간이 줄고, 결과는 batch 크기와 무관하게 같다. 그러나 이 이득은 계속되지 않는다. 고정 비용이 충분히 나뉘면 샘플당 시간은 계산량으로 정해지는 바닥에 닿고, 그 뒤로는 batch를 키워도 메모리만 더 쓴다. 학습에서는 batch 크기가 최적화를 바꾸므로(learning rate를 다시 맞춰야 하고, 같은 성능에 필요한 epoch 수가 달라진다) 처리량만 보고 정하지 않는다.

::quiz{id="batch-throughput"}

**roofline 감각.** 5.5장의 roofline 모형은 연산마다 "원리적으로 메모리 한계인가 연산 한계인가"를 알려 준다(Williams 외, 2009). 큰 행렬 곱은 원소 하나를 읽어 여러 번 쓰므로 연산 강도가 높아 연산 한계에 가깝고, 활성화 함수·정규화·덧셈 같은 원소별 연산은 원소마다 한두 번 계산하고 끝나 메모리 대역폭 한계다. 그래서 원소별 연산을 빠르게 하는 방법은 계산을 줄이는 것이 아니라 **메모리를 덜 오가는 것**이다. 연속된 원소별 연산을 kernel 하나로 합치면(fusion) 중간 결과를 메모리에 썼다 다시 읽는 일이 사라진다.

**mixed precision.** 행렬 곱과 convolution을 16비트 형식(bf16, fp16)으로 하면 최신 GPU의 행렬 연산 장치(Tensor Core 등)를 쓸 수 있고, 메모리 이동량도 절반이 된다. 두 형식은 성질이 다르다(5.5장 수의 표현).

- **fp16**: 지수 비트가 적어 표현 범위가 좁다. 작은 gradient가 0으로 사라지는(underflow) 것을 막으려면 손실에 큰 수를 곱했다가 gradient에서 나누는 **loss scaling**이 필요하다. 가중치의 기준 사본은 fp32로 유지한다(Micikevicius 외, 2018).
- **bf16**: fp32와 같은 지수 비트를 가져 표현 범위가 같고, 대신 가수 비트가 적어 정밀도가 낮다. 대개 loss scaling 없이 쓴다. 하드웨어가 지원하는지 확인한다.

PyTorch에서는 `torch.autocast`가 연산마다 안전한 정밀도를 골라 준다. 행렬 곱은 16비트로, 합·softmax·손실처럼 정밀도에 민감한 연산은 fp32로 두는 식이다. fp16을 쓸 때는 `GradScaler`로 loss scaling을 한다(집필 시점 예시, 최신 API는 공식 문서의 AMP 안내에서 확인한다).

```python
# 집필 시점 예시: CUDA에서 bf16 autocast 학습 step
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward()                     # backward는 autocast 밖에서
opt.step()
opt.zero_grad(set_to_none=True)
```

**torch.compile과 효율적 attention.** `torch.compile(model)`은 모델의 연산 그래프를 잡아 연산을 합치고 하드웨어에 맞는 kernel을 생성한다. 첫 호출에서 컴파일 시간이 들고(warm-up에서 빼야 하는 대표적인 일회성 비용), 입력 shape이 바뀌면 다시 컴파일할 수 있으며, 그래프로 잡을 수 없는 Python 코드에서는 그래프가 끊긴다(17.10장). attention은 시퀀스 길이의 제곱 크기 행렬을 만들기 때문에 메모리 이동이 큰데, `torch.nn.functional.scaled_dot_product_attention`은 하드웨어와 입력에 따라 이 행렬을 메모리에 다 쓰지 않는 fused kernel(FlashAttention 계열 등)을 골라 쓴다. 어느 기법이든 효과는 모델·하드웨어·버전에 따라 크게 다르므로, 같은 조건에서 전후를 재서 판단한다.

::quiz{id="amdahl"}

**어디부터 고치는가.** 전체 시간의 비율이 큰 곳부터다. 한 부분을 $s$배 빠르게 했을 때 전체 속도 향상은 그 부분의 비율 $p$에 묶인다.

$$
\text{속도 향상} = \frac{1}{(1 - p) + p/s} \le \frac{1}{1 - p}
$$

이 Amdahl의 법칙은 단순하지만 우선순위를 정하는 데 충분하다. step의 10%를 차지하는 연산을 아무리 최적화해도 전체는 1.11배를 넘지 못한다. profiler가 보여 준 비율로 이 상한을 먼저 계산하고, 상한이 큰 곳부터 고친다.

## 변화의 검증

**왜 필요한가.** 최적화는 계산을 바꾼다. mixed precision은 정밀도를, fusion과 컴파일은 연산 순서를, 데이터 로더 변경은 샘플 순서와 증강을 바꾼다. 빨라졌지만 답이 달라졌다면 개선이 아니다. 그리고 달라졌는지 판단하려면 "원래 얼마나 흔들리는가"를 알아야 한다(22.2장).

**세 층의 비교.**

1. **출력 수준.** 같은 입력에 대해 최적화 전후의 출력을 비교한다. 기준은 더 높은 정밀도(float64)로 계산한 출력이 좋다. 허용 오차는 바꾼 정밀도에 맞춘다. float32 대 float64는 $10^{-7}$ 근처, bf16은 $10^{-3}$–$10^{-2}$ 근처가 정상이다. 이보다 크게 벗어나면 버그나 수치 문제를 의심한다.
2. **하위 판정.** 분류라면 예측 클래스가 얼마나 일치하는지, 회귀라면 오차 지표가 얼마나 바뀌는지 본다. 경계 근처의 샘플은 작은 수치 차이로도 판정이 바뀐다.
3. **학습 결과.** 최적화한 설정으로 처음부터 학습해, 최종 metric이 seed 간 변동 범위 안에 드는지 확인한다. 정밀도 변화는 학습 경로 전체에 누적되므로 출력 수준 비교만으로는 부족하다.

1–2층을 bf16 autocast로 확인한다. 로컬 PyTorch가 필요하므로 로컬 Python에서 실행하며, 출력은 이 책의 실습 환경(CPU)에서 얻은 것이다.

```python {exec="python" session="prof-torch"}
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))
xb = torch.randn(1024, 64)                                    # (B, D)
with torch.no_grad():
    ref64 = net.double()(xb.double())                         # 기준: float64
    net.float()
    out32 = net(xb)                                           # (B, C) 기본: float32
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out16 = net(xb)                                       # (B, C) bf16 autocast

def rel_err(a, b):
    return ((a.double() - b).norm() / b.norm()).item()

print("출력 dtype:", out32.dtype, out16.dtype)
print(f"float32 대 float64 상대 오차 {rel_err(out32, ref64):.0e}, bf16 대 float64 {rel_err(out16, ref64):.0e}")
agree = (out16.argmax(dim=1) == ref64.argmax(dim=1)).double().mean().item()
print(f"예측 클래스 일치 비율: bf16 {agree:.3f}")
```

```output
출력 dtype: torch.float32 torch.bfloat16
float32 대 float64 상대 오차 3e-07, bf16 대 float64 4e-03
예측 클래스 일치 비율: bf16 0.998
```

float32의 상대 오차는 $10^{-7}$ 수준이고, bf16은 $10^{-3}$ 수준으로 1만 배 가까이 크다. bf16의 가수 비트가 적다는 것에서 예상되는 크기다. 예측 클래스는 1024개 중 2개가 바뀌었다(일치 비율 0.998). 이 차이가 학습 결과에 의미 있는 영향을 주는지는 3층, 즉 seed 여러 개로 학습해 비교해야 알 수 있다.

::quiz{id="speed-vs-accuracy"}

**보고.** 최적화의 결과는 두 줄로 보고한다. "step 시간 X ms → Y ms(같은 하드웨어·같은 batch·warm-up 제외, 중앙값)", 그리고 "최종 검증 metric 평균 ± 표준편차가 전후 각각 A, B(seed n개)". 둘 중 하나만 있으면 독자는 개선인지 판단할 수 없다.

:::note[분야 예시]
기후 모델이나 유체 시뮬레이션의 대리모델(surrogate)을 bf16으로 학습하면 빨라지지만, 장기 rollout에서는 한 step의 작은 오차가 누적된다. 한 step 예측 오차만 비교하면 차이가 없어 보여도, 수백 step을 굴린 뒤의 에너지 보존이나 기후 통계(평균장, 분산)가 달라질 수 있다. 최적화 전후 비교는 연구 질문이 실제로 쓰는 지표, 여기서는 장기 rollout의 물리량으로 해야 한다.
:::

## 실습

학습 스크립트를 profiling해 병목을 찾고, 개선 전후를 같은 조건에서 측정한다.

1. 22.1장 실습의 학습 스크립트에 step마다 "다음 batch를 받기까지의 시간"과 "계산 시간"을 기록하는 코드를 넣는다. GPU라면 각 구간 끝에서 동기화한다. 기록은 `metrics.jsonl`(22.1장)에 남긴다.
2. warm-up을 뺀 처리량과 데이터 대기 비율을 계산한다. 아래 연습문제의 함수가 그 핵심이다.
3. PyTorch profiler(GPU가 있으면 Nsight Systems도)로 몇 step을 기록해, 시간의 대부분이 어느 구간·연산에 쓰이는지 표로 정리한다.
4. 가장 비율이 큰 병목 하나를 골라 Amdahl의 법칙으로 기대 이득의 상한을 계산한 뒤 고친다.
5. 같은 기계·같은 조건에서 전후 step 시간을 재고, 변화의 검증 세 층을 모두 확인한다.

본문 셀들의 측정을 실제 시간으로 보고하는 `labs/experiment-operations/profiling-and-efficiency/bench_report.py`도 있다. 이 기계에서의 예(집필 시점, 다른 job이 함께 도는 공유 CPU 서버, 스레드 1개)로 같은 스크립트를 두 번 연달아 실행한 결과는 다음과 같다.

```output
환경: x86_64, Python 3.13.18, NumPy 2.5.4, PyTorch 2.15.0+cu132, 스레드 1개
[측정] 10만 원소 제곱 평균: Python 반복문 17.28 ms, NumPy 0.106 ms (163배)
[profiler] batch 256: 데이터 준비 10.06 ms, 계산 0.112 ms (데이터 비율 99%)
[연산 효율] 샘플당 시간 — batch 1: 4.68 µs, batch 16: 0.30 µs, batch 256: 0.03 µs
[mixed precision] CPU 학습 step: float32 11.8 ms, bf16 autocast 4.4 ms (bf16/float32 = 0.38)

환경: x86_64, Python 3.13.18, NumPy 2.5.4, PyTorch 2.15.0+cu132, 스레드 1개
[측정] 10만 원소 제곱 평균: Python 반복문 15.84 ms, NumPy 0.088 ms (181배)
[profiler] batch 256: 데이터 준비 9.20 ms, 계산 0.082 ms (데이터 비율 99%)
[연산 효율] 샘플당 시간 — batch 1: 3.28 µs, batch 16: 0.20 µs, batch 256: 0.04 µs
[mixed precision] CPU 학습 step: float32 16.0 ms, bf16 autocast 4.0 ms (bf16/float32 = 0.25)
```

읽는 법은 세 가지다.

- **같은 스크립트를 연달아 돌려도 숫자가 항목에 따라 10–45%가량 흔들린다**(batch 1의 4.68 µs 대 3.28 µs, float32 step의 11.8 ms 대 16.0 ms). 공유 기계의 부하 때문이다. 대소 관계는 두 번 모두 같다. 그래서 본문 셀은 시간 대신 판정을 출력하고, 전후 비교는 같은 시점에 번갈아 여러 번 재서 중앙값으로 한다.
- **데이터 준비가 계산의 수십 배다.** 이 로더에서는 계산을 고치는 것이 의미가 없다.
- **이 CPU에서는 bf16 autocast가 float32보다 빨랐다.** CPU가 bf16 연산을 하드웨어로 지원하는지에 따라 결과는 반대가 될 수도 있다. 자기 하드웨어에서 재 보기 전에는 어느 쪽이라고 말할 수 없다.

이어서 step 시간 기록을 분해하는 함수를 연습문제로 구현한다.

::exercise{id="step-time-breakdown"}

## 실패 사례

- **synchronize 없이 GPU 시간을 측정하는 것.** kernel을 대기열에 넣는 시간만 잰다. 계산 시간은 다음 동기화 지점(`.item()`, `print`, `.cpu()`)에 몰려 엉뚱한 줄이 병목처럼 보인다.
- **속도를 얻고 정확도를 잃은 것을 놓치는 것.** mixed precision·fusion·데이터 로더 변경 후 처리량만 보고하고 최종 metric을 seed 여러 개로 비교하지 않는다.
- **첫 step을 포함해 평균 내는 것.** 컴파일·메모리 할당·캐시 채우기 비용이 섞여 정상 상태 처리량을 과소평가한다.
- **profiler 없이 추측으로 최적화하는 것.** 비율이 작은 부분을 고치면 Amdahl의 법칙에 따라 전체는 거의 빨라지지 않는다.
- **처리량을 학습 속도로 착각하는 것.** batch 크기를 키워 처리량을 높여도 목표 성능에 도달하는 시간이 줄지 않을 수 있다.

## 통과 기준

- 개선 하나에 대해 전후 step 시간(같은 조건, warm-up 제외, 반복 측정의 중앙값)과 수치 결과의 동일성(출력 수준 상대 오차와 seed 여러 개의 최종 metric)을 함께 보고한다.
- 학습 스크립트의 step 시간을 데이터 대기와 계산으로 나누고, 데이터 대기 비율과 prefetch로 얻을 수 있는 이득의 상한을 계산한다.
- 연습문제 `step-time-breakdown`의 테스트를 통과한다.

## 참고 문헌

- Williams, Waterman, Patterson (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4). — 연산 강도와 메모리 대역폭으로 성능 상한을 그리는 roofline 모형
- Micikevicius 외 (2018). Mixed precision training. *ICLR 2018*. — fp16 학습의 loss scaling과 fp32 가중치 사본
