---
id: python-and-computing/computer-architecture
title: "컴퓨터 구조와 병렬 계산"
title_en: "Computer Architecture and Parallelism"
tier: core
status: draft
prerequisites: [python-and-computing/algorithms-and-complexity]
---

# 컴퓨터 구조와 병렬 계산

5.4장의 복잡도는 연산의 **횟수**를 센다. 그러나 같은 횟수의 연산도 어떤 순서로, 어디에 있는 데이터로, 어떤 장치에서 하느냐에 따라 시간이 수십 배 달라진다. 비싼 GPU를 빌렸는데 사용률이 20%에 머물고, 스레드를 늘렸는데 빨라지지 않고, GPU 연산 시간을 쟀더니 말이 안 되게 짧게 나온다. 이런 현상은 알고리즘이 아니라 **기계의 구조**로 설명된다.

이 장은 하드웨어 설계를 가르치지 않는다. 연구 코드의 성능 문제를 진단하는 데 필요한 최소한의 모형을 다룬다. 데이터가 메모리 계층을 오르내리는 비용, GPU가 빠른 조건과 느린 조건, 프로세스·스레드·GIL의 관계, 그리고 이 모두를 묶어 "지금 무엇이 병목인가"를 묻는 틀이다. 마지막으로 수가 비트로 어떻게 표현되는지 개요를 보고, 자세한 수치 오차 분석은 9부로 넘긴다. GPU 학습의 실제 profiling과 메모리 관리는 22.4–22.5장에서 이어진다.

## 학습 목표

- CPU·GPU·메모리 계층을 설명한다
- process·thread·GIL을 이해한다
- 병목의 종류를 구분한다

## 하드웨어 구성

연구용 계산 서버 한 대는 대략 다음 부품으로 이루어진다.

- **CPU**: 수 개에서 수십 개의 코어. 코어 하나는 복잡한 제어 흐름(분기, 함수 호출, 운영체제 작업)을 빠르게 처리하도록 설계되어 있다. Python 인터프리터, 데이터 전처리, 파일 입출력이 여기서 돈다.
- **RAM**(주 메모리): CPU가 직접 읽고 쓰는 메모리. 수십 GB에서 TB 단위.
- **GPU**: 수천 개의 단순한 연산 유닛으로 같은 연산을 많은 데이터에 동시에 적용하도록 설계된 가속기. 자체 메모리인 **VRAM**을 가진다. GPU는 VRAM에 있는 데이터만 계산할 수 있으므로, RAM의 데이터를 PCIe 같은 연결 통로로 복사해 와야 한다.
- **storage**: SSD·HDD·네트워크 파일 시스템. 전원이 꺼져도 남는 대신 가장 느리다.

**메모리 계층.** 핵심은 **빠른 저장소일수록 작고 비싸다**는 것이다. 그래서 컴퓨터는 크고 느린 저장소와 작고 빠른 저장소를 층층이 쌓는다. 아래 수치는 자릿수만 보이기 위한 대략적인 범위이고, 실제 값은 제품마다 다르므로 사양서에서 확인한다.

| 층 | 크기(대략) | 접근 지연(대략) |
|---|---|---|
| 레지스터 | 수백 byte–수 KB | 1 ns 미만 |
| L1·L2 캐시 | 수십 KB–수 MB (코어마다) | 수 ns |
| L3 캐시 | 수 MB–수백 MB (코어들이 공유) | 수십 ns |
| RAM | 수십 GB–TB | 약 100 ns |
| SSD | TB | 수십–수백 µs |
| 네트워크 저장소 | 무제한에 가까움 | ms 이상 |

**지연**(latency)은 데이터 하나를 받기까지 걸리는 시간, **대역폭**(bandwidth)은 초당 옮길 수 있는 데이터의 양이다. 계산 속도는 수십 년간 메모리 지연보다 훨씬 빠르게 좋아졌다. 그래서 현대 프로세서에서 많은 계산은 연산 자체보다 **데이터를 가져오는 시간**이 정한다.

**캐시와 지역성.** CPU는 RAM에서 데이터를 한 byte씩이 아니라 **캐시 라인**(cache line, 흔히 64 byte) 단위로 가져오고, 순서대로 읽는 패턴을 감지하면 다음 라인을 미리 가져온다(prefetch). 그래서 프로그램이 다음 두 성질을 가질수록 빠르다.

- **공간 지역성**: 방금 읽은 데이터 근처를 곧 읽는다. 배열을 메모리 순서대로 훑으면 캐시 라인 하나로 원소 여러 개를 얻는다.
- **시간 지역성**: 방금 읽은 데이터를 곧 다시 읽는다. 캐시에 남아 있을 때 재사용한다.

같은 원소들을 같은 횟수만큼 읽어도, 순서만 바꾸면 시간이 달라진다.

```python {exec="browser" session="arch"}
import time
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=8_000_000)                 # 64 MB: CPU 캐시보다 크다
seq_idx = np.arange(x.size)                    # 0, 1, 2, ... 순서대로
rand_idx = rng.permutation(x.size)             # 같은 원소를 무작위 순서로

def best_time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

t_seq = best_time(lambda: x[seq_idx].sum())
t_rand = best_time(lambda: x[rand_idx].sum())
print("같은 합인가:", np.isclose(x[seq_idx].sum(), x[rand_idx].sum()))
print("무작위 접근이 1.5배 이상 느린가:", t_rand > 1.5 * t_seq)

A = np.zeros((1000, 1000))                     # C order (row-major)
print("C order strides:", A.strides, " Fortran order strides:", np.asfortranarray(A).strides)
```

```output
같은 합인가: True
무작위 접근이 1.5배 이상 느린가: True
C order strides: (8000, 8)  Fortran order strides: (8, 8000)
```

두 계산은 같은 원소 800만 개를 한 번씩 읽어 같은 합을 낸다. 연산 횟수가 같으므로 5.4장의 복잡도로는 구별되지 않는다. 그러나 무작위 순서는 매 접근마다 다른 캐시 라인을 요구하고 prefetch도 소용없어서 느리다(이 셀을 실행한 기계에서는 1.5배를 넘었다. 배율은 기계마다 다르므로 대소 관계만 출력한다).

**배열의 메모리 배치.** 5.3장의 strides가 여기서 성능의 문제가 된다. NumPy의 기본 C order(row-major)에서는 마지막 축이 연속이다. `(1000, 1000)` 배열에서 `A[i, j]`의 `j`를 한 칸 옮기면 8 byte, `i`를 한 칸 옮기면 8000 byte를 건너뛴다. Fortran order(column-major, Fortran·MATLAB·Julia의 기본)는 반대다. 반복문이나 컴파일된 kernel이 배열을 훑을 때는 **가장 안쪽 반복이 연속된 축을 돌게** 한다. NumPy의 내장 연산은 대개 이를 알아서 처리하지만, 전치한 view(`A.T`)처럼 연속이 아닌 배열을 C 확장이나 GPU로 넘길 때는 `np.ascontiguousarray`로 복사가 일어나거나, 복사가 필요하다는 오류가 날 수 있다.

::widget{name="strides-cache"}

::quiz{id="row-major-traversal"}

## GPU의 계산 모델

**왜 GPU인가.** 신경망 학습의 계산은 대부분 큰 행렬 곱과 원소별 연산이다. 같은 연산을 수백만 개의 원소에 독립적으로 적용하므로, 복잡한 제어 흐름보다 **동시에 많이** 계산하는 능력이 중요하다. GPU는 이런 데이터 병렬(data parallel) 계산에 맞게 만든 장치다.

**SIMT와 kernel.** GPU에서 실행되는 함수를 **kernel**이라 한다. kernel을 launch하면 수천~수백만 개의 스레드가 같은 코드를 서로 다른 데이터에 실행한다. NVIDIA GPU는 스레드를 32개씩 묶은 **warp** 단위로 같은 명령을 동시에 실행하는데, 이를 SIMT(single instruction, multiple threads)라 한다. 같은 warp 안의 스레드가 `if` 문에서 다른 갈래로 나뉘면 두 갈래를 차례로 실행해야 해서 느려진다. 그래서 GPU는 분기가 적고 균일한 계산에서 빠르다. PyTorch를 쓸 때 kernel을 직접 쓰는 일은 드물다. `x @ w`나 `torch.relu(x)` 같은 연산 하나하나가 미리 만들어진 kernel의 launch다.

**연산 강도와 roofline.** GPU의 최대 연산 속도(FLOP/s)는 메모리 대역폭(byte/s)보다 훨씬 크다. 그래서 어떤 연산이 GPU의 연산 능력을 다 쓸 수 있는지는 **연산 강도**(arithmetic intensity)

$$
I = \frac{\text{연산 횟수 (FLOP)}}{\text{메모리 이동량 (byte)}}
$$

가 정한다. 메모리 대역폭을 $B$, 최대 연산 속도를 $P_{\text{peak}}$라 하면, 이 연산이 낼 수 있는 속도의 상한은

$$
P_{\text{attainable}} = \min\left(P_{\text{peak}},\ I \cdot B\right)
$$

이다. $I$를 가로축, $P_{\text{attainable}}$을 세로축으로 로그 그래프를 그리면 기울어진 직선(메모리 한계)과 수평선(연산 한계)이 만나는 지붕 모양이 되어 **roofline** 모형이라 부른다. 두 선이 만나는 점 $I^* = P_{\text{peak}} / B$보다 연산 강도가 낮으면 메모리 대역폭이, 높으면 연산 능력이 병목이다.

```python {exec="browser" session="arch"}
def intensity(flops, bytes_moved):
    return flops / bytes_moved                 # FLOP/byte

n, m = 10**7, 4096                             # 벡터 길이, 정사방행렬 크기 (float64 = 8 byte)
ops = {
    "axpy y=2a+b": (2 * n, 3 * 8 * n),         # 곱·합 각 1번 / a, b 읽고 y 쓰기
    "matmul C=AB": (2 * m**3, 3 * 8 * m * m),  # 최소 이동: 세 행렬을 한 번씩
}
peak_flops, bandwidth = 10e12, 1e12            # 가상의 가속기: 10 TFLOP/s, 1 TB/s
ridge = peak_flops / bandwidth                 # 이보다 연산 강도가 낮으면 메모리가 병목
for name, (flops, nbytes) in ops.items():
    I = intensity(flops, nbytes)
    attainable = min(peak_flops, I * bandwidth)          # roofline
    bound = "메모리" if I < ridge else "연산"
    print(f"{name}: 강도 {I:7.3f} FLOP/byte → 최대 {attainable / 1e12:6.3f} TFLOP/s ({bound} 한계)")
```

```output
axpy y=2a+b: 강도   0.083 FLOP/byte → 최대  0.083 TFLOP/s (메모리 한계)
matmul C=AB: 강도 341.333 FLOP/byte → 최대 10.000 TFLOP/s (연산 한계)
```

벡터 연산 $\mathbf{y} = 2\mathbf{a} + \mathbf{b}$는 원소 하나에 2 FLOP을 하려고 24 byte를 옮긴다. 이 가상의 장치에서는 최대 연산 속도의 1% 미만밖에 쓰지 못한다. 반면 $m \times m$ 행렬 곱은 $2m^3$ FLOP을 $3 \cdot 8m^2$ byte로 하므로 연산 강도가 $m/12$로 커져 연산 한계에 닿는다. 일반적인 원리는 다음과 같다.

- **원소별 연산(활성화, 정규화, 덧셈)은 메모리 한계다.** 여러 원소별 연산을 따로 실행하면 매번 VRAM을 왕복한다. 이를 하나의 kernel로 묶는 kernel fusion(예: `torch.compile`이 하는 일 중 하나)이 효과를 내는 이유다.
- **큰 행렬 곱은 연산 한계다.** 행렬이 작으면(batch가 작으면) 연산 강도가 낮아져 GPU를 다 쓰지 못한다.
- 같은 장치에서도 `float16`·`bfloat16`처럼 작은 dtype을 쓰면 옮길 byte가 줄어 연산 강도가 올라간다.

roofline은 상한을 주는 모형이다. 실제 속도는 kernel 구현, launch 비용, 캐시 재사용 등으로 이보다 낮다. 그러나 "이 연산은 원리적으로 메모리 한계인가"를 판단하는 데는 충분하고, 연습문제에서 이 판정을 직접 구현한다.

## 병렬성과 동시성

**정의.** **병렬성**(parallelism)은 여러 일을 **같은 순간에** 실제로 실행하는 것(여러 코어), **동시성**(concurrency)은 여러 일이 **진행 중인 상태**를 다루는 것이다. 한 코어에서도 한 일이 디스크를 기다리는 동안 다른 일을 진행하면 동시성이 있다.

**프로세스와 스레드.**

- **프로세스**(process)는 운영체제가 자원을 나누는 단위다. 프로세스마다 독립된 메모리 공간을 가진다. 한 프로세스가 죽어도 다른 프로세스는 영향을 받지 않지만, 프로세스 사이에 데이터를 주고받으려면 복사(직렬화)가 필요하다. 프로세스를 관찰하고 관리하는 방법은 2.2장에서 다룬다.
- **스레드**(thread)는 한 프로세스 안의 실행 흐름이다. 같은 프로세스의 스레드들은 메모리를 공유하므로 데이터를 복사 없이 주고받지만, 두 스레드가 같은 데이터를 동시에 바꾸면 경쟁 상태(race condition)가 생긴다.

**GIL.** CPython(표준 Python 구현)에는 GIL(global interpreter lock)이 있다. 한 프로세스 안에서 Python 바이트코드는 한 번에 **한 스레드만** 실행할 수 있다는 잠금이다. 인터프리터 내부 자료구조(참조 횟수 등)를 단순하고 안전하게 유지하기 위한 설계다. 결과는 다음과 같다.

- 순수 Python으로 쓴 CPU 계산은 스레드를 늘려도 빨라지지 않는다. 스레드들이 GIL을 번갈아 쥘 뿐이다.
- I/O를 기다리는 동안(파일·네트워크·`sleep`)에는 GIL을 놓는다. 그래서 기다리는 일이 많은 작업은 스레드로 겹쳐 빨라진다.
- NumPy·SciPy의 큰 배열 연산처럼 컴파일된 코드는 계산하는 동안 GIL을 놓는 경우가 많다. 이런 계산은 스레드로도 여러 코어를 쓸 수 있다.

Python 3.13부터는 GIL을 끈 free-threaded 빌드가 실험적으로 제공된다. 아래 셀은 실행 중인 인터프리터의 GIL 상태를 함께 출력한다. 스레드는 브라우저 실행 환경(Pyodide)에서 동작하지 않으므로, 이 셀과 다음 셀은 로컬 Python 런타임에서 실행한다.

```python {exec="python" session="arch-threads"}
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def cpu_task(_=None):               # 순수 Python 계산: 실행 중 GIL을 쥔다
    s = 0
    for i in range(3_000_000):
        s += i * i
    return s

def io_task(_=None):                # 파일·네트워크 대기를 흉내: 기다리는 동안 GIL을 놓는다
    time.sleep(0.2)

def wall(fn, k, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(k)))
    return time.perf_counter() - t0

gil = getattr(sys, "_is_gil_enabled", lambda: True)()
print("GIL 켜짐:", gil)
print("CPU 일 4개: 스레드 4개로 2배 이상 빨라지는가:", wall(cpu_task, 4, 1) > 2 * wall(cpu_task, 4, 4))
print("I/O 일 4개: 스레드 4개로 2배 이상 빨라지는가:", wall(io_task, 4, 1) > 2 * wall(io_task, 4, 4))
```

```output
GIL 켜짐: True
CPU 일 4개: 스레드 4개로 2배 이상 빨라지는가: False
I/O 일 4개: 스레드 4개로 2배 이상 빨라지는가: True
```

**multiprocessing.** CPU 계산을 여러 코어에 나누는 표준적인 방법은 프로세스를 여러 개 띄우는 것이다. `concurrent.futures.ProcessPoolExecutor`나 `multiprocessing.Pool`은 작업과 인자를 자식 프로세스에 보내고 결과를 받아 온다. 주의할 점은 세 가지다.

- **데이터 전달 비용.** 인자와 결과는 pickle로 직렬화되어 복사된다. 작업 하나가 너무 작거나 인자가 너무 크면 복사 비용이 계산 이득을 삼킨다. 큰 배열은 파일로 공유하거나 공유 메모리를 쓴다.
- **시작 방식.** 자식 프로세스를 만드는 방식(fork, spawn, forkserver)의 기본값은 플랫폼과 Python 버전마다 다르다(`multiprocessing.get_start_method()`로 확인한다). spawn 방식에서는 자식이 주 모듈을 다시 import하므로, 실행 코드를 `if __name__ == "__main__":`으로 보호해야 한다(5.2장).
- **스레드를 쓰는 라이브러리와의 중첩.** NumPy의 BLAS나 PyTorch는 내부적으로 여러 스레드를 쓴다. 프로세스 8개가 각각 스레드 8개를 띄우면 코어 수보다 훨씬 많은 스레드가 경쟁해 오히려 느려진다. 이때 `OMP_NUM_THREADS` 같은 환경 변수로 프로세스당 스레드 수를 줄인다.

PyTorch의 `DataLoader(num_workers=k)`는 이 원리를 쓴다. 데이터 읽기·디코딩·증강을 자식 프로세스 $k$개에 맡겨, 주 프로세스가 GPU에 일을 넘기는 동안 다음 batch를 미리 준비한다(17.4장).

**async의 위치.** `asyncio`는 한 스레드 안에서 동시성을 얻는 방법이다. 함수가 `await`에서 기다리는 동안 이벤트 루프가 다른 작업을 진행한다. 수천 개의 네트워크 요청(데이터 내려받기, API 호출)처럼 기다림이 대부분인 작업에 맞고, CPU 계산을 빠르게 하지는 않는다. 정리하면 다음과 같다.

| 작업의 성격 | 도구 |
|---|---|
| 기다림이 많은 I/O, 수십 개 규모 | 스레드 |
| 기다림이 많은 I/O, 수천 개 규모 | asyncio |
| 순수 Python CPU 계산 | 프로세스 |
| 큰 배열 계산 | NumPy·PyTorch(내부 병렬화)에 맡기기, 필요하면 GPU |

## 병목의 종류

성능 문제를 진단한다는 것은 "지금 무엇을 기다리고 있는가"를 찾는 일이다. 연구 코드, 특히 GPU 학습에서 만나는 병목은 다섯 가지로 나눌 수 있다.

| 병목 | 증상 | 확인하는 실험 |
|---|---|---|
| **I/O** | GPU 사용률이 낮고 들쭉날쭉, CPU는 한가하거나 디스크 대기 | 데이터를 메모리에 캐시하거나 같은 batch를 반복해 넣어 본다. 빨라지면 I/O다 |
| **연산** | GPU 사용률이 높고 꾸준함 | batch나 모델 크기를 바꾸면 시간이 연산량에 비례해 변한다 |
| **메모리 대역폭** | 연산량에 비해 느림, 원소별 연산이 많음 | 연산 강도를 계산해 roofline과 비교한다. dtype을 줄이면 빨라진다 |
| **CPU–GPU 전송** | 매 step 큰 텐서를 장치로 옮김 | 전송만 따로 재 본다. pinned memory와 `non_blocking=True`로 겹치게 한다 |
| **동기화** | 특정 줄(`.item()`, `print`)이 이상하게 느림 | 동기화를 일으키는 호출을 줄이고 차이를 잰다 |

**비동기 실행과 동기화.** GPU 연산은 **비동기**다. CPU는 kernel을 GPU의 대기열에 넣고 바로 다음 줄로 넘어간다. 덕분에 CPU가 다음 일을 준비하는 동안 GPU가 계산할 수 있다. 그러나 두 가지 혼란이 생긴다.

1. **시간을 잘못 잰다.** `t0 = time.perf_counter(); y = x @ w; t1 = time.perf_counter()`는 kernel을 대기열에 넣는 시간만 잰다.
2. **엉뚱한 줄이 느려 보인다.** `loss.item()`, `.cpu()`, `print(tensor)`처럼 GPU의 값을 CPU로 가져오는 코드는 앞선 모든 kernel이 끝나기를 기다린다(동기화). 앞선 계산의 시간이 이 줄에 몰려 보인다.

아래 셀은 GPU 없이 이 상황을 흉내 낸다. "kernel launch"는 백그라운드 스레드에 일을 맡기고 바로 돌아오고, `join()`이 `torch.cuda.synchronize()`의 역할을 한다.

```python {exec="python" session="arch-threads"}
import threading

def launch_kernel(duration_s=0.3):
    """GPU 커널 호출처럼, 일을 맡겨 두고 바로 돌아온다."""
    worker = threading.Thread(target=time.sleep, args=(duration_s,))
    worker.start()
    return worker

t0 = time.perf_counter()
job = launch_kernel()
t_launch = time.perf_counter() - t0       # 동기화 없이 잰 시간
job.join()                                # torch.cuda.synchronize()에 해당
t_sync = time.perf_counter() - t0
print(f"동기화 없이 잰 시간: {t_launch:.1f} s,  동기화 후 잰 시간: {t_sync:.1f} s")
```

```output
동기화 없이 잰 시간: 0.0 s,  동기화 후 잰 시간: 0.3 s
```

실제 GPU 코드에서 바르게 재는 방법은 다음과 같다(CUDA GPU와 PyTorch가 있는 환경에서 실행한다).

```python
import torch

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")
for _ in range(3):                      # warm-up: 첫 실행의 초기화 비용 제외
    x @ w
torch.cuda.synchronize()                # 앞선 작업을 모두 끝낸다

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(10):
    y = x @ w
end.record()
torch.cuda.synchronize()                # end 이벤트까지 끝나기를 기다린다
print(start.elapsed_time(end) / 10, "ms per matmul")
```

::quiz{id="async-gpu-timing"}

**연구에서의 의미.** "GPU가 항상 빠르다"는 가정은 연산 한계인 큰 계산에서만 맞다. 데이터가 작으면 kernel launch와 전송 비용이 계산보다 크고, 순차적인 제어 흐름(작은 반복문, 분기가 많은 시뮬레이션)은 CPU가 빠르다. 결정은 측정으로 한다. GPU 학습 전체를 profiling하는 도구(PyTorch profiler, Nsight 계열)는 22.4장에서 다룬다.

:::note[분야 예시]
기상·기후 모델의 격자 계산(stencil)은 격자점마다 이웃 몇 개의 값을 읽어 몇 번의 연산을 하는 구조라 연산 강도가 낮고, 대개 메모리 대역폭이 성능을 정한다. 이런 코드를 GPU로 옮겨 빨라지는 이유는 GPU의 연산 능력보다 VRAM의 높은 대역폭인 경우가 많다. 반대로 양자화학의 큰 행렬 연산이나 분자 표현을 다루는 신경망의 큰 행렬 곱은 연산 한계에 가깝다. 같은 "GPU 가속"이라도 병목의 종류가 다르므로, 최적화 방향(메모리 접근 줄이기 대 연산량 줄이기)도 다르다.
:::

## 수의 표현

컴퓨터는 수를 유한한 비트로 표현한다. 이 사실이 만드는 결과 중 연구 코드에서 바로 부딪히는 것만 정리한다. 원리와 오차 분석은 9.1장(부동소수점과 수치 오차)에서 자세히 다룬다.

**정수.** `int32`는 약 $\pm 2.1 \times 10^9$, `int64`는 약 $\pm 9.2 \times 10^{18}$까지 표현한다. Python의 `int`는 크기 제한이 없지만, NumPy·PyTorch의 정수 배열은 고정 폭이라 범위를 넘으면 조용히 감긴다(wrap around). 큰 데이터의 인덱스, 누적 개수, 해시 값에서 문제가 된다.

**부동소수점.** IEEE 754 부동소수점은 수를 부호, 지수, 가수(유효 숫자)로 나누어 $\pm 1.f \times 2^{e}$ 꼴로 저장한다. 비트 수에 따라 표현 범위와 정밀도가 정해진다.

| 형식 | 비트(지수·가수) | 유효 숫자(10진) | 주된 쓰임 |
|---|---|---|---|
| `float64` | 64 (11·52) | 약 16자리 | 과학 계산의 기본, NumPy 기본값 |
| `float32` | 32 (8·23) | 약 7자리 | 딥러닝의 기본, PyTorch 기본값 |
| `float16` | 16 (5·10) | 약 3자리 | 혼합 정밀도 학습·추론 |
| `bfloat16` | 16 (8·7) | 약 2–3자리 | `float32`와 같은 범위를 유지한 16비트 학습 |

```python {exec="browser" session="arch"}
print("0.1 + 0.2 == 0.3 :", 0.1 + 0.2 == 0.3, "  isclose:", np.isclose(0.1 + 0.2, 0.3))
print(f"0.1 + 0.2 = {0.1 + 0.2:.17f}")
for dt in (np.float16, np.float32, np.float64):
    fi = np.finfo(dt)
    print(f"{dt.__name__:8s} 비트 {fi.bits:2d}  machine eps {float(fi.eps):.2e}  최댓값 {float(fi.max):.2e}")
with np.errstate(over="ignore"):
    print("float16(70000) =", np.float16(70000.0))
counts = np.array([2_000_000_000, 2_000_000_000], dtype=np.int32)
print("int32로 누적한 합:", np.add.reduce(counts, dtype=np.int32), "  int64 합:", counts.sum(dtype=np.int64))
```

```output
0.1 + 0.2 == 0.3 : False   isclose: True
0.1 + 0.2 = 0.30000000000000004
float16  비트 16  machine eps 9.77e-04  최댓값 6.55e+04
float32  비트 32  machine eps 1.19e-07  최댓값 3.40e+38
float64  비트 64  machine eps 2.22e-16  최댓값 1.80e+308
float16(70000) = inf
int32로 누적한 합: -294967296   int64 합: 4000000000
```

출력에서 네 가지를 읽는다.

- **10진 소수는 대개 2진 부동소수점으로 정확히 표현되지 않는다.** 0.1은 2진수로 무한소수라 가장 가까운 값으로 반올림된다. 그래서 부동소수점 결과는 `==`가 아니라 허용 오차를 두고 비교한다(`math.isclose`, `np.isclose`, `np.allclose`).
- **machine epsilon**은 1과 그다음 표현 가능한 수의 간격이다. 상대 정밀도의 척도이고, `float32`는 약 $10^{-7}$, `float64`는 약 $10^{-16}$이다.
- **`float16`의 최댓값은 65504다.** 그보다 큰 값은 `inf`가 된다. 혼합 정밀도 학습에서 loss scaling을 쓰는 이유이고, 같은 16비트라도 지수 비트가 많은 `bfloat16`이 학습에 선호되는 이유다.
- **고정 폭 정수는 넘치면 감긴다.** 20억 두 개를 `int32`로 더하면 음수가 된다. 오류도 경고도 없다.

::quiz{id="float-equality"}

## 실습

"GPU 사용률이 낮다"는 현상을 데이터 공급·전송·연산·동기화 가설로 나누어 실험으로 검증한다. 실제 GPU가 없는 환경에서도 원리를 연습할 수 있도록, `labs/python-and-computing/computer-architecture/gpu_starvation_sim.py`는 학습 한 step을 네 단계의 모형으로 흉내 낸다.

- **load**: 샘플 읽기. 샘플마다 디스크 대기(2 ms)를 흉내 낸다.
- **transfer**: 호스트 → 장치 복사(배열 복사).
- **compute**: 장치 계산(행렬 곱 반복). '장치 사용률' = compute 시간 / 전체 시간.
- **sync**: 매 step 손실 값을 꺼내는 동기화(4 ms 대기).

가설마다 **한 가지만 바꾼** 실험을 돌린다. 이 기계에서의 예는 다음과 같다. 시간 값은 기계마다, 실행마다 다르다.

```output
batch 32, step 30, 샘플당 대기 2 ms, 동기화 대기 4 ms
기준: 순차 로딩, 매 step 동기화              전체  2.432 s   장치 사용률 10.6%
가설 1 확인: 데이터를 메모리에 캐시              전체  0.360 s   장치 사용률 64.6%
가설 1 처방: loader 스레드 8개 + prefetch  전체  0.500 s   장치 사용률 61.4%
가설 2 확인: 동기화를 10 step마다            전체  0.339 s   장치 사용률 76.6%
가설 3 확인: step당 계산량 4배              전체  1.000 s   장치 사용률 92.3%
```

결과를 다음 순서로 읽는다.

1. **데이터 공급 가설.** 데이터를 메모리에 캐시하자(같은 batch를 반복) 사용률이 10.6%에서 64.6%로 뛰었다. 기준 상태의 주된 병목은 데이터 공급이었다. 처방으로 loader 스레드 8개와 prefetch 대기열을 쓰자 캐시와 비슷한 수준까지 올라왔다. 실제 학습에서는 `DataLoader`의 `num_workers`와 `prefetch_factor`가 이 역할을 한다.
2. **동기화 가설.** 매 step 동기화를 10 step마다로 줄이자 사용률이 더 올랐다. 실제 코드에서는 매 step `loss.item()`으로 기록하는 대신 일정 간격으로 기록하거나 GPU에 누적해 두고 가끔 꺼낸다.
3. **연산 가설.** step당 계산량을 4배로 늘리자 사용률이 92%가 되었다. 계산이 작으면 고정 비용(전송·동기화)의 비중이 커진다. 실제로는 batch 크기를 키우는 것이 이에 해당하지만, batch 크기는 최적화 동작도 바꾸므로 성능만 보고 정하지 않는다.
4. **전송 가설**은 이 모형에서 비중이 작다. `transfer`의 복사 횟수를 늘려 사용률이 어떻게 변하는지 직접 확인한다.

실제 GPU 환경이 있다면 같은 순서를 따른다. `nvidia-smi`로 사용률과 메모리를 관찰하고(2.2장), 같은 batch를 반복해 넣는 실험으로 데이터 공급 가설을 먼저 확인하고, `torch.cuda.synchronize()`를 넣어 각 구간의 시간을 바르게 잰다. 측정 도구의 자세한 사용법은 22.4장에서 다룬다.

마지막으로 roofline 판정을 연습문제로 구현한다. 테스트가 바로 채점한다.

::exercise{id="roofline-estimate"}

## 실패 사례

- **GPU가 항상 빠르다고 가정하는 것.** 작은 배열, 작은 batch, 순차적인 반복문, 분기가 많은 계산은 GPU에서 오히려 느릴 수 있다. kernel launch와 CPU–GPU 전송에 고정 비용이 있기 때문이다. 작은 시뮬레이션을 GPU로 옮기기 전에 CPU와 GPU에서 같은 크기로 재 본다.
- **비동기 GPU 연산의 시간을 잘못 측정하는 것.** 동기화 없이 `time.perf_counter()`로 재면 kernel 예약 시간만 잰다. 반대로 `.item()`이나 `print`가 들어간 줄은 앞선 계산 시간을 떠안아 느려 보인다. 구간 앞뒤에서 `torch.cuda.synchronize()`를 부르거나 CUDA event로 재고, warm-up 뒤 여러 번 반복해 평균을 낸다.
- **CPU 계산을 스레드로 병렬화하려는 것.** GIL 때문에 순수 Python 계산은 스레드로 빨라지지 않는다. 프로세스를 쓰거나, NumPy로 벡터화해 컴파일된 코드에 맡긴다.
- **프로세스 수 × 라이브러리 스레드 수가 코어 수를 크게 넘는 것.** 각 프로세스의 BLAS·PyTorch가 코어 수만큼 스레드를 띄우면 서로 경쟁해 느려진다. 프로세스당 스레드 수를 환경 변수로 제한한다.
- **부동소수점을 `==`로 비교하는 것.** 수학적으로 같은 값도 계산 순서에 따라 마지막 비트가 다르다. 테스트와 조건문에서는 허용 오차를 둔 비교를 쓴다.

## 통과 기준

- 자신의 학습·시뮬레이션 코드에서 관찰한 성능 문제 하나를 이 장의 다섯 병목(I/O·연산·메모리 대역폭·전송·동기화) 중 하나로 분류하고, 한 가지만 바꾼 실험의 측정값으로 근거를 제시한다.
- `gpu_starvation_sim.py`의 다섯 실험 결과를 실행 전에 예측하고, 각 결과가 어떤 가설을 지지하거나 기각하는지 설명한다.
- 연습문제 `roofline-estimate`의 테스트를 통과한다.

## 참고 문헌

- Python Software Foundation. *The Python Standard Library* — `threading`, `multiprocessing`, `concurrent.futures`, `asyncio`. Python 공식 문서.
- PyTorch 공식 문서 — CUDA semantics(비동기 실행, 동기화, CUDA event를 이용한 시간 측정). pytorch.org.
- NumPy 공식 문서 — `numpy.finfo`, 배열의 메모리 배치(C order와 Fortran order). numpy.org.
