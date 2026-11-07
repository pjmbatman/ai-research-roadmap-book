---
id: experiment-operations/reproducibility
title: "재현성"
title_en: "Reproducibility"
tier: core
status: draft
prerequisites: [experiment-operations/experiment-tracking, environments/docker]
---

# 재현성

22.1장에서 결과 수치에서 그 run의 설정·코드·데이터로 되돌아가는 경로를 만들었다. 그런데 기록대로 다시 돌렸는데 숫자가 다르게 나온다면 어떻게 해야 하는가. seed를 고정했는데도 결과가 달라지는 일은 딥러닝에서 흔하고, 그 차이가 버그인지 잡음인지 가르지 못하면 디버깅도, 방법 비교도, 다른 사람의 재현도 막힌다.

이 장은 "같은 결과"가 무엇을 뜻하는지부터 정한다. 재현성에는 수준이 있고, 수준마다 기대할 수 있는 일치의 정도가 다르다. 그다음 결과를 바꾸는 무작위성의 출처를 하나씩 찾아 통제하고, 결정적 실행이 무엇을 보장하고 얼마의 비용이 드는지 본다. 마지막으로 통제할 수 없는 변동은 보고하고, 환경은 기록하고 고정한다.

## 학습 목표

- 무작위성의 출처를 통제한다
- 결정성과 성능의 trade-off를 이해한다
- 재현성의 수준을 구분한다

## 재현성의 수준

**왜 필요한가.** "재현이 안 된다"는 말은 여러 뜻이다. 어제 돌린 run을 오늘 같은 서버에서 돌렸더니 마지막 자리가 다르다는 뜻일 수도 있고, 다른 연구실이 논문대로 구현했는데 성능이 5% 낮다는 뜻일 수도 있다. 둘은 원인도, 기대해야 할 일치의 정도도, 해결책도 다르다.

**정의.** 이 책은 재현성을 세 수준으로 나눈다. 같은 단어(repeatability, reproducibility, replicability)를 분야와 기관마다 다른 뜻으로 쓰므로, 용어보다 무엇을 바꾸고 무엇을 고정했는지로 구분한다.

| 수준 | 바뀌는 것 | 고정된 것 | 기대할 수 있는 일치 | 확인하는 것 |
|---|---|---|---|---|
| 같은 환경 재실행 | 실행 시각 | 코드·데이터·설정·seed·환경·하드웨어 | 결정적 설정이면 비트 단위로 같음 | 기록이 빠짐없는가 |
| 다른 환경 재현 | 하드웨어·라이브러리 버전·스레드 수 | 코드·데이터·설정 | 보고된 변동 범위 안 | 결과가 환경에 얼마나 민감한가 |
| 독립 재구현 | 코드(새로 작성)·때로 데이터 | 방법의 설명 | 같은 결론(경향·우열) | 결론이 특정 구현의 우연이 아닌가 |

**직관.** 위로 갈수록 통제가 쉽고 증거는 약하다. 같은 환경 재실행이 실패하면 거의 항상 기록이나 코드의 문제다. 기록되지 않은 설정이 있거나, seed가 닿지 않는 난수가 있거나, 실행마다 다른 알고리즘이 고른다. 다른 환경 재현에서는 비트 단위 일치를 기대하지 않는다. 대신 "원래 결과의 변동 범위 안에 드는가"를 묻는다. 그래서 원 결과가 변동을 보고하지 않았다면 재현을 판정할 근거가 없다. 독립 재구현은 논문이 방법을 충분히 설명했는지까지 시험한다(23.3장).

::quiz{id="reproducibility-levels"}

**연구에서의 의미.** 연구 중에 필요한 것은 주로 첫 번째 수준이다. 코드를 고친 뒤 결과가 바뀌었을 때 그 차이가 수정 때문인지 잡음 때문인지 가르려면, 수정 전 코드로 다시 돌렸을 때 같은 결과가 나와야 한다. 논문을 낼 때 약속하는 것은 두 번째 수준이다. 다른 사람이 다른 기계에서 돌려도 보고한 범위 안의 결과를 얻을 수 있어야 한다. 이 장의 통과 기준이 그것이다.

## 무작위성의 출처

**왜 필요한가.** seed 하나를 고정하면 재현된다고 생각하기 쉽다. 그러나 학습 코드에는 seed가 닿지 않는 무작위성과, 난수가 아닌데도 실행마다 달라지는 연산이 있다. 출처를 모르면 어디를 고정해야 할지 모른다.

**난수 생성기의 출처.** 학습 하나에 여러 난수 생성기가 관여한다.

| 생성기 | 쓰이는 곳 | 고정 방법 |
|---|---|---|
| Python `random` | 파일 목록 섞기, 일부 증강 라이브러리 | `random.seed(s)` |
| NumPy | 데이터 분할, 증강, 전처리 | 전역 `np.random.seed(s)` 또는 `np.random.default_rng(s)` 생성기 |
| PyTorch CPU·GPU | 가중치 초기화, dropout, `randperm` | `torch.manual_seed(s)`(모든 장치에 설정된다) |
| DataLoader worker | worker 안에서 도는 증강 | worker마다 seed를 정하는 `worker_init_fn`, 순서를 정하는 `generator` |

전역 생성기 하나를 여러 용도가 나눠 쓰는 구조는 그 자체로 재현을 깨뜨린다. 앞에서 난수를 몇 개 소비하느냐에 따라 뒤의 모든 난수가 밀리기 때문이다. 증강 코드 한 줄을 추가하면 증강과 무관한 데이터 순서까지 바뀐다. NumPy의 `SeedSequence`로 용도마다 독립된 생성기를 만들면 이 문제가 사라진다.

```python {exec="browser" session="repro"}
import numpy as np

def setup(add_augmentation, per_purpose):
    """가중치 초기화와 데이터 순서를 만든다. add_augmentation이면 그 사이에 증강 난수를 쓴다."""
    if per_purpose:
        init_ss, data_ss, aug_ss = np.random.SeedSequence(0).spawn(3)
        w = np.random.default_rng(init_ss).normal(size=3)
        if add_augmentation:
            np.random.default_rng(aug_ss).uniform(size=5)
        order = np.random.default_rng(data_ss).permutation(6)
    else:
        np.random.seed(0)                              # 전역 상태 하나를 모두가 나눠 쓴다
        w = np.random.normal(size=3)
        if add_augmentation:
            np.random.uniform(size=5)                  # 새로 넣은 증강이 난수를 먼저 소비한다
        order = np.random.permutation(6)
    return w, order

for per_purpose in [False, True]:
    (w0, o0), (w1, o1) = setup(False, per_purpose), setup(True, per_purpose)
    name = "용도별 RNG" if per_purpose else "전역 RNG  "
    print(f"{name}: 초기화 같음 {np.array_equal(w0, w1)}, 데이터 순서 같음 {np.array_equal(o0, o1)}"
          f"  {o0} → {o1}")
```

```output
전역 RNG  : 초기화 같음 True, 데이터 순서 같음 False  [4 0 2 5 3 1] → [4 0 3 5 1 2]
용도별 RNG: 초기화 같음 True, 데이터 순서 같음 True  [4 3 2 0 1 5] → [4 3 2 0 1 5]
```

전역 생성기에서는 증강 한 줄 때문에 데이터 순서가 `[4 0 2 5 3 1]`에서 `[4 0 3 5 1 2]`로 바뀌었다. 초기화는 증강보다 먼저 난수를 썼으므로 운 좋게 같다. 용도별 생성기에서는 둘 다 그대로다. 실습의 연습문제에서 이 아이디어를 "이름으로 정해지는 스트림"으로 확장한다.

::quiz{id="global-rng-shift"}

**DataLoader worker.** `num_workers > 0`이면 데이터 로딩이 별도 프로세스에서 돈다. PyTorch는 worker마다 자기 난수 seed를 다르게 정하지만, NumPy나 `random` 같은 다른 라이브러리의 상태를 어떻게 다룰지는 코드가 책임진다. PyTorch 공식 문서의 재현성 노트는 다음 패턴을 권한다(집필 시점 예시).

```python
# 집필 시점 예시: PyTorch 재현성 노트의 권장 패턴
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32      # worker마다 다른 PyTorch seed에서 유도
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)                                    # shuffle 순서를 정하는 생성기
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker, generator=g)
```

worker 수를 바꾸면 각 worker가 어떤 샘플을 처리하는지가 바뀌므로, worker 안의 증강 난수가 다른 샘플에 배정된다. worker 수도 설정의 일부로 기록한다.

**난수가 아닌 비결정성.** 난수를 모두 고정해도 결과가 달라지는 원인이 남는다. 근본 원인은 **부동소수점 덧셈의 결합법칙이 성립하지 않는다**는 것이다(9.1장). $(a + b) + c$와 $a + (b + c)$는 반올림이 다른 위치에서 일어나 다른 값이 될 수 있다. 병렬 계산은 덧셈을 여러 조각으로 나눠 동시에 하고 나중에 합치므로, 나누는 방식이 바뀌면 결과의 마지막 자리가 바뀐다. 같은 float32 수 10만 개를 조각 수만 바꿔 더해 본다.

```python {exec="browser" session="repro"}
import math

rng = np.random.default_rng(0)
x = (rng.normal(size=100_000) * 1000).astype(np.float32)        # (100000,)

def chunked_sum(x, n_chunks):
    """n개 스레드가 조각을 나눠 더한 뒤 합치는 것을 흉내 낸다(각 조각은 앞에서부터 차례로 더한다)"""
    parts = [np.cumsum(c)[-1] for c in np.array_split(x, n_chunks)]
    return np.cumsum(np.array(parts, dtype=np.float32))[-1]

exact = math.fsum(x.astype(np.float64))
for n in [1, 2, 4, 8]:
    s = chunked_sum(x, n)
    print(f"조각 {n}개: {float(s):.4f}   정확한 합과의 차이 {float(s) - exact:+.4f}")
print("같은 조각 수로 두 번 더하면 같은가?", chunked_sum(x, 4) == chunked_sum(x, 4))
```

```output
조각 1개: -90825.2266   정확한 합과의 차이 -0.1470
조각 2개: -90824.7500   정확한 합과의 차이 +0.3295
조각 4개: -90825.1562   정확한 합과의 차이 -0.0767
조각 8개: -90824.9297   정확한 합과의 차이 +0.1498
같은 조각 수로 두 번 더하면 같은가? True
```

같은 수들의 합이 조각 수에 따라 마지막 자리에서 달라진다. 어느 쪽도 틀린 것이 아니다. 모두 float32 정밀도 안에서 정확한 합 근처에 있다. 그리고 **같은 방식으로 나누면 몇 번을 더해도 같다.** 비결정성은 덧셈 자체가 아니라 덧셈 순서가 실행마다 바뀌는 데서 온다. 실제 학습에서 순서를 바꾸는 것은 다음과 같다.

- **스레드 수.** CPU 연산은 스레드 수에 따라 reduction을 나누는 방식이 바뀐다. 같은 코드를 코어 수가 다른 기계에서 돌리면 결과가 다를 수 있다.
- **GPU atomic 연산.** 여러 스레드가 같은 메모리 위치에 더하는 연산(예: `index_add_`, 일부 역전파 kernel)은 스레드가 끝나는 순서대로 더해지므로 실행마다 순서가 다르다.
- **cuDNN 알고리즘 선택.** `torch.backends.cudnn.benchmark = True`이면 첫 호출 때 여러 convolution 알고리즘의 시간을 재서 가장 빠른 것을 고른다. 시간 측정은 실행마다 달라 다른 알고리즘이 골라질 수 있다.
- **라이브러리 버전과 하드웨어.** 버전이 바뀌면 kernel 구현이 바뀌고, GPU 모델이 바뀌면 같은 라이브러리도 다른 kernel을 쓴다.

PyTorch로 같은 학습을 스레드 수만 바꿔 돌려 본다. 로컬 PyTorch가 필요하므로 브라우저가 아닌 로컬 Python에서 실행하며, 출력은 이 책의 실습 환경(CPU)에서 얻은 것이다.

```python {exec="python"}
import torch

def train(seed, threads, steps=100):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    X = torch.randn(4096, 16)                                     # (N, D)
    Y = torch.sin(X @ torch.randn(16, 16))                        # (N, D)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 16))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])   # (P,)

a, b = train(seed=0, threads=1), train(seed=0, threads=1)
c, d = train(seed=0, threads=2), train(seed=1, threads=1)
print("같은 seed·같은 스레드 수, 두 번: 비트 단위로 같음", torch.equal(a, b))
print("같은 seed·스레드 1개 대 2개:    비트 단위로 같음", torch.equal(a, c),
      f"  최대 차이 {(a - c).abs().max().item():.1e}")
print(f"seed 0 대 seed 1:              최대 차이 {(a - d).abs().max().item():.1e}")
```

```output
같은 seed·같은 스레드 수, 두 번: 비트 단위로 같음 True
같은 seed·스레드 1개 대 2개:    비트 단위로 같음 False   최대 차이 3.0e-08
seed 0 대 seed 1:              최대 차이 4.9e-01
```

같은 seed와 같은 스레드 수로 두 번 돌리면 비트 단위로 같다. 스레드 수만 1개에서 2개로 바꾸면 파라미터가 $10^{-8}$ 크기로 달라진다. seed를 바꿀 때의 차이(약 0.5)와 비교하면 무시할 만하지만, 0은 아니다. 이 작은 차이는 학습이 길고 비볼록할수록 커질 수 있고, early stopping의 시점이나 argmax 같은 경계 판정을 뒤집을 수 있다.

::quiz{id="seed-not-enough"}

::quiz{id="float-order"}

## 결정적 실행

**왜 필요한가.** 같은 환경 재실행을 비트 단위로 같게 만들어야 하는 순간이 있다. 코드를 리팩터링한 뒤 결과가 같은지 확인하는 회귀 테스트, 드물게 나타나는 NaN을 같은 step에서 다시 일으켜 원인을 찾는 디버깅이 그렇다. 실행마다 결과가 흔들리면 "수정 전과 후가 다르다"는 관찰이 아무 정보도 주지 않는다.

**정의.** 결정적(deterministic) 실행은 같은 입력·같은 소프트웨어·같은 하드웨어에서 항상 같은 출력을 내는 실행이다. PyTorch에서는 다음을 함께 설정한다(집필 시점 예시, 최신 목록은 공식 문서의 재현성 노트에서 확인한다).

```python
# 집필 시점 예시: PyTorch 결정적 실행 설정
import os, random
import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # CUDA 10.2 이상의 cuBLAS 결정성(첫 CUDA 호출 전에)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = False              # 알고리즘 자동 선택 끄기
torch.use_deterministic_algorithms(True)            # 결정적 구현 강제, 없으면 RuntimeError
```

`torch.use_deterministic_algorithms(True)`는 결정적 구현이 있는 연산은 그 구현을 쓰게 하고, 비결정적 구현만 있는 연산은 호출하는 순간 `RuntimeError`를 내게 한다. 조용히 비결정적으로 도는 것보다 낫다. 어느 연산이 문제인지 바로 알 수 있기 때문이다. 오류 대신 경고만 받고 싶으면 `warn_only=True`를 준다. cuBLAS 관련 환경변수의 정확한 값은 PyTorch가 오류 메시지와 문서에서 안내하는 값을 따른다.

**비용.** 결정성은 공짜가 아니다. PyTorch 문서도 결정적 연산이 대개 비결정적 연산보다 느리다고 밝힌다. 순서를 고정하려면 atomic 덧셈 대신 정렬이나 별도 reduction을 써야 하고, cuDNN benchmark를 끄면 가장 빠른 알고리즘을 포기한다. 비용은 모델과 하드웨어에 따라 거의 없기도 하고 크기도 하므로, 켜기 전과 후의 step 시간을 같은 조건에서 재 본다(22.4장).

::quiz{id="deterministic-tradeoff"}

**보장의 범위.** 결정적 실행은 "같은 소프트웨어·같은 하드웨어"에서의 일치만 보장한다. PyTorch 문서는 릴리스·커밋·플랫폼이 바뀌면 완전히 같은 결과를 보장하지 않는다고 밝힌다. 다른 GPU 모델로 옮기면 결정적 설정을 켜도 결과는 달라질 수 있다. 결정성은 첫 번째 수준(같은 환경 재실행)의 도구이고, 두 번째 수준은 아래의 변동 보고로 다룬다.

**언제 켜는가.** 실용적인 정책은 다음과 같다.

- **켠다:** 회귀 테스트, 버그 재현, 리팩터링 전후 비교, 작은 규모의 디버깅 run.
- **기본 설정으로 둔다:** 성능이 중요한 본 실험과 대규모 sweep. 대신 seed를 여러 개 돌려 변동을 보고한다.
- 어느 쪽이든 **설정 자체를 기록한다.** 결정적 설정을 켰는지, cuDNN benchmark를 켰는지, 스레드 수가 몇인지가 run 기록(22.1장)에 남아야 한다.

::quiz{id="when-determinism"}

## 변동성의 보고

**왜 필요한가.** 통제할 수 없거나 통제하지 않기로 한 변동은 보고해야 한다. 결정적으로 돌린 run 하나는 "다시 돌리면 같다"는 것만 보장할 뿐, 그 결과가 다른 seed에서도 나오는 대표적인 값인지는 말해 주지 않는다. Henderson 외(2018)는 심층 강화학습에서 같은 알고리즘·같은 설정이라도 seed에 따라 결과가 크게 달라져, seed 몇 개로 낸 비교가 결론을 뒤집을 수 있음을 보였다.

**무엇을 보고하는가.** seed·데이터 분할에 따른 변동을 어떻게 측정하고 비교하는지는 12.5장에서 자세히 다뤘다. 여기서는 재현성의 관점에서 필요한 최소한만 정리한다.

- **실행 횟수와 무엇을 바꿨는지.** "seed 5개(초기화·데이터 순서), 분할 고정"처럼.
- **평균과 퍼짐.** 표준편차 또는 신뢰구간, 그리고 어느 것인지 명시한다.
- **실패한 run의 수.** 발산하거나 중단된 run을 빼고 평균을 냈다면 그 수를 함께 쓴다(22.1장).
- **결정성 설정과 환경.** 결과를 낸 하드웨어와 라이브러리 버전.

**재현 판정.** 변동을 보고하면 다른 환경 재현을 숫자로 판정할 수 있다. 원 결과가 seed $n$개의 평균 $\bar{x}$와 표준편차 $s$로 보고되었고, 재현하는 쪽이 seed $m$개로 평균 $\bar{y}$를 얻었다고 하자. 두 환경의 seed 간 변동이 같다고 가정하면, 새 seed $m$개 평균의 95% 예측구간은

$$
\bar{x} \pm t_{0.975,\,n-1}\, s \sqrt{\frac{1}{n} + \frac{1}{m}}
$$

이다. $t_{0.975,\,n-1}$은 자유도 $n-1$인 t분포의 97.5% 분위수다. 첫 항 $1/n$은 원 평균 자체의 불확실성, 둘째 항 $1/m$은 새 평균의 불확실성이다. $\bar{y}$가 이 구간 안에 있으면 "보고된 변동 범위 안에서 재현되었다"고 할 수 있다.

```python {exec="browser" session="repro"}
from scipy import stats

reported = np.array([0.912, 0.905, 0.918, 0.909, 0.914])         # 논문: seed 5개의 정확도
reproduced = {"기계 B": np.array([0.910, 0.907, 0.913]),           # 다른 기계에서 seed 3개
              "기계 C": np.array([0.893, 0.889, 0.897])}

n, m_bar, s = len(reported), reported.mean(), reported.std(ddof=1)
print(f"보고: {m_bar:.4f} ± {s:.4f} (seed {n}개, 표준편차)")
for name, r in reproduced.items():
    m = len(r)
    half = stats.t.ppf(0.975, df=n - 1) * s * np.sqrt(1 / n + 1 / m)   # 새 seed m개 평균의 95% 예측구간
    inside = abs(r.mean() - m_bar) <= half
    print(f"{name}: 평균 {r.mean():.4f}, 허용 범위 {m_bar - half:.4f}–{m_bar + half:.4f} → 재현 {inside}")
```

```output
보고: 0.9116 ± 0.0049 (seed 5개, 표준편차)
기계 B: 평균 0.9100, 허용 범위 0.9016–0.9216 → 재현 True
기계 C: 평균 0.8930, 허용 범위 0.9016–0.9216 → 재현 False
```

기계 B의 결과는 보고값과 비트 단위로 다르지만 변동 범위 안에 있다. 기계 C는 범위를 벗어났다. 이때는 seed를 더 돌리기 전에 환경 차이를 먼저 의심한다. 라이브러리 버전, 데이터 버전(manifest 해시), 기본값이 바뀐 설정이 흔한 원인이다. 이 판정은 seed 간 변동이 두 환경에서 같다는 가정에 기대며, seed 수가 적으면 구간이 넓어져 판정이 관대해진다는 점도 함께 기억한다.

:::note[분야 예시]
분자동역학 시뮬레이션은 같은 초기 구조에서 출발해도 초기 속도 난수, 병렬 분해 방식, 부동소수점 연산 순서에 따라 궤적이 점점 갈라진다. 혼돈적인 동역학에서는 작은 차이가 지수적으로 커지므로, 개별 궤적을 비트 단위로 맞추는 것보다 여러 궤적에서 구한 평균 물리량(확산 계수, 자유 에너지)과 그 오차 막대를 재현의 단위로 삼는다. 딥러닝 학습의 "seed 여러 개, 평균과 퍼짐" 관행과 같은 생각이다.
:::

## 환경 고정

**왜 필요한가.** 다른 환경 재현이 범위를 벗어났을 때 가장 먼저 확인할 것은 환경 차이다. 그러려면 원래 환경이 기록되어 있어야 하고, 가능하면 그 환경을 다시 만들 수 있어야 한다.

**세 겹의 고정.** 환경은 층마다 다른 도구로 고정한다(3부).

| 층 | 고정 수단 | 기록할 것 | 다루는 곳 |
|---|---|---|---|
| Python 패키지 | lockfile(`uv.lock` 등)로 정확한 버전과 해시를 고정 | lockfile 자체를 저장소에 커밋 | 3.4장 |
| 시스템 라이브러리·CUDA | 컨테이너 image를 digest로 고정 | image digest, 빌드한 커밋 | 3.5장 |
| 하드웨어·드라이버 | 고정할 수 없다. 기록한다 | GPU 모델, 드라이버, CPU, 스레드 수 | 3.6장 |

하드웨어는 고정할 수 없으므로 기록이 전부다. 3.6장의 환경 보고서를 run마다 `env.json`으로 저장하고(22.1장), 결과가 다를 때 두 보고서를 비교한다. 보고서를 짧은 지문(fingerprint)으로 요약해 두면 "이 run들은 같은 환경에서 나왔는가"를 한눈에 가를 수 있다. 지문은 보고서를 키 순서대로 직렬화해 hash한 값이다. 다만 지문은 같은지 다른지만 알려 주므로, 다를 때 무엇이 다른지는 보고서 원본을 비교해 찾는다.

**재현에 필요한 최소 묶음.** 논문과 함께 공개하거나 연구실에 보관할 재현 묶음은 다음과 같다.

1. 코드: 결과를 낸 커밋(태그를 붙여 둔다).
2. 환경: lockfile, 컨테이너를 썼다면 image digest와 Dockerfile.
3. 데이터: 데이터셋 manifest 해시와 분할 파일(15.5장), 데이터를 얻는 방법.
4. 설정: 결과 표의 각 칸을 만든 resolved config와 seed 목록(22.1장).
5. 기대 결과: 평균과 퍼짐, 결과를 낸 하드웨어, 결정성 설정.

5번이 빠지면 나머지가 모두 있어도 재현 판정을 할 수 없다.

:::note[분야 예시]
기후·기상 모델 커뮤니티에서는 컴파일러, 최적화 플래그, 프로세서 수가 바뀌면 같은 모델 코드도 비트 단위로 다른 결과를 낸다는 것이 잘 알려져 있다. 그래서 모델 출력 메타데이터에 컴파일 환경과 병렬 구성을 함께 남기고, 플랫폼을 옮길 때는 개별 출력의 일치 대신 기후 통계(평균장, 변동성)가 통계적으로 구별되지 않는지를 확인하는 방식이 쓰인다.
:::

## 실습

같은 실험을 결정적 설정과 기본 설정으로 각각 여러 번 실행해 결과 변동을 측정한다.

1. 22.1장 실습의 학습 스크립트에 결정성 옵션(`--deterministic`)을 추가한다. 켜면 위 "결정적 실행"의 설정을 모두 적용하고, 그 사실을 run 기록에 남긴다.
2. 같은 seed로 (a) 결정적 설정 5번, (b) 기본 설정 5번 돌린다. GPU가 있으면 `cudnn.benchmark = True`를 기본 설정에 포함한다. CPU뿐이라면 스레드 수(`torch.set_num_threads`)를 바꾼 run을 (b)에 섞는다.
3. 서로 다른 seed 5개로 기본 설정을 5번 돌린다.
4. 세 묶음에서 최종 검증 metric의 범위(최댓값 − 최솟값)와 표준편차를 계산해 표로 만든다. 기대하는 결과는 (a)의 범위가 0, (b)의 범위가 작은 양수, (c)의 범위가 (b)보다 훨씬 큰 것이다. (b)와 (c)의 크기 비교가 "결정성을 포기해도 되는가"에 대한 답이다.
5. (a)와 기본 설정의 step당 시간을 같은 조건에서 재서 결정성의 비용을 기록한다.

이어서 용도별 난수 스트림을 연습문제로 구현한다. 스트림이 seed와 이름만으로 정해지면, 코드에 새 난수 용도를 추가해도 기존 실험의 난수는 바뀌지 않는다.

::exercise{id="keyed-rng-streams"}

## 실패 사례

- **seed 하나 고정으로 재현성이 보장된다고 생각하는 것.** seed는 난수 생성기의 출발점만 고정한다. DataLoader worker의 난수, 스레드 수, cuDNN 알고리즘 선택, GPU의 atomic 덧셈, 라이브러리 버전이 남는다.
- **결정적으로 돌린 run 하나를 결과로 보고하는 것.** 다시 돌리면 같은 값이 나오지만, 그 값이 seed에 따라 얼마나 흔들리는지는 모른다.
- **Python 내장 `hash()`로 seed를 만드는 것.** 문자열의 `hash`는 프로세스마다 다른 값을 낸다(`PYTHONHASHSEED`로 고정하지 않는 한). 한 세션에서는 재현되는 것처럼 보이다가 다음 날 달라진다.
- **다른 기계의 결과가 비트 단위로 다르다는 이유로 재현 실패라고 판단하는 것.** 다른 환경 재현의 기준은 보고된 변동 범위다.
- **환경을 기록하지 않고 "최신 버전을 설치하라"고 안내하는 것.** 몇 달 뒤의 최신 버전은 다른 kernel과 다른 기본값을 가진다.

## 통과 기준

- 다른 사람이 다른 머신에서 재현 묶음(코드 커밋, lockfile 또는 image digest, 데이터 manifest, 설정, 기대 결과)만으로 실험을 다시 돌려, 결과가 보고된 변동 범위 안에 드는지를 예측구간으로 판정한다.
- 실습의 세 묶음(결정적·기본·다른 seed)의 변동을 표로 만들고, 결정적 설정의 step 시간 비용을 함께 보고한다.
- 연습문제 `keyed-rng-streams`의 테스트를 통과한다.

## 참고 문헌

- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1). — seed·구현 차이에 따른 결과 변동과 재현 가능한 보고를 위한 권고
- Bouthillier 외 (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems 3 (MLSys 2021)*. — 데이터 샘플링·초기화·하이퍼파라미터 선택에서 오는 변동과 벤치마크 비교 방법
- Sandve, Nekrutenko, Taylor, Hovig (2013). Ten simple rules for reproducible computational research. *PLoS Computational Biology*, 9(10), e1003285. — 외부 프로그램 버전 보관, 난수 seed 기록 등 계산 재현성의 기본 규칙
