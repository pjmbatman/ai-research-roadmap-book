---
id: deep-learning-pytorch/debugging-training
title: "학습 실패 진단"
title_en: "Debugging Neural Network Training"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/dashboards]
---

# 학습 실패 진단

신경망 학습의 버그는 대부분 오류 메시지를 내지 않는다. label이 입력과 어긋나도, 모델 끝에 softmax가 한 번 더 붙어도, gradient를 지우지 않아도 코드는 끝까지 돌고 손실은 어느 정도 내려간다. 결과가 "기대보다 조금 나쁘다"로 나타나기 때문에, 모델 구조나 하이퍼파라미터 탓으로 돌리고 몇 주를 튜닝에 쓰기 쉽다.

그래서 학습 실패의 진단에는 절차가 필요하다. 6.6장에서 다룬 체계적 디버깅(관찰 고정 → 가설 → 예측과 시험 → 범위 축소 → 회귀 테스트)을 신경망 학습에 맞게 구체화한 것이 이 장의 체크리스트다. 17.8장에서 기록한 숫자들이 이 장의 증거가 된다.

## 학습 목표

- 학습 실패를 체계적으로 진단한다
- 디버깅 체크리스트를 사용한다

## 단계적 접근

큰 모델, 큰 데이터, 긴 학습에서는 실패 하나를 확인하는 데 몇 시간이 걸린다. 진단의 기본 전략은 **실패를 가장 작고 빠른 설정에서 재현**하는 것이고, 그 반대 방향으로 학습을 쌓아 올리는 것이다. 각 단계는 통과 기준이 분명하고, 앞 단계를 통과해야 다음 단계로 간다.

1. **데이터를 확인한다.** 학습 코드를 돌리기 전에 입력과 label을 직접 본다. 값의 범위, 단위, 결측, 상수 특징, class 비율, train과 test 사이의 중복, 그리고 몇 개 표본을 사람이 읽어 label이 맞는지.
2. **단순한 baseline을 돌린다.** 평균 예측, 최다 class 예측, 선형 모델(16장). 신경망이 이것보다 나쁘면 신경망 이전의 문제다. 선형 모델조차 우연 수준이면 입력과 label의 관계부터 의심한다.
3. **초기 손실을 확인한다.** 학습 전 손실이 이론값과 맞는지 본다. $C$개 class 분류에서 모든 class에 같은 확률을 주면 손실은 $\ln C$이고, 회귀에서 0을 예측하면 손실은 목표값의 제곱 평균이다.
4. **batch 하나에 과적합한다.** 표본 수십 개로 수백 step 학습해 손실이 0 근처로 가는지 본다. 모델·손실·역전파·업데이트의 연결이 맞는지 보는 가장 싼 검사다.
5. **규모를 키운다.** 전체 데이터로 짧게 학습해 train 손실이 baseline보다 내려가는지, 그다음 검증 곡선, 그다음 하이퍼파라미터 탐색 순서로 간다.

3과 4를 코드로 본다. class 5개, 표본 16개짜리 batch 하나에서 정상 모델, 끝에 softmax를 하나 더 붙인 모델, label을 무작위로 섞은 경우를 비교한다.

```python {exec="python"}
import math

import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(16, 10)                                   # (B, D) batch 하나
y = torch.randint(0, 5, (16,))                            # (B,) class 5개

def overfit(head, labels, steps=300):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5), *head)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps + 1):
        loss = nn.functional.cross_entropy(model(x), labels)
        first = loss.item() if step == 0 else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first, loss.item()

print(f"기대 초기 손실 ln 5 = {math.log(5):.3f}")
for name, head, labels in [("정상", [], y), ("softmax 두 번", [nn.Softmax(dim=1)], y),
                           ("label 무작위", [], y[torch.randperm(16)])]:
    first, last = overfit(head, labels)
    print(f"{name:12s} 초기 손실 {first:.3f}   300 step 뒤 {last:.4f}")
```

```output
기대 초기 손실 ln 5 = 1.609
정상           초기 손실 1.663   300 step 뒤 0.0003
softmax 두 번  초기 손실 1.617   300 step 뒤 0.9674
label 무작위    초기 손실 1.583   300 step 뒤 0.0004
```

세 경우 모두 초기 손실은 $\ln 5 = 1.609$ 근처라 3단계는 아무것도 걸러내지 못했다. 4단계는 softmax 중복을 잡았다. `cross_entropy`는 내부에서 log-softmax를 계산하므로, 이미 확률인 값에 softmax를 다시 적용하면 출력이 좁은 범위에 갇혀 손실이 0.97 아래로 내려가지 못한다. 그러나 **무작위 label도 0.0004까지 내려갔다.** 신경망은 16개 정도는 아무 label이나 외운다. batch 하나 과적합은 "학습 경로가 연결되어 있는가"를 보는 검사이지, "데이터가 맞는가"를 보는 검사가 아니다. 1, 2단계가 따로 있는 이유다.

::quiz{id="overfit-one-batch-limits"}

1단계의 점검은 몇 줄로 자동화할 수 있다. 실제 데이터에서 흔한 문제를 일부러 심은 표를 점검해 본다.

```python {exec="browser" session="debug"}
import numpy as np

rng = np.random.default_rng(2)
X = rng.normal([300.0, 1.0, 0.5, 7.0], [10.0, 0.1, 0.2, 0.0], size=(500, 4))   # (N, 4)
X[rng.choice(500, 3, replace=False), 1] = np.nan          # 결측 3개
y = (X[:, 0] > 300).astype(int)
y[:40] = 1                                                # 앞쪽 40개 label이 잘못 붙음
train, test = X[:400], np.vstack([X[400:490], X[:10]])    # test에 train 표본 10개가 섞임

print("특징별 평균  ", np.round(np.nanmean(X, 0), 3))
print("특징별 표준편차", np.round(np.nanstd(X, 0), 3))
print("NaN 개수     ", np.isnan(X).sum(0))
print("상수 특징     ", np.where(np.nanstd(X, 0) == 0)[0])
print("label 비율    ", np.bincount(y) / len(y))
dup = {r.tobytes() for r in train} & {r.tobytes() for r in test}
print("train과 test에 함께 있는 행", len(dup))
print("앞 50개와 나머지의 label 1 비율", y[:50].mean().round(3), y[50:].mean().round(3))
```

```output
특징별 평균   [299.266   1.001   0.489   7.   ]
특징별 표준편차 [10.292  0.095  0.199  0.   ]
NaN 개수      [0 3 0 0]
상수 특징      [3]
label 비율     [0.494 0.506]
train과 test에 함께 있는 행 10
앞 50개와 나머지의 label 1 비율 0.92 0.46
```

평균과 표준편차에서 첫 특징이 다른 특징보다 수백 배 큰 스케일(온도 K)이라 정규화가 필요하다는 것, 두 번째 특징에 결측이 3개 있다는 것, 네 번째 특징은 상수라 정보가 없다는 것이 보인다. train과 test에 같은 행이 10개 있고(15.4장), 데이터 앞쪽의 label 비율이 나머지와 크게 다르다. 마지막 것은 파일을 만든 과정(예: 특정 장비나 날짜의 측정이 앞에 모여 있음)이나 label을 붙이는 과정의 오류를 의심하게 한다. 어느 것도 학습 코드의 오류 메시지로는 드러나지 않는다.

## 증상별 진단

곡선에 나타난 증상에서 출발해 가능한 원인을 좁히는 표다. 원인 열은 **확인하기 쉬운 것부터** 적었다.

| 증상 | 먼저 확인할 것 | 그다음 |
|---|---|---|
| 손실이 처음부터 줄지 않음 | 초기 손실이 이론값인가, gradient가 0이 아닌가, optimizer에 파라미터가 등록되었는가, 학습률이 0이나 너무 작지 않은가 | batch 하나 과적합, 입력 정규화, 활성값 포화 |
| 손실이 줄다가 NaN·inf | 기록에서 처음 이상이 나온 step, 그 step의 batch에 NaN·극단값이 있는가 | 학습률·warm-up, gradient clipping, log·나눗셈·sqrt 같은 연산의 정의역, mixed precision |
| 손실이 크게 흔들리거나 발산 | 업데이트 비율, gradient norm의 spike | 학습률, batch 크기, 이상치 batch |
| train은 좋은데 검증이 정체·악화 | train과 검증의 전처리가 같은가, 평가 모드 | 과적합(17.8장), 분포 차이(21.4장) |
| 검증 성능이 기대보다 훨씬 좋음 | train–검증 중복, 정답을 담은 특징, 평가가 train 데이터를 쓰는가 | 분할 단위(시료·환자·분자, 15.4장) |
| 실행마다 결과가 크게 다름 | 시드 고정, 평가의 결정성 | 데이터 순서·초기화에 대한 민감도(22.2장) |

**이상하게 좋은 성능**은 특별히 다룰 가치가 있다. 성능이 나쁠 때는 누구나 코드를 의심하지만, 좋을 때는 의심하지 않는다. 과거 연구가 70%에 머문 문제에서 99%가 나왔다면, 축하하기 전에 누수를 찾는다. 가장 흔한 것은 같은 대상에서 나온 표본이 train과 검증에 나뉘어 들어간 것이고, 그다음이 정답에서 계산된 특징이나 측정 순서처럼 정답과 우연히 연결된 정보다.

::widget{name="learning-curve-reader"}

::quiz{id="suspiciously-good"}

:::note[분야 예시]
의료 영상 분류 모델은 질병 자체가 아니라 병원·장비마다 다른 영상 표식(글자, 눈금자, 촬영 조건)을 학습해 높은 점수를 낼 수 있다. 중증 환자가 특정 병원에서 주로 촬영되면 그 병원의 표식이 곧 label의 단서가 된다. 재료 데이터에서도 같은 연구 그룹이 같은 합성법으로 만든 시료들이 비슷한 특성을 가져, 무작위 분할에서는 점수가 높고 새 그룹의 시료에서는 무너지는 경우가 있다. 이상하게 좋은 성능을 만나면 "모델이 무엇을 보고 맞혔는가"를 묻는다(21.5장).
:::

## 흔한 원인

증상 표의 원인들은 실제로 몇 가지 범주로 모인다. 각 범주에 대해 **확인 방법**을 함께 적는다.

- **데이터·label 오류.** 입력과 label이 어긋남(섞을 때 한쪽만 섞음, 정렬 기준이 다른 두 파일을 합침), label의 단위·부호·인코딩 오류, 결측을 0으로 채움. → 표본을 직접 보고, 선형 baseline의 성능을 확인한다.
- **정규화 누락.** 스케일이 수백 배 다른 특징을 그대로 넣거나, 목표값을 표준화하지 않음. 초기 출력이 커져 초기 손실이 이론값에서 벗어나고, 학습률 하나로는 모든 방향에 맞출 수 없다(10.3장). → 특징별 평균·표준편차, 초기 손실.
- **학습률.** 너무 크면 흔들리거나 발산하고, 너무 작으면 아주 천천히 내려간다. → 업데이트 비율, 학습률 탐색(10.4장).
- **초기화.** 가중치의 스케일이 층을 지날수록 활성값을 포화시키거나 0으로 줄인다(17.1장). → 층별 활성값 통계(아래 도구 절).
- **mask.** padding 위치가 손실이나 attention에 들어감, mask의 0과 1이 뒤바뀜, 미래 시점을 가리지 않음(누수). → 손실을 mask 위치만으로 다시 계산해 비교, 미래를 가린 입력으로 예측이 달라지는지 확인.
- **eval 모드.** 평가에서 `model.eval()` 누락, 평가 뒤 `model.train()` 누락(17.6장). → 같은 입력을 두 번 평가해 같은 값이 나오는지.
- **손실 집계.** batch 평균의 평균, sum과 mean의 혼동(학습률이 batch 크기만큼 바뀐 효과), gradient accumulation에서 나누기 누락(10.4장). → batch 크기를 바꿔도 같은 값이 나오는지.
- **학습 루프 순서.** `zero_grad` 누락으로 gradient 누적, scheduler 호출 주기 오류(17.5장). → 학습률 0으로 같은 batch를 두 번 통과시켜 gradient가 같은지.

## 도구

증상을 원인으로 좁히는 데 쓰는 도구들이다. 모두 "어디서 처음 이상해졌는가"를 찾는 도구라는 공통점이 있다.

**gradient norm 추적.** 17.8장에서처럼 step마다 global gradient norm을 기록하고, 층별 norm은 더 긴 주기로 기록한다. 발산은 손실보다 gradient norm에서 먼저 보이고, 특정 층의 norm만 0에 붙어 있으면 그 층 아래에서 gradient가 사라지고 있다. 기록이 길면 사람이 다 볼 수 없으므로, **최근 정상값의 중앙값보다 몇 배 큰 step**을 자동으로 표시해 둔다(연습문제).

**처음 비유한 값의 위치.** 손실이 NaN이 되면 그 NaN이 어디서 처음 생겼는지 찾아야 한다. forward에서 생긴 것은 모듈마다 출력을 검사하는 forward hook(17.3장)으로, backward에서만 생긴 것은 PyTorch의 anomaly detection으로 찾는다. 아래 예는 첫 입력 특징(농도)의 로그를 취하는 모델에 농도 0인 표본 하나를 넣는다.

```python {exec="python"}
import warnings

class LogFeature(nn.Module):
    """농도 같은 양수 특징의 로그를 취한다(0이 들어오면 -inf)."""
    def forward(self, x):
        return torch.cat([torch.log(x[:, :1]), x[:, 1:]], dim=1)

torch.manual_seed(0)
model = nn.Sequential(LogFeature(), nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
x = torch.rand(6, 3)                                      # (B, 3)
x[4, 0] = 0.0                                             # 농도 0인 표본 하나
first_bad = []
def find_nonfinite(module, inputs, output):
    if not first_bad and not torch.isfinite(output).all():
        rows = (~torch.isfinite(output)).any(1).nonzero().flatten().tolist()
        first_bad.append(f"{module.__class__.__name__} (표본 {rows})")
for m in model:
    m.register_forward_hook(find_nonfinite)
loss = model(x).pow(2).mean()
loss.backward()
print(f"손실 {loss.item():.4f}, 유한한가 {torch.isfinite(loss).item()}")
print("처음 비유한 출력을 낸 모듈:", first_bad[0])
print("NaN gradient:", [n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()])

w = torch.zeros(3, requires_grad=True)                    # backward에서만 생기는 NaN
with warnings.catch_warnings(record=True) as caught, torch.autograd.set_detect_anomaly(True):
    try:
        torch.sqrt((w ** 2).sum()).backward()             # |w|를 w = 0에서 미분
    except RuntimeError as e:
        print("anomaly mode:", str(e))
```

```output
손실 0.1516, 유한한가 True
처음 비유한 출력을 낸 모듈: LogFeature (표본 [4])
NaN gradient: ['1.weight']
anomaly mode: Function 'PowBackward0' returned nan values in its 0th output.
```

이 예는 NaN 진단에서 흔히 놓치는 점을 보여 준다.

- **손실은 유한했다.** $\log 0 = -\infty$가 선형층을 지나 $\pm\infty$가 되었지만, tanh가 이를 $\pm 1$로 눌러 손실은 0.1516이 되었다. 손실만 감시했다면 문제를 몰랐을 것이다.
- **gradient는 NaN이 되었다.** 포화된 tanh의 미분은 0이고, 첫 선형층 가중치의 gradient는 그 0과 입력 $-\infty$의 곱이라 NaN이다. 한 번의 업데이트로 그 가중치 전체가 NaN이 된다.
- **forward hook은 원인의 위치(`LogFeature`, 표본 4)를 바로 짚었다.** 데이터 쪽의 0을 고치거나, $\log(x + \epsilon)$처럼 정의역을 보호하는 처리가 필요하다.
- **anomaly mode는 backward 중 NaN을 처음 반환한 연산(`PowBackward0`)에서 멈춘다.** $\lVert \mathbf{w} \rVert = \sqrt{\sum w_i^2}$를 $\mathbf{w} = 0$에서 미분하면 sqrt의 미분이 무한대가 되고, 그 뒤의 곱에서 NaN이 생긴다. anomaly mode는 그 연산을 만든 forward 코드의 위치도 함께 경고로 출력한다. 모든 연산을 검사하므로 학습이 크게 느려진다. 실패를 재현한 짧은 실행에서만 켠다.

**활성값 통계.** 층마다 활성값의 평균·표준편차, 포화 비율(tanh라면 $\lvert a \rvert > 0.99$, ReLU라면 0인 비율)을 본다. 초기화 스케일이 깊은 층에서 만드는 차이를 tanh 10층으로 본다(초기화의 원리는 17.1장).

```python {exec="browser" session="debug"}
rng = np.random.default_rng(0)
x0 = rng.normal(size=(512, 256))                          # (N, D) 표준화된 입력
for name, std in [("std 1", 1.0), ("std 1/sqrt(D)", 1 / np.sqrt(256)), ("std 0.01", 0.01)]:
    a, rows = x0, []
    for layer in range(1, 11):
        W = rng.normal(0, std, size=(256, 256))           # (D, D)
        a = np.tanh(a @ W)                                # (N, D)
        if layer in (1, 5, 10):
            rows.append(f"층 {layer:2d}: std {a.std():.3g}, 포화 {np.mean(np.abs(a) > 0.99):.0%}")
    print(f"{name:14s} | " + " | ".join(rows))
```

```output
std 1          | 층  1: std 0.975, 포화 87% | 층  5: std 0.974, 포화 87% | 층 10: std 0.974, 포화 86%
std 1/sqrt(D)  | 층  1: std 0.629, 포화 1% | 층  5: std 0.323, 포화 0% | 층 10: std 0.226, 포화 0%
std 0.01       | 층  1: std 0.157, 포화 0% | 층  5: std 0.000104, 포화 0% | 층 10: std 1.1e-08, 포화 0%
```

표준편차 1로 초기화하면 첫 층부터 활성값의 87%가 포화된다. 포화된 tanh의 미분은 거의 0이라 gradient가 흐르지 않는다. 0.01로 초기화하면 층을 지날 때마다 활성값이 줄어 10층에서 $10^{-8}$ 수준이 되고, 깊은 층의 신호와 gradient가 사라진다. $1/\sqrt{D}$로 초기화하면 둘 다 피한다(층을 지나며 조금씩 줄어드는 것은 tanh가 분산을 줄이기 때문이며, 17.1장의 Xavier 초기화가 이를 다룬다). 학습이 처음부터 안 될 때 층별 통계 한 줄이 "모델 구조가 문제"라는 잘못된 결론을 막는다.

## 디버깅 기록

진단은 실험이다. 기록하지 않은 실험은 반복된다. 디버깅을 시작하면 다음 형식으로 한 줄씩 남긴다.

| 시각 | 가설 | 실험(바꾼 것 하나) | 예측 | 결과 | 결론 |
|---|---|---|---|---|---|
| 10:20 | label이 어긋났다 | 선형 baseline 학습 | 정확도가 최다 class 비율 근처 | 0.395 (최다 0.372) | 가설 유지 → 데이터 로딩 코드 확인 |
| 10:35 | `load_data`에서 y를 섞지 않는다 | y도 같은 순서로 섞음 | baseline 정확도 상승 | 0.968 | 원인 확정, 회귀 테스트 추가 |

이 형식에는 지켜야 할 규칙이 두 개 들어 있다.

- **한 번에 하나만 바꾼다.** 학습률, 정규화, 초기화를 한꺼번에 바꿔 학습이 되면, 무엇이 필요했는지 모른다. 다음에 같은 증상이 다른 원인으로 나타나면 그때 바꾼 세 가지를 모두 다시 시도하게 된다. 작은 설정에서 하나씩 시험하는 비용은 크지 않다.
- **예측을 먼저 쓴다.** 결과를 본 뒤에 "그럴 줄 알았다"고 쓰면 모든 결과가 가설을 지지하는 것처럼 보인다. 결과보다 예측을 먼저 적으면, 예측이 틀린 실험이 가장 많은 정보를 준다.

원인을 확정하면 그 버그를 잡는 테스트를 남긴다(6.6장). 실습의 체크리스트가 그 예다. 디버깅 기록은 연구 노트(23.7장)의 일부로 보관하면, 몇 달 뒤 같은 증상을 만났을 때 검색할 수 있다.

::quiz{id="one-change-at-a-time"}

## 실습

일부러 버그 다섯 개를 심은 학습 코드에서 체크리스트로 모든 버그를 찾는다. 코드는 `labs/deep-learning-pytorch/debugging-training/`에 있다.

- `pipeline.py`: (온도 K, 압력 bar, 조성 비율)로 상(phase) 3종을 분류하는 작은 학습 파이프라인. 버그 다섯 개를 켜고 끌 수 있다. 입력과 label의 어긋남, 정규화 누락, softmax 중복, `zero_grad` 누락, 평가 모드 누락이다.
- `checklist.py`: 여섯 항목의 체크리스트. (1) 입력 스케일, (2) 선형 baseline으로 본 label 정합성, (3) 초기 손실 $\approx \ln C$, (4) batch 하나 과적합, (5) 학습률 0으로 같은 batch를 두 번 통과시켜 gradient가 같은지, (6) 같은 모델을 두 번 평가해 같은 값이 나오는지. 끝에는 버그를 하나씩만 켰을 때 어느 검사가 실패하는지 표로 출력한다.

```bash
cd labs/deep-learning-pytorch/debugging-training
uv run --project ../.. --group torch python checklist.py
BUGS=none TABLE=0 uv run --project ../.. --group torch python checklist.py
```

먼저 `pipeline.py`의 버그 분기를 보지 않고 체크리스트의 결과만으로 원인을 추론해 디버깅 기록 형식으로 적은 뒤, 코드에서 확인한다. 집필 시점의 실행 결과(CPU, PyTorch 2.15)는 다음과 같다.

```output
$ python checklist.py            # BUGS=all (기본)
켠 버그: ['double-softmax', 'eval-mode', 'label-misalign', 'no-normalize', 'no-zero-grad']
  실패  1 입력 스케일                     |평균| 최대 330, 표준편차 비 101
  실패  2 label 정합성(선형 baseline)     선형 baseline train 정확도 0.395 (최다 class 비율 0.372)
  통과  3 초기 손실 ≈ ln C               초기 손실 1.236 (ln 3 = 1.099)
  실패  4 batch 하나 과적합               batch 16개를 300 step 학습한 뒤 손실 1.18
  실패  5 gradient 초기화               같은 batch 두 번째 step의 gradient 상대 차이 1
  실패  6 평가의 결정성                    같은 모델을 두 번 평가한 손실 1.2457, 1.2148
  학습 결과: 검증 손실 1.228, 검증 정확도 0.323

버그를 하나씩만 켰을 때 실패하는 검사
  label-misalign  실패한 검사 2, 4      검증 손실 1.122  검증 정확도 0.337
  no-normalize    실패한 검사 1, 3, 4   검증 손실 1.018  검증 정확도 0.347
  double-softmax  실패한 검사 4         검증 손실 0.601  검증 정확도 0.950
  no-zero-grad    실패한 검사 4, 5      검증 손실 2.861  검증 정확도 0.860
  eval-mode       실패한 검사 6         검증 손실 0.142  검증 정확도 0.950
$ BUGS=none TABLE=0 python checklist.py
켠 버그: 없음
  통과  1 입력 스케일                     |평균| 최대 7.15e-09, 표준편차 비 1
  통과  2 label 정합성(선형 baseline)     선형 baseline train 정확도 0.968 (최다 class 비율 0.352)
  통과  3 초기 손실 ≈ ln C               초기 손실 1.212 (ln 3 = 1.099)
  통과  4 batch 하나 과적합               batch 16개를 300 step 학습한 뒤 손실 0.00197
  통과  5 gradient 초기화               같은 batch 두 번째 step의 gradient 상대 차이 0
  통과  6 평가의 결정성                    같은 모델을 두 번 평가한 손실 1.2033, 1.2033
  학습 결과: 검증 손실 0.136, 검증 정확도 0.950
```

결과에서 읽을 것은 세 가지다.

- **모든 버그가 적어도 한 검사에 걸리고, 버그가 없으면 모든 검사를 통과한다.** 이 성질은 `test_checklist.py`가 테스트로 확인한다.
- **검사와 버그는 일대일이 아니다.** 정규화 누락은 세 검사(1, 3, 4)에 걸리고, label 어긋남은 baseline 검사와 함께 batch 과적합 검사에도 걸렸다. 이 파이프라인에는 dropout이 있고 비슷한 입력에 서로 다른 label이 붙어 외우기가 더 어렵기 때문인데, 셀 1에서 보았듯 늘 그렇다고 기대할 수는 없다. 그래서 실패한 검사 하나에서 곧바로 원인을 단정하지 않고, 가설을 세워 하나씩 확인한다.
- **정확도만 보면 놓치는 버그가 있다.** softmax 중복과 평가 모드 누락은 검증 정확도가 정상과 같은 0.950이다. softmax 중복은 검증 손실(0.601, 정상은 0.136)에서, 평가 모드 누락은 평가의 결정성에서 드러난다. 모든 버그를 켠 학습의 정확도 0.323은 우연 수준(1/3)이지만, 그 숫자만으로는 버그가 다섯 개라는 것을 알 수 없다.

이어서 연습문제로 gradient norm 로그의 이상 step 탐지기를 만든다.

::exercise{id="grad-spike-detector"}

## 실패 사례

- **여러 설정을 한꺼번에 바꾸어 원인을 알 수 없게 되는 것.** 학습이 안 될 때 학습률·batch 크기·정규화·초기화를 동시에 바꾸고, 학습이 되면 그대로 진행한다. 실제 원인이 데이터의 결측 처리였다면 그 버그는 남아 있고, 바꾼 설정들은 근거 없는 기본값이 된다.
- **증상을 숨기는 수정.** NaN이 나온 step을 건너뛰거나, gradient clipping 임계값을 아주 작게 해서 발산을 막는다. 원인(정의역 밖의 입력, 잘못된 label)은 남아 학습을 계속 오염시킨다. clipping은 원인을 이해한 뒤의 안전장치로 쓴다.
- **큰 설정에서 디버깅하는 것.** 실패 하나를 확인하는 데 몇 시간이 걸리면 하루에 몇 개의 가설밖에 시험하지 못한다. 실패를 작은 데이터·작은 모델·짧은 학습에서 재현하는 데 먼저 시간을 쓴다.
- **좋은 결과를 디버깅하지 않는 것.** 기대보다 좋은 결과도 같은 체크리스트(특히 데이터 중복과 평가 코드)를 통과시킨다.

## 통과 기준

- 학습 실패 사례 하나(자신의 연구나 실습의 버그)를 원인·근거·해결로 기록한 보고서를 쓴다. 보고서에는 디버깅 기록 표(가설·실험·예측·결과)와, 원인을 잡는 회귀 테스트가 들어간다.
- 실습의 체크리스트로 버그 다섯 개를 모두 찾고, 각 버그가 어느 검사에서 어떤 숫자로 드러났는지 설명한다.
- 연습문제 `grad-spike-detector`의 테스트를 통과한다.
