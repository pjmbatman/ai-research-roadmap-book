---
id: architectures/time-series-deep-learning
title: "시계열 딥러닝"
title_en: "Deep Learning for Time Series"
tier: core
status: draft
prerequisites: [architectures/recurrent-and-ssm, classical-ml/time-series-analysis]
---

# 시계열 딥러닝

기상 관측, 배터리 전압, 지진파, 환자의 생체 신호, 공정 센서처럼 과학 데이터의 상당 부분은 시간 순서가 있는 측정이다. 18.3장의 RNN·TCN·SSM과 18.4장의 Transformer는 모두 시계열에 쓸 수 있고, 이를 바탕으로 한 예측 모델이 많이 발표되었다. 그런데 시계열 예측은 딥러닝이 단순한 방법을 이기지 **못하는** 일이 유난히 자주 보고되는 분야이기도 하다. 한 층짜리 선형 모델이 여러 Transformer 기반 장기 예측 모델보다 나은 성능을 낸다는 보고가 대표적이다(Zeng 외, 2023).

이 장은 그 이유의 상당 부분이 모델이 아니라 **평가**에 있다고 본다. 시계열 데이터는 window를 겹쳐 자르면서 누수가 생기기 쉽고, 나이브 baseline이 의외로 강하며, 여러 시점을 앞서 예측하는 방식에 따라 오차가 다르게 쌓인다. 16.7장이 고전 모델과 rolling-origin 평가를 다뤘다면, 이 장은 딥러닝 모델의 계열과 입력 설계를 정리한 뒤, 같은 평가 원칙을 딥러닝 모델에 그대로 적용하는 법을 계산으로 보인다.

## 학습 목표

- 시계열 딥러닝 모델의 계열과 입력 설계(window·horizon·공변량·patching)를 과제에 맞게 고른다
- 나이브 baseline·시간 순서 분할·다단계 예측 전략을 갖춘 평가 프로토콜로 시계열 특유의 평가 함정을 피한다

## 모델 계열

시계열 예측 모델은 시간 축의 정보를 어떻게 섞는가로 나눌 수 있다. 각 구조의 원리는 18.3장과 18.4장에서 다뤘으므로, 여기서는 시계열 예측에 쓸 때의 성격만 비교한다.

| 계열 | 시간 축을 섞는 방식 | 장점 | 주의할 점 |
|---|---|---|---|
| RNN·LSTM·GRU | 상태를 한 시점씩 갱신 | 길이에 무관한 상태, 온라인 예측 | 순차 계산이라 학습이 느리고, 먼 과거의 gradient가 약해짐 |
| TCN | causal convolution을 dilation으로 쌓음 | 병렬 학습, 수용 영역을 정확히 설계 | 수용 영역 밖의 과거는 보지 못함 |
| Transformer 계열 | attention으로 모든 시점 쌍을 비교 | 먼 시점 사이의 관계를 직접 표현 | 길이의 제곱에 비례하는 비용, 순서 정보는 위치 인코딩에 의존 |
| MLP 기반 | window 전체를 한 벡터로 보고 완전 연결층 | 단순하고 빠름, 강한 baseline | 입력 길이가 고정되고 시점 간 구조를 가정하지 않음 |
| SSM | 선형 상태공간 recurrence를 병렬로 계산 | 긴 열에서 선형 비용, 순차 추론도 가능 | 비교적 새로운 계열이라 과제별 검증이 더 필요 |

**선형 모델이 강한 이유.** 많은 과학 시계열은 추세와 주기, 그리고 짧은 기억을 가진 잡음의 합에 가깝다. 이런 구조는 과거 값의 선형 결합으로 상당 부분 설명된다. 16.7장의 AR 모델이 바로 그것이다. 그래서 lookback window 전체를 받아 horizon 전체를 한 번에 내는 선형층 하나가 강한 baseline이 된다. Zeng 외(2023)의 DLinear는 입력을 추세와 나머지로 나눈 뒤 각각에 선형층 하나를 쓰는 모델로, 그들이 실험한 장기 예측 데이터셋들에서 당시의 Transformer 기반 모델보다 나은 결과를 보고했다. 이 결과는 "Transformer가 시계열에 쓸모없다"가 아니라 "복잡한 모델의 개선을 주장하려면 이런 단순 모델과 같은 프로토콜로 비교해야 한다"로 읽는 것이 맞다.

**patching.** Transformer를 시계열에 쓸 때 시점 하나를 토큰 하나로 쓰면 두 가지 문제가 생긴다. 시점 하나의 값은 의미가 적고(잡음 섞인 숫자 하나), 토큰 수가 lookback 길이와 같아 attention 비용이 길이의 제곱으로 는다. **patching**은 window를 길이 $P$의 조각으로 잘라(간격 $S$로 겹쳐도 된다) 조각 하나를 토큰 하나로 쓴다(Nie 외, 2023). 토큰 수는

$$
n_{\text{patch}} = \Big\lfloor \frac{L - P}{S} \Big\rfloor + 1
$$

이다. $L$은 lookback 길이다. 같은 논문은 여러 변수(채널)를 섞지 않고 채널마다 같은 모델을 따로 적용하는 **channel independence**도 함께 썼다. 아래 코드는 이 두 가지를 shape으로 확인한다.

```python {exec="browser" session="ts"}
import numpy as np

def patchify(x, P, S):                                  # (B, C, L) → (B*C, n_patch, P)
    B, C, L_in = x.shape
    n_patch = (L_in - P) // S + 1
    idx = S * np.arange(n_patch)[:, None] + np.arange(P)[None, :]   # (n_patch, P)
    return x.reshape(B * C, L_in)[:, idx]               # 채널마다 따로: channel independence

x = np.random.default_rng(1).normal(size=(32, 3, 96))  # batch 32, 채널 3, lookback 96
tokens = patchify(x, P=16, S=8)
print("입력", x.shape, "→ 패치 토큰", tokens.shape)
print("첫 두 패치가 원래 시점 0–15, 8–23인가:",
      np.array_equal(tokens[0, 0], x[0, 0, 0:16]), np.array_equal(tokens[0, 1], x[0, 0, 8:24]))
n_tok = tokens.shape[1]
print(f"토큰 수 96 → {n_tok},  attention 행렬 원소 수 비 {96**2 / n_tok**2:.1f}배 감소")
```

```output
입력 (32, 3, 96) → 패치 토큰 (96, 11, 16)
첫 두 패치가 원래 시점 0–15, 8–23인가: True True
토큰 수 96 → 11,  attention 행렬 원소 수 비 76.2배 감소
```

batch 32개 × 채널 3개가 96개의 독립된 단변량 열이 되고, 각 열은 길이 16의 패치 11개로 표현된다. 패치 하나는 16시점의 국소 모양(기울기, 꺾임)을 담으므로 토큰 하나의 의미가 커지고, attention 행렬은 $96^2$에서 $11^2$으로 줄어든다. 비용이 줄어든 만큼 더 긴 lookback을 볼 수 있다. channel independence는 채널 사이의 관계를 모델이 직접 쓰지 못하게 하는 대가로, 채널 수가 많고 데이터가 적을 때 과적합을 줄인다. 채널 사이의 관계가 핵심인 문제(예: 상류–하류 수위)라면 이 가정은 맞지 않는다.

::quiz{id="patch-count"}

## 입력 설계

**window·stride·horizon.** 긴 시계열 하나를 학습 샘플 여러 개로 바꾸려면 잘라야 한다. 시점 $s$에서 시작해 길이 $L$의 **lookback window** $\mathbf{x}_{s:s+L}$를 입력으로, 그 뒤 $H$개 시점 $\mathbf{x}_{s+L:s+L+H}$를 정답으로 삼는다. $H$를 **horizon**이라 한다. 다음 샘플은 **stride**만큼 옮겨 자른다. stride가 1이면 이웃한 두 샘플은 $L + H - 1$개 시점을 공유한다.

```python {exec="browser" session="ts"}
import numpy as np

rng = np.random.default_rng(0)
n_h = 24 * 120                                          # 시간 단위 관측 120일
t_h = np.arange(n_h)
noise = np.zeros(n_h)
for i in range(1, n_h):                                 # AR(1) 잡음: 날씨처럼 며칠 이어지는 편차
    noise[i] = 0.97 * noise[i - 1] + 0.4 * rng.normal()
temp_c = 10 + 0.003 * t_h + 5 * np.sin(2 * np.pi * (t_h - 9) / 24) + noise   # (n_h,)

L, H = 48, 24                                           # lookback 48시간, horizon 24시간
def make_windows(x, L, H, stride=1):
    starts = np.arange(0, len(x) - L - H + 1, stride)
    X = np.stack([x[s:s + L] for s in starts])          # (n_win, L)
    Y = np.stack([x[s + L:s + L + H] for s in starts])  # (n_win, H)
    return X, Y, starts

X, Y, starts = make_windows(temp_c, L, H)
print("X", X.shape, " Y", Y.shape)

idx = rng.permutation(len(X)); n_tr = int(0.8 * len(X))
tr, te = idx[:n_tr], idx[n_tr:]                         # 무작위 분할 (잘못된 예)
covered = np.zeros(n_h, bool)
for s in starts[tr]:
    covered[s:s + L + H] = True                         # train window가 본 시점
frac = np.mean([covered[s + L:s + L + H].all() for s in starts[te]])
print(f"무작위 분할: test 정답 구간을 train window가 이미 본 비율 = {frac:.3f}")
```

```output
X (2809, 48)  Y (2809, 24)
무작위 분할: test 정답 구간을 train window가 이미 본 비율 = 0.998
```

자료는 120일 동안의 시간별 기온을 흉내 낸 합성 시계열이다. 하루 주기(진폭 5 °C), 느린 상승 추세, 며칠씩 이어지는 AR(1) 편차가 더해져 있다. 2,880시간에서 window 2,809개가 나왔다. 이 window들을 무작위로 섞어 80%를 학습에 쓰면, test window 정답 구간의 99.8%가 이미 어떤 train window 안에 들어 있다. 모델은 test의 "미래"를 학습 중에 이미 본 셈이다. 이 누수가 성능 추정을 얼마나 바꾸는지는 평가 절에서 잰다.

**lookback과 horizon의 선택.** $L$은 데이터의 주요 주기와 기억 길이를 덮어야 한다. 하루 주기가 있는 자료에 6시간 lookback을 주면 모델은 지금이 하루 중 어디인지 알 수 없다(평가 절의 다단계 예측 실험). 반대로 $L$을 지나치게 늘리면 샘플 수가 줄고 오래된 정보에 과적합하기 쉽다. $H$는 문제가 정한다. 내일 운전 계획을 세우려면 24시간, 다음 제어 주기에 쓰려면 몇 분이다.

**공변량.** 예측 대상 말고도 입력에 넣을 수 있는 변수를 **공변량**(covariate)이라 한다. 시계열 예측에서는 이 공변량을 **언제 알 수 있는가**로 나눠야 한다.

- **과거에만 관측되는 공변량**: 다른 센서의 측정값, 과거 기상 관측. lookback 구간의 값만 입력에 쓸 수 있다.
- **미래에도 알려진 공변량**(known future input): 달력(시각, 요일, 공휴일), 미리 정해진 운전 계획·설정값, 예측 시점에 이미 발표된 기상 예보. horizon 구간의 값도 입력에 쓸 수 있다.
- **정적 공변량**: 센서 위치, 장비 종류, 환자의 나이처럼 시간에 따라 바뀌지 않는 값.

가장 흔한 누수는 과거에만 관측되는 공변량의 **미래 값**을 입력에 넣는 것이다. 내일의 전력 수요를 예측하면서 내일 실제로 관측된 기온을 넣으면 성능은 좋아 보이지만, 배포 시점에는 그 값이 없다. 배포 시점에 쓸 수 있는 것은 오늘 발표된 내일의 **예보**이고, 학습에도 그 예보(오차 포함)를 써야 같은 조건이 된다.

::quiz{id="known-future-covariates"}

## 불규칙·다변량 시계열

실험실 밖의 시계열은 위 예제처럼 깔끔하지 않다.

**결측과 불규칙 샘플링.** 센서는 고장 나고, 임상 측정은 필요할 때만 하며, 위성 관측은 구름에 가린다. 결측을 앞 값으로 채우거나 보간하면 모델은 "측정되지 않았다"는 정보를 잃는다. 그런데 측정되지 않았다는 사실 자체가 정보인 경우가 많다(상태가 안정적이라 검사를 하지 않았다). 실용적인 방법은 값과 함께 **관측 여부 mask**와 **마지막 관측 이후 경과 시간**을 입력 채널로 넣는 것이다. 관측 시각이 제각각인 자료를 직접 다루는 연속시간 모델은 18.3장과 24.3장에서 다룬다.

**다중 주기.** 전력 수요는 하루·일주일·일 년 주기를 함께 가진다. lookback 하나로 모든 주기를 덮으려면 길이가 너무 길어진다. 시각·요일·연중 날짜를 sin·cos 쌍으로 부호화해 미래에도 알려진 공변량으로 넣으면, 모델이 긴 lookback 없이 주기의 위상을 안다.

**채널 간 관계.** 여러 센서를 함께 예측할 때는 채널 사이 관계를 쓸지 정한다. 위의 channel independence는 관계를 쓰지 않는 쪽의 극단이고, 18.6장의 시공간 그래프 모델은 물리적 연결로 관계를 쓰는 쪽이다. 관계를 쓰는 모델이 실제로 이득을 보는지는 channel independence 모델과의 비교로 확인한다.

:::note[분야 예시]
배터리 셀의 열화 예측에서는 충방전 주기마다 용량을 한 번 측정하지만, 주기 안에서는 전압·전류·온도를 초 단위로 기록한다. 두 시간 척도가 섞인 다변량 시계열이다. 셀마다 수명이 달라 시계열 길이도 제각각이다. 이때 window를 셀 구분 없이 섞어 분할하면, 같은 셀의 앞부분과 뒷부분이 train과 test에 나뉘어 들어가 셀 사이 일반화를 과대평가한다. 시간 순서 분할에 더해 **셀 단위 분할**(15.4장의 group 분할)이 필요하다.
:::

## 시계열 Foundation Model

여러 분야의 대규모 시계열로 사전학습한 뒤 새 시계열에 추가 학습 없이(zero-shot) 또는 약간의 미세조정으로 예측하는 **시계열 foundation model**이 여럿 발표되었다. 구조는 대개 patching한 Transformer이고, 일부는 값을 구간으로 나눠 토큰으로 바꾼 뒤 18.5장의 언어모델처럼 다음 토큰 예측으로 학습한다. 사전학습과 전이의 일반 원리는 19.9장에서 다룬다.

이 모델들을 자기 과제에 쓸지 판단할 때는 다음을 확인한다.

- **평가 데이터가 사전학습 말뭉치에 들어 있었는가.** 공개 벤치마크 시계열이 사전학습 자료에 포함되어 있으면 zero-shot 성능은 기억을 잰 것일 수 있다. 18.5장의 오염 문제와 같다. 발표 문서가 사전학습 자료 목록을 공개하는지 확인하고, 가능하면 공개된 적이 없는 자기 데이터로 평가한다.
- **같은 프로토콜로 나이브 baseline과 비교했는가.** 사전학습 모델이 seasonal naive나 선형 모델보다 나은지, 같은 분할·같은 horizon·같은 metric으로 확인한다.
- **자기 데이터의 성격이 사전학습 자료와 비슷한가.** 사전학습 자료가 주로 경제·교통·에너지 시계열이라면, 물리 실험의 고주파 신호나 불규칙 샘플링 자료에서는 장점이 줄어들 수 있다.

모델 이름과 성능 순위는 빠르게 바뀐다. 위의 세 질문은 바뀌지 않는다.

## 평가의 함정

**나이브 baseline.** 16.1장의 원칙대로 복잡한 모델의 성능은 단순한 기준과 비교해야 의미가 있다. 시계열의 나이브 baseline은 계산이 필요 없는 예측 규칙이다.

- **persistence**: 마지막 관측값을 horizon 전체에 그대로 쓴다. $\hat{y}_{t+h} = y_t$.
- **seasonal naive**: 한 주기 전의 값을 쓴다. 주기가 $m$이면 $\hat{y}_{t+h} = y_{t+h-m}$ ($h \le m$).
- **평균**: 학습 구간의 평균.

여러 시계열이나 단위가 다른 변수를 함께 보고할 때는 **MASE**(mean absolute scaled error)를 쓴다(Hyndman, Koehler, 2006). 예측 오차의 MAE를, **학습 구간**에서 seasonal naive를 한 단계 앞 예측에 썼을 때의 MAE로 나눈다.

$$
\mathrm{MASE} = \frac{\frac{1}{n_{\text{test}}}\sum_{t \in \text{test}} \lvert y_t - \hat{y}_t \rvert}{\frac{1}{n_{\text{train}} - m}\sum_{t=m+1}^{n_{\text{train}}} \lvert y_t - y_{t-m} \rvert}.
$$

분모는 학습 자료만으로 정해지는 척도이므로 단위가 사라지고, 0에 가까운 값이 있어도 MAPE처럼 발산하지 않는다. MASE가 1보다 작으면 학습 구간의 seasonal naive 한 단계 예측보다 평균 오차가 작다는 뜻이다.

```python {exec="browser" session="ts"}
cut = 24 * 96                                           # 앞 96일 학습, 뒤 24일 평가
is_tr = starts + L + H <= cut                           # 정답까지 cut 이전에 끝나는 window
is_te = starts + L >= cut                               # 정답이 모두 cut 이후인 window
Xtr, Ytr, Xte, Yte = X[is_tr], Y[is_tr], X[is_te], Y[is_te]

baselines = {
    "train 평균     ": np.full_like(Yte, temp_c[:cut].mean()),
    "persistence    ": np.repeat(Xte[:, -1:], H, axis=1),   # 마지막 관측값 유지
    "seasonal naive ": Xte[:, -24:],                         # 24시간 전 값 (H = 24)
}
scale = np.abs(temp_c[24:cut] - temp_c[:cut - 24]).mean()  # 학습 구간 seasonal naive 1-step MAE
print("모델              h=1    h=6   h=24   평균 MAE   MASE")
for name, pred in baselines.items():
    mae_h = np.abs(pred - Yte).mean(0)                   # horizon별 MAE (H,)
    print(f"{name}  {mae_h[0]:5.2f}  {mae_h[5]:5.2f}  {mae_h[23]:5.2f}   {mae_h.mean():6.2f}   {mae_h.mean() / scale:5.2f}")
```

```output
모델              h=1    h=6   h=24   평균 MAE   MASE
train 평균        4.44   4.44   4.45     4.45    3.37
persistence       0.88   4.58   1.12     4.21    3.19
seasonal naive    1.13   1.14   1.12     1.13    0.86
```

분할은 시간 순서를 따른다. 정답까지 96일째 이전에 끝나는 window만 학습에, 정답이 모두 그 이후인 window만 평가에 쓴다(평가 window의 입력은 학습 구간의 끝을 볼 수 있다. 예측 시점에는 그 과거가 실제로 주어지기 때문이다). **horizon별로** 보고하는 것이 중요하다. persistence는 1시간 뒤에는 가장 좋지만(0.88 °C) 6시간 뒤에는 하루 주기의 반대편이라 평균보다도 나쁘고(4.58 °C), 24시간 뒤에는 다시 같은 시각이라 좋아진다. 평균 MAE 하나만 보면 이 구조가 보이지 않는다. 이 자료에서 이겨야 할 기준은 MASE 0.86, 평균 MAE 1.13 °C의 seasonal naive다.

**누수: 무작위 분할.** 앞 절에서 무작위 분할은 test 정답 구간의 99.8%를 학습 중에 노출했다. 같은 모델을 두 분할로 평가한다. 선형 모델은 lookback 48시간에서 horizon 24시간을 한 번에 내는 다중 출력 ridge 회귀이고, 1-최근접 이웃은 학습 window 중 입력이 가장 비슷한 것을 찾아 그 window의 미래를 그대로 복사한다.

```python {exec="browser" session="ts"}
def fit_linear(X, Y, lam=1.0):                          # 다중 출력 ridge (direct 전략)
    Xb = np.hstack([X, np.ones((len(X), 1))])           # (n, L+1)
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)   # (L+1, H)
    return lambda Xn: np.hstack([Xn, np.ones((len(Xn), 1))]) @ W

def fit_1nn(X, Y):                                      # 가장 비슷한 과거 window의 미래를 복사
    def predict(Xn):
        d2 = (Xn**2).sum(1)[:, None] - 2 * Xn @ X.T + (X**2).sum(1)[None]
        return Y[np.argmin(d2, axis=1)]
    return predict

splits = {"무작위 분할  ": (X[tr], Y[tr], X[te], Y[te]), "시간 순서 분할": (Xtr, Ytr, Xte, Yte)}
for model_name, fit in [("선형(ridge)   ", fit_linear), ("1-최근접 이웃 ", fit_1nn)]:
    for split_name, (a, b, c, d) in splits.items():
        print(f"{model_name} {split_name}  test MAE {np.abs(fit(a, b)(c) - d).mean():.2f}")
print(f"seasonal naive 시간 순서 분할  test MAE {np.abs(Xte[:, -24:] - Yte).mean():.2f}")
```

```output
선형(ridge)    무작위 분할    test MAE 0.95
선형(ridge)    시간 순서 분할  test MAE 0.89
1-최근접 이웃  무작위 분할    test MAE 1.38
1-최근접 이웃  시간 순서 분할  test MAE 1.79
seasonal naive 시간 순서 분할  test MAE 1.13
```

두 분할의 test 구간이 다르므로 숫자를 한 줄씩 정밀하게 비교할 수는 없다. 그래도 결론은 분명하다.

- **유연한 모델일수록 무작위 분할에서 이득을 본다.** 1-최근접 이웃은 무작위 분할에서 1.38 °C로 보이지만, 실제 미래(시간 순서 분할)에서는 1.79 °C로 seasonal naive(1.13 °C)보다 나쁘다. 무작위 분할에서는 test window와 한 시간만 어긋난 train window가 거의 항상 있어서, 그 미래를 복사하면 정답과 많이 겹친다. 추세가 있는 자료에서 미래의 기온은 학습 구간보다 높은데, 과거를 복사하는 모델은 그 범위를 벗어나지 못한다는 약점도 무작위 분할에서는 가려진다.
- **선형 모델은 누수의 덕을 거의 보지 못한다.** 파라미터가 49 × 24개뿐인 선형 모델은 개별 window를 외우지 못한다. 그래서 "무작위 분할로도 결과가 비슷했다"는 관찰을 다른 모델에 일반화하면 안 된다. 딥러닝 모델은 1-최근접 이웃 쪽에 가깝게 외울 수 있다.

::quiz{id="random-split-windows"}

**다단계 예측 전략.** horizon이 여러 시점이면 두 가지 전략이 있다.

- **recursive**(iterated): 한 시점 앞을 예측하는 모델 하나를 학습하고, 예측값을 입력 끝에 붙여 다시 예측하기를 $H$번 반복한다. 18.3장의 autoregressive rollout과 같다.
- **direct**(multi-output): horizon의 각 시점을 직접 예측한다. 위의 다중 출력 ridge처럼 $H$개 출력을 한 번에 내는 모델이 그렇다.

recursive는 모델 하나로 어떤 horizon이든 예측할 수 있지만, 앞 단계의 오차가 다음 단계의 입력이 되어 쌓인다. 모델이 자료를 잘 설명할 때는 두 전략이 비슷하고, 모델이 틀렸을 때 차이가 커진다.

```python {exec="browser" session="ts"}
def recursive(Xtr, Ytr, Xte):
    one_step = fit_linear(Xtr, Ytr[:, :1])              # 1시간 뒤만 예측하는 모델
    window, out = Xte.copy(), []                        # (n_te, L')
    for h in range(H):                                  # 예측값을 입력 끝에 붙여 24번 반복
        y_next = one_step(window)                       # (n_te, 1)
        out.append(y_next[:, 0])
        window = np.hstack([window[:, 1:], y_next])
    return np.stack(out, axis=1)                        # (n_te, H)

print("lookback  전략         h=1    h=6   h=12   h=24")
for Ls in [48, 6]:                                      # 하루 주기를 보는 입력 vs 못 보는 입력
    a, c = Xtr[:, -Ls:], Xte[:, -Ls:]
    for name, pred in [("recursive", recursive(a, Ytr, c)), ("direct   ", fit_linear(a, Ytr)(c))]:
        m = np.abs(pred - Yte).mean(0)
        print(f"{Ls:4d}시간  {name}   {m[0]:5.2f}  {m[5]:5.2f}  {m[11]:5.2f}  {m[23]:5.2f}")
```

```output
lookback  전략         h=1    h=6   h=12   h=24
  48시간  recursive    0.32   0.76   1.00   1.10
  48시간  direct       0.32   0.75   1.00   1.07
   6시간  recursive    0.42   2.98   5.77   3.72
   6시간  direct       0.42   2.21   2.55   1.10
```

lookback 48시간이면 하루 주기 전체가 입력에 있어 1-step 선형 모델이 자료를 잘 설명하고, 두 전략이 거의 같다. lookback 6시간이면 1-step 모델은 하루 주기를 알 수 없어 짧은 구간의 추세를 이어 그리는 모델이 되고, recursive로 반복하면 그 직선이 계속 뻗어 12시간 뒤 오차가 5.77 °C까지 커진다. direct 모델은 horizon마다 "6시간 입력에서 $h$시간 뒤로 가는 가장 좋은 선형 사상"을 따로 배우므로 오차가 쌓이지 않는다. 24시간 뒤의 값은 지금 값과 같은 시각이라 6시간 입력으로도 잘 맞힌다(1.10 °C). 1-step 오차(h=1)만 보고하면 두 모델의 차이가 전혀 보이지 않는다는 점에 주의한다.

::quiz{id="recursive-vs-direct"}

**데이터셋 선택 편향.** 같은 모델도 데이터셋마다 순위가 바뀐다. 논문이 자기 모델에 유리한 데이터셋만 골라 보고하면, 그 표의 평균 순위는 일반적인 우수성을 말하지 않는다. 자기 연구에서는 실제로 쓸 자료(또는 그와 성격이 같은 자료)로 평가하고, 공개 벤치마크의 결과는 참고로만 쓴다. 정규화 통계(평균·표준편차)를 전체 시계열로 계산하는 것도 미래 정보를 쓰는 누수다. 학습 구간의 통계로만 정규화한다.

## 실습

시계열 예측에서 딥러닝 모델과 단순 선형 baseline을 **같은 프로토콜**로 비교한다. 같은 자료, 같은 시간 순서 분할, 같은 horizon, 같은 metric이다. 정규화 통계는 학습 구간에서만 구한다. 아래는 PyTorch로 학습한 선형층 하나(Zeng 외의 선형 baseline과 같은 형태)와 은닉 256의 2층 MLP를 비교한 것이다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
n_h = 24 * 120; t_h = np.arange(n_h); noise = np.zeros(n_h)
for i in range(1, n_h):                                 # 본문 앞 셀과 같은 자료
    noise[i] = 0.97 * noise[i - 1] + 0.4 * rng.normal()
temp_c = 10 + 0.003 * t_h + 5 * np.sin(2 * np.pi * (t_h - 9) / 24) + noise
L, H, cut = 48, 24, 24 * 96
s = np.arange(n_h - L - H + 1)
X = np.stack([temp_c[i:i + L] for i in s]); Y = np.stack([temp_c[i + L:i + L + H] for i in s])
tr, te = s + L + H <= cut, s + L >= cut                 # 앞과 같은 시간 순서 분할
mu, sd = temp_c[:cut].mean(), temp_c[:cut].std()        # 정규화 통계는 학습 구간에서만
f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
Xtr, Ytr, Xte = f(X[tr]), f(Y[tr]), f(X[te])            # (n_tr, L), (n_tr, H), (n_te, L)

def train(model, epochs=60):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xtr)).split(64):
            loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model(Xte).numpy() * sd + mu             # 원래 단위(°C)로 되돌린다 (n_te, H)

models = {"선형 1층          ": nn.Linear(L, H),
          "MLP 2층 (은닉 256)": nn.Sequential(nn.Linear(L, 256), nn.ReLU(), nn.Linear(256, H))}
for name, m in models.items():
    print(f"{name}  test MAE {np.abs(train(m) - Y[te]).mean():.3f} °C")
print(f"seasonal naive      test MAE {np.abs(X[te][:, -24:] - Y[te]).mean():.3f} °C")
```

```output
선형 1층            test MAE 0.924 °C
MLP 2층 (은닉 256)  test MAE 1.124 °C
seasonal naive      test MAE 1.133 °C
```

이 자료에서는 MLP가 선형층보다 나쁘고 seasonal naive와 비슷하다. 파라미터가 약 1만 9천 개인 MLP가 학습 window 2천여 개의 잡음에 맞춰졌기 때문이다. 이 결과 하나로 "MLP는 시계열에 나쁘다"고 결론 내려서도 안 된다. 같은 프로토콜로 비교하면 어떤 모델이 이 자료에서 더 나은지가 드러난다는 것이 요점이다. 자기 데이터로 다음을 수행한다.

1. 시간 순서 분할(가능하면 16.7장의 rolling origin으로 여러 번)을 정하고, 학습 구간 안에서 다시 검증 구간을 시간 순서로 떼어 하이퍼파라미터를 고른다.
2. persistence·seasonal naive·선형 모델을 먼저 평가해 horizon별 MAE와 MASE 표를 만든다.
3. 딥러닝 모델(MLP, TCN, patching Transformer 중 하나 이상)을 같은 분할·같은 horizon·같은 metric으로 평가한다. 튜닝 예산(시도한 설정 수)도 함께 기록한다.
4. 무작위 분할로도 평가해, 그 추정이 시간 순서 분할보다 얼마나 낙관적인지 보고한다.

이어서 연습문제로 MASE를 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="mase-score"}

## 실패 사례

- **단순 baseline 없이 복잡한 모델의 성능을 주장하는 것.** "MAE 1.1 °C"는 seasonal naive가 1.13 °C인 자료에서는 개선이 아니다. 나이브 baseline과 선형 모델을 같은 표에 넣고, 개선의 크기를 MASE나 상대 오차로 보고한다.
- **겹치는 window를 무작위로 섞어 분할하는 것.** test 정답이 학습 중에 노출되어, 외우는 모델일수록 성능이 부풀려진다. 시간 순서로 분할하고, 필요하면 train과 test 사이에 $L + H$ 이상의 간격을 둔다.
- **one-step 오차만 보고하는 것.** 다단계 예측을 recursive로 할 때 오차는 horizon을 따라 쌓인다. horizon별 오차를 보고한다.
- **배포 시점에 알 수 없는 공변량을 입력에 넣는 것.** 실제 관측된 미래 기상, 전체 시계열로 계산한 정규화 통계, 사후에 보정된 값은 모두 미래 정보다.

## 통과 기준

- 자기 시계열 과제의 lookback·horizon·stride와 공변량(과거 관측·미래에 알려진·정적)을 정하고, 각각을 배포 시점에 알 수 있는지로 정당화한다.
- 시간 순서 분할에서 persistence·seasonal naive·선형 모델·딥러닝 모델을 horizon별 MAE와 MASE로 비교한 표를 만든다.
- 다단계 예측을 recursive와 direct 중 무엇으로 했는지 밝히고, 무작위 분할을 썼다면 얼마나 낙관적인지 수치로 보인다.
- 연습문제 `mase-score`의 테스트를 통과한다.

## 참고 문헌

- Hyndman, Koehler (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. — MASE의 제안
- Zeng, Chen, Zhang, Xu (2023). Are Transformers Effective for Time Series Forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121–11128. — 한 층 선형 모델(DLinear)과 Transformer 기반 장기 예측 모델의 비교
- Nie, Nguyen, Sinthong, Kalagnanam (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. — patching과 channel independence
