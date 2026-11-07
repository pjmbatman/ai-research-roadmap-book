---
id: research-landscape/first-pipeline
title: "첫 번째 연구 파이프라인 완주하기"
title_en: "Your First End-to-End Research Pipeline"
tier: core
status: draft
prerequisites: [research-landscape/problem-formulation]
---

# 첫 번째 연구 파이프라인 완주하기

연구를 처음 시작하는 사람이 가장 자주 멈추는 곳은 어려운 수식이 아니라 **처음부터 끝까지 한 번 가 보지 않은 상태**다. 데이터는 불러왔는데 어떻게 나눌지 모르고, 모델은 학습했는데 무엇과 비교할지 모르고, 숫자는 나왔는데 그 숫자를 믿어도 되는지 모른다. 각 단계를 따로 공부하면 단계 사이의 연결, 즉 앞 단계의 결정이 뒤 단계의 숫자를 어떻게 바꾸는지가 보이지 않는다.

이 장은 작은 과학 데이터 하나로 1.1장의 생애주기를 **한 번에 완주**한다. 진자의 주기를 줄 길이와 진폭으로 예측하는 회귀 문제다. 문제 정의, 데이터, 분할, baseline, 작은 신경망, 평가, 보고까지 NumPy와 scikit-learn으로 직접 실행한다. 신경망의 학습 루프(forward → loss → backward → update)를 처음 만나는 곳도 여기다. 완벽하게 하는 것이 아니라 끝까지 가는 것이 목표다. 각 단계를 깊게 파는 일은 뒤의 부들이 맡고, 이 장은 그 부들이 어디에 붙는지 보여 주는 뼈대다.

## 학습 목표

- 데이터부터 결과 보고까지 최소 파이프라인을 한 번 완주한다
- 이후 각 부가 이 파이프라인의 어느 부분을 깊게 다루는지 안다

## 최소 파이프라인의 구성

**일곱 단계.** 최소 파이프라인은 다음 일곱 단계다. 1.1장의 아홉 단계 중 연구 질문 이후의 단계를 코드로 옮긴 것이다.

| 단계 | 이 장에서 하는 일 | 산출물 | 깊게 다루는 곳 |
|---|---|---|---|
| 문제 정의 | 입력·출력·정답·일반화 대상·지표·성공 기준을 정한다 | 한 장짜리 정의 | 1.3장, 12.3장 |
| 데이터 로드 | 측정 데이터를 표로 읽는다 | `DataFrame` | 15.1–15.3장 |
| 분할 | 학습 60% · 검증 20% · 테스트 20% | 인덱스 배열 세 개 | 15.4장 |
| baseline | 평균 예측, 선형 회귀, 물리 공식, 물리 특징 회귀 | 검증 RMSE | 16.1–16.2장 |
| 신경망 | 은닉층 하나의 MLP를 경사하강으로 학습 | 학습된 파라미터 | 8.4장, 10.3장, 17.5장 |
| 평가 | 같은 분할·같은 지표로 모든 모델을 한 번에 테스트 | 결과 표 | 16.6장, 21.2장 |
| 보고 | 결과 표와 해석, 재실행 명령 | 한 쪽짜리 보고서 | 22.2장, 23.6장 |

**문제 정의.** 1.3장의 양식을 짧게 채운다.

| 항목 | 내용 |
|---|---|
| 연구 질문 | 줄 길이와 처음 진폭으로 진자의 주기를 측정 잡음 수준으로 예측할 수 있는가? |
| 문제 유형 | 정적 예측(회귀) |
| 변수 | 입력: 줄 길이 $L$ [m], 처음 진폭 $\theta_0$ [°] / 출력: 주기 $T$ [s] |
| 정답 정의 | 스톱워치로 잰 주기. 측정 잡음의 표준편차는 0.01 s |
| 정보의 시점 | 두 입력 모두 진자를 놓기 전에 정해진다 |
| 일반화 대상 | 같은 범위($L$ 0.2–2.0 m, $\theta_0$ 5–80°)의 **새 진자**. 행마다 다른 진자이므로 행 단위 무작위 분할이 일반화 대상과 맞는다 |
| 지표 | 주기의 RMSE [s] |
| baseline | 평균 예측, 선형 회귀, 작은 진폭 공식, 물리 특징 회귀 |
| 성공 기준 | 테스트 RMSE가 측정 잡음(0.01 s)의 두 배 이내 |

**데이터.** 이 장의 공통 데이터셋은 진자 300개의 주기 측정을 흉내 낸 시뮬레이션 데이터다. 실제 측정 대신 시뮬레이션을 쓰는 이유는 **참값을 알기 때문이다.** 참값을 알면 모델의 오차가 측정 잡음 때문인지 모델 때문인지 가를 수 있다. 진폭이 큰 진자의 정확한 주기는 제1종 완전 타원적분 $K$로 쓸 수 있다.

$$
T = 4\sqrt{\frac{L}{g}}\; K\!\left(m\right), \qquad m = \sin^2\!\frac{\theta_0}{2}.
$$

여기서 $g$는 중력가속도, $K(m) = \int_0^{\pi/2} \big(1 - m \sin^2\phi\big)^{-1/2} d\phi$다. $\theta_0 \to 0$이면 $K \to \pi/2$이므로 익숙한 $T = 2\pi\sqrt{L/g}$로 돌아간다. 이 참 주기에 표준편차 0.01 s의 측정 잡음을 더한다.

```python {exec="browser" session="pipeline"}
import numpy as np
import pandas as pd
from scipy.special import ellipk

G_MS2 = 9.81                                               # 중력가속도 [m/s^2]

def make_pendulum_data(n, seed):
    """진자 n개의 주기 측정을 흉내 낸다. 참 주기는 타원적분 공식, 측정 잡음은 0.01 s."""
    r = np.random.default_rng(seed)
    length_m = r.uniform(0.2, 2.0, size=n)                 # (n,) 줄 길이
    amp_deg = r.uniform(5.0, 80.0, size=n)                 # (n,) 처음 진폭
    m = np.sin(np.deg2rad(amp_deg) / 2) ** 2
    period_s = 4 * np.sqrt(length_m / G_MS2) * ellipk(m)   # (n,) 참 주기
    period_s = period_s + r.normal(0, 0.01, size=n)        # 스톱워치 측정 잡음
    return pd.DataFrame({"length_m": length_m, "amp_deg": amp_deg, "period_s": period_s})

df = make_pendulum_data(n=300, seed=0)
print(df.shape)
print(df.head(3).round(3))
```

```output
(300, 3)
   length_m  amp_deg  period_s
0     1.347   72.260     2.582
1     0.686   48.749     1.731
2     0.274    8.016     1.042
```

:::note[분야 예시]
같은 일곱 단계가 분야마다 다른 데이터로 반복된다. **재료**: 조성과 공정 조건으로 경도를 예측하고, 분할 단위는 조성 계열이 된다(28.5장). **기후**: 지점의 기상 변수로 다음 날 기온을 예측하고, 분할은 기간 단위가 된다(27.4장). **배터리**: 초기 사이클로 수명을 예측하고, 분할 단위는 셀이 된다(30.3장). 바뀌는 것은 데이터와 분할 단위이고, 파이프라인의 뼈대는 같다.
:::

## 한 번에 완주하기

**하나의 스크립트, 하나의 프로세스.** 처음 완주할 때는 모든 단계를 파일 하나에 위에서 아래로 쓴다. 설정 파일도, 클래스 계층도, 여러 프로세스도 쓰지 않는다. 구조를 나누는 것은 무엇을 나눠야 하는지 알게 된 다음의 일이다(6부). 이 장의 코드 셀을 순서대로 실행하면 완주가 되고, 같은 내용을 스크립트 하나로 묶은 것이 `labs/research-landscape/first-pipeline/pipeline.py`다.

**분할과 baseline.** 300개 행을 한 번 섞어 60:20:20으로 나눈다. 섞는 데 쓰는 난수 생성기에 seed를 고정해, 다시 실행해도 같은 분할이 나오게 한다. 그다음 신경망보다 먼저 baseline 네 개의 검증 RMSE를 잰다.

- **평균 예측**: 학습 데이터의 평균 주기를 모든 진자에 답한다. 어떤 모델도 이보다는 나아야 한다.
- **선형 회귀**: $T \approx w_1 L + w_2 \theta_0 + b$. 입력을 그대로 쓴 가장 단순한 학습 모델이다.
- **작은 진폭 공식**: $2\pi\sqrt{L/g}$. 데이터를 전혀 쓰지 않는 물리 공식이다.
- **물리 특징 + 선형 회귀**: 작은 진폭 전개 $T \approx 2\pi\sqrt{L/g}\,\big(1 + \theta_0^2/16 + 11\theta_0^4/3072\big)$에서 특징 $\sqrt{L}$, $\sqrt{L}\,\theta_0^2$, $\sqrt{L}\,\theta_0^4$를 만들고($\theta_0$는 라디안), 그 계수만 선형 회귀로 학습한다. 분야의 지식으로 만든 baseline이다.

```python {exec="browser" session="pipeline"}
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(1)
idx = rng.permutation(len(df))                             # 한 번 섞고, 이후 고정
tr, va, te = idx[:180], idx[180:240], idx[240:]            # 60% / 20% / 20%
X = df[["length_m", "amp_deg"]].to_numpy()                 # (300, 2)
y = df["period_s"].to_numpy()                              # (300,)

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def physics_features(X_):                                  # √L, √L·θ², √L·θ⁴ (θ: 라디안)
    sq, th = np.sqrt(X_[:, 0]), np.deg2rad(X_[:, 1])
    return np.column_stack([sq, sq * th**2, sq * th**4])

lin = LinearRegression().fit(X[tr], y[tr])
phys = LinearRegression().fit(physics_features(X[tr]), y[tr])
baselines = {
    "평균 예측": lambda X_: np.full(len(X_), y[tr].mean()),
    "선형 회귀 (L, 진폭)": lin.predict,
    "작은 진폭 공식 2π√(L/g)": lambda X_: 2 * np.pi * np.sqrt(X_[:, 0] / G_MS2),
    "물리 특징 + 선형 회귀": lambda X_: phys.predict(physics_features(X_)),
}
for name, predict in baselines.items():
    print(f"검증 RMSE {rmse(predict(X[va]), y[va]):.4f} s   {name}")
```

```output
검증 RMSE 0.5639 s   평균 예측
검증 RMSE 0.0742 s   선형 회귀 (L, 진폭)
검증 RMSE 0.1447 s   작은 진폭 공식 2π√(L/g)
검증 RMSE 0.0091 s   물리 특징 + 선형 회귀
```

이미 여러 가지를 알 수 있다. 평균 예측의 오차(0.56 s)는 이 문제의 "아무것도 모를 때"의 크기다. 작은 진폭 공식은 진폭 효과를 무시해 큰 진폭에서 주기를 체계적으로 짧게 예측하고, 그래서 입력을 그대로 쓴 선형 회귀보다도 나쁘다. 진폭 효과를 넣은 물리 특징 회귀는 측정 잡음(0.01 s) 수준에 이미 도달했다. 신경망을 학습하기 전에 "넘어야 할 선"이 정해진 것이다.

**forward · loss · backward · update의 첫 만남.** 이제 은닉층 하나의 MLP를 학습한다. 표준화한 입력 $\mathbf{X}$(N × 2)에 대해 모델과 손실은 다음과 같다.

$$
\mathbf{H} = \tanh(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1), \qquad
\hat{\mathbf{y}} = \mathbf{H}\mathbf{W}_2 + \mathbf{b}_2, \qquad
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2.
$$

$\mathbf{W}_1$(2 × 32), $\mathbf{b}_1$, $\mathbf{W}_2$(32 × 1), $\mathbf{b}_2$가 학습할 파라미터다. 학습은 네 동작을 반복한다.

1. **forward**: 현재 파라미터로 $\hat{\mathbf{y}}$를 계산한다.
2. **loss**: 예측과 정답의 차이를 숫자 하나 $L$로 줄인다.
3. **backward**: 연쇄법칙으로 각 파라미터에 대한 $L$의 gradient를 계산한다. 출력에서 입력 쪽으로 거꾸로 가며 곱한다.
4. **update**: 파라미터를 gradient의 반대 방향으로 조금 옮긴다. $\theta \leftarrow \theta - \eta\, \nabla_\theta L$, $\eta$는 학습률.

PyTorch 같은 프레임워크는 3번을 자동으로 해 준다(17.2장). 여기서는 그 계산이 무엇인지 보기 위해 손으로 쓴다. 입력과 출력의 표준화 통계는 **학습 데이터에서만** 계산한다.

```python {exec="browser" session="pipeline"}
mu_x, sd_x = X[tr].mean(0), X[tr].std(0)                   # 표준화 통계는 학습 데이터에서만
mu_y, sd_y = y[tr].mean(), y[tr].std()

def train_mlp(seed, hidden=32, steps=3000, lr=0.05, verbose=False):
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, 1 / np.sqrt(2), (2, hidden)), np.zeros(hidden)          # (2, H), (H,)
    W2, b2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), np.zeros(1)          # (H, 1), (1,)
    Xs, ys = (X[tr] - mu_x) / sd_x, ((y[tr] - mu_y) / sd_y)[:, None]             # (N, 2), (N, 1)
    for step in range(steps):
        h = np.tanh(Xs @ W1 + b1)                          # forward  (N, H)
        out = h @ W2 + b2                                  #          (N, 1)
        loss = np.mean((out - ys) ** 2)                    # loss     MSE
        d_out = 2 * (out - ys) / len(ys)                   # backward (N, 1)
        dW2, db2 = h.T @ d_out, d_out.sum(0)
        d_h = (d_out @ W2.T) * (1 - h ** 2)                # (N, H)
        dW1, db1 = Xs.T @ d_h, d_h.sum(0)
        W1, b1, W2, b2 = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2   # update
        if verbose and step in (0, 100, 1000, steps - 1):
            print(f"step {step:4d}  학습 loss {loss:.4f}")
    return lambda X_: (np.tanh((X_ - mu_x) / sd_x @ W1 + b1) @ W2 + b2)[:, 0] * sd_y + mu_y

mlp = train_mlp(seed=0, verbose=True)
print(f"검증 RMSE {rmse(mlp(X[va]), y[va]):.4f} s   MLP")
```

```output
step    0  학습 loss 1.6511
step  100  학습 loss 0.0375
step 1000  학습 loss 0.0039
step 2999  학습 loss 0.0010
검증 RMSE 0.0232 s   MLP
```

학습 loss는 표준화한 단위의 MSE라서 숫자 자체보다 줄어드는 모양을 본다. 처음 100 step에 크게 줄고, 이후로는 천천히 준다. 검증 RMSE 0.0232 s는 입력을 그대로 쓴 선형 회귀(0.0742 s)보다 세 배 좋지만, 물리 특징 회귀(0.0091 s)에는 못 미친다.

::quiz{id="tanh-derivative-factor"}

::quiz{id="mlp-param-count"}

::quiz{id="standardize-train-only"}

**평가와 보고.** 모델 선택이 끝난 뒤, 모든 모델을 같은 분할·같은 지표로 **한 번에** 테스트한다. 학습·검증·테스트 오차를 나란히 놓으면 과적합 여부도 함께 보인다.

```python {exec="browser" session="pipeline"}
models = {**baselines, "MLP (은닉 32)": mlp}
print("RMSE [s]   학습    검증    테스트   모델")
for name, predict in models.items():
    scores = [rmse(predict(X[s]), y[s]) for s in (tr, va, te)]
    print("         " + "  ".join(f"{v:.4f}" for v in scores) + f"   {name}")
print("측정 잡음의 표준편차 0.0100")
```

```output
RMSE [s]   학습    검증    테스트   모델
         0.5810  0.5639  0.5941   평균 예측
         0.0839  0.0742  0.0760   선형 회귀 (L, 진폭)
         0.1300  0.1447  0.1305   작은 진폭 공식 2π√(L/g)
         0.0094  0.0091  0.0089   물리 특징 + 선형 회귀
         0.0180  0.0232  0.0190   MLP (은닉 32)
측정 잡음의 표준편차 0.0100
```

::quiz{id="read-the-table"}

보고서에 쓸 수 있는 결론은 이 표가 지지하는 범위까지다. "이 데이터·이 분할에서 MLP는 입력을 그대로 쓴 선형 회귀보다 테스트 RMSE가 4배 작지만, 진자 주기의 전개식에서 온 특징을 쓴 4-파라미터 선형 모델보다 2배 크다. 후자는 측정 잡음 수준에 도달해 성공 기준을 만족한다." 이 문장은 과장 없이 숫자를 옮기고, 비교 조건을 밝힌다.

:::tip[연구 팁]
이 결과는 과학 문제에서 자주 반복되는 교훈을 작게 보여 준다. 분야의 지식으로 좋은 특징이나 물리 모델을 만들 수 있으면, 그것이 신경망보다 적은 데이터로 더 좋은 성능을 낸다. 신경망의 자리는 그런 지식이 없거나 불완전한 부분이다. 물리 모델과 학습 모델을 결합하는 방법은 24.5장에서 다룬다.
:::

## 의도적으로 생략한 것들

이 파이프라인은 일부러 많은 것을 빼고 만들었다. 빠진 것을 아는 것도 완주의 일부다. 뺀 것마다 그것이 필요해지는 순간과, 그때 볼 곳을 적는다.

| 생략한 것 | 필요해지는 순간 | 다루는 곳 |
|---|---|---|
| 설정 관리 | 학습률·은닉 크기·seed를 바꿔 가며 여러 번 실행할 때 | 6.3장, 6.4장 |
| 재현성 | 다른 컴퓨터·다른 사람이 같은 숫자를 얻어야 할 때 | 3.4장, 22.2장 |
| 실험 추적 | 실행이 수십 번을 넘어 어떤 설정이 어떤 숫자를 냈는지 헷갈릴 때 | 22.1장 |
| 하이퍼파라미터 탐색 | 은닉 크기·학습률을 체계적으로 고를 때 | 16.6장, 22.3장 |
| 불확실성 | 예측 구간이나 "얼마나 확실한가"를 보고해야 할 때 | 12.3장, 21.1–21.2장 |
| 모델 비교의 통계 | 두 모델의 차이가 seed·분할의 흔들림보다 큰지 판단할 때 | 12.4–12.5장 |
| PyTorch와 자동미분 | 모델이 커져 gradient를 손으로 쓸 수 없을 때 | 8.4장, 17.2–17.5장 |
| 분산 학습·GPU | 데이터와 모델이 한 장치의 메모리와 시간을 넘을 때 | 22.4–22.6장 |
| 데이터 버전·문서화 | 데이터가 바뀌거나 남에게 공개할 때 | 15.5장 |

지금 이 중 어느 것도 없어서 생긴 문제는 없다. 데이터 300개, 모델 다섯 개, 실행 시간 몇 초짜리 실험에 실험 추적 도구와 분산 학습을 붙이면, 도구를 다루는 시간이 실험 시간보다 길어진다. 도구는 **그것이 없어서 문제가 생긴 다음에** 들이는 것이 좋은 순서다.

## 결과를 의심하는 첫 연습

숫자가 나왔다고 끝이 아니다. 완주한 다음 할 일은 그 숫자를 의심하는 것이다. 처음에는 세 가지 질문이면 충분하다.

**너무 좋은 결과.** 오차가 측정 잡음보다 훨씬 작거나, 학습 loss가 순식간에 0이 되면 1.3장의 누수를 의심한다. 이 장의 표에서 물리 특징 회귀의 테스트 RMSE(0.0089 s)도 측정 잡음(0.0100 s)보다 작다. 이것도 누수일까? 계산으로 판단할 수 있다. 잡음만 있는 오차 60개의 RMSE는 그 자체로 흔들리고, 그 표준편차는 대략 $\sigma/\sqrt{2n} = 0.01/\sqrt{120} \approx 0.0009$ s다. 0.0089 s는 0.0100 s에서 이 흔들림의 한두 배 거리에 있으므로 우연으로 충분히 설명된다. 1.3장의 중심 이동평균은 테스트 1000개에서 하한보다 훨씬 아래로 내려갔고, 그쪽이 누수의 신호였다. **"잡음보다 작다"는 사실보다, 표본 크기를 고려했을 때 얼마나 작은지가 중요하다.**

**너무 나쁜 결과.** 어떤 모델이 평균 예측보다 나쁘거나, 학습 오차부터 크면 코드나 설정을 먼저 의심한다. 단위 변환(도와 라디안), 표준화를 되돌리지 않은 예측, 학습률이 너무 커서 발산한 loss가 흔한 원인이다. 작은 진폭 공식이 선형 회귀보다 나쁜 것은 버그가 아니라 진폭 효과를 무시한 결과다. 원인을 설명할 수 있는 나쁜 결과와 설명할 수 없는 나쁜 결과를 구분한다(17.9장).

**재실행 시 달라지는 결과.** 같은 코드인데 MLP의 초기화 seed만 바꿔 다시 학습한다.

```python {exec="browser" session="pipeline"}
for seed in range(5):
    print(f"초기화 seed {seed}  MLP 검증 RMSE {rmse(train_mlp(seed)(X[va]), y[va]):.4f} s")
```

```output
초기화 seed 0  MLP 검증 RMSE 0.0232 s
초기화 seed 1  MLP 검증 RMSE 0.0272 s
초기화 seed 2  MLP 검증 RMSE 0.0307 s
초기화 seed 3  MLP 검증 RMSE 0.0297 s
초기화 seed 4  MLP 검증 RMSE 0.0275 s
```

검증 RMSE가 0.0232 s에서 0.0307 s까지 흔들린다. 앞의 표에 쓴 MLP의 숫자는 그중 **가장 좋은** seed 0의 결과였다. 다섯 번 중 가장 좋은 것만 보고하면 MLP를 과대평가하게 된다. 분할의 seed를 바꿔도 비슷한 흔들림이 생긴다(실습 참고). 이 흔들림보다 작은 차이로 "모델 A가 B보다 낫다"고 말할 수는 없다.

::widget{name="seed-variance"}

::quiz{id="suspicious-results"}

::quiz{id="where-next"}

이 세 질문은 끝이 아니라 시작이다. 결과를 체계적으로 의심하는 방법, 즉 여러 seed와 분할로 차이의 크기를 재고, 비교가 공정한지 확인하고, 주장의 범위를 정하는 방법은 12.5장, 16.6장, 23.6장에서 다룬다.

## 실습

공통 데이터셋으로 선형 baseline과 작은 MLP를 학습하고, 같은 분할·같은 metric으로 비교한 **한 쪽짜리 보고서**를 쓴다.

1. **스크립트를 실행한다.** 저장소 루트에서 다음 명령을 실행한다(집필 시점 예시). 환경 구성은 0.2장과 3.3장을 따른다.

   ```bash
   uv run --project labs python labs/research-landscape/first-pipeline/pipeline.py
   ```

   집필 시점의 실행 결과는 다음과 같다. 본문 코드 셀과 같은 숫자다.

   ```output
   # 진자 주기 회귀 — n=300, data_seed=0, split_seed=1
   # python 3.13.18, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1
   # 분할: 학습 180 / 검증 60 / 테스트 60, 측정 잡음 0.01 s
   RMSE [s]   학습    검증    테스트   모델
            0.5810  0.5639  0.5941   평균 예측
            0.0839  0.0742  0.0760   선형 회귀 (L, 진폭)
            0.1300  0.1447  0.1305   작은 진폭 공식 2π√(L/g)
            0.0094  0.0091  0.0089   물리 특징 + 선형 회귀
            0.0180  0.0232  0.0190   MLP (은닉 32, seed 0)
   ```

2. **흔들림을 잰다.** 분할 seed와 초기화 seed를 바꿔 실행한다. 예: `--split-seed 2 --init-seeds 0 1 2 3 4`. 분할 seed 2개 이상, 초기화 seed 5개 이상에서 MLP와 선형 회귀의 테스트 RMSE 범위(최소–최대)를 표로 만든다.
3. **보고서를 쓴다.** 한 쪽 안에 다음을 담는다.
   - 문제 정의(이 장의 표를 줄여서)
   - 결과 표: 모든 모델의 테스트 RMSE와, MLP는 여러 seed의 범위
   - 결론 두세 문장: 표가 지지하는 범위까지만 쓴다
   - 재실행 방법: 명령 한 줄과, 출력 첫 줄들에 찍힌 패키지 버전
4. **다른 사람에게 재실행을 부탁한다.** 보고서의 명령만 보고 같은 숫자가 나오는지 확인한다. 숫자가 다르면 무엇이 달랐는지(패키지 버전, 운영체제, 명령)를 찾아 보고서를 고친다.

이어서 두 연습문제를 푼다. 첫째는 다시 실행해도 같은 분할을 만드는 함수, 둘째는 이 장의 학습 루프 한 step이다.

::exercise{id="reproducible-split"}

::exercise{id="mlp-train-step"}

## 실패 사례

- **처음부터 Docker·분산학습·추상화된 trainer를 도입하는 것.** 첫 완주에 컨테이너, 설정 프레임워크, 범용 학습 클래스를 붙이면 도구를 다루는 데 시간을 쓰고 정작 한 번도 끝까지 가지 못한다. 또 도구가 무엇을 해결하는지 모른 채 쓰게 된다. 파일 하나, 프로세스 하나로 먼저 완주하고, 같은 불편을 두 번 겪은 부분부터 구조를 나눈다.
- **baseline 없이 신경망부터 학습하는 것.** 이 장에서 MLP만 학습했다면 "검증 RMSE 0.0232 s"는 좋은 숫자처럼 보였을 것이다. 4-파라미터 물리 특징 회귀가 그 절반 이하의 오차를 낸다는 사실은 baseline이 있어야 보인다. 평균 예측, 단순한 학습 모델, 분야의 지식으로 만든 모델을 먼저 돌린다(16.1장).
- **가장 좋은 seed의 숫자만 보고하는 것.** 다섯 번 중 가장 좋은 결과는 다섯 번의 평균보다 체계적으로 좋다. 여러 seed의 범위나 평균과 표준편차를 함께 보고한다.
- **테스트 데이터를 보고 모델을 고르는 것.** 테스트 점수를 보며 은닉 크기나 학습률을 바꾸면 테스트 데이터가 검증 데이터가 된다. 모델 선택은 검증 데이터로 끝내고, 테스트는 마지막에 한 번 연다.

## 통과 기준

- 다른 사람이 보고서의 명령 한 줄로 같은 결과 표의 숫자를 다시 얻는다.
- 한 쪽짜리 보고서에 문제 정의, 같은 분할·같은 지표로 비교한 baseline과 MLP의 결과 표, 여러 seed에서의 흔들림 범위, 표가 지지하는 범위의 결론이 들어 있다.
- 연습문제 두 개(`reproducible-split`, `mlp-train-step`)의 테스트를 통과한다.
