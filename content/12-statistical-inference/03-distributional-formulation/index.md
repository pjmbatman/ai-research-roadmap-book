---
id: statistical-inference/distributional-formulation
title: "분포 기반 문제 정의"
title_en: "Formulating Problems with Distributions"
tier: core
status: draft
prerequisites: [statistical-inference/likelihood]
---

# 분포 기반 문제 정의

회귀 모델이 "다음 측정값은 1.01이다"라고 답했다고 하자. 이 숫자는 무엇일까. 가장 자주 나오는 값일까, 평균일까, 절반의 측정이 이보다 작다는 뜻일까. 그리고 실제 측정이 0.7이나 2.3이 나올 가능성은 얼마나 될까. 점 예측 하나로는 어느 것도 알 수 없다. 이 장은 문제를 **분포**로 정의하는 법을 다룬다. 무엇의 분포가 필요한지 정하고, 점 예측이 그 분포에 대한 의사결정이라는 관점을 세우고, 분포를 출력하는 모델과 그 평가 방법을 설계한다.

## 학습 목표

- 문제에 필요한 분포가 무엇인지 정한다
- 점 예측을 분포에 대한 의사결정으로 이해한다
- 분포를 출력하는 모델을 설계한다

## 무엇의 분포가 필요한가

같은 데이터로도 질문에 따라 필요한 분포가 다르다. 잡음이 섞인 감쇠 진동자의 위치를 시간 $t$마다 측정한 데이터 $\mathcal{D} = \{(t_i, y_i)\}$를 예로 든다.

| 연구 질문 | 필요한 분포 | 다루는 곳 |
|---|---|---|
| 시각 $t$에서 측정값은 어떤 범위에 있을까? | 조건부 예측분포 $p(y \mid t)$ | 이 장 |
| 이 진동자의 감쇠비 $\zeta$는 얼마이고, 얼마나 확실한가? | 파라미터 사후분포 $p(\zeta \mid \mathcal{D})$ | 13장 |
| 그럴듯한 궤적을 새로 만들어라 | 데이터 분포 $p(\mathbf{x})$ | 19장 |
| 측정 뒤에 숨은 진짜 상태(위치·속도)는? | 잠재변수의 사후분포 $p(\mathbf{z} \mid \mathbf{y})$ | 24.6장 |
| 입력과 출력이 함께 어떻게 분포하는가? | 결합분포 $p(t, y)$ | 생성모델·결측 처리 |

모델의 출력도 세 종류로 구분해야 한다.

$$
\hat{y} = f_\theta(x), \qquad p_\theta(y \mid x), \qquad p(\theta \mid \mathcal{D}).
$$

각각 **점 예측**, **조건부 예측분포**, **파라미터의 사후분포**다. 첫째는 숫자 하나, 둘째는 입력마다 정해지는 출력의 분포, 셋째는 모델 파라미터 자체에 대한 불확실성이다. "모델이 불확실성을 출력한다"고 말할 때 둘째인지 셋째인지 밝혀야 한다.

::quiz{id="distribution-statements"}

:::note[과학자를 위해]
ML의 "모델"은 흔히 $p_\theta(y \mid x)$ 하나를 가리키고, 데이터가 생성된 물리 과정에 대한 가정은 손실함수 안에 숨어 있다. 예를 들어 제곱손실을 쓰는 순간, 측정 잡음이 가우시안이고 분산이 일정하다고 가정한 것과 같다(12.2장).
:::

## 결정 이론

조건부분포 $p(y \mid x)$를 안다고 해도, 결국 숫자 하나를 골라야 할 때가 있다. 어떤 숫자를 고를지는 **틀렸을 때의 비용**, 즉 손실에 달려 있다. 손실 $\ell(\hat{y}, y)$의 기대값을 최소화하는 선택이 최적 결정이다.

$$
\hat{y}^\star(x) = \arg\min_{\hat{y}} \ \mathbb{E}_{y \sim p(y \mid x)}\big[\ell(\hat{y}, y)\big].
$$

::quiz{id="squared-loss-optimum"}

**제곱손실.** $\mathbb{E}[(y - \hat{y})^2 \mid x]$를 $\hat{y}$로 미분하면 $-2\,\mathbb{E}[y - \hat{y} \mid x]$이고, 이를 0으로 두면

$$
\hat{y}^\star = \mathbb{E}[y \mid x],
$$

즉 **조건부 평균**이다. MSE로 학습한 회귀 모델은 (용량과 데이터가 충분하다면) 조건부 평균을 추정한다.

**절댓값손실.** $\mathbb{E}[\,\lvert y - \hat{y} \rvert \mid x]$의 최적해는 **조건부 중앙값**이다. $\hat{y}$를 조금 올리면, $\hat{y}$보다 큰 $y$에 대한 손실은 줄고 작은 $y$에 대한 손실은 는다. 두 확률이 같아지는 점, 즉 중앙값에서 균형이 맞는다.

**Pinball loss.** 잔차 $u = y - \hat{y}$에 대해

$$
\rho_\tau(u) = u\,\big(\tau - \mathbb{1}[u < 0]\big)
$$

로 정의하면, 과소 예측($u > 0$)에는 $\tau$, 과대 예측($u < 0$)에는 $1 - \tau$의 가중치가 붙는다. 같은 균형 논리로 최적해는 $p(y \mid x)$의 **$\tau$ 분위수**가 된다. $\tau = 0.5$이면 절댓값손실의 절반이므로 중앙값과 같다.

::widget{name="loss-to-estimator"}

오른쪽 꼬리가 긴 측정 잡음에서 세 손실이 고르는 값을 직접 확인한다. 후보 격자에서 각 손실을 가장 작게 만드는 값을 찾아 표본의 평균·중앙값·분위수와 비교한다.

```python {exec="browser" session="dist"}
import numpy as np

rng = np.random.default_rng(0)
y = rng.exponential(scale=1.0, size=2000)     # 오른쪽 꼬리가 긴 측정 잡음

grid = np.linspace(0, 4, 801)
def argmin(loss):
    return grid[np.argmin([loss(q) for q in grid])]

tau = 0.9
q_sq = argmin(lambda q: np.mean((y - q) ** 2))
q_ab = argmin(lambda q: np.mean(np.abs(y - q)))
q_pb = argmin(lambda q: np.mean((y - q) * (tau - (y < q))))

print(f"제곱손실 최소  {q_sq:.3f}   표본 평균       {y.mean():.3f}")
print(f"절댓값손실 최소 {q_ab:.3f}   표본 중앙값     {np.median(y):.3f}")
print(f"pinball(0.9)  {q_pb:.3f}   표본 90% 분위수 {np.quantile(y, tau):.3f}")
```

```output
제곱손실 최소  1.010   표본 평균       1.012
절댓값손실 최소 0.705   표본 중앙값     0.704
pinball(0.9)  2.315   표본 90% 분위수 2.314
```

비대칭 분포에서는 세 대표값이 크게 다르다. "예측값"이라는 말만으로는 이 중 무엇인지 알 수 없다.

::quiz{id="absolute-loss-optimum"}

분포가 **봉우리 두 개**(쌍봉)이면 문제가 더 심각하다. $y$가 $-1$ 근처나 $+1$ 근처에서만 나온다면 평균은 0 근처, 즉 실제로는 거의 관측되지 않는 값이다. 이때 MSE 모델이 0을 예측하는 것은 학습 실패가 아니라 문제 정의(점 예측 + 제곱손실)가 데이터에 맞지 않는다는 신호다.

::quiz{id="bimodal-mean"}

## 분포를 출력하는 모델

분포를 직접 출력하게 하면 대표값을 고르기 전의 정보를 잃지 않는다. 방법은 간단하다. 모델의 출력을 숫자 하나가 아니라 **분포의 파라미터**로 바꾸고, 손실을 그 분포의 음의 로그우도(NLL)로 둔다(12.2장).

**이분산 가우시안 회귀.** 평균과 표준편차를 함께 출력한다.

$$
p_\theta(y \mid x) = \mathcal{N}\big(\mu_\theta(x), \sigma_\theta(x)^2\big), \qquad
-\log p_\theta(y \mid x) = \log \sigma_\theta(x) + \frac{(y - \mu_\theta(x))^2}{2\sigma_\theta(x)^2} + \text{상수}.
$$

$\sigma$가 입력마다 달라질 수 있으므로 이분산(heteroscedastic)이라 한다. $\sigma$는 양수여야 하므로 신경망에서는 출력에 softplus나 지수 함수를 씌운다. $\sigma$가 상수이면 NLL 최소화는 제곱오차 최소화와 같아진다.

신호가 클수록 센서 잡음도 커지는 측정을 흉내 낸 데이터로 확인한다. 참 모델은 평균 $2x$, 표준편차 $0.1 + 0.5x$다. 평균은 직선, $\log\sigma$도 $x$의 직선으로 두고 NLL을 경사하강으로 최소화한다.

```python {exec="browser" session="dist"}
rng = np.random.default_rng(1)
x = rng.uniform(0, 1, size=2000)
y = 2.0 * x + rng.normal(size=2000) * (0.1 + 0.5 * x)   # 잡음이 x에 비례해 커진다

w, c = np.polyfit(x, y, 1)                  # 평균은 최소제곱으로 시작
alpha, beta = 0.0, np.log(np.std(y - (w * x + c)))
lr = 0.1
for _ in range(3000):
    mu = w * x + c
    s = alpha * x + beta                    # log sigma
    r2 = (y - mu) ** 2 * np.exp(-2 * s)
    d_mu = -(y - mu) * np.exp(-2 * s)       # dNLL/dmu
    d_s = 1.0 - r2                          # dNLL/d(log sigma)
    w, c = w - lr * np.mean(d_mu * x), c - lr * np.mean(d_mu)
    alpha, beta = alpha - lr * np.mean(d_s * x), beta - lr * np.mean(d_s)

for x0 in (0.1, 0.5, 0.9):
    print(f"x = {x0}:  평균 {w * x0 + c:.3f}   sigma {np.exp(alpha * x0 + beta):.3f}   (참값 sigma {0.1 + 0.5 * x0:.3f})")
```

```output
x = 0.1:  평균 0.203   sigma 0.170   (참값 sigma 0.150)
x = 0.5:  평균 0.997   sigma 0.316   (참값 sigma 0.350)
x = 0.9:  평균 1.791   sigma 0.588   (참값 sigma 0.550)
```

모델은 $\sigma$가 $x$와 함께 커진다는 사실을 배웠다. 그런데 참값과 조금씩 어긋난다. 참 $\sigma$는 $x$의 직선인데 모델은 $\log\sigma$를 직선으로 두었기 때문이다. 데이터를 아무리 늘려도 이 차이는 사라지지 않는다. 이 장의 마지막 절에서 다룰 **모델 구조 불확실성**의 작은 예다.

**분위수 회귀.** 분포의 모양을 가정하지 않고, 원하는 분위수마다 출력 하나를 두고 pinball loss로 학습한다. $\tau = 0.1$과 $\tau = 0.9$ 두 출력이 있으면 80% 예측 구간이 된다. 서로 다른 $\tau$의 출력이 엇갈리는 분위수 교차(quantile crossing)가 생길 수 있어, 출력 사이에 순서 제약을 두기도 한다.

::quiz{id="pinball-quantile"}

**혼합 밀도 네트워크.** 쌍봉처럼 봉우리가 여럿인 분포는 가우시안 하나로 표현할 수 없다. mixture density network는 가우시안 $K$개의 혼합 가중치·평균·표준편차를 모두 출력한다.

## 적정 점수 규칙

분포를 출력하는 모델은 RMSE 같은 점 예측 지표로 평가할 수 없다. 평균만 맞으면 분포의 폭이 엉망이어도 RMSE는 같기 때문이다. 예측분포 $P$와 실제 관측 $y$로 점수 $S(P, y)$를 매기는 규칙 중, **참 분포를 그대로 보고할 때 기대 점수가 가장 좋은** 규칙을 적정 점수 규칙(proper scoring rule)이라 한다. 적정 규칙으로 평가하면 모델이 분포를 일부러 좁히거나 넓혀서 점수를 올릴 수 없다.

가장 많이 쓰는 두 가지는 다음과 같다(둘 다 작을수록 좋다).

- **NLL**(음의 로그우도): $-\log p(y)$. 관측값이 예측분포에서 드문 곳에 떨어지면 크게 벌한다.
- **CRPS**(continuous ranked probability score): 예측 누적분포함수와 관측값의 계단함수 사이의 제곱 차이를 적분한 값이다. 단위가 $y$와 같아 해석이 쉽고, 점 예측의 절댓값 오차를 분포로 일반화한 것으로 볼 수 있다. 가우시안 예측분포 $\mathcal{N}(\mu, \sigma^2)$에 대해서는 $z = (y - \mu)/\sigma$로 두고 $\mathrm{CRPS} = \sigma\big[z(2\Phi(z) - 1) + 2\varphi(z) - 1/\sqrt{\pi}\big]$이다. 여기서 $\Phi$, $\varphi$는 표준정규분포의 누적분포함수와 밀도함수다.

참 분포가 $\mathcal{N}(0, 1)$일 때, 평균은 맞고 폭만 다른 세 예측분포를 비교한다.

::quiz{id="overconfident-score"}

```python {exec="browser" session="dist"}
from math import erf, pi, sqrt

Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
def crps_gauss(y, mu, s):
    z = (y - mu) / s
    pdf = np.exp(-z**2 / 2) / sqrt(2 * pi)
    return np.mean(s * (z * (2 * Phi(z) - 1) + 2 * pdf - 1 / sqrt(pi)))

def nll_gauss(y, mu, s):
    return np.mean(0.5 * np.log(2 * pi * s**2) + (y - mu) ** 2 / (2 * s**2))

y = np.random.default_rng(2).normal(size=5000)     # 참 분포: N(0, 1)
for s in (0.3, 1.0, 3.0):
    print(f"예측 N(0, {s}^2)  NLL {nll_gauss(y, 0, s):.3f}   CRPS {crps_gauss(y, 0, s):.3f}")
```

```output
예측 N(0, 0.3^2)  NLL 5.286   CRPS 0.668
예측 N(0, 1.0^2)  NLL 1.420   CRPS 0.566
예측 N(0, 3.0^2)  NLL 2.073   CRPS 0.831
```

두 규칙 모두 참 분포($\sigma = 1$)에 가장 좋은 점수를 준다. 과신한 분포($\sigma = 0.3$)는 NLL에서 특히 크게 벌을 받는다. 예측분포의 평가는 21.2장에서 calibration과 함께 더 자세히 다룬다.

## 불확실성의 출처 구분

예측분포의 폭에는 성격이 다른 불확실성이 섞여 있다.

| 출처 | 진동자 예 | 데이터를 늘리면 | 대응 |
|---|---|---|---|
| **관측 잡음** (aleatoric) | 센서 잡음 | 줄지 않는다 | 더 좋은 센서, 잡음을 분포로 모델링 |
| **파라미터 불확실성** (epistemic) | 짧은 관측 구간 때문에 감쇠비 $\zeta$가 불확실 | 줄어든다 | 데이터 추가, 베이지안 추론(13장), 앙상블(21.1장) |
| **모델 구조 불확실성** | 실제로는 비선형 감쇠인데 선형 감쇠 모델을 씀 | 줄지 않는다 | 모델 비교·진단, 물리 지식 반영(24장) |

::quiz{id="irreducible-uncertainty"}

출처를 구분하면 다음에 할 일이 달라진다. 관측 잡음이 지배적이면 모델을 키워도 소용이 없다. 파라미터 불확실성이 지배적이면 어떤 조건에서 데이터를 더 모을지(25.3장 실험 설계)가 핵심 질문이 된다. 모델 구조가 틀렸다면, 앞 절의 이분산 예제처럼 데이터를 늘려도 오차가 남는다.

:::note[AI 연구자를 위해]
과학 데이터에서는 측정 장비의 사양서나 반복 측정으로 관측 잡음의 크기를 따로 알 수 있는 경우가 많다. 이 정보를 쓰면 모델이 학습한 $\sigma$가 측정 잡음보다 지나치게 작거나 큰지 검증할 수 있다.
:::

## 실습

같은 데이터에 세 모델을 학습하고, 각 모델이 내놓는 80% 예측 구간이 실제로 관측값의 80%를 포함하는지 입력 구간별로 확인한다. 코드는 `labs/statistical-inference/distributional-formulation/compare_models.py`다. 데이터는 앞 절과 같이 $y = 2x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, (0.1 + 0.5x)^2)$이고, 이번에는 $\sigma$를 $x$의 직선(softplus로 양수 보장)으로 두어 참 모델과 같은 꼴로 학습한다. 집필 시점의 실행 결과는 다음과 같다(난수 시드 0, 시험 데이터 20,000개).

```output
80% 예측 구간의 실제 포함률 (목표 0.80)
모델              x<1/3   1/3~2/3  x>2/3   전체
MSE + 등분산       0.982   0.832    0.644   0.818
이분산 가우시안        0.819   0.763    0.818   0.800
분위수 회귀          0.822   0.794    0.785   0.800
```

MSE 모델에 일정한 폭의 구간을 붙이면 **전체** 포함률은 0.818로 목표에 가깝다. 그러나 구간별로 보면 잡음이 작은 곳에서는 지나치게 넓고(0.982), 잡음이 큰 곳에서는 지나치게 좁다(0.644). 평균 지표 하나로는 이 실패가 보이지 않는다. 이분산 모델과 분위수 회귀는 구간별로도 목표 근처를 유지한다.

이어서 두 연습문제를 푼다.

::exercise{id="pinball-loss"}

::exercise{id="gaussian-nll"}

## 실패 사례

- **다봉 분포에서 조건부 평균을 대표값으로 쓰는 것.** 상태가 두 가지로 갈리는 현상(예: 고장·정상)에서 MSE 모델의 예측은 어느 상태에도 해당하지 않는 중간값이 된다.
- **예측 구간과 신뢰 구간을 혼동하는 것.** 신뢰 구간은 평균 같은 고정된 미지량의 범위이고, 데이터가 늘면 좁아진다. 예측 구간은 새 관측값의 범위이고, 관측 잡음 때문에 데이터가 늘어도 일정 폭 아래로 좁아지지 않는다.
- **점 예측 지표만으로 예측분포를 평가하는 것.** RMSE는 분포의 폭을 보지 않는다. 적정 점수 규칙과 구간별 포함률을 함께 본다.
- **전체 포함률만 보고 구간이 맞다고 결론 내리는 것.** 실습 결과처럼 전체는 맞아도 조건별로는 크게 틀릴 수 있다.

## 통과 기준

- 자신의 연구 문제를 '어떤 분포를 학습하고 어떤 결정에 쓰는가'로 다시 쓴다.
- 연습문제 두 개(`pinball-loss`, `gaussian-nll`)의 테스트를 통과한다.
