---
id: classical-ml/time-series-analysis
title: "시계열 분석"
title_en: "Time Series Analysis"
tier: core
status: draft
prerequisites: [classical-ml/model-selection, probability/stochastic-processes]
---

# 시계열 분석

기온, 전력 수요, 배터리 전압, 지진계 신호, 유전자 발현의 시간 변화는 모두 시계열이다. 시계열의 샘플은 서로 독립이 아니다. 지금 값은 직전 값과 닮았고, 하루나 일 년 주기로 반복되며, 천천히 흘러가는 추세를 가진다. 이 구조는 예측의 근거가 되지만, 독립 샘플을 가정한 도구(무작위 분할, 표준오차 공식)를 그대로 쓰면 결과를 체계적으로 왜곡한다.

이 장은 11.7장의 AR 과정과 정상성에서 출발해, 시계열을 구성 요소로 분해하고, 고전 예측 모델과 상태공간 모델을 적용하고, 마지막으로 시간 순서를 지키는 평가 프로토콜을 설계한다. 딥러닝 시계열 모델(18장)을 쓰더라도 이 장의 baseline과 평가 방식은 그대로 필요하다.

## 학습 목표

- 시계열의 구조를 분해한다
- 고전 예측 모델을 적용한다
- 시계열 평가 방식을 설계한다

## 시계열의 구조

**왜 필요한가.** 시계열 예측의 첫 질문은 "무엇이 반복되고 무엇이 남는가"다. 하루 주기가 분명한 기온을 주기 없이 모델링하면 모델은 매일 같은 패턴을 다시 배워야 하고, 추세를 무시하면 학습 기간 밖에서 체계적으로 틀린다. 구조를 먼저 분해하면 각 부분에 맞는 도구를 쓸 수 있다.

**정의.** 고전적인 가법 분해는 시계열을 세 부분의 합으로 본다.

$$
y_t = T_t + S_t + R_t.
$$

$T_t$는 **추세**(trend, 천천히 변하는 수준), $S_t$는 **계절성**(seasonality, 주기 $m$으로 반복되는 패턴: 하루 $m = 24$시간, 일 년 $m = 12$개월), $R_t$는 나머지(remainder)다. 변동의 크기가 수준에 비례해 커지면 로그를 취한 뒤 가법 분해를 하거나 곱셈 분해 $y_t = T_t S_t R_t$를 쓴다.

나머지 $R_t$도 대개 독립 잡음이 아니다. 이를 재는 도구가 **자기상관 함수**(ACF, autocorrelation function)다. 시차(lag) $k$에서

$$
\hat{\rho}(k) = \frac{\sum_{t=1}^{T-k}(y_t - \bar{y})(y_{t+k} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}
$$

이다. 11.7장에서 본 것처럼 AR(1) 과정 $y_t = \phi y_{t-1} + \varepsilon_t$의 ACF는 $\phi^k$로 감소하고, 평균·분산·자기공분산이 시간에 따라 변하지 않는 (약)정상(weakly stationary) 과정에서만 ACF가 시차만의 함수로 잘 정의된다. 추세가 있으면 정상이 아니고, 계절성이 있어도 평균이 시각에 따라 바뀌므로 정상이 아니다.

60일 동안의 시간당 기온을 만든다. 선형 추세(하루 0.05 °C), 하루 주기(진폭 3 °C), AR(1) 잡음($\phi = 0.8$)의 합이다. 선형 추세를 최소제곱으로 빼고, 시각(0–23시)별 평균을 계절성으로 빼서 잔차를 얻은 뒤, 원 시계열과 잔차의 ACF를 비교한다.

```python {exec="browser" session="ts"}
import numpy as np

rng = np.random.default_rng(0)
n = 24 * 60                                                    # 60일 × 24시간
t_h = np.arange(n)                                             # 시간 [h]
ar = np.zeros(n)
for t in range(1, n):
    ar[t] = 0.8 * ar[t - 1] + rng.normal(0, 0.5)               # AR(1) 잡음
temp_c = 15 + 0.05 * t_h / 24 + 3 * np.sin(2 * np.pi * t_h / 24) + ar   # 기온 [°C], (n,)

def acf(x, lags):
    x = x - x.mean()
    return np.array([np.sum(x[:-k] * x[k:]) / np.sum(x * x) for k in lags])

coef = np.polyfit(t_h, temp_c, 1)                              # 선형 추세 [°C/h, °C]
detr = temp_c - np.polyval(coef, t_h)
season = np.array([detr[t_h % 24 == h].mean() for h in range(24)])   # (24,) 시각별 평균
resid = detr - season[t_h % 24]                                # (n,) 추세·계절성을 뺀 잔차
lags = [1, 2, 6, 12, 24]
print("시차          ", lags)
print("원 시계열 ACF ", np.round(acf(temp_c, lags), 2))
print("잔차 ACF      ", np.round(acf(resid, lags), 2))
print("AR(1) 이론값 0.8^k", np.round(0.8 ** np.array(lags), 2))
print(f"추정한 추세 {24 * coef[0]:.3f} °C/일 (참값 0.05)")
```

```output
시차           [1, 2, 6, 12, 24]
원 시계열 ACF  [ 0.95  0.86  0.17 -0.6   0.86]
잔차 ACF       [ 0.81  0.66  0.26  0.06 -0.01]
AR(1) 이론값 0.8^k [0.8  0.64 0.26 0.07 0.  ]
추정한 추세 0.052 °C/일 (참값 0.05)
```

원 시계열의 ACF는 시차 12에서 $-0.60$, 시차 24에서 $0.86$으로, 하루 주기가 지배한다. 시차 1의 0.95도 대부분은 "한 시간 뒤에도 비슷한 시각대"라는 계절성 때문이다. 추세와 계절성을 뺀 잔차의 ACF는 0.81, 0.66, 0.26, 0.06으로 AR(1) 이론값 $0.8^k$를 따른다. 분해가 제대로 되었다면 잔차에는 단기 의존성만 남고, 그 구조는 다음 절의 AR 모델이 담당한다.

::quiz{id="acf-seasonal-lag"}

**분해의 선택.** 이 예는 추세를 직선으로, 계절성을 시각별 평균으로 잡았다. 고전적인 분해는 추세를 주기 길이의 이동평균으로 잡고, STL처럼 국소 회귀로 추세와 계절성을 함께 추정하는 방법도 널리 쓰인다. 방법에 따라 잔차의 모양이 달라진다. 예를 들어 앞뒤 12시간을 쓰는 24시간 중심 이동평균은 잔차의 느린 변동까지 추세로 흡수해, 잔차의 자기상관을 실제보다 작게 만든다. 그리고 **중심 이동평균은 미래 값을 쓴다.** 분해는 이미 관측한 기간을 이해하는 분석 도구이고, 이렇게 만든 성분을 예측 모델의 입력 특징으로 쓰면 미래 정보가 새어 들어간다(이 장의 "평가" 절).

:::note[분야 예시]
대기 CO₂ 농도의 장기 관측 기록은 분해의 교과서적인 예다. 화석연료 배출에 따른 꾸준한 증가(추세)에, 북반구 식생의 광합성과 호흡에 따른 연 주기(계절성)가 겹쳐 있다. 계절 진폭 자체가 수십 년에 걸쳐 조금씩 변한다는 분석도 있는데, 이런 질문은 계절성을 고정된 패턴으로 가정하는 분해로는 답할 수 없고, 시간에 따라 변하는 계절 성분을 허용하는 방법이 필요하다.
:::

## 고전 모델

**AR 모델.** 자기회귀(autoregressive) 모델 AR($p$)는 현재 값을 과거 $p$개 값의 선형 결합과 새 잡음으로 쓴다.

$$
y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2) \text{ 독립}.
$$

$\phi_k$는 시차 $k$의 계수, $c$는 상수항, $\sigma^2$는 잡음의 분산이다. 계수는 lag 행렬에 대한 최소제곱으로 추정할 수 있다. 행 $t$의 특징이 $(y_{t-1}, \dots, y_{t-p})$이고 목표가 $y_t$인 선형 회귀다. 앞의 잔차에 AR(1)–AR(3)을 맞춘다.

```python {exec="browser" session="ts"}
def fit_ar(x, p):                                              # x_t = φ_1 x_{t-1} + ... + φ_p x_{t-p} + e_t
    L = np.column_stack([x[p - k:len(x) - k] for k in range(1, p + 1)])   # (n-p, p) lag 행렬
    phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None)
    sigma = np.std(x[p:] - L @ phi)
    return phi, sigma

for p in (1, 2, 3):
    phi, sigma = fit_ar(resid, p)
    print(f"AR({p}): φ = {np.round(phi, 3)}   잔차 표준편차 {sigma:.3f}")
```

```output
AR(1): φ = [0.81]   잔차 표준편차 0.490
AR(2): φ = [0.808 0.003]   잔차 표준편차 0.490
AR(3): φ = [ 0.807 -0.004  0.009]   잔차 표준편차 0.490
```

AR(1)의 계수는 0.81로 참값 0.8에 가깝다. AR(2), AR(3)에서 추가한 계수는 0에 가깝고 잔차 표준편차도 0.490으로 줄지 않는다. 참 과정이 AR(1)이므로 더 긴 시차는 정보를 더하지 않는다. 실제 데이터에서는 차수를 AIC·BIC나 시간 순서를 지킨 검증 오차로 고르고, 잔차의 ACF가 0 근처인지 확인한다(12.6장의 잔차 진단, Durbin–Watson 검정과 같은 생각이다).

**ARIMA.** 추세나 단위근(11.7장)이 있는 비정상 시계열은 차분 $\Delta y_t = y_t - y_{t-1}$을 $d$번 해서 정상으로 만든 뒤 모델링한다. 여기에 과거 잡음의 선형 결합인 이동평균(MA) 항 $\theta_1\varepsilon_{t-1} + \cdots + \theta_q\varepsilon_{t-q}$를 더한 것이 ARIMA($p, d, q$)다. 계절 차분 $y_t - y_{t-m}$과 계절 AR·MA 항을 추가한 계절 ARIMA도 있다. 모델의 식별과 적합은 Box–Jenkins 방법론으로 정리되어 있고, 실제 적합은 `statsmodels` 같은 라이브러리를 쓴다(이 책의 실습 환경에는 포함되지 않으므로, 쓰려면 설치하고 그 버전의 문서를 확인한다).

**지수평활.** 단순 지수평활(simple exponential smoothing)은 수준 $\ell_t$를 새 관측과 이전 수준의 가중 평균으로 갱신한다.

$$
\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}, \qquad \hat{y}_{t+h} = \ell_t.
$$

$\alpha \in (0, 1]$이 크면 최근 관측을 많이 믿고, 작으면 과거를 오래 기억한다. 전개하면 과거 관측에 지수적으로 줄어드는 가중치 $\alpha(1-\alpha)^j$를 주는 가중 평균이다. 추세와 계절 성분에도 같은 갱신을 적용한 것이 Holt–Winters 방법이다. 지수평활과 ARIMA는 계산이 가볍고 해석이 쉬워, 복잡한 모델을 평가할 때 반드시 함께 돌려 볼 baseline이다. 두 계열의 이론과 실무는 Hyndman, Athanasopoulos(2021)가 공개 교과서로 정리했다.

**lag 특징.** 트리 앙상블이나 신경망 같은 일반 회귀 모델로 시계열을 예측할 때는 과거 값을 특징으로 만든다. 직전 값들($y_{t-1}, y_{t-2}$), 계절 시차($y_{t-24}$), 과거 구간의 평균·최댓값, 달력 특징(시각, 요일), 외생 변수(예보된 기온)가 흔하다. 원칙은 하나다. **예측 시점에 실제로 알 수 있는 값만** 특징으로 쓴다. $h$시간 앞을 예측한다면 lag 특징은 $y_{t-h}$보다 오래된 값이어야 하고, 이동평균은 과거 쪽으로만 계산해야 한다.

## 상태공간 모델

**왜 필요한가.** 많은 과학 측정에서 관심 있는 양(배터리의 충전 상태, 위성의 위치, 생태계의 개체 수)은 직접 관측되지 않고, 잡음 섞인 측정을 통해서만 보인다. 상태공간 모델(state-space model)은 이 구분을 명시한다.

**정의.** 선형 가우스 상태공간 모델은 두 방정식으로 이루어진다.

$$
\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}) \qquad \text{(상태 방정식)}
$$

$$
\mathbf{y}_t = \mathbf{H}\mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}) \qquad \text{(관측 방정식)}
$$

$\mathbf{x}_t$는 숨은 상태, $\mathbf{F}$는 상태 전이 행렬, $\mathbf{y}_t$는 관측, $\mathbf{H}$는 관측 행렬, $\mathbf{Q}$와 $\mathbf{R}$은 상태 잡음과 관측 잡음의 공분산이다. AR·ARIMA·지수평활은 모두 이 형태로 쓸 수 있다. 예를 들어 단순 지수평활은 상태가 random walk로 움직이고($\mathbf{F} = 1$) 관측에 잡음이 더해지는 **local level 모델**과 대응한다.

**Kalman filter.** 모델이 선형이고 잡음이 가우스이면, 관측 $\mathbf{y}_1, \dots, \mathbf{y}_t$가 주어졌을 때 상태의 조건부분포는 가우스이고 그 평균과 공분산을 재귀적으로 계산할 수 있다(Kalman, 1960). 스칼라 local level 모델에서 한 step은 다음과 같다.

1. **예측**: $m_{t|t-1} = m_{t-1|t-1}$, $P_{t|t-1} = P_{t-1|t-1} + q$
2. **갱신**: Kalman gain $K_t = P_{t|t-1}/(P_{t|t-1} + r)$로 $m_{t|t} = m_{t|t-1} + K_t(y_t - m_{t|t-1})$, $P_{t|t} = (1 - K_t)P_{t|t-1}$

$K_t$는 예측의 불확실성 $P_{t|t-1}$과 관측 잡음 $r$의 비로 "새 관측을 얼마나 믿을지"를 정한다. 11.4장의 다변량 가우스 조건부분포를 시간에 따라 반복 적용한 것이다. 정상 상태에서 $K_t$가 상수가 되면 갱신식은 $\alpha = K$인 단순 지수평활과 같은 꼴이다.

## 예측·필터링·평활

같은 상태공간 모델로 세 가지 다른 질문을 할 수 있다. 어떤 관측을 쓸 수 있느냐가 다르다.

| 문제 | 추정 대상 | 쓰는 관측 | 쓰임 |
|---|---|---|---|
| **예측**(prediction, forecasting) | $\mathbf{x}_{t+h}$ | $\mathbf{y}_1, \dots, \mathbf{y}_t$ | 미래 계획, 제어 |
| **필터링**(filtering) | $\mathbf{x}_t$ | $\mathbf{y}_1, \dots, \mathbf{y}_t$ | 실시간 상태 추정 (예: 배터리 관리 시스템의 SOC) |
| **평활**(smoothing) | $\mathbf{x}_t$ | $\mathbf{y}_1, \dots, \mathbf{y}_T$ ($T > t$) | 사후 분석, 데이터 재구성 |

평활은 Kalman filter를 앞으로 돌린 뒤 뒤에서부터 미래 관측의 정보를 거꾸로 전파하는 Rauch–Tung–Striebel(RTS) smoother로 계산한다(Rauch 외, 1965). 숨은 상태가 random walk(상태 잡음 표준편차 0.05)이고 관측 잡음 표준편차가 0.5인 시계열 500개 시점에서 네 가지 추정의 오차를 비교한다.

```python {exec="browser" session="ts"}
q, r = 0.05**2, 0.5**2                                         # 상태 잡음, 관측 잡음 분산
x_true = np.cumsum(rng.normal(0, np.sqrt(q), 500))             # 숨은 상태: random walk, (T,)
y_obs = x_true + rng.normal(0, np.sqrt(r), 500)                # 관측, (T,)

m_pred, P_pred, m_filt, P_filt = (np.zeros(500) for _ in range(4))
m, P = 0.0, 10.0                                               # 사전 평균·분산
for t in range(500):
    m_pred[t], P_pred[t] = m, P + q                            # 예측: x_t | y_1..y_{t-1}
    K = P_pred[t] / (P_pred[t] + r)                            # Kalman gain
    m = m_pred[t] + K * (y_obs[t] - m_pred[t])                 # 필터링: x_t | y_1..y_t
    P = (1 - K) * P_pred[t]
    m_filt[t], P_filt[t] = m, P
m_smooth = m_filt.copy()                                       # 평활: x_t | y_1..y_T (RTS)
for t in range(498, -1, -1):
    G = P_filt[t] / P_pred[t + 1]
    m_smooth[t] = m_filt[t] + G * (m_smooth[t + 1] - m_pred[t + 1])

def rmse(a):
    return np.sqrt(np.mean((a[50:] - x_true[50:]) ** 2))       # 초기 50개 제외
print(f"상태 추정 RMSE  관측 그대로 {rmse(y_obs):.3f}   한 step 예측 {rmse(m_pred):.3f}"
      f"   필터링 {rmse(m_filt):.3f}   평활 {rmse(m_smooth):.3f}")
```

```output
상태 추정 RMSE  관측 그대로 0.500   한 step 예측 0.143   필터링 0.139   평활 0.101
```

관측을 그대로 쓰면 오차가 관측 잡음 크기(0.5)다. 과거 관측만 쓰는 필터링은 0.139로 오차를 3분의 1 이하로 줄이고, 아직 $y_t$를 보지 않은 한 step 예측(0.143)은 그보다 약간 나쁘다. 앞뒤 관측을 모두 쓰는 평활은 0.101로 가장 정확하다. **평활의 정확도는 미래를 보았기 때문에 얻은 것이다.** 평활된 시계열로 예측 모델을 학습·평가하거나, 평활 성능을 실시간 시스템의 성능으로 보고하면 미래 정보 누수다. 연구 질문이 "지금 알 수 있는가"(필터링)인지 "나중에 재구성할 수 있는가"(평활)인지를 먼저 정한다.

::quiz{id="filter-vs-smoother"}

:::note[분야 예시]
배터리 관리 시스템(BMS)은 직접 잴 수 없는 충전 상태(SOC)를 전류 적산과 전압 측정으로 실시간 추정한다. 등가회로 모델을 상태 방정식으로, 단자 전압을 관측 방정식으로 두고 Kalman filter 계열(비선형이면 extended·unscented Kalman filter)을 쓰는 것이 대표적인 접근이다. 이때 성능 평가는 필터링 기준이어야 한다. 실험이 끝난 뒤 전체 데이터로 평활한 SOC는 "참값" 근사로 쓸 수는 있어도, 온라인 추정 성능의 근거가 될 수는 없다. 배터리 모델 자체는 30.2장에서 다룬다.
:::

## 평가

**왜 필요한가.** 시계열 모델의 쓰임은 미래 예측이다. 그러므로 평가는 "과거만 보고 미래를 예측했을 때"의 오차를 재야 한다. 무작위 분할은 이 조건을 깬다.

**rolling origin.** 예측 원점(forecast origin) $o$를 시간 순서대로 옮겨 가며, 원점마다 $o$까지의 데이터로 모델을 학습(또는 갱신)하고 $o+1, \dots, o+H$를 예측해 실제 값과 비교한다(Tashman, 2000). 이것을 **rolling-origin 평가** 또는 backtesting이라 한다. 학습 구간은 처음부터 원점까지 늘어나는 확장 창(expanding window)이나, 최근 일정 기간만 쓰는 이동 창(sliding window)으로 잡는다. 결과는 **horizon $h$별 오차**로 정리한다. 한 시간 뒤와 하루 뒤의 예측은 다른 과제이고, 평균 오차 하나로 묶으면 어느 horizon에서 어떤 모델이 나은지가 사라진다.

앞의 기온 시계열에서 40일째부터 하루 간격으로 원점 20개를 두고 24시간 앞까지 예측한다. 세 모델을 비교한다. persistence는 마지막 관측을 반복하고, seasonal naive는 하루 전 같은 시각의 값을 쓴다(16.1장의 단순 baseline). 세 번째는 원점까지의 데이터로 추세·계절성·AR(1)을 매번 다시 적합한다.

```python {exec="browser" session="ts"}
def forecast_all(y, o, H):                                     # 시점 o까지만 보고 o+1..o+H를 예측
    th = np.arange(o + 1)
    c = np.polyfit(th, y[: o + 1], 1)
    d = y[: o + 1] - np.polyval(c, th)
    s = np.array([d[th % 24 == h].mean() for h in range(24)])
    e = d - s[th % 24]
    phi = fit_ar(e, 1)[0][0]
    fut = o + np.arange(1, H + 1)                              # 예측할 시각, (H,)
    return {"persistence": np.full(H, y[o]),
            "seasonal naive": y[fut - 24],
            "추세+계절+AR(1)": np.polyval(c, fut) + s[fut % 24] + phi ** np.arange(1, H + 1) * e[-1]}

H, origins = 24, range(24 * 40 - 1, n - 24, 24)                # 40일째부터 하루 간격, 원점 20개
err = {}
for o in origins:
    for name, f in forecast_all(temp_c, o, H).items():
        err.setdefault(name, []).append(f - temp_c[o + 1: o + 1 + H])
print(f"원점 {len(origins)}개, RMSE [°C]   h=1    h=3    h=6   h=12   h=24")
for name, e in err.items():
    rmse = np.sqrt(np.mean(np.array(e) ** 2, axis=0))          # (H,) horizon별
    print(f"{name:18s}", " ".join(f"{rmse[h - 1]:6.2f}" for h in (1, 3, 6, 12, 24)))
```

```output
원점 20개, RMSE [°C]   h=1    h=3    h=6   h=12   h=24
persistence          1.16   2.55   4.05   1.76   1.07
seasonal naive       1.17   1.10   1.14   0.96   1.07
추세+계절+AR(1)          0.63   0.79   1.04   0.77   0.84
```

horizon마다 순위가 다르다. persistence는 1시간 뒤에는 seasonal naive와 비슷하지만(1.16), 6시간 뒤에는 오차가 4.05까지 커지고, 24시간 뒤에는 다시 같은 시각이 되어 seasonal naive와 같아진다(1.07). 원점이 모두 매일 같은 시각(23시)이라, 6시간 뒤는 하루 주기의 정점 근처여서 23시 값과의 차이가 가장 크다. 즉 이 표의 horizon별 오차는 "23시에 발행한 예측"에 대한 것이다. 예측을 하루 중 여러 시각에 발행한다면 원점의 시각도 다양하게 두어야 한다. 추세+계절+AR(1) 모델은 모든 horizon에서 가장 낫다. 그 이점은 짧은 horizon에서 크고(1시간 0.63 대 1.16), AR(1) 성분의 기억이 $0.8^h$로 사라지는 긴 horizon에서는 작아진다(24시간 0.84 대 1.07). "평균 RMSE 하나"로는 이 구조가 보이지 않는다.

::quiz{id="persistence-horizon"}

**backtesting 설계의 체크리스트.**

- **원점마다 과거만 쓴다.** 전처리 통계량(정규화 평균, 계절 프로파일), 특징 선택, 하이퍼파라미터 선택도 원점 이전 데이터로만 한다.
- **horizon과 원점의 시각은 실제 사용 조건과 맞춘다.** 내일 계획을 위한 예측이면 24–48시간이 중요하다. 예측을 발행하는 시각, 그리고 그 시점에 어떤 데이터가 도착해 있는지(데이터 지연)도 반영한다.
- **원점 사이의 상관을 기억한다.** 이웃한 원점의 예측 구간은 겹치고 같은 날씨 패턴을 공유하므로, 원점별 오차는 독립이 아니다. 원점 수로 나눈 표준오차는 불확실성을 과소평가한다(12.5장의 반복 CV와 같은 문제).
- **구조 변화 이후의 원점을 따로 본다.** 장비 교체, 정책 변화, 기후 체제 전환 뒤에는 오래된 데이터가 오히려 해가 될 수 있다(실습).

**무작위 분할과 미래 정보.** 시계열에 샘플 단위 무작위 $k$-fold를 쓰면 시험 시점의 앞뒤 시점이 학습에 들어간다. 모델은 미래 쪽 이웃까지 보고 "보간"하므로 오차가 실제 예측보다 작게 나온다. 다만 이 낙관의 크기는 모델에 따라 다르다. Bergmeir 외(2018)는 오차가 무상관인 순수 자기회귀 모델에서는 표준 $k$-fold CV도 타당하다는 것을 보였다. 문제는 오차에 자기상관이 남거나 모델이 유연해 시간적 이웃을 외울 수 있을 때다. 더 직접적인 누수는 **특징 자체가 미래 값을 담는 경우**다. 같은 기온 시계열로 한 시간 뒤 값을 예측하며 비교한다. 세 번째 모델은 시계열을 중심 이동평균(앞뒤 2시간)으로 다듬은 뒤 lag 특징을 만든다.

```python {exec="browser" session="ts"}
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def lag_table(series, target, lags=(1, 2, 24)):                # 행 t: series[t-1], series[t-2], series[t-24]
    t0 = max(lags)
    F = np.column_stack([series[t0 - k: len(series) - k] for k in lags])   # (n - 24, 3)
    return F, target[t0:]

F, y_t = lag_table(temp_c, temp_c)
smooth = np.convolve(temp_c, np.ones(5) / 5, mode="same")      # 중심 이동평균: t-2..t+2를 사용
F_leak, _ = lag_table(smooth, temp_c)                          # smooth[t-1]에 y_t, y_{t+1}이 들어 있다
cvs = {"무작위 5-fold": KFold(5, shuffle=True, random_state=0), "시간 순서 5-fold": TimeSeriesSplit(5)}
for name, model, X_ in [("kNN, lag 특징", KNeighborsRegressor(5), F), ("Ridge, lag 특징", Ridge(1.0), F),
                        ("Ridge, 중심 평활한 lag", Ridge(1.0), F_leak)]:
    res = {k: np.sqrt(-cross_val_score(model, X_, y_t, cv=cv, scoring="neg_mean_squared_error").mean())
           for k, cv in cvs.items()}
    print(f"{name:16s} RMSE [°C]  " + "   ".join(f"{k} {v:.3f}" for k, v in res.items()))
```

```output
kNN, lag 특징      RMSE [°C]  무작위 5-fold 0.687   시간 순서 5-fold 0.749
Ridge, lag 특징    RMSE [°C]  무작위 5-fold 0.608   시간 순서 5-fold 0.609
Ridge, 중심 평활한 lag RMSE [°C]  무작위 5-fold 0.406   시간 순서 5-fold 0.410
```

선형 AR 형태의 Ridge 모델은 무작위 분할과 시간 순서 분할에서 거의 같은 오차(0.608, 0.609)를 낸다. Bergmeir 외(2018)의 결과와 맞는다. 유연한 $k$-NN은 무작위 분할에서 0.687, 시간 순서 분할에서 0.749로 무작위 분할이 낙관적이다. 추세 때문에 뒤쪽 시험 구간의 값이 학습 구간보다 높아지는데, 무작위 분할에서는 같은 기간의 이웃 시점이 학습에 있어 그 수준을 "미리" 알고 있다. 가장 극적인 것은 중심 평활한 lag다. 시간 순서를 지킨 분할에서도 오차가 0.410으로 정직한 모델(0.609)보다 33% 작다. 다듬은 값 $\tilde{y}_{t-1}$이 $y_t$와 $y_{t+1}$을 평균에 포함하기 때문이다. 이 모델은 실제 예측 시점에는 만들 수 없는 특징으로 평가된 것이고, 배포하면 성능이 사라진다. **분할을 시간 순서로 해도 특징 생성 단계의 누수는 막지 못한다.**

## 실습

시계열 예측 문제를 rolling-origin backtesting으로 평가하고 horizon별 오차를 분석한다. 코드는 `labs/classical-ml/time-series-analysis/backtest.py`다. 120일 시간당 기온에 80일째 +2 °C의 수준 변화(센서 이전이나 체제 변화 같은 구조 변화)를 넣었다. 60일째부터 하루 간격의 원점 59개에서 48시간 앞까지 예측하고, 확장 창과 최근 14일 이동 창을 비교한다. 괄호 안은 원점별 제곱오차로 근사한 RMSE의 표준오차인데, 이웃 원점의 오차가 상관되어 있으므로 실제 불확실성보다 작다. 집필 시점의 실행 결과는 다음과 같다.

```output
원점 59개 (60일째부터 하루 간격), horizon 1–48시간, 구조 변화 80일째 +2 °C

1. horizon별 RMSE [°C]  (괄호: 원점별 제곱오차 평균의 표준오차로 근사한 RMSE의 ±)
                                h=1        h=6        h=12       h=24       h=48       
persistence                     0.87(±0.07) 3.80(±0.16) 2.01(±0.16) 1.10(±0.10) 1.19(±0.10)
seasonal naive (전날 같은 시각)       0.95(±0.08) 1.01(±0.09) 1.02(±0.10) 1.10(±0.10) 1.19(±0.10)
추세+계절+AR(1), 확장 창               0.52(±0.05) 0.87(±0.10) 1.07(±0.11) 1.17(±0.11) 1.21(±0.11)
추세+계절+AR(1), 최근 14일 창           0.52(±0.05) 0.85(±0.09) 0.89(±0.11) 0.93(±0.08) 1.00(±0.09)

2. 구조 변화 전후의 RMSE (h = 1–24 평균) [°C]
persistence                     변화 전 2.59   변화 후 2.49
seasonal naive (전날 같은 시각)       변화 전 1.13   변화 후 1.17
추세+계절+AR(1), 확장 창               변화 전 0.91   변화 후 1.09
추세+계절+AR(1), 최근 14일 창           변화 전 0.99   변화 후 0.90

3. 이웃 원점의 오차 상관 (확장 창 모델, h = 24)
   하루 간격 원점 사이 상관 0.44
   (h = 1에서는 0.27)
```

결과를 읽는 법은 다음과 같다.

- **짧은 horizon에서는 AR 성분이, 긴 horizon에서는 수준 추정이 성능을 정한다.** 두 AR 모델은 1시간 뒤 0.52로 같지만, 24시간 뒤에는 확장 창(1.17)이 seasonal naive(1.10)보다도 나쁘고 14일 창(0.93)이 가장 낫다.
- **차이의 원인은 구조 변화다.** 확장 창 모델은 변화 전 0.91에서 변화 후 1.09로 나빠졌다. 선형 추세 하나로 +2 °C 계단을 맞추려다 추세선이 기울어지고, 오래된 데이터가 새 수준을 희석한다. 최근 14일만 쓰는 모델은 변화 후 오히려 나아졌다(0.90). 데이터가 많을수록 좋다는 가정은 과정이 변하지 않을 때만 성립한다. 반대로 과정이 안정적이면 짧은 창은 추정 분산 때문에 손해를 본다(변화 전 0.99 대 0.91). 창 길이도 backtesting으로 고르는 하이퍼파라미터다.
- **persistence와 seasonal naive는 24시간과 48시간에서 같은 예측이다.** 마지막 관측이 정확히 하루(이틀) 전 같은 시각이기 때문이다. baseline의 정의가 horizon에 따라 겹칠 수 있으므로 표를 읽을 때 확인한다.
- **원점별 오차는 독립이 아니다.** 하루 간격 원점의 24시간 오차 상관이 0.44다. 괄호 안의 표준오차는 이를 무시했으므로 실제 불확실성보다 작다. 모델 사이의 차이를 주장하려면 같은 원점에서의 짝지은 차이(12.5장)를 보고, 블록 단위로 재표본화한다.

이어서 두 연습문제를 푼다. 첫째는 AR($p$)를 최소제곱으로 적합하고 재귀 예측을 만들고, 둘째는 rolling-origin 평가와 horizon별 RMSE를 구현한다.

::exercise{id="ar-forecast"}

::exercise{id="rolling-origin"}

## 실패 사례

- **시계열에 무작위 분할을 쓰는 것.** 유연한 모델은 시험 시점의 앞뒤 이웃을 보고 보간해 오차를 과소평가한다. 이 장의 예에서 $k$-NN은 무작위 분할에서 8% 낙관적이었고, 추세나 구조 변화가 클수록 차이가 커진다. 시간 순서를 지키는 rolling-origin 평가를 기본으로 한다.
- **미래 값을 사용한 특징을 만드는 것.** 중심 이동평균, 전체 기간으로 계산한 정규화 통계량, 평활한 상태 추정값, 나중에 수정된 데이터(재분석 자료)가 대표적이다. 중심 평활한 lag는 시간 순서 분할에서도 오차를 33% 줄여 보였다. 특징마다 "예측 시점에 이 값을 알 수 있는가"를 묻는다.
- **평균 오차 하나만 보고하는 것.** horizon마다 모델의 순위가 바뀌었다. horizon별 오차와 baseline(persistence, seasonal naive)을 함께 보고한다.
- **구조 변화를 무시하고 모든 과거 데이터를 쓰는 것.** 확장 창 모델은 변화 뒤에 baseline보다 나빠졌다. 구조 변화 전후의 오차를 나누어 보고, 창 길이를 backtesting으로 정한다.
- **원점별 오차를 독립 표본으로 보는 것.** 겹치는 예측 구간의 오차는 상관되어 있다. 원점 수로 나눈 표준오차는 낙관적이다.

## 통과 기준

- 자신의 시계열 문제에서 예측 시점에 사용할 수 있는 데이터, 예측 horizon, 원점의 간격과 학습 창을 정한 rolling-origin 평가 프로토콜을 쓰고, 각 선택을 연구 목적으로 정당화한다.
- 같은 프로토콜로 persistence·seasonal naive baseline과 자신의 모델을 horizon별 오차로 비교하고, 특징 생성 단계에 미래 정보가 없음을 점검한 결과를 제시한다.
- 연습문제 두 개(`ar-forecast`, `rolling-origin`)의 테스트를 통과한다.

## 참고 문헌

- Kalman (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. — Kalman filter
- Rauch, Tung, Striebel (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445–1450. — RTS smoother
- Tashman (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437–450. — 고정 원점과 rolling origin, 표본 외 평가의 설계
- Bergmeir, Hyndman, Koo (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*, 120, 70–83. — 순수 자기회귀 모델에서 $k$-fold CV가 타당한 조건
- Hyndman, Athanasopoulos (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. — 분해, 지수평활, ARIMA, 시계열 교차검증을 다루는 공개 교과서
