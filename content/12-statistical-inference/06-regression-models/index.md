---
id: statistical-inference/regression-models
title: "회귀모형의 통계적 관점"
title_en: "A Statistical View of Regression"
tier: advanced
status: draft
prerequisites: [statistical-inference/likelihood]
---

# 회귀모형의 통계적 관점

ML에서 회귀는 주로 예측 도구다. 시험 세트의 RMSE가 낮으면 좋은 모델이다. 과학 연구에서는 회귀에 다른 질문을 던지는 경우가 많다. "온도가 10 °C 오르면 열화 속도가 몇 배가 되는가", "이 첨가제의 효과는 측정 잡음과 구별되는가"처럼 **계수 자체**와 그 불확실성이 결과다. 이때는 예측 성능이 좋아도 계수의 표준오차가 틀리면 결론이 틀린다.

이 장은 회귀를 12.2장의 우도 관점에서 다시 본다. 선형모형의 가정과 계수의 불확실성, 출력의 종류에 맞춰 우도를 바꾸는 일반화 선형모형(GLM), 가정이 깨졌는지 확인하는 잔차 진단, 그리고 같은 셀·같은 장비에서 반복 측정한 데이터를 다루는 혼합효과 모형을 다룬다. 마지막 주제는 과학 데이터에서 특히 흔한 실수, 즉 반복 측정을 독립 샘플로 세어 불확실성을 과소평가하는 문제와 직결된다.

## 학습 목표

- 선형모형과 GLM을 통계적으로 해석한다
- 잔차 진단을 수행한다

## 선형모형

**모형과 가정.** 선형모형은 출력이 입력의 선형결합에 잡음이 더해진 것이라고 본다.

$$
y_i = \mathbf{x}_i^\top\boldsymbol\beta + \varepsilon_i, \qquad i = 1, \dots, n.
$$

$\mathbf{x}_i \in \mathbb{R}^p$는 절편을 위한 1을 포함한 입력, $\boldsymbol\beta$는 계수, $\varepsilon_i$는 오차다. 표준적인 가정은 네 가지이고, 무엇이 깨지면 무엇이 틀리는지가 다르다.

| 가정 | 내용 | 깨지면 |
|---|---|---|
| **선형성** | $\mathbb{E}[y \mid \mathbf{x}] = \mathbf{x}^\top\boldsymbol\beta$, 즉 $\mathbb{E}[\varepsilon \mid \mathbf{x}] = 0$ | 계수가 편향되고 해석이 무너진다 |
| **독립성** | $\varepsilon_i$들이 서로 독립 | 계수는 불편이지만 표준오차가 틀린다(대개 과소평가) |
| **등분산** | $\operatorname{Var}(\varepsilon_i) = \sigma^2$로 일정 | 계수는 불편이지만 표준오차가 틀리고 비효율적이다 |
| **정규성** | $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ | 작은 표본에서 $t$ 구간이 부정확하다. 큰 표본에서는 영향이 작다 |

위에서 아래로 갈수록 덜 치명적이다. 잔차의 히스토그램(정규성)만 확인하고 넘어가는 것은 가장 덜 중요한 가정만 확인하는 셈이다. 네 가정이 모두 성립하면 최소제곱은 고정 분산 가우시안 우도의 MLE와 같다(12.2장).

**계수와 불확실성.** 최소제곱 해와 그 공분산은

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}, \qquad
\operatorname{Cov}(\hat{\boldsymbol\beta}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}, \qquad
\hat\sigma^2 = \frac{\lVert \mathbf{y} - \mathbf{X}\hat{\boldsymbol\beta} \rVert^2}{n - p}
$$

이다. $\mathbf{X}$는 $\mathbf{x}_i^\top$를 행으로 쌓은 $n \times p$ 행렬이다. 공분산 식은 $\hat{\boldsymbol\beta} = \boldsymbol\beta + (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol\varepsilon$에 $\operatorname{Cov}(\boldsymbol\varepsilon) = \sigma^2\mathbf{I}$를 넣어 얻는다. 이 단계에 독립성과 등분산이 쓰인다. $\hat\sigma^2$을 $n-p$로 나누는 이유는 12.2장 실습에서 본 MLE($n$으로 나눔)의 bias를 없애기 위해서다. 계수 $j$의 표준오차는 공분산 행렬 대각 성분의 제곱근이고, $(\hat\beta_j - \beta_j)/\mathrm{SE}_j$는 정규 오차 아래에서 자유도 $n-p$인 $t$ 분포를 따른다.

배터리 셀 하나의 용량을 0–200 사이클에서 10 사이클마다 측정한 데이터로 열화 속도를 추정한다. 같은 실험을 5,000번 반복해, 공식으로 계산한 표준오차가 기울기의 실제 퍼짐과 맞는지도 확인한다.

```python {exec="browser" session="reg"}
import numpy as np

def ols(X, y):                                   # 계수, 표준오차, 잔차 분산
    XtX_inv = np.linalg.inv(X.T @ X)             # (p, p)
    beta = XtX_inv @ X.T @ y                     # (p,)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])   # σ̂² = RSS / (n - p)
    return beta, np.sqrt(s2 * np.diag(XtX_inv)), s2

rng = np.random.default_rng(0)
cycle = np.arange(0, 201, 10.0)                  # 사이클 번호, (n,) n = 21
X = np.column_stack([np.ones_like(cycle), cycle])                 # (n, 2)
cap_ah = 2.50 - 0.0012 * cycle + rng.normal(0, 0.02, cycle.size)  # 용량 [Ah]
beta, se, _ = ols(X, cap_ah)
print(f"절편 {beta[0]:.4f} ± {se[0]:.4f} Ah,  기울기 {beta[1] * 1000:.3f} ± {se[1] * 1000:.3f} mAh/cycle")

slopes, ses = [], []
for _ in range(5000):                            # 같은 실험을 반복해 SE 공식을 확인한다
    b, s, _ = ols(X, 2.50 - 0.0012 * cycle + rng.normal(0, 0.02, cycle.size))
    slopes.append(b[1]); ses.append(s[1])
slopes, ses = np.array(slopes), np.array(ses)
cover = np.mean(np.abs(slopes + 0.0012) <= 2.093 * ses)          # t_{0.975, 19} = 2.093
print(f"기울기의 실제 sd {slopes.std() * 1000:.4f},  보고된 SE 평균 {ses.mean() * 1000:.4f} mAh/cycle,  95% 구간 포함률 {cover:.3f}")
```

```output
절편 2.5010 ± 0.0072 Ah,  기울기 -1.247 ± 0.062 mAh/cycle
기울기의 실제 sd 0.0715,  보고된 SE 평균 0.0712 mAh/cycle,  95% 구간 포함률 0.955
```

가정이 모두 성립하는 시뮬레이션에서는 공식이 정확하다. 보고된 표준오차의 평균이 기울기의 실제 표준편차와 맞고, 95% 구간이 참값을 95% 덮는다. 이 장의 나머지는 가정이 깨질 때 이 일치가 어떻게 무너지는지에 대한 것이다.

**계수의 해석.** $\beta_j$는 "다른 입력을 고정하고 $x_j$만 1단위 바꿀 때 $\mathbb{E}[y]$의 변화"다. 이 해석에는 두 가지 함정이 있다. 첫째, 입력끼리 강하게 상관되어 있으면(다중공선성) 하나만 바꾸는 상황이 데이터에 거의 없어 계수의 표준오차가 커진다. $(\mathbf{X}^\top\mathbf{X})^{-1}$이 커지는 것이다. 둘째, 계수는 연관성이지 인과가 아니다. 관측 데이터에서 계수를 인과 효과로 읽으려면 추가 가정이 필요하다(16.8장).

::quiz{id="ols-assumptions"}

## 일반화 선형모형

선형모형은 출력이 연속이고 대칭인 잡음을 가진다고 본다. 그러나 과학 데이터의 출력은 자주 그렇지 않다. 불량 여부(0/1), 검출기 계수·결함 수(0 이상의 정수), 수명·농도(양수, 오른쪽으로 치우침)가 그렇다. 이런 출력에 최소제곱을 쓰면 예측이 범위를 벗어나고(음의 계수, 1보다 큰 확률), 분산 구조가 틀린다.

**일반화 선형모형**(generalized linear model, GLM)은 세 부분으로 이 문제를 푼다(Nelder & Wedderburn, 1972).

1. **확률 분포**: $y$가 지수족 분포(12.2장)를 따른다. 정규, Bernoulli, Poisson, Gamma 등.
2. **선형 예측자**: $\eta = \mathbf{x}^\top\boldsymbol\beta$.
3. **link function** $g$: 평균과 선형 예측자를 잇는다. $g(\mathbb{E}[y \mid \mathbf{x}]) = \eta$.

| 출력 | 분포 | 대표 link | 모형 | 계수의 해석 |
|---|---|---|---|---|
| 연속 | 정규 | identity | 선형회귀 | $x$ 1단위당 평균의 변화량 |
| 0/1 | Bernoulli | logit $\log\frac{\pi}{1-\pi}$ | 로지스틱 회귀 | $x$ 1단위당 log-odds의 변화, odds는 $e^{\beta}$배 |
| 계수 | Poisson | log | Poisson 회귀 | $x$ 1단위당 발생률이 $e^{\beta}$배 |

link function 덕분에 $\eta$는 실수 전체를 오가도 평균은 올바른 범위(확률은 0–1, 발생률은 양수)에 머문다. 대가는 해석이다. log link의 계수는 더하는 양이 아니라 **곱하는 비율의 로그**이고, logit link의 계수는 확률이 아니라 odds를 곱한다.

**적합.** GLM의 MLE는 닫힌 해가 없지만, 지수족의 NLL이 볼록하므로 Newton 방법으로 안정적으로 풀린다. GLM에서 Newton 방법은 매 단계 가중최소제곱을 푸는 꼴이 되어 **IRLS**(iteratively reweighted least squares)라 한다. Poisson 회귀(log link)에서는 현재 평균 $\mu_i = e^{\eta_i}$로 작업 반응변수 $z_i = \eta_i + (y_i - \mu_i)/\mu_i$와 가중치 $w_i = \mu_i$를 만들고, $\boldsymbol\beta \leftarrow (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{z}$를 반복한다. 수렴하면 $(\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}$이 Fisher 정보의 역행렬, 즉 계수의 근사 공분산이다(12.2장의 점근 정규성).

온도별로 시편 8개씩, 시편마다 미세 균열 수를 센 데이터에 Poisson 회귀를 적합한다. 시뮬레이션의 참값은 $\beta_1 = 0.35$, 즉 10 °C마다 발생률 $e^{0.35} \approx 1.42$배다.

```python {exec="browser" session="reg"}
def poisson_glm(X, y, iters=25):                 # log link: log E[y] = X β
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(X @ beta)                    # (n,)
        z = X @ beta + (y - mu) / mu             # 작업 반응변수
        W = mu                                   # 가중치 = 1 / (Var(y) · (dη/dμ)²) = μ
        beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
    cov = np.linalg.inv(X.T @ (np.exp(X @ beta)[:, None] * X))   # Fisher 정보의 역
    return beta, np.sqrt(np.diag(cov))

temp_c = np.repeat([20.0, 30.0, 40.0, 50.0, 60.0], 8)              # 온도 [°C], (n,) n = 40
Xp = np.column_stack([np.ones_like(temp_c), (temp_c - 40) / 10])   # 10 °C 단위로 중심화
counts = rng.poisson(np.exp(1.5 + 0.35 * Xp[:, 1]))                # 시편당 미세 균열 수
b, s = poisson_glm(Xp, counts)
print(f"β1 = {b[1]:.3f} ± {s[1]:.3f}   → 10 °C마다 균열 발생률 × {np.exp(b[1]):.2f}"
      f"  (95% 구간 × {np.exp(b[1] - 1.96 * s[1]):.2f} ~ {np.exp(b[1] + 1.96 * s[1]):.2f})")
print(f"온도별 평균 균열 수: {[float(counts[temp_c == t].mean()) for t in (20, 30, 40, 50, 60)]}")
```

```output
β1 = 0.317 ± 0.051   → 10 °C마다 균열 발생률 × 1.37  (95% 구간 × 1.24 ~ 1.52)
온도별 평균 균열 수: [2.375, 3.125, 6.5, 6.25, 9.125]
```

결과는 "10 °C마다 약 1.37배, 95% 구간 1.24–1.52배"로 읽는다. 구간은 계수 척도에서 대칭으로 만든 뒤 지수를 씌웠으므로 비율 척도에서는 비대칭이다. 참값 1.42는 구간 안에 있다.

::quiz{id="poisson-rate-ratio"}

**로지스틱 회귀.** 같은 틀에서 Bernoulli 분포와 logit link를 쓰면 로지스틱 회귀다. 계수 $\beta$는 log-odds의 변화이므로 odds를 $e^{\beta}$배 한다. 확률의 변화량은 기준 확률에 따라 달라서, 확률이 0.5 근처일 때 가장 크고(약 $\beta/4$) 0이나 1 근처에서는 작다. 분류 모델로서의 로지스틱 회귀는 16.2장에서 다룬다.

::quiz{id="logistic-coefficient"}

:::note[분야 예시]
역학(epidemiology)에서는 노출과 질병 발생의 관계를 로지스틱 회귀로 추정하고 결과를 odds ratio로 보고하는 것이 표준이다. 질병이 드물면 odds ratio가 상대위험(risk ratio)에 가깝지만, 흔한 질병에서는 odds ratio가 상대위험보다 1에서 더 멀리 떨어져 효과를 과장해 보이게 할 수 있다. 같은 계수를 어떤 척도로 보고하느냐가 독자의 해석을 바꾼다.
:::

**과대산포.** Poisson 분포는 분산이 평균과 같다고 가정한다. 실제 계수 데이터는 측정 단위 사이의 이질성 때문에 분산이 평균보다 큰 경우(과대산포, overdispersion)가 흔하다. 이때 Poisson 회귀의 표준오차는 과소평가된다. 잔차 deviance를 자유도와 비교해 확인하고, 음이항(negative binomial) 회귀나 분산을 보정한 quasi-Poisson으로 대응한다.

## 잔차 진단

가정은 데이터에서 확인해야 한다. 오차 $\varepsilon_i$는 볼 수 없지만 그 추정인 **잔차** $e_i = y_i - \hat{y}_i$는 볼 수 있다. 가정이 맞으면 잔차에는 아무 구조도 없어야 한다. 구조가 보이면 그것이 어떤 가정이 깨졌는지 말해 준다.

| 그림 | 보이는 패턴 | 깨진 가정 | 대응 |
|---|---|---|---|
| 잔차 vs 적합값(또는 입력) | 곡선(U자, 역U자) | 선형성 | 항 추가(2차·상호작용), 변환, 비선형 모델 |
| 잔차 vs 적합값 | 깔때기(퍼짐이 커짐) | 등분산 | 가중최소제곱, 분산이 평균과 함께 커지는 GLM, 로그 변환, robust 표준오차 |
| 잔차 vs 시간(측정 순서) | 이웃한 잔차가 닮음 | 독립성 | 시계열 오차 모형, 블록 단위 불확실성(16.7장) |
| 정규 Q-Q plot | 꼬리가 직선에서 벗어남 | 정규성 | 큰 표본이면 대개 무시 가능, 작은 표본이면 robust 방법 |
| leverage vs 잔차 | 혼자 떨어진 영향점 | — (가정이 아니라 데이터) | 원인 확인, 넣고 뺀 결과를 함께 보고 |

**곡률과 이분산.** 열화가 점점 빨라지는(참 모형이 2차인) 용량 데이터에 직선을 맞추고, 잡음도 사이클과 함께 커지게 한다. 그림 대신 적합값 순서로 5구간을 나눠 잔차의 평균과 표준편차를 숫자로 본다.

```python {exec="browser" session="reg"}
cyc = np.arange(0, 501, 5.0)                                        # (n,) n = 101
fade = 2.5 - 0.0005 * cyc - 1.5e-6 * cyc**2                         # 가속 열화 (참 모형은 2차)
y3 = fade + rng.normal(0, 1, cyc.size) * (0.005 + 0.00006 * cyc)   # 잡음도 사이클과 함께 커진다
X3 = np.column_stack([np.ones_like(cyc), cyc])
b3, _, _ = ols(X3, y3)
fitted, r = X3 @ b3, y3 - X3 @ b3
bins = np.array_split(np.argsort(fitted), 5)                        # 적합값 순서로 5구간
print("적합값 구간      잔차 평균    잔차 sd")
for idx in bins:
    print(f"{fitted[idx].min():.3f}–{fitted[idx].max():.3f}    {r[idx].mean():+.4f}     {r[idx].std():.4f}")
```

```output
적합값 구간      잔차 평균    잔차 sd
1.943–2.066    -0.0315     0.0350
2.072–2.189    +0.0194     0.0197
2.196–2.313    +0.0252     0.0209
2.319–2.436    +0.0182     0.0180
2.442–2.560    -0.0297     0.0207
```

잔차 평균이 양 끝에서 음수, 가운데에서 양수인 역U자 패턴은 직선이 곡선을 따라가지 못한다는 신호(선형성 위반)다. 잔차 표준편차는 적합값이 가장 낮은 구간(가장 늦은 사이클)에서 다른 구간의 약 두 배다(이분산). 두 패턴이 섞여 있으면 곡률이 퍼짐처럼 보이기도 하므로, 먼저 평균 구조를 고친 뒤 다시 퍼짐을 본다.

::quiz{id="residual-funnel"}

**자기상관과 영향점.** 시간 순서로 측정한 데이터에서는 이웃한 측정의 오차가 닮는 경우가 흔하다(온도 변동, 장비 drift). 잔차의 lag-1 자기상관과 **Durbin–Watson** 통계량 $d = \sum_t (e_t - e_{t-1})^2 / \sum_t e_t^2 \approx 2(1 - \hat\rho)$로 확인한다. 독립이면 $d \approx 2$다.

**영향점**(influential point)은 하나만 빼도 적합 결과가 크게 바뀌는 관측이다. 두 요소가 곱해져 생긴다. 하나는 **leverage** $h_{ii}$, 즉 hat 행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$의 대각 성분으로, 입력이 다른 점들에서 멀수록 크다($\sum_i h_{ii} = p$). 다른 하나는 잔차의 크기다. 둘을 합친 지표가 **Cook's distance**다.

$$
D_i = \frac{e_i^2}{p\,\hat\sigma^2}\cdot\frac{h_{ii}}{(1 - h_{ii})^2}.
$$

관례적으로 $D_i$가 1 근처 이상이거나 다른 점들보다 두드러지게 크면 들여다본다.

```python {exec="browser" session="reg"}
e = np.zeros(cyc.size)
for t in range(1, cyc.size):                     # AR(1) 잡음: 이웃한 측정의 오차가 닮는다
    e[t] = 0.7 * e[t - 1] + rng.normal(0, 0.01)
r4 = (2.5 - 0.001 * cyc + e) - X3 @ ols(X3, 2.5 - 0.001 * cyc + e)[0]
dw = np.sum(np.diff(r4) ** 2) / np.sum(r4**2)
print(f"잔차의 lag-1 자기상관 {np.corrcoef(r4[:-1], r4[1:])[0, 1]:.2f},  Durbin–Watson {dw:.2f}  (독립이면 약 2)")

y5 = 2.5 - 0.001 * cyc + rng.normal(0, 0.01, cyc.size)
y5[-1] -= 0.15                                   # 마지막 사이클(가장자리 x)에 측정 오류 하나
H = X3 @ np.linalg.inv(X3.T @ X3) @ X3.T         # hat 행렬, (n, n)
h, (b5, _, s2) = np.diag(H), ols(X3, y5)
r5 = y5 - X3 @ b5
cook = r5**2 / (2 * s2) * h / (1 - h) ** 2       # Cook's distance (p = 2)
i = np.argmax(cook)
print(f"최대 Cook's distance: 사이클 {cyc[i]:.0f}, D = {cook[i]:.2f}, leverage h = {h[i]:.3f} (평균 {h.mean():.3f})")
print(f"기울기: 전체 {b5[1] * 1000:.3f}, 이 점을 빼면 {ols(X3[:-1], y5[:-1])[0][1] * 1000:.3f} mAh/cycle (참값 -1.000)")
```

```output
잔차의 lag-1 자기상관 0.61,  Durbin–Watson 0.77  (독립이면 약 2)
최대 Cook's distance: 사이클 500, D = 1.30, leverage h = 0.039 (평균 0.020)
기울기: 전체 -1.017, 이 점을 빼면 -1.000 mAh/cycle (참값 -1.000)
```

첫 줄의 Durbin–Watson 0.77은 강한 양의 자기상관을 뜻한다. 이대로 표준오차를 계산하면 이웃한 측정이 사실상 같은 정보를 반복하는데도 독립 정보로 세게 된다. 둘째 줄에서 마지막 사이클의 측정 오류 하나는 leverage가 평균의 두 배인 가장자리에 있어 Cook's distance 1.30으로 두드러진다. 이 점 하나가 기울기를 1.7% 바꿨다. 같은 크기의 오류가 데이터 가운데에 있었다면 영향은 훨씬 작았을 것이다.

::quiz{id="durbin-watson"}

영향점을 찾은 뒤에 할 일은 삭제가 아니라 **확인**이다. 측정 기록에서 오류의 독립적인 근거(장비 경고, 기록 실수)를 찾으면 뺀다. 근거가 없으면 그 점이 모형의 결함을 드러내는 것일 수 있다. 예를 들어 마지막 구간에서 열화가 실제로 가속된다면 그 점은 오류가 아니라 정보다. 어느 쪽이든 넣은 결과와 뺀 결과를 함께 보고한다.

::quiz{id="high-leverage-point"}

## 혼합효과 모형

과학 데이터는 자주 **그룹 구조**를 가진다. 배터리 셀마다 여러 사이클을 측정하고, 환자마다 여러 번 검사하고, 장비마다 여러 시료를 측정한다. 같은 그룹의 측정은 그룹 고유의 특성(제조 편차, 체질, 장비 보정 상태)을 공유하므로 서로 닮는다. 이것을 무시하고 모든 측정을 독립 샘플로 세면 불확실성을 크게 과소평가한다. 생태학에서는 이 문제를 **pseudoreplication**이라 부르며, 처리가 반복되지 않았거나 반복이 통계적으로 독립이 아닌 데이터로 처리 효과를 검정하는 오류로 정의했다(Hurlbert, 1984).

**random intercept 모형.** 가장 단순한 혼합효과 모형은 그룹마다 절편이 무작위로 다르다고 본다.

$$
y_{ij} = \mathbf{x}_{ij}^\top\boldsymbol\beta + u_i + \varepsilon_{ij}, \qquad u_i \sim \mathcal{N}(0, \tau^2), \quad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2).
$$

$i$는 그룹(셀), $j$는 그룹 안의 측정이다. $\boldsymbol\beta$는 모든 그룹에 공통인 **고정 효과**, $u_i$는 그룹마다 다른 **무작위 효과**다. 같은 그룹의 두 측정은 $u_i$를 공유하므로 상관계수가

$$
\rho = \frac{\tau^2}{\tau^2 + \sigma^2}
$$

이다. 이를 그룹 내 상관(intraclass correlation, ICC)이라 한다. 그룹마다 측정이 $m$개일 때, 그룹 단위 효과를 추정하는 분산은 독립을 가정한 값의 약 $1 + (m-1)\rho$배다. 이를 **설계 효과**(design effect)라 한다. $\rho$가 0.5이고 $m = 20$이면 설계 효과가 10.5로, 측정 200개가 독립 측정 약 19개의 정보밖에 담지 않는다.

::quiz{id="design-effect"}

셀 12개(상온 6개, 고온 6개)에서 셀마다 15번 측정해 고온의 효과를 추정한다. 고온 여부는 셀 단위 공변량이다. 측정 180개를 독립으로 본 OLS와, 셀마다 평균을 내 12개로 회귀한 결과의 95% 구간 포함률을 2,000번 반복으로 비교한다.

```python {exec="browser" session="reg"}
G, m = 12, 15                                    # 셀 12개 × 셀마다 측정 15번
hot = np.repeat(np.r_[np.zeros(6), np.ones(6)], m)                  # 고온 셀 여부 (셀 단위 공변량)
Xg = np.column_stack([np.ones(G * m), hot])
naive_cover, mean_cover = [], []
for _ in range(2000):
    u = np.repeat(rng.normal(0, 0.03, G), m)                        # 셀 고유 편차 (공유됨)
    yg = 2.45 - 0.05 * hot + u + rng.normal(0, 0.02, G * m)
    b, s, _ = ols(Xg, yg)                                            # 측정 180개를 독립으로 취급
    naive_cover.append(abs(b[1] + 0.05) <= 1.96 * s[1])
    bm, sm, _ = ols(Xg[::m], yg.reshape(G, m).mean(axis=1))          # 셀 평균 12개로 회귀
    mean_cover.append(abs(bm[1] + 0.05) <= 2.228 * sm[1])            # t_{0.975, 10}
print(f"고온 효과의 95% 구간 포함률:  측정 단위 OLS {np.mean(naive_cover):.3f}   셀 평균 회귀 {np.mean(mean_cover):.3f}")
print(f"설계 효과 1 + (m - 1)ρ = {1 + (m - 1) * 0.03**2 / (0.03**2 + 0.02**2):.1f}  (ρ = 셀 내 상관)")
```

```output
고온 효과의 95% 구간 포함률:  측정 단위 OLS 0.413   셀 평균 회귀 0.949
설계 효과 1 + (m - 1)ρ = 10.7  (ρ = 셀 내 상관)
```

측정 단위 OLS의 "95% 구간"은 참값을 41%만 덮는다. 설계 효과가 10.7이므로 표준오차를 약 $\sqrt{10.7} \approx 3.3$배 과소평가한 것이다. 셀 평균으로 회귀하면 독립 단위가 맞으므로 0.949로 돌아온다.

셀 평균 회귀는 셀 단위 공변량에는 맞지만, 셀 **안에서** 변하는 공변량(사이클 번호)의 효과는 추정할 수 없고 셀마다 측정 수가 다르면 가중치 문제가 생긴다. 혼합효과 모형은 두 종류의 공변량을 한 모형에서 다룬다. $\tau^2$과 $\sigma^2$을 데이터에서 추정하고(보통 REML), 그 공분산 구조로 GLS를 풀어 계수와 표준오차를 얻는다. 셀 단위 효과는 셀 수만큼의 정보로, 셀 안 효과는 셀 고유 편차가 상쇄된 셀 안 비교로 추정된다(실습).

**partial pooling.** 혼합효과 모형에서 그룹별 절편의 추정값은 그 그룹의 데이터만 쓴 값(no pooling)과 전체 평균(complete pooling) 사이로 끌려간다. 측정이 적은 그룹일수록 전체 평균 쪽으로 더 많이 끌린다. 이것이 12.1장의 bias–variance trade-off를 그룹 구조에 적용한 수축(shrinkage)이고, 13.3장의 계층 모델이 같은 구조를 베이지안으로 다룬다.

:::note[분야 예시]
신경과학에서는 동물 몇 마리에서 뉴런 수백 개를 기록하고 뉴런을 독립 샘플로 세어 검정하는 일이 흔했다. 같은 동물의 뉴런은 동물 고유의 상태를 공유하므로, 동물 간 변동이 크면 이런 분석은 위양성을 크게 늘린다. 동물을 무작위 효과로 둔 혼합효과 모형이나 동물 단위 요약으로 분석하면 결론이 달라질 수 있다. 셀·장비·환자·관측소처럼 이름만 다를 뿐 구조는 같다.
:::

:::tip[연구 팁]
혼합효과 모형은 직접 구현하기보다 검증된 패키지를 쓴다. 집필 시점 예시로 Python에서는 statsmodels의 `mixedlm`(formula 인터페이스), R에서는 lme4의 `lmer`가 널리 쓰인다. 쓰기 전에 사용하는 버전의 문서에서 REML·ML 선택 인자와 무작위 효과 지정 방법을 확인한다. 그룹이 몇 개뿐(예: 5개 미만)이면 $\tau^2$의 추정 자체가 불안정하므로 결과를 조심해서 해석한다.
:::

## 실습

반복 측정 데이터에 일반 회귀와 혼합효과 모형을 적합해 계수와 불확실성을 비교한다. 코드는 `labs/statistical-inference/regression-models/mixed_effects.py`다. 배터리 셀 12개(상온 6개, 고온 6개)에서 0–140 사이클 동안 10 사이클마다 용량을 잰다.

$$
y_{ij} = \beta_0 + \beta_1\,\mathrm{hot}_i + \beta_2\,\mathrm{cycle}_{ij} + u_i + \varepsilon_{ij}, \qquad \tau = 0.03,\ \sigma = 0.01\ \text{Ah}.
$$

$\beta_1$(고온 효과)은 셀 단위 공변량, $\beta_2$(사이클당 변화)는 셀 안에서 변하는 공변량의 계수다. 세 방법을 비교한다. (1) 측정 180개를 독립으로 본 OLS, (2) 셀 평균 12개로 회귀, (3) random intercept 모형. 혼합효과 모형은 NumPy만으로 구현했다. $\lambda = \tau^2/\sigma^2$을 격자에서 REML로 고르고, 그 공분산 구조로 GLS를 푼다. 같은 데이터에서 statsmodels의 `mixedlm`(REML)과 계수가 일치하고 표준오차가 0.3% 이내로 맞는 것을 확인했다. 집필 시점의 실행 결과는 다음과 같다.

```output
참값: 고온 효과 -0.0500 Ah, 사이클당 -0.400 mAh,  τ = 0.03, σ = 0.01 Ah
혼합효과 모형이 추정한 τ = 0.0232, σ = 0.0098 Ah

방법             고온 효과 ± SE [Ah]      사이클당 변화 ± SE [mAh/cycle]
OLS (독립 가정)  -0.0562 ± 0.0035       -0.4178 ± 0.0405
셀 평균 회귀     -0.0562 ± 0.0134       (추정 불가)
혼합효과 모형    -0.0562 ± 0.0135       -0.4178 ± 0.0169

95% 구간의 포함률 (새 데이터셋 500개)
방법         고온 효과   사이클당 변화
OLS          0.392       1.000
셀 평균         0.946         -  
혼합효과         0.948       0.934
사이클 계수의 평균 SE: OLS 0.0503, 혼합효과 0.0172 mAh/cycle
```

결과를 읽는 법은 세 가지다.

- **계수 추정값은 세 방법이 같다.** 균형 설계에서는 OLS도 계수 자체는 불편이다. 다른 것은 불확실성이다.
- **셀 단위 효과에서 OLS는 표준오차를 약 4배 과소평가한다.** OLS의 고온 효과 SE(0.0035)는 혼합효과 모형(0.0135)의 4분의 1이고, 95% 구간의 포함률은 0.392에 그친다. 고온 효과의 정보는 셀 12개에서 오는데 OLS는 측정 180개에서 온다고 센 것이다. 혼합효과 모형과 셀 평균 회귀는 0.95 근처다.
- **셀 안 효과에서는 반대로 OLS가 표준오차를 과대평가한다.** 사이클 효과는 같은 셀 안의 측정끼리 비교해 추정되므로 셀 고유 편차 $u_i$가 상쇄된다. OLS는 $u_i$를 잡음으로 취급해 SE를 3배 가까이 크게 잡았고(0.0503 대 0.0172), 포함률이 1.000이다. 보수적이어서 안전해 보이지만, 실제로 있는 효과를 놓치게 만든다. 혼합효과 모형은 두 방향의 오류를 모두 바로잡는다.

이어서 두 연습문제를 푼다. 첫째는 최소제곱 계수의 표준오차를 구현하고, 둘째는 leverage와 Cook's distance를 구현한다.

::exercise{id="ols-standard-errors"}

::exercise{id="cooks-distance"}

## 실패 사례

- **반복 측정을 독립 샘플로 취급해 불확실성을 과소평가하는 것.** 셀 12개에서 측정 180개를 얻었다면, 셀 단위 효과에 대한 독립 정보는 측정 180개가 아니라 셀 12개 수준이다. 실습에서 이 실수는 "95% 구간"의 실제 포함률을 0.39로 떨어뜨렸다. 독립 단위를 먼저 정하고, 혼합효과 모형이나 그룹 단위 요약으로 분석한다.
- **잔차의 정규성만 확인하는 것.** 정규성은 가장 덜 중요한 가정이다. 잔차를 적합값·입력·측정 순서에 대해 그려 곡률·이분산·자기상관을 먼저 확인한다.
- **log link 계수를 더하는 양으로 해석하는 것.** Poisson·로지스틱 회귀의 계수는 비율(발생률·odds)을 곱하는 양의 로그다. 결과는 $e^{\beta}$와 그 구간으로 보고한다.
- **영향점을 근거 없이 지우는 것.** 결과가 나빠지는 점만 지우면 12.4장의 선택 편향이 된다. 지운다면 독립적인 근거를 기록하고, 넣은 결과와 뺀 결과를 함께 보고한다.
- **계수를 인과 효과로 읽는 것.** 관측 데이터의 회귀계수는 다른 입력을 고정한 조건부 연관성이다. 측정하지 않은 교란 변수가 있으면 인과 해석이 틀린다.

## 통과 기준

- 자신의 데이터(또는 이 장의 열화 예제)에 회귀를 적합하고, 잔차 진단으로 가정 위반(곡률·이분산·자기상관·영향점) 중 적어도 하나를 찾아내 근거 숫자와 함께 설명하고 대안 모형을 제시한다.
- 데이터의 독립 단위를 명시하고, 그 단위를 반영한 방법(혼합효과 모형 또는 그룹 단위 요약)으로 계수의 표준오차를 다시 계산해 독립 가정 결과와 비교한다.
- 연습문제 두 개(`ols-standard-errors`, `cooks-distance`)의 테스트를 통과한다.

## 참고 문헌

- Nelder, Wedderburn (1972). Generalized linear models. *Journal of the Royal Statistical Society: Series A*, 135(3), 370–384. — 선형·로지스틱·Poisson 회귀를 지수족과 link function으로 통합하고, 반복 가중최소제곱으로 MLE를 구하는 틀
- Hurlbert (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187–211. — 독립이 아닌 반복으로 처리 효과를 검정하는 오류(pseudoreplication)의 정의
