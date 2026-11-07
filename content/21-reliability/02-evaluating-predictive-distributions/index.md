---
id: reliability/evaluating-predictive-distributions
title: "예측분포와 구간의 평가"
title_en: "Evaluating Predictive Distributions"
tier: core
status: draft
prerequisites: [reliability/uncertainty-quantification]
---

# 예측분포와 구간의 평가

21.1장에서 불확실성을 추정하는 방법을 여럿 보았다. 그런데 어느 방법의 불확실성이 맞는지는 어떻게 알까. 모델이 "90% 구간"이라고 한 것이 실제로 90%를 덮는지, "비 올 확률 30%"라고 한 날들에 실제로 30%쯤 비가 왔는지 확인하지 않으면, 불확실성 추정은 그럴듯한 그림에 그친다. 불확실성은 그 자체로 **검증 대상인 예측**이다.

이 장은 예측분포와 예측 구간을 평가하는 도구를 다룬다. 12.3장에서 적정 점수 규칙(NLL·CRPS)을 소개했으므로, 여기서는 그 원리를 더 분명히 하고 표본 기반 CRPS·Brier score·interval score로 넓힌다. 이어서 점수 하나로는 보이지 않는 두 성질, calibration(보정)과 sharpness(예리함)를 따로 측정하고, 조건별 coverage로 평균에 숨은 실패를 찾고, 마지막으로 보정이 틀렸을 때 고치는 재보정을 다룬다.

## 학습 목표

- 적정 점수 규칙으로 예측분포를 평가한다
- calibration과 sharpness를 측정한다

## 적정 점수 규칙

**왜 필요한가.** 예측분포를 평가하는 지표를 잘못 고르면, 그 지표를 최적화한 모델은 일부러 틀린 분포를 낸다. 예를 들어 이진 사건의 확률 예측을 "예측 확률과 결과의 차이의 절댓값"으로 평가하면, 참 확률이 0.7이어도 1을 보고하는 쪽이 점수가 좋다. 지표가 모델에게 거짓말을 시키는 것이다.

**정의.** 예측분포 $P$와 관측 $y$에 점수 $S(P, y)$를 매긴다(이 장에서는 작을수록 좋은 방향으로 쓴다). 참 분포가 $Q$일 때 기대 점수 $\mathbb{E}_{y \sim Q}[S(P, y)]$가 $P = Q$에서 가장 작으면 적정(proper), $P = Q$에서만 가장 작으면 엄밀하게 적정(strictly proper)하다고 한다(Gneiting & Raftery, 2007). 적정 점수로 평가하면 참이라고 믿는 분포를 그대로 보고하는 것이 최선이다.

**직관.** 참 확률이 0.7인 사건에서 보고값 $q$마다 기대 점수를 계산해 본다. Brier score $(q - y)^2$(Brier, 1950)와 절댓값 점수 $\lvert q - y\rvert$를 비교한다.

```python {exec="browser" session="eval"}
import numpy as np

p_true = 0.7                                   # 사건이 실제로 일어날 확률
print("보고한 q   기대 Brier   기대 |q - y|")
for q in (0.5, 0.7, 0.9, 1.0):
    brier = p_true * (1 - q) ** 2 + (1 - p_true) * q ** 2
    absol = p_true * (1 - q) + (1 - p_true) * q
    print(f"  {q:.1f}      {brier:.3f}        {absol:.3f}")
```

```output
보고한 q   기대 Brier   기대 |q - y|
  0.5      0.250        0.500
  0.7      0.210        0.420
  0.9      0.250        0.340
  1.0      0.300        0.300
```

Brier score의 기대값은 $q = 0.7$에서 가장 작다(0.210). 절댓값 점수는 $q$에 대한 직선이라 $q = 1$에서 가장 작다(0.300). 절댓값 점수로 학습하거나 모델을 고르면, 확률을 0과 1로 밀어붙이는 과신 모델이 이긴다.

::quiz{id="improper-absolute-score"}

**대표 규칙.** 실제로 많이 쓰는 엄밀하게 적정한 규칙은 다음과 같다.

| 규칙 | 대상 | 식 | 성질 |
|---|---|---|---|
| **NLL**(log score) | 밀도·확률 | $-\log p(y)$ | 드문 곳에 떨어진 관측을 무한히 크게 벌한다. 밀도가 필요하다 |
| **CRPS** | 연속 실수 | $\int (F(t) - \mathbb{1}[y \le t])^2 dt$ | $y$와 같은 단위. 표본만으로 계산된다. 점 예측이면 절댓값 오차가 된다 |
| **Brier score** | 이진·범주 | $\sum_k (q_k - \mathbb{1}[y = k])^2$ | 확률의 제곱 오차. 0–1 범위(이진) |
| **interval score** | 중심 $(1-\alpha)$ 구간 $[l, u]$ | $(u-l) + \frac{2}{\alpha}(l-y)\mathbb{1}[y<l] + \frac{2}{\alpha}(y-u)\mathbb{1}[y>u]$ | 구간만 내는 모델용. 최적 구간은 참 분포의 $\alpha/2$, $1-\alpha/2$ 분위수 |

::quiz{id="brier-single"}

**표본 기반 CRPS.** 앙상블·MC dropout·생성모델처럼 예측분포를 표본으로만 내는 모델은 밀도를 모르므로 NLL을 바로 계산할 수 없다. CRPS는 다음 등가식 덕분에 표본으로 계산된다. $X, X'$이 예측분포 $F$에서 독립으로 뽑은 두 표본일 때

$$
\mathrm{CRPS}(F, y) = \mathbb{E}\,\lvert X - y\rvert - \tfrac{1}{2}\,\mathbb{E}\,\lvert X - X'\rvert
$$

이다(Gneiting & Raftery, 2007). 첫째 항은 예측 표본이 관측에서 얼마나 먼가(정확도), 둘째 항은 예측 표본끼리 얼마나 퍼져 있는가(퍼짐에 대한 보상)다. 퍼짐을 무조건 보상하는 것은 아니다. 퍼짐을 키우면 첫째 항도 함께 커지므로, 둘의 균형이 참 분포에서 맞는다. 12.3장의 가우시안 닫힌 형태와 비교한다.

```python {exec="browser" session="eval"}
from scipy.stats import norm

def crps_samples(x, y):
    """x: (S,) 예측분포의 표본, y: 관측값. CRPS = E|X - y| - E|X - X'| / 2"""
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))

def crps_gauss(y, mu, s):
    z = (y - mu) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

rng = np.random.default_rng(0)
x = rng.normal(0.0, 1.0, size=2000)            # (S,) N(0, 1) 예측분포에서 뽑은 표본
for y in (0.0, 1.0, 3.0):
    print(f"y = {y}  표본 CRPS {crps_samples(x, y):.3f}   닫힌 형태 {crps_gauss(y, 0, 1):.3f}")
print(f"표본 1개(점 예측 0)일 때 CRPS = |0 - 1| = {crps_samples(np.zeros(1), 1.0):.3f}")
```

```output
y = 0.0  표본 CRPS 0.237   닫힌 형태 0.234
y = 1.0  표본 CRPS 0.620   닫힌 형태 0.602
y = 3.0  표본 CRPS 2.464   닫힌 형태 2.437
표본 1개(점 예측 0)일 때 CRPS = |0 - 1| = 1.000
```

표본 2000개로 계산한 값이 닫힌 형태와 소수 둘째 자리까지 맞는다. 차이는 표본 수가 유한해서 생기는 잡음이다. 표본이 하나(점 예측)이면 둘째 항이 0이 되어 CRPS는 절댓값 오차 $\lvert x - y\rvert$가 된다. 그래서 CRPS로는 점 예측 모델과 분포 예측 모델을 같은 표에서 비교할 수 있다.

**NLL과 CRPS 중 무엇을 쓰나.** NLL은 꼬리에 민감하다. 예측분포가 관측을 거의 0의 확률로 배제했다면 그 하나의 관측이 평균 NLL을 지배한다. 이상치가 많은 데이터에서는 CRPS가 더 안정적이다. 반대로 드문 사건을 놓치는 것이 치명적인 문제(예: 극한 강수)라면 NLL의 그 민감함이 원하는 성질일 수 있다. 어떤 규칙을 쓰든 같은 규칙으로 모든 방법을 비교하고, 무엇을 썼는지 밝힌다.

**연구에서의 의미.** 적정 점수는 모델 선택과 학습 손실 양쪽에 쓴다. 21.1장의 이분산 head는 NLL로 학습하고, 딥 앙상블 논문은 적정 점수로 학습하는 것을 방법의 일부로 제시했다(Lakshminarayanan 외, 2017). 그러나 점수 하나는 "얼마나 좋은가"를 말할 뿐 "어디가 왜 틀렸는가"를 말하지 않는다. 다음 두 절이 그 진단이다.

## calibration

**왜 필요한가.** "확률 0.8"이라는 예측이 쓸모 있으려면, 그런 예측을 받은 사례 중 실제로 약 80%가 일어나야 한다. 이것이 보정(calibration)이다. 보정이 맞지 않으면 확률을 결정에 그대로 쓸 수 없다. 21.6장의 비용 기반 threshold는 확률이 보정되어 있다는 가정 위에서만 최적이다.

**정의.** 분류에서 예측 확률 $q$가 보정되어 있다는 것은

$$
\Pr[Y = 1 \mid q(X) = p] = p \quad \text{(모든 } p\text{에 대해)}
$$

라는 뜻이다. 회귀에서는 예측 누적분포함수 $F_x$에 대해 **PIT**(probability integral transform) 값 $F_x(y)$가 $[0, 1]$에서 균등분포를 따르면 보정되어 있다고 한다. 예측분포가 참 조건부분포와 같다면 관측을 자신의 CDF에 넣은 값은 균등분포가 되기 때문이다.

**reliability diagram과 ECE.** 분류 확률을 같은 폭의 구간 $B_1, \dots, B_K$(보통 10개)로 나누고, 구간마다 평균 예측 확률과 관측 빈도를 비교한 그림이 reliability diagram이다. 대각선 위의 막대는 과소신뢰, 아래의 막대는 과신이다. 차이를 표본 수로 가중 평균한 값이 ECE(expected calibration error)다.

$$
\mathrm{ECE} = \sum_{k=1}^{K} \frac{\lvert B_k\rvert}{N}\,\Big\lvert \bar{y}_{B_k} - \bar{q}_{B_k} \Big\rvert.
$$

$\bar{y}_{B_k}$는 구간 $B_k$에 든 사례의 관측 빈도, $\bar{q}_{B_k}$는 평균 예측 확률이다. 아래 위젯에서 과신·과소신뢰 모델을 고르고 구간 수와 온도 $T$를 바꾸며 막대와 ECE가 어떻게 바뀌는지 확인한다.

::widget{name="calibration-diagram" model="over"}

**ECE의 한계.** ECE는 구간 수에 따라 값이 바뀌고, 표본이 적은 구간에서는 잡음이 크다. 그리고 평균적인 보정만 본다. 늘 기저율(예: 0.3)만 예보하는 모델도 ECE가 0이다. 보정은 좋은 예측의 **필요조건**일 뿐이다.

::quiz{id="ece-zero-limits"}

**PIT 히스토그램.** 회귀에서는 PIT 값의 히스토그램을 그린다. 평평하면 보정, U자이면 예측분포가 너무 좁고(관측이 양 끝에 몰린다), 가운데가 솟으면 너무 넓고, 한쪽으로 기울면 평균이 치우쳐 있다. 다음 절의 실험에서 숫자로 본다.

:::note[분야 예시]
일기예보는 확률 예보의 보정을 가장 오래 검증해 온 분야다. Brier score가 처음 제안된 곳도 기상학이다(Brier, 1950). 강수 확률 예보의 reliability diagram은 지금도 예보 기관의 표준 검증 도구이고, 앙상블 예보는 PIT 히스토그램과 같은 원리의 rank histogram으로 퍼짐이 적절한지 검사한다. 의료 위험 예측 모델에서도 판별력(AUC)과 별도로 calibration plot을 보고하는 것이 권장되는데, 같은 AUC라도 위험도를 체계적으로 과대평가하는 모델은 과잉 치료로 이어지기 때문이다.
:::

## sharpness

**왜 필요한가.** 보정만으로는 좋은 예측을 고를 수 없다. 늘 과거 전체의 분포를 내는 **기후값**(climatology) 예보는 장기적으로는 보정되어 있지만, 오늘이 특별한 날인지 알려 주지 않는다.

**정의.** sharpness는 예측분포가 얼마나 좁은가, 즉 예측분포 자체의 집중도다. 구간 폭의 평균이나 예측 표준편차의 평균으로 잰다. 관측과 무관하게 예측만 보고 계산한다는 점이 calibration과 다르다. Gneiting, Balabdaoui, Raftery(2007)는 확률 예측의 목표를 **보정이라는 제약 아래에서 sharpness를 최대화하는 것**으로 정리했다. 좁기만 하고 틀린 예측도, 맞기만 하고 넓은 예측도 아닌, 맞으면서 가능한 한 좁은 예측이다. 적정 점수는 이 두 성질을 함께 반영한다.

**직관.** 잡음이 $x$에 따라 커지는 데이터 $y = 2x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, (0.1 + 0.5x)^2)$(12.3장 실습과 같은 꼴)에 세 예측을 비교한다. 과거 관측의 분포를 그대로 쓰는 기후값, 평균은 맞지만 폭이 일정한 등분산 예측, 참 분포와 같은 이분산 예측이다.

```python {exec="browser" session="eval"}
xr = rng.uniform(0, 1, size=20000)             # (N,)
sd = 0.1 + 0.5 * xr                            # 참 잡음 sd: x에 따라 커진다
yr = 2 * xr + rng.normal(size=20000) * sd
u = rng.uniform(size=20000)
past = np.sort(2 * u + rng.normal(size=20000) * (0.1 + 0.5 * u))   # 과거 관측: 기후값 예측에 쓴다
s_pool = np.sqrt(np.mean(sd ** 2))
forecasts = {   # 이름: (PIT 값 (N,), 90% 구간 하한 (N,), 상한 (N,))
    "기후값": (np.searchsorted(past, yr) / len(past), *np.quantile(past, [0.05, 0.95])),
    "등분산": (norm.cdf(yr, 2 * xr, s_pool), *norm.ppf([[0.05], [0.95]], 2 * xr, s_pool)),
    "이분산": (norm.cdf(yr, 2 * xr, sd), *norm.ppf([[0.05], [0.95]], 2 * xr, sd)),
}
print("예측    90%포함 평균폭  포함(x<1/3) 포함(x>2/3)  PIT 5구간 비율")
for name, (pit, lo, hi) in forecasts.items():
    inside = (lo < yr) & (yr < hi)
    hist = np.histogram(pit, bins=5, range=(0, 1))[0] / len(yr)
    print(f"{name:6s}  {inside.mean():.3f}  {np.mean(hi - lo):.3f}     {inside[xr < 1/3].mean():.3f}"
          f"       {inside[xr > 2/3].mean():.3f}     {np.round(hist, 2)}")
for name, s in (("등분산", s_pool), ("이분산", sd)):
    print(f"{name}  NLL {np.mean(-norm.logpdf(yr, 2 * xr, s)):.3f}  CRPS {np.mean(crps_gauss(yr, 2 * xr, s)):.3f}")
```

```output
예측    90%포함 평균폭  포함(x<1/3) 포함(x>2/3)  PIT 5구간 비율
기후값     0.903  2.217     0.854       0.859     [0.19 0.2  0.19 0.21 0.2 ]
등분산     0.896  1.248     0.995       0.776     [0.17 0.2  0.26 0.21 0.17]
이분산     0.899  1.155     0.898       0.900     [0.2 0.2 0.2 0.2 0.2]
등분산  NLL 0.457  CRPS 0.209
이분산  NLL 0.280  CRPS 0.200
```

표를 세 방향으로 읽는다.

- **전체 포함률은 셋 모두 0.90 근처다.** 이 숫자 하나로는 세 예측을 구분할 수 없다.
- **sharpness는 다르다.** 기후값의 평균 폭(2.217)은 이분산 예측(1.155)의 두 배 가까이다. 기후값의 PIT는 평평하다. 보정되었지만 정보가 없다.
- **등분산 예측은 PIT가 가운데로 솟는다**(0.26). 잡음이 작은 곳에서는 지나치게 넓고, 큰 곳에서는 지나치게 좁은 실패가 섞여, 전체로는 약간 넓은 분포처럼 보인다. 아래 두 줄의 NLL·CRPS는 이분산 예측을 분명히 더 좋게 평가한다.

::quiz{id="calibrated-not-sharp"}

**연구에서의 의미.** 불확실성 방법을 비교하는 표에는 적정 점수(NLL 또는 CRPS) 외에 **보정 지표와 sharpness 지표를 따로** 둔다. 점수가 좋아졌을 때 그것이 더 정확해서인지 더 좁아져서인지를 구분해야 하고, 보정이 무너지면서 좁아진 것이라면 받아들이지 않는다.

## 구간 평가

**왜 필요한가.** 많은 연구는 분포 전체 대신 "90% 예측 구간"만 보고한다. 구간의 평가는 두 숫자, **coverage**(실제 포함률)와 **폭**으로 시작한다. 목표 포함률을 맞추면서 폭이 좁은 구간이 좋다. 둘을 하나로 묶은 적정 점수가 interval score다.

**조건별·집단별 calibration.** 위 표의 핵심은 넷째·다섯째 열이다. 등분산 예측의 포함률은 $x < 1/3$에서 0.995, $x > 2/3$에서 0.776이다. 전체 0.896은 이 두 실패의 평균일 뿐이다. 기후값도 양 끝에서 0.85 근처로 떨어진다. 참 분포와 같은 이분산 예측만 두 구간 모두 0.90 근처다. **주변**(marginal) 포함률이 맞는다고 **조건부**(conditional) 포함률이 맞는 것은 아니다. 이 구분은 21.3장 conformal prediction의 보장을 이해하는 데도 핵심이다.

조건을 나누는 기준은 연구 질문이 정한다.

- 입력 변수의 구간(위의 $x$ 구간, 온도 범위, 농도 범위)
- 예측 불확실성 자체의 크기(예측 sd가 큰 사례와 작은 사례) — 불확실성 추정이 순위라도 맞는지 본다
- 집단(환자군, 장비, 실험실, 재료 계열) — 15.4장의 독립성 단위와 같은 기준

집단이 작으면 포함률 추정 자체의 오차가 크다. $n$개 중 포함 비율의 표준오차는 약 $\sqrt{0.9 \times 0.1 / n}$이므로, $n = 50$이면 0.04 정도다. 집단별 포함률에는 이 정도의 흔들림을 함께 적는다(12.4장의 구간 추정).

::quiz{id="interval-score-numeric"}

:::note[분야 예시]
배터리 수명 예측에서 셀 화학·충전 프로토콜별로 coverage를 따로 보면, 전체 포함률은 맞아도 고속 충전 셀에서만 구간이 체계적으로 좁은 경우가 흔하다. 신약 개발의 물성 예측에서는 학습 데이터와 구조가 비슷한 분자와 먼 분자를 나눠 coverage를 본다. 먼 분자에서 포함률이 무너진다면, 그 모델의 불확실성은 21.4장의 분포 이동을 반영하지 못하는 것이다.
:::

## 재보정

**왜 필요한가.** 보정이 틀린 모델을 다시 학습하지 않고, 별도의 보정 데이터로 출력 확률만 고칠 수 있다. 현대의 깊은 분류 신경망은 흔히 과신한다는 보고가 있고(Guo 외, 2017), 21.1장에서 보았듯 앙상블도 퍼짐이 부족할 수 있다.

**temperature scaling.** 로짓 $z$를 하나의 양수 $T$로 나눈다: $q = \sigma(z / T)$(다중 class면 softmax). $T > 1$이면 확률이 0.5 쪽으로 당겨지고(과신 교정), $T < 1$이면 바깥으로 밀린다. $T$는 보정 데이터의 NLL을 최소화해 정한다. 파라미터가 하나뿐이라 보정 데이터가 적어도 과적합 위험이 작고, 로짓의 순서를 바꾸지 않으므로 **정확도와 AUC는 그대로**다(Guo 외, 2017).

위젯과 같은 설정으로, 참 로짓을 2.5배로 부풀린 과신 모델에 temperature scaling을 적용한다. 보정용 2000개로 $T$를 정하고 따로 떼어 둔 5000개로 평가한다.

```python {exec="browser" session="eval"}
sig = lambda t: 1 / (1 + np.exp(-t))
z = rng.normal(0, 1.5, size=7000)              # (N,) 참 로짓
yb = (rng.uniform(size=7000) < sig(z)).astype(float)
logit = 2.5 * z                                # 과신하는 모델: 로짓을 2.5배로 낸다
cal, te = slice(0, 2000), slice(2000, None)    # 보정용 2000개, 평가용 5000개

def scores(q, y, bins=10):
    b = np.minimum((q * bins).astype(int), bins - 1)
    ece = sum(abs(y[b == k].mean() - q[b == k].mean()) * np.mean(b == k) for k in range(bins) if np.any(b == k))
    nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return f"정확도 {np.mean((q > 0.5) == y):.3f}  ECE {ece:.3f}  Brier {np.mean((q - y) ** 2):.3f}  NLL {nll:.3f}"

Ts = np.linspace(0.5, 5, 451)
nll_cal = [-np.mean(yb[cal] * np.log(sig(logit[cal] / T)) + (1 - yb[cal]) * np.log(1 - sig(logit[cal] / T))) for T in Ts]
T_best = Ts[np.argmin(nll_cal)]
print("보정 전     ", scores(sig(logit[te]), yb[te]))
print(f"T = {T_best:.2f}    ", scores(sig(logit[te] / T_best), yb[te]))
```

```output
보정 전      정확도 0.738  ECE 0.130  Brier 0.195  NLL 0.660
T = 2.57     정확도 0.738  ECE 0.013  Brier 0.175  NLL 0.524
```

보정 데이터에서 찾은 $T = 2.57$은 참값 2.5에 가깝다. 정확도는 그대로이고, ECE·Brier·NLL이 모두 줄었다. 보정 데이터와 평가 데이터를 나눈 것이 중요하다. 같은 데이터로 $T$를 정하고 평가하면 15.4장의 누수와 같은 이유로 보정이 실제보다 좋아 보인다.

::quiz{id="temperature-keeps-accuracy"}

**isotonic regression.** 예측 확률을 관측 빈도로 보내는 **단조 증가 함수**를 데이터에서 직접 맞춘다. 모양을 가정하지 않으므로 temperature scaling이 고칠 수 없는 비대칭 왜곡도 고친다. 대신 계단 함수라 보정 데이터가 적으면 과적합하고, 동점이 생겨 순위 정보가 일부 사라진다. 집필 시점 예시로, scikit-learn은 `sklearn.isotonic.IsotonicRegression`과, 분류기에 sigmoid(Platt) 또는 isotonic 보정을 씌우는 `sklearn.calibration.CalibratedClassifierCV`를 제공한다. 사용하는 버전의 문서에서 인자를 확인한다.

**회귀의 재보정.** 회귀 예측분포는 PIT 값의 경험적 분포로 재보정한다. 보정 데이터에서 PIT가 0.9 이하인 비율이 0.8이라면, 이후에 "90% 분위수"를 낼 때 원래 모델의 다른 분위수를 써서 실제로 90%가 되게 맞춘다. 같은 원리를 분포 가정 없이 유한 표본 보장까지 붙인 것이 21.3장의 conformal prediction이다.

**재보정이 고치지 못하는 것.** 위 방법은 모두 **보정 데이터의 분포에서** 평균적인 보정을 맞춘다. 조건별 실패(등분산 예측의 $x$ 구간별 실패)는 출력 확률의 단조 변환 하나로 고쳐지지 않고, 분포가 바뀐 새 데이터에서는 보정이 다시 무너진다.

## 실습

네 가지 불확실성 방법을 NLL·CRPS·coverage·폭·조건별 coverage로 비교한다. 코드는 `labs/reliability/evaluating-predictive-distributions/compare_methods.py`다. 데이터는 $y = \sin 2x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, (0.05 + 0.2x)^2)$이고 $x \in [0, 3]$에서 300개로 학습한다. 네 방법은 모두 평균을 5차 다항식으로 맞추고, 분산을 하나로 두거나(등분산) 21.1장의 방법으로 $x$에 따라 바꾼다(이분산). 각각 하나의 모델과 부트스트랩 앙상블 20개로 만든다. 평가는 학습 범위 안과 밖(x ∈ [3, 3.5])에서 따로 한다. 집필 시점의 실행 결과는 다음과 같다(시드 0).

```output
학습 범위 안 (x ∈ [0, 3])                             90% 포함률 (x 구간별)
방법            NLL     CRPS   포함    폭     [0,1)  [1,2)  [2,3)
등분산           0.496  0.216  0.886  1.268  0.997  0.916  0.747
이분산           0.233  0.203  0.888  1.130  0.899  0.895  0.870
등분산 앙상블       0.486  0.215  0.884  1.253  0.997  0.913  0.744
이분산 앙상블       0.236  0.203  0.891  1.162  0.884  0.901  0.887

학습 범위 밖 (x ∈ [3, 3.5])
방법            NLL     CRPS   포함    폭
등분산           1.935  0.470  0.589  1.268
이분산           1.449  0.453  0.707  1.628
등분산 앙상블       1.315  0.479  0.881  2.805
이분산 앙상블       1.190  0.449  0.900  2.613
```

결과는 앞 절들의 주장을 한 표로 보여 준다.

- **학습 범위 안에서는 전체 포함률이 넷 다 0.88–0.89로 비슷하다.** 그러나 등분산 방법은 $[0, 1)$에서 0.997, $[2, 3)$에서 0.747로 조건별로 무너진다. NLL(0.496 대 0.233)이 이 차이를 잡는다.
- **학습 범위 안에서 앙상블은 거의 아무것도 바꾸지 않는다.** 데이터가 있는 곳에서는 epistemic 불확실성이 작기 때문이다.
- **학습 범위 밖에서는 순서가 바뀐다.** 단일 모델은 포함률이 0.589(등분산), 0.707(이분산)로 떨어진다. 외삽 오차를 모르기 때문이다. 앙상블은 구간을 두 배 넘게 넓혀 포함률을 0.88–0.90으로 지킨다. 학습 범위 밖에서 가장 좋은 NLL은 이분산 앙상블(1.190)이다.

평가를 학습 범위 안에서만 했다면 앙상블의 가치는 보이지 않았을 것이다. **어디서 평가하느냐가 결론을 바꾼다.** 이 점은 21.4장에서 다시 다룬다.

이어서 두 연습문제를 푼다.

::exercise{id="crps-ensemble"}

::exercise{id="interval-score"}

## 실패 사례

- **평균 coverage만 보고 조건별 실패를 놓치는 것.** 전체 포함률 0.90은 잡음이 작은 곳의 과대 포함과 큰 곳의 과소 포함이 상쇄된 결과일 수 있다. 입력 구간·집단·예측 불확실성 크기별로 나눠 본다.
- **적정하지 않은 점수로 확률을 평가·학습하는 것.** 예측 확률과 결과의 절댓값 차이, 정확도처럼 확률의 크기를 보지 않는 지표는 과신을 보상하거나 무시한다.
- **보정과 평가에 같은 데이터를 쓰는 것.** temperature나 isotonic 함수를 맞춘 데이터로 ECE를 재면 보정이 실제보다 좋아 보인다.
- **ECE 하나로 보정을 결론짓는 것.** 구간 수에 따라 값이 바뀌고, 판별력이 없는 모델도 ECE가 0일 수 있다. reliability diagram과 적정 점수를 함께 본다.
- **학습 범위 안에서만 불확실성을 평가하는 것.** 불확실성이 필요한 곳은 대개 학습 범위의 가장자리와 바깥이다. 실습처럼 범위 밖 평가를 따로 둔다.

## 통과 기준

- 불확실성 방법 두 개 이상을 적정 점수(NLL 또는 CRPS), 전체 coverage, 평균 폭, 조건별(또는 집단별) coverage를 모두 포함한 한 표로 비교한다.
- 분류 모델이면 reliability diagram과 ECE를, 회귀 모델이면 PIT 히스토그램을 보정 데이터와 분리한 평가 데이터에서 제시한다.
- 연습문제 두 개(`crps-ensemble`, `interval-score`)의 테스트를 통과한다.

## 참고 문헌

- Brier (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. — Brier score
- Gneiting, Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. — 적정 점수 규칙, CRPS의 표본 표현, interval score
- Gneiting, Balabdaoui, Raftery (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B*, 69(2), 243–268. — 보정 아래 sharpness 최대화 원칙, PIT 히스토그램
- Guo, Pleiss, Sun, Weinberger (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330. — 신경망의 과신과 temperature scaling
- Lakshminarayanan, Pritzel, Blundell (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. — 적정 점수로 학습하는 분포 예측
