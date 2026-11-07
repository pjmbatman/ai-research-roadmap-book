---
id: reliability/uncertainty-quantification
title: "불확실성 정량화"
title_en: "Uncertainty Quantification"
tier: core
status: draft
prerequisites: [statistical-inference/distributional-formulation, bayesian-inference/bayesian-thinking]
---

# 불확실성 정량화

배터리 수명 모델이 "이 셀은 812 사이클을 버틴다"고 답했다. 이 숫자를 믿고 시험을 줄여도 될까. 답은 모델이 이 셀과 비슷한 셀을 얼마나 많이 보았는지, 같은 조건의 셀들이 원래 얼마나 흩어지는지, 그리고 이 셀이 학습 데이터와 다른 조건에서 만들어졌는지에 달려 있다. 숫자 하나로는 셋 중 어느 것도 알 수 없다. 이 장은 예측의 불확실성을 **출처별로** 정량화하는 방법을 다룬다.

12.3장은 분포를 출력하는 모델과 aleatoric·epistemic의 구분을 소개했고, 13부는 파라미터 사후분포로 epistemic 불확실성을 표현하는 베이지안 방법을 다뤘다. 이 장은 그 둘을 딥러닝 규모의 모델에서 실제로 계산하는 방법, 즉 이분산 출력 head, 앙상블, 근사 베이지안 방법을 구현하고, 두 불확실성을 분리해 보고하는 법을 다룬다. 불확실성이 맞는지 평가하는 법은 21.2장, 분포 가정 없이 구간을 보장하는 법은 21.3장에서 이어진다.

## 학습 목표

- aleatoric·epistemic 불확실성을 구분한다
- 딥러닝 불확실성 추정 방법을 구현한다

## 불확실성의 종류

**왜 필요한가.** 예측 구간이 넓다는 사실만으로는 다음 행동을 정할 수 없다. 측정 잡음 때문에 넓다면 모델을 키워도 소용이 없고 센서를 바꿔야 한다. 데이터가 부족해서 넓다면 그 영역에서 측정을 더 하면 된다. 학습 범위 밖이라서 넓다면(혹은 넓어야 하는데 좁다면) 그 예측을 쓰지 말아야 한다. 불확실성을 출처별로 나눠야 행동이 정해진다.

**정의.** 12.3장의 구분을 다시 쓴다.

- **aleatoric 불확실성**(aleatoric uncertainty): 같은 입력 $x$에서도 출력 $y$가 흩어지는 정도. 측정 잡음, 제조 편차, 관측되지 않은 변수처럼 데이터 생성 과정 자체의 무작위성에서 온다. 조건부분포 $p(y \mid x)$의 폭이다. 같은 조건의 데이터를 늘려도 줄지 않는다.
- **epistemic 불확실성**(epistemic uncertainty): 모델이 데이터를 충분히 보지 못해서 생기는 불확실성. 파라미터 $\theta$를 하나로 정할 수 없다는 사실, 즉 사후분포 $p(\theta \mid \mathcal{D})$의 폭에서 온다. 그 영역의 데이터가 늘면 줄어든다.

**수식.** 파라미터 사후분포(또는 그 근사)로 예측분포를 평균 내면

$$
p(y \mid x, \mathcal{D}) = \int p(y \mid x, \theta)\, p(\theta \mid \mathcal{D})\, d\theta
$$

이다(13.1장의 사후 예측분포). 이 분포의 분산을 전분산 법칙(law of total variance)으로 나누면 두 불확실성이 항 하나씩으로 갈라진다.

$$
\operatorname{Var}[y \mid x] = \underbrace{\mathbb{E}_{\theta}\big[\sigma^2_\theta(x)\big]}_{\text{aleatoric}} + \underbrace{\operatorname{Var}_{\theta}\big[\mu_\theta(x)\big]}_{\text{epistemic}}.
$$

여기서 $\mu_\theta(x)$와 $\sigma^2_\theta(x)$는 파라미터 $\theta$를 가진 모델이 낸 예측분포의 평균과 분산이고, 기대값과 분산은 $\theta$의 사후분포에 대해 취한다. 첫째 항은 "각 모델이 생각하는 잡음 크기의 평균", 둘째 항은 "모델들이 평균에 대해 서로 얼마나 다른 말을 하는가"다. 앙상블에서는 $\theta$의 사후분포 대신 구성원 $M$개를 쓰고 기대값을 표본 평균으로 바꾼다.

::quiz{id="more-data-reduces-which"}

**직관.** 두 불확실성은 관측값에 **함께 섞여** 나타난다. 측정 하나가 예측에서 0.5만큼 벗어났을 때, 그 0.5 중 얼마가 잡음이고 얼마가 모델의 무지인지는 그 관측 하나로는 알 수 없다. 분리할 수 있는 것은 모델 구조 덕분이다. aleatoric은 "같은 입력 근처의 관측들이 얼마나 흩어지는가"로, epistemic은 "데이터를 조금 바꿔 다시 학습하면 예측이 얼마나 바뀌는가"로 추정한다. 그래서 분해 결과는 모델 가정에 의존한다. 잡음 모델이 틀리면 aleatoric 추정이 틀리고, 그 오차의 일부가 epistemic처럼 보이기도 한다.

**코드.** 왼쪽은 잡음이 작고 오른쪽은 잡음이 큰 1차원 데이터에 부트스트랩 앙상블을 학습한다. 가운데 $[-1, 1]$에는 데이터가 없다. 구성원마다 평균은 5차 다항식으로, $\log$ 분산은 잔차 제곱의 로그에 2차 다항식을 맞춰 추정한다. 표준정규 $\varepsilon$에 대해 $\mathbb{E}[\log \varepsilon^2] \approx -1.27$이므로 그만큼 더해 편향을 보정한다.

```python {exec="browser" session="uq"}
import numpy as np

rng = np.random.default_rng(0)
n = 300
x = np.concatenate([rng.uniform(-3, -1, n // 2), rng.uniform(1, 3, n // 2)])  # (N,) [-1, 1]은 비어 있다
sd_true = lambda x: 0.05 + 0.1 * (x + 3)             # 참 잡음 sd: 오른쪽으로 커진다
y = np.sin(x) + rng.normal(size=n) * sd_true(x)      # (N,)

def fit_member(xb, yb):
    mu_c = np.polyfit(xb, yb, 5)                     # 평균: 5차 다항식
    r2 = (yb - np.polyval(mu_c, xb)) ** 2
    lv_c = np.polyfit(xb, np.log(r2) + 1.27, 2)      # log 분산: 2차 다항식 (E[log χ²₁] ≈ -1.27 보정)
    return mu_c, lv_c

M = 20
members = [fit_member(x[i], y[i]) for i in rng.integers(0, n, size=(M, n))]  # 부트스트랩 재표본 M개

x0 = np.array([-2.0, 2.0, 0.0, 4.0])
mu = np.array([np.polyval(a, x0) for a, _ in members])           # (M, 4) 구성원별 평균
var = np.array([np.exp(np.polyval(b, x0)) for _, b in members])  # (M, 4) 구성원별 분산
alea, epi = var.mean(axis=0), mu.var(axis=0)                     # 전분산 법칙의 두 항
for xi, a, e in zip(x0, alea, epi):
    print(f"x = {xi:4.1f}  aleatoric sd {np.sqrt(a):.3f}  epistemic sd {np.sqrt(e):.3f}  참 잡음 sd {sd_true(xi):.3f}")
```

```output
x = -2.0  aleatoric sd 0.140  epistemic sd 0.017  참 잡음 sd 0.150
x =  2.0  aleatoric sd 0.528  epistemic sd 0.067  참 잡음 sd 0.550
x =  0.0  aleatoric sd 0.394  epistemic sd 0.092  참 잡음 sd 0.350
x =  4.0  aleatoric sd 0.395  epistemic sd 1.516  참 잡음 sd 0.750
```

결과를 영역별로 읽는다.

- **$x = -2$ (데이터 많음, 잡음 작음)**: 두 불확실성이 모두 작다. aleatoric 추정(0.140)이 참값(0.150)에 가깝다.
- **$x = 2$ (데이터 많음, 잡음 큼)**: aleatoric이 크고(0.528) epistemic은 작다. 구간이 넓은 이유는 잡음이고, 데이터를 늘려도 좁아지지 않는다.
- **$x = 0$ (빈 구간)**: epistemic이 데이터가 있는 곳보다 크지만(0.092) 기대만큼 크지는 않다. 구성원이 모두 같은 매끄러운 다항식으로 빈틈을 메우기 때문이다. 앙상블의 불일치는 **모델 클래스가 허용하는 다양성**만 잡는다.
- **$x = 4$ (외삽)**: epistemic이 압도적으로 크다(1.516). 그런데 aleatoric 추정(0.395)은 참값(0.750)의 절반쯤이다. 잡음 모델도 외삽하고 있기 때문이다. 학습 범위 밖에서는 aleatoric 추정도 믿을 수 없다.

**연구에서의 의미.** 분해 결과는 다음 측정을 어디서 할지 정하는 근거가 된다. epistemic이 큰 영역은 측정을 더 하면 좁아지고, aleatoric이 큰 영역은 측정 방법을 바꿔야 좁아진다. 능동 학습과 실험 설계(25.3장)는 이 구분을 직접 쓴다. 단, 위 결과처럼 분해는 모델 가정 위의 추정이다. "epistemic이 작다"는 말은 "이 모델 클래스 안에서 구성원들이 합의했다"는 뜻이지 "참값을 안다"는 뜻이 아니다.

:::note[분야 예시]
기계학습 원자간 퍼텐셜(machine-learned interatomic potential)로 분자동역학을 돌릴 때, 여러 퍼텐셜 모델의 힘 예측이 크게 갈리는 원자 배치를 골라 양자화학 계산으로 레이블을 붙이고 학습 데이터에 추가하는 방식이 널리 쓰인다. 구성원 간 불일치(epistemic)를 "모델이 아직 모르는 배치"의 신호로 쓰는 것이다. 반대로 양자화학 계산 자체의 수치 오차는 같은 배치를 다시 계산해도 남는 aleatoric 성분에 가깝다.
:::

## 확률적 출력

**왜 필요한가.** aleatoric 불확실성은 입력마다 다르다. 위 데이터에서 잡음은 오른쪽으로 갈수록 커졌다. 출력 분산을 상수 하나로 두는 모델은 이를 표현할 수 없고, 12.3장 실습에서 보았듯 전체 포함률은 맞아도 조건별로는 크게 틀린다.

**정의.** 신경망의 마지막 층이 분포의 파라미터를 출력하게 한다. 이를 **분포 파라미터 head**라 한다. 회귀에서 가장 흔한 것은 평균과 로그 분산 두 출력이다.

$$
f_\theta(x) = \big(\mu_\theta(x),\ s_\theta(x)\big), \qquad \sigma^2_\theta(x) = \exp\big(s_\theta(x)\big).
$$

손실은 12.3장의 이분산 가우시안 NLL이다. $\exp$ 대신 softplus를 써도 된다. 분류에서는 softmax 출력이 이미 범주형 분포의 파라미터다. 꼬리가 두꺼우면 Student-t, 양수만 가능하면 log-normal이나 gamma처럼 데이터에 맞는 분포를 고른다.

**학습의 불안정성.** 이분산 NLL을 평균으로 미분하면

$$
\frac{\partial}{\partial \mu}\left[\tfrac{1}{2}\log\sigma^2 + \frac{(y-\mu)^2}{2\sigma^2}\right] = -\frac{y-\mu}{\sigma^2}
$$

이다. 평균의 gradient가 $1/\sigma^2$로 가중된다. 그래서 두 가지 일이 생긴다.

- 학습 초기에 평균을 잘 못 맞히는 영역에서는 $\sigma$를 키우는 것이 NLL을 빨리 줄이는 길이다. 모델이 평균을 고치는 대신 "여기는 잡음이 크다"고 설명해 버리면, 그 영역의 평균 gradient는 작아지고 학습이 멈춘 것처럼 보인다.
- 반대로 몇몇 점에서 $\sigma$가 아주 작아지면 그 점들의 gradient가 폭발한다. 로그 분산을 일정 범위로 자르거나(clamp) 최솟값을 더하는 이유다.

실무에서는 평균을 먼저 MSE로 학습한 뒤 분산 head를 학습하는 warm-up, 로그 분산 범위 제한, 평균 gradient의 $1/\sigma^2$ 가중을 약하게 만드는 손실 변형을 쓴다. 이 장의 실습 코드는 앞의 두 가지를 쓴다.

::quiz{id="heteroscedastic-gradient"}

분포 head 하나가 내는 것은 aleatoric 불확실성뿐이다. 파라미터를 하나로 정했으므로 epistemic 항 $\operatorname{Var}_\theta[\mu_\theta(x)]$는 0이다. 학습 데이터가 없는 영역에서도 $\sigma_\theta(x)$는 어떤 값이든 낼 수 있고, 그 값에는 근거가 없다. epistemic을 얻으려면 $\theta$의 불확실성을 표현하는 다음 절의 방법이 필요하다.

## 앙상블과 근사 베이지안

**왜 필요한가.** 13부의 방법(켤레 사전분포, MCMC, 변분 추론)은 파라미터 수백만 개의 신경망에 그대로 쓰기 어렵다. 사후분포를 정확히 구하는 대신, 사후분포에서 뽑은 것처럼 **서로 다르면서 모두 그럴듯한 모델 여러 개**를 싸게 얻는 방법들이 쓰인다. 모든 방법은 결국 구성원 $\theta_1, \dots, \theta_M$을 만들고 위의 전분산 법칙으로 두 불확실성을 계산한다. 차이는 구성원을 어떻게 만드느냐다.

| 방법 | 구성원을 만드는 법 | 비용 | 특징 |
|---|---|---|---|
| **deep ensemble** | 무작위 초기값(과 데이터 순서)을 바꿔 독립 학습 $M$번 | 학습 $M$배, 추론 $M$배 | 구현이 단순하고 경험적으로 강하다(Lakshminarayanan 외, 2017) |
| **부트스트랩 앙상블** | 복원추출한 데이터로 각각 학습(12.4장) | 학습 $M$배 | 데이터 변동을 직접 반영. 작은 모델·표 데이터에서 흔하다 |
| **MC dropout** | 학습한 dropout 신경망에서 예측 때도 dropout을 켜고 $T$번 forward | 학습 1번, 추론 $T$배 | 근사 변분 추론으로 해석된다(Gal & Ghahramani, 2016). dropout 비율이 불확실성 크기를 좌우한다 |
| **Laplace 근사** | 학습한 가중치를 최빈값으로 두고 손실의 곡률로 가우시안 근사(13.6장) | 학습 1번 + 곡률 계산 | 사후에 붙일 수 있다. 마지막 층만 쓰는 근사가 흔하다(13.2장) |
| **SWAG** | SGD 궤적의 가중치 평균과 공분산으로 가우시안 근사 | 학습 1번 + 추가 epoch | 저차원 + 대각 공분산을 쓴다(Maddox 외, 2019) |
| **BNN** | 가중치 사후분포를 변분 추론·MCMC로 직접 근사 | 크다 | 원리가 가장 분명하지만 큰 모델에서는 근사가 거칠어진다 |

**직관.** 신경망의 손실면에는 성능이 비슷한 해가 많다(10.6장). deep ensemble은 서로 다른 초기값에서 출발해 서로 다른 골짜기에 도착한 해들을 모은다. 이 해들은 데이터가 있는 곳에서는 같은 예측을 하지만(그렇게 학습되었으므로), 데이터가 없는 곳에서는 각자 다르게 외삽한다. 그 차이가 epistemic 불확실성의 추정이다. MC dropout·Laplace·SWAG는 한 골짜기 **안**의 퍼짐을 근사하므로, 서로 다른 골짜기를 모으는 앙상블보다 불일치가 작게 나오는 경향이 있다.

::quiz{id="ensemble-diversity-source"}

**앙상블 크기.** 구성원 $M$개로 계산한 epistemic 표준편차는 그 자체가 표본 $M$개짜리 추정량이다. 같은 데이터에 앙상블을 여러 번 새로 만들어, 외삽점 $x = 4$에서 epistemic 표준편차 추정이 얼마나 흔들리는지 본다.

```python {exec="browser" session="uq"}
def epistemic_sd(M, x_eval):
    idx = rng.integers(0, n, size=(M, n))
    mus = [np.polyval(np.polyfit(x[i], y[i], 5), x_eval) for i in idx]
    return np.std(mus)

R = 100
for M in (2, 5, 20):
    est = np.array([epistemic_sd(M, 4.0) for _ in range(R)])   # (R,) 같은 데이터, 다른 앙상블
    print(f"M = {M:2d}  epistemic sd 평균 {est.mean():.2f}  반복 간 sd {est.std():.2f}  상대 흔들림 {est.std() / est.mean():.0%}")
```

```output
M =  2  epistemic sd 평균 0.84  반복 간 sd 0.64  상대 흔들림 76%
M =  5  epistemic sd 평균 1.23  반복 간 sd 0.43  상대 흔들림 35%
M = 20  epistemic sd 평균 1.42  반복 간 sd 0.24  상대 흔들림 17%
```

$M = 2$이면 추정이 평균의 76%만큼 흔들리고, 평균 자체도 $M = 20$일 때보다 작다. 표본 표준편차는 표본이 적을수록 아래로 치우치기 때문이다. 구성원이 적은 앙상블은 epistemic 불확실성을 **작게, 그리고 불안정하게** 추정한다. 실무에서 흔한 5개는 순위를 매기는 데는 쓸 만하지만, 표준편차의 크기를 수치로 보고할 때는 이 흔들림을 고려해야 한다.

**softmax는 불확실성이 아니다.** 분류 모델의 softmax 확률을 그대로 불확실성으로 쓰는 일이 흔하다. 두 class를 구분하는 로지스틱 회귀의 부트스트랩 앙상블로, 결정 경계 근처·학습 범위 안·학습 범위에서 아주 먼 점의 예측을 비교한다.

```python {exec="browser" session="uq"}
def fit_logreg(X, t, steps=2000, lr=0.5):
    Xb = np.hstack([X, np.ones((len(X), 1))])          # (N, 3) bias 열 추가
    w = np.zeros(3)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - t) / len(t)              # 평균 교차 엔트로피의 gradient
    return w

Xc = np.vstack([rng.normal([-1, 0], 1.0, (200, 2)), rng.normal([1, 0], 1.0, (200, 2))])  # (400, 2)
tc = np.repeat([0.0, 1.0], 200)                                                         # (400,)
ens = [fit_logreg(Xc[i], tc[i]) for i in rng.integers(0, 400, size=(10, 400))]          # 부트스트랩 10개

queries = {"경계 근처 (0, 0)": [0.0, 0.0], "학습 범위 안 (1, 0)": [1.0, 0.0], "멀리 밖 (20, 30)": [20.0, 30.0]}
for name, q in queries.items():
    p = np.array([1 / (1 + np.exp(-(w[:2] @ q + w[2]))) for w in ens])
    print(f"{name:14s}  p(class 1) 평균 {p.mean():.3f}  구성원 간 sd {p.std():.3f}")
```

```output
경계 근처 (0, 0)    p(class 1) 평균 0.485  구성원 간 sd 0.052
학습 범위 안 (1, 0)  p(class 1) 평균 0.912  구성원 간 sd 0.030
멀리 밖 (20, 30)   p(class 1) 평균 1.000  구성원 간 sd 0.000
```

경계 근처에서는 확률이 0.5 근처이고 구성원 간 차이도 있다. 그런데 학습 데이터에서 아주 먼 점 $(20, 30)$에서는 확률이 1.000이고 구성원 간 차이도 0이다. 선형 모델은 경계에서 멀어질수록 로짓이 커지고, 모든 구성원이 **같은 방식으로** 외삽하기 때문이다. softmax 확률은 "경계의 어느 쪽에 얼마나 멀리 있는가"를 말할 뿐 "학습 범위 안에 있는가"를 말하지 않는다. 앙상블도 구성원들이 같은 방식으로 틀리는 곳에서는 이를 잡지 못한다. 입력이 학습 범위 밖인지는 21.4장의 방법으로 따로 확인한다.

::quiz{id="softmax-confidence"}

:::note[분야 예시]
기상·기후 예보는 오래전부터 앙상블 예보를 쓴다. 초기 조건과 모델 물리 설정을 조금씩 바꾼 수십 개의 수치 예보를 돌리고, 예보들이 갈리는 정도를 예보 불확실성으로 제시한다. 같은 문제가 그대로 나타난다. 모든 구성원이 공유하는 모델 오차(예: 같은 구름 모수화)는 앙상블의 퍼짐에 나타나지 않으므로, 앙상블은 흔히 실제 오차보다 좁게 퍼진다(under-dispersion). 그래서 기상 예보는 앙상블 출력을 과거 관측으로 다시 보정한다(21.2장).
:::

**연구에서의 의미.** 방법을 고를 때는 "구성원 사이의 차이가 무엇을 반영하는가"를 먼저 묻는다. 무작위 초기값만 바꾼 앙상블은 최적화의 다양성을, 부트스트랩은 데이터 표본의 변동을, MC dropout은 dropout이 허용하는 퍼짐을 반영한다. 어떤 방법도 모든 구성원이 공유하는 가정(입력 특징, 모델 구조, 학습 데이터의 범위)에서 오는 오차는 반영하지 못한다. 그래서 불확실성 추정은 반드시 별도의 데이터로 평가해야 한다(21.2장).

## 불확실성 전파

**왜 필요한가.** 모델의 **입력**에도 불확실성이 있다. 측정한 온도에는 센서 오차가, 문헌에서 가져온 활성화 에너지에는 실험 오차가 있다. 입력의 불확실성이 출력에서 얼마나 커지는지 모르면, 모델이 아무리 정확해도 최종 예측의 불확실성을 과소평가한다.

**정의.** 입력 $X$가 분포 $p(x)$를 따를 때 출력 $Y = g(X)$의 분포를 구하는 문제다. 두 가지 방법이 있다.

- **선형화**(1차 전파, delta method): $g$를 평균 $\mu_X$ 근처에서 1차 Taylor 근사(7.2장)하면 $\operatorname{Var}[Y] \approx g'(\mu_X)^2 \operatorname{Var}[X]$다. 다변수에서는 Jacobian $\mathbf{J}$로 $\boldsymbol\Sigma_Y \approx \mathbf{J}\boldsymbol\Sigma_X\mathbf{J}^\top$이다. 계산이 싸지만 $g$가 입력 불확실성의 범위 안에서 거의 선형일 때만 맞다.
- **Monte Carlo 전파**: $p(x)$에서 표본 $x^{(s)}$를 많이 뽑아 $y^{(s)} = g(x^{(s)})$를 계산하고, 그 표본으로 분포를 요약한다. $g$의 모양과 상관없이 맞지만 $g$를 여러 번 평가해야 한다.

**직관.** 화학 반응 속도 상수의 Arrhenius 식 $k = A\exp(-E_a / (RT))$에서 활성화 에너지 $E_a$가 불확실하다고 하자. $k$는 $E_a$의 지수 함수이므로, $E_a$의 대칭인 불확실성이 $k$에서는 오른쪽으로 긴 꼬리가 된다. 선형화는 이 비대칭을 볼 수 없다.

```python {exec="browser" session="uq"}
R_J_PER_MOL_K = 8.314
temp_k = 300.0
ea_j_per_mol = rng.normal(80e3, 2e3, size=100_000)          # (S,) 활성화 에너지의 불확실성
ratio = np.exp(-(ea_j_per_mol - 80e3) / (R_J_PER_MOL_K * temp_k))   # (S,) k / k(평균 Ea)

sd_lin = 2e3 / (R_J_PER_MOL_K * temp_k)                      # 1차 선형화: d ln k = -dEa / (R T)
lo, med, hi = np.percentile(ratio, [2.5, 50, 97.5])
print(f"Monte Carlo  평균 {ratio.mean():.2f}  중앙값 {med:.2f}  95% 구간 [{lo:.2f}, {hi:.2f}]")
print(f"선형화       평균 1.00  sd {sd_lin:.2f}  95% 구간 [{1 - 1.96 * sd_lin:.2f}, {1 + 1.96 * sd_lin:.2f}]")
```

```output
Monte Carlo  평균 1.37  중앙값 1.00  95% 구간 [0.21, 4.74]
선형화       평균 1.00  sd 0.80  95% 구간 [-0.57, 2.57]
```

$E_a = 80 \pm 2$ kJ/mol(표준편차)일 때, $300$ K에서 $k$를 평균 $E_a$로 계산한 값의 비율로 나타냈다. Monte Carlo로 구한 95% 구간 $[0.21, 4.74]$는 중앙값 1을 기준으로 크게 비대칭이고, 평균(1.37)은 중앙값보다 크다. 선형화는 대칭 구간 $[-0.57, 2.57]$을 주는데, 하한이 **음수인 반응 속도**다. 물리적으로 불가능한 값이 나왔다는 것 자체가 선형화가 이 범위에서 맞지 않는다는 신호다. 이 경우에는 $\ln k$가 $E_a$의 선형 함수이므로 $\ln k$ 척도에서 선형화하면 정확하다. 어떤 척도에서 선형에 가까운지 먼저 생각하는 것이 좋은 습관이다.

**연구에서의 의미.** 학습된 모델을 시뮬레이션이나 다른 모델의 입력으로 연결할 때(예: 물성 예측 → 공정 시뮬레이션), 앞 단계의 예측 불확실성은 다음 단계의 입력 불확실성이 된다. 앞 단계가 앙상블이면 구성원마다 다음 단계를 돌리는 것이 가장 단순한 Monte Carlo 전파다. 이때 앞 단계의 aleatoric·epistemic을 모두 표본으로 넘겨야 한다. 평균만 넘기면 불확실성이 사라진다.

## 실습

데이터 밀도·관측 잡음·도메인 변화가 다른 다섯 영역에서 딥 앙상블과 MC dropout의 불확실성을 비교한다. 코드는 `labs/reliability/uncertainty-quantification/compare_uq.py`이고 PyTorch가 필요하다(CPU로 충분하다). 두 방법 모두 $(\mu, \log\sigma^2)$를 출력하는 MLP를 이분산 NLL로 학습하고, 평균을 먼저 MSE로 맞추는 warm-up과 로그 분산 범위 제한을 쓴다. 데이터는 $y = \sin x + \varepsilon$이고, 영역마다 데이터 수와 잡음 크기가 다르다.

| 영역 | $x$ 범위 | 학습 데이터 | 참 잡음 sd |
|---|---|---|---|
| 밀집·저잡음 | $[-3, -1]$ | 400개 | 0.05 |
| 밀집·고잡음 | $[1, 3]$ | 400개 | 0.40 |
| 희소 | $[3, 4]$ | 15개 | 0.05 |
| 빈 구간 | $[-1, 1]$ | 0개 | — |
| 도메인 밖 | $[5, 6]$ | 0개 | — |

앙상블은 무작위 초기값만 다른 MLP 5개, MC dropout은 dropout 비율 0.1인 MLP 하나에서 예측 때 50번 표본을 뽑는다. 영역 안 50개 점에서 평균 낸 aleatoric·epistemic 표준편차를 비교한다. 집필 시점의 실행 결과는 다음과 같다(시드 0, CPU 스레드 1개).

```output
영역별 평균 sd (aleatoric / epistemic),  참 잡음 sd
영역                     앙상블 5개    MC dropout T=50   참값
밀집·저잡음         0.046 /  0.000     0.077 /  0.058   0.05
밀집·고잡음         0.395 /  0.015     0.437 /  0.095   0.40
희소             0.265 /  0.041     0.554 /  0.103   0.05
빈 구간           0.131 /  0.046     0.281 /  0.077   (데이터 없음)
도메인 밖          0.226 /  0.245     0.705 /  0.092   (데이터 없음)
```

결과를 읽을 때 확인할 것은 세 가지다.

- **데이터가 많은 영역에서는 두 방법 모두 aleatoric을 대체로 맞힌다.** 앙상블(0.046, 0.395)이 참값(0.05, 0.40)에 더 가깝다. 앙상블의 epistemic은 거의 0인데, MC dropout의 epistemic은 데이터가 많은 곳에서도 0.058–0.095에 머문다. MC dropout의 퍼짐에는 dropout 비율이 정하는 바닥이 있어서, 데이터를 늘려도 그 아래로 줄지 않는다.
- **희소 영역에서는 두 방법 모두 불확실성의 출처를 잘못 분류한다.** 참 잡음은 0.05인데 aleatoric이 0.265(앙상블), 0.554(MC dropout)로 나온다. 바로 옆 고잡음 영역의 잡음 추정이 데이터가 적은 영역으로 번진 것이다. "데이터가 부족하다"는 epistemic 신호가 "잡음이 크다"는 aleatoric 신호로 둔갑했다. 두 불확실성이 관측에 함께 섞여 있다는 첫 절의 경고가 그대로 나타난다.
- **도메인 밖에서 차이가 가장 크다.** 앙상블의 epistemic(0.245)은 데이터가 많은 영역의 열 배가 넘어, 학습 범위를 벗어났다는 신호가 분명하다. MC dropout의 epistemic(0.092)은 데이터가 많은 고잡음 영역(0.095)과 구별되지 않는다. 대신 aleatoric이 0.705로 커졌는데, 이것만 보고는 "잡음이 큰 영역"과 "모르는 영역"을 구분할 수 없다. 빈 구간에서는 두 방법 모두 epistemic이 작다. 매끄러운 신경망이 짧은 빈틈을 비슷하게 메우기 때문이다.

이 결과는 이 데이터·이 구조·이 시드에서의 결과다. dropout 비율, 구성원 수, 신경망 폭을 바꾸면 숫자가 달라진다. 실습의 요점은 특정 방법의 우열이 아니라, **영역을 나눠 aleatoric·epistemic을 따로 보고해야 이런 실패가 보인다**는 것이다. 시드와 dropout 비율을 바꿔 결론이 유지되는지 확인한다.

이어서 연습문제를 푼다. 앙상블 구성원의 평균과 분산에서 두 불확실성을 계산하는 함수를 직접 쓴다.

::exercise{id="variance-decomposition"}

## 실패 사례

- **softmax 확률을 불확실성으로 그대로 쓰는 것.** 확률은 결정 경계로부터의 거리이지 학습 범위에 대한 정보가 아니다. 학습 데이터에서 먼 입력에도 0.999가 나온다. 보정(21.2장)과 OOD 점검(21.4장)을 따로 한다.
- **앙상블 크기가 너무 작은 것.** 구성원 2–3개로 계산한 표준편차는 작게 치우치고 크게 흔들린다. 불확실성의 크기를 수치로 보고하려면 구성원 수를 늘리거나, 적어도 크기에 따른 변화를 확인한다.
- **구성원 간 다양성이 없는 앙상블.** 같은 초기값, 같은 데이터 순서로 학습한 구성원은 거의 같은 모델이다. 무엇을 바꿔 구성원을 만들었는지 기록하고 보고한다.
- **분해 결과를 참값처럼 해석하는 것.** aleatoric·epistemic의 분리는 잡음 모델과 모델 클래스에 의존한다. 학습 범위 밖에서는 aleatoric 추정도 외삽이다.
- **입력 불확실성을 무시하는 것.** 측정·문헌 값의 오차를 전파하지 않으면 최종 예측 구간이 지나치게 좁아진다. 비선형 모델에서는 선형화 대신 Monte Carlo 전파를 쓴다.

## 통과 기준

- 자신의 모델(또는 실습 모델)의 예측 불확실성을 aleatoric과 epistemic으로 분해해, 데이터가 많은 영역·적은 영역·학습 범위 밖에서 각각 보고한다.
- 사용한 불확실성 방법이 구성원 간 다양성을 무엇으로 만들었는지(초기값·재표본·dropout 등)와 구성원 수를 밝힌다.
- 연습문제 `variance-decomposition`의 테스트를 통과한다.

## 참고 문헌

- Lakshminarayanan, Pritzel, Blundell (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. — deep ensemble과 적정 점수 규칙으로 학습하는 분포 head
- Gal, Ghahramani (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48, 1050–1059. — MC dropout과 근사 베이지안 해석
- Kendall, Gal (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. — aleatoric·epistemic을 함께 모델링하는 딥러닝 설정
- Maddox, Garipov, Izmailov, Vetrov, Wilson (2019). A simple baseline for Bayesian uncertainty in deep learning. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. — SWAG
