---
id: classical-ml/linear-models
title: "선형 모델과 정규화"
title_en: "Linear Models and Regularization"
tier: core
status: draft
prerequisites: [classical-ml/baselines, linear-algebra/inner-products-and-projection, statistical-inference/likelihood]
---

# 선형 모델과 정규화

선형 모델은 가장 오래되고 가장 많이 쓰이는 baseline이다. "선형"이라는 이름 때문에 직선밖에 못 맞춘다고 오해하기 쉽지만, 선형인 것은 입력이 아니라 **파라미터**다. 입력을 다항식·스플라인·도메인 지식으로 변환하면 선형 모델로도 곡선과 상호작용을 표현할 수 있고, 해는 여전히 7.4장의 사영 하나로 구해진다. 문제는 특징이 늘어날수록 과적합이 쉬워진다는 것이고, 이를 막는 표준 도구가 **정규화**(regularization)다.

12.6장은 선형모형을 계수 추정과 그 불확실성의 관점에서 다루었다. 이 장은 같은 모델을 **예측**의 관점에서 다룬다. 특징을 어떻게 만들고, 정규화 강도를 어떻게 고르고, 정규화된 계수를 어디까지 해석할 수 있는지가 주제다. 16.1장의 의미에서, 잘 튜닝된 정규화 선형 모델은 거의 모든 회귀·분류 문제에서 먼저 세워야 할 강한 baseline이다.

## 학습 목표

- 선형·로지스틱 회귀와 정규화를 사용한다
- 정규화를 기하·확률적으로 해석한다
- 정규화 강도를 교차검증으로 고르고, 특징 스케일과 선택의 안정성을 점검한 뒤 계수를 해석한다

## 선형회귀

**정의.** 입력 $\mathbf{x}$를 특징 벡터 $\boldsymbol\phi(\mathbf{x}) \in \mathbb{R}^p$로 바꾸고, 그 선형결합으로 출력을 예측한다.

$$
\hat y = \mathbf{w}^\top\boldsymbol\phi(\mathbf{x}) + b.
$$

$\boldsymbol\phi$는 항등함수일 수도($\boldsymbol\phi(\mathbf{x}) = \mathbf{x}$), 다항식 $(x, x^2, \dots, x^d)$일 수도, 스플라인 basis 함수일 수도 있다. 이를 **basis 확장**(basis expansion)이라 한다. $\boldsymbol\phi$가 무엇이든 모델은 $\mathbf{w}$에 대해 선형이므로, 학습 데이터의 특징을 행으로 쌓은 $N \times p$ 행렬 $\boldsymbol\Phi$에 대해 최소제곱 해는

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2 = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf{y}
$$

이다(절편은 $\boldsymbol\Phi$에 1의 열을 넣어 처리한다). 예측 $\boldsymbol\Phi\hat{\mathbf{w}}$는 $\mathbf{y}$를 $\boldsymbol\Phi$의 column space에 사영한 것이다(7.4장). 손실이 볼록한 이차식이라 국소 최솟값이 없고, 해가 닫힌 형태로 나온다. 이 두 성질이 선형 모델이 강한 baseline이 되는 이유다. 학습이 실패하는 경우가 거의 없고, 결과가 초기화나 seed에 의존하지 않는다.

**직관.** 특징을 늘리면 column space가 커져 학습 데이터에 더 가까이 붙을 수 있다. 특징 수 $p$가 데이터 수 $N$에 가까워지면 잡음까지 맞추기 시작한다. 감쇠 진동 $f(t) = e^{-t}\cos 4t$에서 측정한 25개 점에 네 가지 basis로 선형회귀를 맞춘다. 학습 구간 $[0, 2]$ s 안과, 학습 구간 밖 $[2, 2.5]$ s에서의 오차를 따로 잰다.

```python {exec="browser" session="lin"}
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

rng = np.random.default_rng(0)
f = lambda t: np.exp(-t) * np.cos(4 * t)                   # 감쇠 진동 (참 함수)
t_tr = np.sort(rng.uniform(0, 2, 25))[:, None]             # 학습 입력 [s], (25, 1)
y_tr = f(t_tr[:, 0]) + rng.normal(0, 0.05, 25)
t_in = np.linspace(0, 2, 200)[:, None]                     # 학습 구간 안, (200, 1)
t_out = np.linspace(2, 2.5, 50)[:, None]                   # 학습 구간 밖, (50, 1)
rmse = lambda m, t: np.sqrt(np.mean((m.predict(t) - f(t[:, 0])) ** 2))

for name, feat in [("원 특징 t", PolynomialFeatures(1)), ("다항 5차", PolynomialFeatures(5)),
                   ("다항 12차", PolynomialFeatures(12)), ("스플라인", SplineTransformer(n_knots=11))]:
    m = make_pipeline(StandardScaler(), feat, LinearRegression()).fit(t_tr, y_tr)
    print(f"특징 {m[-1].coef_.size:2d}개  구간 안 RMSE {rmse(m, t_in):.3f}  구간 밖 RMSE {rmse(m, t_out):7.2f}  {name}")
```

```output
특징  2개  구간 안 RMSE 0.413  구간 밖 RMSE    0.08  원 특징 t
특징  6개  구간 안 RMSE 0.055  구간 밖 RMSE    2.81  다항 5차
특징 13개  구간 안 RMSE 0.475  구간 밖 RMSE 2072.26  다항 12차
특징 13개  구간 안 RMSE 0.047  구간 밖 RMSE    0.19  스플라인
```

- **원 특징 $t$**(직선)는 곡선을 표현하지 못한다(구간 안 0.413). 구간 밖 오차가 작은 것은 그곳에서 참 함수가 0 근처라 직선이 우연히 가깝기 때문이다.
- **다항 5차**는 구간 안에서 잘 맞는다(0.055). **다항 12차**는 파라미터 13개로 점 25개를 맞추며 잡음까지 따라가 구간 안 오차가 오히려 커졌다(0.475).
- **스플라인**은 같은 13개 특징인데도 가장 정확하다(0.047). 스플라인 basis 함수는 각자 좁은 구간에서만 0이 아니어서, 한 곳의 잡음이 먼 곳의 예측을 흔들지 않는다. 반면 다항식 basis $t^k$는 모든 곳에서 0이 아니어서 계수 하나가 곡선 전체를 움직인다.
- **구간 밖**에서 다항식은 발산한다(12차에서 RMSE 2000 이상). 다항식은 $|t|$가 커지면 최고차항이 지배하기 때문이다. 이 스플라인은 구간 밖에서 경계값을 상수로 연장하도록 설정되어 있어 발산하지 않지만, 외삽을 잘하는 것도 아니다.

다항 12차의 실패에는 두 가지 처방이 있다. 특징을 줄이거나(모델 선택), 특징은 두고 계수의 크기에 벌점을 주는 것이다(정규화). 두 번째가 이 장의 중심이다.

**연구에서의 의미.** basis 확장은 "선형 모델 대 비선형 모델"이라는 구분을 흐린다. 선형 모델의 표현력은 특징에서 나오고, 좋은 특징은 대개 도메인 지식에서 나온다(뒤의 특징 공학 절). 수치 계산의 주의점 하나도 기억해 둔다. 이 셀은 다항 특징을 만들기 전에 $t$를 표준화했다. 원래 척도의 $t \in [0, 2]$로 12차 다항식을 만들면 $\boldsymbol\Phi^\top\boldsymbol\Phi$의 조건수가 매우 커져, 같은 코드가 플랫폼마다 다른 계수를 낼 수 있다(9.2장).

## 정규화

**왜 필요한가.** 특징이 많거나 서로 강하게 상관되어 있으면 $\boldsymbol\Phi^\top\boldsymbol\Phi$가 특이행렬에 가까워지고, 최소제곱 계수는 잡음에 크게 흔들린다. 큰 양수 계수와 큰 음수 계수가 서로 상쇄하며 학습 데이터에 맞추는 해가 나온다. 정규화는 계수가 커지는 것 자체에 벌점을 주어 이런 해를 막는다.

**정의.** 정규화된 최소제곱은 손실에 계수의 노름을 더한다. 세 가지가 표준이다.

$$
\begin{aligned}
\text{Ridge:} &\quad \min_{\mathbf{w}} \lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2 + \lambda \lVert \mathbf{w} \rVert_2^2 \\
\text{Lasso:} &\quad \min_{\mathbf{w}} \tfrac{1}{2N}\lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2 + \alpha \lVert \mathbf{w} \rVert_1 \\
\text{Elastic net:} &\quad \min_{\mathbf{w}} \tfrac{1}{2N}\lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2 + \alpha\rho \lVert \mathbf{w} \rVert_1 + \tfrac{\alpha(1-\rho)}{2} \lVert \mathbf{w} \rVert_2^2
\end{aligned}
$$

$\lambda, \alpha \ge 0$는 정규화 강도, $\rho \in [0, 1]$은 L1과 L2의 비율이다. 이 표기는 scikit-learn의 `Ridge(alpha=λ)`, `Lasso(alpha=α)`, `ElasticNet(alpha=α, l1_ratio=ρ)`를 따랐다. Lasso와 Elastic net은 제곱오차를 $2N$으로 나누므로 같은 숫자의 `alpha`라도 Ridge와 강도가 다르다. 세 경우 모두 **절편 $b$에는 벌점을 주지 않는다.** 출력의 평균 수준을 0쪽으로 당길 이유가 없기 때문이다.

Ridge는 닫힌 해를 가진다.

$$
\hat{\mathbf{w}}_{\text{ridge}} = (\boldsymbol\Phi^\top\boldsymbol\Phi + \lambda\mathbf{I})^{-1}\boldsymbol\Phi^\top\mathbf{y}.
$$

$\lambda > 0$이면 $\boldsymbol\Phi^\top\boldsymbol\Phi + \lambda\mathbf{I}$는 항상 가역이다. 특징이 데이터보다 많아도($p > N$) 해가 하나로 정해진다. Lasso는 $\lVert\mathbf{w}\rVert_1$이 0에서 미분 불가능해 닫힌 해가 없고, 좌표 하강법(coordinate descent) 같은 반복법으로 푼다. 그 대신 일부 계수를 **정확히 0**으로 만든다. Lasso라는 이름은 least absolute shrinkage and selection operator의 약자다(Tibshirani, 1996).

**기하적 해석.** 벌점 형태는 같은 해를 주는 제약 형태 $\min \lVert \mathbf{y} - \boldsymbol\Phi\mathbf{w} \rVert^2$ s.t. $\lVert\mathbf{w}\rVert \le t$로 바꿔 쓸 수 있다(10.2장의 Lagrange 승수). 손실의 등고선은 최소제곱 해를 중심으로 한 타원이고, 해는 타원이 제약 영역에 처음 닿는 점이다.

- L2 제약 영역은 **원**(고차원에서는 구)이다. 경계가 매끄러워서 타원이 축 위에서 닿을 특별한 이유가 없다. 계수는 줄지만 0이 되지 않는다.
- L1 제약 영역은 **마름모**(고차원에서는 꼭짓점이 축 위에 있는 다면체)다. 타원은 꼭짓점에 먼저 닿기 쉽고, 꼭짓점에서는 일부 좌표가 정확히 0이다.

Ridge의 줄이는 방식은 SVD(7.6장)로 정확히 볼 수 있다. $\boldsymbol\Phi = \mathbf{U}\mathbf{D}\mathbf{V}^\top$이면 Ridge 예측은 $\sum_j \mathbf{u}_j \frac{d_j^2}{d_j^2 + \lambda}\mathbf{u}_j^\top\mathbf{y}$다. 최소제곱은 모든 주성분 방향의 성분을 그대로 쓰고, Ridge는 특잇값 $d_j$가 작은 방향, 즉 데이터가 거의 퍼져 있지 않아 추정이 불안정한 방향의 성분을 강하게 줄인다.

**확률적 해석.** 12.2장에서 본 것처럼 정규화된 손실은 MAP 추정이다. 잡음이 $\mathcal{N}(0, \sigma^2)$이고 계수의 사전분포가 $\mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$이면 Ridge의 $\lambda = \sigma^2/\tau^2$이다. 계수의 사전분포가 Laplace 분포이면 Lasso가 된다. Laplace 분포는 0에 뾰족한 봉우리가 있어 "대부분의 계수는 0이거나 0에 가깝다"는 믿음을 표현한다. 이 해석은 정규화 강도에 의미를 준다. $\lambda$를 키우는 것은 "계수가 클 수 없다"는 사전 믿음을 강하게 하는 것이다.

::quiz{id="ridge-prior-lambda"}

**정규화 경로.** 강도를 큰 값에서 작은 값으로 줄이며 계수가 어떻게 변하는지 그린 것을 **정규화 경로**(regularization path)라 한다. 당뇨병 환자 442명의 기초 측정 10개(나이, 성별, BMI, 혈압 bp, 혈청 측정 s1–s6)로 1년 뒤 질병 진행 지표를 예측하는 데이터(Efron 외, 2004; scikit-learn에 포함)에서 확인한다. 특징은 표준화했다.

```python {exec="browser" session="lin"}
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge

data = load_diabetes()
names = np.array(data.feature_names)
X = (data.data - data.data.mean(0)) / data.data.std(0)     # 표준화한 특징, (442, 10)
y = data.target                                            # 1년 뒤 질병 진행 지표, (442,)
for alpha in [40.0, 20.0, 10.0, 3.0, 0.3]:
    w_l = Lasso(alpha=alpha).fit(X, y).coef_               # (10,)
    w_r = Ridge(alpha=2 * len(y) * alpha).fit(X, y).coef_  # 같은 척도의 L2 벌점
    print(f"alpha {alpha:4.1f}  0 아닌 계수 Lasso {np.sum(w_l != 0):2d}개, Ridge {np.sum(w_r != 0):2d}개  "
          f"Lasso 선택: {', '.join(names[w_l != 0])}")
```

```output
alpha 40.0  0 아닌 계수 Lasso  2개, Ridge 10개  Lasso 선택: bmi, s5
alpha 20.0  0 아닌 계수 Lasso  3개, Ridge 10개  Lasso 선택: bmi, bp, s5
alpha 10.0  0 아닌 계수 Lasso  4개, Ridge 10개  Lasso 선택: bmi, bp, s3, s5
alpha  3.0  0 아닌 계수 Lasso  7개, Ridge 10개  Lasso 선택: sex, bmi, bp, s1, s3, s5, s6
alpha  0.3  0 아닌 계수 Lasso  8개, Ridge 10개  Lasso 선택: sex, bmi, bp, s1, s3, s4, s5, s6
```

강한 정규화에서 Lasso는 BMI와 s5 두 개만 남기고, 강도를 줄이면 bp, s3, 그리고 나머지가 차례로 들어온다. Ridge는 같은 범위에서 계수 10개를 모두 0이 아닌 값으로 둔다. Lasso가 특징을 들여보내는 순서는 "이 데이터에서 예측에 가장 빨리 기여하는 특징"의 순서이고, 특징 선택의 출발점이 된다. 다만 그 순서를 "중요도"로 읽을 때의 함정은 아래에서 본다.

::quiz{id="lasso-exact-zeros"}

**Elastic net.** Lasso는 강하게 상관된 특징들 중 하나만 고르는 경향이 있다. 이 데이터에서 s1(총 콜레스테롤)과 s2(LDL)의 상관계수는 0.90이다. 어느 하나를 고를지는 데이터의 작은 변화로 바뀔 수 있다. Elastic net은 L2 항을 섞어 상관된 특징들이 함께 들어오고 함께 나가게 만든다(Zou & Hastie, 2005). 과학 데이터에서 같은 현상을 여러 센서·여러 지표가 중복해서 재는 경우가 흔하므로, 해석이 목적이면 Elastic net을 함께 시도한다.

**특징 스케일.** 벌점 $\lVert\mathbf{w}\rVert$는 모든 계수를 같은 척도로 본다. 그런데 계수의 크기는 특징의 단위에 따라 바뀐다. 길이를 m에서 mm로 바꾸면 같은 효과를 내는 계수가 1000분의 1로 줄고, 벌점도 그만큼 덜 받는다. 즉 **표준화하지 않은 정규화는 측정 단위에 따라 다른 모델을 낸다.** 같은 당뇨병 데이터를 원래 단위(나이는 년, 혈압은 값의 범위로 보아 mmHg)로 다시 불러, bp만 kPa로 환산했을 때 Lasso의 선택이 어떻게 바뀌는지 본다.

```python {exec="browser" session="lin"}
raw = load_diabetes(scaled=False).data                     # 원래 단위(나이 [년], 혈압 [mmHg] ...), (442, 10)
raw_kpa = raw.copy()
raw_kpa[:, 3] *= 0.1333                                    # 혈압 bp: mmHg → kPa
for label, Xr in [("원래 단위", raw), ("bp를 kPa로", raw_kpa)]:
    w = Lasso(alpha=10.0).fit(Xr, y).coef_
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=10.0)).fit(Xr, y)
    print(f"[{label}] 표준화 없이: {', '.join(names[w != 0])} / 표준화 후: {', '.join(names[pipe[-1].coef_ != 0])}")
```

```output
[원래 단위] 표준화 없이: bmi, bp, s1, s2, s3, s6 / 표준화 후: bmi, bp, s3, s5
[bp를 kPa로] 표준화 없이: age, bmi, bp, s1, s2, s3, s6 / 표준화 후: bmi, bp, s3, s5
```

표준화 없이 적합한 Lasso는 표준화하면 가장 먼저 들어오는 두 특징 중 하나인 s5를 버린다. s5는 원래 단위에서 표준편차가 0.52로 작아서, 같은 효과를 내려면 큰 계수가 필요하고 그만큼 큰 벌점을 받기 때문이다. bp의 단위를 바꾸기만 해도 age가 새로 선택된다. 측정 단위는 물리적으로 아무 의미 없는 선택인데 모델이 바뀐 것이다. 표준화한 뒤에는 두 단위에서 선택이 같다. 정규화 모델은 `StandardScaler`와 함께 pipeline으로 묶어, 표준화 통계량을 학습 데이터에서만 추정하게 한다(15.4장의 "fit은 학습 데이터에서만").

**강도 선택과 안정성.** 정규화 강도는 하이퍼파라미터다. 학습 데이터 안에서 교차검증(16.6장)으로 고른다. `LassoCV`는 강도의 격자 전체에서 경로를 계산하고 fold별 검증 오차가 가장 작은 강도를 고른다. 고른 뒤에는 **선택된 특징 집합이 얼마나 안정적인지** 확인한다. 부트스트랩 표본마다 같은 강도로 다시 적합해, 각 특징이 선택되는 비율을 센다.

```python {exec="browser" session="lin"}
from sklearn.linear_model import LassoCV

cv_model = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(raw, y)
alpha_cv = cv_model[-1].alpha_
print(f"CV로 고른 alpha = {alpha_cv:.2f}, 선택된 특징: {', '.join(names[cv_model[-1].coef_ != 0])}")
boot_rng = np.random.default_rng(0)
freq = np.zeros(10)
for _ in range(200):                                       # 부트스트랩 표본마다 같은 alpha로 다시 적합
    i = boot_rng.integers(0, len(y), len(y))
    freq += make_pipeline(StandardScaler(), Lasso(alpha=alpha_cv)).fit(raw[i], y[i])[-1].coef_ != 0
print("부트스트랩 200번 중 선택 비율:", "  ".join(f"{n} {p:.2f}" for n, p in zip(names, freq / 200)))
```

```output
CV로 고른 alpha = 0.08, 선택된 특징: age, sex, bmi, bp, s1, s2, s4, s5, s6
부트스트랩 200번 중 선택 비율: age 0.95  sex 1.00  bmi 1.00  bp 1.00  s1 0.98  s2 0.89  s3 0.87  s4 0.97  s5 1.00  s6 0.98
```

교차검증은 약한 정규화를 골랐고 특징 9개가 선택되었다. 전체 데이터에서 빠진 s3는 부트스트랩 표본의 87%에서 선택되었고, 선택된 s2도 89%에 그친다. 이 특징들의 포함 여부는 표본의 작은 변화로 바뀐다는 뜻이다. s1–s4는 콜레스테롤 관련 지표로, s1과 s2(상관계수 0.90), s3과 s4(−0.74)처럼 강하게 상관된 쌍이 있어서 그중 무엇을 쓰든 예측은 비슷하다. 그래서 "Lasso가 s3를 버렸으므로 HDL은 질병 진행과 무관하다"는 해석은 근거가 없다. 계수를 해석하려면 선택 비율 같은 안정성 정보를 함께 보고한다.

:::note[분야 예시]
분광학에서 시료의 농도를 수백–수천 개 파장의 흡광도로 예측하는 검량 문제는 특징 수가 시료 수보다 많고($p > N$), 이웃한 파장의 흡광도가 거의 같은 정보를 담아 강하게 상관된다. 최소제곱은 해가 하나로 정해지지 않고, Ridge나 주성분 기반 회귀처럼 상관된 방향을 줄이는 방법이 표준이다. Lasso로 "중요한 파장"을 고르면 이웃 파장 중 하나가 임의로 뽑히므로, 고른 파장을 분자 구조의 흡수 띠와 연결하는 해석에는 조심해야 한다.
:::

## 로지스틱 회귀

**왜 필요한가.** 출력이 0/1(파손 여부, 합성 성공 여부, 질병 유무)이면 최소제곱 선형 모델은 0–1 밖의 값을 내고, 오차 분포 가정도 맞지 않는다. 필요한 것은 확률을 내는 선형 모델이다.

**정의.** 로지스틱 회귀(logistic regression)는 선형 예측자를 sigmoid 함수로 확률로 바꾼다.

$$
p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

$\mathbf{w}^\top\mathbf{x} + b$는 log-odds $\log\frac{p}{1-p}$다. 12.6장에서 본 Bernoulli 분포와 logit link의 GLM이다. 학습은 Bernoulli NLL, 즉 cross-entropy를 최소화한다.

$$
L(\mathbf{w}, b) = -\sum_{i=1}^N \big[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \big] + \frac{1}{2C}\lVert\mathbf{w}\rVert^2.
$$

scikit-learn의 `LogisticRegression`은 **기본으로 L2 정규화를 건다.** 강도는 $C$의 역수로 정하므로 $C$가 클수록 약하다. 정규화는 로지스틱 회귀에서 특히 중요하다. 학습 데이터가 어떤 초평면으로 완전히 분리되면, 정규화 없는 최대우도 해는 존재하지 않는다. $\mathbf{w}$를 키울수록 모든 점의 확률이 0이나 1에 가까워져 NLL이 계속 줄기 때문이다.

**결정 경계와 threshold.** 확률을 0/1 판정으로 바꾸려면 threshold $c$가 필요하다. $p \ge c$이면 1로 판정한다. 결정 경계 $\{\mathbf{x} : \sigma(\mathbf{w}^\top\mathbf{x} + b) = c\}$는 $\mathbf{w}^\top\mathbf{x} + b = \log\frac{c}{1-c}$인 **초평면**이다. 그래서 로지스틱 회귀는 선형 분류기다. 확률을 매기는 모델과 판정 규칙은 별개이고, threshold는 확률을 낸 다음에 결정 비용으로 고른다. 놓친 양성(false negative)의 비용이 $c_{\text{FN}}$, 잘못된 경보(false positive)의 비용이 $c_{\text{FP}}$이면, 확률 $p$인 시편에 경보를 낼 때의 기대 비용 $(1 - p)c_{\text{FP}}$가 경보를 내지 않을 때의 $p\,c_{\text{FN}}$보다 작을 조건에서

$$
c^* = \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}}
$$

를 얻는다. 비용이 같으면 0.5이고, 놓친 파손이 잘못된 경보보다 9배 비싸면 0.1이다. 이 식은 확률이 보정(calibration)되어 있을 때, 즉 "확률 0.2"라고 말한 시편 중 실제로 약 20%가 파손될 때 성립한다.

::quiz{id="threshold-from-costs"}

금속 시편 600개의 피로 시험을 흉내 낸다. 응력 진폭이 크고 온도가 높을수록 파손 확률이 높고, 참 모형에서 70 °C의 결정 경계(파손 확률 0.5)는 330 MPa이다. 400개로 학습하고 200개로 평가한다.

```python {exec="browser" session="lin"}
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(3)
stress_mpa = rng.uniform(200, 400, 600)                    # 응력 진폭 [MPa], (600,)
temp_c = rng.uniform(20, 120, 600)                         # 시험 온도 [°C], (600,)
logit = 0.05 * (stress_mpa - 330) + 0.03 * (temp_c - 70)   # 파손의 log-odds (참 모형)
failed = rng.random(600) < 1 / (1 + np.exp(-logit))        # 파손 여부, (600,)
Xf = np.column_stack([stress_mpa, temp_c])                 # (600, 2)
clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xf[:400], failed[:400])
p = clf.predict_proba(Xf[400:])[:, 1]                      # 시험 시편의 파손 확률, (200,)
yt = failed[400:]
for thr in (0.5, 0.2):
    alarm = p >= thr
    print(f"threshold {thr}: 실제 파손 중 경보 비율 {alarm[yt].mean():.2f}, 경보 중 실제 파손 비율 {yt[alarm].mean():.2f}")
w = clf[-1].coef_[0] / clf[0].scale_                       # 원래 단위의 계수 [1/MPa, 1/°C]
b = clf[-1].intercept_[0] - w @ clf[0].mean_
print(f"70 °C에서 결정 경계(p = 0.5) 응력 = {-(b + w[1] * 70) / w[0]:.0f} MPa")
```

```output
threshold 0.5: 실제 파손 중 경보 비율 0.82, 경보 중 실제 파손 비율 0.79
threshold 0.2: 실제 파손 중 경보 비율 0.93, 경보 중 실제 파손 비율 0.68
70 °C에서 결정 경계(p = 0.5) 응력 = 325 MPa
```

threshold를 0.5에서 0.2로 내리면 실제 파손 중 경보를 받은 비율(recall)은 0.82에서 0.93으로 오르고, 경보 중 실제 파손 비율(precision)은 0.79에서 0.68로 내려간다. 어느 쪽이 나은지는 모델이 아니라 비용이 정한다. 표준화된 계수를 원래 단위로 되돌리면 결정 경계를 물리적인 양으로 읽을 수 있다. 70 °C에서 325 MPa로, 참값 330 MPa에 가깝다. 분류 metric과 threshold 선택은 16.6장에서 다시 다룬다.

## 특징 공학

선형 모델의 성능은 대부분 특징에서 결정된다. 특징 공학(feature engineering)의 표준 도구는 세 가지이고, 과학 데이터에서는 네 번째가 가장 강력하다.

- **다항 특징.** $x, x^2, \dots, x^d$. 매끄러운 곡률을 표현하지만, 앞에서 보았듯 차수가 높으면 구간 안에서도 진동하고 구간 밖에서 발산한다. 입력을 표준화한 뒤 만들고, 차수는 3 이하에서 시작한다.
- **상호작용 특징.** $x_1 x_2$처럼 두 특징의 곱. "온도가 높을 때만 응력의 효과가 커진다"처럼 한 특징의 효과가 다른 특징의 값에 따라 달라지는 관계를 표현한다. 특징이 $p$개면 쌍별 상호작용은 $p(p-1)/2$개라 빠르게 늘어나므로 정규화와 함께 쓴다.
- **스플라인 특징.** 구간을 매듭(knot)으로 나누고 구간마다 저차 다항식을 이어 붙인다. B-스플라인 basis 함수는 몇 개의 이웃 구간에서만 0이 아니어서, 앞의 실험처럼 같은 특징 수의 다항식보다 안정적이다. 매듭 수가 유연성을 정하고, 매듭이 많으면 Ridge로 계수를 매끄럽게 묶는다.
- **도메인 특징.** 물리 법칙이 알려 주는 변환. 반응 속도의 온도 의존성(Arrhenius 식)은 $\log k$가 $1/T$에 선형이므로, $T$ 대신 $1/T$를, $k$ 대신 $\log k$를 쓰면 직선 하나로 맞는다. 무차원수(Reynolds 수 등), 비율, 로그 변환도 같은 역할을 한다. 좋은 도메인 특징 하나가 고차 다항식 수십 개보다 정확하고 외삽에도 강하다.

:::note[분야 예시]
기후 자료로 지역 기온을 회귀할 때, 연중 계절 주기는 날짜 $d$를 그대로 넣는 대신 $\sin(2\pi d/365.25)$, $\cos(2\pi d/365.25)$ 두 특징으로 넣는다. 12월 31일과 1월 1일이 특징 공간에서 가깝게 놓이고, 진폭과 위상이 계수 두 개로 표현된다. 이것도 basis 확장(Fourier basis)이다. 반주기, 1/3주기 항을 더하면 더 복잡한 계절 모양을 표현한다.
:::

특징 공학에는 누수의 위험이 따른다. 표준화 통계량, 스플라인 매듭 위치, 특징 선택은 모두 데이터에서 추정하는 것이므로 학습 fold 안에서만 추정해야 한다. scikit-learn의 pipeline으로 특징 변환과 모델을 묶으면 교차검증이 fold마다 이를 다시 수행한다(15.4장).

## 실습

Ridge와 Lasso의 정규화 경로를 그리고 계수 선택 결과를 해석한다. 앞의 당뇨병 데이터에 대해 다음을 진행한다.

1. `sklearn.linear_model.lasso_path`와 Ridge의 강도 격자로, 강도($\log$ 척도)를 가로축, 계수를 세로축으로 한 경로 그림 두 장을 그린다(`matplotlib`). 같은 그림에 `LassoCV`가 고른 강도를 세로선으로 표시한다.
2. s1과 s2(상관계수 0.90)의 경로를 비교한다. Lasso에서 둘 중 하나가 먼저 들어오는 구간이 있는지, Ridge에서는 어떻게 움직이는지 설명한다. `ElasticNet(l1_ratio=0.5)`의 경로도 그려 비교한다.
3. 부트스트랩 선택 비율이 0.9 미만인 특징을 표시하고, 그 특징들의 계수를 "효과 크기"로 해석할 수 있는지 한 문단으로 쓴다.

이어서 두 연습문제를 푼다. 첫 문제는 표준화와 벌점 없는 절편을 포함한 Ridge 회귀를 NumPy로 구현하고, 두 번째 문제는 Lasso가 계수를 정확히 0으로 만드는 이유인 soft-thresholding을 좌표 하강법으로 구현한다.

::exercise{id="standardized-ridge"}

::exercise{id="lasso-coordinate-descent"}

## 실패 사례

- **표준화 없이 정규화를 적용하는 것.** 벌점은 계수의 크기를 보는데 계수의 크기는 단위에 따라 바뀐다. 측정 단위를 바꾸기만 해도 Lasso의 선택이 바뀐다(앞의 실험). 표준화를 pipeline에 넣고 학습 데이터에서만 추정한다. 반대로 원-핫 인코딩한 범주형 특징처럼 척도에 의미가 있는 특징은 표준화 여부를 의식적으로 정한다.
- **절편에 벌점을 주는 것.** 직접 구현한 Ridge에서 1의 열에도 $\lambda$를 더하면 예측 전체가 0쪽으로 끌려간다. 출력의 평균이 큰 문제(예: 2.0 Ah 근처의 용량)에서 특히 크게 틀린다.
- **Lasso가 고른 특징을 "진짜 원인", 버린 특징을 "무관한 변수"로 해석하는 것.** 상관된 특징 중 하나는 임의로 버려진다. 선택은 예측을 위한 것이지 인과 판단이 아니다(16.8장). 선택 비율과 상관 구조를 함께 보고한다.
- **정규화 강도를 시험 데이터로 고르는 것.** 시험 점수가 낙관적으로 편향된다. 교차검증 또는 별도 검증 세트로 고른다.
- **로지스틱 회귀의 기본 정규화를 모르고 계수를 보고하는 것.** scikit-learn의 기본값은 $C = 1$의 L2 정규화다. 12.6장처럼 계수의 표준오차와 구간을 해석하려면 정규화 없는 MLE를 쓰거나 정규화 강도를 명시한다.

## 통과 기준

- 정규화 강도를 교차검증으로 고르고(표준화를 포함한 pipeline), 선택된 계수의 부호와 크기를 해석하면서 부트스트랩 선택 비율로 그 해석의 안정성을 함께 보고한다.
- Ridge와 Lasso의 차이를 제약 영역의 기하(원과 마름모)와 사전분포(가우시안과 Laplace)로 설명한다.
- 연습문제 두 개(`standardized-ridge`, `lasso-coordinate-descent`)의 테스트를 통과한다.

## 참고 문헌

- Hoerl, Kennard (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. — $\mathbf{X}^\top\mathbf{X}$의 대각에 양수를 더하는 ridge 추정량과 ridge trace
- Tibshirani (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288. — 계수의 절댓값 합 제약으로 축소와 선택을 함께 하는 lasso
- Zou, Hastie (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320. — 상관된 특징을 함께 선택하는 elastic net
- Efron, Hastie, Johnstone, Tibshirani (2004). Least angle regression. *Annals of Statistics*, 32(2), 407–499. — LARS 알고리즘과 lasso 경로. 이 장의 당뇨병 데이터의 출처
