---
id: classical-ml/kernels-and-neighbors
title: "커널 방법과 최근접 이웃"
title_en: "Kernel Methods and Nearest Neighbors"
tier: core
status: draft
prerequisites: [classical-ml/linear-models, linear-algebra/function-spaces]
---

# 커널 방법과 최근접 이웃

16.2장의 선형 모델은 특징을 사람이 정해야 했다. 어떤 다항식, 어떤 스플라인, 어떤 도메인 변환을 쓸지 고르는 일이 모델 성능의 대부분을 결정했다. 이 장의 방법들은 다른 길을 택한다. 특징을 직접 만드는 대신 **두 입력이 얼마나 비슷한가**를 정의하고, 예측을 "비슷한 학습 데이터의 출력"으로 만든다. 최근접 이웃(k-nearest neighbors, kNN)은 이 생각을 가장 직접적으로 구현하고, 커널 방법은 같은 생각을 내적과 정규화의 언어로 정리해 SVM, 커널 릿지 회귀, Gaussian process(GP)로 이어진다.

이 방법들은 데이터가 수백–수천 개이고 입력 차원이 낮은 과학 문제에서 지금도 강한 baseline이다. 동시에 두 가지 약점이 분명하다. "비슷하다"의 정의가 특징의 단위와 차원에 민감하고, 학습 데이터 수 $N$에 대해 계산 비용이 빠르게 커진다. 7.9장에서 커널, RKHS, 표현자 정리를 함수공간의 관점에서 보았다면, 이 장은 그것을 예측 도구로 쓰는 법과 그 한계를 다룬다.

## 학습 목표

- kNN·SVM·커널 릿지 회귀를 사용하고 각 방법이 거리와 유사도에 거는 가정을 설명한다
- 커널과 길이 척도의 선택이 예측에 담는 가정을 이해한다
- 데이터 크기와 차원에 따라 계산 비용을 따지고 근사 방법을 고른다

## 최근접 이웃

**정의.** kNN 회귀는 새 입력 $\mathbf{x}$에서 가장 가까운 학습 점 $k$개의 집합 $\mathcal{N}_k(\mathbf{x})$를 찾아 그 출력의 평균을 예측으로 쓴다.

$$
\hat y(\mathbf{x}) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i.
$$

분류에서는 다수결을 쓴다. 학습 단계에서 하는 일이 없고(데이터를 저장할 뿐이다), 가정은 하나뿐이다. **입력 공간에서 가까운 점은 출력도 비슷하다.** $k$는 유연성을 정한다. $k = 1$이면 학습 데이터를 그대로 외우고, $k = N$이면 16.1장의 평균 baseline이 된다. 그 사이의 $k$는 교차검증으로 고른다.

**거리와 스케일.** "가깝다"는 거리로 정의되고, 보통의 Euclid 거리 $\lVert\mathbf{x} - \mathbf{x}'\rVert = \sqrt{\sum_j (x_j - x'_j)^2}$는 모든 특징을 같은 무게로 더한다. 그런데 특징의 단위가 다르면 숫자의 크기가 다르고, 숫자가 큰 특징이 거리를 지배한다. 반응 속도를 농도(0.01–0.2 mol/L)와 온도(280–360 K)로 예측하는 문제에서 확인한다. 속도는 농도의 제곱에 비례하고 온도에는 Arrhenius 식처럼 지수적으로 의존한다.

```python {exec="browser" session="kern"}
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n = 600
conc_mol_l = rng.uniform(0.01, 0.2, n)                     # 농도 [mol/L]
temp_k = rng.uniform(280, 360, n)                          # 온도 [K]
rate = 50 * conc_mol_l**2 * np.exp(-2000 * (1 / temp_k - 1 / 320))   # 반응 속도 (2차 반응 + Arrhenius)
rate += rng.normal(0, 0.02, n)                             # 측정 잡음
X = np.column_stack([conc_mol_l, temp_k])                  # (600, 2)
tr, te = slice(0, 400), slice(400, None)
for label, model in [("표준화 없이", KNeighborsRegressor(10)),
                     ("표준화 후", make_pipeline(StandardScaler(), KNeighborsRegressor(10)))]:
    model.fit(X[tr], rate[tr])
    print(f"{label}: 시험 R² {model.score(X[te], rate[te]):.3f}")
```

```output
표준화 없이: 시험 R² 0.176
표준화 후: 시험 R² 0.991
```

표준화 없이 계산한 거리에서 농도 차이는 최대 0.19, 온도 차이는 최대 80이다. 거리는 사실상 온도만 보고, 가장 중요한 특징인 농도는 무시된다. 표준화하면 두 특징이 같은 척도에 놓여 $R^2$가 0.176에서 0.991로 오른다. 16.2장에서 정규화 벌점이 단위를 탔던 것과 같은 문제다. 거리 기반 방법은 표준화를 pipeline에 넣고, 학습 데이터의 통계량으로만 표준화한다.

표준화는 "모든 특징이 같은 정도로 중요하다"는 가정이기도 하다. 중요하지 않은 특징도 거리에 같은 무게로 들어간다. 이것이 다음 문제로 이어진다.

**차원의 저주.** 입력 차원이 높아지면 "가까운 이웃"이라는 개념 자체가 무너진다. 단위 초입방체 $[0, 1]^d$에 데이터가 고르게 퍼져 있을 때, 데이터의 비율 $r$을 담는 작은 초입방체의 한 변 길이는 $r^{1/d}$이다. 데이터의 1%를 이웃으로 모으려면 $d = 2$에서는 한 변이 0.1이면 되지만, $d = 10$에서는 $0.01^{1/10} \approx 0.63$, 즉 각 축 범위의 63%를 덮어야 한다. 이웃이 더 이상 국소적이지 않다. 또한 차원이 높으면 한 점에서 가장 가까운 점과 가장 먼 점까지의 거리가 비슷해진다(Beyer 외, 1999). 출력이 앞의 두 특징에만 의존하고 나머지는 무관한 잡음 특징인 데이터에서, 잡음 특징을 늘려 가며 kNN과 선형 모델을 비교한다.

```python {exec="browser" session="kern"}
from sklearn.linear_model import RidgeCV

def task(n, d_noise, rng):                                 # 출력은 앞의 두 특징에만 의존
    X = rng.uniform(-1, 1, (n, 2 + d_noise))               # (n, 2 + d_noise)
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, n)
    return X, y

for d_noise in (0, 4, 16, 64):
    r = np.random.default_rng(1)
    Xa, ya = task(500, d_noise, r)
    Xb, yb = task(500, d_noise, r)
    knn = KNeighborsRegressor(10, algorithm="kd_tree").fit(Xa, ya).score(Xb, yb)
    lin = RidgeCV().fit(Xa, ya).score(Xb, yb)
    d = np.linalg.norm(Xb[:1] - Xa, axis=1)                # 시험점 하나에서 학습점까지 거리, (500,)
    print(f"차원 {2 + d_noise:2d}: 최근접/최원 거리 비 {d.min() / d.max():.2f}  kNN R² {knn:.2f}  선형 R² {lin:.2f}")
```

```output
차원  2: 최근접/최원 거리 비 0.06  kNN R² 0.96  선형 R² 0.56
차원  6: 최근접/최원 거리 비 0.21  kNN R² 0.76  선형 R² 0.54
차원 18: 최근접/최원 거리 비 0.42  kNN R² 0.41  선형 R² 0.57
차원 66: 최근접/최원 거리 비 0.69  kNN R² 0.19  선형 R² 0.50
```

잡음 특징이 0개일 때 kNN의 $R^2$는 0.96으로, 비선형 관계를 표현하지 못하는 선형 모델(0.56)보다 훨씬 낫다. 잡음 특징이 늘면 최근접/최원 거리 비가 0.06에서 0.69로 커지고, kNN은 0.19까지 떨어져 선형 모델보다 나빠진다. 선형 모델은 무관한 특징의 계수를 0 근처로 두면 되므로 거의 영향을 받지 않는다. 데이터 수가 같다면, 무관한 차원이 늘수록 국소적인 방법이 먼저 무너진다.

::quiz{id="knn-irrelevant-features"}

**연구에서의 의미.** kNN은 입력이 저차원이고 의미 있는 거리가 있을 때 강한 baseline이다. 차원이 높으면 특징 선택, 도메인 지식으로 만든 저차원 표현, 또는 학습된 표현(embedding) 위의 거리를 쓴다. 표현 학습 뒤 최근접 이웃 검색은 큰 데이터셋에서 지금도 표준 도구다.

:::note[분야 예시]
기상학의 **유사 예보**(analog forecasting)는 kNN 회귀와 같은 생각이다. 오늘의 대기 상태와 가장 비슷한 과거의 날들을 찾아, 그 다음 날 실제로 일어난 일을 예보로 쓴다. 대기 상태는 차원이 매우 높아서, 충분히 비슷한 과거 사례를 찾기 어렵다는 점이 이 방법의 오랜 한계로 꼽혀 왔다. 차원의 저주가 실제 과학 문제에서 나타나는 예다. 실무에서는 관심 지역의 몇몇 변수로 차원을 줄인 뒤 유사 사례를 찾는다.
:::

## SVM

**왜 필요한가.** 두 클래스를 가르는 초평면은 데이터가 분리 가능하면 무수히 많다. 로지스틱 회귀(16.2장)는 확률의 우도로 하나를 고른다. 서포트 벡터 머신(support vector machine, SVM)은 다른 기준을 쓴다. **두 클래스에서 가장 멀리 떨어진 초평면**, 즉 마진(margin)이 가장 큰 초평면을 고른다(Cortes & Vapnik, 1995).

**정의.** 라벨을 $y_i \in \{-1, +1\}$로 두고 결정 함수를 $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$라 하자. 경계 $f = 0$에서 $f = \pm 1$인 두 평행 초평면까지의 거리를 합한 마진 폭은 $2/\lVert\mathbf{w}\rVert$이다. 마진을 최대화하는 것은 $\lVert\mathbf{w}\rVert$를 최소화하는 것이고, 마진 안으로 들어오거나 반대편에 놓이는 점은 슬랙 변수 $\xi_i \ge 0$으로 허용한다.

$$
\min_{\mathbf{w}, b, \boldsymbol\xi} \ \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 - \xi_i, \ \ \xi_i \ge 0.
$$

최적에서 $\xi_i = \max(0, 1 - y_i f(\mathbf{x}_i))$이므로, 이 문제는

$$
\min_{\mathbf{w}, b} \ \sum_{i=1}^N \max\big(0,\ 1 - y_i f(\mathbf{x}_i)\big) + \frac{1}{2C}\lVert\mathbf{w}\rVert^2
$$

와 같다. 첫 항이 **hinge loss**다. 올바른 쪽에 마진 이상 떨어진 점($y_i f(\mathbf{x}_i) \ge 1$)은 손실이 정확히 0이다. 둘째 항은 16.2장의 L2 정규화다. 즉 SVM은 "hinge loss + Ridge 벌점"인 선형 분류기이고, $C$가 클수록 정규화가 약하다. 로지스틱 회귀의 cross-entropy와 비교하면, cross-entropy는 확실히 맞힌 점에도 작은 손실을 계속 주지만 hinge loss는 마진 밖의 점을 완전히 무시한다.

**쌍대 문제와 서포트 벡터.** 위 문제의 Lagrange 쌍대(10.2장)는

$$
\max_{\boldsymbol\alpha} \ \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j\,\mathbf{x}_i^\top\mathbf{x}_j \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \ \ \sum_i \alpha_i y_i = 0
$$

이고, 해에서 $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$다. 상보 여유 조건 때문에 마진 밖에 있는 점은 $\alpha_i = 0$이다. $\alpha_i > 0$인 점, 즉 마진 위나 안쪽에 있는 점만 해에 기여하고, 이들을 **서포트 벡터**라 한다. 나머지 점은 지워도 해가 바뀌지 않는다.

```python {exec="browser" session="kern"}
from sklearn.svm import SVC

rng = np.random.default_rng(2)
Xs = np.vstack([rng.normal([-1.5, 0], 0.7, (100, 2)), rng.normal([1.5, 0], 0.7, (100, 2))])   # (200, 2)
ys = np.repeat([0, 1], 100)
for C in (0.01, 1.0, 100.0):
    svm = SVC(kernel="linear", C=C).fit(Xs, ys)
    w = svm.coef_[0]
    print(f"C = {C:6.2f}: 서포트 벡터 {svm.n_support_.sum():3d}개, 마진 폭 2/||w|| = {2 / np.linalg.norm(w):.2f}, 학습 정확도 {svm.score(Xs, ys):.3f}")
svm = SVC(kernel="linear", C=1.0, tol=1e-6).fit(Xs, ys)    # 해를 정밀하게 구한다
keep = np.ones(200, bool)
keep[np.setdiff1d(np.arange(200), svm.support_)[:50]] = False   # 서포트 벡터가 아닌 점 50개 제거
svm2 = SVC(kernel="linear", C=1.0, tol=1e-6).fit(Xs[keep], ys[keep])
same = np.allclose(svm.coef_, svm2.coef_, atol=1e-4) and np.allclose(svm.intercept_, svm2.intercept_, atol=1e-4)
print("서포트 벡터가 아닌 점 50개를 지워도 경계가 같은가:", same)
```

```output
C =   0.01: 서포트 벡터  88개, 마진 폭 2/||w|| = 2.68, 학습 정확도 0.990
C =   1.00: 서포트 벡터  14개, 마진 폭 2/||w|| = 0.91, 학습 정확도 0.985
C = 100.00: 서포트 벡터   7개, 마진 폭 2/||w|| = 0.34, 학습 정확도 0.990
서포트 벡터가 아닌 점 50개를 지워도 경계가 같은가: True
```

$C$가 작으면(정규화가 강하면) 마진이 넓어지고(2.68) 마진 안에 들어오는 점이 많아져 서포트 벡터가 88개다. $C$를 키우면 마진이 좁아지고(0.34) 서포트 벡터가 7개로 준다. 서포트 벡터가 아닌 점 50개를 지우고 다시 학습해도 경계가 같다. 해가 소수의 경계 근처 점으로 정해진다는 것은 SVM의 장점(예측이 서포트 벡터 수에만 비례)이자 약점(경계 근처의 라벨 오류에 민감)이다.

::quiz{id="non-support-vector-removal"}

쌍대 문제에서 입력은 내적 $\mathbf{x}_i^\top\mathbf{x}_j$로만 나타난다. 이를 커널 $k(\mathbf{x}_i, \mathbf{x}_j)$로 바꾸면 SVM은 커널이 정한 특징 공간에서 마진을 최대화하는 비선형 분류기가 된다. 이것이 다음 절의 커널 트릭이다. SVM의 출력은 확률이 아니라 부호 있는 점수 $f(\mathbf{x})$라는 점도 기억한다. 확률이 필요하면 별도의 보정 단계가 필요하다.

## 커널 방법

**커널 트릭.** 7.9장에서 본 것처럼, 양의 정부호 커널 $k(\mathbf{x}, \mathbf{x}') = \langle\boldsymbol\phi(\mathbf{x}), \boldsymbol\phi(\mathbf{x}')\rangle$는 특징 사상 $\boldsymbol\phi$를 계산하지 않고 특징 공간의 내적을 준다. 내적만 쓰는 알고리즘은 모두 커널화할 수 있다. 자주 쓰는 커널은 다음과 같다.

| 커널 | $k(\mathbf{x}, \mathbf{x}')$ | 담는 가정 |
|---|---|---|
| 선형 | $\mathbf{x}^\top\mathbf{x}'$ | 16.2장의 선형 모델과 같다 |
| 다항 | $(\mathbf{x}^\top\mathbf{x}' + c)^d$ | 차수 $d$ 이하의 상호작용 |
| RBF(가우시안) | $\exp\!\big(-\lVert\mathbf{x} - \mathbf{x}'\rVert^2 / (2\ell^2)\big)$ | 길이 척도 $\ell$ 안에서 매끄러운 함수, 거리가 멀면 무관 |
| Matérn | 거리의 함수, 매끄러움 차수 $\nu$ | RBF보다 덜 매끄러운(유한 번 미분 가능한) 함수 |
| 주기 | $\exp\!\big(-2\sin^2(\pi\lvert x - x'\rvert/p)/\ell^2\big)$ | 주기 $p$로 반복하는 함수 |

**커널 릿지 회귀.** 7.9장의 표현자 정리에 따라, RKHS에서 정규화된 최소제곱의 해는 $f(\mathbf{x}) = \sum_i \alpha_i k(\mathbf{x}, \mathbf{x}_i)$이고 계수는

$$
\boldsymbol\alpha = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}, \qquad \hat f(\mathbf{x}) = \mathbf{k}(\mathbf{x})^\top(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}
$$

로 구한다. $\mathbf{K}$는 학습 점들의 $N \times N$ Gram 행렬, $\mathbf{k}(\mathbf{x}) = (k(\mathbf{x}, \mathbf{x}_1), \dots, k(\mathbf{x}, \mathbf{x}_N))^\top$이다. 이것이 **커널 릿지 회귀**(kernel ridge regression, KRR)다. 선형 커널이면 16.2장의 Ridge와 같은 예측을 낸다.

**GP와의 관계.** Gaussian process 회귀는 함수 $f$ 자체에 사전분포를 둔다. 평균 0, 공분산 함수 $k$인 GP 사전분포와 분산 $\sigma^2$인 가우시안 관측 잡음을 가정하면, 관측 $\mathbf{y}$가 주어졌을 때 $f(\mathbf{x})$의 사후분포는 가우시안이고

$$
\mathbb{E}[f(\mathbf{x}) \mid \mathbf{y}] = \mathbf{k}(\mathbf{x})^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}, \qquad
\operatorname{Var}[f(\mathbf{x}) \mid \mathbf{y}] = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}(\mathbf{x})^\top(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}(\mathbf{x})
$$

이다(유도는 13.7장). 사후평균은 $\lambda = \sigma^2$인 KRR의 예측과 **같은 식**이다. 16.2장에서 Ridge가 가우시안 사전분포의 MAP였던 것과 같은 관계가 함수공간에서 성립한다. 차이는 GP가 사후분산, 즉 예측의 불확실성을 함께 준다는 것이다. 사후분산은 학습 점 근처에서 작고, 멀어지면 사전분산 $k(\mathbf{x}, \mathbf{x})$로 돌아간다.

```python {exec="browser" session="kern"}
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

rng = np.random.default_rng(3)
x_tr = np.sort(rng.uniform(0, 10, 30))[:, None]            # 측정 위치, (30, 1)
y_tr = np.sin(x_tr[:, 0]) + rng.normal(0, 0.1, 30)
x_te = np.linspace(-2, 12, 141)[:, None]                   # 학습 범위 밖 포함, (141, 1)
ell, noise_var = 1.0, 0.1**2
krr = KernelRidge(kernel="rbf", gamma=1 / (2 * ell**2), alpha=noise_var).fit(x_tr, y_tr)
gp = GaussianProcessRegressor(RBF(ell), alpha=noise_var, optimizer=None).fit(x_tr, y_tr)
mean, std = gp.predict(x_te, return_std=True)              # (141,), (141,)
print("KRR 예측 = GP 사후평균:", np.allclose(krr.predict(x_te), mean, atol=1e-8))
for x0 in (5.0, 11.0, 12.0):
    i = np.argmin(np.abs(x_te[:, 0] - x0))
    print(f"x = {x0:4.1f}: 사후평균 {mean[i]:+.2f}, 사후표준편차 {std[i]:.2f}, 참값 {np.sin(x0):+.2f}")
for ell_try in (0.1, 1.0, 10.0):
    m = KernelRidge(kernel="rbf", gamma=1 / (2 * ell_try**2), alpha=noise_var).fit(x_tr, y_tr)
    rmse = np.sqrt(np.mean((m.predict(x_te[20:121]) - np.sin(x_te[20:121, 0])) ** 2))
    print(f"길이 척도 {ell_try:4.1f}: 학습 범위 안 RMSE {rmse:.3f}")
```

```output
KRR 예측 = GP 사후평균: True
x =  5.0: 사후평균 -0.90, 사후표준편차 0.06, 참값 -0.96
x = 11.0: 사후평균 -0.27, 사후표준편차 0.80, 참값 -1.00
x = 12.0: 사후평균 -0.08, 사후표준편차 0.99, 참값 -0.54
길이 척도  0.1: 학습 범위 안 RMSE 0.421
길이 척도  1.0: 학습 범위 안 RMSE 0.100
길이 척도 10.0: 학습 범위 안 RMSE 0.595
```

같은 커널과 $\lambda = \sigma^2$에서 KRR 예측과 GP 사후평균이 일치한다. 학습 범위 $[0, 10]$ 안의 $x = 5$에서 사후표준편차는 0.06으로 작다. 범위를 벗어난 $x = 11, 12$에서는 사후평균이 0(사전평균)으로 돌아가고 사후표준편차가 사전표준편차 1에 다가간다. RBF 커널의 예측은 데이터에서 멀어지면 "모른다"로 돌아가고, GP는 그것을 불확실성으로 정직하게 보고한다. KRR은 같은 예측을 하면서 그 사실을 알려 주지 않는다.

::quiz{id="krr-equals-gp-mean"}

**커널 선택의 의미.** 커널은 "어떤 함수가 그럴듯한가"에 대한 사전 가정이다. 길이 척도를 바꾼 마지막 세 줄이 그 효과를 보여 준다. $\ell = 0.1$은 측정 간격(평균 약 0.33)보다 짧아 학습 점 사이에서 예측이 0으로 꺼지고(RMSE 0.421), $\ell = 10$은 사인 곡선의 주기보다 길어 곡선을 펴 버린다(0.595). $\ell = 1$이 데이터의 실제 변동 척도와 맞는다(0.100). 커널 선택의 실무 원칙은 다음과 같다.

- **입력을 표준화하거나 특징마다 길이 척도를 둔다.** RBF 커널도 거리의 함수이므로 kNN과 같은 스케일 문제가 있다. 특징마다 다른 길이 척도를 두는 ARD(automatic relevance determination) 커널은 무관한 특징의 길이 척도를 길게 만들어 그 특징을 사실상 무시한다.
- **하이퍼파라미터는 데이터로 고른다.** KRR에서는 교차검증으로, GP에서는 주변우도(marginal likelihood)를 최대화해 길이 척도와 잡음 분산을 함께 고른다(13.7장).
- **알고 있는 구조를 커널로 넣는다.** 주기성(일·연 주기), 매끄러움의 정도(Matérn), 추세(선형 커널과의 합)는 커널의 합과 곱으로 표현한다. 커널의 합은 성분 함수들의 합을, 곱은 상호작용을 뜻한다.

:::note[분야 예시]
지구과학의 **kriging**은 관측소 몇 곳의 값으로 그 사이 지점의 광물 농도·강수량·지하수위를 보간하는 방법으로, GP 회귀와 같은 수식이다. 공간 상관 구조를 나타내는 variogram이 커널의 역할을 하고, kriging 분산이 GP의 사후분산이다. 관측소에서 멀어지면 분산이 커진다는 성질 덕분에, 다음 관측소를 어디에 세울지 정하는 데도 쓰인다. 사후분산이 가장 큰 곳에서 측정하는 것이 25장에서 다룰 능동 학습과 베이지안 실험 설계의 출발점이다.
:::

## 확장성

**문제.** 정확한 KRR과 GP는 $N \times N$ Gram 행렬을 만들고 $(\mathbf{K} + \lambda\mathbf{I})$의 선형계를 푼다. 메모리는 $O(N^2)$, 시간은 Cholesky 분해에 $O(N^3)$이다. $N = 10^4$이면 Gram 행렬만 float64로 800 MB이고, $N$이 두 배가 되면 시간은 약 8배가 된다. 예측도 학습 점 전체와의 커널 계산이 필요해 $O(N)$이다. kNN은 학습이 없지만 예측마다 이웃을 찾아야 하고, 고차원에서는 트리 기반 검색이 효율을 잃는다.

**저랭크 근사.** 해결책은 Gram 행렬을 랭크 $m \ll N$인 행렬로 근사하는 것이다. 그러면 $N \times m$ 특징 행렬 위의 Ridge 회귀가 되어 시간은 $O(Nm^2)$, 메모리는 $O(Nm)$이다.

- **Nyström 근사**는 학습 점 중 $m$개를 기준점(landmark)으로 골라 $\mathbf{K} \approx \mathbf{K}_{Nm}\mathbf{K}_{mm}^{-1}\mathbf{K}_{mN}$으로 근사한다(Williams & Seeger, 2001). 7.6장의 저랭크 근사를 열 일부만으로 하는 셈이다.
- **Random features**는 shift-invariant 커널이 어떤 분포의 Fourier 변환이라는 사실(Bochner 정리)을 쓴다. RBF 커널이면 $\boldsymbol\omega_j \sim \mathcal{N}(\mathbf{0}, \ell^{-2}\mathbf{I})$, $b_j \sim \mathrm{U}(0, 2\pi)$를 뽑아 $z_j(\mathbf{x}) = \sqrt{2/m}\cos(\boldsymbol\omega_j^\top\mathbf{x} + b_j)$를 만들면 $\mathbf{z}(\mathbf{x})^\top\mathbf{z}(\mathbf{x}') \approx k(\mathbf{x}, \mathbf{x}')$이다(Rahimi & Recht, 2007). 데이터와 무관하게 특징을 만든다.

```python {exec="browser" session="kern"}
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(4)
Xl = rng.uniform(-3, 3, (2500, 2))                         # (2500, 2)
yl = np.sin(Xl[:, 0]) * np.cos(Xl[:, 1]) + rng.normal(0, 0.1, 2500)
Xa, ya, Xb, yb = Xl[:1500], yl[:1500], Xl[1500:], yl[1500:]
exact = KernelRidge(kernel="rbf", gamma=0.5, alpha=0.01).fit(Xa, ya)
print(f"정확한 KRR (1500×1500 Gram 행렬): 시험 R² {exact.score(Xb, yb):.3f}")
for m in (10, 50, 200):
    for name, feat in [("Nyström", Nystroem(gamma=0.5, n_components=m, random_state=0)),
                       ("random features", RBFSampler(gamma=0.5, n_components=m, random_state=0))]:
        model = make_pipeline(feat, Ridge(alpha=0.01)).fit(Xa, ya)
        print(f"m = {m:3d} {name:15s}: 시험 R² {model.score(Xb, yb):.3f}")
```

```output
정확한 KRR (1500×1500 Gram 행렬): 시험 R² 0.961
m =  10 Nyström        : 시험 R² 0.805
m =  10 random features: 시험 R² 0.254
m =  50 Nyström        : 시험 R² 0.961
m =  50 random features: 시험 R² 0.957
m = 200 Nyström        : 시험 R² 0.961
m = 200 random features: 시험 R² 0.962
```

특징 10개로는 두 근사 모두 정확한 KRR에 못 미친다. 데이터에 맞춰 기준점을 고르는 Nyström(0.805)이 데이터와 무관한 random features(0.254)보다 훨씬 낫다. 특징 50개에서 두 방법 모두 정확한 KRR(0.961)에 거의 도달하고, 200개면 차이가 없다. 이 문제는 입력이 2차원이고 함수가 매끄러워 적은 특징으로 충분하다. 입력 차원이 높고 함수가 복잡하면 필요한 $m$이 커진다. $m$은 정확도와 비용 사이의 하이퍼파라미터이므로, $m$을 늘려 가며 검증 성능이 포화하는 지점을 찾는다.

**연구에서의 의미.** 데이터가 수천 개 이하이면 정확한 KRR·GP가 현실적이고, 불확실성이 필요하면 GP가 기본 선택이다. 수만 개 이상이면 Nyström·random features 같은 근사나, 16.4장의 트리 앙상블, 17장 이후의 신경망이 후보가 된다. random features 위의 선형 모델은 은닉층의 가중치를 고정한 2층 신경망과 같은 꼴이라, 커널 방법과 신경망을 잇는 다리 역할도 한다.

## 실습

같은 비선형 데이터에 kNN·커널 릿지·GP를 적합해 예측과 계산 비용을 비교한다. 코드는 `labs/classical-ml/kernels-and-neighbors/compare_cost.py`다. 학습 데이터 수 $N$을 250에서 2000까지 두 배씩 늘리며, 참 함수에 대한 시험 RMSE와 학습+예측 시간을 잰다. GP는 주변우도로 길이 척도와 잡음 분산을 최적화하므로 커널 릿지보다 비싸다. 집필 시점에 BLAS 스레드 하나로 실행한 결과는 다음과 같다(시간은 기계마다 다르다).

```output
    N  kNN (k=10): RMSE / 시간[s]  커널 릿지 (고정 커널): RMSE / 시간[s]  GP (하이퍼파라미터 최적화): RMSE / 시간[s]
  250  0.131 /   0.01   0.059 /   0.01   0.027 /   0.06
  500  0.085 /   0.00   0.044 /   0.02   0.026 /   0.28
 1000  0.058 /   0.00   0.032 /   0.04   0.021 /   1.48
 2000  0.049 /   0.01   0.019 /   0.21   (생략)
```

읽는 법은 세 가지다.

- **정확도.** 같은 $N$에서 오차는 GP가 가장 작고 kNN이 가장 크다. GP는 길이 척도와 잡음을 데이터에 맞췄고, kNN은 이웃 10개의 평균이라 매끄러운 함수를 계단 모양으로 근사한다.
- **데이터 효율.** 커널 릿지는 $N = 500$에서 이미 $N = 2000$의 kNN보다 정확하다. 매끄러움에 대한 가정이 맞으면 그 가정을 쓰는 방법이 데이터를 덜 쓴다.
- **비용.** kNN은 이 크기에서 거의 공짜다. 커널 릿지는 $N$이 1000에서 2000으로 두 배가 될 때 시간이 약 5배가 되었다. $O(N^3)$ 항이 지배하기 시작하는 구간이다. GP는 하이퍼파라미터 최적화의 매 단계마다 Cholesky 분해를 반복하므로 같은 $N$에서 수십 배 비싸다.

다음을 직접 해 본다.

1. 데이터에 무관한 잡음 특징 8개를 더하고 같은 비교를 반복한다. 어느 방법이 가장 크게 나빠지는가. GP의 커널을 특징마다 길이 척도를 두는 `RBF(length_scale=np.ones(10))`로 바꾸면 어떻게 되는가.
2. 커널 릿지의 `gamma`와 `alpha`를 고정하지 말고 `GridSearchCV`로 고른 뒤, GP가 주변우도로 고른 값($\ell$은 `gp.kernel_`에서 읽는다)과 비교한다.

이어서 두 연습문제를 푼다. 첫 문제는 표준화를 포함한 kNN 회귀를 NumPy로 구현하고, 두 번째 문제는 RBF 커널의 random Fourier features를 구현해 커널 근사 오차가 특징 수에 따라 줄어드는지 확인한다.

::exercise{id="knn-standardized"}

::exercise{id="random-fourier-features"}

## 실패 사례

- **스케일이 다른 특징으로 거리를 계산하는 것.** 숫자가 큰 특징이 거리를 지배해, 가장 중요한 특징이 무시될 수 있다(앞의 실험에서 $R^2$ 0.176 대 0.991). kNN, RBF 커널, SVM 모두 같은 문제를 가진다. 표준화를 pipeline에 넣고 학습 데이터에서만 추정한다.
- **무관한 특징을 모두 넣고 거리 기반 방법을 쓰는 것.** 차원이 늘면 이웃이 국소성을 잃는다. 특징 선택, ARD 커널, 도메인 지식으로 차원을 줄인다.
- **GP의 작은 사후분산을 "예측이 맞다"는 보장으로 읽는 것.** 사후분산은 커널과 잡음 모형이 맞다는 가정 아래의 불확실성이다. 길이 척도가 너무 길게 추정되면 데이터에서 먼 곳에서도 분산이 작게 나온다. 불확실성의 보정은 따로 검증한다(21장).
- **RBF 커널 모델로 외삽하는 것.** 학습 범위 밖에서 예측은 사전평균(보통 0)으로 돌아간다. 출력에 큰 평균이나 추세가 있으면 평균을 빼거나 선형 커널을 더한다.

## 통과 기준

- 커널 릿지 회귀와 GP 사후평균이 같은 식임을 수식으로 보이고, 두 방법이 주는 정보의 차이(사후분산)를 코드로 확인한다.
- 같은 데이터에서 kNN·커널 릿지·GP의 정확도와 계산 시간을 $N$에 따라 비교하고, 데이터 크기에 맞는 방법(정확한 계산 또는 근사)을 근거와 함께 고른다.
- 연습문제 두 개(`knn-standardized`, `random-fourier-features`)의 테스트를 통과한다.

## 참고 문헌

- Cortes, Vapnik (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. — soft margin SVM과 커널을 이용한 비선형 분류
- Beyer, Goldstein, Ramakrishnan, Shaft (1999). When is "nearest neighbor" meaningful? *Database Theory — ICDT'99*, LNCS 1540, 217–235. — 차원이 높아질 때 최근접·최원 거리의 비가 1에 가까워지는 조건
- Williams, Seeger (2001). Using the Nyström method to speed up kernel machines. *Advances in Neural Information Processing Systems 13*, 682–688. — Gram 행렬의 Nyström 저랭크 근사
- Rahimi, Recht (2007). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems 20*, 1177–1184. — shift-invariant 커널의 random Fourier features 근사
