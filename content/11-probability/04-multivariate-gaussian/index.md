---
id: probability/multivariate-gaussian
title: "다변량 가우시안"
title_en: "The Multivariate Gaussian"
tier: core
status: draft
prerequisites: [probability/conditional-probability, linear-algebra/eigen-and-symmetric]
---

# 다변량 가우시안

Gaussian process 회귀, Kalman filter, 베이지안 선형회귀, 기상 예보의 데이터 동화, VAE의 잠재공간. 서로 다른 분야의 도구처럼 보이지만 계산의 핵심은 같다. 여러 변수가 함께 가우시안 분포를 따를 때, 일부를 관측하면 나머지의 분포가 어떻게 바뀌는가. 다변량 가우시안은 이 질문에 **닫힌 식**으로 답할 수 있는 거의 유일한 분포이고, 그 답이 행렬 연산 몇 줄이라는 점에서 계산이 실용적이다.

이 장은 11.3장의 주변화·조건화를 다변량 가우시안에서 수행하는 공식을 유도하고, 7.5장의 대칭행렬·양의 정부호성·Cholesky 분해와 연결한다. 공식이 무엇을 말하는지는 기하(타원)로, 공식이 맞는지는 샘플링으로 확인한다. 마지막으로 실제 계산에서 공분산 행렬이 수치적으로 양의 정부호가 아니게 되는 문제를 다룬다.

## 학습 목표

- 다변량 가우시안의 주변·조건부 분포를 유도한다
- 선형-가우시안 모델을 다룬다
- Cholesky 분해로 샘플링하고 공분산을 수치적으로 안정하게 다룬다

## 정의와 기하

**정의.** $D$차원 확률벡터 $\mathbf{x}$가 평균 $\boldsymbol{\mu} \in \mathbb{R}^D$, 공분산 $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$(대칭, 양의 정부호)인 다변량 가우시안(multivariate Gaussian)을 따른다는 것은 밀도가

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}\,|\boldsymbol{\Sigma}|^{1/2}} \exp\Big(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Big)
$$

라는 것이고, $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$라 쓴다. $|\boldsymbol{\Sigma}|$는 행렬식이다. $\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}]$, $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top]$이고, $\Sigma_{ii}$는 성분 $i$의 분산, $\Sigma_{ij}$는 두 성분의 공분산이다. 2차원이면 표준편차 $\sigma_1, \sigma_2$와 상관계수 $\rho$로 $\boldsymbol{\Sigma} = \begin{bmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}$이다.

**Mahalanobis 거리.** 지수 안의 이차형식

$$
d_M(\mathbf{x})^2 = (\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})
$$

의 제곱근을 Mahalanobis 거리라 한다. 밀도는 이 거리만의 함수이므로 **등고선은 $d_M$이 일정한 곳, 즉 타원**이다. 7.5장에서 본 것처럼 타원의 축은 $\boldsymbol{\Sigma}$의 고유벡터 방향이고, 반축의 길이는 $\sqrt{\lambda_i}$(공분산의 고윳값의 제곱근, 즉 그 방향의 표준편차)에 비례한다. Mahalanobis 거리는 "각 방향의 표준편차를 단위로 잰 거리"다. 분산이 큰 방향으로 같은 유클리드 거리만큼 떨어진 점은 덜 이례적이다.

**샘플링.** 표준정규 벡터 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$는 성분마다 독립인 표준정규 난수로 만든다. 여기에 아핀 변환 $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$를 적용하면 평균은 $\boldsymbol{\mu}$, 공분산은 $\mathbf{L}\mathbf{L}^\top$이다. 따라서 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$인 $\mathbf{L}$이 있으면 된다. Cholesky 분해(7.5장)가 바로 그런 하삼각행렬 $\mathbf{L}$을 준다. 반대 방향 $\mathbf{z} = \mathbf{L}^{-1}(\mathbf{x} - \boldsymbol{\mu})$는 **백색화**(whitening)이고, $d_M(\mathbf{x})^2 = \lVert\mathbf{z}\rVert^2$이다. 표준정규 성분 $D$개의 제곱합이므로 $d_M^2$은 자유도 $D$인 카이제곱 분포 $\chi^2_D$를 따른다.

```python {exec="browser" session="mvn"}
import numpy as np
from scipy import stats

mu = np.array([1.0, -1.0])                     # (2,)
Sigma = np.array([[1.0, 1.2],
                  [1.2, 2.25]])                # (2, 2) sigma1 = 1, sigma2 = 1.5, rho = 0.8
L = np.linalg.cholesky(Sigma)                  # (2, 2) 하삼각, Sigma = L L^T
rng = np.random.default_rng(0)
Z = rng.standard_normal((200_000, 2))          # (N, 2) 표준정규
X = mu + Z @ L.T                               # (N, 2) x = mu + L z
print("표본 평균   :", np.round(X.mean(axis=0), 3))
print("표본 공분산 :", np.round(np.cov(X, rowvar=False), 3).tolist())

D = X - mu                                     # (N, 2)
W = np.linalg.solve(L, D.T).T                  # (N, 2) 백색화 z = L^{-1}(x - mu)
d2 = (W**2).sum(axis=1)                        # (N,) Mahalanobis 거리 제곱
for r in (1.0, 2.0, 3.0):
    print(f"Mahalanobis <= {r}: 표본 {np.mean(d2 <= r**2):.4f}   chi2(2) {stats.chi2.cdf(r**2, df=2):.4f}"
          f"   | 유클리드 <= {r}: {np.mean((D**2).sum(1) <= r**2):.4f}")
```

```output
표본 평균   : [ 1.001 -0.999]
표본 공분산 : [[1.003, 1.201], [1.201, 2.249]]
Mahalanobis <= 1.0: 표본 0.3922   chi2(2) 0.3935   | 유클리드 <= 1.0: 0.3674
Mahalanobis <= 2.0: 표본 0.8646   chi2(2) 0.8647   | 유클리드 <= 2.0: 0.7339
Mahalanobis <= 3.0: 표본 0.9887   chi2(2) 0.9889   | 유클리드 <= 3.0: 0.9131
```

표본의 평균과 공분산이 목표값과 맞는다. 표본 행렬 `Z`는 행마다 표본 하나인 `(N, 2)`이므로 $\mathbf{L}\mathbf{z}$를 한꺼번에 `Z @ L.T`로 계산했다. Mahalanobis 거리는 역행렬 $\boldsymbol{\Sigma}^{-1}$을 만들지 않고 삼각 시스템 $\mathbf{L}\mathbf{w} = \mathbf{x} - \boldsymbol{\mu}$를 풀어 백색화한 뒤 제곱합으로 구했다. Mahalanobis 거리 $r$ 안에 든 비율은 $\chi^2_2$ 분포의 CDF와 일치한다. 1차원에서 "평균 ± 2σ 안에 95%"였던 직관은 2차원에서 그대로 성립하지 않는다. Mahalanobis 거리 2 안에는 86.5%만 들어 있다. 차원이 높아질수록 이 비율은 더 줄어, 고차원 가우시안의 표본은 대부분 평균에서 멀리, 반지름 $\sqrt{D}$ 근처의 껍질에 놓인다. 유클리드 원은 분포의 모양을 무시하므로 같은 반지름에서 들어가는 비율이 다르다.

**연구에서의 의미.** 다변량 측정값의 이상치를 판정할 때 변수마다 따로 "3σ 밖"을 보면 상관을 무시하게 된다. 기온이 높으면 습도가 낮은 것이 정상인 데이터에서, 기온도 습도도 높은 점은 각 변수로는 정상 범위여도 Mahalanobis 거리로는 멀다. $d_M^2$을 $\chi^2_D$의 분위수와 비교하면 차원을 고려한 판정 기준이 된다(단 $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$를 같은 데이터로 추정하면 이 분포가 근사가 되고, 이상치 자체가 추정값을 끌어당긴다는 점에 주의한다).

## 주변화와 조건부

$\mathbf{x}$를 두 블록 $\mathbf{x}_a$(관측하지 않은 것)와 $\mathbf{x}_b$(관측한 것)로 나누고 평균과 공분산도 맞춰 나눈다.

$$
\begin{bmatrix}\mathbf{x}_a \\ \mathbf{x}_b\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}\boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b\end{bmatrix}, \begin{bmatrix}\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}\right), \qquad \boldsymbol{\Sigma}_{ba} = \boldsymbol{\Sigma}_{ab}^\top.
$$

유도에 필요한 사실은 두 가지다. (1) 가우시안 벡터의 아핀 변환 $\mathbf{M}\mathbf{x} + \mathbf{c}$는 다시 가우시안이고, 평균은 $\mathbf{M}\boldsymbol{\mu} + \mathbf{c}$, 공분산은 $\mathbf{M}\boldsymbol{\Sigma}\mathbf{M}^\top$이다($\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$를 대입하면 $\mathbf{z}$의 아핀 변환이 된다). (2) 결합 가우시안인 두 블록의 교차 공분산이 0이면, 공분산이 블록 대각이 되어 밀도가 두 블록의 곱으로 갈라지므로 **독립**이다. 일반적으로는 상관 0이 독립을 뜻하지 않지만(11.3장), 결합 가우시안에서는 같다.

**주변분포.** $\mathbf{x}_a = [\mathbf{I}\ \mathbf{0}]\,\mathbf{x}$는 선형변환이므로 사실 (1)에서 바로

$$
\mathbf{x}_a \sim \mathcal{N}(\boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})
$$

이다. 주변화는 평균과 공분산의 해당 블록을 **잘라 내는 것**으로 끝난다. 적분을 할 필요가 없다.

::quiz{id="marginal-is-subblock"}

**조건부분포.** $\mathbf{x}_b$로 설명되는 부분을 $\mathbf{x}_a$에서 뺀 잔차 $\mathbf{w} = \mathbf{x}_a - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\mathbf{x}_b$를 생각한다. $\mathbf{w}$와 $\mathbf{x}_b$의 교차 공분산은

$$
\operatorname{Cov}[\mathbf{w}, \mathbf{x}_b] = \boldsymbol{\Sigma}_{ab} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{bb} = \mathbf{0}
$$

이고, $(\mathbf{w}, \mathbf{x}_b)$는 $\mathbf{x}$의 선형변환이라 결합 가우시안이므로 사실 (2)에 의해 $\mathbf{w}$는 $\mathbf{x}_b$와 독립이다. 따라서 $\mathbf{x}_b$를 고정해도 $\mathbf{w}$의 분포는 그대로이고, $\mathbf{x}_a = \mathbf{w} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\mathbf{x}_b$에서

$$
\mathbf{x}_a \mid \mathbf{x}_b \sim \mathcal{N}\Big(\boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b),\ \ \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}\Big)
$$

를 얻는다. 조건부 공분산 $\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$를 $\boldsymbol{\Sigma}_{bb}$에 대한 **Schur complement**라 한다. 식을 읽는 법은 세 가지다.

- **조건부 평균은 $\mathbf{x}_b$의 아핀 함수다.** 관측값이 평균에서 벗어난 만큼($\mathbf{x}_b - \boldsymbol{\mu}_b$), 회귀 계수 $\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}$를 곱해 $\mathbf{x}_a$의 예측을 옮긴다. 2차원이면 $\mu_2 + \rho\frac{\sigma_2}{\sigma_1}(x_1 - \mu_1)$, 즉 회귀직선이다.
- **조건부 공분산은 관측값에 의존하지 않는다.** 무엇을 관측했는지만 중요하고, 관측값이 얼마였는지는 불확실성에 영향을 주지 않는다. 2차원이면 $\sigma_2^2(1 - \rho^2)$이다.
- **조건부 공분산은 주변 공분산보다 작거나 같다.** 빼는 항 $\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$는 PSD다. 상관이 클수록 관측이 불확실성을 많이 줄인다.

앞 절의 2차원 분포에서 $x_1 = 2.2$를 관측한 경우를 공식으로 계산하고, $x_1$이 2.2 근처(폭 ±0.05)인 표본만 골라 센 값과 비교한다.

```python {exec="browser" session="mvn"}
x1 = 2.2                                       # 관측한 x1 값
cond_mean = mu[1] + Sigma[1, 0] / Sigma[0, 0] * (x1 - mu[0])
cond_var = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]     # Schur complement
band = np.abs(X[:, 0] - x1) < 0.05                          # x1 근처 띠 안의 표본
print(f"공식     E[x2 | x1={x1}] = {cond_mean:.3f}   Var = {cond_var:.3f}")
print(f"샘플링   E[x2 | x1≈{x1}] = {X[band, 1].mean():.3f}   Var = {X[band, 1].var():.3f}   (표본 {band.sum()}개)")
print(f"주변분포 E[x2] = {mu[1]:.3f}   Var = {Sigma[1, 1]:.3f}")
```

```output
공식     E[x2 | x1=2.2] = 0.440   Var = 0.810
샘플링   E[x2 | x1≈2.2] = 0.453   Var = 0.828   (표본 3801개)
주변분포 E[x2] = -1.000   Var = 2.250
```

$x_1$을 관측하자 $x_2$의 평균은 $-1$에서 $0.44$로 옮겨 가고, 분산은 2.25에서 0.81로, 즉 $1 - \rho^2 = 0.36$배로 줄었다. 샘플링 추정은 공식과 가깝지만 조금 크다. 띠의 폭 때문에 $x_1$이 정확히 2.2가 아닌 표본이 섞이기 때문이다. 이 차이는 실습에서 따져 본다.

::quiz{id="conditional-variance-numeric"}

아래 위젯에서 상관계수 $\rho$와 관측값 $x_1$을 바꾸며 등고선 위의 회귀선 $\mathbb{E}[x_2 \mid x_1]$과 오른쪽의 주변분포·조건부분포를 비교한다. $x_1$을 움직이면 조건부분포가 이동만 하고 폭은 그대로라는 것, $|\rho|$를 키우면 폭이 좁아진다는 것을 확인한다. 표본을 켜면 $x_1$ 근처 띠 안의 표본으로 추정한 조건부분포도 함께 보인다.

::widget{name="gaussian-conditioning" rho="0.8"}

## 선형 변환과 선형-가우시안 모델

과학의 측정은 흔히 "보이지 않는 상태 $\mathbf{x}$를 선형 장치로 측정하고 잡음이 더해진" 꼴이다.

$$
\mathbf{x} \sim \mathcal{N}(\mathbf{m}_0, \mathbf{P}_0), \qquad \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}),
$$

$\boldsymbol{\varepsilon}$은 $\mathbf{x}$와 독립이다. $\mathbf{P}_0$는 측정 전의 불확실성, $\mathbf{A}$는 측정 장치(어떤 조합을 재는가), $\mathbf{R}$은 측정 잡음의 공분산이다. 이것을 **선형-가우시안 모델**(linear-Gaussian model)이라 한다.

**결합분포.** $(\mathbf{x}, \boldsymbol{\varepsilon})$은 독립인 가우시안이므로 결합 가우시안이고, $(\mathbf{x}, \mathbf{y})$는 그 선형변환이므로 역시 결합 가우시안이다. 평균과 공분산을 계산하면

$$
\begin{bmatrix}\mathbf{x} \\ \mathbf{y}\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}\mathbf{m}_0 \\ \mathbf{A}\mathbf{m}_0\end{bmatrix}, \begin{bmatrix}\mathbf{P}_0 & \mathbf{P}_0\mathbf{A}^\top \\ \mathbf{A}\mathbf{P}_0 & \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R}\end{bmatrix}\right)
$$

이다. $\operatorname{Cov}[\mathbf{x}, \mathbf{y}] = \operatorname{Cov}[\mathbf{x}, \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}] = \mathbf{P}_0\mathbf{A}^\top$이고, $\operatorname{Cov}[\mathbf{y}] = \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R}$이다.

**주변분포.** 블록을 잘라 내면 $\mathbf{y} \sim \mathcal{N}(\mathbf{A}\mathbf{m}_0, \mathbf{S})$, $\mathbf{S} = \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R}$이다. 측정값의 분포는 "상태의 불확실성이 측정 장치를 거친 것"과 "측정 잡음"의 합이다. 이것이 베이지안 모델의 evidence $p(\mathbf{y})$이고, 13.2장의 모델 비교에 쓰인다.

**사후분포.** 앞 절의 조건부 공식을 그대로 적용하면

$$
\mathbf{K} = \mathbf{P}_0\mathbf{A}^\top\mathbf{S}^{-1}, \qquad
\mathbf{x} \mid \mathbf{y} \sim \mathcal{N}\big(\mathbf{m}_0 + \mathbf{K}(\mathbf{y} - \mathbf{A}\mathbf{m}_0),\ \ \mathbf{P}_0 - \mathbf{K}\mathbf{A}\mathbf{P}_0\big)
$$

이다. $\mathbf{y} - \mathbf{A}\mathbf{m}_0$는 **예측과 관측의 차이**(innovation)이고, $\mathbf{K}$(Kalman gain)는 그 차이를 상태 공간으로 되돌리는 가중치다. 측정 잡음 $\mathbf{R}$이 작을수록 $\mathbf{K}$가 커져 관측을 더 믿는다. 같은 결과를 행렬 역의 항등식(Woodbury)으로 바꿔 쓰면 정밀도 형태가 된다.

$$
\mathbf{P}^{-1} = \mathbf{P}_0^{-1} + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{A}, \qquad
\mathbf{m} = \mathbf{P}\big(\mathbf{P}_0^{-1}\mathbf{m}_0 + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{y}\big).
$$

**사후 정밀도 = 사전 정밀도 + 측정이 주는 정밀도**이고, 사후 평균은 사전 평균과 측정을 정밀도로 가중한 평균이다. 1차원에서 사전 $\mathcal{N}(0, 4)$, 측정 $y = x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$이면 사후 정밀도는 $1/4 + 1 = 1.25$다. 두 형태 중 어느 것이 빠른지는 차원에 달려 있다. gain 형태는 $M \times M$(측정 차원) 행렬을, 정밀도 형태는 $D \times D$(상태 차원) 행렬을 푼다.

::quiz{id="linear-gaussian-posterior-mean"}

물체의 초기 위치(m)와 속도(m/s)를 추정하는 예로 확인한다. 사전 표준편차는 위치 2 m, 속도 1 m/s이고, 2초 뒤의 위치 하나만 잡음 표준편차 0.5 m로 측정한다. 측정 장치는 $\mathbf{A} = [1\ \ 2]$(위치 + 2초 × 속도)다.

```python {exec="browser" session="mvn"}
m0 = np.array([0.0, 1.0])                      # (2,) 사전 평균: 위치 m, 속도 m/s
P0 = np.diag([4.0, 1.0])                       # (2, 2) 사전 공분산
dt_s = 2.0
A = np.array([[1.0, dt_s]])                    # (1, 2) 2초 뒤 위치 = 위치 + dt * 속도
R = np.array([[0.25]])                         # (1, 1) 측정 잡음 분산
y = np.array([5.0])                            # (1,) 측정한 위치 m

S = A @ P0 @ A.T + R                           # (1, 1) y의 주변 공분산
K = P0 @ A.T @ np.linalg.inv(S)                # (2, 1) gain
m_post = m0 + K @ (y - A @ m0)
P_post = P0 - K @ A @ P0
print("사후 평균      :", np.round(m_post, 3))
print("사후 표준편차  :", np.round(np.sqrt(np.diag(P_post)), 3), "  (사전:", np.sqrt(np.diag(P0)), ")")
print("사후 상관계수  :", round(P_post[0, 1] / np.sqrt(P_post[0, 0] * P_post[1, 1]), 3))

J = np.block([[P0, P0 @ A.T], [A @ P0, S]])   # (3, 3) (x, y)의 결합 공분산
m_joint = m0 + J[:2, 2:] @ np.linalg.solve(J[2:, 2:], y - A @ m0)
print("결합분포를 조건화한 결과와 같은가:", np.allclose(m_joint, m_post))
```

```output
사후 평균      : [1.455 1.727]
사후 표준편차  : [1.435 0.718]   (사전: [2. 1.] )
사후 상관계수  : -0.941
결합분포를 조건화한 결과와 같은가: True
```

예측한 2초 뒤 위치는 $0 + 2 \times 1 = 2$ m였는데 5 m가 측정되었다. 이 3 m의 차이를 사후분포는 위치와 속도에 나누어 반영했다(위치 +1.45 m, 속도 +0.73 m/s). 두 성분의 표준편차는 조금씩만 줄었는데, 사후 상관계수가 $-0.94$로 강해졌다. 측정 하나는 "위치 + 2 × 속도"라는 **한 방향**만 정확히 알려 주기 때문이다. 그 조합은 잘 정해졌지만 위치와 속도 각각은 여전히 불확실하고, "위치가 크면 속도가 작다"는 관계로 얽힌다. 마지막 줄은 이 계산이 결국 $(\mathbf{x}, \mathbf{y})$ 결합분포의 조건화라는 것을 확인한다.

**연구에서의 의미.** 사후 공분산 $\mathbf{P}$에는 관측값 $\mathbf{y}$가 없다. 측정하기 전에 측정 설계($\mathbf{A}$, $\mathbf{R}$)별 사후 불확실성을 계산할 수 있다는 뜻이다. 어느 위치에 센서를 둘지, 어느 시점에 측정할지를 정하는 실험 설계(25.3장)가 이 성질을 쓴다. 측정을 시간 순서로 하나씩 반영해도(지난 사후를 다음 사전으로) 한꺼번에 반영한 결과와 같다는 점도 중요하다. 상태가 시간에 따라 선형으로 변하는 과정을 끼워 넣으면 이 순차 갱신이 Kalman filter가 된다(Kalman, 1960; 24.6장).

## 정밀도 행렬

공분산의 역행렬 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$을 **정밀도 행렬**(precision matrix)이라 한다. 밀도의 지수가 $-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Lambda}(\mathbf{x} - \boldsymbol{\mu})$이므로, 정밀도 행렬은 변수들이 밀도 안에서 **직접** 어떻게 엮이는지를 적는다.

**0과 조건부 독립.** 지수를 전개하면 $x_i$와 $x_j$가 함께 나오는 항은 $\Lambda_{ij}x_ix_j$뿐이다. $\Lambda_{ij} = 0$이면 나머지 변수 $\mathbf{x}_{\setminus ij}$를 고정했을 때 밀도가 $x_i$의 함수와 $x_j$의 함수의 곱으로 갈라지므로

$$
\Lambda_{ij} = 0 \iff x_i \perp x_j \mid \mathbf{x}_{\setminus ij}
$$

이다. **공분산의 0은 주변 독립, 정밀도의 0은 조건부 독립**이다. 정밀도 행렬의 0이 아닌 성분을 변으로 그린 무방향 그래프를 가우시안 그래프 모델(Gaussian Markov random field)이라 한다. 11.3장의 방향 그래프와 짝을 이루는 표현이다.

**블록 공식.** 정밀도 행렬로 쓰면 조건화가 간단해진다. 조건부분포는

$$
\mathbf{x}_a \mid \mathbf{x}_b \sim \mathcal{N}\big(\boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}(\mathbf{x}_b - \boldsymbol{\mu}_b),\ \ \boldsymbol{\Lambda}_{aa}^{-1}\big)
$$

이다. 공분산 표현에서는 주변화가 블록 잘라 내기였고 조건화에 Schur complement가 나왔다. 정밀도 표현에서는 반대로 **조건화가 블록 잘라 내기**($\boldsymbol{\Lambda}_{aa}$)이고 주변화에 Schur complement가 나온다. $a = \{i, j\}$로 두고 $2 \times 2$ 조건부 공분산 $\boldsymbol{\Lambda}_{aa}^{-1}$의 상관계수를 계산하면 $-\Lambda_{ij}/\sqrt{\Lambda_{ii}\Lambda_{jj}}$가 나온다. 나머지 변수를 모두 조건으로 했을 때의 상관계수, 즉 **편상관계수**(partial correlation)다.

$x_1 \to x_2 \to \cdots \to x_5$로 이어지는 AR(1) 사슬 $x_{t+1} = 0.8\,x_t + \varepsilon_t$로 확인한다(정상 상태의 공분산 $\Sigma_{ij} = 0.8^{|i-j|}/(1 - 0.8^2)$를 썼다).

```python {exec="browser" session="mvn"}
phi, n = 0.8, 5                                # AR(1) 사슬 x1 -> x2 -> ... -> x5
C = phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / (1 - phi**2)   # (5, 5) 공분산
Lam = np.linalg.inv(C)                         # (5, 5) 정밀도 행렬
print("공분산 (소수 둘째 자리):\n", np.round(C, 2))
print("정밀도 행렬:\n", np.round(Lam, 2) + 0.0)          # + 0.0: -0.0 표시를 0.0으로
d = np.sqrt(np.diag(Lam))
print("x1-x3 상관계수   :", round(C[0, 2] / np.sqrt(C[0, 0] * C[2, 2]), 3))
print("x1-x3 편상관계수 :", round(-Lam[0, 2] / (d[0] * d[2]), 3) + 0.0, " (나머지가 주어졌을 때)")
```

```output
공분산 (소수 둘째 자리):
 [[2.78 2.22 1.78 1.42 1.14]
 [2.22 2.78 2.22 1.78 1.42]
 [1.78 2.22 2.78 2.22 1.78]
 [1.42 1.78 2.22 2.78 2.22]
 [1.14 1.42 1.78 2.22 2.78]]
정밀도 행렬:
 [[ 1.   -0.8   0.    0.    0.  ]
 [-0.8   1.64 -0.8   0.    0.  ]
 [ 0.   -0.8   1.64 -0.8   0.  ]
 [ 0.    0.   -0.8   1.64 -0.8 ]
 [ 0.    0.    0.   -0.8   1.  ]]
x1-x3 상관계수   : 0.64
x1-x3 편상관계수 : 0.0  (나머지가 주어졌을 때)
```

공분산 행렬은 모든 성분이 0이 아니다. 사슬로 이어진 모든 변수가 서로 상관되어 있다. 정밀도 행렬은 **삼중대각**이다. 사슬에서 이웃한 변수만 직접 연결되어 있다는 그래프 구조가 정밀도 행렬의 0 패턴에 그대로 드러난다. $x_1$과 $x_3$의 상관계수는 0.64이지만, $x_2$(와 나머지)를 알면 편상관은 0이다. 11.3장의 사슬 구조 $A \to B \to C$에서 본 $A \perp C \mid B$와 같은 결론이다.

::quiz{id="precision-zero-meaning"}

**연구에서의 의미.** 유전자 발현, 뇌 영역의 활동, 주식 수익률처럼 변수가 많은 데이터에서 "어떤 변수들이 직접 연결되어 있는가"를 묻는다면, 상관 행렬이 아니라 정밀도 행렬의 구조를 봐야 한다. 상관은 간접 경로로도 생기기 때문이다. 표본 수가 변수 수보다 적으면 표본 공분산의 역행렬이 존재하지 않으므로, 정밀도 행렬이 희소하다고 가정하고 L1 벌점으로 추정하는 방법(graphical lasso)을 쓴다. 또 공간 격자나 시계열처럼 이웃끼리만 직접 연결된 모델은 정밀도 행렬이 희소해, 공분산 행렬보다 훨씬 큰 문제를 다룰 수 있다.

## 활용

이 장의 공식은 여러 장에서 다른 이름으로 다시 나온다. 공통점은 "결합 가우시안을 세우고, 관측으로 조건화한다"는 것이다.

| 방법 | 잠재 $\mathbf{x}$ | 관측 $\mathbf{y}$ | 이 장에서 쓰는 것 | 다루는 곳 |
|---|---|---|---|---|
| 베이지안 선형회귀 | 회귀 가중치 $\mathbf{w}$ | 목표값, $\mathbf{A}$ = 설계 행렬 | 선형-가우시안 사후분포 | 13.2장 |
| Gaussian process 회귀 | 모든 입력에서의 함수값 | 학습 점의 함숫값 + 잡음 | 결합 가우시안의 조건화 | 13.7장, 11.7장 |
| Kalman filter | 시각 $t$의 상태 | 시각 $t$의 측정 | 선형-가우시안 갱신의 반복 | 24.6장 |
| VAE | 잠재변수 $\mathbf{z}$ | 데이터 $\mathbf{x}$ | 대각 가우시안의 샘플링 $\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ | 19.3장 |

베이지안 선형회귀는 $\mathbf{y} = \boldsymbol{\Phi}\mathbf{w} + \boldsymbol{\varepsilon}$에서 $\mathbf{x} = \mathbf{w}$, $\mathbf{A} = \boldsymbol{\Phi}$로 둔 선형-가우시안 모델 그 자체다. 정밀도 형태의 $\mathbf{P}^{-1} = \mathbf{P}_0^{-1} + \boldsymbol{\Phi}^\top\mathbf{R}^{-1}\boldsymbol{\Phi}$에서 사전 정밀도가 ridge 벌점 역할을 한다. Gaussian process는 함수값 $f(t_1), \dots, f(t_n)$의 결합분포를 커널 함수로 정한 공분산 행렬로 두는 다변량 가우시안이고, 예측은 조건부 공식 그대로다. VAE의 encoder가 출력하는 대각 가우시안에서 표본을 뽑는 재매개변수화(11.6장)는 $\mathbf{L}$이 대각인 Cholesky 샘플링이다.

:::note[분야 예시]
**기상·해양.** 수치예보의 데이터 동화(data assimilation)는 모델 예보(사전)와 관측을 결합해 현재 대기 상태(사후)를 추정한다. 상태 차원이 수억에 이르러 공분산을 직접 저장할 수는 없지만, 계산의 뼈대는 이 장의 선형-가우시안 갱신이다. 앙상블 Kalman filter는 공분산을 예보 앙상블의 표본 공분산으로 근사한다.

**지구통계학.** 광물 매장량이나 토양 오염도를 측정 지점 몇 곳에서 공간 전체로 보간하는 kriging은, 공간 상관을 공분산 함수로 둔 가우시안의 조건부 평균이다. Gaussian process 회귀와 같은 계산이 지구과학에서 먼저 독립적으로 발전했다.
:::

**수치 문제: 양의 정부호가 깨지는 공분산.** 위의 모든 계산은 공분산이 양의 정부호라는 가정 위에 있다. 실제로는 두 가지 이유로 이 가정이 깨진다. 첫째, GP에서는 입력이 가까운 점이 많으면 커널 행렬의 행들이 거의 같아져, 수학적으로는 양의 정부호인데 고윳값이 반올림 오차 수준까지 작아진다. 둘째, 결측 때문에 변수 쌍마다 다른 표본으로 상관계수를 추정해 모으면, 어떤 분포의 공분산도 될 수 없는 행렬이 나올 수 있다. 입력 50개의 RBF 커널 행렬과 쌍별로 추정한 3×3 상관 행렬로 확인한다.

```python {exec="browser" session="mvn"}
from scipy import linalg as sla

def try_cholesky(M, name):                     # scipy는 실패 시 항상 LinAlgError를 낸다
    try:
        sla.cholesky(M, lower=True)
        print(f"{name:24s} Cholesky 성공")
    except np.linalg.LinAlgError:
        print(f"{name:24s} Cholesky 실패")

t = np.linspace(0, 1, 50)                      # (50,) 입력 위치
K = np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2 / 0.3**2)   # (50, 50) RBF 커널 행렬
w = np.linalg.eigvalsh(K)
print(f"RBF 커널: 최대 고윳값 {w.max():.1f},  |최소 고윳값| < 1e-12: {abs(w.min()) < 1e-12}")
try_cholesky(K + 1e-6 * np.eye(50), "RBF + jitter 1e-6")

C = np.array([[1.0, 0.9, 0.9],                 # 변수 쌍마다 다른 표본으로 추정한 상관계수
              [0.9, 1.0, 0.2],
              [0.9, 0.2, 1.0]])                # (3, 3)
w, Q = np.linalg.eigh(C)
print("쌍별 상관 행렬 고윳값:", np.round(w, 3))
try_cholesky(C + 1e-6 * np.eye(3), "쌍별 상관 + jitter 1e-6")
C_psd = Q @ np.diag(np.clip(w, 1e-3, None)) @ Q.T   # 음수 고윳값을 잘라 낸 PSD 근사
try_cholesky(C_psd, "고윳값 clip")
print("clip 뒤 최대 성분 변화:", round(np.abs(C_psd - C).max(), 3))
```

```output
RBF 커널: 최대 고윳값 29.1,  |최소 고윳값| < 1e-12: True
RBF + jitter 1e-6        Cholesky 성공
쌍별 상관 행렬 고윳값: [-0.177  0.8    2.377]
쌍별 상관 + jitter 1e-6      Cholesky 실패
고윳값 clip                 Cholesky 성공
clip 뒤 최대 성분 변화: 0.096
```

**RBF 커널 행렬**의 최소 고윳값은 크기가 $10^{-12}$보다 작다. 가장 큰 고윳값(29.1)과 비교하면 부동소수점 반올림 오차 수준이고, 그 부호와 정확한 값은 실행 환경(선형대수 라이브러리)마다 다르다. 그래서 jitter 없이 Cholesky가 성공하는지도 환경에 따라 갈리므로 출력하지 않았다(집필 시점에 로컬 CPython에서는 실패했다). 확인하는 도구도 환경을 탄다. 집필 시점 브라우저(Pyodide)의 `np.linalg.cholesky`는 양의 정부호가 아닌 행렬에서 오류를 내지 않고 `nan`이 든 행렬을 돌려주었다. 위 코드가 실패 시 항상 `LinAlgError`를 내는 `scipy.linalg.cholesky`를 쓴 이유다. 이런 경계의 행렬에는 대각에 아주 작은 값(**jitter**)을 더하면 어느 환경에서나 안정하게 성공한다. jitter는 각 함숫값에 아주 작은 독립 잡음이 있다고 가정하는 것과 같다. 실제 코드에서는 작은 값부터 키워 가며 시도하고 사용한 값을 기록한다. jitter가 측정 잡음 분산과 비슷한 크기가 되면 더 이상 수치 보정이 아니라 모델을 바꾸는 것이다.

**쌍별 상관 행렬**은 다르다. 첫 변수가 다른 두 변수와 각각 0.9로 상관되어 있다면 그 두 변수끼리도 강하게 상관되어 있어야 하는데, 0.2라는 값은 이와 모순된다. 고윳값 $-0.177$은 반올림 오차가 아니라 행렬 자체가 틀렸다는 신호이고, 작은 jitter로는 고쳐지지 않는다. 음수 고윳값을 작은 양수로 잘라 내 가장 가까운 PSD 행렬로 바꾸면 Cholesky는 성공하지만, 성분이 최대 0.096만큼 바뀐다. 이는 데이터가 말한 상관을 바꾼 것이므로 보고해야 하는 수정이다.

## 실습

2차원 가우시안($\sigma_1 = 1$, $\sigma_2 = 1.5$, $\rho = 0.8$, 평균 0)에서 조건부분포를 공식과 샘플링으로 각각 구해 비교한다. 샘플링은 $|x_1 - x_1^\star| < h$인 띠 안의 표본으로 추정하고, 관측값 $x_1^\star$와 띠 폭 $h$를 바꾼다. 코드는 `labs/probability/multivariate-gaussian/conditioning_band.py`이고, 표본과 회귀선 $\pm 2$ 표준편차를 `outputs/conditioning_band.png`로 그린다. 마지막 열은 띠 안의 $x_1$이 고르게 퍼져 있다고 보았을 때의 근사 분산 $\sigma_{2|1}^2 + (\text{기울기})^2 h^2/3$이다. 집필 시점의 실행 결과는 다음과 같다(시드 0, 표본 20만 개).

```output
N = 200,000, seed 0, sigma1 = 1.0, sigma2 = 1.5, rho = 0.8
formula: E[x2|x1] = 1.20 x1,  Var[x2|x1] = 0.810 (x1과 무관)
   x1     h       n     mean  (formula)     var  var+slope^2 h^2/3
  0.0  0.02    3156    0.030      0.000   0.802              0.810
  0.0  0.10   15820    0.004      0.000   0.803              0.815
  0.0  0.50   76279   -0.001      0.000   0.929              0.930
  1.2  0.02    1536    1.451      1.440   0.820              0.810
  1.2  0.10    7804    1.432      1.440   0.807              0.815
  1.2  0.50   39625    1.318      1.440   0.923              0.930
  2.5  0.02     159    2.937      3.000   0.947              0.810
  2.5  0.10     757    2.973      3.000   0.824              0.815
  2.5  0.50    4364    2.786      3.000   0.884              0.930
```

- **띠가 좁으면 표본이 적다.** $x_1 = 2.5$, $h = 0.02$에서는 표본이 159개뿐이라 분산 추정(0.947)이 공식(0.810)에서 크게 벗어난다. 분포의 꼬리에서 조건화하면 더 심하다.
- **띠가 넓으면 치우친다.** $h = 0.5$에서는 띠 안의 $x_1$이 한 값이 아니라서 분산이 부풀고(0.93 근처), $x_1^\star$가 평균에서 멀면 띠 안의 표본이 평균 쪽으로 몰려 조건부 평균도 끌려간다($x_1 = 2.5$에서 2.786, 공식 3.000).
- **공식은 이 trade-off가 없다.** 가우시안에서 조건부분포를 닫힌 식으로 쓸 수 있다는 것의 가치가 여기에 있다. 데이터에서 조건부분포를 추정하는 비모수적 방법(근방 평균, 커널 회귀)은 언제나 이 띠 폭 선택 문제를 안고 있다.

이어서 두 연습문제를 푼다. 첫째는 일반 차원의 조건부 공식, 둘째는 선형-가우시안 모델의 사후분포다.

::exercise{id="condition-gaussian"}

::exercise{id="linear-gaussian-posterior"}

## 실패 사례

- **공분산 행렬이 PSD가 아니게 되는 수치 문제.** 원인은 여러 가지다. (1) 거의 같은 입력이 있는 커널 행렬처럼 고윳값이 0에 가까운 행렬의 반올림 오차, (2) 결측 때문에 변수 쌍마다 다른 표본으로 상관을 추정해 모은 행렬, (3) $\mathbf{P}_0 - \mathbf{K}\mathbf{A}\mathbf{P}_0$ 같은 뺄셈 갱신을 반복하며 쌓인 오차와 깨진 대칭성. 대응은 Cholesky로 먼저 확인하기, 대칭화 $(\mathbf{P} + \mathbf{P}^\top)/2$, 작은 jitter, 그리고 근본적으로는 수치적으로 안정한 형태(Joseph 형태의 공분산 갱신, square-root filter)를 쓰는 것이다.
- **역행렬을 직접 계산하는 것.** `np.linalg.inv(S) @ b` 대신 `np.linalg.solve(S, b)`나 Cholesky 분해 후의 삼각 풀이를 쓴다. 더 빠르고 오차가 작다. 로그 행렬식도 `np.log(np.linalg.det(S))` 대신 Cholesky 인자로 $2\sum_i \log L_{ii}$를 계산한다. 차원이 크면 행렬식 자체가 overflow나 underflow를 일으킨다.
- **표본 수보다 차원이 큰 표본 공분산을 그대로 쓰는 것.** $N \le D$이면 표본 공분산은 rank가 $N - 1$ 이하라 특이행렬이고, 역행렬(Mahalanobis 거리, 정밀도 행렬)을 구할 수 없다. shrinkage 추정이나 희소성 가정이 필요하다.
- **각 변수가 가우시안이면 결합분포도 가우시안이라고 가정하는 것.** 주변분포가 모두 가우시안이어도 결합분포는 가우시안이 아닐 수 있다. 이 장의 성질(상관 0 ⇒ 독립, 조건부 평균이 선형)은 **결합** 가우시안에서만 성립한다.

## 통과 기준

- 블록 공분산에서 주변분포와 조건부분포를 유도하고, 2차원에서 공식과 샘플링 결과를 비교해 띠 폭에 따른 차이를 설명한다.
- 선형-가우시안 모델 $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}$의 결합분포를 세우고 사후분포를 gain 형태와 정밀도 형태로 직접 유도한다.
- Cholesky 분해로 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 표본을 만들고, 정밀도 행렬의 0 패턴을 조건부 독립으로 해석한다.
- 연습문제 두 개(`condition-gaussian`, `linear-gaussian-posterior`)의 테스트를 통과한다.

## 참고 문헌

- Bishop (2006). *Pattern Recognition and Machine Learning*. Springer. — 2.3절(가우시안 분포): 조건부 가우시안 분포를 포함한 다변량 가우시안의 성질
- Rasmussen, Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press. — 부록 A(수학적 배경)의 가우시안 항등식(A.2)과 Cholesky 분해. 저자 웹사이트에서 전문 공개
- Kalman (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. — 선형-가우시안 상태공간 모델의 순차 추정
