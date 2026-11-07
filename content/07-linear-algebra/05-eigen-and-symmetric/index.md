---
id: linear-algebra/eigen-and-symmetric
title: "고윳값과 대칭행렬"
title_en: "Eigenvalues and Symmetric Matrices"
tier: core
status: draft
prerequisites: [linear-algebra/inner-products-and-projection]
---

# 고윳값과 대칭행렬

같은 행렬을 반복해서 곱하는 계산은 연구 곳곳에 있다. 개체군 모델을 한 세대씩 진행하고, 시뮬레이터를 한 step씩 굴리고, RNN이 hidden state를 시간마다 갱신하고, 역전파가 gradient를 층마다 되돌려 보낸다. 100번 곱한 뒤의 결과는 어떻게 될까. 행렬 성분만 보아서는 알 수 없지만, **고윳값**을 알면 바로 답할 수 있다. 커지는지, 줄어드는지, 진동하는지, 어느 방향으로 수렴하는지가 고윳값과 고유벡터에 적혀 있다.

이 장의 후반은 대칭행렬에 집중한다. 공분산, Hessian, 커널 행렬, 그래프 Laplacian처럼 연구에서 만나는 중요한 행렬은 대부분 대칭이고, 대칭행렬에는 서로 직교하는 축이 반드시 있다는 강력한 정리(스펙트럴 정리)가 성립한다. 이 정리가 7.6장의 SVD와 PCA의 토대가 된다.

## 학습 목표

- 고유분해와 대각화를 이해한다
- 대칭행렬의 스펙트럴 정리를 사용한다
- 양의 (준)정부호 행렬과 이차형식을 해석한다

## 고윳값과 고유벡터

**정의.** 정사각행렬 $\mathbf{A}$에 대해

$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v}, \qquad \mathbf{v} \neq \mathbf{0}
$$

을 만족하는 수 $\lambda$를 **고윳값**(eigenvalue), $\mathbf{v}$를 그에 대응하는 **고유벡터**(eigenvector)라 한다. 대부분의 벡터는 $\mathbf{A}$를 곱하면 방향이 바뀌지만, 고유벡터는 방향이 그대로이고 길이만 $\lambda$배가 된다($\lambda < 0$이면 반대 방향). 고유벡터는 방향만 의미가 있다. $\mathbf{v}$가 고유벡터이면 $2\mathbf{v}$, $-\mathbf{v}$도 고유벡터다.

**특성방정식.** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$는 $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$이고, 0이 아닌 해 $\mathbf{v}$가 있으려면 $\mathbf{A} - \lambda\mathbf{I}$의 null space가 $\{\mathbf{0}\}$이 아니어야 한다(7.3장). 즉

$$
\det(\mathbf{A} - \lambda\mathbf{I}) = 0
$$

이다. 좌변은 $\lambda$에 대한 $n$차 다항식이므로 고윳값은 (중복과 복소수를 포함해) $n$개다. 고윳값의 합은 trace, 곱은 행렬식과 같다. 7.1장에서 닮은 행렬이 trace와 행렬식을 공유한다고 했는데, 고윳값 자체가 basis에 무관한 양이기 때문이다. 실제 계산에서는 특성다항식의 근을 구하지 않는다. 다항식의 근은 계수의 작은 오차에 매우 민감하기 때문이다. 라이브러리는 직교 변환을 반복하는 알고리즘(QR 알고리즘 등, 9.3장)을 쓴다.

**대각화.** $n \times n$ 행렬이 선형독립인 고유벡터 $n$개를 가지면, 고유벡터를 열로 쌓은 $\mathbf{V}$와 고윳값을 대각에 놓은 $\boldsymbol{\Lambda}$로

$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}
$$

이라 쓸 수 있다. 이를 **고유분해**(eigendecomposition)라 하고, 이런 행렬을 **대각화 가능**하다고 한다. 7.1장의 언어로는 "고유벡터 basis에서 $\mathbf{A}$의 행렬이 대각행렬"이라는 뜻이다. 고윳값이 모두 서로 다르면 대각화 가능하다. 모든 행렬이 대각화 가능한 것은 아니다. 전단(shear) $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$은 고윳값 1이 두 번 겹치는데 고유벡터 방향은 $(1, 0)$ 하나뿐이다.

**반복 적용.** 고유분해의 가장 큰 쓸모는 거듭제곱이다. $\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$이고, $\boldsymbol{\Lambda}^k$는 대각 성분을 $k$제곱하면 된다. 초기 벡터를 고유벡터로 전개해 $\mathbf{x}_0 = \sum_i c_i\mathbf{v}_i$로 쓰면

$$
\mathbf{A}^k\mathbf{x}_0 = \sum_i c_i\lambda_i^k\mathbf{v}_i
$$

이다. 각 성분이 자기 고윳값의 $k$제곱으로 독립적으로 커지거나 줄어든다. $k$가 크면 절댓값이 가장 큰 고윳값(**지배 고윳값**)의 항이 나머지를 압도한다.

예로 두 지역 사이의 인구 이동을 본다. 매년 지역 1 인구의 20%가 지역 2로, 지역 2 인구의 30%가 지역 1로 옮긴다. 각 열의 합이 1이므로 총인구는 보존된다.

```python {exec="browser" session="eig"}
import numpy as np

A = np.array([[0.8, 0.3],
              [0.2, 0.7]])                     # (2, 2) 두 지역 사이 이동 비율(열의 합 = 1)
lam, V = np.linalg.eig(A)                      # 고윳값 (2,), 고유벡터 열 (2, 2)
lam, V = lam.real, V.real                      # 이 행렬의 고윳값은 실수다
print("고윳값        :", lam)
print("A v = lam v ? :", np.allclose(A @ V, V * lam))
print("A = V L V^-1 ?:", np.allclose(A, V @ np.diag(lam) @ np.linalg.inv(V)))

x0 = np.array([1.0, 0.0])                      # 처음에 모두 지역 1에
for k in [1, 5, 20]:
    xk = V @ np.diag(lam**k) @ np.linalg.solve(V, x0)
    print(f"k = {k:2d}  x_k = {np.round(xk, 4)}")
print("lam = 1 고유벡터(합 1로 정규화):", np.round(V[:, 0] / V[:, 0].sum(), 4))
```

```output
고윳값        : [1.  0.5]
A v = lam v ? : True
A = V L V^-1 ?: True
k =  1  x_k = [0.8 0.2]
k =  5  x_k = [0.6125 0.3875]
k = 20  x_k = [0.6 0.4]
lam = 1 고유벡터(합 1로 정규화): [0.6 0.4]
```

고윳값 1의 성분은 그대로 남고, 고윳값 0.5의 성분은 해마다 절반으로 줄어든다. 그래서 어디서 시작하든 인구 분포는 고윳값 1의 고유벡터 $(0.6, 0.4)$로 수렴한다. 수렴 속도는 둘째 고윳값 0.5가 정한다. 20년 뒤에는 $0.5^{20} \approx 10^{-6}$만 남는다.

::quiz{id="repeated-application"}

**거듭제곱법.** 위 관찰을 거꾸로 쓰면 지배 고유벡터를 구하는 알고리즘이 된다. 아무 벡터에서 시작해 $\mathbf{A}$를 곱하고 길이를 1로 맞추기를 반복한다.

$$
\mathbf{v}_{k+1} = \frac{\mathbf{A}\mathbf{v}_k}{\lVert\mathbf{A}\mathbf{v}_k\rVert}, \qquad \lambda \approx \frac{\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k}{\mathbf{v}_k^\top\mathbf{v}_k}.
$$

정규화는 수가 넘치거나(overflow) 0으로 사라지는 것을 막을 뿐 방향은 바꾸지 않는다. 오차는 대략 $|\lambda_2/\lambda_1|^k$로 줄어든다. 두 고윳값의 크기가 비슷하면 수렴이 느리다. 고윳값 추정에 쓴 비율을 **Rayleigh quotient**라 하며, 셋째 절에서 다시 다룬다. 거듭제곱법은 행렬 곱만 있으면 되므로, 행렬을 직접 저장할 수 없을 만큼 큰 문제(웹 그래프의 PageRank, 신경망 Hessian의 최대 고윳값 추정)에서도 쓸 수 있다. 실습의 연습문제에서 직접 구현한다.

## 스펙트럴 정리

**왜 필요한가.** 일반 행렬의 고유벡터는 서로 비스듬하고, 고윳값이 복소수일 수 있고, 대각화가 안 될 수도 있다. 대칭행렬은 이 모든 문제가 없다. 그리고 연구에서 만나는 중요한 행렬은 대부분 대칭이다.

**정리(스펙트럴 정리).** 실대칭행렬 $\mathbf{S} = \mathbf{S}^\top$는 다음을 만족한다.

1. 고윳값이 모두 실수다.
2. 정규직교인 고유벡터 $n$개를 고를 수 있다.
3. 따라서 직교행렬 $\mathbf{Q}$로 대각화된다.

$$
\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top = \sum_{i=1}^n \lambda_i\,\mathbf{q}_i\mathbf{q}_i^\top.
$$

일반 고유분해의 $\mathbf{V}^{-1}$이 $\mathbf{Q}^\top$로 바뀐 것이 핵심이다(7.4장의 직교행렬). 서로 다른 고윳값의 고유벡터가 직교하는 이유는 짧게 보일 수 있다. $\mathbf{S}\mathbf{q}_1 = \lambda_1\mathbf{q}_1$, $\mathbf{S}\mathbf{q}_2 = \lambda_2\mathbf{q}_2$이면

$$
\lambda_1\mathbf{q}_1^\top\mathbf{q}_2 = (\mathbf{S}\mathbf{q}_1)^\top\mathbf{q}_2 = \mathbf{q}_1^\top\mathbf{S}^\top\mathbf{q}_2 = \mathbf{q}_1^\top\mathbf{S}\mathbf{q}_2 = \lambda_2\mathbf{q}_1^\top\mathbf{q}_2
$$

이므로 $(\lambda_1 - \lambda_2)\mathbf{q}_1^\top\mathbf{q}_2 = 0$이고, $\lambda_1 \neq \lambda_2$이면 $\mathbf{q}_1^\top\mathbf{q}_2 = 0$이다. 셋째 등호에서 대칭성 $\mathbf{S}^\top = \mathbf{S}$를 썼다.

**직관.** 오른쪽 합 $\sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^\top$에서 $\mathbf{q}_i\mathbf{q}_i^\top$는 $\mathbf{q}_i$ 방향으로의 사영행렬이다(7.4장). 대칭행렬은 공간을 서로 직교하는 축으로 나누고, 축마다 $\lambda_i$배로 늘이거나 줄이는 변환이다. 회전이나 비스듬한 전단이 섞이지 않는다.

```python {exec="browser" session="eig"}
rng = np.random.default_rng(0)
B = rng.normal(size=(4, 4))
S = (B + B.T) / 2                              # (4, 4) 대칭행렬
w, Q = np.linalg.eigh(S)                       # 실수 고윳값(오름차순), 정규직교 고유벡터
print("대칭  고윳값:", np.round(w, 3))
print("Q^T Q = I ?  ", np.allclose(Q.T @ Q, np.eye(4)), "  S = Q W Q^T ?", np.allclose(S, Q @ np.diag(w) @ Q.T))

_, V = np.linalg.eig(np.array([[2.0, 1.0], [0.0, 3.0]]))
V = V.real                                     # 고윳값 2, 3 (실수)
print("비대칭 고유벡터 사이 cos =", round(abs(V[:, 0] @ V[:, 1]), 4))
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print("30도 회전의 고윳값:", np.round(np.linalg.eigvals(R), 4))
```

```output
대칭  고윳값: [-1.748 -0.452  0.115  1.216]
Q^T Q = I ?   True   S = Q W Q^T ? True
비대칭 고유벡터 사이 cos = 0.7071
30도 회전의 고윳값: [0.866+0.5j 0.866-0.5j]
```

대칭행렬의 고유벡터는 정확히 직교한다. 비대칭 행렬 $\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$의 두 고유벡터 $(1, 0)$, $(1, 1)/\sqrt{2}$는 45° 기울어져 있다(cos = 0.7071). 회전행렬은 방향을 바꾸지 않는 실벡터가 없으므로 고윳값이 복소수 $\cos\theta \pm i\sin\theta$다. `eigh`는 대칭(Hermitian) 행렬 전용 함수로, 이 구조를 이용해 `eig`보다 빠르고 정확하며 고윳값을 오름차순으로 정렬해 준다.

:::note[분야 예시]
분자의 진동을 분석할 때는 퍼텐셜 에너지를 평형 구조 근처에서 2차까지 전개하고, 원자 좌표에 대한 2계 도함수 행렬(Hessian)을 질량으로 보정해 고유분해한다. Hessian은 대칭이므로 고유벡터가 서로 직교하는 **정규 진동 모드**(normal mode)가 되고, 고윳값은 각 모드 진동수의 제곱에 비례한다. 분자의 복잡한 진동은 이 독립적인 모드들의 중첩이다. 진동 분광의 피크 위치를 계산할 때, 그리고 구조물의 공진 주파수를 해석할 때(강성 행렬과 질량 행렬의 고유문제) 같은 수학을 쓴다.
:::

## 양의 정부호성과 이차형식

**정의.** 대칭행렬 $\mathbf{S}$가 모든 $\mathbf{x} \neq \mathbf{0}$에 대해

$$
\mathbf{x}^\top\mathbf{S}\mathbf{x} > 0
$$

이면 **양의 정부호**(positive definite, PD), $\ge 0$이면 **양의 준정부호**(positive semidefinite, PSD)라 한다. $\mathbf{x}^\top\mathbf{S}\mathbf{x}$를 **이차형식**(quadratic form)이라 한다. 성분으로 쓰면 $\sum_{ij} s_{ij}x_ix_j$로, 변수들의 2차 다항식이다.

**판정.** 스펙트럴 정리로 $\mathbf{y} = \mathbf{Q}^\top\mathbf{x}$라 두면

$$
\mathbf{x}^\top\mathbf{S}\mathbf{x} = \mathbf{x}^\top\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top\mathbf{x} = \sum_i \lambda_i y_i^2
$$

이다. 따라서 **PD ⟺ 모든 고윳값 > 0, PSD ⟺ 모든 고윳값 ≥ 0**이다. 계산에서는 Cholesky 분해 $\mathbf{S} = \mathbf{L}\mathbf{L}^\top$($\mathbf{L}$은 대각이 양수인 하삼각행렬)를 시도하는 것이 가장 빠른 PD 판정법이다. Cholesky는 PD일 때만 성공한다. 자주 만나는 PSD 행렬은 $\mathbf{A}^\top\mathbf{A}$다. $\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x} = \lVert\mathbf{A}\mathbf{x}\rVert^2 \ge 0$이기 때문이다. 성분이 모두 양수라고 PD가 되는 것도, 음수 성분이 있다고 PD가 아닌 것도 아니다. 고윳값을 봐야 한다.

**기하.** 위 식에서 $\mathbf{x}^\top\mathbf{S}\mathbf{x} = 1$인 점들의 집합은 고유벡터 좌표 $\mathbf{y}$에서 $\sum_i \lambda_i y_i^2 = 1$이다. 모든 $\lambda_i > 0$이면 이는 **타원**(고차원에서는 타원체)이고, 축은 고유벡터 $\mathbf{q}_i$ 방향, 반축의 길이는 $1/\sqrt{\lambda_i}$이다. 고윳값이 **작은** 방향으로 타원이 **길다.** 고윳값에 음수가 섞이면 등위선은 쌍곡선이 되고, 이차형식의 그래프는 안장(saddle) 모양이 된다.

이 그림은 최적화에서 그대로 쓰인다. 손실함수를 최솟점 근처에서 2차로 근사하면 $L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}^\star) + \frac{1}{2}\boldsymbol{\delta}^\top\mathbf{H}\boldsymbol{\delta}$이고, Hessian $\mathbf{H}$의 고윳값이 각 방향의 곡률이다. 등고선은 타원이고, 가장 큰 고윳값과 가장 작은 고윳값의 비가 크면 길쭉한 골짜기가 되어 경사하강이 지그재그로 느리게 내려간다(10.3장). 임계점의 Hessian이 PD이면 극소, 부호가 섞이면 안장점이다.

**Rayleigh quotient.** 대칭행렬 $\mathbf{S}$에 대해

$$
R(\mathbf{x}) = \frac{\mathbf{x}^\top\mathbf{S}\mathbf{x}}{\mathbf{x}^\top\mathbf{x}}
$$

를 Rayleigh quotient라 한다. 위의 $\sum_i \lambda_i y_i^2$ 표현에서 $\lVert\mathbf{y}\rVert = \lVert\mathbf{x}\rVert$이므로 $R(\mathbf{x})$는 고윳값들의 가중평균이고, 따라서

$$
\lambda_{\min} \le R(\mathbf{x}) \le \lambda_{\max}
$$

이며, 양 끝은 해당 고유벡터에서 정확히 달성된다. 즉 **최대 고윳값은 이차형식을 최대화하는 단위 벡터 방향의 값**이다. 이 변분 표현이 7.6장의 PCA("분산을 최대화하는 방향")의 수학적 근거다.

```python {exec="browser" session="eig"}
S2 = np.array([[2.0, 1.0], [1.0, 2.0]])
T2 = np.array([[1.0, 2.0], [2.0, 1.0]])
print("S2 고윳값:", np.linalg.eigvalsh(S2), "  T2 고윳값:", np.linalg.eigvalsh(T2))
x = np.array([1.0, -1.0])
print("x^T S2 x =", x @ S2 @ x, "  x^T T2 x =", x @ T2 @ x)

X = np.random.default_rng(1).normal(size=(10000, 2))           # (N, 2) 무작위 방향
rq = np.einsum("ni,ij,nj->n", X, S2, X) / np.einsum("ni,ni->n", X, X)
print(f"Rayleigh quotient 범위: [{rq.min():.4f}, {rq.max():.4f}]")
for M, name in [(S2, "S2"), (T2, "T2")]:
    try:
        np.linalg.cholesky(M)
        print(name, "Cholesky 성공 → 양의 정부호")
    except np.linalg.LinAlgError:
        print(name, "Cholesky 실패 → 양의 정부호 아님")
```

```output
S2 고윳값: [1. 3.]   T2 고윳값: [-1.  3.]
x^T S2 x = 2.0   x^T T2 x = -2.0
Rayleigh quotient 범위: [1.0000, 3.0000]
S2 Cholesky 성공 → 양의 정부호
T2 Cholesky 실패 → 양의 정부호 아님
```

$\mathbf{T}_2$는 성분이 모두 양수이지만 고윳값 $-1$이 있어 PD가 아니다. 고유벡터 방향 $\mathbf{x} = (1, -1)$에서 이차형식이 음수가 된다. 무작위 방향 10000개에서 계산한 $\mathbf{S}_2$의 Rayleigh quotient는 정확히 두 고윳값 1과 3 사이에 있다.

::quiz{id="psd-test"}

## 공분산 행렬

**정의.** 데이터 점 $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^D$의 표본 공분산 행렬은

$$
\mathbf{C} = \frac{1}{N - 1}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^\top
$$

이다($\boldsymbol{\mu}$는 표본 평균). 대각 성분은 각 특징의 분산, 비대각 성분은 두 특징의 공분산이다. $\mathbf{C}$는 대칭이고 PSD다. 단위 벡터 $\mathbf{u}$에 대해 $\mathbf{u}^\top\mathbf{C}\mathbf{u}$는 데이터를 $\mathbf{u}$ 방향으로 사영한 값 $\mathbf{u}^\top(\mathbf{x}_n - \boldsymbol{\mu})$의 표본 분산이고, 분산은 음수가 될 수 없기 때문이다.

**주축.** 이 해석과 Rayleigh quotient를 합치면 공분산의 고유구조가 무엇을 말하는지 분명해진다.

- 최대 고윳값의 고유벡터 $\mathbf{q}_1$은 **데이터의 분산이 가장 큰 방향**이고, 그 고윳값은 그 방향의 분산이다.
- 다음 고유벡터들은 앞의 방향과 직교하면서 분산이 가장 큰 방향을 차례로 준다.
- 고윳값의 합 $\operatorname{tr}\mathbf{C}$는 모든 특징의 분산을 더한 총분산이다.

이 고유벡터들을 데이터의 **주축**(principal axes)이라 한다. 데이터 구름을 타원으로 그리면 그 타원의 축이다. 기온(°C)과 상대습도(%)를 함께 측정한 가상의 기상 데이터로 확인한다. 기온이 오르면 상대습도가 내려가도록 만들었다.

```python {exec="browser" session="eig"}
rng = np.random.default_rng(2)
temp_c = rng.normal(20.0, 3.0, size=500)                       # (500,) 기온
rh_pct = 60.0 - 2.0 * (temp_c - 20.0) + rng.normal(0.0, 4.0, size=500)   # (500,) 상대습도
D = np.column_stack([temp_c, rh_pct])                          # (500, 2)
C = np.cov(D, rowvar=False)                                    # (2, 2) 표본 공분산
w, Q = np.linalg.eigh(C)
print("공분산:\n", np.round(C, 2))
print("고윳값(분산):", np.round(w, 2))
print("제1주축     :", np.round(Q[:, 1], 3), " 기울기 =", round(Q[1, 1] / Q[0, 1], 3))
proj = (D - D.mean(0)) @ Q[:, 1]
print("제1주축 방향 표본 분산 =", round(proj.var(ddof=1), 2))
```

```output
공분산:
 [[  9.24 -18.64]
 [-18.64  54.04]]
고윳값(분산): [ 2.5  60.78]
제1주축     : [-0.34  0.94]  기울기 = -2.764
제1주축 방향 표본 분산 = 60.78
```

제1주축 방향으로 사영한 데이터의 분산이 최대 고윳값 60.78과 정확히 같다. 두 가지를 눈여겨본다. 첫째, 제1주축의 기울기($-2.76$)는 데이터를 만든 관계의 기울기($-2$)와 다르다. 주축은 점에서 직선까지의 **수직 거리**를 최소화하는 방향이고, 회귀는 $y$ 방향의 거리를 최소화하므로 서로 다른 질문에 답한다. 둘째, 이 결과는 단위에 의존한다. 습도를 %가 아니라 0–1 비율로 적으면 습도의 분산이 $10^{-4}$배가 되어 주축이 거의 기온 축으로 돌아간다. 단위가 다른 특징을 섞을 때의 이 문제는 7.6장의 PCA에서 다시 다룬다.

:::note[분야 예시]
기후과학에서는 해수면 온도 같은 장(field)을 격자점마다 시계열로 모은 뒤, 평균을 뺀 편차(anomaly)의 공간 공분산 행렬을 고유분해한다. 그 고유벡터를 경험적 직교 함수(EOF, empirical orthogonal function)라 부르며, 각 EOF는 함께 변동하는 공간 패턴이고 고윳값은 그 패턴이 설명하는 분산이다. 수학적으로는 이 절의 주축 분석, 그리고 7.6장의 PCA와 같은 계산이다.
:::

## 동역학과 안정성

**왜 필요한가.** 이산 시간 선형 시스템 $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t$는 시뮬레이터의 한 step, 평형점 근처에서 선형화한 비선형 시스템, 선형 RNN의 hidden state 갱신을 모두 포함한다. 이런 시스템이 장기적으로 어떻게 될지는 첫째 절의 반복 적용 공식이 답한다.

**스펙트럼 반지름과 안정성.** $\mathbf{x}_t = \mathbf{A}^t\mathbf{x}_0 = \sum_i c_i\lambda_i^t\mathbf{v}_i$이므로, 모든 성분이 0으로 줄어들려면 모든 $|\lambda_i| < 1$이어야 한다. 고윳값 절댓값의 최댓값을 **스펙트럼 반지름**(spectral radius) $\rho(\mathbf{A}) = \max_i |\lambda_i|$라 한다.

- $\rho(\mathbf{A}) < 1$: 모든 초기 상태에서 $\mathbf{x}_t \to \mathbf{0}$ (점근적으로 안정)
- $\rho(\mathbf{A}) > 1$: 해당 고유벡터 성분이 있는 초기 상태에서 $\lVert\mathbf{x}_t\rVert \to \infty$
- 복소 고윳값 $re^{i\phi}$: 크기 $r^t$로 변하면서 각속도 $\phi$로 회전(진동)
- 음의 실수 고윳값: 매 step 부호가 바뀌며 진동

연속 시간 $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$에서는 같은 역할을 $\operatorname{Re}\lambda_i < 0$이 한다. 시간 간격 $\Delta t$로 이산화하면 $e^{\lambda\Delta t}$의 크기가 1보다 작은지가 이산 시간 조건이 되고, 이 대응이 ODE solver의 안정성 분석(9.6장)으로 이어진다.

**비정규 행렬의 일시적 증폭.** 고윳값은 $t \to \infty$의 거동을 말할 뿐, 그 사이의 크기는 말하지 않는다. 고유벡터가 직교하지 않는 행렬(비정규 행렬)은 모든 고윳값이 1보다 작아도 일시적으로 크게 증폭할 수 있다.

```python {exec="browser" session="eig"}
def run(A, x0, T):
    xs = [x0]
    for _ in range(T):
        xs.append(A @ xs[-1])
    return np.array([np.linalg.norm(x) for x in xs])           # (T + 1,)

x0 = np.array([1.0, 1.0])
stable = np.array([[0.9, 0.0], [0.0, 0.5]])
unstable = np.array([[1.05, 0.0], [0.0, 0.5]])
nonnormal = np.array([[0.9, 5.0], [0.0, 0.8]])                  # 고윳값 0.9, 0.8
for name, M in [("stable", stable), ("unstable", unstable), ("nonnormal", nonnormal)]:
    rho = max(abs(np.linalg.eigvals(M)))
    n = run(M, x0, 60)
    print(f"{name:9s} rho = {rho:.2f}  |x_10| = {n[10]:7.3f}  |x_60| = {n[60]:9.3e}  최대 = {n.max():.2f}")
```

```output
stable    rho = 0.90  |x_10| =   0.349  |x_60| = 1.797e-03  최대 = 1.41
unstable  rho = 1.05  |x_10| =   1.629  |x_60| = 1.868e+01  최대 = 18.68
nonnormal rho = 0.90  |x_10| =  12.414  |x_60| = 9.157e-02  최대 = 14.00
```

`nonnormal`은 `stable`과 스펙트럼 반지름이 같은 0.90이지만, 초기 크기 $\sqrt{2} \approx 1.41$에서 최대 14까지 약 10배 커졌다가 나중에야 줄어든다. 비대각 성분 5가 한 성분을 다른 성분으로 밀어 넣기 때문이다. 유체의 난류 천이, 생태계의 교란 반응처럼 "결국 안정하지만 그 전에 크게 튀는" 현상이 이 구조에서 나온다. 단기 거동을 판단하려면 고윳값 대신 $\lVert\mathbf{A}^t\rVert$를 직접 보거나, 한 step의 최대 증폭률인 최대 특이값(7.6장)을 본다.

::quiz{id="stability-spectral-radius"}

**RNN의 gradient.** 활성화가 없는 선형 RNN $\mathbf{h}_t = \mathbf{W}\mathbf{h}_{t-1}$에서 마지막 상태 $\mathbf{h}_K$에서 계산한 손실의 gradient를 $K$ step 거슬러 보내면

$$
\frac{\partial L}{\partial \mathbf{h}_0} = (\mathbf{W}^\top)^K\,\frac{\partial L}{\partial \mathbf{h}_K}
$$

이다(8.4장의 역전파). 같은 행렬을 $K$번 곱하므로, $\rho(\mathbf{W}) < 1$이면 gradient가 지수적으로 사라지고(vanishing gradient), $\rho(\mathbf{W}) > 1$이면 지수적으로 폭발한다(exploding gradient). 비선형 RNN에서는 매 step 활성화 함수의 도함수가 대각행렬로 곱해져 상황이 더 복잡하지만, 같은 행렬 곱의 반복이라는 구조는 그대로다. gradient clipping, gating(LSTM·GRU), 가중치를 직교행렬 근처로 두는 초기화가 모두 이 문제를 다루기 위한 기법이다(18.3장).

## 실습

거듭제곱법을 구현하고, 선형 동역학 시스템의 장기 거동을 고윳값으로 예측해 시뮬레이션과 비교한다. 코드는 `labs/linear-algebra/eigen-and-symmetric/long_term.py`다. 1부는 생태학의 연령 구조 개체군 모델(Leslie 행렬)이다. 세 연령 계층의 개체 수 $\mathbf{n}_t$가 $\mathbf{n}_{t+1} = \mathbf{L}\mathbf{n}_t$로 변한다. $\mathbf{L}$의 첫 행은 계층별 출산율 $(0, 1.5, 1.2)$, 아래 대각은 다음 계층으로의 생존율 $(0.6, 0.4)$이다. 2부는 선형 RNN에서 gradient를 $K$ step 거슬러 보냈을 때의 크기를 잰다. 집필 시점의 실행 결과는 다음과 같다.

```output
[1부] Leslie 행렬
  numpy 고윳값       : [ 1.0801+0.j -0.6982+0.j -0.3819+0.j]
  거듭제곱법 지배 고윳값: 1.0801
  안정 연령 분포(예측): [0.5678 0.3154 0.1168]
  t =  1  총 개체 수      60.0  성장률 0.6000  연령 분포 [0. 1. 0.]
  t =  2  총 개체 수     114.0  성장률 1.9000  연령 분포 [0.7895 0.     0.2105]
  t =  5  총 개체 수     107.4  성장률 0.8955  연령 분포 [0.4829 0.4527 0.0644]
  t = 10  총 개체 수     172.3  성장률 1.1047  연령 분포 [0.5767 0.3008 0.1226]
  t = 40  총 개체 수    1723.9  성장률 1.0801  연령 분포 [0.5678 0.3154 0.1168]
[2부] 선형 RNN h_t = W h_(t-1)의 gradient 크기 |(W^T)^K g|
  rho = 0.9  K = 10:  3.50e-01  K = 50:  2.11e-03  K = 100:  7.87e-06
  rho = 1.0  K = 10:  1.00e+00  K = 50:  4.10e-01  K = 100:  2.96e-01
  rho = 1.1  K = 10:  2.60e+00  K = 50:  4.82e+01  K = 100:  4.08e+03
```

결과를 읽는 법은 세 가지다.

- **장기 성장률은 지배 고윳값이다.** 초기에는 성장률이 0.6, 1.9, 0.9로 크게 출렁이지만, 40세대 뒤에는 지배 고윳값 1.0801과 소수점 넷째 자리까지 같다. 연령 분포도 지배 고유벡터 $(0.5678, 0.3154, 0.1168)$로 수렴한다. 초기 상태가 무엇이든 장기 예측은 행렬 하나로 정해진다.
- **초기의 출렁임은 다른 고윳값에서 온다.** 둘째 고윳값 $-0.6982$는 음수라 세대마다 부호가 바뀌며 진동하고, 그 크기가 지배 고윳값 대비 $|{-0.6982}/1.0801| \approx 0.65$배씩 줄어든다. 이 비율이 거듭제곱법의 수렴 속도이기도 하다.
- **gradient의 크기는 대략 스펙트럼 반지름의 $K$제곱을 따라간다.** $\rho = 0.9$에서 100 step 뒤 gradient는 $10^{-5}$ 수준으로 사라지고, $\rho = 1.1$에서는 $10^{3}$ 수준으로 폭발한다. $\rho = 1.0$에서도 크기가 정확히 유지되지는 않는다. 크기가 1보다 작은 나머지 고윳값 방향의 성분이 사라지고, 고유벡터가 직교하지 않아 노름이 정확히 보존되지 않기 때문이다.

이어서 두 연습문제를 푼다.

::exercise{id="power-iteration"}

::exercise{id="ellipse-axes"}

## 실패 사례

- **비대칭 행렬에 대칭행렬의 성질을 적용하는 것.** 비대칭 행렬의 고유벡터는 직교하지 않으므로 $\mathbf{V}^{-1} \neq \mathbf{V}^\top$이고, 고윳값이 복소수일 수 있으며, 고윳값만으로 단기 증폭을 판단할 수 없다. 특히 `np.linalg.eigh`에 비대칭 행렬을 넣으면 오류 없이 **아래 삼각 부분만 읽어** 엉뚱한 답을 준다. 수치 오차로 미세하게 비대칭이 된 행렬은 `(S + S.T) / 2`로 대칭화한 뒤 `eigh`를 쓰고, 원래 비대칭인 행렬에는 `eig`를 쓴다.
- **고윳값과 특이값을 혼동하는 것.** $\begin{bmatrix} 1 & 10 \\ 0 & 1 \end{bmatrix}$의 고윳값은 모두 1이지만, 이 행렬은 벡터를 최대 약 10.1배까지 늘인다(최대 특이값). 고윳값은 "반복해서 곱할 때의 장기 증가율", 특이값은 "한 번 곱할 때의 최대 증폭률"이다. 두 값이 일치하는 것은 대칭 PSD 행렬처럼 특별한 경우뿐이다(7.6장).
- **고유벡터의 부호와 크기에 의미를 두는 것.** $\mathbf{v}$와 $-\mathbf{v}$는 같은 고유벡터다. 라이브러리 버전이나 플랫폼에 따라 부호가 바뀔 수 있으므로, 고유벡터를 비교하거나 시각화할 때는 부호를 정하는 규칙(예: 절댓값이 가장 큰 성분을 양수로)을 명시한다.

## 통과 기준

- 공분산 행렬의 고유분해로 데이터의 주축과 각 축의 분산을 구하고, 주축 방향으로 사영한 표본 분산이 고윳값과 같음을 확인한다.
- 선형 시스템 $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t$의 스펙트럼 반지름으로 장기 안정성을 판정하고, 시뮬레이션으로 장기 성장률이 지배 고윳값과 일치함을 확인한다.
- 연습문제 두 개(`power-iteration`, `ellipse-axes`)의 테스트를 통과한다.
