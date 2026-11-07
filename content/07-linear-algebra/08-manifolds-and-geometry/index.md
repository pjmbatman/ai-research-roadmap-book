---
id: linear-algebra/manifolds-and-geometry
title: "다양체와 표현의 기하"
title_en: "Manifolds and the Geometry of Representations"
tier: advanced
status: draft
prerequisites: [linear-algebra/affine-and-nonlinear]
---

# 다양체와 표현의 기하

앞 장들은 데이터를 벡터공간의 점으로, 모델을 그 공간 위의 함수로 다뤘다. 그런데 과학 데이터의 상당수는 벡터공간 전체를 채우지 않는다. 단위벡터로 정규화한 embedding은 구면 위에 있고, 강체의 자세는 회전 행렬의 집합 위에 있으며, 분자의 모양은 몇 개의 비틀림 각으로 정해진다. 이런 집합은 평평하지 않다. 두 점의 평균이 집합 밖으로 나가고, 직선으로 보간한 점이 의미 없는 상태가 된다.

**다양체**(manifold)는 "전체로는 휘어 있지만 가까이서 보면 평평한" 집합을 다루는 언어다. 이 장은 다양체의 정의를 직관 수준에서 세우고, 7.2장에서 본 "미분은 국소적인 아핀 근사"라는 관점을 다양체 위로 옮긴다. 그리고 연구에서 자주 섞여 쓰이는 네 가지 "차원"—행렬 rank, 벡터공간 차원, 잠재변수 차원, 다양체 차원—을 구분한다.

## 학습 목표

- 다양체를 직관적으로 이해한다
- Jacobian을 국소 선형화로 해석한다
- rank·차원·잠재 차원·다양체 차원을 구분한다

## 거리공간과 위상의 직관

다양체를 정의하려면 먼저 "가까이서 본다"는 말을 정확히 해야 한다. 좌표가 없는 집합에서도 가까움을 말할 수 있게 해 주는 것이 거리와 위상이다.

**거리공간.** 집합 $\mathcal{X}$ 위의 거리(metric) $d: \mathcal{X} \times \mathcal{X} \to [0, \infty)$는 모든 $x, y, z$에 대해 다음을 만족하는 함수다.

$$
d(x, y) = 0 \iff x = y, \qquad d(x, y) = d(y, x), \qquad d(x, z) \le d(x, y) + d(y, z).
$$

마지막 조건이 삼각부등식이다. 7.4장의 노름 $\lVert\cdot\rVert$이 있으면 $d(\mathbf{x}, \mathbf{y}) = \lVert\mathbf{x} - \mathbf{y}\rVert$가 거리가 된다. 하지만 거리는 노름보다 일반적이다. 구면 위 두 점 사이의 거리를 "구면을 따라 가는 최단 경로의 길이"로 정하면, 이 거리는 어떤 노름에서도 오지 않는다. 같은 집합에 여러 거리를 줄 수 있고, 어떤 거리를 쓰느냐가 "가까운 이웃"을 바꾼다.

**열린 집합과 위상.** 점 $x$를 중심으로 반지름 $r$ 안의 점들 $B(x, r) = \{y : d(x, y) < r\}$을 열린 공(open ball)이라 한다. 모든 점 주위에 그 집합 안에 들어가는 작은 열린 공이 있으면 그 집합을 열린 집합이라 한다. 열린 집합들의 모임을 위상(topology)이라 부르는데, 이 장에서 필요한 직관은 하나다. **위상은 "어떤 점들이 서로 가까운가"만 기억하고, 정확한 거리값은 잊는다.** 연속함수는 가까운 점을 가까운 점으로 보내는 함수이고, 연속이고 연속인 역함수를 가진 일대일 대응(위상동형사상, homeomorphism)은 두 집합을 위상적으로 같게 만든다. 커피잔과 도넛이 같다는 유명한 비유가 이 뜻이다.

## 다양체

**정의(직관).** $m$차원 다양체는 모든 점 근처가 $\mathbb{R}^m$의 열린 집합과 위상동형인 집합이다. 즉 **국소적으로 유클리드 공간처럼 보이는 공간**이다. 점 근처의 이웃을 $\mathbb{R}^m$으로 보내는 위상동형사상을 chart, 그 출력을 국소 좌표(local coordinates)라 한다. chart들로 다양체 전체를 덮은 모음을 atlas라 한다. 지구 전체를 한 장의 평평한 지도로 왜곡 없이 그릴 수 없지만, 지도책(atlas) 여러 장으로는 모든 지역을 덮을 수 있다는 비유에서 온 이름이다.

이 장에서 다루는 다양체는 모두 $\mathbb{R}^n$ 안에 놓인 매끄러운 다양체다. 대표적인 예는 다음과 같다.

| 다양체 | 놓인 공간 | 다양체 차원 | 국소 좌표의 예 |
|---|---|---|---|
| 원 $S^1$ | $\mathbb{R}^2$ | 1 | 각 $\varphi$ |
| 단위구면 $S^2$ | $\mathbb{R}^3$ | 2 | 구면좌표 $(\theta, \varphi)$ |
| 회전군 $SO(3)$ | $\mathbb{R}^{3 \times 3} \cong \mathbb{R}^9$ | 3 | 회전축 × 회전각 |
| 토러스 $S^1 \times S^1$ | $\mathbb{R}^4$ 또는 $\mathbb{R}^3$ | 2 | 두 각 $(\varphi_1, \varphi_2)$ |

$SO(3) = \{\mathbf{R} \in \mathbb{R}^{3\times 3} : \mathbf{R}^\top\mathbf{R} = \mathbf{I}, \det\mathbf{R} = 1\}$은 3차원 회전 행렬의 집합이다. 성분은 9개지만 $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$가 독립 조건 6개를 걸어 자유도 3이 남는다. 7.10장에서 이 집합을 "군"으로 다시 만난다.

::quiz{id="so3-dimension"}

반대로 다양체가 **아닌** 집합도 있다. 8자 모양 곡선은 교차점 근처에서 네 갈래로 갈라져 어떤 열린 구간과도 위상동형이 아니다. 원뿔은 꼭짓점에서 매끄럽지 않다. 실제 데이터도 이런 교차와 뾰족한 점을 가질 수 있고, 그 근처에서는 "국소적으로 평평하다"는 가정이 깨진다.

**chart는 다양체가 아니다.** 구면좌표 $(\theta, \varphi) \mapsto (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)$는 구면의 대부분을 덮는 chart다. 그런데 북극($\theta = 0$)에서는 $\varphi$를 바꿔도 점이 움직이지 않는다. 이것은 좌표의 결함이지 구면의 결함이 아니다. 구면은 회전에 대해 모든 점이 같으므로 북극도 다른 점과 똑같이 매끄럽다. 다음 셀은 일반적인 점과 북극에서 구면좌표의 Jacobian을 계산한다.

```python {exec="browser" session="manifold"}
import numpy as np

def sphere(u):                       # 국소 좌표 u = (θ, φ) [rad] → R^3 위의 점
    th, ph = u
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

def jacobian(f, u, h=1e-6):          # 중심 차분 Jacobian, (출력 차원, 입력 차원)
    u = np.asarray(u, dtype=float)
    cols = [(f(u + h * e) - f(u - h * e)) / (2 * h) for e in np.eye(len(u))]
    return np.stack(cols, axis=1)

u0 = np.array([1.0, 0.5])
p = sphere(u0)                       # (3,)
J = jacobian(sphere, u0)             # (3, 2)
print("|p| = 1     :", np.isclose(np.linalg.norm(p), 1.0))
print("J shape     :", J.shape, " rank =", np.linalg.matrix_rank(J))
print("J^T p = 0   :", np.allclose(J.T @ p, 0, atol=1e-8))
print("북극의 rank :", np.linalg.matrix_rank(jacobian(sphere, [0.0, 0.5]), tol=1e-6))
```

```output
|p| = 1     : True
J shape     : (3, 2)  rank = 2
J^T p = 0   : True
북극의 rank : 1
```

일반적인 점에서 Jacobian은 $3 \times 2$, rank 2이고, 두 열이 모두 $\mathbf{p}$에 수직이다. 북극에서는 rank가 1로 떨어진다. 회전을 오일러 각으로 매개화할 때 생기는 gimbal lock도 같은 현상이다. 특정 자세에서 세 각 중 둘이 같은 회전을 만들어 자유도 하나가 사라진 것처럼 보인다.

::quiz{id="chart-singularity"}

## 접공간과 Jacobian

**왜 필요한가.** 다양체 위에서 최적화하거나, 다양체 위의 데이터를 미분하거나, 한 점 근처의 작은 변화를 다루려면 "그 점에서 움직일 수 있는 방향"의 공간이 필요하다. 다양체 자체는 벡터공간이 아니지만, 한 점 근처의 방향들은 벡터공간을 이룬다.

**정의.** 다양체 $\mathcal{M}$ 위의 점 $\mathbf{p}$를 지나는 매끄러운 곡선 $\gamma(t)$, $\gamma(0) = \mathbf{p}$를 생각한다. 그 속도 $\gamma'(0)$들을 모두 모은 집합이 $\mathbf{p}$에서의 **접공간**(tangent space) $T_{\mathbf{p}}\mathcal{M}$이다. $m$차원 다양체의 접공간은 $m$차원 벡터공간이다. $\mathbb{R}^n$ 안의 다양체라면 접공간은 $\mathbb{R}^n$의 $m$차원 부분공간으로, 그 점에서 다양체에 접하는 평면을 원점으로 옮긴 것이다.

**Jacobian과의 관계.** chart의 역방향, 즉 국소 좌표 $\mathbf{u} \in \mathbb{R}^m$에서 다양체 위의 점으로 가는 매개화 $\phi: \mathbb{R}^m \to \mathbb{R}^n$를 생각한다. 좌표가 $\mathbf{u} + \Delta\mathbf{u}$로 조금 바뀌면 점은 1차 근사로

$$
\phi(\mathbf{u} + \Delta\mathbf{u}) \approx \phi(\mathbf{u}) + \mathbf{J}_\phi(\mathbf{u})\,\Delta\mathbf{u}
$$

만큼 움직인다. 7.2장의 1차 Taylor 근사 그대로다. 따라서 **접공간은 Jacobian $\mathbf{J}_\phi(\mathbf{u})$의 열공간**이고, 매개화가 좋은 chart라면 $\mathbf{J}_\phi$의 rank가 $m$이다. 구면에서 이 열공간은 $\{\mathbf{v} : \mathbf{p}^\top\mathbf{v} = 0\}$, 즉 $\mathbf{p}$에 수직인 평면이다. 위 셀의 `J^T p = 0`이 이것을 확인했다.

더 일반적으로, 다양체 $\mathcal{M}$에서 $\mathcal{N}$으로 가는 매끄러운 사상 $f$의 Jacobian은 **접공간 사이의 선형사상** $T_{\mathbf{p}}\mathcal{M} \to T_{f(\mathbf{p})}\mathcal{N}$이다. 곡선의 속도를 상(image) 곡선의 속도로 보낸다. 신경망 한 층 $f: \mathbb{R}^n \to \mathbb{R}^k$의 Jacobian도 같은 방식으로 읽을 수 있다. 입력이 어떤 다양체 위에 있으면, 그 접방향을 Jacobian에 넣어 출력에서 어느 방향으로 얼마나 늘어나는지 본다(Jacobian의 일반적인 계산은 8.1장).

::quiz{id="jacobian-tangent-plane"}

**회전군의 접공간과 지수사상.** $SO(3)$의 항등원 $\mathbf{I}$를 지나는 회전 곡선 $\mathbf{R}(t)$, $\mathbf{R}(0) = \mathbf{I}$를 생각한다. $\mathbf{R}(t)^\top\mathbf{R}(t) = \mathbf{I}$를 $t = 0$에서 미분하면 $\mathbf{R}'(0)^\top + \mathbf{R}'(0) = \mathbf{0}$이다. 즉 항등원에서의 접공간은 **반대칭 행렬**($\mathbf{A}^\top = -\mathbf{A}$)의 공간이다. $3 \times 3$ 반대칭 행렬은 성분 3개로 정해지므로, 벡터 $\boldsymbol{\omega} \in \mathbb{R}^3$와 일대일로 대응한다.

$$
\hat{\boldsymbol{\omega}} =
\begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix},
\qquad \hat{\boldsymbol{\omega}}\mathbf{x} = \boldsymbol{\omega} \times \mathbf{x}.
$$

접벡터 $\hat{\boldsymbol{\omega}}$에서 다양체 위의 점으로 돌아가는 사상이 **지수사상**(exponential map) $\exp(\hat{\boldsymbol{\omega}}) = \sum_{k \ge 0} \hat{\boldsymbol{\omega}}^k / k!$이다. 물리적으로는 일정한 각속도 $\boldsymbol{\omega}$로 단위 시간 동안 돈 결과로, 축 $\boldsymbol{\omega}/\lVert\boldsymbol{\omega}\rVert$ 둘레로 각 $\theta = \lVert\boldsymbol{\omega}\rVert$만큼의 회전이다. $\hat{\boldsymbol{\omega}}^3 = -\theta^2\hat{\boldsymbol{\omega}}$라는 성질 때문에 무한급수가 닫힌 형태로 정리되는데, 이것이 **Rodrigues 공식**이다. 단위 회전축의 반대칭 행렬 $\mathbf{K} = \hat{\boldsymbol{\omega}}/\theta$로 쓰면

$$
\exp(\hat{\boldsymbol{\omega}}) = \mathbf{I} + \sin\theta\,\mathbf{K} + (1 - \cos\theta)\,\mathbf{K}^2.
$$

```python {exec="browser" session="manifold"}
from scipy.linalg import expm

def hat(w):                          # (3,) → (3, 3) 반대칭 행렬
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])

def rodrigues(w):                    # w = 회전축 방향 × 회전각 [rad]
    th = np.linalg.norm(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

w = np.array([0.3, -0.5, 0.8])
R = rodrigues(w)                     # (3, 3)
print("expm(hat w)와 일치 :", np.allclose(R, expm(hat(w))))
print("R^T R = I          :", np.allclose(R.T @ R, np.eye(3)))
print("det R              :", np.linalg.det(R).round(6))
print("회전축 고정 R w = w:", np.allclose(R @ w, w))

h = 1e-6
dR = (rodrigues(h * w) - rodrigues(-h * w)) / (2 * h)   # 곡선 R(t)의 t = 0 속도
print("dR/dt = hat(w)     :", np.allclose(dR, hat(w), atol=1e-6))
```

```output
expm(hat w)와 일치 : True
R^T R = I          : True
det R              : 1.0
회전축 고정 R w = w: True
dR/dt = hat(w)     : True
```

마지막 줄은 곡선 $t \mapsto \exp(t\hat{\boldsymbol{\omega}})$의 출발 속도가 접벡터 $\hat{\boldsymbol{\omega}}$ 자체임을 확인한다. 지수사상은 "접공간의 직선을 다양체 위의 곡선으로 감는" 사상이다. 이 셀의 `rodrigues`는 $\boldsymbol{\omega} = \mathbf{0}$에서 0으로 나누어 실패한다. 이 문제를 고치는 것이 연습문제 `so3-exp`다.

연구에서 이 구조는 회전을 학습하거나 최적화할 때 쓰인다. 회전 행렬의 9개 성분을 그대로 경사하강으로 갱신하면 한 step 뒤에 더 이상 회전 행렬이 아니다. 대신 접공간에서 $\boldsymbol{\omega}$를 갱신하고 $\mathbf{R} \leftarrow \mathbf{R}\exp(\hat{\boldsymbol{\omega}})$로 다양체 위로 돌아오면 제약이 항상 지켜진다.

:::note[분야 예시]
다결정 재료의 결정립(grain)마다 결정 격자의 방위(orientation)는 $SO(3)$의 원소로 기술된다. 방위 데이터를 평균하거나 보간할 때 행렬 성분이나 오일러 각을 산술 평균하면 회전이 아닌 행렬이 나오거나, 각의 주기성 때문에 엉뚱한 평균이 나온다. 결정의 대칭(예: 입방정의 회전 대칭) 때문에 서로 다른 행렬이 물리적으로 같은 방위를 나타낸다는 점까지 고려해야 한다(7.10장).
:::

## 데이터 다양체 가설

**가설.** 데이터 다양체 가설(manifold hypothesis)은 고차원 공간의 실제 데이터가 공간 전체에 퍼져 있지 않고, 그보다 훨씬 낮은 차원의 다양체 근처에 몰려 있다는 가정이다. 64 × 64 흑백 이미지는 4096차원 벡터지만, 각 픽셀을 독립적으로 무작위로 뽑은 이미지는 거의 언제나 잡음처럼 보이고 자연 이미지처럼 보이지 않는다. 자연 이미지는 4096차원 공간의 극히 작은 부분을 차지한다는 뜻이다.

**근거.** 과학 데이터에서는 이 가설에 물리적인 이유가 있는 경우가 많다.

- 측정값이 소수의 물리 파라미터로 정해진다. 진자의 영상은 픽셀이 수만 개여도 각도와 각속도 두 개로 정해진다.
- 동역학이 저차원 끌개(attractor)로 수렴한다. 주기적으로 반복되는 흐름의 상태는 시간이 지나면 닫힌 곡선(1차원) 근처에 머문다.
- 보존 법칙과 제약이 자유도를 줄인다. 에너지가 보존되는 계의 궤적은 에너지가 일정한 곡면 위에만 있다.

다만 가설은 가설이다. 데이터가 여러 다양체의 합집합이거나(클래스마다 다른 다양체), 차원이 위치마다 다르거나, 잡음 때문에 다양체 "근처"의 두께가 무시할 수 없을 수 있다.

:::note[분야 예시]
작은 분자의 가능한 모양(conformation)은 원자 좌표 $3N$개로 기술하지만, 결합 길이와 결합각은 거의 고정되어 있어 실제 자유도는 주로 몇 개의 비틀림 각(torsion angle)이다. 비틀림 각 하나는 원 $S^1$ 위의 값이므로, 비틀림 각이 두 개인 분자의 모양은 토러스 $S^1 \times S^1$에 가까운 다양체를 이룬다. 각을 실수로 다뤄 $179°$와 $-179°$를 먼 값으로 취급하면 이 구조가 깨진다.
:::

**내재 차원 추정.** 데이터가 놓인 다양체의 차원을 내재 차원(intrinsic dimension)이라 한다. 직접 볼 수 없으므로 데이터에서 추정한다. 원리는 "작은 이웃만 보면 다양체는 평평하다"는 것이다.

- **국소 PCA.** 각 점의 $k$-최근접 이웃에 PCA(7.6장)를 하고, 분산의 대부분을 설명하는 성분 수를 센다. 이웃이 작으면 휘어짐이 무시되어 다양체 차원만큼의 성분에 분산이 몰린다.
- **이웃 거리의 증가율.** $m$차원 다양체 위에서 반지름 $r$ 안에 든 이웃 수는 작은 $r$에서 대략 $r^m$에 비례한다. $k$번째 이웃까지의 거리가 $k$에 따라 늘어나는 속도에서 $m$을 거꾸로 추정한다. 실습의 최대우도 추정이 이 원리를 쓴다.

Swiss roll은 2차원 직사각형 시트를 3차원에서 말아 놓은 합성 데이터다. 전역 PCA와 국소 PCA를 비교한다.

```python {exec="browser" session="manifold"}
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

X, t_roll = make_swiss_roll(n_samples=1500, random_state=0)    # X (N, 3)
Xc = X - X.mean(axis=0)
s = np.linalg.svd(Xc, compute_uv=False)
print("전역 PCA 분산 비율:", np.round(s**2 / np.sum(s**2), 3))

k = 12
_, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)  # (N, k)
local = []
for nb in idx[::10]:                                           # 150개 점에서
    P = X[nb] - X[nb].mean(axis=0)                             # (k, 3)
    v = np.linalg.svd(P, compute_uv=False) ** 2
    local.append(v / v.sum())
print("국소 PCA 분산 비율 평균:", np.round(np.mean(local, axis=0), 3))
```

```output
전역 PCA 분산 비율: [0.408 0.307 0.285]
국소 PCA 분산 비율 평균: [0.671 0.327 0.002]
```

전역 PCA는 세 방향에 분산을 고르게 나눠, 데이터를 담는 선형 부분공간이 3차원이라고 말한다. 12개 이웃 안에서는 셋째 방향의 분산이 0.2%뿐이다. 국소적으로 데이터는 2차원 시트다.

::quiz{id="pca-on-swiss-roll"}

추정치는 **척도에 의존한다**. 이웃이 너무 작으면 잡음의 차원(측정 잡음은 모든 방향으로 퍼진다)이 보이고, 너무 크면 휘어짐 때문에 차원이 커 보인다. 내재 차원은 하나의 숫자라기보다 "어떤 척도에서의 차원"으로 보고, 이웃 크기를 바꿔 가며 보고한다(실습).

## 표현의 기하

신경망의 중간 표현이나 생성모델의 잠재 공간(latent space)은 보통 $\mathbb{R}^d$의 벡터로 다룬다. 그 공간에서 두 점의 유클리드 거리와 직선 보간 $(1 - t)\mathbf{z}_1 + t\mathbf{z}_2$는 계산하기 쉽다. 문제는 그것이 데이터 쪽에서 의미 있는 거리와 경로인가다.

**측지선과 직선.** 다양체 위에서 두 점을 잇는 국소적으로 가장 짧은 곡선을 측지선(geodesic)이라 하고, 그 길이가 측지 거리(geodesic distance)다. 단위구면에서 측지선은 대원(great circle)의 호이고, 두 단위벡터 $\mathbf{p}, \mathbf{q}$ 사이의 측지 거리는 사잇각 $\arccos(\mathbf{p}^\top\mathbf{q})$다. 측지선을 따라 같은 속도로 보간하는 방법이 구면 선형 보간(slerp)이다.

$$
\mathrm{slerp}(\mathbf{p}, \mathbf{q}; t) = \frac{\sin\big((1-t)\Omega\big)}{\sin\Omega}\mathbf{p} + \frac{\sin(t\Omega)}{\sin\Omega}\mathbf{q}, \qquad \Omega = \arccos(\mathbf{p}^\top\mathbf{q}).
$$

```python {exec="browser" session="manifold"}
p = np.array([1.0, 0.0, 0.0])
q = np.array([0.0, 1.0, 0.0])
ang = np.arccos(p @ q)               # 두 단위벡터 사이 각 [rad]

for t in [0.25, 0.5, 0.75]:
    lerp = (1 - t) * p + t * q
    slerp = (np.sin((1 - t) * ang) * p + np.sin(t * ang) * q) / np.sin(ang)
    print(f"t = {t:.2f}  |lerp| = {np.linalg.norm(lerp):.3f}  |slerp| = {np.linalg.norm(slerp):.3f}")
print(f"측지 거리 = {ang:.4f}  직선 거리 = {np.linalg.norm(p - q):.4f}")
```

```output
t = 0.25  |lerp| = 0.791  |slerp| = 1.000
t = 0.50  |lerp| = 0.707  |slerp| = 1.000
t = 0.75  |lerp| = 0.791  |slerp| = 1.000
측지 거리 = 1.5708  직선 거리 = 1.4142
```

직선 보간은 구면 안쪽을 가로질러 중점에서 노름이 $1/\sqrt{2}$로 줄어든다. 정규화된 embedding(예: cosine similarity로 학습한 표현)을 직선으로 보간하면 학습 중에 한 번도 본 적 없는 노름의 벡터가 만들어진다. 고차원 표준 가우시안에서 뽑은 잠재 벡터도 비슷하다. $d$차원 표준 가우시안 표본의 노름은 $\sqrt{d}$ 근처에 몰리는데, 독립인 두 표본의 중점은 노름이 약 $\sqrt{d/2}$로, 사전분포에서 거의 나오지 않는 영역에 놓인다.

::quiz{id="lerp-midpoint-norm"}

**잠재 공간 거리의 함정.** 잠재 공간이 평평한 $\mathbb{R}^d$라고 해서 그 위의 유클리드 거리가 데이터 사이의 의미 있는 거리라는 보장은 없다. decoder $g: \mathbb{R}^d \to \mathbb{R}^n$이 비선형이면, 잠재 공간의 같은 길이 $\lVert\Delta\mathbf{z}\rVert$가 데이터 공간에서는 $\lVert\mathbf{J}_g(\mathbf{z})\Delta\mathbf{z}\rVert$만큼의 변화를 만들고, 이 값은 위치와 방향에 따라 크게 다르다. 데이터 공간에서 측정한 길이로 잠재 공간의 경로를 재려면 $\mathbf{J}_g^\top\mathbf{J}_g$를 국소 내적으로 쓰는 방법(잠재 공간에 decoder가 유도하는 Riemann 계량을 주는 것)이 있다. 이 장에서는 "잠재 거리는 decoder의 Jacobian을 거쳐야 데이터 거리가 된다"는 점만 기억한다.

## 차원 개념의 구분

연구 논문과 대화에서 "차원"은 적어도 네 가지 뜻으로 쓰인다. 서로 같을 때도 있지만, 다를 때가 더 흥미롭다.

| 개념 | 정의 | 무엇을 재는가 |
|---|---|---|
| 벡터공간 차원 | basis의 원소 수 (7.1장) | 데이터를 담는 그릇의 크기, 예: $\mathbb{R}^{4096}$ |
| 행렬 rank | 열공간의 차원 (7.3장) | 데이터를 담는 가장 작은 **선형** 부분공간의 차원 |
| 잠재변수 차원 | 모델이 쓰는 $\mathbf{z}$의 길이 | 설계자가 고른 모델의 용량 |
| 다양체(내재) 차원 | 국소 좌표의 개수 | 데이터의 **국소적인** 자유도 |

데이터가 선형 부분공간 위에 있으면 rank와 다양체 차원이 같다. 데이터가 휘어 있으면 rank가 더 크다. 다음 셀은 곡선 $(t, t^2, t^3)$ 위의 점으로 이것을 확인한다.

```python {exec="browser" session="manifold"}
t = np.linspace(-1, 1, 200)
curve = np.stack([t, t**2, t**3], axis=1)                      # (200, 3)
C = curve - curve.mean(axis=0)
print("특이값          :", np.round(np.linalg.svd(C, compute_uv=False), 2))
print("데이터 행렬 rank:", np.linalg.matrix_rank(C))

J_curve = np.stack([np.ones_like(t), 2 * t, 3 * t**2], axis=1)  # 각 점의 Jacobian (200, 3) = 200개의 (3, 1)
print("곡선 Jacobian rank (모든 점):", sorted({int(np.linalg.matrix_rank(j[:, None])) for j in J_curve}))
```

```output
특이값          : [9.66 4.26 1.84]
데이터 행렬 rank: 3
곡선 Jacobian rank (모든 점): [1]
```

데이터 행렬의 rank는 3이지만 곡선은 매개변수 하나로 정해지는 1차원 다양체이고, 모든 점에서 Jacobian의 rank는 1이다. 잠재변수 차원은 여기에 또 다른 축을 더한다. 잠재 차원 32인 모델이 이 곡선을 완벽히 재구성해도 곡선의 자유도는 여전히 1이다. 반대로 잠재 차원이 내재 차원보다 작으면 재구성이 불가능하거나, decoder가 다양체를 접어 서로 먼 점을 같은 잠재 좌표에 겹쳐 놓아야 한다. 원 $S^1$을 1차원 잠재 공간 $\mathbb{R}$에 연속적으로 일대일로 펼칠 수는 없다는 점(어딘가를 끊어야 한다)이 그 예다.

::quiz{id="curve-rank-vs-dimension"}

## 실습

Swiss roll 데이터에서 PCA와 비선형 방법의 차이를 확인하고 내재 차원을 추정한다. 코드는 `labs/linear-algebra/manifolds-and-geometry/swiss_roll.py`다.

1. PCA와 Isomap(`sklearn.manifold.Isomap`)으로 각각 2차원 embedding을 만든다. Isomap은 최근접 이웃 그래프 위의 최단 경로 길이로 측지 거리를 근사한 뒤, 그 거리를 보존하는 좌표를 찾는다. 시트가 말린 방향의 좌표 $t$를 embedding이 얼마나 보존하는지 Spearman 순위 상관으로 잰다.
2. 잡음 크기와 이웃 수 $k$를 바꿔 가며 국소 PCA(분산 95%를 설명하는 성분 수)와 이웃 거리 비율 기반 최대우도 추정으로 내재 차원을 추정한다. 최대우도 추정치는 점마다 $\hat{m}(\mathbf{x}) = \big[\frac{1}{k-1}\sum_{j<k}\log(T_k(\mathbf{x})/T_j(\mathbf{x}))\big]^{-1}$($T_j$는 $j$번째 이웃까지의 거리)를 계산해 평균한다.

집필 시점의 실행 결과는 다음과 같다. 그림은 `outputs/swiss_roll.png`에 저장된다.

```output
말린 방향 좌표 t와의 |Spearman 상관| (2차원 embedding 성분 중 최대)
  PCA    0.222
  Isomap 1.000

내재 차원 추정 (잡음 표준편차 × 이웃 수 k)
  noise    k   국소 PCA(95%)   거리 비율 MLE
   0.0     8         2.00           2.29
   0.0    20         2.00           2.05
   0.0    50         2.01           1.97
   0.5     8         2.77           2.98
   0.5    20         2.69           2.45
   0.5    50         2.29           2.19
   1.0     8         2.87           3.34
   1.0    20         2.99           2.86
   1.0    50         3.00           2.53
```

결과를 읽는 법은 세 가지다.

- **PCA는 시트를 펴지 못한다.** 선형 사영은 말린 시트를 눌러 겹치게 하므로, 말린 방향의 순서가 거의 보존되지 않는다(0.222). 측지 거리를 쓰는 Isomap은 순서를 완전히 보존한다(1.000).
- **잡음이 없으면 두 추정 방법 모두 2에 가깝다.** 다만 최대우도 추정은 이웃이 작을 때(k = 8) 2.29로 약간 크게 나온다. 추정 방법에도 편향이 있다.
- **잡음이 있으면 추정치는 이웃 크기에 따라 달라진다.** 잡음 표준편차 0.5에서 작은 이웃은 잡음 방향까지 차원으로 세어 3에 가깝게, 큰 이웃은 2에 가깝게 추정한다. 잡음 1.0에서 국소 PCA는 이웃이 커질수록 오히려 3에 가까워지는데, 이웃이 커지면 시트의 휘어짐까지 담기 때문이다. "내재 차원은 얼마인가"라는 질문에는 "어떤 척도에서"를 함께 답해야 한다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="so3-exp"}

::exercise{id="sphere-exp-map"}

## 실패 사례

- **잠재 차원을 데이터의 실제 자유도로 단정하는 것.** 잠재 차원 16인 모델이 잘 동작했다고 "이 시스템의 자유도는 16"이라고 쓰면 안 된다. 잠재 차원은 설계 선택이고 자유도의 상한조차 아닐 수 있다(재구성 오차에 거의 영향을 주지 않는 자유도는 버려진다). 자유도를 주장하려면 잠재 차원을 바꿔 가며 재구성 오차가 어디서 꺾이는지 보거나, 내재 차원을 여러 척도에서 추정해 함께 보고한다.
- **latent space의 유클리드 거리를 의미 있는 거리로 가정하는 것.** 잠재 공간에서 가까운 두 점이 데이터에서도 비슷하다는 보장은 없다. decoder의 Jacobian이 위치마다 다르게 늘이고 줄이기 때문이다. 잠재 공간의 최근접 이웃으로 "비슷한 물질"이나 "비슷한 기후 패턴"을 찾는 분석은, 데이터 공간에서 직접 측정한 거리로 검증한다.
- **휘어진 공간의 값을 산술 평균하는 것.** 각도($359°$와 $1°$의 평균은 $180°$가 아니다), 회전 행렬, 단위벡터를 성분별로 평균하면 집합 밖의 값이 나온다. 원 위의 각은 단위벡터 $(\cos\varphi, \sin\varphi)$로 바꿔 평균한 뒤 방향을 다시 구하고, 회전은 접공간에서 평균한 뒤 지수사상으로 돌아온다.
- **chart의 특이점을 데이터의 특이점으로 해석하는 것.** 오일러 각이 gimbal lock 근처에서 크게 튀는 것은 좌표의 문제다. 각이 튄다고 물체의 자세가 급변한 것은 아니다.

## 통과 기준

- 곡선 $(t, t^2, t^3)$ 위의 데이터, 단위구면 위의 embedding, Swiss roll을 예로 들어 행렬 rank·벡터공간 차원·잠재 차원·다양체 차원이 각각 얼마이고 언제 서로 다른지 설명한다.
- 구면좌표의 Jacobian 열공간이 접평면임을 계산으로 보이고, 북극에서 rank가 떨어지는 것이 chart의 결함임을 설명한다.
- 연습문제 두 개(`so3-exp`, `sphere-exp-map`)의 테스트를 통과한다.
