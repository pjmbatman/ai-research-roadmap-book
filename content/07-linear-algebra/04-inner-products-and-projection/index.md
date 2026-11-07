---
id: linear-algebra/inner-products-and-projection
title: "내적·노름·직교성·사영"
title_en: "Inner Products, Norms, and Projection"
tier: core
status: draft
prerequisites: [linear-algebra/rank-and-subspaces]
---

# 내적·노름·직교성·사영

7.1장의 벡터공간에는 덧셈과 스칼라배만 있었다. 길이도, 거리도, 각도도 없었다. 그런데 연구의 거의 모든 질문에는 "얼마나 가까운가"가 들어 있다. 예측이 측정에 얼마나 가까운가, 두 분자의 embedding이 얼마나 비슷한가, 모델이 데이터를 얼마나 잘 설명하는가. 이 장은 벡터공간에 **내적**을 더해 길이와 각도를 정의하고, 그 위에서 가장 많이 쓰이는 계산인 **사영**과 **최소제곱**을 유도한다.

7.3장에서 잡음이 섞인 과결정 시스템에는 정확한 해가 없고, 미결정 시스템에는 해가 너무 많다는 것을 보았다. 이 장은 두 경우에 각각 "가장 가까운 해"와 "가장 짧은 해"라는 답을 주고, 둘을 하나로 묶는 pseudoinverse로 마무리한다.

## 학습 목표

- 내적과 노름의 종류를 구분한다
- 사영과 최소제곱을 유도한다
- pseudoinverse를 사용한다

## 내적과 노름

**왜 필요한가.** 손실함수, 거리 기반 분류, 유사도 검색, 수렴 판정은 모두 "크기"를 잰다. 어떤 크기를 쓰느냐에 따라 결과가 달라진다. 이상치 몇 개에 민감한 척도도 있고, 가장 큰 오차 하나만 보는 척도도 있다. 척도를 고르려면 먼저 종류를 구분해야 한다.

**내적.** 실벡터공간 $V$ 위의 **내적**(inner product) $\langle \cdot, \cdot \rangle$은 두 벡터를 실수로 보내는 함수로, 다음을 만족한다.

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle, \qquad
\langle \alpha\mathbf{x} + \beta\mathbf{y}, \mathbf{z} \rangle = \alpha\langle \mathbf{x}, \mathbf{z} \rangle + \beta\langle \mathbf{y}, \mathbf{z} \rangle, \qquad
\langle \mathbf{x}, \mathbf{x} \rangle > 0 \ (\mathbf{x} \neq \mathbf{0}).
$$

대칭성, 선형성, 양의 정부호성이다. 가장 익숙한 예는 $\mathbb{R}^n$의 점곱 $\mathbf{x}^\top\mathbf{y} = \sum_i x_i y_i$이다. 그 밖에도

- **가중 내적** $\langle \mathbf{x}, \mathbf{y} \rangle_{\mathbf{W}} = \mathbf{x}^\top\mathbf{W}\mathbf{y}$: $\mathbf{W}$가 대칭 양의 정부호 행렬(7.5장)이면 내적이다. 성분마다 측정 정밀도가 다를 때 쓴다.
- **함수의 내적** $\langle f, g \rangle = \int_a^b f(t)g(t)\,dt$: $C[a, b]$ 위의 내적이다. Fourier 급수와 Legendre 다항식의 "직교성"이 이 내적에 대한 것이다.

내적이 있으면 길이 $\lVert\mathbf{x}\rVert = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$와 각도를 정의할 수 있다. 각도는 **Cauchy–Schwarz 부등식** $|\langle \mathbf{x}, \mathbf{y} \rangle| \le \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert$ 덕분에 정의된다. 양변의 비가 $-1$과 $1$ 사이이므로 $\cos\theta = \langle \mathbf{x}, \mathbf{y} \rangle / (\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert)$로 둘 수 있다. $\langle \mathbf{x}, \mathbf{y} \rangle = 0$이면 두 벡터는 **직교**한다.

**노름.** 길이의 개념만 따로 떼어 낸 것이 **노름**(norm)이다. $\lVert\mathbf{x}\rVert \ge 0$이고 $\mathbf{x} = \mathbf{0}$일 때만 0, $\lVert\alpha\mathbf{x}\rVert = |\alpha|\,\lVert\mathbf{x}\rVert$, 삼각부등식 $\lVert\mathbf{x} + \mathbf{y}\rVert \le \lVert\mathbf{x}\rVert + \lVert\mathbf{y}\rVert$을 만족하면 노름이다. 모든 내적은 노름을 만들지만, 모든 노름이 내적에서 오는 것은 아니다. 대표적인 벡터 노름은 다음 셋이다.

| 노름 | 정의 | 단위 공(2차원) | 쓰임 |
|---|---|---|---|
| L1 | $\sum_i \lvert x_i \rvert$ | 마름모 | MAE, LASSO, 희소성 |
| L2 | $\sqrt{\sum_i x_i^2}$ | 원 | MSE, ridge, 유클리드 거리 |
| L∞ | $\max_i \lvert x_i \rvert$ | 정사각형 | 최대 오차, 허용오차 검사 |

세 노름 중 내적에서 오는 것은 L2뿐이다. 단위 공의 모양은 정규화가 어떤 해를 선호하는지 알려 준다. L1 공은 축 위에 뾰족한 꼭짓점이 있어, L1 정규화는 일부 성분이 정확히 0인 희소한 해를 선호한다(16.2장).

**행렬 노름.** 행렬도 벡터공간의 원소이므로 노름을 잴 수 있다. 세 가지가 자주 쓰인다.

- **Frobenius 노름** $\lVert\mathbf{A}\rVert_F = \sqrt{\sum_{ij} a_{ij}^2}$: 행렬을 긴 벡터로 펴서 잰 L2 노름이다. 가중치 감쇠(weight decay)가 이 노름의 제곱을 벌점으로 준다.
- **spectral 노름** $\lVert\mathbf{A}\rVert_2 = \max_{\mathbf{x} \neq \mathbf{0}} \lVert\mathbf{A}\mathbf{x}\rVert / \lVert\mathbf{x}\rVert$: 행렬이 벡터를 최대 몇 배까지 늘이는가. 층의 Lipschitz 상수, 반복 적용의 안정성에 쓰인다.
- **nuclear 노름** $\lVert\mathbf{A}\rVert_*$: 특이값의 합. rank의 볼록한 대리 척도로, 저랭크 행렬 복원에 쓰인다.

세 노름은 모두 특이값 $\sigma_i$(7.6장)로 쓸 수 있다. Frobenius는 $\sqrt{\sum \sigma_i^2}$, spectral은 $\sigma_1$, nuclear는 $\sum \sigma_i$이다. 아래 행렬 $\mathbf{M}$의 특이값은 $3\sqrt{5} \approx 6.708$과 $\sqrt{5} \approx 2.236$이다.

```python {exec="browser" session="proj"}
import numpy as np

x = np.array([3.0, -4.0, 0.0, 1.0])
print("L1 =", np.linalg.norm(x, 1), " L2 =", round(np.linalg.norm(x), 3), " Linf =", np.linalg.norm(x, np.inf))

M = np.array([[3.0, 0.0],
              [4.0, 5.0]])                                   # (2, 2)
print("Frobenius =", round(np.linalg.norm(M, "fro"), 3),
      " spectral =", round(np.linalg.norm(M, 2), 3),
      " nuclear =", round(np.linalg.norm(M, "nuc"), 3))

s = np.array([0.1, 0.8, 0.3, 0.05])                          # 스펙트럼 하나
t = 2.0 * s                                                  # 농도가 두 배인 같은 물질
cos = s @ t / (np.linalg.norm(s) * np.linalg.norm(t))
print("cosine =", round(cos, 6), " 유클리드 거리 =", round(np.linalg.norm(s - t), 3))
```

```output
L1 = 8.0  L2 = 5.099  Linf = 4.0
Frobenius = 7.071  spectral = 6.708  nuclear = 8.944
cosine = 1.0  유클리드 거리 = 0.862
```

**거리와 유사도.** 마지막 두 줄은 같은 물질의 흡광 스펙트럼을 농도만 두 배로 측정한 경우다. 모양이 같으므로 cosine similarity는 정확히 1이지만, 유클리드 거리는 0이 아니다. **cosine similarity**는 크기를 무시하고 방향만 비교한다. 두 벡터를 단위 길이로 정규화한 뒤의 점곱과 같고, 정규화된 벡터 사이에서는 $\lVert\hat{\mathbf{x}} - \hat{\mathbf{y}}\rVert^2 = 2 - 2\cos\theta$이므로 유클리드 거리와 같은 순서를 준다. 어느 쪽을 쓸지는 크기가 의미 있는 정보인가로 정한다. 스펙트럼의 모양으로 물질을 식별하려면 cosine이, 농도까지 맞추려면 유클리드 거리가 맞다. 텍스트·분자 embedding 검색에서 cosine을 흔히 쓰는 것도 크기보다 방향에 의미를 두기 때문이다.

::quiz{id="cosine-vs-euclid"}

## 직교성

**왜 필요한가.** 7.1장에서 일반 basis의 좌표를 구하려면 연립방정식을 풀어야 했다. basis 벡터가 서로 직교하고 길이가 1이면 이 계산이 내적 몇 번으로 끝난다. 수치적으로도 가장 안정적인 basis다.

**정의.** 벡터 $\mathbf{q}_1, \dots, \mathbf{q}_k$가 $\langle \mathbf{q}_i, \mathbf{q}_j \rangle = 0$ ($i \neq j$)이고 $\lVert\mathbf{q}_i\rVert = 1$이면 **정규직교**(orthonormal)라 한다. 정규직교 basis에서 좌표는

$$
\mathbf{v} = \sum_i c_i\mathbf{q}_i, \qquad c_i = \langle \mathbf{q}_i, \mathbf{v} \rangle
$$

이다. 양변에 $\mathbf{q}_j$를 내적하면 $j$번째 항만 남기 때문이다. 연립방정식이 필요 없다.

**Gram–Schmidt.** 독립인 벡터 $\mathbf{a}_1, \dots, \mathbf{a}_n$에서 같은 공간을 span하는 정규직교 벡터를 만드는 절차다. 각 벡터에서 앞서 만든 방향 성분을 빼고 길이를 1로 맞춘다.

$$
\tilde{\mathbf{a}}_k = \mathbf{a}_k - \sum_{j < k} \langle \mathbf{q}_j, \mathbf{a}_k \rangle\,\mathbf{q}_j, \qquad \mathbf{q}_k = \frac{\tilde{\mathbf{a}}_k}{\lVert\tilde{\mathbf{a}}_k\rVert}.
$$

빼는 항 $\langle \mathbf{q}_j, \mathbf{a}_k \rangle\mathbf{q}_j$는 $\mathbf{a}_k$를 $\mathbf{q}_j$ 방향으로 사영한 것이다(다음 절). 이 과정을 행렬로 쓰면 $\mathbf{A} = \mathbf{Q}\mathbf{R}$이다. $\mathbf{Q}$의 열은 정규직교이고, $\mathbf{R}$은 계수 $\langle \mathbf{q}_j, \mathbf{a}_k \rangle$를 담은 상삼각행렬이다. 이를 **QR 분해**라 한다. 수식 그대로의 구현(classical Gram–Schmidt)은 반올림 오차로 직교성을 잃기 쉬워, 실제 라이브러리는 같은 결과를 더 안정적으로 내는 방법(Householder 변환 등, 9.3장)을 쓴다.

**직교행렬.** 정사각행렬 $\mathbf{Q}$의 열이 정규직교이면 $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, 즉 $\mathbf{Q}^{-1} = \mathbf{Q}^\top$이다. 이런 행렬을 **직교행렬**이라 한다. 직교행렬은 내적을 보존한다.

$$
\langle \mathbf{Q}\mathbf{u}, \mathbf{Q}\mathbf{v} \rangle = \mathbf{u}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{v} = \mathbf{u}^\top\mathbf{v}.
$$

따라서 길이와 각도도 보존한다. 기하적으로는 회전과 반사뿐이다($\det\mathbf{Q} = \pm 1$).

```python {exec="browser" session="proj"}
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))                 # (4, 4) 직교행렬
u, v = rng.normal(size=4), rng.normal(size=4)
print("Q^T Q = I ?      ", np.allclose(Q.T @ Q, np.eye(4)))
print("|Qu|, |u|        ", round(np.linalg.norm(Q @ u), 6), round(np.linalg.norm(u), 6))
print("<Qu,Qv>, <u,v>   ", round((Q @ u) @ (Q @ v), 6), round(u @ v, 6))
```

```output
Q^T Q = I ?       True
|Qu|, |u|         1.285511 1.285511
<Qu,Qv>, <u,v>    -0.269617 -0.269617
```

**연구에서의 의미.** 직교행렬은 오차를 키우지 않는다. 입력의 반올림 오차가 $\lVert\mathbf{Q}\boldsymbol{\delta}\rVert = \lVert\boldsymbol{\delta}\rVert$로 그대로 전달되기 때문이다. 그래서 수치 알고리즘은 가능하면 직교 변환으로 문제를 바꾼다(QR, SVD). 신경망에서도 가중치를 직교행렬로 초기화하거나 제약하는 기법이 있는데, 깊이 쌓아도 신호의 크기가 폭발하거나 사라지지 않게 하려는 것이다(7.5장의 안정성 논의와 연결된다).

## 사영

**왜 필요한가.** 7.3장에서 잡음 섞인 $\mathbf{b}$는 column space 밖에 있어 $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해가 없었다. 할 수 있는 최선은 column space 안에서 $\mathbf{b}$에 **가장 가까운 점**을 찾는 것이다. 그 점이 사영이다.

**직선으로의 사영.** 벡터 $\mathbf{a}$ 방향의 직선 위에서 $\mathbf{b}$에 가장 가까운 점 $\mathbf{p} = \hat{x}\mathbf{a}$는, 오차 $\mathbf{b} - \mathbf{p}$가 $\mathbf{a}$와 직교하는 점이다. $\mathbf{a}^\top(\mathbf{b} - \hat{x}\mathbf{a}) = 0$을 풀면

$$
\hat{x} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}, \qquad \mathbf{p} = \frac{\mathbf{a}\mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{b}.
$$

**부분공간으로의 사영.** 열이 독립인 $m \times n$ 행렬 $\mathbf{A}$의 column space로 사영한다. 사영점은 $\mathbf{p} = \mathbf{A}\hat{\mathbf{x}}$ 꼴이고, 오차 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$가 모든 열과 직교해야 한다.

$$
\mathbf{A}^\top(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0}
\;\Longrightarrow\;
\hat{\mathbf{x}} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}, \qquad
\mathbf{P} = \mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top.
$$

$\mathbf{A}^\top\mathbf{A}$가 가역인 것은 7.3장의 $\operatorname{rank}(\mathbf{A}^\top\mathbf{A}) = \operatorname{rank}(\mathbf{A}) = n$ 덕분이다. 오차가 모든 열과 직교한다는 조건은 "오차가 left null space $N(\mathbf{A}^\top)$에 있다"는 말과 같다.

**사영행렬의 성질.** $\mathbf{P}$는 두 성질로 특징지어진다.

- $\mathbf{P}^2 = \mathbf{P}$: 이미 부분공간 안에 있는 점은 다시 사영해도 그대로다.
- $\mathbf{P}^\top = \mathbf{P}$: 오차가 부분공간과 **직교**하는 사영(직교사영)이다.

$\mathbf{I} - \mathbf{P}$도 사영행렬이며, 직교여공간(여기서는 $N(\mathbf{A}^\top)$)으로 사영한다. $\mathbf{b} = \mathbf{P}\mathbf{b} + (\mathbf{I} - \mathbf{P})\mathbf{b}$는 7.3장의 분해 $\mathbb{R}^m = C(\mathbf{A}) \oplus N(\mathbf{A}^\top)$을 구체적으로 계산한 것이다. 열이 정규직교인 $\mathbf{Q}$라면 $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$이므로 $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$로 단순해진다.

```python {exec="browser" session="proj"}
A = np.array([[1.0, 0.0],
              [1.0, 1.0],
              [1.0, 2.0]])                                   # (3, 2), 열이 평면을 span
P = A @ np.linalg.solve(A.T @ A, A.T)                        # (3, 3) 사영행렬
b = np.array([6.0, 0.0, 0.0])
p = P @ b
print("P b       =", np.round(p, 4))
print("P^2 = P ? ", np.allclose(P @ P, P), "  P^T = P ?", np.allclose(P.T, P))
print("A^T (b-p) = 0 ?", np.allclose(A.T @ (b - p), 0))
```

```output
P b       = [ 5.  2. -1.]
P^2 = P ?  True   P^T = P ? True
A^T (b-p) = 0 ? True
```

$\mathbf{b} = (6, 0, 0)$의 사영은 $(5, 2, -1)$이고, 오차 $(1, -2, 1)$은 두 열 $(1, 1, 1)$, $(0, 1, 2)$와 모두 직교한다.

::widget{name="projection-lstsq"}

## 최소제곱

**정의.** 과결정 시스템 $\mathbf{A}\mathbf{x} \approx \mathbf{b}$의 **최소제곱 해**(least-squares solution)는 잔차의 제곱합을 최소화하는 $\hat{\mathbf{x}}$이다.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert^2.
$$

**유도.** 두 가지 방법으로 같은 식에 도착한다.

- **미분.** $L(\mathbf{x}) = (\mathbf{A}\mathbf{x} - \mathbf{b})^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$의 gradient는 $2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$이다(8.2장). 이를 0으로 두면 **정규방정식**(normal equations) $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$를 얻는다.
- **기하.** $\mathbf{A}\mathbf{x}$는 column space 위를 움직이고, 그중 $\mathbf{b}$에 가장 가까운 점은 사영 $\mathbf{P}\mathbf{b}$다. 앞 절의 직교 조건 $\mathbf{A}^\top(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0}$이 곧 정규방정식이다.

기하적 해석이 주는 핵심은 **잔차의 직교성**이다. 최소제곱 잔차 $\mathbf{r} = \mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$는 모든 특징 열과 직교한다. 모델에 절편 열(모두 1인 열)이 있으면 잔차의 합이 0이라는 것도 여기서 나온다. 반대로 잔차가 어떤 특징과 상관되어 있다면 아직 최소제곱 해에 도달하지 않았거나, 그 특징이 모델에 없다는 뜻이다.

**선형회귀는 사영이다.** 특징 행렬 $\mathbf{X}$로 관측 $\mathbf{y}$를 설명하는 선형회귀의 예측 $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$는 $\mathbf{y}$를 $\mathbf{X}$의 column space에 사영한 것이다. 통계에서 이 사영행렬 $\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$을 hat 행렬이라 부르고(12.6장), 그 대각 성분으로 각 데이터 점이 자기 예측에 미치는 영향(leverage)을 잰다.

예로 용수철의 늘어난 길이와 힘을 측정해 Hooke 법칙 $F = kx + c$의 용수철 상수 $k$를 추정한다($c$는 영점 보정 오프셋). 세 가지 방법으로 푼다.

```python {exec="browser" session="proj"}
rng = np.random.default_rng(1)
extension_m = np.linspace(0.01, 0.10, 12)                    # (12,) 늘어난 길이
force_n = 250.0 * extension_m + 0.3 + rng.normal(scale=0.4, size=12)
X = np.column_stack([extension_m, np.ones(12)])              # (12, 2)

w_ne = np.linalg.solve(X.T @ X, X.T @ force_n)               # 정규방정식
Qx, R = np.linalg.qr(X)                                      # X = Q R
w_qr = np.linalg.solve(R, Qx.T @ force_n)                    # R w = Q^T y
w_ls = np.linalg.lstsq(X, force_n)[0]
print("k [N/m], c [N]  정규방정식:", np.round(w_ne, 3), " QR:", np.round(w_qr, 3), " lstsq:", np.round(w_ls, 3))
r = force_n - X @ w_ls
print("max |X^T r| < 1e-10:", np.abs(X.T @ r).max() < 1e-10)
```

```output
k [N/m], c [N]  정규방정식: [250.205   0.383]  QR: [250.205   0.383]  lstsq: [250.205   0.383]
max |X^T r| < 1e-10: True
```

데이터를 만들 때 쓴 참값은 $k = 250$ N/m, $c = 0.3$ N이다. 세 방법이 같은 답을 주고, 잔차는 두 열과 반올림 오차 수준까지 직교한다. QR 방법은 $\mathbf{A} = \mathbf{Q}\mathbf{R}$을 정규방정식에 넣어 얻는다. $\mathbf{R}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{R}\hat{\mathbf{x}} = \mathbf{R}^\top\mathbf{Q}^\top\mathbf{b}$에서 $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$이고 $\mathbf{R}^\top$이 가역이므로 $\mathbf{R}\hat{\mathbf{x}} = \mathbf{Q}^\top\mathbf{b}$이다. $\mathbf{A}^\top\mathbf{A}$를 만들지 않는다는 점이 중요하다(실습).

::quiz{id="residual-orthogonal"}

:::note[분야 예시]
분석화학의 검량선(calibration curve)은 표준 시료의 농도와 신호(예: 흡광도)로 직선을 맞추는 최소제곱 문제다. 측정 정밀도가 농도마다 다르면(높은 농도에서 잡음이 크면) 모든 점을 같은 무게로 다루는 것이 맞지 않는다. 이때 각 점의 잡음 분산 $\sigma_i^2$의 역수를 가중치로 한 **가중 최소제곱** $\sum_i (b_i - \mathbf{a}_i^\top\mathbf{x})^2 / \sigma_i^2$를 쓴다. 이는 가중 내적 $\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{W}} = \mathbf{u}^\top\mathbf{W}\mathbf{v}$, $\mathbf{W} = \operatorname{diag}(1/\sigma_i^2)$에 대한 사영이다. 잡음이 가우시안이면 이 해가 최대우도 추정과 같다(12.2장). 어떤 내적으로 사영하느냐가 곧 측정 오차에 대한 가정이다.
:::

## pseudoinverse

**왜 필요한가.** 정규방정식은 $\mathbf{A}^\top\mathbf{A}$가 가역일 때, 즉 열이 독립일 때만 해가 하나다. 미결정 시스템이나 rank가 부족한 회귀(7.3장의 섭씨·화씨 예)에서는 최소제곱 해가 무수히 많다. 그중 하나를 일관되게 고르는 규칙이 필요하다.

**정의.** 최소제곱 해들 가운데 **노름이 가장 작은 해**를 $\mathbf{x}^+$라 하고, $\mathbf{b}$를 $\mathbf{x}^+$로 보내는 행렬을 $\mathbf{A}$의 **pseudoinverse**(Moore–Penrose 유사역행렬) $\mathbf{A}^+$라 한다. 경우별로 다음과 같다.

| 경우 | $\mathbf{A}^+$ | 의미 |
|---|---|---|
| 정사각 가역 | $\mathbf{A}^{-1}$ | 보통의 역행렬 |
| full column rank ($r = n$) | $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ | 유일한 최소제곱 해 |
| full row rank ($r = m$) | $\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}$ | 무수한 해 중 최소 노름 해 |
| 일반 | SVD로 계산(7.6장) | 최소제곱 해 중 최소 노름 해 |

**직관.** 해 전체는 특수해에 null space를 더한 아핀 집합이다(7.3장). 각 해는 row space 성분과 null space 성분으로 나뉘는데, null space 성분은 $\mathbf{A}\mathbf{x}$에 영향을 주지 않고 노름만 키운다. 그래서 최소 노름 해는 null space 성분이 0인 해, 즉 **row space 안에 있는 유일한 해**다. full row rank 공식이 $\mathbf{x}^+ = \mathbf{A}^\top(\cdots)$ 꼴인 것도 $\mathbf{x}^+$가 행들의 선형결합이기 때문이다.

방정식 하나 $x_1 + x_2 = 2$로 확인한다. 해는 직선 위의 모든 점이고, 그중 원점에 가장 가까운 점은 $(1, 1)$이다.

```python {exec="browser" session="proj"}
A1 = np.array([[1.0, 1.0]])                                  # (1, 2) 방정식 x1 + x2 = 2
b1 = np.array([2.0])
x_pinv = np.linalg.pinv(A1) @ b1
print("pinv 해 =", x_pinv, " 노름 =", round(np.linalg.norm(x_pinv), 4))
for x_other in [np.array([2.0, 0.0]), np.array([3.0, -1.0])]:
    print("다른 해 =", x_other, " 노름 =", round(np.linalg.norm(x_other), 4), " A x =", A1 @ x_other)
```

```output
pinv 해 = [1. 1.]  노름 = 1.4142
다른 해 = [2. 0.]  노름 = 2.0  A x = [2.]
다른 해 = [ 3. -1.]  노름 = 3.1623  A x = [2.]
```

세 해 모두 방정식을 만족하지만 pseudoinverse 해의 노름이 가장 작다. $(1, 1)$은 행 $(1, 1)$의 배수이므로 row space 안에 있다. 7.3장 실습에서 `lstsq`가 준 해와 ridge 해가 $\lambda \to 0$에서 수렴한 해가 바로 이 최소 노름 해다.

::quiz{id="pinv-min-norm"}

**성질과 주의점.** $\mathbf{A}\mathbf{A}^+$는 column space로의 사영, $\mathbf{A}^+\mathbf{A}$는 row space로의 사영이다. 그러나 $\mathbf{A}^+\mathbf{A} = \mathbf{I}$는 열이 독립일 때만 성립한다. 일반적으로 pseudoinverse는 역행렬이 아니다. 또 실무에서 `np.linalg.pinv`는 아주 작은 특이값을 0으로 보고 버린다. 그 기준은 허용오차 인자로 조절하며, 인자 이름은 NumPy 버전에 따라 다를 수 있으므로 사용하는 버전의 문서를 확인한다. 이 허용오차가 7.3장의 수치적 rank를 정하고, 잡음 수준의 방향을 버린다는 점에서 일종의 정규화 역할을 한다. 역문제에서 "pseudoinverse로 풀었다"고 할 때는 이 허용오차가 결과를 크게 바꿀 수 있으므로 함께 보고한다(24.7장).

## 실습

최소제곱 해를 정규방정식·QR·pseudoinverse(SVD 기반 `lstsq`)로 각각 구하고 결과와 수치 안정성을 비교한다. 코드는 `labs/linear-algebra/inner-products-and-projection/lstsq_stability.py`다. 구간 $[0, 1]$의 50개 점에서 다항식 회귀의 설계 행렬(Vandermonde 행렬, 열이 $1, t, t^2, \dots$)을 만들고, 참 계수가 모두 1인 데이터를 **잡음 없이** 만든다. 그러면 정확한 답을 알고 있으므로, 각 방법이 계수를 얼마나 정확히 되찾는지 잴 수 있다. 집필 시점의 실행 결과는 다음과 같다(마지막 자릿수는 컴퓨터의 수치 라이브러리에 따라 조금 다를 수 있다).

```output
  차수   cond(X) cond(X^T X)       정규방정식        QR     lstsq
   3   1.2e+02     1.4e+04     8.4e-13   1.9e-15   1.5e-15
   6   2.0e+04     3.9e+08     2.7e-09   6.2e-14   5.4e-13
   9   3.6e+06     1.3e+13     2.4e-04   4.4e-11   1.5e-10
  12   6.8e+08     8.5e+17     1.1e+01   8.4e-09   1.6e-08
(오차 = |w - w_true| / |w_true|, 계수의 상대 오차)
```

**조건수**(condition number) $\kappa(\mathbf{A}) = \sigma_{\max} / \sigma_{\min}$은 입력의 작은 상대 오차가 해에서 최대 몇 배로 커질 수 있는지를 잰다(9.2장). 결과를 읽는 법은 세 가지다.

- **$\kappa(\mathbf{X}^\top\mathbf{X}) \approx \kappa(\mathbf{X})^2$이다.** 차수 3에서 $1.2 \times 10^2$의 제곱은 약 $1.4 \times 10^4$이다. 차수 12에서는 제곱하면 $4.6 \times 10^{17}$인데 표에는 $8.5 \times 10^{17}$이 나온다. $\mathbf{X}^\top\mathbf{X}$를 만드는 계산 자체가 반올림 오차에 묻혀 조건수조차 정확히 잴 수 없는 영역이다.
- **정규방정식은 오차가 조건수의 제곱에 비례해 커진다.** 차수 9에서 이미 계수의 상대 오차가 $10^{-4}$이고, 차수 12에서는 1을 넘어 계수가 완전히 틀린다. float64의 정밀도(약 $10^{-16}$)에 $\kappa^2 \approx 10^{18}$을 곱하면 오차가 1을 넘는 것이 당연하다.
- **QR과 SVD는 $\mathbf{X}$를 직접 다뤄 오차가 대략 조건수에 비례한다.** 같은 차수 12에서도 $10^{-8}$ 수준이다. 예측값 $\mathbf{X}\mathbf{w}$의 정확도는 계수보다 훨씬 덜 나빠질 수 있지만, 계수 자체를 해석하는 과학 문제(물리 상수 추정 등)에서는 이 차이가 결정적이다.

이어서 두 연습문제를 푼다. 하나는 사영을 반복해 정규직교 basis를 만들고, 다른 하나는 미결정 시스템의 최소 노름 해를 구한다.

::exercise{id="gram-schmidt"}

::exercise{id="min-norm-solution"}

## 실패 사례

- **정규방정식을 그대로 풀어 조건수를 제곱하는 것.** `np.linalg.solve(X.T @ X, X.T @ y)`는 짧고 빠르지만, $\kappa(\mathbf{X})$가 $10^{6}$만 넘어도 float64에서 계수의 유효숫자 대부분을 잃는다. 고차 다항식, 거의 종속인 특징, 단위가 크게 다른 특징이 흔한 원인이다. 기본값으로 `np.linalg.lstsq`(SVD)나 QR을 쓰고, 특징을 표준화해 조건수를 줄인다.
- **스케일이 다른 특징에 유클리드 거리를 쓰는 것.** 온도(K, 수백 범위)와 압력(Pa, $10^5$ 범위)을 그대로 벡터로 묶으면 유클리드 거리는 사실상 압력 차이만 잰다. k-최근접 이웃, 군집화, RBF 커널이 모두 영향을 받는다. 특징을 표준화하거나, 측정 불확도에 맞춘 가중 내적을 쓴다. 어떤 거리를 쓰는지가 곧 "어떤 차이를 중요하게 보는가"에 대한 선택이다.
- **pseudoinverse 해를 유일한 해로 해석하는 것.** 미결정 시스템에서 $\mathbf{A}^+\mathbf{b}$는 "데이터가 정한 해"가 아니라 "최소 노름이라는 기준이 고른 해"다. null space 방향의 성분은 데이터에 정보가 없어 0으로 둔 것이다. 역문제에서 복원한 장(field)이 매끈해 보이는 것은 실제가 매끈해서가 아니라 최소 노름 기준 때문일 수 있다.

## 통과 기준

- 선형회귀를 column space로의 사영으로 설명하고, 최소제곱 잔차가 모든 특징 열과 직교함($\mathbf{X}^\top\mathbf{r} \approx \mathbf{0}$)을 계산으로 보인다.
- 같은 최소제곱 문제를 정규방정식·QR·`lstsq`로 풀어, 조건수가 클 때 정규방정식의 오차가 커지는 것을 수치로 확인한다.
- 연습문제 두 개(`gram-schmidt`, `min-norm-solution`)의 테스트를 통과한다.
