---
id: linear-algebra/rank-and-subspaces
title: "행렬의 네 부분공간과 rank"
title_en: "The Four Subspaces and Rank"
tier: core
status: draft
prerequisites: [linear-algebra/vector-spaces]
---

# 행렬의 네 부분공간과 rank

회귀 모델을 학습했더니 계수 벡터가 $(-0.03, 0.49, 0.004)$일 때와 $(320, 18.5, -10)$일 때 예측이 반올림 오차 수준까지 같다고 하자. 어느 계수가 "진짜"일까. 이 질문에 답하려면 행렬이 입력 공간의 어떤 방향을 지우고(null space), 출력 공간의 어떤 방향에는 절대 닿지 못하는지(left null space)를 알아야 한다. 이 장은 행렬 하나에 딸린 네 부분공간과, 그 크기를 하나의 수로 요약한 rank를 다룬다.

네 부분공간은 선형 방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$에 대한 세 질문, 즉 해가 있는가, 하나뿐인가, 여러 개라면 어떻게 생겼는가에 모두 답한다. 그리고 rank는 데이터와 모델의 "실제 차원"을 재는 도구가 되어, 저랭크 근사(7.6장)와 저랭크 미세조정 같은 현대 기법으로 이어진다.

## 학습 목표

- column·row·null·left null space를 구한다
- rank–nullity로 해의 존재와 유일성을 판단한다
- 딥러닝에서의 저랭크 구조를 이해한다

## 네 부분공간

$m \times n$ 행렬 $\mathbf{A}$는 선형사상 $\mathbb{R}^n \to \mathbb{R}^m$이다(7.1장). 7.1장에서 선형사상의 핵과 상을 정의했다. 행렬에서는 여기에 $\mathbf{A}^\top$의 핵과 상을 더해 네 부분공간을 함께 본다. 넷을 함께 보아야 입력과 출력 공간이 각각 어떻게 나뉘는지가 드러난다.

**정의.**

| 부분공간 | 정의 | 사는 곳 | 차원 |
|---|---|---|---|
| column space $C(\mathbf{A})$ | 열들의 span $= \{\mathbf{A}\mathbf{x}\}$ (상) | $\mathbb{R}^m$ | $r$ |
| row space $C(\mathbf{A}^\top)$ | 행들의 span | $\mathbb{R}^n$ | $r$ |
| null space $N(\mathbf{A})$ | $\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$ (핵) | $\mathbb{R}^n$ | $n - r$ |
| left null space $N(\mathbf{A}^\top)$ | $\{\mathbf{y} : \mathbf{A}^\top\mathbf{y} = \mathbf{0}\}$, 즉 $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$ | $\mathbb{R}^m$ | $m - r$ |

여기서 $r$은 rank다(다음 절). left null space라는 이름은 $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$처럼 $\mathbf{y}$가 **왼쪽**에서 곱해지기 때문에 붙었다.

**직교 관계.** 두 벡터의 내적을 $\mathbf{u}^\top\mathbf{v} = \sum_i u_i v_i$로 쓰고, 이 값이 0이면 직교한다고 한다(일반적인 내적은 7.4장에서 다룬다). $\mathbf{A}\mathbf{x} = \mathbf{0}$을 행별로 읽으면 "$\mathbf{A}$의 모든 행과 $\mathbf{x}$의 내적이 0"이다. 따라서 null space의 모든 벡터는 row space의 모든 벡터와 직교한다. 같은 논리를 $\mathbf{A}^\top$에 적용하면 left null space는 column space와 직교한다. 차원까지 더하면 $r + (n - r) = n$, $r + (m - r) = m$이므로, 두 쌍은 각각 입력 공간과 출력 공간을 빈틈없이 나눈다.

$$
\mathbb{R}^n = C(\mathbf{A}^\top) \oplus N(\mathbf{A}), \qquad \mathbb{R}^m = C(\mathbf{A}) \oplus N(\mathbf{A}^\top).
$$

$\oplus$는 두 부분공간이 영벡터만 공유하고, 합쳐서 전체 공간을 이룬다는 뜻이다. 이를 선형대수의 기본 정리라 부르기도 한다.

**직관.** 모든 입력 $\mathbf{x}$는 row space 성분 $\mathbf{x}_r$과 null space 성분 $\mathbf{x}_n$으로 나뉜다. $\mathbf{A}$는 $\mathbf{x}_n$을 지우고, $\mathbf{x}_r$만 column space로 보낸다. row space와 column space는 차원이 $r$로 같고, 그 사이에서 $\mathbf{A}$는 일대일 대응이다. 출력 쪽에서 left null space 방향은 $\mathbf{A}$가 어떤 입력으로도 만들어 내지 못하는 방향이다.

**예.** 다음 $3 \times 4$ 행렬은 셋째 행이 첫째 행과 둘째 행의 합이다.

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 2 & 4 & 1 & 3 \\ 3 & 6 & 1 & 4 \end{bmatrix}.
$$

행 연산(가우스 소거)으로 $\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix}$을 얻는다. pivot이 둘이므로 rank는 2다. pivot이 없는 열에 대응하는 $x_2, x_4$를 자유 변수로 두고 $x_1 = -2x_2 - x_4$, $x_3 = -x_4$를 풀면, null space의 basis는 $(-2, 1, 0, 0)$과 $(-1, 0, -1, 1)$이다. 셋째 행 = 첫째 + 둘째라는 관계는 $\mathbf{y} = (1, 1, -1)$에 대해 $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$이라는 뜻이므로, $\mathbf{y}$가 left null space의 basis다.

```python {exec="browser" session="rank"}
import numpy as np

A = np.array([[1.0, 2.0, 0.0, 1.0],
              [2.0, 4.0, 1.0, 3.0],
              [3.0, 6.0, 1.0, 4.0]])          # (3, 4), 셋째 행 = 첫째 행 + 둘째 행
N = np.array([[-2.0, 1.0, 0.0, 0.0],
              [-1.0, 0.0, -1.0, 1.0]]).T     # (4, 2) null space의 basis
y = np.array([1.0, 1.0, -1.0])               # (3,) left null space의 basis

r = np.linalg.matrix_rank(A)
print("rank =", r, "  rank(A^T) =", np.linalg.matrix_rank(A.T))
print("dim: C(A) =", r, " C(A^T) =", r, " N(A) =", A.shape[1] - r, " N(A^T) =", A.shape[0] - r)
print("A @ N =\n", A @ N)                    # 각 행(row space)이 null space와 직교
print("y @ A =", y @ A)                      # 각 열(column space)이 y와 직교
```

```output
rank = 2   rank(A^T) = 2
dim: C(A) = 2  C(A^T) = 2  N(A) = 2  N(A^T) = 1
A @ N =
 [[0. 0.]
 [0. 0.]
 [0. 0.]]
y @ A = [0. 0. 0. 0.]
```

손으로는 가우스 소거로 basis를 구하지만, 수치 계산에서는 SVD로 네 부분공간의 정규직교 basis를 한꺼번에 구한다(7.6장). 소거법은 작은 반올림 오차에도 pivot 선택이 바뀌어 불안정할 수 있기 때문이다.

::quiz{id="four-subspace-dims"}

:::note[분야 예시]
화학 반응 네트워크는 화학량론 행렬(stoichiometric matrix) $\mathbf{S}$로 적는다. 행은 화학종, 열은 반응이고, 성분은 그 반응에서 화학종이 생기거나(양수) 소모되는(음수) 개수다. 반응 속도 벡터를 $\mathbf{v}$라 하면 농도 변화는 $\dot{\mathbf{c}} = \mathbf{S}\mathbf{v}$이다. **null space**의 $\mathbf{v}$는 모든 농도를 일정하게 유지하는 정상 상태 flux이고, 대사 네트워크의 flux 분석은 이 공간 안에서 해를 찾는다. **left null space**의 $\mathbf{y}$는 $\mathbf{y}^\top\dot{\mathbf{c}} = \mathbf{y}^\top\mathbf{S}\mathbf{v} = 0$을 만족하므로, $\mathbf{y}^\top\mathbf{c}$는 어떤 반응 속도에서도 보존된다. 원소 보존이나 보조 인자(예: ATP와 ADP의 합) 보존 같은 보존 법칙이 left null space의 basis로 나타난다.
:::

## rank

**정의.** 행렬의 **rank**는 column space의 차원이다. 같은 수를 여러 방법으로 정의할 수 있고, 모두 일치한다.

- 선형독립인 열의 최대 개수 $= \dim C(\mathbf{A})$
- 선형독립인 행의 최대 개수 $= \dim C(\mathbf{A}^\top)$
- 가우스 소거 후 pivot의 개수
- 0이 아닌 특이값의 개수(7.6장)

첫째와 둘째가 같다는 것, 즉 **열 rank = 행 rank**는 자명하지 않은 정리다. $3 \times 4$ 행렬의 네 열 중 독립인 것이 둘이면, 세 행 중에서도 독립인 것이 정확히 둘이다. 위 코드의 `rank(A^T) = 2`가 이를 확인한다.

**곱의 rank.** $\mathbf{A}\mathbf{B}$의 열은 $\mathbf{A}$의 열의 선형결합이므로 $C(\mathbf{A}\mathbf{B}) \subseteq C(\mathbf{A})$이고, 행에 대해서도 같은 논리가 성립한다. 그래서

$$
\operatorname{rank}(\mathbf{A}\mathbf{B}) \le \min\big(\operatorname{rank}\mathbf{A}, \operatorname{rank}\mathbf{B}\big)
$$

이다. 7.2장에서 활성화 없는 층을 쌓을 때 좁은 중간 층이 병목이 된 이유다. 같은 장에서 자주 쓰는 등식이 하나 더 있다.

$$
\operatorname{rank}(\mathbf{A}^\top\mathbf{A}) = \operatorname{rank}(\mathbf{A}).
$$

증명은 null space가 같음을 보이면 된다. $\mathbf{A}^\top\mathbf{A}\mathbf{x} = \mathbf{0}$이면 양변에 $\mathbf{x}^\top$을 곱해 $\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x} = \lVert\mathbf{A}\mathbf{x}\rVert^2 = 0$, 즉 $\mathbf{A}\mathbf{x} = \mathbf{0}$이다. 반대 방향은 자명하다. 두 행렬의 열 수가 $n$으로 같으므로 rank–nullity(다음 절)에 의해 rank도 같다. 이 등식 덕분에 최소제곱의 정규방정식 $\mathbf{A}^\top\mathbf{A}\mathbf{w} = \mathbf{A}^\top\mathbf{b}$(7.4장)가 언제 유일한 해를 갖는지 $\mathbf{A}$만 보고 판단할 수 있다.

**full rank.** $m \times n$ 행렬의 rank는 $\min(m, n)$을 넘지 못한다. 이 상한에 닿으면 full rank라 한다. 경우를 나누면 의미가 분명해진다.

| 조건 | 의미 |
|---|---|
| $r = n$ (full column rank) | 열이 독립. $N(\mathbf{A}) = \{\mathbf{0}\}$. 해가 있다면 유일. $\mathbf{A}^\top\mathbf{A}$ 가역 |
| $r = m$ (full row rank) | 행이 독립. $C(\mathbf{A}) = \mathbb{R}^m$. 모든 $\mathbf{b}$에 대해 해가 있음 |
| $r = m = n$ | 정사각 가역행렬. 모든 $\mathbf{b}$에 대해 해가 정확히 하나 |

rank는 행렬의 **크기가 아니다.** $1000 \times 1000$ 행렬도 rank가 1일 수 있다($\mathbf{u}\mathbf{v}^\top$ 꼴). 크기는 저장 비용을, rank는 담긴 정보의 차원을 말한다.

## rank–nullity와 선형시스템

**정리(rank–nullity).** $m \times n$ 행렬 $\mathbf{A}$에 대해

$$
\operatorname{rank}(\mathbf{A}) + \dim N(\mathbf{A}) = n.
$$

입력 공간의 $n$차원 가운데 $r$차원은 출력으로 전달되고, 나머지 $n - r$차원은 지워진다. 7.1장의 $\dim\ker T + \dim\operatorname{im} T = \dim V$를 행렬로 쓴 것이다.

**해의 존재.** $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해가 있다는 것은 $\mathbf{b}$가 열들의 선형결합이라는 것, 즉 $\mathbf{b} \in C(\mathbf{A})$라는 뜻이다. 앞 절의 직교 관계로 바꿔 말하면 다음과 같다.

$$
\mathbf{A}\mathbf{x} = \mathbf{b} \text{의 해가 있다} \iff \text{모든 } \mathbf{y} \in N(\mathbf{A}^\top) \text{에 대해 } \mathbf{y}^\top\mathbf{b} = 0.
$$

left null space의 basis 벡터 하나하나가 $\mathbf{b}$에 대한 **양립 조건**(compatibility condition)이다. 앞의 예에서 $\mathbf{y} = (1, 1, -1)$이므로, $\mathbf{b}$의 셋째 성분이 첫째와 둘째의 합이어야 해가 있다. 행렬로 쓰면 $\operatorname{rank}[\mathbf{A} \mid \mathbf{b}] = \operatorname{rank}\mathbf{A}$와 같은 조건이다.

**해의 유일성.** $\mathbf{x}_p$가 해이면 null space의 아무 벡터 $\mathbf{x}_n$을 더해도 $\mathbf{A}(\mathbf{x}_p + \mathbf{x}_n) = \mathbf{b}$이다. 반대로 두 해의 차는 항상 null space에 있다. 따라서 해 전체는

$$
\{\mathbf{x}_p + \mathbf{x}_n : \mathbf{x}_n \in N(\mathbf{A})\}
$$

이고, 이는 null space를 $\mathbf{x}_p$만큼 평행이동한 아핀 집합이다(7.2장). 해가 유일한 것은 $N(\mathbf{A}) = \{\mathbf{0}\}$, 즉 $r = n$일 때뿐이다.

```python {exec="browser" session="rank"}
x0 = np.array([1.0, 0.0, 2.0, 0.0])
b1 = A @ x0                                  # column space 안의 b
b2 = np.array([1.0, 0.0, 0.0])               # column space 밖의 b
print("y @ b1 =", y @ b1, "  y @ b2 =", y @ b2)

x1 = x0 + 3.0 * N[:, 0] - 1.0 * N[:, 1]      # 특수해 + null space 성분
print("A @ x0 =", A @ x0, "  A @ x1 =", A @ x1)

x_ls = np.linalg.lstsq(A, b2)[0]             # 가장 가까운 근사해(7.4장)
print("b2의 최소제곱 잔차 =", np.round(b2 - A @ x_ls, 4))
```

```output
y @ b1 = 0.0   y @ b2 = 1.0
A @ x0 = [1. 4. 5.]   A @ x1 = [1. 4. 5.]
b2의 최소제곱 잔차 = [ 0.3333  0.3333 -0.3333]
```

$\mathbf{b}_1$은 양립 조건을 만족해 해가 무수히 많고($\mathbf{x}_0$와 $\mathbf{x}_1$이 모두 해), $\mathbf{b}_2$는 만족하지 않아 정확한 해가 없다. 최소제곱으로 가장 가까운 근사해를 구하면 잔차가 $\mathbf{y}/3$, 즉 **left null space 방향**으로 남는다. column space 안에서는 오차를 없앨 수 있고, 남는 오차는 column space에 직교하는 방향뿐이기 때문이다. 이 사실이 7.4장의 최소제곱 이론의 핵심이다.

정리하면, 해의 구조는 $r$과 $m, n$의 관계로 정해진다.

| 경우 | 해의 존재 | 해의 개수 | 예 |
|---|---|---|---|
| $r = m = n$ | 항상 | 1 | 가역 정사각 시스템 |
| $r = n < m$ (과결정) | $\mathbf{b} \in C(\mathbf{A})$일 때만 | 0 또는 1 | 측정이 미지수보다 많은 회귀 |
| $r = m < n$ (미결정) | 항상 | 무한 | 미지수가 측정보다 많은 역문제 |
| $r < m, r < n$ | $\mathbf{b} \in C(\mathbf{A})$일 때만 | 0 또는 무한 | 위의 예, 종속인 특징이 있는 회귀 |

::quiz{id="rank-nullity-solutions"}

과결정 시스템에 잡음이 섞이면 $\mathbf{b}$는 거의 확실히 column space 밖에 있으므로 정확한 해는 없고, "가장 가까운" 해를 찾아야 한다(최소제곱, 7.4장). 미결정 시스템은 해가 너무 많으므로 "가장 그럴듯한" 해를 고를 기준이 필요하다(최소 노름, 정규화, 사전분포). 과학의 역문제(inverse problem)는 흔히 두 문제를 함께 가진다. 측정은 잡음이 있고, 측정이 닿지 않는 방향(null space)도 있다(24.7장).

## 수치적 rank

**왜 필요한가.** rank의 정의는 "독립인가 아닌가"라는 예·아니오 질문이다. 그런데 부동소수점 계산과 잡음 섞인 데이터에서는 열이 정확히 종속인 일이 거의 없다. 두 센서가 같은 양을 측정하면 두 열은 거의 같지만, 잡음 때문에 정확히 같지는 않다. 이런 행렬의 rank를 그대로 세면 의미 없는 답이 나온다.

**정의.** 실무에서는 특이값 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$(7.6장)을 계산하고, 허용오차 $\tau$보다 큰 특이값의 개수를 **수치적 rank**라 한다. 지금은 특이값을 "각 독립 방향이 얼마나 강하게 살아 있는지 재는 수"로 이해하면 충분하다. 정확한 rank는 0이 아닌 특이값의 개수이고, 수치적 rank는 $\tau$보다 큰 특이값의 개수다. NumPy의 `matrix_rank`는 기본값으로 $\tau = \sigma_1 \cdot \max(m, n) \cdot \epsilon$($\epsilon$은 기계 엡실론, float64에서 약 $2.2 \times 10^{-16}$)을 쓴다. 반올림 오차만 걸러 내는 매우 작은 값이다.

```python {exec="browser" session="rank"}
rng = np.random.default_rng(0)
c1, c2 = rng.normal(size=100), rng.normal(size=100)        # (100,), (100,)
exact = np.column_stack([c1, c2, c1 + c2])                 # (100, 3)
noisy = np.column_stack([c1, c2, c1 + c2 + 1e-9 * rng.normal(size=100)])

for name, X in [("정확히 종속", exact), ("1e-9 잡음", noisy)]:
    s = np.linalg.svd(X, compute_uv=False)                 # 특이값(7.6장)
    print(f"{name}: 특이값 {s[0]:.2f} {s[1]:.2f} {s[2]:.1e}  "
          f"rank(기본 tol) = {np.linalg.matrix_rank(X)}  "
          f"rank(tol=1e-6) = {np.linalg.matrix_rank(X, tol=1e-6)}")
```

```output
정확히 종속: 특이값 17.07 9.37 3.4e-15  rank(기본 tol) = 2  rank(tol=1e-6) = 2
1e-9 잡음: 특이값 17.07 9.37 6.5e-09  rank(기본 tol) = 3  rank(tol=1e-6) = 2
```

정확히 종속인 행렬도 셋째 특이값이 정확히 0이 아니라 $3.4 \times 10^{-15}$이다. 반올림 오차다. 기본 허용오차는 이것을 0으로 보고 rank 2를 준다. 셋째 열에 $10^{-9}$ 크기의 잡음을 더하면 셋째 특이값이 $6.5 \times 10^{-9}$이 되어 기본 허용오차를 넘고, rank는 3이 된다. 셋째 열은 "수학적으로는 독립"이지만, 그 독립성은 잡음 수준에 불과하다. 허용오차를 $10^{-6}$으로 정하면 다시 rank 2다.

::quiz{id="numerical-rank"}

**연구에서의 의미.** 수치적 rank는 허용오차에 의존하고, 허용오차는 데이터가 결정한다. 측정 잡음이 $10^{-3}$ 수준이면 그보다 작은 특이값의 방향은 측정으로 구분할 수 없으므로, 허용오차도 그 수준에서 정해야 한다. 그래서 "rank가 몇인가"보다 **특이값이 어떻게 감소하는가**를 보는 편이 더 많은 정보를 준다. 특이값이 어느 지점에서 급격히 떨어지면 그 앞까지가 실질적인 차원이고, 완만하게 줄면 뚜렷한 경계가 없다. 이 분석은 7.6장에서 이어진다. 거의 종속인 열은 rank를 바꾸지 않더라도 계수 추정을 매우 불안정하게 만든다. 이는 9.2장의 조건수 문제다.

## 저랭크 구조

rank가 $r$인 $d \times k$ 행렬은 언제나 $d \times r$ 행렬과 $r \times k$ 행렬의 곱 $\mathbf{B}\mathbf{A}$로 쓸 수 있다($\mathbf{B}$의 열로 column space의 basis를 고르면 된다). 저장해야 하는 수는 $dk$개에서 $r(d + k)$개로 줄어든다. $r$이 $d, k$보다 훨씬 작으면 이 차이는 매우 크다.

**저랭크 가중치와 업데이트.** 신경망의 가중치 행렬이나 학습 중 가중치의 **변화량**이 근사적으로 저랭크라는 관찰은 여러 기법의 출발점이다. 대표적인 예가 LoRA(low-rank adaptation)다. 사전학습된 가중치 $\mathbf{W}_0$은 고정하고, 미세조정에서 업데이트 $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$만 학습한다. $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$이므로 $\Delta\mathbf{W}$의 rank는 곱의 rank 부등식에 의해 $r$ 이하다.

```python {exec="browser" session="rank"}
d, k, r = 1024, 1024, 8
rng = np.random.default_rng(1)
B = rng.normal(size=(d, r))                  # (d, r)
A_lr = rng.normal(size=(r, k))               # (r, k)
dW = B @ A_lr                                # (d, k) 저랭크 업데이트
print("rank(dW)            =", np.linalg.matrix_rank(dW))
print("전체 행렬 파라미터 수 =", d * k)
print("두 인수 파라미터 수   =", d * r + r * k, f"({(d * r + r * k) / (d * k):.2%})")
```

```output
rank(dW)            = 8
전체 행렬 파라미터 수 = 1048576
두 인수 파라미터 수   = 16384 (1.56%)
```

$1024 \times 1024$ 행렬 전체를 학습하는 대신 1.56%의 파라미터만 학습한다. 그 대가로 업데이트는 $r$차원 부분공간 안의 열만 만들 수 있다. 이 제약이 미세조정에 충분한지는 과제마다 다르며, 실험으로 확인해야 하는 가정이다.

**저랭크 행렬로서의 데이터.** 데이터 행렬(행 = 샘플, 열 = 특징)도 흔히 근사적으로 저랭크다. 측정 변수는 수백 개여도, 그 변화를 만드는 요인은 몇 개뿐인 경우가 많기 때문이다. 이때 데이터 행렬의 특이값은 앞의 몇 개 뒤로 빠르게 작아진다. 7.6장의 PCA와 저랭크 근사가 이 구조를 찾아 쓰는 방법이다.

:::note[분야 예시]
유체나 열전달 시뮬레이션에서 시각마다 저장한 장(field)을 열로 쌓은 행렬을 snapshot 행렬이라 한다. 격자점이 수십만 개여도, 주기적인 흐름처럼 구조가 뚜렷한 경우에는 특이값이 빠르게 감소해 소수의 공간 패턴으로 대부분의 변동을 설명할 수 있다. 이 패턴을 찾는 방법을 유체역학에서는 POD(proper orthogonal decomposition)라 부르며, 계산 방법은 7.6장의 SVD·PCA와 같다. 축소된 좌표로 동역학을 학습하면 대리모델(24.5장)의 비용이 크게 줄어든다.
:::

## 실습

rank가 부족한 회귀 문제를 만들고, 해가 유일하지 않다는 것과 정규화가 해를 고르는 방식을 확인한다. 같은 온도를 섭씨와 화씨로 모두 특징에 넣고 절편 열을 더하면, 화씨 = 1.8 × 섭씨 + 32 관계 때문에 세 열이 종속이 된다. 측정 장비 로그를 그대로 특징으로 쓸 때 실제로 일어나는 일이다.

```python {exec="browser" session="rank"}
rng = np.random.default_rng(2)
temp_c = rng.uniform(10, 40, size=50)                      # (50,) 섭씨
temp_f = 1.8 * temp_c + 32.0                               # (50,) 같은 온도의 화씨
X = np.column_stack([np.ones(50), temp_c, temp_f])         # (50, 3), 열이 종속
y_obs = 0.5 * temp_c + rng.normal(scale=0.3, size=50)      # (50,)
print("rank(X) =", np.linalg.matrix_rank(X))

n_vec = np.array([32.0, 1.8, -1.0])                        # X @ n_vec = 0
w_a = np.linalg.lstsq(X, y_obs)[0]
w_b = w_a + 10.0 * n_vec
print("w_a =", np.round(w_a, 3), " w_b =", np.round(w_b, 3))
print("예측 차이 최대 =", f"{np.abs(X @ w_a - X @ w_b).max():.1e}")

for lam in [1e-6, 1e-2, 1e2]:
    w = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y_obs)
    rmse = np.sqrt(np.mean((y_obs - X @ w) ** 2))
    print(f"lambda = {lam:6.0e}  w = {np.round(w, 3)}  |w| = {np.linalg.norm(w):.3f}  RMSE = {rmse:.4f}")
```

```output
rank(X) = 2
w_a = [-0.027  0.487  0.004]  w_b = [319.973  18.487  -9.996]
예측 차이 최대 = 1.7e-13
lambda =  1e-06  w = [-0.027  0.487  0.004]  |w| = 0.488  RMSE = 0.2735
lambda =  1e-02  w = [-0.027  0.487  0.004]  |w| = 0.488  RMSE = 0.2735
lambda =  1e+02  w = [-0.022  0.411  0.029]  |w| = 0.413  RMSE = 0.3688
```

결과를 읽는 법은 세 가지다.

- **해가 유일하지 않다.** $\mathbf{X}$의 null space는 $(32, 1.8, -1)$ 방향이다. 이 방향으로 아무리 움직여도 예측은 반올림 오차 수준($10^{-13}$)까지 같다. 도입의 두 계수 벡터가 $\mathbf{w}_a$와 $\mathbf{w}_b$다. 데이터로는 두 계수를 구분할 수 없으므로, "화씨 계수가 음수이니 화씨 온도가 결과를 줄인다" 같은 해석은 의미가 없다.
- **`lstsq`는 그중 노름이 가장 작은 해를 준다.** 무수한 해 가운데 null space 성분이 0인, 즉 row space 안에 있는 해다. 7.4장의 pseudoinverse가 바로 이 해를 준다.
- **ridge 정규화 $\lambda\lVert\mathbf{w}\rVert^2$는 해를 하나로 고른다.** $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$는 $\lambda > 0$이면 항상 가역이다. $\lambda$가 작으면 최소 노름 해와 같아지고, $\lambda$가 크면 계수를 0 쪽으로 줄이는 대가로 학습 오차(RMSE)가 커진다. 정규화는 데이터가 정하지 못한 방향을 "작은 계수가 좋다"는 사전 선호로 정한다. 그 선호가 물리적으로 맞는지는 별개의 질문이다.

이어서 두 연습문제를 푼다.

::exercise{id="solution-structure"}

::exercise{id="ridge-solution"}

## 실패 사례

- **수치적으로 거의 의존적인 열을 독립으로 취급하는 것.** `matrix_rank`가 full rank를 돌려줘도, 특이값이 잡음 수준이면 그 방향의 계수는 사실상 정해지지 않는다. 증상은 학습할 때마다(시드·데이터 부분집합마다) 크게 바뀌는 계수, 서로 반대 부호로 큰 값을 갖는 한 쌍의 계수다. rank 하나만 보지 말고 특이값 전체를 출력해 감소 양상을 본다.
- **rank와 행렬 크기를 혼동하는 것.** "$1000 \times 50$ 데이터이니 50차원 정보가 있다"는 말은 rank가 50일 때만 맞다. 반대로 $\Delta\mathbf{W}$가 $4096 \times 4096$이라도 두 인수로 만들었다면 rank는 인수의 내부 차원 이하다.
- **해가 여러 개인 문제에서 한 해를 골라 계수를 해석하는 것.** null space가 $\{\mathbf{0}\}$이 아니면, 개별 계수는 데이터가 아니라 알고리즘(최소 노름, 정규화, 초기값)이 정한다. 해석하기 전에 null space를 확인하고, 필요하면 특징을 줄이거나 식별 가능한 조합(위의 예에서는 섭씨 계수와 화씨 계수의 특정 결합)만 보고한다.

## 통과 기준

- 주어진 행렬의 네 부분공간의 basis와 차원을 구하고, row space ⟂ null space, column space ⟂ left null space를 계산으로 확인한다.
- rank와 행렬 크기로 $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해가 없는지, 하나인지, 무한히 많은지 판정한다(연습문제 `solution-structure`).
- 종속인 특징이 있는 회귀에서 해가 유일하지 않음을 보이고, ridge 정규화가 고르는 해를 계산한다(연습문제 `ridge-solution`).
