---
id: calculus-and-autodiff/matrix-calculus
title: "행렬 미분"
title_en: "Matrix Calculus"
tier: core
status: draft
prerequisites: [calculus-and-autodiff/multivariable-calculus]
---

# 행렬 미분

신경망의 파라미터는 행렬이고, 손실은 그 행렬들의 스칼라 함수다. 가중치 행렬의 성분 하나하나로 편미분해 모을 수도 있지만, 성분이 수만 개를 넘으면 그렇게 유도한 식은 읽을 수도 구현할 수도 없다. 행렬 미분(matrix calculus)은 이 계산을 행렬 연산의 언어로 한 번에 하는 방법이다. 결과는 `X.T @ (X @ w - y)`처럼 그대로 코드가 되는 식이다.

이 장의 어려움은 계산보다 **규약**에 있다. 같은 미분을 교재마다 다른 shape으로 쓰고, 그 차이가 전치 하나로 나타난다. 이 장은 책 전체의 규약을 정하고, 미분형(differential)으로 gradient를 기계적으로 읽어 내는 방법을 익힌 뒤, 선형회귀·softmax cross-entropy·2층 MLP의 gradient를 손으로 유도한다. 8.4장의 역전파는 이 유도를 자동화한 것이다.

## 학습 목표

- 행렬·벡터에 대한 미분을 유도한다
- layout 규약을 일관되게 쓴다
- 신경망 gradient를 손으로 유도한다

## layout 규약

8.1장에서 스칼라 함수의 Jacobian은 $1 \times n$ 행이고 gradient는 그것을 전치한 $n \times 1$ 열이라고 했다. 둘은 같은 정보이지만 shape이 다르다. 교재와 논문은 둘 중 하나를 골라 "미분"이라 부르는데, 그 선택을 **layout 규약**이라 한다. 규약을 모르고 공식을 옮기면 전치가 하나씩 어긋난다.

두 가지 대표 규약이 있다. $\mathbf{y} \in \mathbb{R}^m$을 $\mathbf{x} \in \mathbb{R}^n$으로 미분한 $\partial\mathbf{y}/\partial\mathbf{x}$를

- **numerator layout**은 분자($\mathbf{y}$)의 차원을 행으로 놓아 $m \times n$으로 쓴다. 8.1장의 Jacobian과 같다. 스칼라를 벡터로 미분하면 $1 \times n$ 행벡터가 된다.
- **denominator layout**은 분모($\mathbf{x}$)의 차원을 행으로 놓아 $n \times m$으로 쓴다. Jacobian의 전치다. 스칼라를 벡터로 미분하면 $n \times 1$ 열벡터가 된다.

**이 책의 규약**은 다음 표로 고정한다([부록 A](../../appendices/a-notation/index.md)와 같다).

| 미분 | 이 책의 표기 | shape | 비고 |
|---|---|---|---|
| 벡터 $\mathbf{y} \in \mathbb{R}^m$를 벡터 $\mathbf{x} \in \mathbb{R}^n$로 | Jacobian $\mathbf{J} = \partial\mathbf{y}/\partial\mathbf{x}$ | $m \times n$ | numerator layout |
| 스칼라 $L$을 벡터 $\mathbf{x} \in \mathbb{R}^n$로 | gradient $\nabla_{\mathbf{x}} L$ | $n \times 1$ | $= \mathbf{J}^\top$, 열벡터 |
| 스칼라 $L$을 행렬 $\mathbf{W} \in \mathbb{R}^{m \times n}$로 | gradient $\nabla_{\mathbf{W}} L$ | $m \times n$ | $\mathbf{W}$와 같은 shape |

Jacobian은 numerator layout을 따르고, 스칼라의 gradient는 파라미터와 같은 shape을 따른다. 엄밀히는 두 규약을 섞어 쓰는 셈이지만, 머신러닝에서 가장 실용적인 조합이다. 경사하강 $\mathbf{W} \leftarrow \mathbf{W} - \eta\nabla_{\mathbf{W}}L$이 성분별로 바로 맞고, PyTorch의 `.grad`도 파라미터와 같은 shape이다.

::quiz{id="grad-shape-matrix"}

벡터를 행렬로 미분하는 경우(예: 층의 출력 $\mathbf{W}\mathbf{x}$를 $\mathbf{W}$로)는 결과가 3차원 배열이 되어 행렬로 쓸 수 없다. 이 책은 그런 대상을 직접 만들지 않는다. 손실이라는 스칼라까지 가는 경로에서 다음 절의 미분형을 쓰면 3차원 배열 없이 gradient를 얻는다. 8.4장의 역전파도 같은 이유로 Jacobian 전체를 만들지 않는다.

## 미분형(differential) 접근

성분별 편미분으로 행렬 gradient를 유도하면 첨자가 서너 개씩 붙고 실수하기 쉽다. 미분형은 "입력이 $d\mathbf{X}$만큼 변할 때 출력이 얼마나 변하는가"를 행렬식 그대로 추적한 뒤, 마지막에 gradient를 읽어 내는 방법이다.

**정의.** 스칼라 함수 $L(\mathbf{X})$의 미분형 $dL$은 8.1장의 1차 근사 $L(\mathbf{X} + d\mathbf{X}) - L(\mathbf{X})$에서 $d\mathbf{X}$에 대해 선형인 부분이다. 성분으로 쓰면 $dL = \sum_{ij} \frac{\partial L}{\partial X_{ij}}\,dX_{ij}$이고, 이것은 두 행렬의 성분별 곱의 합이므로 trace로 쓸 수 있다.

$$
dL = \operatorname{tr}\big(\mathbf{G}^\top d\mathbf{X}\big) \quad \Longleftrightarrow \quad \nabla_{\mathbf{X}} L = \mathbf{G}.
$$

여기서 $\operatorname{tr}(\mathbf{A}^\top\mathbf{B}) = \sum_{ij} A_{ij}B_{ij}$를 썼다. 벡터에서는 $dL = \mathbf{g}^\top d\mathbf{x}$이면 $\nabla_{\mathbf{x}}L = \mathbf{g}$다. 그래서 절차는 두 단계다. **(1) 규칙으로 $dL$을 계산하고, (2) $\operatorname{tr}(\mathbf{G}^\top d\mathbf{X})$ 꼴로 정리해 $\mathbf{G}$를 읽는다.**

**규칙.** 일변수 미분 규칙이 행렬 곱의 순서를 지키는 형태로 그대로 성립한다.

$$
d(\mathbf{X}\mathbf{Y}) = (d\mathbf{X})\mathbf{Y} + \mathbf{X}(d\mathbf{Y}), \qquad d(\mathbf{X}^\top) = (d\mathbf{X})^\top, \qquad d\operatorname{tr}(\mathbf{X}) = \operatorname{tr}(d\mathbf{X}), \qquad d\mathbf{A} = \mathbf{0}\ (\text{상수}).
$$

정리 단계에서 쓰는 trace 성질은 셋이다. 스칼라는 자기 자신의 trace다($a = \operatorname{tr}(a)$). 순환해도 같다($\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$, shape이 맞을 때). 전치해도 같다($\operatorname{tr}(\mathbf{A}^\top) = \operatorname{tr}(\mathbf{A})$).

**예: 한 층의 제곱오차.** $L = \tfrac{1}{2}\lVert \mathbf{W}\mathbf{x} - \mathbf{y} \rVert^2$을 $\mathbf{W} \in \mathbb{R}^{m \times n}$로 미분한다. 잔차를 $\mathbf{r} = \mathbf{W}\mathbf{x} - \mathbf{y}$로 두면

$$
dL = \mathbf{r}^\top d\mathbf{r} = \mathbf{r}^\top (d\mathbf{W})\,\mathbf{x} = \operatorname{tr}\big(\mathbf{x}\mathbf{r}^\top d\mathbf{W}\big) = \operatorname{tr}\big((\mathbf{r}\mathbf{x}^\top)^\top d\mathbf{W}\big)
$$

이다. 셋째 등호에서 스칼라를 trace로 쓰고 순환시켰다. 따라서 $\nabla_{\mathbf{W}}L = \mathbf{r}\mathbf{x}^\top$이고, shape은 $(m \times 1)(1 \times n) = m \times n$으로 $\mathbf{W}$와 같다. **출력 쪽 gradient와 입력의 외적**이라는 이 모양은 모든 선형층의 가중치 gradient에서 반복된다.

**역행렬과 행렬식.** $\mathbf{X}\mathbf{X}^{-1} = \mathbf{I}$의 양변에 미분형을 취하면 $(d\mathbf{X})\mathbf{X}^{-1} + \mathbf{X}\,d(\mathbf{X}^{-1}) = \mathbf{0}$이므로

$$
d(\mathbf{X}^{-1}) = -\mathbf{X}^{-1}(d\mathbf{X})\,\mathbf{X}^{-1}
$$

이다. 일변수의 $d(1/x) = -dx/x^2$에 해당한다. 행렬식에 대해서는 Jacobi 공식 $d\det\mathbf{X} = \det\mathbf{X}\,\operatorname{tr}(\mathbf{X}^{-1}d\mathbf{X})$가 성립하고, 따라서 $d\log\det\mathbf{X} = \operatorname{tr}(\mathbf{X}^{-1}d\mathbf{X})$, 즉 $\nabla_{\mathbf{X}}\log\det\mathbf{X} = \mathbf{X}^{-\top}$이다. 2×2 행렬로 확인하면 $\det\mathbf{X} = X_{11}X_{22} - X_{12}X_{21}$이므로 $\partial\det\mathbf{X}/\partial X_{11} = X_{22}$이고, 이것은 $\det\mathbf{X}\,(\mathbf{X}^{-1})_{11}$과 같다. $\log\det$는 8.3장의 변수변환과 normalizing flow에서 다시 나온다.

## 자주 쓰는 항등식

다음 표는 미분형 규칙으로 모두 유도할 수 있다. 외우기보다 유도할 수 있어야 하고, 쓰기 전에 수치미분으로 확인하는 습관이 중요하다. 이 책의 규약(gradient는 미분 대상과 같은 shape)으로 쓴 것이다. $\mathbf{a}, \mathbf{b}, \mathbf{A}$는 상수다.

| 함수 | gradient | 유도의 핵심 |
|---|---|---|
| $\mathbf{a}^\top\mathbf{x}$ | $\nabla_{\mathbf{x}} = \mathbf{a}$ | $d = \mathbf{a}^\top d\mathbf{x}$ |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $\nabla_{\mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | $\mathbf{x}$가 양쪽에 있어 항이 둘 |
| $\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2$ | $\nabla_{\mathbf{x}} = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ | 잔차의 미분 후 전치 |
| $\mathbf{a}^\top\mathbf{X}\mathbf{b}$ | $\nabla_{\mathbf{X}} = \mathbf{a}\mathbf{b}^\top$ | $\operatorname{tr}(\mathbf{b}\mathbf{a}^\top d\mathbf{X})$ |
| $\operatorname{tr}(\mathbf{A}\mathbf{X})$ | $\nabla_{\mathbf{X}} = \mathbf{A}^\top$ | $\operatorname{tr}(\mathbf{A}\,d\mathbf{X})$ |
| $\operatorname{tr}(\mathbf{X}^\top\mathbf{A}\mathbf{X})$ | $\nabla_{\mathbf{X}} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{X}$ | 곱의 규칙 |
| $\log\det\mathbf{X}$ | $\nabla_{\mathbf{X}} = \mathbf{X}^{-\top}$ | Jacobi 공식 |
| $\operatorname{tr}(\mathbf{A}\mathbf{X}^{-1})$ | $\nabla_{\mathbf{X}} = -\mathbf{X}^{-\top}\mathbf{A}^\top\mathbf{X}^{-\top}$ | 역행렬의 미분형 |

$\mathbf{A}$가 대칭이면 둘째 줄은 $2\mathbf{A}\mathbf{x}$, 여섯째 줄은 $2\mathbf{A}\mathbf{X}$로 줄어든다. 대칭을 가정한 공식을 대칭이 아닌 행렬에 쓰는 것이 흔한 실수다.

::quiz{id="quadratic-form-gradient"}

표의 세 줄을 중앙차분 gradient와 비교한다. `num_grad`는 입력과 같은 shape의 gradient를 돌려주므로, 이 책의 규약을 그대로 확인할 수 있다.

```python {exec="browser" session="matcalc"}
import numpy as np

def num_grad(f, X, eps=1e-6):
    """스칼라 함수 f의 X에 대한 gradient를 중앙차분으로. 결과 shape = X.shape"""
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X)
        E[idx] = eps
        G[idx] = (f(X + E) - f(X - E)) / (2 * eps)
    return G

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 3))                       # (3, 3) 대칭이 아님
x = rng.normal(size=3)                            # (3,)
X = rng.normal(size=(3, 3)) + 3 * np.eye(3)       # (3, 3) det > 0

checks = {
    "x^T A x  -> (A + A^T) x": (num_grad(lambda v: v @ A @ v, x), (A + A.T) @ x),
    "tr(A X)  -> A^T        ": (num_grad(lambda M: np.trace(A @ M), X), A.T),
    "log det X -> X^{-T}    ": (num_grad(lambda M: np.log(np.linalg.det(M)), X), np.linalg.inv(X).T),
}
for name, (num, formula) in checks.items():
    print(f"{name}  최대 차이 {np.abs(num - formula).max():.1e}")
```

```output
x^T A x  -> (A + A^T) x  최대 차이 5.0e-11
tr(A X)  -> A^T          최대 차이 4.2e-10
log det X -> X^{-T}      최대 차이 1.0e-10
```

:::note[분야 예시]
기후·지구과학에서 관측소 사이의 값을 보간할 때 쓰는 가우시안 과정(Gaussian process) 회귀는 커널 행렬 $\mathbf{K}(\theta)$의 hyperparameter $\theta$(상관 거리, 잡음 분산 등)를 로그 주변우도 $\log p(\mathbf{y}) = -\tfrac{1}{2}\mathbf{y}^\top\mathbf{K}^{-1}\mathbf{y} - \tfrac{1}{2}\log\det\mathbf{K} + \text{상수}$를 최대화해 정한다. 표의 역행렬·$\log\det$ 규칙과 chain rule을 쓰면 $\dfrac{\partial \log p(\mathbf{y})}{\partial\theta} = \tfrac{1}{2}\mathbf{y}^\top\mathbf{K}^{-1}\dfrac{\partial\mathbf{K}}{\partial\theta}\mathbf{K}^{-1}\mathbf{y} - \tfrac{1}{2}\operatorname{tr}\Big(\mathbf{K}^{-1}\dfrac{\partial\mathbf{K}}{\partial\theta}\Big)$을 얻는다. 첫 항은 데이터에 맞추는 힘, 둘째 항은 모델 복잡도에 대한 벌점이다.
:::

## 신경망의 gradient

이제 같은 도구로 실제 모델의 gradient를 유도한다. 순서는 선형회귀, softmax cross-entropy, 2층 MLP다. 세 유도 모두 **출력 쪽 gradient를 먼저 구하고, 층을 하나씩 거슬러 내려가며 Jacobian의 전치를 곱한다**는 같은 구조를 가진다.

**선형회귀.** 설계행렬 $\mathbf{X} \in \mathbb{R}^{N \times D}$(행이 샘플), 타깃 $\mathbf{y} \in \mathbb{R}^N$, 가중치 $\mathbf{w} \in \mathbb{R}^D$에 대해 $L = \tfrac{1}{2N}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$이다. 잔차 $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$로 두면 $dL = \tfrac{1}{N}\mathbf{r}^\top\mathbf{X}\,d\mathbf{w}$이므로

$$
\nabla_{\mathbf{w}} L = \frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) \qquad (D \times N)(N \times 1) = D \times 1.
$$

이 gradient를 0으로 두면 정규방정식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$가 나온다. 최소제곱해가 잔차를 $\mathbf{X}$의 열공간에 직교하게 만든다는 사영의 관점(7.4장)과 같은 식이다.

```python {exec="browser" session="matcalc"}
N, D = 50, 3
Xd = rng.normal(size=(N, D))                      # (N, D) 설계행렬
y = Xd @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=N)   # (N,)
w = np.zeros(D)                                   # (D,)

def mse(w):
    r = Xd @ w - y                                # (N,) 잔차
    return 0.5 * np.mean(r**2)

grad_w = Xd.T @ (Xd @ w - y) / N                  # (D, N)(N,) = (D,)
print("손 유도  :", grad_w.round(6))
print("수치미분 :", num_grad(mse, w).round(6))
```

```output
손 유도  : [-1.283416  1.29817  -0.589093]
수치미분 : [-1.283416  1.29817  -0.589093]
```

**softmax cross-entropy.** 분류 모델은 logit $\mathbf{z} \in \mathbb{R}^K$를 확률 $p_k = e^{z_k}/\sum_j e^{z_j}$로 바꾸고, one-hot 정답 $\mathbf{y}$에 대해 $L = -\sum_k y_k\log p_k$를 최소화한다. $\log p_k = z_k - \log\sum_j e^{z_j}$이므로

$$
\frac{\partial \log p_k}{\partial z_i} = \delta_{ki} - \frac{e^{z_i}}{\sum_j e^{z_j}} = \delta_{ki} - p_i
$$

이다. $\delta_{ki}$는 $k = i$일 때 1, 아니면 0이다. 이를 손실에 넣으면

$$
\frac{\partial L}{\partial z_i} = -\sum_k y_k(\delta_{ki} - p_i) = -y_i + p_i\sum_k y_k = p_i - y_i, \qquad \nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}.
$$

마지막 등호에 $\sum_k y_k = 1$을 썼다. 결과는 놀랍도록 단순하다. **예측 확률에서 정답을 뺀 것**이다. 정답 클래스의 성분은 $p_c - 1 < 0$이라 그 logit을 올리고, 나머지 성분은 $p_k > 0$이라 낮춘다. 성분의 합은 $1 - 1 = 0$이다. 같은 결과를 softmax의 Jacobian $\partial\mathbf{p}/\partial\mathbf{z} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$과 $\partial L/\partial\mathbf{p} = -\mathbf{y}/\mathbf{p}$(성분별 나눗셈)의 곱으로 유도해도 된다. 이 단순함 때문에 프레임워크는 softmax와 cross-entropy를 한 함수로 묶어 계산한다. 따로 계산하면 $1/p_k$에서 수치 문제가 생길 수 있다.

::quiz{id="softmax-ce-gradient"}

```python {exec="browser" session="matcalc"}
def softmax(z):
    e = np.exp(z - z.max())                       # 큰 logit에서의 overflow 방지
    return e / e.sum()

z = np.array([2.0, 1.0, -1.0, 0.5])               # (K,) logits
y_onehot = np.array([0.0, 1.0, 0.0, 0.0])         # (K,) 정답은 인덱스 1

ce = lambda z: -np.sum(y_onehot * np.log(softmax(z)))
p = softmax(z)
print("p - y    :", (p - y_onehot).round(6))
print("수치미분 :", num_grad(ce, z).round(6))
print("성분 합  :", f"{abs((p - y_onehot).sum()):.1e}")
```

```output
p - y    : [ 0.60946  -0.775792  0.030343  0.135989]
수치미분 : [ 0.60946  -0.775792  0.030343  0.135989]
성분 합  : 2.8e-17
```

**2층 MLP.** 한 샘플에 대해 $\mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1$, $\mathbf{h} = \mathrm{ReLU}(\mathbf{a})$, $\mathbf{z} = \mathbf{W}_2\mathbf{h} + \mathbf{b}_2$, $L = \mathrm{CE}(\mathbf{z}, \mathbf{y})$이다. 가중치는 (출력 × 입력) shape이다. 출력 쪽부터 거슬러 내려간다.

$$
\begin{aligned}
\mathbf{g}_z &= \mathbf{p} - \mathbf{y}, &
\nabla_{\mathbf{W}_2} L &= \mathbf{g}_z\mathbf{h}^\top, &
\nabla_{\mathbf{b}_2} L &= \mathbf{g}_z, \\
\mathbf{g}_h &= \mathbf{W}_2^\top\mathbf{g}_z, &
\mathbf{g}_a &= \mathbf{g}_h \odot \mathbb{1}[\mathbf{a} > 0], & & \\
\nabla_{\mathbf{W}_1} L &= \mathbf{g}_a\mathbf{x}^\top, &
\nabla_{\mathbf{b}_1} L &= \mathbf{g}_a. & &
\end{aligned}
$$

각 줄은 앞 절의 결과를 그대로 쓴다. 가중치 gradient는 "출력 쪽 gradient와 입력의 외적"이고, 선형층을 거슬러 갈 때는 가중치의 전치 $\mathbf{W}_2^\top$을 곱하며($\mathbf{z}$의 $\mathbf{h}$에 대한 Jacobian이 $\mathbf{W}_2$이므로 gradient는 그 전치로 전달된다), ReLU를 거슬러 갈 때는 켜진 뉴런의 gradient만 통과시킨다(ReLU의 Jacobian은 0과 1로 된 대각행렬이다).

**batch로 쓰기.** 코드에서는 샘플 $N$개를 행으로 쌓은 $\mathbf{X} \in \mathbb{R}^{N \times D}$를 쓰고, 순전파는 $\mathbf{A} = \mathbf{X}\mathbf{W}_1^\top + \mathbf{1}\mathbf{b}_1^\top$이다. 손실이 batch 평균이면 출력 쪽 gradient는 $\mathbf{G}_Z = (\mathbf{P} - \mathbf{Y})/N$이고, 샘플별 외적의 합 $\sum_n \mathbf{g}_n\mathbf{h}_n^\top$은 행렬 곱 $\mathbf{G}_Z^\top\mathbf{H}$가 된다. bias gradient는 샘플 축으로 더한 것이다. 이 식을 그대로 구현하고 파라미터마다 중앙차분과 비교한다. 비교에는 **상대오차** $\lVert \mathbf{g} - \mathbf{g}_{\text{num}} \rVert / (\lVert \mathbf{g} \rVert + \lVert \mathbf{g}_{\text{num}} \rVert)$를 쓴다. 값의 크기와 무관하게 판단할 수 있어서다(8.4장).

```python {exec="browser" session="matcalc"}
N, D, H, K = 8, 3, 5, 4
Xb = rng.normal(size=(N, D))                      # (N, D)
labels = rng.integers(0, K, size=N)               # (N,)
Y = np.eye(K)[labels]                             # (N, K) one-hot
P0 = {"W1": rng.normal(size=(H, D)), "b1": rng.normal(size=H),
      "W2": rng.normal(size=(K, H)), "b2": rng.normal(size=K)}

def forward(P):
    A1 = Xb @ P["W1"].T + P["b1"]                 # (N, H)
    Hh = np.maximum(A1, 0)                        # (N, H)
    Z = Hh @ P["W2"].T + P["b2"]                  # (N, K)
    Z = Z - Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)   # (N, K)
    return A1, Hh, Pr, -np.mean(np.sum(Y * np.log(Pr), axis=1))

A1, Hh, Pr, loss = forward(P0)
Gz = (Pr - Y) / N                                 # (N, K) dL/dZ
Ga = (Gz @ P0["W2"]) * (A1 > 0)                   # (N, H) dL/dA1
grads = {"W2": Gz.T @ Hh, "b2": Gz.sum(0), "W1": Ga.T @ Xb, "b1": Ga.sum(0)}
for k, g in grads.items():
    f = lambda v, k=k: forward({**P0, k: v})[3]
    g_num = num_grad(f, P0[k])
    rel = np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num))
    print(f"{k}: shape {str(g.shape):7}  상대오차 {rel:.1e}")
```

```output
W2: shape (4, 5)   상대오차 2.7e-10
b2: shape (4,)     상대오차 3.2e-10
W1: shape (5, 3)   상대오차 3.6e-10
b1: shape (5,)     상대오차 1.5e-10
```

모든 gradient의 shape이 해당 파라미터와 같고, 상대오차가 $10^{-10}$ 수준이다. 행 벡터 batch에서는 `Gz @ W2`처럼 전치 없이 곱하는 것이 단일 샘플 식의 $\mathbf{W}_2^\top\mathbf{g}_z$와 같다는 점에 주의한다. $(\mathbf{W}_2^\top\mathbf{g}_z)^\top = \mathbf{g}_z^\top\mathbf{W}_2$이기 때문이다. 단일 샘플 식과 batch 식 사이를 오갈 때 전치가 한 번씩 뒤집힌다.

:::note[분야 예시]
천문학과 분광학에서 관측 스펙트럼 $\mathbf{y}$를 알려진 성분 스펙트럼들(설계행렬 $\mathbf{X}$의 열)의 선형 결합으로 분해하는 문제는 위의 선형회귀와 같은 구조다. 성분 비율이 음수가 될 수 없다는 제약을 더하면 비음수 최소제곱(non-negative least squares)이 되고, 이 경우에도 gradient $\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$는 그대로 쓰이며 사영 경사하강(projected gradient descent) 같은 방법이 제약을 처리한다(10장).
:::

손으로 유도한 gradient는 두 가지 쓸모가 있다. 첫째, 자동미분이 없는 환경(일부 시뮬레이션 코드, 임베디드 장치)에서 직접 구현할 수 있다. 둘째, 더 중요하게는 **자동미분의 결과를 읽을 수 있게** 된다. gradient가 0이 되는 이유(꺼진 ReLU, 포화된 softmax)나 커지는 이유(큰 가중치의 전치가 반복 곱해짐)를 식에서 바로 찾을 수 있다. 8.4장은 이 유도를 계산 그래프 위에서 자동으로 수행하는 방법이다.

## 실습

선형회귀와 2층 MLP의 gradient를 손으로 유도하고 NumPy로 구현한 뒤 수치미분으로 검증한다. 코드는 `labs/calculus-and-autodiff/matrix-calculus/gradcheck_models.py`이고 세 부분으로 되어 있다.

1. 여러 크기·시드에서 선형회귀와 MLP의 손 유도 gradient를 중앙차분과 비교한다.
2. 일부러 틀린 구현 두 가지를 같은 검사에 넣는다. ReLU 마스크를 빠뜨린 경우와, 은닉층 폭과 클래스 수가 같아($H = K$) shape이 우연히 맞는 상태에서 $\mathbf{W}_2$의 전치를 잘못 붙인 경우다.
3. 손 유도 gradient만으로 3개 클래스의 나선형 데이터에 2층 MLP를 학습한다.

집필 시점의 실행 결과는 다음과 같다.

```output
[1] 선형회귀: 손 유도 vs 중앙차분 (상대오차)
  N=  20 D=  2   w 1.3e-11   b 8.0e-12
  N= 100 D=  5   w 1.1e-10   b 5.2e-10
  N= 500 D= 10   w 1.5e-10   b 2.1e-09
[2] 2층 MLP: 손 유도 vs 중앙차분 (파라미터별 상대오차)
  N= 8 D=3 H= 5 K=4 정상             W1 2.3e-10  b1 1.9e-10  W2 7.8e-10  b2 1.5e-10
  N= 8 D=3 H= 5 K=4 no-mask        W1 3.0e-01  b1 4.2e-01  W2 7.8e-10  b2 1.5e-10
  N=16 D=4 H= 4 K=4 정상             W1 6.0e-10  b1 1.0e-09  W2 5.4e-10  b2 2.6e-10
  N=16 D=4 H= 4 K=4 no-mask        W1 3.6e-01  b1 5.3e-01  W2 5.4e-10  b2 2.6e-10
  N=16 D=4 H= 4 K=4 no-transpose   W1 8.5e-01  b1 9.5e-01  W2 5.4e-10  b2 2.6e-10
  N=32 D=6 H=10 K=3 정상             W1 7.3e-10  b1 5.2e-10  W2 3.5e-10  b2 1.9e-10
  N=32 D=6 H=10 K=3 no-mask        W1 4.1e-01  b1 3.4e-01  W2 3.5e-10  b2 1.9e-10
[3] 손 유도 gradient로 2층 MLP 학습 (3개 클래스, 2차원 입력)
  step    0  loss 1.1987  정확도 0.240
  step  500  loss 0.0503  정확도 0.997
  step 1000  loss 0.0338  정확도 0.997
  step 1500  loss 0.0274  정확도 0.997
  step 2000  loss 0.0239  정확도 0.997
```

결과를 읽는 법은 두 가지다.

- **버그는 그 버그가 지나는 파라미터에서만 드러난다.** 두 버그 모두 은닉층 아래(`W1`, `b1`)의 상대오차만 0.3–0.95로 키우고, 출력층(`W2`, `b2`)은 정상과 같다. gradient check를 파라미터별로 해야 버그의 위치를 좁힐 수 있다.
- **전치 버그는 shape 검사를 통과한다.** $H = K$인 경우에만 실행되었고 오류 없이 틀린 값을 냈다. shape이 맞는다는 것은 정답의 필요조건일 뿐이다.

이어서 두 연습문제를 푼다.

::exercise{id="softmax-ce-grad"}

::exercise{id="mlp-backward"}

## 실패 사례

- **전치를 빠뜨려 shape가 맞지 않는 gradient.** $\nabla_{\mathbf{W}}L$을 $\mathbf{x}\mathbf{g}^\top$으로 쓰면 shape이 $\mathbf{W}$의 전치가 된다. 정사각 행렬이면 오류 없이 지나가므로 더 위험하다. 유도한 식마다 shape을 주석으로 적고, gradient의 shape이 파라미터와 같은지 확인한다.
- **서로 다른 layout 규약의 공식을 섞는 것.** denominator layout 교재의 Jacobian 공식(입력 × 출력)을 numerator layout의 chain rule($\mathbf{J}_g\mathbf{J}_f$)에 넣으면 곱의 순서와 전치가 모두 어긋난다. 공식을 옮길 때는 출처의 규약을 먼저 확인하고, 작은 예제로 수치미분과 비교한다.
- **대칭 행렬을 가정한 공식을 일반 행렬에 쓰는 것.** $\nabla_{\mathbf{x}}\mathbf{x}^\top\mathbf{A}\mathbf{x} = 2\mathbf{A}\mathbf{x}$는 $\mathbf{A}$가 대칭일 때만 맞다. 반대로 공분산처럼 대칭이라는 제약이 있는 행렬을 미분할 때는, 독립 성분이 줄어들어 성분별 gradient의 해석이 달라진다. 그런 경우는 대칭 행렬을 $\mathbf{L}\mathbf{L}^\top$ 같은 제약 없는 파라미터로 다시 쓰고 미분하는 편이 안전하다.
- **batch 평균의 $1/N$을 빠뜨리는 것.** 손실은 평균인데 gradient는 합으로 구현하면 gradient가 batch 크기만큼 커진다. 학습률이 batch 크기에 따라 달라 보이는 이상한 현상의 원인이 된다.

## 통과 기준

- softmax cross-entropy의 gradient가 $\mathbf{p} - \mathbf{y}$임을 유도하고, 2층 MLP의 모든 파라미터 gradient를 미분형 또는 chain rule로 유도한다.
- 유도한 gradient를 NumPy로 구현해 파라미터별 상대오차가 $10^{-6}$ 이하임을 중앙차분으로 확인한다.
- 연습문제 두 개(`softmax-ce-grad`, `mlp-backward`)의 테스트를 통과한다.
