---
id: calculus-and-autodiff/multivariable-calculus
title: "다변수 미분"
title_en: "Multivariable Differentiation"
tier: core
status: draft
prerequisites: [linear-algebra/vector-spaces]
---

# 다변수 미분

신경망의 손실은 수백만 개의 파라미터를 입력으로 받는 함수이고, 학습은 그 함수의 기울기를 따라 내려가는 과정이다. 시뮬레이터의 민감도 분석, 분자의 힘 계산, 역문제 풀이도 모두 "입력을 조금 바꾸면 출력이 얼마나 바뀌는가"라는 같은 질문이다. 일변수 미분은 이 질문에 숫자 하나로 답하지만, 입력과 출력이 벡터가 되면 답은 벡터(gradient)와 행렬(Jacobian, Hessian)이 된다.

이 장은 그 세 대상을 정의하고 계산하는 법, 합성 함수에서 이들을 곱해 나가는 chain rule, 그리고 한 점 근처의 함수를 1차·2차 다항식으로 근사하는 Taylor 전개를 다룬다. [7.2장](../../07-linear-algebra/02-affine-and-nonlinear/index.md)에서 "미분은 한 점 근처에서 가장 좋은 아핀 근사"라고 했던 관점을 여러 변수로 넓히는 장이다.

## 학습 목표

- gradient·Jacobian·Hessian을 정의하고 계산한다
- chain rule로 합성 함수의 Jacobian을 유도한다
- Taylor 전개로 국소 근사를 한다

## 극한·연속·미분 가능성

일변수에서는 "도함수가 존재한다"와 "미분 가능하다"가 같은 말이다. 여러 변수에서는 그렇지 않다. 축 방향으로 각각 미분할 수 있어도 함수가 그 점에서 매끄럽지 않을 수 있다. 이 차이를 모르면 gradient가 계산된다는 사실만으로 그 gradient가 함수의 국소적 행동을 대표한다고 잘못 믿게 된다.

**정의.** $f: \mathbb{R}^n \to \mathbb{R}^m$이 점 $\mathbf{a}$에서 **미분 가능**(differentiable)하다는 것은, 어떤 선형사상 $\mathbf{L}: \mathbb{R}^n \to \mathbb{R}^m$이 있어

$$
\lim_{\mathbf{h} \to \mathbf{0}} \frac{\lVert f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) - \mathbf{L}\mathbf{h} \rVert}{\lVert \mathbf{h} \rVert} = 0
$$

이 성립한다는 뜻이다. $\mathbf{h}$가 **어느 방향에서** 0으로 다가가든 오차가 $\lVert \mathbf{h} \rVert$보다 빨리 사라져야 한다. 이 $\mathbf{L}$이 다음 절들에서 다룰 Jacobian이고, 식은 "$f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \mathbf{L}\mathbf{h}$가 1차까지 정확하다"는 말이다. 미분 가능하면 연속이다. 분자가 0으로 가야 하기 때문이다.

**편미분과의 차이.** 편미분(partial derivative) $\partial f / \partial x_j$는 $j$번째 축 방향으로만 움직였을 때의 변화율이다. 축 방향 $n$개만 보므로, 그 사이 방향에서 함수가 어떻게 행동하는지는 알려 주지 않는다. 고전적인 반례는

$$
f(x, y) = \frac{xy}{x^2 + y^2} \quad ((x, y) \neq (0, 0)), \qquad f(0, 0) = 0
$$

이다. 두 축 위에서 $f$는 항상 0이므로 원점에서 두 편미분은 모두 0이다. 그런데 대각선 $y = x$ 위에서는 $f(t, t) = 1/2$이므로, 원점에 아무리 가까이 가도 함수값이 $1/2$로 남는다. 원점에서 연속도 아니니 미분 가능할 수 없다.

```python {exec="browser" session="calc"}
import numpy as np

def f(x, y):
    if x == 0 and y == 0:
        return 0.0                                  # 원점에서는 0으로 정의
    return x * y / (x**2 + y**2)

h = 1e-6
print("df/dx(0,0) ≈", (f(h, 0.0) - f(-h, 0.0)) / (2 * h))
print("df/dy(0,0) ≈", (f(0.0, h) - f(0.0, -h)) / (2 * h))
for t in [1e-1, 1e-3, 1e-6]:
    print(f"f({t:g}, {t:g}) = {f(t, t):.3f}")
```

```output
df/dx(0,0) ≈ 0.0
df/dy(0,0) ≈ 0.0
f(0.1, 0.1) = 0.500
f(0.001, 0.001) = 0.500
f(1e-06, 1e-06) = 0.500
```

::quiz{id="partials-not-differentiable"}

실용적인 충분조건이 있다. 모든 편미분이 $\mathbf{a}$ 근처에서 존재하고 **연속**이면 $f$는 $\mathbf{a}$에서 미분 가능하다. 이런 함수를 $C^1$ 함수라 한다. 다항식, $\exp$, $\tanh$, softplus처럼 신경망에 쓰는 매끄러운 함수의 합성은 모두 $C^1$이다.

연구에서 문제가 되는 곳은 꺾이는 점이다. $\mathrm{ReLU}(z) = \max(0, z)$, $\lvert z \rvert$, $\max$ pooling은 특정 점에서 미분 가능하지 않다. 딥러닝 프레임워크는 그런 점에서도 정해진 값(예: PyTorch의 ReLU는 $z = 0$에서 0)을 gradient로 돌려준다. 이는 수학적 도함수가 아니라 구현의 규약이다. 꺾이는 점은 측도가 0인 집합이라 학습 중에 정확히 걸리는 일이 드물어 대부분 문제가 되지 않지만, 입력이나 파라미터가 정확히 0에 모이는 구조(예: 희소성을 유도하는 L1 항)에서는 이 규약이 결과에 영향을 준다.

## 편미분과 방향미분

손실을 줄이려면 "어느 방향으로 움직여야 가장 빨리 줄어드는가"를 알아야 한다. 방향미분이 그 질문의 답이고, gradient는 모든 방향의 답을 벡터 하나에 압축한다.

**정의.** 스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 점 $\mathbf{a}$에서 단위벡터 $\mathbf{u}$ 방향의 **방향미분**(directional derivative)은

$$
D_{\mathbf{u}} f(\mathbf{a}) = \lim_{t \to 0} \frac{f(\mathbf{a} + t\mathbf{u}) - f(\mathbf{a})}{t}
$$

이다. $\mathbf{u}$가 $j$번째 표준기저 $\mathbf{e}_j$이면 편미분 $\partial f / \partial x_j$가 된다. 편미분들을 모은 열벡터가 **gradient**다.

$$
\nabla f(\mathbf{a}) = \begin{bmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{bmatrix} \in \mathbb{R}^n.
$$

이 책에서 gradient는 항상 **열벡터**다([부록 A](../../appendices/a-notation/index.md)). $f$가 미분 가능하면 모든 방향미분은 gradient와의 내적이다.

$$
D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u}.
$$

**기하적 의미.** Cauchy–Schwarz 부등식으로 $\nabla f^\top \mathbf{u} \le \lVert \nabla f \rVert \lVert \mathbf{u} \rVert = \lVert \nabla f \rVert$이고, 등호는 $\mathbf{u} = \nabla f / \lVert \nabla f \rVert$일 때 성립한다. 그래서 두 가지가 나온다.

- **gradient는 가장 가파르게 올라가는 방향**이고, 그 방향의 변화율은 $\lVert \nabla f \rVert$다. 반대 방향 $-\nabla f$가 가장 가파르게 내려가는 방향이며, 경사하강법(gradient descent)의 이름이 여기서 왔다.
- **gradient는 등고선(level set)에 수직**이다. 등고선 위의 곡선 $\boldsymbol{\gamma}(t)$를 따라가면 $f(\boldsymbol{\gamma}(t))$가 일정하므로, 미분하면 $\nabla f^\top \boldsymbol{\gamma}'(t) = 0$이다. 등고선의 접선 방향으로는 함수값이 1차까지 변하지 않는다.

등고선이 타원인 $f(x, y) = x^2 + 3y^2$에서 점 $(1, 1)$의 모든 방향미분을 계산해 확인한다. 이 함수의 gradient는 $(2x, 6y)$다.

```python {exec="browser" session="calc"}
def loss(p):
    x, y = p
    return x**2 + 3 * y**2

def grad(p):
    x, y = p
    return np.array([2 * x, 6 * y])                 # (2,) 열벡터

p, h = np.array([1.0, 1.0]), 1e-6
g = grad(p)
angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
U = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (3600, 2) 단위벡터
D = (np.array([loss(p + h * u) for u in U]) - loss(p)) / h  # (3600,) 방향미분
u_best = U[np.argmax(D)]
print("가장 가파른 방향 ≈", u_best.round(3), " g/|g| =", (g / np.linalg.norm(g)).round(3))
print("최대 방향미분   ≈", round(D.max(), 3), " |g| =", round(np.linalg.norm(g), 3))
tangent = np.array([-g[1], g[0]]) / np.linalg.norm(g)  # 등고선의 단위 접선
print("접선 방향의 방향미분 ≈", round((loss(p + h * tangent) - loss(p)) / h, 4))
```

```output
가장 가파른 방향 ≈ [0.316 0.949]  g/|g| = [0.316 0.949]
최대 방향미분   ≈ 6.325  |g| = 6.325
접선 방향의 방향미분 ≈ 0.0
```

3600개 방향을 모두 시도해 찾은 가장 가파른 방향이 $\nabla f / \lVert \nabla f \rVert$와 같고, 그 변화율이 $\lVert \nabla f \rVert = \sqrt{40}$이다. 이 방향은 원점에서 멀어지는 방향 $(1, 1)/\sqrt{2}$가 아니다. 등고선이 y 방향으로 납작하므로 gradient가 y축 쪽으로 기운다.

:::note[분야 예시]
분자동역학에서 원자들이 받는 힘은 위치에 대한 퍼텐셜 에너지 $U(\mathbf{r})$의 음의 gradient, $\mathbf{F} = -\nabla U$다. 원자 $N$개면 $\mathbf{r} \in \mathbb{R}^{3N}$이고 힘도 $3N$차원 벡터다. 에너지를 신경망으로 학습하는 머신러닝 퍼텐셜은 힘을 따로 예측하지 않고 에너지 모델의 gradient로 계산하는 경우가 많다. 그러면 힘이 어떤 에너지 함수의 gradient라는 물리적 성질(보존력)이 구조적으로 보장된다.
:::

## Jacobian과 Hessian

출력이 벡터인 함수, 예를 들어 신경망의 한 층이나 좌표 변환은 출력 성분마다 gradient가 있다. 이것들을 한 행렬에 모은 것이 Jacobian이다. 스칼라 함수의 gradient를 다시 미분하면 곡률을 담은 Hessian이 된다.

**Jacobian.** $f: \mathbb{R}^n \to \mathbb{R}^m$의 Jacobian은 $(i, j)$ 성분이 $\partial f_i / \partial x_j$인 $m \times n$ 행렬이다.

$$
\mathbf{J}_f(\mathbf{x}) = \begin{bmatrix} \nabla f_1(\mathbf{x})^\top \\ \vdots \\ \nabla f_m(\mathbf{x})^\top \end{bmatrix} \in \mathbb{R}^{m \times n}.
$$

행은 출력 성분, 열은 입력 성분이다(출력 차원 × 입력 차원). $i$번째 행은 $f_i$의 gradient를 눕힌 것이고, $j$번째 열은 $x_j$만 움직일 때 출력 벡터가 움직이는 속도다. 미분 가능성 정의의 선형사상 $\mathbf{L}$이 바로 $\mathbf{J}_f(\mathbf{a})$이므로

$$
f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \mathbf{J}_f(\mathbf{a})\,\mathbf{h}
$$

다. 스칼라 함수($m = 1$)에서는 $\mathbf{J}_f = \nabla f^\top$, 즉 Jacobian은 gradient를 전치한 $1 \times n$ 행이다. 이 전치 관계가 다음 장(8.2)의 layout 규약 문제의 출발점이다.

예를 들어 극좌표에서 직교좌표로 가는 사상 $(r, \theta) \mapsto (r\cos\theta, r\sin\theta)$의 Jacobian은

$$
\mathbf{J} = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}, \qquad \det \mathbf{J} = r
$$

이다. 행렬식 $r$은 이 변환이 작은 넓이를 얼마나 늘리는지를 나타내고, 8.3장의 변수변환에서 그대로 쓰인다. 선형사상 $\mathbf{A}\mathbf{x}$의 Jacobian은 어디서나 $\mathbf{A}$이고, 아핀 사상 $\mathbf{A}\mathbf{x} + \mathbf{b}$도 마찬가지다.

**Hessian.** 두 번 미분 가능한 스칼라 함수 $f$의 Hessian은 2계 편미분을 모은 $n \times n$ 행렬이다.

$$
\mathbf{H}_f(\mathbf{x})_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j}, \qquad \mathbf{H}_f = \mathbf{J}_{\nabla f}.
$$

Hessian은 gradient라는 벡터 함수의 Jacobian이다. 2계 편미분이 연속이면($C^2$) 미분 순서를 바꿔도 같으므로(Schwarz 정리) Hessian은 **대칭행렬**이다. 그래서 실수 고윳값과 서로 직교하는 고유벡터를 가진다(7.5장).

**곡률.** 단위 방향 $\mathbf{u}$로 두 번 미분하면 $D_{\mathbf{u}}^2 f = \mathbf{u}^\top \mathbf{H} \mathbf{u}$다. 이 값이 그 방향의 곡률이다. $\mathbf{u}$가 고유벡터 $\mathbf{v}_i$이면 곡률은 고윳값 $\lambda_i$가 되고, 가장 큰 고윳값과 가장 작은 고윳값이 곡률의 최댓값과 최솟값이다. gradient가 0인 점(임계점, critical point)에서는

- 모든 고윳값이 양수이면 국소 최소점,
- 모두 음수이면 국소 최대점,
- 양수와 음수가 섞이면 안장점(saddle point)

이다. 고윳값 중 0이 있으면 2차 정보만으로는 판정할 수 없다.

Rosenbrock 함수 $f(x, y) = (1 - x)^2 + b(y - x^2)^2$(여기서는 $b = 10$)으로 확인한다. 최소점은 $(1, 1)$이고, 바나나 모양으로 휜 좁은 골짜기를 가진다.

```python {exec="browser" session="calc"}
def rosen_hess(x, y, b=10.0):
    return np.array([[2 - 4 * b * y + 12 * b * x**2, -4 * b * x],
                     [-4 * b * x, 2 * b]])          # (2, 2)

for pt in [(1.0, 1.0), (0.0, 0.5)]:
    H = rosen_hess(*pt)
    lam, V = np.linalg.eigh(H)                      # 오름차순 고윳값, 열이 고유벡터
    print(f"점 {pt}: 고윳값 {lam.round(3)}, 최소 고윳값의 방향 {V[:, 0].round(3)}")
lam = np.linalg.eigvalsh(rosen_hess(1.0, 1.0))
print("최소점의 조건수 λmax/λmin =", round(lam[1] / lam[0], 1))
```

```output
점 (1.0, 1.0): 고윳값 [  0.394 101.606], 최소 고윳값의 방향 [-0.44  -0.898]
점 (0.0, 0.5): 고윳값 [-18.  20.], 최소 고윳값의 방향 [1. 0.]
최소점의 조건수 λmax/λmin = 258.1
```

최소점 $(1, 1)$에서 두 고윳값은 모두 양수지만 크기가 250배 넘게 다르다. 곡률이 작은 방향 $(-0.44, -0.90)$(부호는 임의)으로는 함수가 거의 평평하고, 그 수직 방향으로는 가파르다. 등고선이 이 방향으로 길게 늘어진 좁은 타원이 되고, 10장에서 보듯 경사하강법은 이런 골짜기에서 벽 사이를 오가며 느리게 나아간다. 점 $(0, 0.5)$는 임계점이 아니지만(gradient가 0이 아님) 고윳값 하나가 음수다. 이 점 근처에서 $x$ 방향 단면은 위로 볼록하지 않고 아래로 휜다. 손실 지형에서 음의 곡률 방향은 임계점이 아닌 곳에도 흔하다.

:::note[분야 예시]
계산화학에서 분자의 퍼텐셜 에너지 곡면(potential energy surface) 위의 안정 구조는 에너지의 국소 최소점이고, 그 점의 Hessian(질량으로 가중한 것)의 고윳값에서 분자의 진동 진동수를 얻는다. 반응의 전이 상태(transition state)는 고윳값 하나만 음수인 안장점이다. "최소점인가 안장점인가"를 Hessian의 부호로 판정하는 일이 화학 반응 경로 탐색의 기본 단계다.
:::

::widget{name="gradient-hessian-contour"}

## chain rule

신경망은 층을 합성한 함수이고, 손실은 그 끝에 붙은 또 하나의 함수다. 합성 함수의 미분을 각 부분의 미분으로 나누어 계산하는 규칙이 chain rule이고, 역전파(8.4장)는 이 규칙을 효율적인 순서로 적용하는 알고리즘이다.

**정리.** $f: \mathbb{R}^n \to \mathbb{R}^k$가 $\mathbf{x}$에서, $g: \mathbb{R}^k \to \mathbb{R}^m$이 $f(\mathbf{x})$에서 미분 가능하면 $h = g \circ f$도 $\mathbf{x}$에서 미분 가능하고

$$
\mathbf{J}_h(\mathbf{x}) = \mathbf{J}_g\big(f(\mathbf{x})\big)\, \mathbf{J}_f(\mathbf{x}) \qquad (m \times n) = (m \times k)(k \times n)
$$

이다. 바깥 함수의 Jacobian이 왼쪽에 오고, **안쪽 함수의 출력에서** 계산한다. shape을 맞춰 보면 곱의 순서가 저절로 정해진다.

**직관.** 각 Jacobian은 그 함수의 국소 선형 근사다. 선형 근사끼리 합성하면 행렬 곱이므로, 합성 함수의 선형 근사는 Jacobian의 곱이다. 입력의 작은 변화 $\mathbf{h}$가 $f$를 지나며 $\mathbf{J}_f\mathbf{h}$가 되고, 다시 $g$를 지나며 $\mathbf{J}_g\mathbf{J}_f\mathbf{h}$가 된다.

**성분으로 보면.** 한 변수가 여러 경로로 결과에 영향을 주면 경로마다의 기여를 더한다. $u = xy$, $x = s + t$, $y = s - t$이면

$$
\frac{\partial u}{\partial s} = \frac{\partial u}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial u}{\partial y}\frac{\partial y}{\partial s} = y + x
$$

다. 행렬 곱의 한 성분 $\sum_k (\mathbf{J}_g)_{ik}(\mathbf{J}_f)_{kj}$가 바로 "중간 변수 $k$를 거치는 모든 경로의 합"이다.

::quiz{id="chain-rule-order"}

**신경망 층.** 은닉층 $\mathbf{x} \mapsto \mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 \mapsto \tanh(\mathbf{a}) \mapsto \mathbf{W}_2\tanh(\mathbf{a})$는 세 함수의 합성이다. 아핀 사상의 Jacobian은 $\mathbf{W}_1$, 성분별 함수 $\tanh$의 Jacobian은 대각행렬 $\operatorname{diag}(1 - \tanh^2(\mathbf{a}))$, 마지막 선형사상의 Jacobian은 $\mathbf{W}_2$다. 따라서

$$
\mathbf{J}(\mathbf{x}) = \mathbf{W}_2 \, \operatorname{diag}\big(1 - \tanh^2(\mathbf{a})\big)\, \mathbf{W}_1.
$$

```python {exec="browser" session="calc"}
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)   # (4, 3), (4,)
W2 = rng.normal(size=(2, 4))                            # (2, 4)

def net(x):
    return W2 @ np.tanh(W1 @ x + b1)                    # (3,) -> (2,)

x = rng.normal(size=3)
a = W1 @ x + b1                                         # (4,)
J = W2 @ np.diag(1 - np.tanh(a) ** 2) @ W1              # (2,4)(4,4)(4,3) = (2, 3)

eps = 1e-6
J_fd = np.stack([(net(x + eps * e) - net(x - eps * e)) / (2 * eps)
                 for e in np.eye(3)], axis=1)           # (2, 3)
print("J shape:", J.shape)
print("chain rule와 유한차분의 최대 차이 < 1e-8:", np.abs(J - J_fd).max() < 1e-8)
```

```output
J shape: (2, 3)
chain rule와 유한차분의 최대 차이 < 1e-8: True
```

손으로 유도한 Jacobian과 중앙차분으로 근사한 Jacobian의 차이가 $10^{-8}$보다 작다. 남은 차이($10^{-10}$ 안팎)는 유한차분의 반올림 오차라서 정확한 값은 실행 환경마다 조금씩 다르므로, 셀은 판정만 찍는다. 이렇게 유한차분으로 해석적 미분을 확인하는 절차를 gradient check라 하고, 8.4장에서 정밀도 문제까지 포함해 자세히 다룬다.

**곱의 순서와 비용.** 층이 $L$개인 신경망의 Jacobian은 $\mathbf{J}_L \cdots \mathbf{J}_2 \mathbf{J}_1$이다. 행렬 곱은 결합법칙을 만족하므로 오른쪽부터 곱해도(입력 쪽부터) 왼쪽부터 곱해도(출력 쪽부터) 결과는 같지만, 계산량은 다르다. 손실처럼 출력이 스칼라 하나이면 맨 왼쪽 행렬이 $1 \times k$ 행이므로, 왼쪽부터 곱하면 매 단계가 "행벡터 × 행렬"이 되어 큰 행렬끼리의 곱을 피할 수 있다. 이것이 reverse mode 자동미분, 즉 역전파의 핵심 아이디어이고 8.4장의 주제다.

:::note[분야 예시]
생태학의 포식자–피식자 모형처럼 상태 $\mathbf{s}$가 $\dot{\mathbf{s}} = F(\mathbf{s})$를 따르는 동역학계에서, 평형점 $\mathbf{s}^\ast$($F(\mathbf{s}^\ast) = \mathbf{0}$) 근처의 작은 교란 $\boldsymbol{\delta}$는 $\dot{\boldsymbol{\delta}} \approx \mathbf{J}_F(\mathbf{s}^\ast)\boldsymbol{\delta}$를 따른다. Jacobian의 고윳값 실수부가 모두 음수이면 교란이 줄어들어 평형이 안정하다. 기후 모델, 화학 반응망, 전력망의 안정성 분석이 모두 같은 선형화 절차를 쓴다.
:::

## Taylor 전개

1차 근사는 gradient만 쓰고, 2차 근사는 Hessian까지 쓴다. 최적화 알고리즘은 결국 "현재 위치에서 손실을 어떤 다항식으로 근사하고, 그 근사를 얼마나 믿고 움직일 것인가"의 선택이다.

**식.** $f$가 $\mathbf{a}$ 근처에서 충분히 매끄러우면

$$
f(\mathbf{a} + \mathbf{h}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^\top \mathbf{h} + \tfrac{1}{2}\mathbf{h}^\top \mathbf{H}_f(\mathbf{a})\,\mathbf{h} + O(\lVert \mathbf{h} \rVert^3)
$$

이다. 둘째 항까지 쓰면 1차 근사, 셋째 항까지 쓰면 2차 근사다. 1차 근사의 오차는 $\tfrac{1}{2}\mathbf{h}^\top\mathbf{H}\mathbf{h}$가 지배하므로 $\lVert \mathbf{h} \rVert^2$에 비례하고, 2차 근사의 오차는 $\lVert \mathbf{h} \rVert^3$에 비례한다. 유도는 일변수 Taylor 전개를 직선 $t \mapsto f(\mathbf{a} + t\mathbf{u})$에 적용하면 된다. 이 함수의 1계·2계 도함수가 $\nabla f^\top \mathbf{u}$와 $\mathbf{u}^\top \mathbf{H} \mathbf{u}$이기 때문이다.

Rosenbrock 함수의 점 $(0, 0.5)$에서 방향 $(0.6, 0.8)$로 거리 $\lVert \mathbf{h} \rVert$만큼 움직이며 두 근사의 오차를 잰다.

```python {exec="browser" session="calc"}
def rosen(p, b=10.0):
    x, y = p
    return (1 - x) ** 2 + b * (y - x**2) ** 2

def rosen_grad(p, b=10.0):
    x, y = p
    return np.array([-2 * (1 - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])

a0 = np.array([0.0, 0.5])
g0, H0 = rosen_grad(a0), rosen_hess(*a0)
d = np.array([0.6, 0.8])                                # 단위 방향
for t in [0.1, 0.05, 0.025]:
    s = t * d
    e1 = abs(rosen(a0 + s) - (rosen(a0) + g0 @ s))
    e2 = abs(rosen(a0 + s) - (rosen(a0) + g0 @ s + 0.5 * s @ H0 @ s))
    print(f"|h| = {t:5.3f}  1차 오차 {e1:.2e} (/|h|^2 {e1 / t**2:.2f})  2차 오차 {e2:.2e} (/|h|^3 {e2 / t**3:.2f})")
```

```output
|h| = 0.100  1차 오차 2.60e-02 (/|h|^2 2.60)  2차 오차 5.63e-03 (/|h|^3 5.63)
|h| = 0.050  1차 오차 7.19e-03 (/|h|^2 2.88)  2차 오차 7.12e-04 (/|h|^3 5.70)
|h| = 0.025  1차 오차 1.89e-03 (/|h|^2 3.02)  2차 오차 8.95e-05 (/|h|^3 5.73)
```

거리를 절반으로 줄일 때 1차 오차는 약 1/4, 2차 오차는 약 1/8로 줄어든다. 1차 오차를 $\lVert \mathbf{h} \rVert^2$으로 나눈 값은 $\tfrac{1}{2}\mathbf{u}^\top\mathbf{H}\mathbf{u} = \tfrac{1}{2}(-18 \cdot 0.36 + 20 \cdot 0.64) = 3.16$에 다가간다.

**최적화와의 연결.** 두 근사는 서로 다른 최적화 방법으로 이어진다.

- **1차 근사**는 $\mathbf{h}$에 대해 선형이라 최소점이 없다. 대신 가장 빨리 내려가는 방향 $-\nabla f$를 알려 주고, 얼마나 움직일지(학습률)는 근사가 믿을 만한 범위를 보고 따로 정해야 한다. 경사하강법이다(10.3장).
- **2차 근사**는 $\mathbf{H}$가 양의 정부호이면 최소점이 있다. $\mathbf{h}$로 미분해 0으로 두면 $\nabla f + \mathbf{H}\mathbf{h} = \mathbf{0}$, 즉 $\mathbf{h} = -\mathbf{H}^{-1}\nabla f$다. Newton 방법이다(10.5장). 곡률이 큰 방향은 덜, 작은 방향은 더 움직여 좁은 골짜기를 한 번에 가로지른다.
- **임계점**($\nabla f = \mathbf{0}$)에서는 1차 항이 사라지고 2차 항 $\tfrac{1}{2}\mathbf{h}^\top\mathbf{H}\mathbf{h}$가 함수 모양을 결정한다. 앞 절의 고윳값 판정법이 이 식에서 나온다.

::quiz{id="taylor-quadratic-model"}

Taylor 근사의 유효 범위는 3차 이상의 항이 무시할 만한 거리까지다. 학습률이 너무 크면 1차 근사가 맞지 않는 거리까지 뛰어 손실이 오히려 커지고, Newton 방법도 $\mathbf{H}$가 양의 정부호가 아닌 곳에서는 엉뚱한 방향을 가리킨다. 실제 연구의 최적화 기법 대부분(학습률 스케줄, trust region, 감쇠 Newton)은 근사를 믿을 거리를 조절하는 장치다.

## 실습

2변수 손실함수의 등고선 위에 gradient와 Hessian 고유벡터를 그려 곡률을 시각화한다. 코드는 `labs/calculus-and-autodiff/multivariable-calculus/curvature_plot.py`이고, Rosenbrock 함수($b = 10$)의 다섯 점에서 해석적 gradient·Hessian을 유한차분으로 검증한 뒤 수치를 출력하고 그림을 `outputs/curvature.png`에 저장한다. 그림에서 빨간 화살표는 $-\nabla f$, 검은 선분은 Hessian의 고유벡터(실선은 양의 곡률, 점선은 음의 곡률)다.

집필 시점의 실행 결과는 다음과 같다.

```output
             점        f   |grad|        λ1        λ2  곡률
    (1.0, 1.0)    0.000    0.000     0.394   101.606  볼록(양의 정부호)
    (0.0, 0.5)    3.500   10.198   -18.000    20.000  부정부호
   (-1.0, 1.5)    6.500   18.868    -4.177    86.177  부정부호
   (0.5, -0.3)    3.275   14.866     8.676    55.324  볼록(양의 정부호)
   (-0.6, 0.0)    3.856   13.857     5.494    59.706  볼록(양의 정부호)
그림 저장: labs/calculus-and-autodiff/multivariable-calculus/outputs/curvature.png
```

그림과 표를 보며 세 가지를 확인한다.

- 빨간 화살표가 각 점에서 등고선에 수직인가.
- 최소점 $(1, 1)$에서 곡률이 작은 고유벡터(긴 축)가 골짜기 바닥이 뻗은 방향과 일치하는가.
- 부정부호인 점 두 곳에서 점선(음의 곡률) 방향으로 단면이 아래로 휘는가. 이 점들은 임계점이 아니므로 안장점은 아니다.

이어서 두 연습문제를 푼다. 첫째는 어떤 함수에도 쓸 수 있는 유한차분 Jacobian을 만들고, 둘째는 2층 신경망의 Jacobian을 chain rule로 유도한다. 첫째에서 만든 함수가 둘째의 답을 검증하는 도구가 된다.

::exercise{id="numerical-jacobian"}

::exercise{id="tanh-layer-jacobian"}

## 실패 사례

- **gradient를 행벡터·열벡터로 섞어 쓰는 것.** 어떤 교재는 스칼라 함수의 미분을 행벡터 $\partial f / \partial \mathbf{x}$(Jacobian, $1 \times n$)로, 어떤 교재는 열벡터 $\nabla f$($n \times 1$)로 쓴다. 두 규약의 공식을 섞으면 chain rule이 $\mathbf{J}_g\mathbf{J}_f$가 아니라 $\mathbf{J}_f^\top\mathbf{J}_g^\top$ 꼴로 바뀌고, 차원이 우연히 맞는 정사각 행렬에서는 오류가 드러나지 않은 채 전치된 결과가 나온다. 이 책은 gradient를 열벡터, Jacobian을 (출력 × 입력)으로 고정한다. 공식을 옮겨 올 때는 shape을 먼저 확인한다.
- **편미분이 존재하면 미분 가능하다고 가정하는 것.** 축 방향 변화율만으로 국소 선형 근사가 성립한다고 보면, 꺾이거나 방향마다 극한이 다른 점에서 잘못된 결론을 낸다.
- **gradient가 0이면 최소점이라고 결론 내리는 것.** 임계점은 안장점일 수 있고, 고차원 손실 지형에서는 안장점이 흔하다. Hessian의 고윳값 부호(또는 그 근사)를 확인해야 한다.
- **Hessian이 작은 곳만 보고 지형이 평평하다고 판단하는 것.** 조건수가 큰 지형은 한 방향으로는 평평하고 다른 방향으로는 가파르다. 고윳값 하나가 아니라 가장 큰 값과 가장 작은 값의 비를 본다.

## 통과 기준

- 임의의 합성 함수에 대해 chain rule로 Jacobian을 유도하고, 유한차분 Jacobian과 비교해 일치를 확인한다.
- 2변수 함수의 한 점에서 gradient와 Hessian을 계산하고, Hessian 고윳값으로 그 점의 곡률(볼록·부정부호)과 등고선 모양을 설명한다.
- 연습문제 두 개(`numerical-jacobian`, `tanh-layer-jacobian`)의 테스트를 통과한다.
