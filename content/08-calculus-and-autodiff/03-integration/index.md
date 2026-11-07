---
id: calculus-and-autodiff/integration
title: "적분과 변수변환"
title_en: "Integration and Change of Variables"
tier: core
status: draft
prerequisites: [calculus-and-autodiff/multivariable-calculus]
---

# 적분과 변수변환

확률은 적분이다. 기댓값, 주변분포, 정규화 상수, 우도가 모두 적분으로 정의되고, 그 적분은 대부분 여러 변수에 걸쳐 있다. 머신러닝에서 미분만큼 자주 보이지 않는 이유는 적분을 직접 계산하기보다 표본 평균이나 정규화된 모델로 우회하기 때문인데, 그 우회가 맞는지 판단하려면 적분이 무엇을 하는지 알아야 한다.

이 장의 중심은 **변수변환**(change of variables)이다. 좌표를 바꾸면 작은 부피가 늘거나 줄고, 그 배율이 Jacobian 행렬식이다. 이 사실 하나가 극좌표 적분, 가우시안의 정규화 상수, 확률밀도의 변환 공식을 모두 설명하고, 생성모델의 한 갈래인 normalizing flow의 로그우도로 그대로 이어진다.

## 학습 목표

- 다중적분을 계산한다
- 변수변환과 Jacobian 행렬식을 이해한다
- 확률밀도 변환에 적용한다

## 다중적분

2차원 이상의 확률분포에서 확률·기댓값·주변분포를 구하려면 여러 변수에 대한 적분이 필요하다. 다중적분을 계산하는 실제 방법은 거의 항상 **한 변수씩 차례로 적분하는 것**이고, 이때 영역의 경계를 정확히 기술하는 것이 핵심이다.

**정의.** 영역 $\mathcal{R} \subset \mathbb{R}^2$ 위의 함수 $f$의 적분 $\iint_{\mathcal{R}} f\,dA$는 영역을 작은 칸으로 나누고 (칸 넓이) × (칸 안의 함수값)을 더한 합(Riemann 합)의 극한이다. 1차원 적분이 곡선 아래 넓이라면, 2차원 적분은 곡면 아래 부피다. $f$가 확률밀도이면 그 영역에 떨어질 확률이다.

**반복적분과 Fubini 정리.** $f$가 영역 위에서 연속이거나, 더 일반적으로 $\iint \lvert f \rvert\,dA < \infty$이면 다중적분은 반복적분(iterated integral)과 같고, 적분 순서를 바꿔도 값이 같다(Fubini 정리). 직사각형 영역에서는 한계가 상수이므로 순서 교환이 단순하다.

$$
\iint_{[a, b] \times [c, d]} f\,dA = \int_a^b\!\!\int_c^d f(x, y)\,dy\,dx = \int_c^d\!\!\int_a^b f(x, y)\,dx\,dy.
$$

**영역이 직사각형이 아닐 때.** 안쪽 적분의 한계가 바깥 변수에 의존한다. 삼각형 $0 \le y \le x \le 1$에서 $x$를 바깥에 두면 $y$는 $0$부터 $x$까지, $y$를 바깥에 두면 $x$는 $y$부터 $1$까지 움직인다.

$$
\int_0^1\!\!\int_0^x f(x, y)\,dy\,dx = \int_0^1\!\!\int_y^1 f(x, y)\,dx\,dy.
$$

순서를 바꿀 때는 한계를 기계적으로 옮기지 말고, 영역을 그린 뒤 바깥 변수를 고정한 선분이 영역에 들어가고 나가는 위치를 읽는다. $f(x, y) = xy^2$이면 앞의 순서로 $\int_0^1 x \cdot x^3/3\,dx = 1/15$, 뒤의 순서로 $\int_0^1 y^2(1 - y^2)/2\,dy = 1/15$다.

::quiz{id="iterated-integral-limits"}

```python {exec="browser" session="integ"}
import numpy as np
from scipy import integrate

f = lambda x, y: x * y**2
# dblquad(func(안쪽, 바깥), 바깥 하한, 바깥 상한, 안쪽 하한(바깥), 안쪽 상한(바깥))
I_dy_dx, _ = integrate.dblquad(lambda y, x: f(x, y), 0, 1, lambda x: 0, lambda x: x)
I_dx_dy, _ = integrate.dblquad(lambda x, y: f(x, y), 0, 1, lambda y: y, lambda y: 1)

n = 2000                                            # 격자 Riemann 합
c = (np.arange(n) + 0.5) / n                        # (n,) 칸의 중심
Xg, Yg = np.meshgrid(c, c, indexing="ij")           # (n, n)
riemann = np.sum(f(Xg, Yg) * (Yg <= Xg)) / n**2
print(f"dy 먼저 {I_dy_dx:.6f}   dx 먼저 {I_dx_dy:.6f}   Riemann {riemann:.6f}   1/15 = {1/15:.6f}")
```

```output
dy 먼저 0.066667   dx 먼저 0.066667   Riemann 0.066729   1/15 = 0.066667
```

두 순서의 반복적분이 같고, 격자 Riemann 합도 가깝다. Riemann 합이 네 번째 자리에서 어긋나는 것은 대각선 경계에 걸친 칸을 통째로 넣거나 빼기 때문이다. 영역의 경계를 칸으로 근사하는 오차는 칸 크기에 비례해 줄어 내부 오차보다 느리게 줄어든다. 수치적분 방법은 9.5장에서 다룬다.

:::caution[함정]
Fubini 정리의 조건 $\iint \lvert f \rvert\,dA < \infty$는 장식이 아니다. $f(x, y) = (x^2 - y^2)/(x^2 + y^2)^2$을 $[0, 1]^2$에서 적분하면, $y$를 먼저 적분할 때 $\int_0^1 \frac{dx}{1 + x^2} = \pi/4$, $x$를 먼저 적분할 때 $-\pi/4$가 나온다. 원점 근처에서 $\lvert f \rvert$의 적분이 발산하기 때문이다. 확률밀도처럼 음이 아닌 함수에서는 이런 일이 생기지 않지만, 부호가 바뀌는 피적분함수(예: 점수 함수나 gradient의 기댓값)에서 적분과 극한·미분의 순서를 바꿀 때는 조건을 확인해야 한다.
:::

## 변수변환

적분 영역이 원이나 회전된 타원이면 직교좌표의 한계가 복잡해진다. 영역이나 피적분함수의 대칭에 맞는 좌표로 바꾸면 적분이 단순해지는데, 그 대가로 **부피 배율**을 곱해야 한다.

**선형 변환의 경우.** 선형사상 $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$는 단위 정사각형을 두 열벡터가 만드는 평행사변형으로 보내고, 그 넓이는 $\lvert\det\mathbf{A}\rvert$다. 모든 영역은 작은 정사각형으로 채울 수 있으므로, 선형사상은 **모든 영역의 넓이를 똑같이 $\lvert\det\mathbf{A}\rvert$배** 한다. 행렬식의 부호는 방향(시계·반시계)을 뒤집는지를 나타낼 뿐 넓이와는 관계없다.

**비선형 변환의 경우.** 8.1장에서 본 것처럼 미분 가능한 사상 $g$는 한 점 근처에서 아핀 사상 $g(\mathbf{u} + \mathbf{h}) \approx g(\mathbf{u}) + \mathbf{J}_g(\mathbf{u})\mathbf{h}$처럼 행동한다. 그래서 점 $\mathbf{u}$ 근처의 작은 칸은 넓이가 $\lvert\det\mathbf{J}_g(\mathbf{u})\rvert$배가 된다. 배율이 점마다 다를 뿐 원리는 같다.

**변수변환 공식.** $g: \mathcal{U} \to \mathcal{X}$가 가역이고 연속적으로 미분 가능하며 역사상도 그렇다면

$$
\int_{\mathcal{X}} f(\mathbf{x})\,d\mathbf{x} = \int_{\mathcal{U}} f\big(g(\mathbf{u})\big)\,\big\lvert\det\mathbf{J}_g(\mathbf{u})\big\rvert\,d\mathbf{u}
$$

이다. 왼쪽은 $\mathbf{x}$ 공간의 칸으로 더한 합, 오른쪽은 $\mathbf{u}$ 공간의 칸으로 더한 합이다. $\mathbf{u}$ 공간의 칸 $d\mathbf{u}$는 $\mathbf{x}$ 공간에서 넓이 $\lvert\det\mathbf{J}_g\rvert d\mathbf{u}$인 칸에 대응하므로 그 배율을 곱한다. 1차원에서는 익숙한 치환적분 $dx = \lvert g'(u) \rvert du$다.

**극좌표.** $(r, \theta) \mapsto (r\cos\theta, r\sin\theta)$의 Jacobian 행렬식은 8.1장에서 계산했듯 $r$이다. 그래서

$$
dx\,dy = r\,dr\,d\theta.
$$

반지름 $r$에서 각도 $d\theta$가 만드는 호의 길이가 $r\,d\theta$라는 기하와 맞는다. 원점에서 멀수록 같은 $(dr, d\theta)$ 칸이 넓은 영역에 대응한다.

::quiz{id="polar-jacobian"}

두 사실을 계산으로 확인한다. 첫째, 선형사상 $\mathbf{A}$가 단위 정사각형을 보낸 평행사변형의 넓이를 Monte Carlo로 재서 $\lvert\det\mathbf{A}\rvert$와 비교한다. 둘째, 극좌표 격자로 단위원의 넓이를 더할 때 $r$을 곱한 경우와 빠뜨린 경우를 비교한다.

```python {exec="browser" session="integ"}
rng = np.random.default_rng(0)
A = np.array([[2.0, 1.0], [0.5, 1.5]])
corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ A.T   # (4, 2) 정사각형 꼭짓점의 상
lo, hi = corners.min(0), corners.max(0)
box = rng.uniform(lo, hi, size=(200_000, 2))        # (N, 2) 경계 상자 위 균등 표본
pre = np.linalg.solve(A, box.T).T                    # (N, 2) 역사상으로 되돌린 점
frac = np.mean(np.all((pre >= 0) & (pre <= 1), axis=1))   # 평행사변형 안의 비율
print(f"평행사변형 넓이 (MC) {frac * np.prod(hi - lo):.3f}   |det A| = {abs(np.linalg.det(A)):.3f}")

m = 400                                              # 극좌표 격자로 단위원의 넓이
r = (np.arange(m) + 0.5) / m                         # (m,)
dr, dth = 1 / m, 2 * np.pi / m
print(f"원의 넓이: r 포함 {np.sum(r) * m * dr * dth:.4f}   r 빠뜨림 {m * m * dr * dth:.4f}   pi = {np.pi:.4f}")
```

```output
평행사변형 넓이 (MC) 2.500   |det A| = 2.500
원의 넓이: r 포함 3.1416   r 빠뜨림 6.2832   pi = 3.1416
```

$r$을 빠뜨리면 $(r, \theta)$ 직사각형 $[0, 1] \times [0, 2\pi]$의 넓이 $2\pi$를 계산하게 된다. 좌표를 바꿨다는 사실만으로 넓이가 두 배로 틀린다.

:::note[분야 예시]
기후 데이터의 위도·경도 격자는 같은 각도 간격이어도 칸의 넓이가 다르다. 구면에서 넓이 요소는 $dA = R^2\cos\varphi\,d\varphi\,d\lambda$($\varphi$는 위도, $\lambda$는 경도)이므로 고위도 칸일수록 좁다. 전 지구 평균 기온을 구할 때 격자값을 단순 평균하면 극지방이 과대 대표된다. $\cos\varphi$로 가중한 평균이 변수변환 공식을 그대로 적용한 것이다. 같은 이유로 위경도 격자에서 학습하는 모델의 손실도 넓이 가중을 할지 결정해야 한다.
:::

## 가우시안 적분

정규분포의 밀도 앞에 붙는 $1/\sqrt{2\pi\sigma^2}$은 어디서 오는가. 정규화 상수를 직접 유도해 보면 변수변환이 왜 강력한지 보이고, 다변량 가우시안의 $\det(\boldsymbol{\Sigma})^{-1/2}$이 무엇을 뜻하는지도 이해된다.

**1차원.** $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$는 초등함수로 된 부정적분이 없다. 대신 $I^2$을 2차원 적분으로 쓰고 극좌표로 바꾼다.

$$
I^2 = \int\!\!\int e^{-(x^2 + y^2)/2}\,dx\,dy = \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta = 2\pi\Big[-e^{-r^2/2}\Big]_0^\infty = 2\pi.
$$

따라서 $I = \sqrt{2\pi}$다. 여기서 Jacobian $r$이 결정적이다. $r e^{-r^2/2}$는 $-e^{-r^2/2}$의 도함수라서 적분이 닫힌 형태로 풀린다. 이어서 $x = \mu + \sigma u$로 치환하면 $dx = \sigma\,du$이므로 $\int e^{-(x - \mu)^2/(2\sigma^2)}dx = \sigma\sqrt{2\pi}$이고, 정규분포의 밀도는

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\Big(-\frac{(x - \mu)^2}{2\sigma^2}\Big)
$$

가 된다.

**$d$차원.** 공분산 $\boldsymbol{\Sigma}$(양의 정부호)를 Cholesky 분해 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$하고 $\mathbf{x} = \mathbf{L}\mathbf{u}$로 바꾸면 $\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} = \lVert\mathbf{u}\rVert^2$이고 $d\mathbf{x} = \lvert\det\mathbf{L}\rvert\,d\mathbf{u} = \det(\boldsymbol{\Sigma})^{1/2}d\mathbf{u}$다. 남는 적분은 1차원 가우시안 적분 $d$개의 곱이다.

$$
\int_{\mathbb{R}^d}\exp\!\big(-\tfrac{1}{2}\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}\big)\,d\mathbf{x} = (2\pi)^{d/2}\det(\boldsymbol{\Sigma})^{1/2}.
$$

그래서 다변량 정규분포의 밀도 앞에는 $(2\pi)^{-d/2}\det(\boldsymbol{\Sigma})^{-1/2}$이 붙는다. $\det\boldsymbol{\Sigma}$는 분포가 차지하는 부피의 제곱이고, 넓게 퍼진 분포일수록 밀도의 높이가 낮아져야 전체 확률이 1로 유지된다. 다변량 가우시안의 성질은 11.4장에서 자세히 다룬다.

```python {exec="browser" session="integ"}
I1, _ = integrate.quad(lambda x: np.exp(-x**2 / 2), -np.inf, np.inf)
print(f"1차원: {I1:.6f}   sqrt(2 pi) = {np.sqrt(2 * np.pi):.6f}")

Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])           # (2, 2) 공분산
P = np.linalg.inv(Sigma)
g = np.linspace(-10, 10, 801)
dx = g[1] - g[0]
G = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)   # (801, 801, 2)
q = np.einsum("...i,ij,...j->...", G, P, G)               # (801, 801) 이차형식
I2 = np.exp(-0.5 * q).sum() * dx**2
print(f"2차원: {I2:.6f}   2 pi sqrt(det) = {2 * np.pi * np.sqrt(np.linalg.det(Sigma)):.6f}")
```

```output
1차원: 2.506628   sqrt(2 pi) = 2.506628
2차원: 7.327390   2 pi sqrt(det) = 7.327390
```

정규화 상수는 학습에서 사라지지 않는다. 분산을 학습하는 모델(12.3장의 이분산 회귀, 13장의 베이지안 모델)에서 음의 로그우도에 $\log\sigma$ 또는 $\tfrac{1}{2}\log\det\boldsymbol{\Sigma}$ 항이 남는 것이 이 정규화 상수 때문이다. 이 항이 없으면 모델은 분산을 무한히 키워 오차 항을 0으로 만들 수 있다.

## 확률밀도의 변환

측정값의 단위를 바꾸거나, 로그를 취하거나, 신경망으로 잠재변수를 데이터로 보낼 때마다 확률변수가 변환된다. 확률밀도는 단위 부피당 확률이므로, 부피가 늘어나는 곳에서는 밀도가 그만큼 낮아져야 한다.

**1차원 유도.** $X$의 밀도가 $p_X$이고 $Y = g(X)$, $g$는 증가하는 가역 함수라 하자. 누적분포함수로 쓰면

$$
P(Y \le y) = P\big(X \le g^{-1}(y)\big) \quad \Rightarrow \quad p_Y(y) = p_X\big(g^{-1}(y)\big)\,\frac{d g^{-1}(y)}{dy}
$$

이다. 감소 함수이면 부호가 바뀌므로 일반적으로는 절댓값을 쓴다. 역함수의 도함수는 $1/g'(x)$이므로 $p_Y(y) = p_X(x)/\lvert g'(x)\rvert$($x = g^{-1}(y)$)로도 쓴다.

**다변량.** $\mathbf{Y} = g(\mathbf{X})$이고 $g$가 가역이며 미분 가능하면, 변수변환 공식에서 바로

$$
p_Y(\mathbf{y}) = p_X(\mathbf{x})\,\big\lvert\det\mathbf{J}_g(\mathbf{x})\big\rvert^{-1}, \qquad \mathbf{x} = g^{-1}(\mathbf{y})
$$

를 얻는다. 임의의 영역 $\mathcal{B}$에 대해 $P(\mathbf{Y} \in \mathcal{B}) = P(\mathbf{X} \in g^{-1}(\mathcal{B}))$가 성립해야 하고, 오른쪽 적분을 $\mathbf{y}$로 변수변환하면 이 식이 나온다. 로그를 취하면

$$
\log p_Y(\mathbf{y}) = \log p_X(\mathbf{x}) - \log\big\lvert\det\mathbf{J}_g(\mathbf{x})\big\rvert.
$$

::quiz{id="density-transform-scale"}

**예: 로그정규분포.** $X \sim \mathcal{N}(0, 1)$, $Y = e^X$이면 $g^{-1}(y) = \log y$, $\lvert d g^{-1}/dy\rvert = 1/y$이므로 $p_Y(y) = p_X(\log y)/y$다. 표본의 구간별 비율과 공식의 적분을 비교하고, Jacobian 항 $1/y$를 빠뜨린 "밀도"의 전체 적분도 계산한다.

```python {exec="browser" session="integ"}
x = rng.normal(size=1_000_000)                       # X ~ N(0, 1)
y = np.exp(x)                                        # Y = exp(X), 로그정규분포
p_x = lambda t: np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
p_y = lambda s: p_x(np.log(s)) / s                   # |d g^{-1}/dy| = 1/y
p_wrong = lambda s: p_x(np.log(s))                   # Jacobian 항을 빠뜨림

edges = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
counts, _ = np.histogram(y, bins=edges)
for a, b, cnt in zip(edges[:-1], edges[1:], counts):
    exact, _ = integrate.quad(p_y, a, b)
    print(f"P({a:4.2f} < Y < {b:4.2f})  표본 {cnt / y.size:.4f}  공식 {exact:.4f}")
print(f"전체 적분: 올바른 밀도 {integrate.quad(p_y, 0, np.inf)[0]:.4f}"
      f"   Jacobian 누락 {integrate.quad(p_wrong, 0, np.inf)[0]:.4f}")
```

```output
P(0.25 < Y < 0.50)  표본 0.1616  공식 0.1613
P(0.50 < Y < 1.00)  표본 0.2556  공식 0.2559
P(1.00 < Y < 2.00)  표본 0.2565  공식 0.2559
P(2.00 < Y < 4.00)  표본 0.1613  공식 0.1613
전체 적분: 올바른 밀도 1.0000   Jacobian 누락 1.6487
```

Jacobian을 빠뜨린 함수의 전체 적분은 $\int p_X(\log y)\,dy = \int p_X(x)\,e^x\,dx = e^{1/2} \approx 1.6487$이다. 확률밀도가 아니다. 이 함수로 최대우도 학습을 하면 손실이 "확률"이 아닌 양을 키우는 방향으로 움직인다.

:::note[분야 예시]
생물학의 유전자 발현량이나 약물 농도처럼 양수이고 오른쪽 꼬리가 긴 측정값은 로그를 취해 분석하는 경우가 많다. 로그 공간에서 정규분포로 맞춘 모델의 우도를 원래 단위의 우도와 비교하려면(예: 로그 변환한 모델과 변환하지 않은 모델의 비교) Jacobian 항 $-\log y$를 더해야 한다. 이 항을 빼먹으면 두 모델의 우도가 서로 다른 단위의 밀도가 되어 비교가 의미를 잃는다.
:::

**normalizing flow.** 이 공식은 생성모델의 한 갈래인 normalizing flow의 출발점이다. 다루기 쉬운 분포(보통 표준정규) $\mathbf{z} \sim p_Z$에서 시작해 학습 가능한 가역 변환 $\mathbf{x} = f_\theta(\mathbf{z})$로 데이터 분포를 표현한다. 데이터 $\mathbf{x}$의 로그우도는

$$
\log p_\theta(\mathbf{x}) = \log p_Z\big(f_\theta^{-1}(\mathbf{x})\big) - \log\big\lvert\det\mathbf{J}_{f_\theta}(\mathbf{z})\big\rvert
$$

이고, 이것을 최대화하는 것이 학습이다. $f_\theta = f_K \circ \cdots \circ f_1$처럼 여러 층을 합성하면 chain rule에 의해 Jacobian은 층별 Jacobian의 곱이고, 곱의 행렬식은 행렬식의 곱이므로 로그 행렬식은 층별로 더해진다.

$$
\log p_\theta(\mathbf{x}) = \log p_Z(\mathbf{z}_0) - \sum_{k=1}^{K}\log\big\lvert\det\mathbf{J}_{f_k}(\mathbf{z}_{k-1})\big\rvert, \qquad \mathbf{z}_k = f_k(\mathbf{z}_{k-1}).
$$

설계의 핵심 제약은 계산량이다. $d \times d$ 행렬의 행렬식은 일반적으로 $O(d^3)$ 연산이 들어 고차원 데이터에서 매 step 계산할 수 없다. 그래서 flow의 층은 Jacobian이 **삼각행렬**이 되도록 설계하는 경우가 많다. 삼각행렬의 행렬식은 대각 성분의 곱이므로 $O(d)$로 계산된다. 실습에서 쓰는 변환이 그 가장 작은 예이고, 구체적인 구조는 19.4장에서 다룬다.

## 실습

2차원 분포에 가역 변환을 적용해 밀도 변환 공식을 Monte Carlo 표본으로 검증한다. 코드는 `labs/calculus-and-autodiff/integration/density_transform.py`다. $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$에

$$
y_1 = z_1, \qquad y_2 = z_2\,e^{a z_1} + b z_1^2 \qquad (a = 0.5,\ b = 0.6)
$$

을 적용해 바나나 모양 분포를 만든다. Jacobian은 하삼각행렬 $\begin{bmatrix} 1 & 0 \\ \ast & e^{a z_1} \end{bmatrix}$이므로 $\log\lvert\det\mathbf{J}\rvert = a z_1$이고, 역변환은 $z_1 = y_1$, $z_2 = (y_2 - b y_1^2)e^{-a y_1}$로 닫힌 형태다. 표본 200만 개로 세 가지를 확인한다.

1. 다섯 개 상자에 떨어진 표본의 비율과, 공식으로 얻은 밀도를 그 상자 위에서 적분한 값(Jacobian 누락 버전 포함)
2. 넓은 격자 위에서 밀도의 전체 질량
3. 표본의 평균 로그우도. normalizing flow가 최대화하는 바로 그 양이다.

집필 시점의 실행 결과는 다음과 같다. 괄호 안은 표본 비율의 표준오차의 두 배다.

```output
[1] 상자 확률: 표본 비율 vs 변환 공식의 적분
  [-1.0, 0.0]x[-1.0, 1.0]  표본 0.2613 (±0.0006)  공식 0.2608  Jacobian 누락 0.2081
  [ 0.0, 1.0]x[ 0.0, 2.0]  표본 0.1598 (±0.0005)  공식 0.1599  Jacobian 누락 0.2029
  [ 1.0, 2.0]x[ 0.5, 3.0]  표본 0.0594 (±0.0003)  공식 0.0594  Jacobian 누락 0.1187
  [-2.0,-1.0]x[ 0.0, 2.0]  표본 0.1123 (±0.0004)  공식 0.1126  Jacobian 누락 0.0579
  [-0.5, 0.5]x[-0.5, 0.5]  표본 0.1476 (±0.0005)  공식 0.1473  Jacobian 누락 0.1461
[2] 전체 질량 ([-6, 6] x [-15, 40] 격자)
  올바른 밀도 1.0000   Jacobian 누락 1.1331
[3] 평균 로그우도 (normalizing flow의 학습 목표와 같은 양)
  공식으로 계산한 mean log p_Y(y)   -2.8378
  log p_Z(z) - log|det J| 의 평균   -2.8378
  이론값 -(1 + log 2 pi)            -2.8379
그림 저장: labs/calculus-and-autodiff/integration/outputs/density_transform.png
```

결과를 읽는 법은 세 가지다.

- **공식은 표본과 오차 범위 안에서 일치한다.** Jacobian을 빠뜨린 버전은 $z_1 > 0$ 쪽(공간이 늘어나는 쪽) 상자의 확률을 과대평가하고(세 번째 상자에서 약 두 배), $z_1 < 0$ 쪽은 과소평가한다(네 번째 상자에서 약 절반).
- **Jacobian을 빠뜨린 함수의 질량은 $\mathbb{E}[e^{a Z_1}] = e^{a^2/2} = e^{0.125} \approx 1.1331$이다.** 1이 아니므로 밀도가 아니다. 원점 근처 상자처럼 $\lvert\det\mathbf{J}\rvert \approx 1$인 곳에서는 차이가 작아 보여, 한두 곳만 확인하면 오류를 놓칠 수 있다.
- **평균 로그우도의 이론값.** $\mathbb{E}[\log p_Z(\mathbf{Z})] = -(1 + \log 2\pi)$이고 $\mathbb{E}[a Z_1] = 0$이므로 $\mathbb{E}[\log p_Y(\mathbf{Y})] = -(1 + \log 2\pi)$다. 가역 변환은 정보를 잃지 않지만 로그우도의 값은 바꿀 수 있다. 이 변환에서는 로그 행렬식의 평균이 0이라 값이 우연히 같다.

이어서 연습문제를 푼다.

::exercise{id="affine-flow-logpdf"}

## 실패 사례

- **밀도 변환에서 Jacobian 항을 빠뜨리는 것.** 결과는 적분해서 1이 되지 않는 함수이고, 그것으로 계산한 우도·사후분포·모델 비교는 모두 의미를 잃는다. 변환한 밀도를 쓰기 전에 격자 적분이나 표본 비율로 전체 질량과 몇 개 영역의 확률을 확인한다.
- **Jacobian의 방향을 뒤집는 것.** $p_Y(\mathbf{y}) = p_X(\mathbf{x})\lvert\det\mathbf{J}_g(\mathbf{x})\rvert$처럼 정변환의 행렬식을 곱하면 부피가 늘어나는 곳에서 밀도가 오히려 커진다. 정변환 $g$의 행렬식으로 나누거나, 역변환 $g^{-1}$의 행렬식을 곱한다. 어느 쪽인지 헷갈리면 $Y = 3X$ 같은 1차원 예로 확인한다.
- **가역이 아닌 변환에 공식을 쓰는 것.** $Y = X^2$처럼 여러 $x$가 같은 $y$로 가면 각 가지(branch)의 기여를 더해야 한다. 차원을 줄이는 변환(예: $\mathbb{R}^3 \to \mathbb{R}^2$ 사영)에는 이 공식을 쓸 수 없고 주변화 적분이 필요하다.
- **위경도 격자처럼 칸 넓이가 다른 좌표에서 단순 평균을 내는 것.** 좌표의 부피 요소를 가중치로 써야 공간 평균이 된다.

## 통과 기준

- 확률밀도 변환 공식을 변수변환 공식에서 유도하고, normalizing flow의 로그우도가 층별 로그 행렬식의 합으로 쓰이는 이유를 설명한다.
- 2차원 가역 변환 하나를 골라 밀도 변환 공식을 구현하고, Monte Carlo 표본의 영역 비율과 공식의 적분이 표준오차 범위 안에서 일치함을 확인한다.
- 연습문제 `affine-flow-logpdf`의 테스트를 통과한다.
