---
id: linear-algebra/vector-spaces
title: "벡터공간과 선형사상"
title_en: "Vector Spaces and Linear Maps"
tier: core
status: draft
prerequisites: []
---

# 벡터공간과 선형사상

NumPy 배열 `[1.0, 2.0, 3.0]`은 무엇을 나타낼까. 공간의 한 점일 수도, 다항식 $1 + 2x + 3x^2$의 계수일 수도, 세 시각에 측정한 전압일 수도 있다. 숫자는 같아도 대상은 다르다. 선형대수가 연구에서 강력한 이유는 이 서로 다른 대상들이 **더하고 수를 곱하는** 같은 구조를 공유하고, 그 구조만으로 많은 것을 계산할 수 있기 때문이다. 이 장은 그 구조인 벡터공간과, 구조를 보존하는 함수인 선형사상을 정의한다.

이 장의 핵심 메시지는 두 가지다. 첫째, 벡터는 숫자 배열이 아니라 **더하고 늘일 수 있는 대상**이다. 함수와 신호도 벡터다. 둘째, 숫자 배열은 벡터를 **basis에 대해 적은 좌표**이고, 행렬은 선형사상을 basis에 대해 적은 표이다. basis를 바꾸면 숫자는 바뀌지만 대상은 그대로다.

## 학습 목표

- 벡터공간의 공리와 예를 설명한다
- span·독립·basis·dimension을 정의한다
- 선형사상과 행렬 표현을 연결한다

## 집합·함수·벡터공간

연구 데이터의 상당수는 "벡터처럼" 다룰 수 있다. 두 스펙트럼의 평균을 내고, 신호에서 기준선(baseline)을 빼고, 여러 해를 섞어 새 해를 만든다. 이런 조작이 의미가 있으려면 대상이 어떤 규칙을 따라야 하는지 정확히 알아야 한다. 그 규칙을 모은 것이 벡터공간의 정의다.

**정의.** 실수 위의 **벡터공간**(vector space) $V$는 덧셈 $\mathbf{u} + \mathbf{v}$와 스칼라배 $\alpha\mathbf{v}$가 정의되고, 그 결과가 다시 $V$에 속하며(닫혀 있음), 모든 $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$와 $\alpha, \beta \in \mathbb{R}$에 대해 다음 여덟 공리를 만족하는 집합이다.

| 덧셈 | 스칼라배 |
|---|---|
| $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ | $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$ |
| $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ | $1\mathbf{v} = \mathbf{v}$ |
| 영벡터 $\mathbf{0}$가 있어 $\mathbf{v} + \mathbf{0} = \mathbf{v}$ | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ |
| 각 $\mathbf{v}$에 $-\mathbf{v}$가 있어 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ | $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$ |

스칼라를 복소수로 바꾸면 복소 벡터공간이 된다(양자역학의 상태, Fourier 계수). 이 장은 실수만 다룬다.

공리는 낯선 대상을 벡터로 다뤄도 되는지 확인하는 체크리스트다. 몇 가지 예를 본다.

- $\mathbb{R}^n$: 성분별 덧셈과 스칼라배. 가장 익숙한 예다.
- **다항식** $\mathcal{P}_n$: 차수 $n$ 이하인 실계수 다항식 전체. $(1 + 2x) + (3 - x^2) = 4 + 2x - x^2$처럼 더해도 차수 $n$ 이하로 남는다.
- **함수** $C[a, b]$: 구간 $[a, b]$에서 연속인 함수 전체. 덧셈은 점별로 $(f + g)(t) = f(t) + g(t)$, 영벡터는 항상 0인 함수다.
- **신호**: 길이 $T$인 시계열 $(s_1, \dots, s_T)$. 모양은 $\mathbb{R}^T$이지만, 성분의 순서가 시간이라는 의미를 갖는다.
- **행렬** $\mathbb{R}^{m \times n}$: 같은 크기의 행렬끼리 더하고 수를 곱한다. 신경망의 가중치 업데이트 $\mathbf{W} - \eta\nabla L$이 이 공간 안의 연산이다.

함수나 다항식이 벡터라는 말은 비유가 아니다. 다항식 $\mathcal{P}_2$의 원소를 계수 벡터로 적으면, 함수로 더하고 곱한 결과와 계수 벡터로 더하고 곱한 결과가 정확히 일치한다.

```python {exec="browser" session="vs"}
import numpy as np
from numpy.polynomial import polynomial as P

p = np.array([1.0, 2.0, 3.0])     # 1 + 2x + 3x^2  (상수항부터)
q = np.array([-1.0, 0.0, 1.0])    # -1 + x^2

x = np.linspace(-1, 1, 5)         # (5,) 함수를 확인할 점들
lhs = P.polyval(x, p) + 2 * P.polyval(x, q)   # 함수로 더하고 곱한다
rhs = P.polyval(x, p + 2 * q)                 # 계수 벡터로 더하고 곱한다
print("p + 2q 의 계수 :", p + 2 * q)
print("두 계산이 같은가:", np.allclose(lhs, rhs))
```

```output
p + 2q 의 계수 : [-1.  2.  5.]
두 계산이 같은가: True
```

**부분공간.** 벡터공간 $V$의 부분집합 $U$가 그 자체로 벡터공간이면 **부분공간**(subspace)이라 한다. 공리는 $V$에서 물려받으므로, 확인할 것은 세 가지뿐이다.

$$
\mathbf{0} \in U, \qquad \mathbf{u}, \mathbf{v} \in U \Rightarrow \mathbf{u} + \mathbf{v} \in U, \qquad \mathbf{u} \in U \Rightarrow \alpha\mathbf{u} \in U.
$$

$\mathbb{R}^3$에서 $\{\mathbf{x} : x_1 + x_2 + x_3 = 0\}$은 원점을 지나는 평면이고 부분공간이다. 반면 $\{\mathbf{x} : x_1 + x_2 + x_3 = 1\}$은 원점을 지나지 않으므로 부분공간이 아니다. 이런 "평행이동한 부분공간"은 7.2장의 아핀 집합이다. 선형 방정식의 해집합은 우변이 0일 때(동차) 부분공간이고, 우변이 0이 아니면(비동차) 아핀 집합이 된다.

::quiz{id="subspace-test"}

:::note[분야 예시]
감쇠 진동자의 운동 방정식 $\ddot{x} + 2\zeta\omega\dot{x} + \omega^2 x = 0$을 생각한다. $x_1(t)$와 $x_2(t)$가 해이면 $\alpha x_1 + \beta x_2$도 해다. 방정식의 좌변이 $x$에 대해 선형이기 때문이다. 따라서 해 전체는 함수공간의 부분공간이고, 물리에서 **중첩 원리**(superposition principle)라 부르는 성질이 바로 이것이다. 외력 항이 붙은 $\ddot{x} + 2\zeta\omega\dot{x} + \omega^2 x = F$의 해집합은 부분공간이 아니라, 특수해 하나만큼 평행이동한 아핀 집합이다.
:::

벡터공간의 구조가 **없는** 대상도 연구에서 흔하다. 농도, 절대온도, 확률처럼 음수가 될 수 없는 양은 $-1$을 곱하면 의미를 잃는다. 확률 벡터(성분이 0 이상이고 합이 1)는 두 개를 더하면 합이 2가 된다. 이런 대상을 아무 생각 없이 벡터처럼 더하고 빼면, 모델이 음의 농도나 합이 1이 아닌 확률을 내놓는다. 그런 대상은 변환(로그, softmax 등)을 거쳐 벡터공간으로 옮긴 뒤 다루는 경우가 많다.

## span·선형독립·basis·dimension

벡터공간의 원소는 무한히 많지만, 몇 개의 벡터로 전체를 만들어 낼 수 있다면 그 몇 개만 기억하면 된다. 그 "몇 개"를 고르는 규칙이 이 절의 내용이다.

**정의.** 벡터 $\mathbf{v}_1, \dots, \mathbf{v}_k$의 **선형결합**(linear combination)은 $\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k$ 꼴의 벡터이고, 가능한 모든 선형결합의 집합을 **span**이라 한다. span은 항상 부분공간이다.

벡터들이 **선형독립**(linearly independent)이라는 것은

$$
\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k = \mathbf{0} \;\Longrightarrow\; \alpha_1 = \cdots = \alpha_k = 0
$$

이라는 뜻이다. 하나라도 0이 아닌 계수로 영벡터를 만들 수 있으면 **선형종속**이고, 그때는 어떤 벡터가 나머지의 선형결합으로 표현된다. 즉 그 벡터는 span에 새로운 방향을 보태지 않는다.

$V$를 span하는 선형독립인 벡터 모음을 $V$의 **basis**라 한다. 한 벡터공간의 basis는 무수히 많지만, 모든 basis의 원소 수는 같다. 이 수가 **차원**(dimension) $\dim V$이다. $\dim \mathbb{R}^n = n$이고, $\mathcal{P}_n$은 $1, x, \dots, x^n$이 basis이므로 $\dim \mathcal{P}_n = n + 1$이다. 연속함수 전체 $C[a, b]$처럼 유한 개의 basis가 없는 공간은 무한차원이다(7.9장).

::quiz{id="independence-check"}

**좌표.** basis $\mathcal{B} = (\mathbf{b}_1, \dots, \mathbf{b}_n)$가 정해지면 모든 벡터는 단 한 가지 방법으로

$$
\mathbf{v} = c_1\mathbf{b}_1 + \cdots + c_n\mathbf{b}_n
$$

으로 쓸 수 있다. 존재는 span에서, 유일성은 선형독립에서 나온다(두 표현을 빼면 영벡터의 선형결합이 되므로 계수 차이가 모두 0이다). 이 계수의 배열 $[\mathbf{v}]_{\mathcal{B}} = (c_1, \dots, c_n)$을 $\mathbf{v}$의 **좌표**(coordinates)라 한다. 좌표는 벡터와 basis가 함께 정한다. 같은 다항식 $1 + 2x + 3x^2$도 단항식 basis $(1, x, x^2)$에서는 좌표가 $(1, 2, 3)$이고, basis $(1,\ 1 + x,\ 1 + x + x^2)$에서는 다르다. 새 basis 벡터의 계수를 열로 쌓은 행렬 $\mathbf{B}$에 대해 $\mathbf{B}\mathbf{c} = \mathbf{p}$를 풀면 된다.

```python {exec="browser" session="vs"}
B = np.array([[1.0, 1.0, 1.0],    # 열: 1, 1+x, 1+x+x^2 의 계수 벡터
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]])   # (3, 3)
c = np.linalg.solve(B, p)         # B c = p 를 푼다
print("단항식 basis 좌표:", p)
print("새 basis 좌표    :", c)
print("되돌리기         :", B @ c)
```

```output
단항식 basis 좌표: [1. 2. 3.]
새 basis 좌표    : [-1. -1.  3.]
되돌리기         : [1. 2. 3.]
```

$-1 \cdot 1 - 1 \cdot (1 + x) + 3(1 + x + x^2) = 1 + 2x + 3x^2$이다. 숫자 배열 $(-1, -1, 3)$과 $(1, 2, 3)$은 다르지만 가리키는 다항식은 하나다.

차원은 "자유롭게 고를 수 있는 수의 개수"다. 감쇠 진동자의 해공간이 2차원이라는 것은, 초기 위치와 초기 속도 두 수를 정하면 해가 하나로 정해진다는 뜻이다. 데이터 분석에서 "이 데이터는 사실상 3차원이다"라는 말도 같은 뜻이다. 관측은 수백 개의 수로 이루어져 있어도, 그 변화를 만드는 독립적인 방향은 몇 개뿐이라는 주장이다. 이 주장을 데이터에서 확인하는 도구는 7.3장(rank)과 7.6장(SVD·PCA)에서 다룬다.

:::note[분야 예시]
분광 측정에서 혼합물의 흡광 스펙트럼은 흔히 순수 성분 스펙트럼의 선형결합으로 모델링한다(Beer–Lambert 법칙이 성립하는 범위에서, 계수는 농도에 비례). 파장 500개에서 측정한 스펙트럼은 $\mathbb{R}^{500}$의 벡터이지만, 성분이 3개라면 모든 혼합물 스펙트럼은 순수 스펙트럼 3개가 span하는 3차원 부분공간 안에 있다. 측정 스펙트럼이 그 부분공간에서 크게 벗어나면, 모르는 성분이 섞였거나 선형 혼합 가정이 깨졌다는 신호다.
:::

## 선형사상

벡터공간 사이의 함수 중에서 덧셈과 스칼라배를 보존하는 것이 선형사상이다. 미분, 적분, 샘플링, 필터링, 행렬 곱이 모두 여기에 속한다. 선형사상은 basis 벡터에서의 값만 알면 전체가 정해지므로, 무한히 많은 입력에 대한 행동을 유한 개의 수로 적을 수 있다.

**정의.** 함수 $T: V \to W$가 모든 $\mathbf{x}, \mathbf{y} \in V$와 $\alpha \in \mathbb{R}$에 대해 다음 두 조건을 만족하면 **선형사상**(linear map)이라 한다.

$$
T(\mathbf{x} + \mathbf{y}) = T(\mathbf{x}) + T(\mathbf{y}), \qquad T(\alpha\mathbf{x}) = \alpha T(\mathbf{x}).
$$

두 번째 조건에 $\alpha = 0$을 넣으면 $T(\mathbf{0}) = \mathbf{0}$이다. 원점을 원점으로 보내지 않는 함수는 선형이 아니다. 두 조건을 합치면 $T(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha T(\mathbf{x}) + \beta T(\mathbf{y})$, 즉 **선형사상은 선형결합을 보존한다.**

예와 반례를 나란히 본다.

| 선형사상 | 선형이 아닌 함수 |
|---|---|
| 미분 $p \mapsto p'$ ($\mathcal{P}_n \to \mathcal{P}_{n-1}$) | 평행이동 $\mathbf{x} \mapsto \mathbf{x} + \mathbf{b}$, $\mathbf{b} \neq \mathbf{0}$ (7.2장의 아핀 사상) |
| 정적분 $f \mapsto \int_0^1 f(t)\,dt$ ($C[0,1] \to \mathbb{R}$) | 노름 $\mathbf{x} \mapsto \lVert\mathbf{x}\rVert$ |
| 한 점에서의 값 $f \mapsto f(t_0)$ | 성분별 제곱 $\mathbf{x} \mapsto \mathbf{x} \odot \mathbf{x}$ |
| 행렬 곱 $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ | ReLU $\mathbf{x} \mapsto \max(\mathbf{0}, \mathbf{x})$ |

::quiz{id="which-maps-linear"}

**합성.** $T: U \to V$와 $S: V \to W$가 선형이면 합성 $S \circ T$도 선형이다. $S(T(\alpha\mathbf{x} + \beta\mathbf{y})) = S(\alpha T\mathbf{x} + \beta T\mathbf{y}) = \alpha S(T\mathbf{x}) + \beta S(T\mathbf{y})$이기 때문이다. 7.2장에서 활성화 없는 층을 쌓아도 하나의 아핀 사상이 되는 이유가 이 성질에서 나온다.

**핵과 상.** 선형사상 $T: V \to W$에는 두 부분공간이 딸려 있다.

$$
\ker T = \{\mathbf{v} \in V : T\mathbf{v} = \mathbf{0}\}, \qquad \operatorname{im} T = \{T\mathbf{v} : \mathbf{v} \in V\}.
$$

**핵**(kernel)은 $T$가 0으로 지워 버리는 입력, **상**(image)은 $T$가 만들어 낼 수 있는 출력이다. 둘 다 부분공간임은 정의에서 바로 확인된다. 핵은 정보 손실을 잰다. $T\mathbf{x} = T\mathbf{y}$이면 $\mathbf{x} - \mathbf{y} \in \ker T$이므로, 핵이 $\{\mathbf{0}\}$일 때에만 $T$는 서로 다른 입력을 구분한다(단사). 미분 $D: \mathcal{P}_3 \to \mathcal{P}_3$의 핵은 상수 함수 전체(1차원)이고, 상은 차수 2 이하 다항식 $\mathcal{P}_2$(3차원)이다. 미분하면 상수항 정보를 잃고, 그래서 적분에는 적분상수가 붙는다. $1 + 3 = 4 = \dim \mathcal{P}_3$인 것은 우연이 아니다. 유한차원에서 항상

$$
\dim \ker T + \dim \operatorname{im} T = \dim V
$$

가 성립하고(rank–nullity 정리), 7.3장에서 행렬의 언어로 다시 다룬다.

연구에서 선형성은 강력한 가정이다. 측정 장치가 선형이면 입력 두 개에 대한 응답만으로 그 합에 대한 응답을 예측할 수 있고, 선형 PDE는 해를 기본해의 중첩으로 만든다. 반대로 시스템이 선형이 아닌데 선형이라고 가정하면, 개별 조건에서는 잘 맞던 모델이 조건을 섞을 때 틀린다. 선형성은 데이터로 검증해야 하는 가정이다. 위의 두 조건을 측정값에 대고 직접 확인할 수 있다.

## 행렬 표현

선형사상은 basis에서의 값으로 완전히 정해진다. $\mathbf{v} = \sum_j c_j\mathbf{b}_j$이면 $T\mathbf{v} = \sum_j c_j T\mathbf{b}_j$이기 때문이다. 그래서 각 $T\mathbf{b}_j$를 $W$의 basis로 적어 두면 $T$를 수의 표, 즉 행렬로 저장할 수 있다.

**정의.** $V$의 basis $\mathcal{B} = (\mathbf{b}_1, \dots, \mathbf{b}_n)$, $W$의 basis $\mathcal{C} = (\mathbf{c}_1, \dots, \mathbf{c}_m)$에 대해 선형사상 $T: V \to W$의 **행렬 표현**은 $j$번째 열이 $[T\mathbf{b}_j]_{\mathcal{C}}$인 $m \times n$ 행렬 $[T]_{\mathcal{C} \leftarrow \mathcal{B}}$이다. 그러면

$$
[T\mathbf{v}]_{\mathcal{C}} = [T]_{\mathcal{C} \leftarrow \mathcal{B}}\,[\mathbf{v}]_{\mathcal{B}}
$$

가 성립한다. 행렬 곱 $\mathbf{A}\mathbf{c}$는 $\mathbf{A}$의 열을 $\mathbf{c}$의 성분으로 섞은 선형결합이라는 점을 떠올리면, 이 식은 $T\mathbf{v} = \sum_j c_j T\mathbf{b}_j$를 좌표로 옮겨 적은 것일 뿐이다. 외울 문장은 하나다. **행렬의 $j$번째 열은 $j$번째 basis 벡터가 가는 곳이다.**

아래 위젯에서 표준 basis 벡터 $\mathbf{e}_1, \mathbf{e}_2$의 끝점을 끌어 본다. 행렬의 두 열이 끌어 놓은 위치와 같다는 것, 격자 전체가 그 두 벡터로 정해진다는 것을 확인한다.

::widget{name="linear-map-2d"}

**미분 연산자의 행렬.** 미분 $D: \mathcal{P}_3 \to \mathcal{P}_3$를 단항식 basis $(1, x, x^2, x^3)$로 표현한다. $D(x^j) = j x^{j-1}$이므로 $j$번째 열(0부터 센다)은 $(j-1)$번째 성분만 $j$이다.

```python {exec="browser" session="vs"}
n = 3                                   # P_3: 차수 3 이하 다항식, 차원 4
D = np.diag(np.arange(1.0, n + 1), k=1) # (4, 4) j번째 열 = d/dx x^j 의 좌표
print(D)
r = np.array([1.0, 2.0, 3.0, 4.0])      # 1 + 2x + 3x^2 + 4x^3
print("D @ r     :", D @ r)
print("polyder   :", P.polyder(r))
print("D^4 = 0 ? :", np.allclose(np.linalg.matrix_power(D, 4), 0))
```

```output
[[0. 1. 0. 0.]
 [0. 0. 2. 0.]
 [0. 0. 0. 3.]
 [0. 0. 0. 0.]]
D @ r     : [ 2.  6. 12.  0.]
polyder   : [ 2.  6. 12.]
D^4 = 0 ? : True
```

행렬 곱 결과 $(2, 6, 12, 0)$은 $2 + 6x + 12x^2$, 즉 $(1 + 2x + 3x^2 + 4x^3)'$이다. `polyder`는 결과를 $\mathcal{P}_2$의 좌표로 돌려주므로 길이가 하나 짧다. 같은 다항식을 다른 공간의 좌표로 적은 차이다. 마지막 줄은 **합성이 행렬 곱**이라는 사실을 보여 준다. $D^4$는 네 번 미분하는 사상이고, 3차 이하 다항식을 네 번 미분하면 0이므로 행렬도 영행렬이다. 일반적으로 $[S \circ T] = [S][T]$이다. 행렬 곱의 정의가 그렇게 생긴 이유가 바로 이것이다.

**기저 변환.** 같은 선형사상도 basis를 바꾸면 행렬이 달라진다. 새 basis 벡터들을 옛 basis의 좌표로 적어 열로 쌓은 가역행렬을 $\mathbf{S}$라 하면, 한 벡터의 두 좌표는 $[\mathbf{v}]_{\text{old}} = \mathbf{S}[\mathbf{v}]_{\text{new}}$로 이어진다. $V \to V$인 사상 $T$에 대해서는 "새 좌표 → 옛 좌표로 바꾸고($\mathbf{S}$), 옛 행렬을 적용하고($\mathbf{A}$), 다시 새 좌표로 돌아온다($\mathbf{S}^{-1}$)"를 차례로 하면 된다.

$$
[T]_{\text{new}} = \mathbf{S}^{-1}\,[T]_{\text{old}}\,\mathbf{S}.
$$

$\mathbf{A}' = \mathbf{S}^{-1}\mathbf{A}\mathbf{S}$ 꼴로 이어진 두 행렬을 **닮음**(similar)이라 한다. 닮은 행렬은 같은 선형사상의 두 표현이므로, basis에 의존하지 않는 양을 공유한다. trace, 행렬식, rank, 고윳값(7.5장)이 그렇다.

예로 $\mathcal{P}_2$ 위의 사상 $T(p) = x\,p'(x)$를 본다. $T(x^j) = j x^j$이므로 단항식 basis에서는 대각행렬 $\operatorname{diag}(0, 1, 2)$이다. 이 사상을 Legendre 다항식 basis $(1,\ x,\ (3x^2 - 1)/2)$로 옮긴다.

```python {exec="browser" session="vs"}
T = np.diag([0.0, 1.0, 2.0])            # x d/dx 를 P_2 의 단항식 basis로
S = np.array([[1.0, 0.0, -0.5],         # 열: Legendre 다항식 1, x, (3x^2-1)/2
              [0.0, 1.0,  0.0],
              [0.0, 0.0,  1.5]])        # (3, 3)
T_new = np.linalg.solve(S, T @ S)       # S^{-1} T S
print(T_new)
v_new = np.array([2.0, -1.0, 1.0])      # 새 basis 좌표로 준 다항식
print("같은 결과?", np.allclose(S @ (T_new @ v_new), T @ (S @ v_new)))
print("trace:", np.trace(T), np.trace(T_new))
```

```output
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 2.]]
같은 결과? True
trace: 3.0 3.0
```

새 행렬의 셋째 열 $(1, 0, 2)$는 $T\big((3x^2 - 1)/2\big) = 3x^2 = 1 + 2 \cdot \tfrac{3x^2 - 1}{2}$을 새 basis로 적은 좌표다. 단항식 basis에서 대각이던 행렬이 Legendre basis에서는 대각이 아니다. 그래도 사상은 같고, trace는 둘 다 3이다. 코드에서 $\mathbf{S}^{-1}$을 직접 구하지 않고 `np.linalg.solve(S, T @ S)`로 $\mathbf{S}\mathbf{X} = \mathbf{T}\mathbf{S}$를 푼 것은, 역행렬을 만들기보다 연립방정식을 푸는 편이 수치적으로 정확하고 빠르기 때문이다(9.3장).

연구에서 basis 선택은 계산의 난이도를 바꾼다. 같은 사상이 어떤 basis에서는 대각행렬이 되어 성분별 곱셈으로 끝나고, 다른 basis에서는 꽉 찬 행렬이 된다. 주기 함수 위의 미분 연산자를 복소 지수함수 $e^{ikx}$로 이루어진 Fourier basis로 표현하면 대각행렬이 되어 PDE를 성분별로 풀 수 있고, 대칭행렬은 고유벡터 basis에서 대각이 된다(7.5장). "좋은 basis를 찾는 것"은 선형대수 뒷부분 전체의 주제다.

:::note[분야 예시]
결정학에서 원자의 위치는 흔히 단위 격자 벡터 $\mathbf{a}, \mathbf{b}, \mathbf{c}$를 basis로 한 분수 좌표로 기록한다. 단사정계처럼 격자 벡터가 서로 직각이 아니면, 같은 원자 위치도 분수 좌표와 직교 좌표(Å 단위)의 숫자가 다르다. 두 좌표는 격자 벡터를 열로 쌓은 행렬로 이어진다. 원자 사이 거리를 분수 좌표 그대로 계산하면 틀린 값이 나온다. 좌표를 같은 basis로 맞춘 뒤 계산해야 한다.
:::

## 실습

같은 선형사상을 두 basis에서 행렬로 표현하고, 기저 변환 공식으로 서로 변환되는지 확인한다. 다음 순서로 진행한다.

1. 미분 연산자 $D$를 $\mathcal{P}_n$의 단항식 basis에서 행렬로 만든다. 무작위 계수의 다항식에 적용해 `numpy.polynomial.polynomial.polyder`와 결과를 비교한다.
2. 핵과 상의 차원을 행렬에서 읽는다. $D$의 핵이 상수 함수뿐임을 확인한다.
3. 다른 basis(예: Legendre 다항식, 또는 $(1, 1 + x, 1 + x + x^2, \dots)$)를 열로 쌓은 행렬 $\mathbf{S}$를 만들고, $\mathbf{S}^{-1}\mathbf{D}\mathbf{S}$가 새 basis에서 직접 만든 행렬과 같은지 확인한다. trace와 행렬식이 바뀌지 않는지도 본다.

두 연습문제가 1번과 3번을 자동 채점한다.

::exercise{id="derivative-matrix"}

::exercise{id="change-of-basis"}

## 실패 사례

- **벡터를 숫자 배열로만 생각하는 것.** 함수·신호·다항식도 벡터다. 이 관점이 없으면 "함수를 basis 함수의 계수로 표현한다"는 아이디어(Fourier 급수, 스펙트럼 방법, 7.9장의 함수공간과 neural operator)를 받아들이기 어렵다. 반대로 숫자 배열이라고 해서 모두 벡터공간의 원소인 것도 아니다. 확률 벡터나 농도처럼 더하거나 음수를 곱하면 의미가 깨지는 배열을 그대로 선형결합하면, 모델이 물리적으로 불가능한 값을 낸다.
- **좌표와 벡터 자체를 혼동하는 것.** 배열 `[1, 2, 3]`만 보고는 어떤 basis에서 적은 좌표인지 알 수 없다. 서로 다른 basis(단위계, 정규화 방식, 격자 좌표계)로 적은 배열을 성분별로 비교하거나 더하면 틀린 결과가 나온다. 데이터를 저장할 때는 숫자와 함께 basis(좌표계·단위·정규화 방식)를 기록한다.
- **기저 변환 공식의 방향을 뒤집는 것.** $\mathbf{S}^{-1}\mathbf{A}\mathbf{S}$와 $\mathbf{S}\mathbf{A}\mathbf{S}^{-1}$은 일반적으로 다르다. $\mathbf{S}$의 열이 무엇인지(새 basis를 옛 좌표로 적은 것인지, 그 반대인지)를 먼저 정하고, 벡터 하나에 대해 양쪽 계산이 같은지 확인한다.

## 통과 기준

- 다항식 공간 $\mathcal{P}_n$이나 함수공간 $C[a, b]$를 예로 들어, 덧셈·스칼라배에 대한 닫힘과 영벡터의 존재를 확인하고 부분공간인 예와 아닌 예를 하나씩 든다.
- 미분 연산자를 $\mathcal{P}_n$의 행렬로 표현하고, 그 행렬의 핵과 상의 차원을 설명한다(연습문제 `derivative-matrix`).
- 같은 선형사상의 두 행렬 표현이 $\mathbf{S}^{-1}\mathbf{A}\mathbf{S}$로 이어짐을 계산으로 확인한다(연습문제 `change-of-basis`).
