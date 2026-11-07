---
id: linear-algebra/symmetry-and-groups
title: "대칭과 군 표현"
title_en: "Symmetry and Group Representations"
tier: advanced
status: draft
prerequisites: [linear-algebra/eigen-and-symmetric]
---

# 대칭과 군 표현

물 분자를 돌려도 에너지는 같다. 실험실을 옮겨도 물리 법칙은 같다. 원자에 붙인 번호를 바꿔도 분자는 같은 분자다. 과학 데이터에는 이런 **대칭**(symmetry)이 가득하고, 대칭은 학습에 두 가지 정보를 준다. 하나는 "이 변환을 해도 답이 같다(또는 정해진 방식으로 바뀐다)"는 제약이고, 다른 하나는 "그러니 이 방향의 변화는 따로 배울 필요가 없다"는 데이터 절약이다.

대칭을 모델에 제대로 넣으려면 먼저 대칭을 정확한 언어로 써야 한다. 이 장은 변환들의 모임을 **군**(group)으로, 군이 데이터에 작용하는 방식을 **표현**(representation)으로 쓰고, 모델이 대칭을 존중한다는 말을 **불변성**과 **등변성**이라는 등식으로 정의한다. 7.5장의 고윳값과 직교행렬, 7.8장의 회전군 $SO(3)$가 여기서 다시 쓰인다.

## 학습 목표

- 대칭을 군으로 표현한다
- 불변성과 등변성을 수학적으로 정의한다
- 과학 데이터의 대칭을 모델 설계로 연결한다

## 대칭과 군

**왜 필요한가.** "회전 대칭이 있다"는 말은 회전 하나가 아니라 모든 회전에 대한 말이다. 회전을 두 번 하면 또 하나의 회전이고, 회전은 되돌릴 수 있으며, 아무것도 안 하는 것도 회전(0°)이다. 이런 성질을 갖춘 변환의 모임이 군이다. 군의 언어로 쓰면 "어떤 대칭이 있는가"를 빠짐없이 나열하고, 대칭끼리의 관계(합성, 역)를 계산할 수 있다.

**정의.** 집합 $G$와 연산 $\cdot$가 다음 네 조건을 만족하면 군이라 한다.

1. **닫힘**: $g, h \in G$이면 $g \cdot h \in G$
2. **결합법칙**: $(g \cdot h) \cdot k = g \cdot (h \cdot k)$
3. **항등원**: 모든 $g$에 대해 $e \cdot g = g \cdot e = g$인 $e \in G$가 있다
4. **역원**: 모든 $g$에 대해 $g \cdot g^{-1} = g^{-1} \cdot g = e$인 $g^{-1} \in G$가 있다

변환들의 군에서 연산은 합성이고, 항등원은 아무것도 하지 않는 변환이다. 교환법칙 $g \cdot h = h \cdot g$는 요구하지 않는다. 3차원 회전은 순서를 바꾸면 결과가 달라진다(책을 x축으로 90° 돌린 뒤 z축으로 90° 돌린 것과 그 반대 순서를 비교해 보라).

**예.** 과학 데이터에서 자주 만나는 군은 다음과 같다.

| 군 | 원소 | 종류 | 대표적인 데이터 |
|---|---|---|---|
| $SO(2)$, $SO(3)$ | 2·3차원 회전 | 연속 | 분자 좌표, 점구름, 방향 데이터 |
| $O(3)$ | 회전과 반사 | 연속 | 거울상을 같게 볼 때의 분자 |
| $\mathbb{R}^3$ (평행이동) | $\mathbf{x} \mapsto \mathbf{x} + \mathbf{t}$ | 연속 | 위치에 의존하지 않는 물리 |
| $E(3)$ | 회전·반사·평행이동 | 연속 | 3차원 강체 운동 전체 |
| $C_n$ | $360°/n$ 회전 $n$개 | 이산 | 결정의 회전 대칭, 고리 분자 |
| $S_n$ | $n$개 원소의 치환 | 이산 | 집합, 점구름, 같은 종류의 원자 |

연속군(continuous group)은 원소가 연속적인 매개변수로 이어져 있는 군이다. 회전각을 조금씩 바꾸면 회전도 조금씩 바뀐다. 이런 군은 7.8장의 다양체이기도 해서 Lie 군이라 부르고, 항등원에서의 접공간(회전군에서는 반대칭 행렬)이 군의 국소 구조를 정한다. 이산군(discrete group)은 원소가 셀 수 있게 떨어져 있다.

## 군의 표현

**정의.** 군 원소는 추상적인 대상이지만, 데이터에 작용할 때는 행렬이 된다. 군 $G$의 **표현**은 각 원소 $g$에 가역 행렬 $\rho(g)$를 대응시키되 군의 연산을 행렬 곱으로 보존하는 사상이다.

$$
\rho(g \cdot h) = \rho(g)\,\rho(h), \qquad \rho(e) = \mathbf{I}.
$$

2차원 회전군 $SO(2)$의 표현으로 회전 행렬 $\rho(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$가 있고, $\rho(a)\rho(b) = \rho(a + b)$다. $S_n$은 치환 행렬로 표현된다. 치환 행렬은 단위행렬의 행 순서를 바꾼 행렬로, 벡터의 성분 순서를 바꾼다.

**회전과 반사.** 3차원 직교행렬 $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$의 모임이 $O(3)$이고, 행렬식은 $\pm 1$이다. 행렬식이 $+1$인 것이 회전($SO(3)$), $-1$인 것이 반사를 포함한 변환이다. 7.5장에서 본 대로 직교행렬의 고윳값은 절댓값이 1이다. 회전 행렬은 회전축 방향의 고윳값 1과 $e^{\pm i\theta}$($\theta$는 회전각)를 가진다. 반사 두 번은 행렬식 $(-1)(-1) = 1$이므로 회전이 된다. 반사만 모은 집합이 군이 아닌 이유다.

```python {exec="browser" session="groups"}
import numpy as np
from itertools import permutations

def rot(theta):                                     # 2차원 회전의 표현 ρ(θ), (2, 2)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

a, b = 0.7, -1.9
print("ρ(a) ρ(b) = ρ(a + b) :", np.allclose(rot(a) @ rot(b), rot(a + b)))

C4 = [rot(k * np.pi / 2) for k in range(4)]         # 90° 회전 4개
in_C4 = lambda M: any(np.allclose(M, g) for g in C4)
print("C4 닫힘              :", all(in_C4(g @ h) for g in C4 for h in C4))

S3 = [np.eye(3)[list(p)] for p in permutations(range(3))]   # 치환 행렬 6개, 각 (3, 3)
in_S3 = lambda M: any(np.array_equal(M, P) for P in S3)
print("S3 원소 수, 닫힘     :", len(S3), all(in_S3(P @ Q) for P in S3 for Q in S3))
print("치환 행렬은 직교     :", all(np.allclose(P.T @ P, np.eye(3)) for P in S3))

F1 = np.diag([1.0, -1.0])                           # x축에 대한 반사
F2 = rot(0.6) @ F1 @ rot(-0.6)                      # 기울어진 축에 대한 반사
print("det F1, det(F1 F2)   :", round(np.linalg.det(F1)), round(np.linalg.det(F1 @ F2)))
```

```output
ρ(a) ρ(b) = ρ(a + b) : True
C4 닫힘              : True
S3 원소 수, 닫힘     : 6 True
치환 행렬은 직교     : True
det F1, det(F1 F2)   : -1 1
```

**같은 군, 다른 표현.** 하나의 군이 서로 다른 대상에 서로 다른 행렬로 작용할 수 있다. 회전 $\mathbf{R} \in SO(3)$를 예로 들면 다음과 같다.

| 대상 | 변환 규칙 | 표현의 차원 | 물리량의 예 |
|---|---|---|---|
| 스칼라 | $s \mapsto s$ | 1 (자명한 표현) | 에너지, 온도, 질량 |
| 벡터 | $\mathbf{v} \mapsto \mathbf{R}\mathbf{v}$ | 3 | 힘, 속도, 전기 쌍극자 모멘트 |
| 2차 텐서 | $\mathbf{T} \mapsto \mathbf{R}\mathbf{T}\mathbf{R}^\top$ | 9 | 응력, 관성 텐서, 분극률 |

2차 텐서의 변환 $\mathbf{R}\mathbf{T}\mathbf{R}^\top$은 7.1장의 기저 변환(닮음 변환)과 같은 꼴이다. 좌표축을 돌리면 같은 물리적 텐서의 행렬 성분이 이렇게 바뀐다. 각운동량이나 자기장 같은 유사벡터(pseudovector)는 회전에는 벡터처럼 변하지만 반사에서는 부호가 한 번 더 바뀐다. 그래서 $SO(3)$만 볼 때와 $O(3)$를 볼 때 변환 규칙이 다르다.

## 불변성과 등변성

**정의.** 군 $G$가 입력 공간에 표현 $\rho_{\text{in}}$으로, 출력 공간에 표현 $\rho_{\text{out}}$으로 작용한다고 하자. 함수 $f$가 모든 $g \in G$에 대해

$$
f\big(\rho_{\text{in}}(g)\,\mathbf{x}\big) = f(\mathbf{x})
$$

를 만족하면 **불변**(invariant),

$$
f\big(\rho_{\text{in}}(g)\,\mathbf{x}\big) = \rho_{\text{out}}(g)\,f(\mathbf{x})
$$

를 만족하면 **등변**(equivariant)이라 한다. 불변은 출력 쪽 표현이 자명한($\rho_{\text{out}}(g) = \mathbf{I}$) 등변의 특수한 경우다. 말로 하면 등변은 "입력을 변환한 뒤 $f$를 적용하는 것과, $f$를 적용한 뒤 출력을 변환하는 것이 같다"이다. 두 경로가 같은 결과에 닿는다는 뜻에서 교환 가능한 그림(commutative diagram)으로 그리곤 한다.

**직관.** 점구름 $\mathbf{X} \in \mathbb{R}^{N \times 2}$에 회전 $\mathbf{R}$과 평행이동 $\mathbf{t}$를 적용한 $g\mathbf{X}$(모든 점 $\mathbf{x}_n \mapsto \mathbf{R}\mathbf{x}_n + \mathbf{t}$)를 생각한다. 정렬한 쌍별 거리는 점의 상대 배치만 보므로 불변이다. 무게중심은 점구름과 함께 움직이므로 등변이다. 좌표를 한 줄로 펼쳐 MLP에 넣으면 어느 쪽도 아니다.

```python {exec="browser" session="groups"}
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))                         # 점구름 (N, D)
R, t = rot(1.1), np.array([0.5, -2.0])
gX = X @ R.T + t                                    # g·X: 모든 점을 회전한 뒤 평행이동

def pair_dists(X):                                  # 불변 후보: 정렬한 쌍별 거리 (N(N-1)/2,)
    d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    return np.sort(d[np.triu_indices(len(X), 1)])

centroid = lambda X: X.mean(axis=0)                 # 등변 후보: 무게중심 (D,)
W = rng.normal(size=(10, 2))
mlp = lambda X: np.tanh(X.reshape(-1) @ W)          # 좌표를 펼쳐 넣는 일반 MLP 층 (2,)

print("거리   불변  f(gX) = f(X)       :", np.allclose(pair_dists(gX), pair_dists(X)))
print("중심   등변  f(gX) = R f(X) + t :", np.allclose(centroid(gX), R @ centroid(X) + t))
print("MLP    불변                     :", np.allclose(mlp(gX), mlp(X)))
print("MLP    등변(회전만)             :", np.allclose(mlp(X @ R.T), R @ mlp(X)))
```

```output
거리   불변  f(gX) = f(X)       : True
중심   등변  f(gX) = R f(X) + t : True
MLP    불변                     : False
MLP    등변(회전만)             : False
```

::quiz{id="centroid-equivariant"}

**예측 대상의 변환 규칙.** 모델이 불변이어야 하는지 등변이어야 하는지는 모델이 아니라 **예측 대상**이 정한다. 분자를 돌렸을 때 에너지는 그대로이고(스칼라, 불변), 원자마다의 힘은 함께 돌며(벡터, 등변), 분극률 텐서는 $\mathbf{R}\mathbf{T}\mathbf{R}^\top$로 바뀐다. 힘이 에너지의 음의 gradient라는 관계 $\mathbf{F}_n = -\partial E/\partial\mathbf{x}_n$에서도 이것이 보인다. 불변인 함수를 미분하면 gradient는 등변이다.

**합성.** 등변 함수끼리 합성하면 등변이고, 등변 함수 뒤에 불변 함수를 붙이면 전체가 불변이다. 신경망을 등변 층 여러 개와 마지막 불변 pooling으로 쌓는 설계가 이 성질을 쓴다. 층 하나하나가 대칭을 지키면, 몇 층을 쌓아도 전체가 대칭을 지킨다.

## 대칭을 모델에 넣는 방법

대칭을 모델에 넣는 방법은 크게 세 가지다.

**데이터 증강.** 학습할 때 입력에 무작위 군 원소를 적용하고, 정답도 대상의 변환 규칙대로 바꿔(불변이면 그대로, 벡터면 $\mathbf{R}\mathbf{v}$로) 함께 넣는다. 모델 구조를 바꾸지 않아도 되는 대신, 대칭은 손실을 통해 권장될 뿐 보장되지 않는다.

**불변 특징.** 입력을 먼저 대칭에 불변인 양(거리, 각, 정렬한 값)으로 바꾼 뒤 아무 모델에나 넣는다. 불변성은 정확히 보장된다. 다만 좋은 불변 특징을 설계하기가 어렵다. 정보를 너무 버리면 구별해야 할 입력을 구별하지 못하고, 불변 특징만으로는 벡터처럼 등변인 출력을 만들 수 없다.

**등변 구조.** 층 자체를 등변으로 설계한다. 핵심은 **가중치 공유**(weight sharing)다. 순열 대칭을 예로 들면, 집합의 원소 $N$개에 같은 가중치 $\mathbf{W}_{\text{self}}$를 적용하고, 순서에 무관한 요약(평균)을 같은 가중치 $\mathbf{W}_{\text{pool}}$로 모든 원소에 더한다.

$$
f(\mathbf{H}) = \mathbf{H}\mathbf{W}_{\text{self}} + \mathbf{1}_N\,\bar{\mathbf{h}}^\top\mathbf{W}_{\text{pool}} + \mathbf{1}_N\mathbf{b}^\top, \qquad \bar{\mathbf{h}} = \frac{1}{N}\sum_n \mathbf{h}_n.
$$

치환 행렬 $\mathbf{P}$에 대해 $(\mathbf{P}\mathbf{H})\mathbf{W}_{\text{self}} = \mathbf{P}(\mathbf{H}\mathbf{W}_{\text{self}})$이고 평균은 순서에 무관하므로 $f(\mathbf{P}\mathbf{H}) = \mathbf{P}f(\mathbf{H})$다.

```python {exec="browser" session="groups"}
N, D_in, D_out = 6, 4, 3
H = rng.normal(size=(N, D_in))                      # 집합 원소 N개의 특징 (N, D_in)
W_self = rng.normal(size=(D_in, D_out))
W_mean = rng.normal(size=(D_in, D_out))
bias = rng.normal(size=D_out)

def equi_layer(H):                                  # (N, D_in) → (N, D_out)
    return H @ W_self + H.mean(axis=0, keepdims=True) @ W_mean + bias

P = np.eye(N)[rng.permutation(N)]                   # 치환 행렬 (N, N)
print("등변  f(PH) = P f(H)      :", np.allclose(equi_layer(P @ H), P @ equi_layer(H)))
print("불변  sum f(PH) = sum f(H):", np.allclose(equi_layer(P @ H).sum(0), equi_layer(H).sum(0)))

W_flat = rng.normal(size=(N * D_in, N * D_out))     # 펼친 집합에 쓰는 일반 dense 층
dense = lambda H: (H.reshape(-1) @ W_flat).reshape(N, D_out)
print("dense 층 등변             :", np.allclose(dense(P @ H), P @ dense(H)))
print("파라미터 수  등변 층 =", W_self.size + W_mean.size + bias.size, "  dense 층 =", W_flat.size)
```

```output
등변  f(PH) = P f(H)      : True
불변  sum f(PH) = sum f(H): True
dense 층 등변             : False
파라미터 수  등변 층 = 27   dense 층 = 432
```

등변 층의 파라미터 수는 원소 수 $N$과 무관하고, 같은 층을 원소 수가 다른 집합에 그대로 쓸 수 있다. 가중치 공유는 대칭을 보장하는 동시에 파라미터를 줄인다. CNN이 평행이동에 등변인 것(18.2장), 그래프 신경망이 노드 번호 바꾸기에 등변인 것(18.6장)도 같은 원리다.

::quiz{id="deepsets-parameters"}

세 방법을 비교하면 다음과 같다.

| 방법 | 대칭 보장 | 장점 | 단점 |
|---|---|---|---|
| 데이터 증강 | 근사적 | 구조 변경 없음, 어떤 모델에도 적용 | 학습 비용 증가, 분포 밖에서 깨질 수 있음 |
| 불변 특징 | 정확 | 단순, 기존 모델 재사용 | 특징 설계가 어렵고 정보 손실, 등변 출력 불가 |
| 등변 구조 | 정확 | 데이터 효율, 등변 출력 가능 | 구현 복잡, 대칭마다 구조를 새로 설계 |

## 과학에서의 대칭

**어떤 대칭이 있는가.** 대칭은 데이터의 종류에서 바로 나오지 않는다. 같은 분자라도 무엇을 예측하느냐에 따라 필요한 대칭이 다르다.

- **분자.** 원자 좌표 전체의 회전·평행이동($E(3)$ 또는 $SO(3)$와 평행이동), 같은 원소 원자끼리의 번호 바꾸기($S_n$의 일부). 거울상 이성질체(chirality)를 구별해야 하는 물성에는 반사를 포함하면 안 된다(28장).
- **결정.** 격자 벡터만큼의 이산 평행이동과, 결정 구조마다 정해진 회전·반사의 유한군(점군). 단위 세포를 고르는 방법이 여러 가지라는 것도 일종의 대칭이다.
- **유체.** 경계와 외력이 허락하는 범위의 평행이동·회전, 그리고 일정한 속도로 움직이는 관찰자에게도 같은 법칙(Galilei 불변성). 경계조건과 외력은 이 대칭을 깨는 경우가 많다.
- **구면 위의 장(field).** 전 지구 기상장은 구면 위의 함수다. 그러나 지구 자전(Coriolis 힘), 지형, 해륙 분포 때문에 대기 역학은 구면의 회전에 대해 대칭이 아니다. 대칭은 데이터의 기하가 아니라 **지배 방정식**에서 확인해야 한다.

:::note[분야 예시]
결정 재료의 물성 예측에서 같은 결정 구조를 서로 다른 단위 세포나 원자 순서로 기록한 파일은 같은 물질이다. 원자 좌표를 그대로 펼쳐 넣는 모델은 같은 물질에 서로 다른 예측을 낼 수 있고, 데이터셋에 같은 물질이 다른 표기로 중복되어 있으면 학습–평가 분할에서 누수가 생긴다. 표현을 대칭에 불변으로 만드는 일은 모델 설계이자 데이터 정리 문제다.
:::

:::note[분야 예시]
등방성 재료의 응력–변형률 관계를 학습할 때, 응력과 변형률은 모두 2차 텐서다. 시편을 회전하면 두 텐서가 모두 $\mathbf{R}\mathbf{T}\mathbf{R}^\top$로 바뀌고, 학습한 구성 관계식도 이 규칙에 등변이어야 한다. 텐서 성분 6개를 독립적인 스칼라 6개로 다루면 이 제약이 사라지고, 학습 데이터에 없는 하중 방향에서 물리적으로 불가능한 응답을 낼 수 있다.
:::

**Noether 정리의 직관.** 물리 법칙의 연속 대칭은 보존 법칙과 짝을 이룬다. 대략적으로 말하면 계의 Lagrangian(운동을 결정하는 함수)이 어떤 연속 변환에 대해 변하지 않으면, 그 변환에 대응하는 양이 시간에 따라 보존된다.

| 대칭 | 보존량 |
|---|---|
| 시간 이동 | 에너지 |
| 공간 평행이동 | 선운동량 |
| 회전 | 각운동량 |

다음 셀은 질량 1 kg인 세 입자가 서로 용수철(상수 1 N/m)로 당기는 2차원 계를 velocity Verlet 방법으로 적분한다. 입자 사이의 힘은 거리에만 의존하는 중심력이라 계 전체가 회전 대칭을 가진다. 여기에 $x$ 방향으로만 작용하는 외부 퍼텐셜 $V = \frac{k}{2}x^2$을 더하면 회전 대칭이 깨진다.

```python {exec="browser" session="groups"}
def forces(x, k_ext):                               # x (N, 2) [m] → 힘 (N, 2) [N]
    diff = x[:, None, :] - x[None, :, :]            # (N, N, 2)
    f = -diff.sum(axis=1)                           # 쌍별 용수철(상수 1 N/m): 중심력, 회전 불변
    f[:, 0] -= k_ext * x[:, 0]                      # 외부장 V = k x²/2: x 방향만, 회전 불변 아님
    return f

def simulate(k_ext, steps=2000, dt_s=0.01):         # 질량 1 kg, velocity Verlet
    x = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.4, -0.9]])
    v = np.array([[0.0, 0.6], [-0.5, -0.2], [0.4, -0.3]])
    L_z = lambda x, v: np.sum(x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])   # 각운동량 [kg m²/s]
    L = [L_z(x, v)]
    for _ in range(steps):
        v_half = v + 0.5 * dt_s * forces(x, k_ext)
        x = x + dt_s * v_half
        v = v_half + 0.5 * dt_s * forces(x, k_ext)
        L.append(L_z(x, v))
    return np.array(L)

for k_ext in [0.0, 0.5]:
    L = simulate(k_ext)
    dL = np.abs(L - L[0]).max()
    print(f"외부장 k = {k_ext}:  L(0) = {L[0]:.4f}  최대 |L(t) - L(0)| =", "< 1e-12" if dL < 1e-12 else f"{dL:.2f}")
```

```output
외부장 k = 0.0:  L(0) = 1.5800  최대 |L(t) - L(0)| = < 1e-12
외부장 k = 0.5:  L(0) = 1.5800  최대 |L(t) - L(0)| = 3.04
```

회전 대칭이 있으면 각운동량은 반올림 오차 수준으로 보존되고, 대칭을 깨면 크게 변한다. 이 적분법(velocity Verlet)은 중심력 계에서 각운동량을 이산 시간에서도 정확히 보존하는 성질이 있다. 일반적인 적분법은 대칭이 있어도 보존량이 조금씩 흘러갈 수 있다. 학습한 동역학 모델이 대칭을 존중하지 않으면 긴 rollout에서 보존량이 흘러가는 것도 같은 이유다. 물리 지식을 모델에 넣는 방법으로서의 대칭은 24.8장에서 다시 다룬다.

::quiz{id="noether-correspondence"}

## 실습

2차원 점구름을 회전시키며 일반 MLP와 회전 불변 특징 기반 모델의 출력이 어떻게 달라지는지 측정한다. 코드는 `labs/linear-algebra/symmetry-and-groups/rotation_test.py`다.

- 점구름은 점 6개로, 학습 데이터는 모두 $x$ 방향으로 길쭉하게 정렬되어 있다(표준편차 $x$ 2.0, $y$ 0.5). 실제 데이터가 실험 설정 때문에 특정 방향으로 치우친 상황을 흉내 낸 것이다.
- 스칼라 목표는 회전 반경 $R_g = \sqrt{\frac{1}{N}\sum_n\lVert\mathbf{x}_n - \mathbf{c}\rVert^2}$(회전 불변), 벡터 목표는 $\mathbf{v} = \frac{1}{N}\sum_n\lVert\mathbf{x}_n - \mathbf{c}\rVert^2(\mathbf{x}_n - \mathbf{c})$(회전 등변)다. $\mathbf{c}$는 무게중심이다.
- 모델 A는 좌표를 펼친 MLP, B는 A에 무작위 회전 증강을 더한 것, C는 정렬한 쌍별 거리(불변 특징)를 받는 MLP다. 벡터 목표에는 불변 특징 MLP(C')와 등변 구성 $\hat{\mathbf{v}} = \frac{1}{N}\sum_n g(r_n)(\mathbf{x}_n - \mathbf{c})$(D, $g$는 거리 $r_n = \lVert\mathbf{x}_n - \mathbf{c}\rVert$의 함수)를 비교한다.
- 회전 민감도는 test 점구름 하나를 36개 각도로 돌렸을 때 예측의 표준편차(50개 점구름 평균), 등변성 오차는 $\lVert f(g\mathbf{X}) - \mathbf{R}f(\mathbf{X})\rVert$의 RMS다.

집필 시점의 실행 결과는 다음과 같다(MLP 학습 결과는 환경에 따라 조금 다를 수 있다).

```output
스칼라 목표 R_g (회전 불변)
  모델                       test RMSE   회전한 test RMSE   회전 민감도
  A 좌표 MLP                  0.0817             0.2936           1.64e-01
  B 좌표 MLP + 회전 증강      0.0932             0.0915           8.15e-02
  C 불변 특징 MLP              0.0167             0.0167            < 1e-12

벡터 목표 v (회전 등변)
  모델                       test RMSE   회전한 test RMSE   등변성 오차
  C' 불변 특징 MLP             4.5729             4.5510           4.49e-01
  D 등변 구성                  0.0783             0.0783            < 1e-12
  (참고: 항상 0을 예측하면 test RMSE = 4.5227)
```

결과를 읽는 법은 네 가지다.

- **좌표 MLP는 학습 방향에서만 맞는다.** 같은 방향의 test에서는 오차 0.082이지만, 회전한 test에서는 0.294로 3.6배 나빠진다.
- **증강은 민감도를 줄이지만 없애지 못한다.** 회전한 test 오차는 0.092로 좋아졌지만 회전 민감도는 A의 약 절반(0.082)으로 남는다. 학습 데이터가 같아도 모든 방향을 배우느라 같은 방향 test의 오차는 오히려 조금 나빠졌다.
- **불변 특징 모델은 정확히 불변이고 더 정확하다.** 회전 민감도는 부동소수점 수준이고, 모델이 회전을 따로 배울 필요가 없어 오차도 가장 작다.
- **불변 모델로는 벡터를 예측할 수 없다.** 불변 특징만 보는 모델은 점구름을 180° 돌린 입력(정답은 $-\mathbf{v}$)과 원래 입력(정답은 $\mathbf{v}$)에 같은 값을 내야 하므로, 항상 0을 예측하는 것과 비슷한 수준(4.57 대 4.52)에 머문다. 불변인 가중치 $g(r_n)$를 등변인 벡터 $\mathbf{x}_n - \mathbf{c}$에 곱하는 등변 구성은 정확히 등변이고 오차도 작다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="perm-equivariant-layer"}

::exercise{id="invariant-features"}

## 실패 사례

- **증강만으로 대칭이 보장된다고 생각하는 것.** 증강한 모델도 회전한 입력에 다른 답을 낸다(실습에서 회전 민감도 0.08). 학습 분포에서 멀어질수록 차이가 커진다. 대칭이 결론에 중요하다면 테스트 시 입력을 여러 방향으로 돌려 예측의 흩어짐을 보고하거나, 구조로 대칭을 보장한다.
- **벡터 물리량에 스칼라용 불변성을 적용하는 것.** 힘·속도·쌍극자 모멘트를 예측하면서 회전 불변 특징만 쓰면, 모델은 출력 방향을 정할 근거가 없다(실습의 C'). 예측 대상마다 스칼라·벡터·텐서 중 무엇인지 먼저 적고, 그에 맞는 변환 규칙을 요구한다. 증강할 때도 입력만 돌리고 벡터 정답을 돌리지 않으면 같은 실수가 된다.
- **데이터에 없는 대칭을 강제하는 것.** 반사 대칭을 넣으면 거울상 이성질체를 구별할 수 없고, 구면 회전 대칭을 넣으면 지형과 자전의 효과를 무시하게 된다. 대칭은 지배 방정식과 측정 설정에서 확인하고, 깨진 대칭은 깨진 대로 둔다.
- **좌표계 선택이 결과를 바꾸는 것을 모르고 지나치는 것.** 시료를 장비에 놓는 방향, 단위 세포 선택, 원자 번호 순서 같은 임의의 선택에 결과가 의존하면, 그 의존성은 물리가 아니라 표기의 산물이다. 같은 대상을 다른 표기로 넣어 예측이 같은지 확인하는 테스트를 둔다.

## 통과 기준

- 과학 데이터 하나(예: 분자, 결정, 점구름, 유체장)의 대칭군을 쓰고, 예측 대상 각각이 불변인지(스칼라), 등변인지(벡터·텐서), 어느 표현으로 변환되는지 판별한다.
- 함수 하나가 주어졌을 때 불변성·등변성을 무작위 군 원소로 수치 검증하는 코드를 쓰고, 실습처럼 증강·불변 특징·등변 구성의 차이를 회전 민감도로 보인다.
- 연습문제 두 개(`perm-equivariant-layer`, `invariant-features`)의 테스트를 통과한다.
