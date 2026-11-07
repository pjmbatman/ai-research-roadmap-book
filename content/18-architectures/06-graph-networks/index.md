---
id: architectures/graph-networks
title: "그래프 신경망과 기하학적 딥러닝"
title_en: "Graph Neural Networks and Geometric Deep Learning"
tier: advanced
status: draft
prerequisites: [architectures/inductive-bias]
---

# 그래프 신경망과 기하학적 딥러닝

분자는 원자와 결합, 센서망은 측정 지점과 그 사이의 연결, 조직 절편은 세포와 이웃 관계로 이루어진다. 이런 데이터는 격자(이미지)나 열(시계열)이 아니라 **그래프**다. 노드의 수가 샘플마다 다르고, 노드에 붙인 번호에는 아무 의미가 없다. 이 구조를 무시하고 노드 특징을 한 줄로 펼쳐 MLP에 넣으면, 같은 분자를 원자 순서만 바꿔 적었을 때 다른 예측이 나온다.

그래프 신경망(graph neural network, GNN)은 7.10장의 등변 층을 그래프로 일반화한 것이다. 각 노드가 이웃에게서 정보를 모으는 **message passing**을 같은 가중치로 모든 노드에 적용하면, 노드 번호를 어떻게 붙여도 결과가 일관된다. 이 장은 그 구조를 직접 계산으로 확인하고, GNN에서 가장 자주 틀리는 결정인 **그래프를 어떻게 만들 것인가**를 데이터의 물리적 구조로 정당화하는 법을 다룬다.

## 학습 목표

- 그래프를 인접행렬·특징 행렬로 표현하고 message passing을 행렬 연산으로 계산한다
- message passing의 순열 등변성을 확인하고, 집계 함수가 구별할 수 있는 이웃 구조를 설명한다
- 그래프 구성 방식을 데이터의 물리적 구조로 정당화하고, over-smoothing·over-squashing을 진단한다

## 그래프 표현

**정의.** 그래프 $G = (\mathcal{V}, \mathcal{E})$는 노드 집합 $\mathcal{V}$와 간선 집합 $\mathcal{E}$로 이루어진다. 노드가 $N$개이면 연결 관계를 **인접행렬**(adjacency matrix) $\mathbf{A} \in \{0, 1\}^{N \times N}$로 쓴다. 노드 $i$와 $j$가 연결되어 있으면 $A_{ij} = 1$이다. 방향이 없는 그래프에서는 $\mathbf{A}$가 대칭이다. 노드 $i$의 **차수**(degree) $d_i = \sum_j A_{ij}$는 이웃의 수다. 그 밖에 세 종류의 특징이 붙는다.

- **노드 특징** $\mathbf{X} \in \mathbb{R}^{N \times D}$: 원자 종류, 센서의 고도, 세포의 유전자 발현.
- **간선 특징** $\mathbf{e}_{ij}$: 결합 차수, 센서 사이 거리, 배관의 지름.
- **그래프 특징**: 분자 전체의 전하, 측정 조건.

실제 구현은 $N \times N$ 행렬 대신 간선 목록(edge list, 연결된 노드 쌍의 목록)을 쓰는 경우가 많다. 분자나 센서망처럼 간선이 노드 수에 비례하는 희소 그래프에서는 $N^2$ 크기의 행렬이 대부분 0이기 때문이다. 수식과 작은 예제에서는 인접행렬이 투명하다.

**코드.** 6원자 고리에 곁사슬이 붙은 분자 골격을 간선 목록으로 적고 인접행렬로 바꾼다. 수소는 생략하고, 노드 특징은 원소 종류의 one-hot이다.

```python {exec="browser" session="gnn"}
import numpy as np

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (2, 6), (6, 7)]  # 6원자 고리 + 곁사슬
N = 8
A = np.zeros((N, N))                                    # 인접행렬 (N, N)
for i, j in edges:
    A[i, j] = A[j, i] = 1.0                             # 무방향 그래프: 대칭
types = np.array([0, 0, 0, 0, 0, 1, 0, 2])              # 0 = C, 1 = N, 2 = O
X = np.eye(3)[types]                                    # 노드 특징: 원소 one-hot (N, 3)

print("차수     :", A.sum(1).astype(int))
print("대칭     :", np.array_equal(A, A.T), "  간선 수:", int(A.sum() / 2))
print("이웃 합 A @ X  (열 = 이웃의 C, N, O 개수)")
print((A @ X).astype(int))

A_hat = A + np.eye(N)                                   # self-loop 추가
d = A_hat.sum(1)
A_norm = A_hat / np.sqrt(np.outer(d, d))                # D^-1/2 (A+I) D^-1/2 (N, N)
print("A_norm[2, 6] =", round(A_norm[2, 6], 4), " = 1/sqrt(4*3):", round(1 / np.sqrt(12), 4))
```

```output
차수     : [2 2 3 2 2 2 2 1]
대칭     : True   간선 수: 8
이웃 합 A @ X  (열 = 이웃의 C, N, O 개수)
[[1 1 0]
 [2 0 0]
 [3 0 0]
 [2 0 0]
 [1 1 0]
 [2 0 0]
 [1 0 1]
 [1 0 0]]
A_norm[2, 6] = 0.2887  = 1/sqrt(4*3): 0.2887
```

**직관.** 행렬 곱 $\mathbf{A}\mathbf{X}$의 $i$번째 행은 노드 $i$의 이웃들의 특징을 더한 것이다. 고리의 분기점인 노드 2는 이웃 탄소가 3개, 산소에 붙은 노드 6은 탄소 1개와 산소 1개를 이웃으로 가진다. 인접행렬을 곱하는 것이 곧 "이웃에게서 모으기"다. 이것이 message passing의 계산 핵심이다.

마지막 두 줄은 다음 절의 GCN이 쓰는 정규화다. self-loop를 더한 $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$와 그 차수 $\tilde{d}_i$로 $\hat{A}_{ij} = \tilde{A}_{ij}/\sqrt{\tilde{d}_i\tilde{d}_j}$를 만든다. 차수가 큰 노드에서 합이 커지는 것을 막고, 행렬의 고윳값을 $[-1, 1]$ 안에 두어 여러 번 곱해도 값이 폭발하지 않게 한다.

## message passing

**정의.** message passing 층은 노드마다 세 단계를 거친다(Gilmer 외, 2017).

$$
\mathbf{m}_{ij} = \phi(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}), \qquad
\mathbf{m}_i = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}, \qquad
\mathbf{h}_i' = \psi(\mathbf{h}_i, \mathbf{m}_i).
$$

$\mathbf{h}_i$는 노드 $i$의 현재 표현, $\mathcal{N}(i)$는 이웃 집합, $\phi$는 이웃 $j$가 보내는 메시지를 만드는 함수, $\bigoplus$는 메시지를 모으는 **집계 함수**(합·평균·최댓값), $\psi$는 모은 메시지로 표현을 갱신하는 함수다. $\phi$와 $\psi$는 모든 노드와 간선에 **같은 가중치**로 적용된다. 층 하나를 지나면 각 노드는 1-hop 이웃의 정보를, $k$층을 지나면 $k$-hop 이내 노드의 정보를 담는다.

대표적인 세 구조는 $\phi$와 $\bigoplus$의 선택만 다르다.

- **GCN**(graph convolutional network)은 정규화한 이웃 합에 선형층을 씌운다(Kipf, Welling, 2017): $\mathbf{H}' = \sigma(\hat{\mathbf{A}}\mathbf{H}\mathbf{W})$. 이웃마다 고정된 가중치 $1/\sqrt{\tilde{d}_i\tilde{d}_j}$로 평균에 가까운 집계를 한다.
- **GAT**(graph attention network)는 이웃마다의 가중치를 고정하지 않고, 두 노드의 표현으로 계산한 attention 점수를 이웃에 대해 softmax한 값으로 쓴다. 18.4장의 attention을 이웃 집합에 제한한 것이다.
- **GIN**(graph isomorphism network)은 합 집계 뒤에 MLP를 둔다: $\mathbf{h}_i' = \mathrm{MLP}\big((1 + \epsilon)\mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j\big)$. 아래에서 보듯 합 집계가 이웃의 개수를 구별하기 때문이다(Xu 외, 2019).

::quiz{id="gcn-normalized-weight"}

**코드.** 자기 자신과 이웃 합에 서로 다른 가중치를 거는 가장 단순한 층을 만들고, 노드 번호를 바꿔도 결과가 일관되는지, 합과 평균 집계가 무엇을 구별하는지 본다.

```python {exec="browser" session="gnn"}
rng = np.random.default_rng(0)
W_self, W_nei = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))   # 노드 수와 무관한 가중치

def aggregate(A, H, agg="sum"):                         # 이웃 특징 모으기 (N, D_in)
    M = A @ H                                           # 이웃의 합
    return M / np.maximum(A.sum(1, keepdims=True), 1) if agg == "mean" else M

def mp_layer(A, H, agg="sum"):                          # (N, N), (N, D_in) → (N, D_out)
    return np.maximum(0, H @ W_self + aggregate(A, H, agg) @ W_nei)

P = np.eye(N)[rng.permutation(N)]                       # 노드 번호 바꾸기 (N, N)
H1 = mp_layer(A, X)
print("등변  f(PAP^T, PX) = P f(A, X):", np.allclose(mp_layer(P @ A @ P.T, P @ X), P @ H1))
print("불변  readout 합              :", np.allclose(mp_layer(P @ A @ P.T, P @ X).sum(0), H1.sum(0)))
W_flat = rng.normal(size=(N * 3, 4))                    # 펼친 특징에 쓰는 MLP 층
print("펼친 MLP 불변                 :", np.allclose((P @ X).ravel() @ W_flat, X.ravel() @ W_flat))

# 중심 노드 0의 이웃이 탄소 1개인 그래프 vs 탄소 2개인 그래프
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.]]); A2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0.]])
Xc = np.tile(np.eye(3)[[0]], (3, 1))                    # 모든 노드가 탄소 (3, 3)
for agg in ["sum", "mean"]:
    same = np.allclose(aggregate(A1, Xc, agg)[0], aggregate(A2, Xc, agg)[0])
    print(f"{agg:4} 집계: 이웃 1개와 2개를 {'구별 못 함' if same else '구별함'}")
```

```output
등변  f(PAP^T, PX) = P f(A, X): True
불변  readout 합              : True
펼친 MLP 불변                 : False
sum  집계: 이웃 1개와 2개를 구별함
mean 집계: 이웃 1개와 2개를 구별 못 함
```

앞의 세 줄은 다음 절의 주제다. 마지막 두 줄은 **집계 함수가 표현력을 정한다**는 것을 보여 준다. 이웃이 모두 탄소일 때, 이웃 탄소 1개와 2개의 평균은 똑같이 "탄소"다. 평균 집계는 이웃 특징의 분포만 보고 개수를 잃는다. 최댓값 집계는 더 많이 잃어, 이웃에 어떤 종류가 있는지만 남긴다. 합 집계는 개수까지 보존한다. GIN 논문은 이 관찰을 바탕으로, 합 집계와 MLP를 쓰는 message passing이 그래프 동형 판정의 고전 알고리즘인 Weisfeiler–Lehman 검사만큼 그래프를 구별할 수 있고, message passing GNN은 그 이상을 구별할 수 없음을 보였다(Xu 외, 2019).

그렇다고 합이 늘 좋은 것은 아니다. 노드의 차수가 크게 다른 그래프(사회망, 일부 센서망)에서는 합의 크기가 차수에 따라 크게 달라져 학습이 불안정해진다. 이웃 수가 의미 있는 정보인지(분자의 결합 수는 의미가 있다), 측정 밀도의 우연인지(센서를 촘촘히 설치한 지역)를 먼저 따진다.

::quiz{id="mean-cannot-count"}

## 대칭성

**순열 등변성.** 노드 번호를 바꾸는 것은 치환 행렬 $\mathbf{P}$로 쓸 수 있다. 특징 행렬은 $\mathbf{P}\mathbf{X}$로, 인접행렬은 행과 열이 함께 바뀌어 $\mathbf{P}\mathbf{A}\mathbf{P}^\top$이 된다. 7.10장의 정의를 그대로 적용하면, 노드마다 출력을 내는 층 $f$가

$$
f(\mathbf{P}\mathbf{A}\mathbf{P}^\top, \mathbf{P}\mathbf{X}) = \mathbf{P}\,f(\mathbf{A}, \mathbf{X})
$$

를 만족할 때 **순열 등변**(permutation equivariant)이다. 위 층은 $(\mathbf{P}\mathbf{A}\mathbf{P}^\top)(\mathbf{P}\mathbf{X}) = \mathbf{P}\mathbf{A}\mathbf{X}$이고($\mathbf{P}^\top\mathbf{P} = \mathbf{I}$), 가중치를 오른쪽에서 곱하고, ReLU를 성분마다 적용하므로 등변이다. 등변 층을 쌓은 뒤 노드에 대해 합이나 평균을 내는 **readout**을 붙이면 그래프 전체의 출력은 순열 **불변**(invariant)이 된다. 코드의 첫 두 줄이 이것을 확인했다. 노드 특징을 펼쳐 넣는 MLP는 번호를 바꾸면 출력이 바뀐다(셋째 줄).

7.10장의 등변 층(자기 자신 + 집합 전체의 평균)은 모든 노드가 서로 연결된 완전 그래프 위의 message passing으로 볼 수 있다. GNN은 그 평균을 **이웃에 대한 집계**로 바꿔, 그래프의 연결 구조라는 추가 정보를 쓴다.

**무엇을 등변으로, 무엇을 불변으로.** 7.10장의 원칙대로 예측 대상이 정한다. 원자마다의 부분 전하나 센서마다의 다음 시점 값은 노드 번호와 함께 움직이므로 등변 출력이고, 분자의 총 에너지나 그래프 전체의 분류 레이블은 불변 출력이다.

::quiz{id="node-vs-graph-target"}

**E(n) 등변 네트워크.** 분자나 입자계처럼 노드가 공간 좌표 $\mathbf{x}_i \in \mathbb{R}^n$를 가지면 대칭이 하나 더 생긴다. 계 전체를 회전·반사·평행이동(유클리드 군 $E(n)$)해도 에너지는 그대로이고 힘은 함께 돈다. 좌표를 노드 특징에 그대로 넣으면 이 대칭이 깨진다. EGNN(Satorras 외, 2021)은 7.10장의 등변 구성 $g(r)(\mathbf{x} - \mathbf{c})$와 같은 원리를 쓴다. 메시지는 불변량인 거리 $\lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2$와 노드 특징으로만 만들고, 좌표는 상대 벡터 $\mathbf{x}_i - \mathbf{x}_j$에 불변 스칼라 가중치를 곱해 더하는 방식으로 갱신한다. 거리는 회전에 불변이고 상대 벡터는 회전에 등변이므로, 층 전체가 노드 특징에는 불변, 좌표에는 등변이다. 이런 기하학적 모델은 24.8장에서 과학 문제와 함께 다시 다룬다.

## 시공간 그래프

**왜 필요한가.** 센서망은 공간적으로 그래프이고, 각 노드는 시간에 따라 값을 낸다. 기상 관측소, 전력망의 변전소, 하천 수위계, 교통 검지기가 그렇다. 한 노드의 미래는 자기 과거뿐 아니라 이웃의 과거에도 의존한다. 상류의 수위가 오르면 몇 시간 뒤 하류가 오른다.

**구조.** 시공간 그래프 모델은 보통 두 가지 연산을 번갈아 쌓는다.

- **공간 연산**: 각 시점에서 message passing으로 이웃 노드의 정보를 섞는다.
- **시간 연산**: 각 노드에서 18.3장의 RNN·TCN이나 18.4장의 attention으로 시간 축의 정보를 섞는다.

입력은 $(T, N, D)$ 텐서(시간 × 노드 × 특징)가 된다. 평가는 18.7장의 시계열 평가 원칙을 그대로 따른다. 시간 순서로 분할하고, 각 노드의 persistence(마지막 값 유지) baseline과 비교한다. 거기에 공간 일반화라는 질문이 더해진다. 학습에 쓰지 않은 **새 노드**(새로 설치한 센서)에서 성능을 따로 보고해야 한다.

:::note[분야 예시]
공간 전사체(spatial transcriptomics) 데이터에서는 조직 절편 위의 각 위치(또는 세포)가 유전자 발현 벡터를 가진다. 위치를 노드로, 물리적으로 가까운 위치끼리를 간선으로 잇는 그래프를 만들면, 세포가 이웃 세포와 주고받는 신호를 message passing으로 모델링할 수 있다. 이때 간선은 "거리가 몇 μm 이내"라는 물리적 가정이다. 같은 데이터에서 발현 유사도로 그래프를 만들면 "비슷한 세포끼리 연결"이 되어, 이웃의 영향이라는 질문 자체가 바뀐다.
:::

## 그래프 구성

**왜 중요한가.** 분자의 결합처럼 그래프가 데이터에 주어진 경우도 있지만, 많은 과학 데이터에서 그래프는 **연구자가 만드는 가정**이다. 어떤 두 센서를 연결할 것인가는 "어떤 측정이 서로에게 정보를 주는가"에 대한 가설이고, 그 가설이 모델의 귀납적 편향이 된다(18.1장).

그래프를 만드는 방법은 크게 세 가지다.

| 방법 | 근거 | 장점 | 위험 |
|---|---|---|---|
| 물리적 연결 | 거리, 배관·하천·선로의 위상, 화학 결합 | 새 조건에도 유지되는 구조 | 물리적으로 먼 상호작용(원격 상관)을 놓침 |
| 학습된 연결 | 학습 구간 데이터의 상관, 또는 학습 중 최적화한 인접행렬 | 데이터에 있는 관계를 직접 반영 | 학습 구간의 조건에 맞춰져 조건이 바뀌면 깨짐 |
| 완전 연결 | 가정 없음 | 놓치는 연결이 없음 | 국소성 정보가 사라지고 계산량은 $N^2$ |

**코드.** 30개 센서가 $100 \times 100\,\mathrm{m}$ 영역에 흩어져 매끄러운 장(field)을 측정한다. 장은 몇 개의 공간 모드(파수 벡터 $\mathbf{k}$)의 합이고, 시점마다 모드의 진폭만 바뀐다. 각 센서 값을 이웃 센서 값의 평균으로 예측하는(빠진 센서 채우기) 가장 단순한 message passing으로 네 그래프를 비교한다. 학습 상관 그래프는 학습 구간 200시점의 센서 간 상관에서 가장 상관이 높은 4개를 잇는다. 마지막 열은 파수 벡터를 90° 돌려, 학습 구간과 공간 패턴이 다른 새 조건을 만든 것이다.

```python {exec="browser" session="gnn"}
rng = np.random.default_rng(2)
S, T_train, T_test = 30, 200, 100                       # 센서 30개, 학습·평가 시점 수
pos_m = rng.random((S, 2)) * 100.0                      # 센서 위치 (S, 2), 단위 m
k_rad_m = rng.normal(size=(6, 2)) * 0.04                # 공간 파수 6개 (6, 2), 단위 rad/m
phase = rng.random(6) * 2 * np.pi
def field(T, k_rad_m):                                  # 모드 진폭만 시점마다 바뀌는 매끄러운 장 (T, S)
    amp = rng.normal(size=(T, 6))
    return amp @ np.sin(pos_m @ k_rad_m.T + phase).T
Y_train, Y_test = field(T_train, k_rad_m), field(T_test, k_rad_m)
Y_shift = field(T_test, k_rad_m[:, ::-1] * [1, -1])     # 파수 벡터를 90° 돌린 새 조건

def knn_graph(Dist, k=4):                               # 각 노드에서 가장 가까운 k개로 연결 (S, S)
    G = np.zeros((S, S))
    G[np.arange(S)[:, None], np.argsort(Dist, 1)[:, 1:k + 1]] = 1.0
    return G

dist_m = np.linalg.norm(pos_m[:, None] - pos_m[None], axis=-1)     # 물리적 거리 (S, S)
corr = np.corrcoef(Y_train.T)                                      # 학습 구간의 센서 간 상관 (S, S)
graphs = {"물리적 거리 kNN": knn_graph(dist_m), "학습 상관 kNN  ": knn_graph(-corr),
          "완전 연결      ": 1.0 - np.eye(S), "무작위 4-이웃  ": knn_graph(rng.random((S, S)) - np.eye(S))}
mae = lambda Y, G: np.abs(Y @ G.T / G.sum(1) - Y).mean()          # 각 센서를 이웃 평균으로 예측
print("그래프            같은 조건 MAE   돌린 조건 MAE")
for name, G in graphs.items():
    print(f"{name}       {mae(Y_test, G):.3f}          {mae(Y_shift, G):.3f}")
print(f"(참고) 0 예측         {np.abs(Y_test).mean():.3f}          {np.abs(Y_shift).mean():.3f}")
```

```output
그래프            같은 조건 MAE   돌린 조건 MAE
물리적 거리 kNN       0.614          0.495
학습 상관 kNN         0.451          0.904
완전 연결             1.063          1.134
무작위 4-이웃         1.150          1.228
(참고) 0 예측         1.329          1.370
```

결과는 세 가지로 읽는다.

- **완전 연결은 무작위 그래프와 비슷한 수준이다.** 모든 센서의 평균은 영역 전체의 평균이라 위치 정보가 사라진다. 이웃 평균이라는 연산이 국소성을 쓰려면 그래프가 국소적이어야 한다.
- **같은 조건에서는 학습된 연결이 물리적 거리보다 낫다(0.451 대 0.614).** 이 장은 방향에 따라 변화의 빠르기가 다르다(파수 벡터가 등방적이지 않다). 상관 그래프는 그 방향성을 학습 데이터에서 읽어 낸다. 물리적 거리만 쓰면 이 정보를 버린다.
- **조건이 바뀌면 순위가 뒤집힌다(0.904 대 0.495).** 공간 패턴의 방향이 바뀌자, 학습 구간의 상관으로 만든 그래프는 더 이상 맞는 이웃을 가리키지 않는다. 물리적 거리는 장의 패턴과 무관한 가정이라 그대로 유효하다.

어느 그래프가 "옳은가"는 데이터만으로 정해지지 않는다. **배포할 조건이 학습 조건과 같은가**에 달려 있다. 계절·운전 조건·실험 설정이 바뀌는 과학 데이터에서는, 학습된 연결의 성능을 학습 조건과 다른 구간에서 따로 평가해야 한다.

::quiz{id="complete-graph-mean"}

:::note[분야 예시]
결정 재료의 물성 예측에서는 원자를 노드로, 일정 거리(cutoff) 안의 원자 쌍을 간선으로 잇는 그래프를 흔히 쓴다. 주기 경계 조건 때문에 단위 세포 밖의 이웃 원자까지 고려해야 하고, cutoff를 너무 작게 잡으면 물성에 중요한 상호작용을 놓치며, 너무 크게 잡으면 그래프가 거의 완전 연결이 되어 계산량이 급증하고 국소성이 흐려진다. cutoff는 하이퍼파라미터이자 "이 물성은 몇 Å 거리의 상호작용으로 정해진다"는 물리적 가정이므로, 값을 바꿔 가며 결과가 얼마나 민감한지 보고한다.
:::

## 한계

**over-smoothing.** GCN의 전파 $\hat{\mathbf{A}}\mathbf{H}$는 각 노드를 이웃과의 가중 평균으로 바꾸는 라플라시안 평활화(Laplacian smoothing)의 한 형태다(Li, Han, Wu, 2018). 몇 번은 잡음을 줄이는 데 도움이 되지만, 반복하면 연결된 모든 노드의 표현이 같은 방향으로 수렴한다. 연결 그래프에서 $\hat{\mathbf{A}}^k\mathbf{H}$는 $k \to \infty$일 때 $\hat{\mathbf{A}}$의 고윳값 1에 해당하는 고유벡터(성분이 $\sqrt{\tilde{d}_i}$에 비례) 방향으로 모이므로, 노드의 표현은 차수에 따른 크기만 다르고 방향이 같아진다. 이것이 **over-smoothing**이다.

두 공동체(같은 공동체끼리 더 자주 연결)로 이루어진 40노드 그래프에서, 잡음이 큰 노드 특징에 학습 파라미터 없는 GCN 전파를 반복한다. 공동체 평균에 가까운 쪽으로 분류하는 정확도와, 서로 다른 공동체 노드 사이의 cosine 유사도를 본다.

```python {exec="browser" session="gnn"}
rng = np.random.default_rng(1)
n, D = 40, 8                                            # 공동체 2개 × 20 노드, 특징 8차원
y = np.repeat([0, 1], n // 2)
p_edge = np.where(y[:, None] == y[None, :], 0.25, 0.05) # 같은 공동체끼리 더 자주 연결
A = np.triu(rng.random((n, n)) < p_edge, 1).astype(float)
A = A + A.T
mu = rng.normal(size=(2, D))
X = mu[y] + 2.0 * rng.normal(size=(n, D))               # 잡음이 큰 노드 특징 (n, D)

A_hat = A + np.eye(n); d = A_hat.sum(1)
A_norm = A_hat / np.sqrt(np.outer(d, d))                # (n, n)

def centroid_acc(H):                                    # 공동체 평균에 가까운 쪽으로 분류
    c = np.stack([H[y == k].mean(0) for k in (0, 1)])
    return (np.argmin(((H[:, None] - c) ** 2).sum(-1), 1) == y).mean()

H = X.copy()
for k in range(65):
    if k in (0, 1, 2, 4, 8, 16, 32, 64):
        Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
        cos = (Hn @ Hn.T)[y[:, None] != y[None, :]].mean()   # 다른 공동체 노드 사이 cosine
        print(f"전파 {k:2d}번  분류 정확도 {centroid_acc(H):.2f}   다른 공동체 간 cos {cos:+.3f}")
    H = A_norm @ H                                      # 학습 파라미터 없는 GCN 전파
```

```output
전파  0번  분류 정확도 0.88   다른 공동체 간 cos +0.039
전파  1번  분류 정확도 0.97   다른 공동체 간 cos +0.264
전파  2번  분류 정확도 1.00   다른 공동체 간 cos +0.460
전파  4번  분류 정확도 1.00   다른 공동체 간 cos +0.687
전파  8번  분류 정확도 0.97   다른 공동체 간 cos +0.892
전파 16번  분류 정확도 0.88   다른 공동체 간 cos +0.989
전파 32번  분류 정확도 0.62   다른 공동체 간 cos +1.000
전파 64번  분류 정확도 0.62   다른 공동체 간 cos +1.000
```

전파 2–4번까지는 이웃 평균이 잡음을 줄여 정확도가 0.88에서 1.00으로 오른다. 그 뒤로는 공동체 사이의 경계까지 평균되어, 32번이면 다른 공동체 노드끼리도 cosine이 1에 이르고 정확도는 0.62로 떨어진다. 실제 GNN 층에는 가중치와 비선형성이 있어 수렴이 이렇게 단순하지는 않지만, 깊게 쌓은 GNN이 얕은 GNN보다 나빠지는 흔한 이유가 이것이다. 진단 지표로는 층별 노드 표현 사이의 평균 거리나 **Dirichlet 에너지** $\tfrac{1}{2}\sum_{i,j} A_{ij}\lVert\mathbf{h}_i - \mathbf{h}_j\rVert^2$를 쓴다. 층이 깊어질수록 이 값이 0으로 떨어지면 over-smoothing이다. 대응으로는 층 수를 줄이기, 잔차 연결(residual connection), 각 층의 표현을 모두 이어 붙여 쓰기 등이 있다.

**over-squashing.** $k$층 GNN의 노드 하나는 $k$-hop 이웃 전체의 정보를 고정된 크기의 벡터 하나에 담아야 한다. 나무처럼 가지를 치는 그래프에서는 $k$-hop 이웃의 수가 $k$에 대해 지수적으로 늘어난다. 먼 노드의 정보가 좁은 경로(병목)를 지나며 다른 정보와 함께 눌려 거의 전달되지 않는 현상을 **over-squashing**이라 한다(Alon, Yahav, 2021). over-smoothing이 "모든 노드가 같아지는" 문제라면, over-squashing은 "먼 곳의 정보가 도착하지 않는" 문제다. 그래서 층을 늘리는 것으로는 해결되지 않고(over-smoothing이 심해진다), 먼 노드 사이에 간선을 추가하거나 그래프 전체에 대한 attention을 함께 쓰는 식으로 그래프 구조 자체를 바꾼다.

::quiz{id="oversmoothing-vs-squashing"}

## 실습

물리적 연결로 만든 그래프와 학습된 그래프로 같은 문제를 풀어 비교한다. 출발점은 본문의 센서 실험(`labs/architectures/graph-networks/cells.py`의 cell 3)이다.

1. 이웃 수 $k$를 2, 4, 8로 바꿔 네 그래프의 MAE가 어떻게 변하는지 표로 만든다. $k$가 커질수록 kNN 그래프가 완전 연결에 가까워지는 것을 확인한다.
2. 학습 구간 길이(`T_train`)를 20, 50, 200으로 바꾼다. 학습 상관 그래프는 학습 자료가 적을 때 얼마나 흔들리는가. 물리적 그래프는 학습 자료가 필요 없다는 점과 비교한다.
3. 파수 벡터의 회전 각도를 0°에서 90°까지 바꿔, 학습 상관 그래프가 물리적 그래프보다 나빠지는 지점을 찾는다.
4. 결과를 바탕으로 "이 데이터에 어떤 그래프를 쓰고, 왜 다른 그래프를 기각하는가"를 세 문장으로 쓴다. 근거는 데이터의 물리적 구조와 배포 조건이어야 한다.

이어서 over-smoothing 진단 지표를 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="dirichlet-energy"}

## 실패 사례

- **근거 없이 완전 연결 그래프를 쓰는 것.** "연결을 다 넣으면 모델이 알아서 고른다"는 생각은 이웃 평균·합 집계에서는 성립하지 않는다. 본문 실험에서 완전 연결은 무작위 그래프와 비슷했다. attention으로 가중치를 학습하더라도 계산량은 $N^2$이고, 국소성이라는 귀납적 편향은 사라진다.
- **학습 구간에서 만든 상관 그래프를 다른 조건에 그대로 쓰는 것.** 같은 조건에서 가장 좋았던 그래프가 조건이 바뀌자 가장 나빠질 수 있다. 그래프를 학습 데이터로 만들었다면, 그 그래프의 일반화도 평가 대상이다. 평가 구간의 데이터로 그래프를 만들면 누수가 된다.
- **층을 깊게 쌓으면 더 넓은 정보를 쓸 것이라 기대하는 것.** 깊이는 수용 영역을 넓히지만 over-smoothing과 over-squashing을 부른다. 층 수에 따른 검증 성능과 노드 표현의 Dirichlet 에너지를 함께 보고한다.
- **노드 순서에 의존하는 연산을 섞는 것.** 노드 특징을 펼쳐 MLP에 넣거나, 노드 번호를 특징으로 쓰면 순열 등변성이 깨진다. 노드 순서를 무작위로 바꿔 출력이 일관되는지 테스트로 확인한다.

## 통과 기준

- 간선 목록에서 인접행렬과 정규화 인접행렬을 만들고, message passing 층이 순열 등변이고 readout이 불변임을 코드로 확인한다.
- 합·평균·최댓값 집계가 구별하는 이웃 구조의 차이를 예로 설명한다.
- 자기 데이터(또는 본문 센서 실험)에서 물리적 그래프와 학습된 그래프를 같은 조건과 바뀐 조건에서 비교하고, 그래프 구성을 데이터의 물리적 구조로 정당화한다.
- 연습문제 `dirichlet-energy`의 테스트를 통과하고, 층 수에 따른 over-smoothing을 지표로 보인다.

## 참고 문헌

- Gilmer, Schoenholz, Riley, Vinyals, Dahl (2017). Neural Message Passing for Quantum Chemistry. *ICML 2017*, PMLR 70, 1263–1272. — message passing 신경망의 공통 틀
- Kipf, Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. — GCN과 self-loop를 더한 정규화
- Xu, Hu, Leskovec, Jegelka (2019). How Powerful are Graph Neural Networks? *ICLR 2019*. — 집계 함수와 Weisfeiler–Lehman 검사, GIN
- Li, Han, Wu (2018). Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. *AAAI 2018*. — GCN 전파를 라플라시안 평활화로 보는 관점과 over-smoothing
- Alon, Yahav (2021). On the Bottleneck of Graph Neural Networks and its Practical Implications. *ICLR 2021*. — over-squashing
- Satorras, Hoogeboom, Welling (2021). E(n) Equivariant Graph Neural Networks. *ICML 2021*, PMLR 139, 9323–9332. — 거리와 상대 벡터로 만드는 E(n) 등변 message passing
