---
id: scientific-ml/geometric-models
title: "대칭성을 반영한 과학 모델"
title_en: "Symmetry-Aware Models for Science"
tier: advanced
status: draft
prerequisites: [architectures/graph-networks, linear-algebra/symmetry-and-groups]
---

# 대칭성을 반영한 과학 모델

분자를 통째로 돌려도 에너지는 그대로이고, 원자에 걸리는 힘은 분자와 함께 돈다. 실험실을 옮겨도 물리 법칙은 바뀌지 않는다. 이런 대칭은 과학 데이터가 가진 가장 확실한 사전 지식이다. 대칭을 모르는 모델은 같은 구조를 여러 방향에서 다시 배워야 하고, 학습 때 보지 못한 방향에서는 틀린다. 대칭을 아는 모델은 한 방향에서 배운 것을 모든 방향에 쓴다.

7.10장은 군·표현·등변성의 정의와 집합 위의 등변 층을, 18.6장은 그래프 위의 message passing과 순열 등변성을 다루었다. 이 장은 그 도구를 과학 문제에 맞게 고르는 일에 집중한다. 데이터가 어떤 기하 구조를 가지는지, 공간 좌표가 있을 때 회전·이동에 등변인 층을 어떻게 만드는지, 대칭을 넣은 효과를 어떻게 실험으로 보이는지, 그리고 문제의 대칭이 깨졌을 때 무엇을 해야 하는지다.

## 학습 목표

- 과학 데이터에 맞는 표현과 등변 모델을 선택한다
- E(n) 등변 message passing을 구현하고 등변성을 검증한다
- 대칭 제약의 효과를 실험으로 검증한다

## 기하 데이터의 표현

**왜 필요한가.** 모델에 대칭을 넣으려면 먼저 데이터가 어떤 공간 위에 있고 어떤 변환에 대해 대칭인지를 정해야 한다. 같은 분자도 원자 좌표의 점구름, 결합의 그래프, 전자 밀도의 3차원 격자로 표현할 수 있고, 표현마다 자연스러운 대칭과 맞는 모델이 다르다.

| 표현 | 과학 예 | 자연스러운 대칭 | 대표 모델 |
|---|---|---|---|
| 점구름 | 원자 좌표, 입자 시뮬레이션, 단백질 구조 | 회전·반사·평행이동 $E(3)$, 순열 | 등변 GNN(EGNN, 구면 조화함수 기반 모델) |
| 그래프 | 분자의 결합, 센서망, 반응망 | 순열 | GNN(18.6장) |
| 격자 | 영상, 기후·유체의 격자장, 전자 밀도 | 평행이동(경계 제외), 90° 회전·반사 | CNN, 군 등변 CNN |
| 메시 | 유한요소·유한체적 시뮬레이션 | 순열, 물리 공간의 $E(3)$ | 메시 그래프 네트워크 |
| 구면 | 지구 전체의 기상장, 천구 관측 | 구의 회전 $SO(3)$ | 구면 CNN, 구면 메시 위의 GNN |

**점구름과 반경 그래프.** 원자나 입자의 좌표 $\mathbf{x}_i \in \mathbb{R}^3$과 종류·전하 같은 특징 $\mathbf{h}_i$의 모음이 점구름이다. 상호작용이 거리에 따라 빠르게 약해지면, 거리가 차단 반경 $r_c$보다 가까운 쌍만 잇는 **반경 그래프**를 만들어 message passing을 한다(18.6장의 그래프 구성). 거리는 회전·이동에 불변이므로 반경 그래프의 연결 구조도 불변이다. 결정처럼 주기 경계 조건이 있으면 이웃 셀의 상(image)까지 포함해 거리를 계산한다.

**무엇을 불변으로, 무엇을 등변으로.** 7.10장의 원칙대로 예측 대상이 정한다. 에너지·쌍극자 크기·결합 에너지는 스칼라(불변), 원자에 걸리는 힘과 쌍극자 벡터는 벡터(등변), 분극률과 응력은 2차 텐서($\mathbf{R}\mathbf{T}\mathbf{R}^\top$로 변환)다. 원자 $N$개의 배치는 $3N$개의 좌표로 쓰지만, 전체 회전(3)과 평행이동(3)을 빼면 에너지를 정하는 내부 자유도는 $3N - 6$개다(일직선 분자는 $3N - 5$). 불변 모델의 입력은 결국 이 내부 자유도의 함수다.

## 등변 신경망

**왜 필요한가.** 좌표를 그대로 MLP에 넣으면 회전된 입력에 다른 답을 낸다(7.10장의 MLP 예). 불변 특징(거리·각)만 쓰면 불변성은 보장되지만, 힘처럼 벡터인 출력을 직접 낼 수 없다. **등변 층**은 좌표와 함께 돌고 이동하는 벡터 특징을 층 안에서 유지한다.

**정의.** 유클리드 군 $E(n)$의 원소 $g$를 회전·반사 $\mathbf{Q}$($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$)와 평행이동 $\mathbf{t}$로 쓰면, 좌표는 $\mathbf{x}_i \mapsto \mathbf{Q}\mathbf{x}_i + \mathbf{t}$로 변한다. 층 $(\mathbf{h}, \mathbf{x}) \mapsto (\mathbf{h}', \mathbf{x}')$가 모든 $g$와 노드 치환 $\mathbf{P}$에 대해

$$
\mathbf{h}'(g\mathbf{x}) = \mathbf{h}'(\mathbf{x}), \qquad \mathbf{x}'(g\mathbf{x}) = \mathbf{Q}\mathbf{x}'(\mathbf{x}) + \mathbf{t}
$$

(그리고 노드 순서를 바꾸면 출력 순서도 함께 바뀜)을 만족하면 노드 특징에는 불변, 좌표에는 등변이다.

**EGNN.** Satorras, Hoogeboom, Welling(2021)의 E(n) 등변 그래프 신경망은 18.6장에서 개념만 본 층이다. 한 층은 다음과 같다.

$$
\mathbf{m}_{ij} = \phi_e\big(\mathbf{h}_i, \mathbf{h}_j, \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2\big), \qquad
\mathbf{x}_i' = \mathbf{x}_i + C\sum_{j \neq i} (\mathbf{x}_i - \mathbf{x}_j)\,\phi_x(\mathbf{m}_{ij}), \qquad
\mathbf{h}_i' = \phi_h\Big(\mathbf{h}_i, \sum_{j \neq i} \mathbf{m}_{ij}\Big).
$$

$\phi_e$, $\phi_x$, $\phi_h$는 MLP, $C$는 정규화 상수(예: $1/(N-1)$)다. 등변성은 두 줄로 확인된다. 거리 제곱은 $\lVert (\mathbf{Q}\mathbf{x}_i + \mathbf{t}) - (\mathbf{Q}\mathbf{x}_j + \mathbf{t}) \rVert^2 = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2$로 불변이므로 메시지 $\mathbf{m}_{ij}$와 $\mathbf{h}'$이 불변이다. 상대 벡터는 $\mathbf{Q}(\mathbf{x}_i - \mathbf{x}_j)$로 회전만 따라가고(평행이동은 상쇄), 불변 스칼라 $\phi_x(\mathbf{m}_{ij})$를 곱해 더하므로 $\mathbf{x}'$은 $\mathbf{Q}\mathbf{x}' + \mathbf{t}$가 된다. 핵심은 **좌표를 MLP에 직접 넣지 않는 것**이다. 좌표는 불변량(거리)으로만 MLP에 들어가고, 출력 쪽에서는 등변량(상대 벡터)의 계수로만 쓰인다.

**코드.** 학습되지 않은 무작위 가중치로도 등변성은 구조에서 나오므로 검증할 수 있다. 노드 6개의 입력을 무작위 회전·반사·평행이동하고 노드 순서까지 섞은 뒤, 두 경로의 출력을 비교한다.

```python {exec="browser" session="geometric"}
import numpy as np

rng = np.random.default_rng(0)
N, D_H, D_M = 6, 4, 8                                    # 노드 수, 노드 특징 차원, 메시지 차원
def mlp(d_in, d_out):                                    # 무작위 가중치의 작은 MLP (학습은 생략)
    W1, W2 = rng.normal(size=(d_in, 16)) / np.sqrt(d_in), rng.normal(size=(16, d_out)) / 4
    return lambda z: np.tanh(z @ W1) @ W2
phi_e, phi_x, phi_h = mlp(2 * D_H + 1, D_M), mlp(D_M, 1), mlp(D_H + D_M, D_H)

def egnn_layer(h, x, r_cut=2.5):                         # h: (N, D_H) 불변 특징, x: (N, 3) 좌표
    diff = x[:, None, :] - x[None, :, :]                 # (N, N, 3) x_i - x_j: 회전에 등변
    d2 = np.sum(diff**2, axis=-1, keepdims=True)         # (N, N, 1) 거리 제곱: 불변
    adj = (d2 < r_cut**2) & ~np.eye(N, dtype=bool)[..., None]          # (N, N, 1) 반경 그래프
    pair = np.concatenate([np.repeat(h[:, None], N, 1), np.repeat(h[None], N, 0), d2], -1)
    m = phi_e(pair) * adj                                # (N, N, D_M) 메시지: 불변량으로만 만든다
    x_new = x + np.sum(diff * phi_x(m) * adj, axis=1) / (N - 1)        # 상대 벡터 × 불변 스칼라
    h_new = h + phi_h(np.concatenate([h, m.sum(axis=1)], -1))
    return h_new, x_new

def rotation(q):                                         # quaternion (4,) → 회전 행렬 (3, 3)
    w, a, b, c = q / np.linalg.norm(q)
    return np.array([[1 - 2*(b*b + c*c), 2*(a*b - c*w), 2*(a*c + b*w)],
                     [2*(a*b + c*w), 1 - 2*(a*a + c*c), 2*(b*c - a*w)],
                     [2*(a*c - b*w), 2*(b*c + a*w), 1 - 2*(a*a + b*b)]])
h, x = rng.normal(size=(N, D_H)), rng.normal(size=(N, 3))
Q = rotation(rng.normal(size=4)) @ np.diag([-1.0, 1, 1])  # 회전 + 반사 (3, 3)
t, P = rng.normal(size=3), np.eye(N)[rng.permutation(N)]  # 평행이동 (3,), 치환 행렬 (N, N)
h1, x1 = egnn_layer(h, x)
h2, x2 = egnn_layer(P @ h, P @ x @ Q.T + t)               # 순서를 바꾸고 회전·반사·이동한 입력
print("h' 불변 (순서만 함께 바뀜) :", np.allclose(h2, P @ h1))
print("x' 등변 (x' -> Q x' + t)  :", np.allclose(x2, P @ x1 @ Q.T + t))
```

```output
h' 불변 (순서만 함께 바뀜) : True
x' 등변 (x' -> Q x' + t)  : True
```

등변성 검증은 모델을 만들 때마다 단위 테스트로 넣는다. 구현 실수(좌표를 실수로 $\phi_e$에 넣거나, 상대 벡터 대신 절대 좌표를 곱하는 것)는 학습 손실로는 잘 드러나지 않지만 이 검사에서는 바로 `False`가 된다. 반사까지 포함한 $\mathbf{Q}$로 검사하는 것도 중요하다. EGNN은 반사에도 등변이므로 거울상 분자를 구별하지 못한다(아래 "대칭이 깨질 때").

::quiz{id="egnn-message-inputs"}

**더 풍부한 기하 특징.** EGNN의 메시지는 거리만 보므로 결합각·이면각 같은 방향 정보는 여러 층의 좌표 갱신을 거쳐 간접적으로만 들어간다. 원자 간 퍼텐셜처럼 방향 정보가 중요한 문제에서는 특징을 스칼라($\ell = 0$)뿐 아니라 벡터($\ell = 1$), 2차 텐서($\ell = 2$) 등으로 나누어 들고 다니는 모델을 쓴다. 상대 벡터의 방향을 **구면 조화함수**(spherical harmonics) $Y_\ell^m$로 전개하고, 차수가 다른 특징끼리는 군 표현 이론이 허용하는 방식(텐서곱과 Clebsch–Gordan 계수)으로만 결합해 등변성을 지킨다. NequIP(Batzner 외, 2022)는 이런 $E(3)$ 등변 GNN으로 적은 학습 데이터에서도 정확한 원자 간 퍼텐셜을 얻었다고 보고했다. 대가는 계산 비용이다. 높은 차수 $\ell$의 텐서곱은 차수가 오를수록 비용이 빠르게 늘어나, 표현력과 속도 사이의 선택이 된다.

**힘은 에너지의 gradient로.** 원자 간 퍼텐셜에서는 힘을 등변 출력으로 따로 예측하기보다, 불변 에너지 모델 $E_\theta(\mathbf{x})$를 만들고 $\mathbf{F}_i = -\partial E_\theta/\partial \mathbf{x}_i$를 자동미분(8.4장)으로 구하는 경우가 많다. 불변 함수의 gradient는 자동으로 등변이고(7.10장), 힘이 퍼텐셜의 gradient이므로 에너지 보존(보존력)도 구조적으로 보장된다. 분자 동역학을 오래 돌릴 때 이 차이가 크다.

## 격자·메시·구면 위의 모델

**격자: 군 등변 CNN.** CNN은 가중치 공유로 평행이동에 등변이다(18.2장). 회전에도 등변이게 하려면 필터를 회전시킨 사본들로 합성곱하고, 출력에 "어느 회전에 반응했는가"라는 축을 하나 더 둔다. Cohen과 Welling(2016)의 군 등변 CNN(G-CNN)은 격자와 맞는 90° 회전·반사 군에서 이를 정확히 구현했다. 연속 회전은 격자와 맞지 않아 근사가 필요하다. 현미경 영상, 천체 영상처럼 "위쪽"이 정해지지 않은 영상에 쓴다.

**메시: 메시 그래프 네트워크.** 유한요소 시뮬레이션의 메시는 불규칙해서 CNN을 쓸 수 없다. 메시의 절점을 노드, 요소의 변을 간선으로 둔 그래프에서 message passing으로 다음 시간 단계를 예측한다. Pfaff 외(2021)의 MeshGraphNets는 간선 특징에 절점 사이의 **상대 위치**를 넣어 평행이동 불변성을 얻었고, 공기역학·구조역학·천 시뮬레이션에서 학습에 쓴 시뮬레이터보다 한두 자릿수 빠르게 예측한다고 보고했다. 메시 위 모델은 24.4장의 neural operator와 함께 26.2장(대리모델링)에서 다시 다룬다.

**구면.** 지구 전체의 기상장을 위경도 격자에 놓으면 극 근처에서 격자 간격이 0으로 줄어 평면 CNN의 가정(어디서나 같은 간격)이 깨진다. 구의 회전에 등변인 구면 합성곱이나, 구를 거의 균일하게 덮는 정이십면체 메시 위의 그래프 신경망을 쓴다. 다만 지구 시스템에서는 자전축·지형·해륙 분포 때문에 구의 회전 대칭이 정확히 성립하지 않는다는 점도 기억한다.

:::note[분야 예시]
- **분자·재료**: 원자 간 퍼텐셜은 $E(3)$ 불변 에너지와 등변 힘을 예측한다. 결정은 주기 경계 조건 때문에 연속 회전 대칭 대신 격자의 이산 대칭을 가진다(28.3장).
- **유체·구조**: 메시 그래프 네트워크는 절점의 순서와 메시의 위치에 무관하게 같은 국소 규칙을 배운다. 중력이나 유입 경계가 있으면 그 방향은 입력으로 명시한다(아래 "대칭이 깨질 때").
- **기후**: 전구 모델은 구면 위에서 동작해야 하지만, 지형과 자전 때문에 위도·경도에 따라 다른 동역학을 가진다. 대칭은 근사적이다(27.2장).
:::

## 대칭 제약의 효과

**왜 필요한가.** "등변 모델이 더 낫다"는 주장은 무엇이 어떻게 나은지 측정할 때 의미가 있다. 측정할 것은 세 가지다. 같은 성능에 필요한 **데이터 양**(데이터 효율), 학습 때 보지 못한 방향·위치에서의 **일반화**, 그리고 학습·추론의 **계산 비용**이다.

**실험 설계.** 대칭의 효과를 보이려면 학습 분포가 대칭을 모두 덮지 않아야 한다. 입자 4개가 Morse 쌍 퍼텐셜로 상호작용하는 계의 에너지를 예측한다. 학습 구조는 $x$축으로 길게 정렬된 분포에서 뽑는다(실제 데이터셋도 주축을 맞춰 저장하거나 특정 방향으로 성장한 구조만 담는 일이 흔하다). 시험은 같은 방향의 구조와 무작위로 회전한 구조 두 벌로 한다. 모델은 같은 커널 ridge 회귀로 고정하고 입력 표현만 바꾼다. 좌표를 그대로 넣기, 좌표에 회전 증강(학습 구조마다 무작위 회전 2개 추가), 정렬한 쌍별 거리(회전·반사·평행이동·순열에 정확히 불변)다.

```python {exec="browser" session="geometric"}
iu = np.triu_indices(4, 1)                               # 입자 4개의 쌍 6개
def energy(X):                                           # X: (B, 4, 3) → (B,) Morse 쌍 퍼텐셜의 합
    d = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)[:, iu[0], iu[1]]   # (B, 6)
    return np.sum((1 - np.exp(-(d - 1.0)))**2 - 1, axis=1)
def rotate_all(X, r):                                    # 표본마다 무작위 회전
    return np.stack([xb @ rotation(r.normal(size=4)).T for xb in X])
raw = lambda X: X.reshape(len(X), -1)                    # (B, 12) 좌표를 펼친 표현
inv = lambda X: np.sort(np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)[:, iu[0], iu[1]], axis=1)  # (B, 6)

def krr(F_tr, y_tr, F_te, lam=1e-3):                     # RBF 커널 ridge, 길이척도는 중앙값 휴리스틱
    sq = lambda a, b: np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T
    K = lambda a, b: np.exp(-0.5 * sq(a, b) / np.median(sq(F_tr, F_tr)))
    alpha = np.linalg.solve(K(F_tr, F_tr) + lam * np.eye(len(F_tr)), y_tr - y_tr.mean())
    return K(F_te, F_tr) @ alpha + y_tr.mean()

r = np.random.default_rng(1)
scale = np.array([1.5, 0.8, 0.4])                        # 학습 구조는 x축으로 길게 정렬되어 있다
X_pool, X_te = r.normal(size=(300, 4, 3)) * scale, r.normal(size=(300, 4, 3)) * scale
X_te_rot = rotate_all(X_te, r)
print("RMSE (같은 방향 시험셋 / 회전된 시험셋)")
for n_tr in [25, 100, 300]:
    X_tr = X_pool[:n_tr]
    X_aug = np.concatenate([X_tr, rotate_all(X_tr, r), rotate_all(X_tr, r)])   # 회전 증강 3배
    row = []
    for name, f, Xa in [("좌표", raw, X_tr), ("좌표+증강", raw, X_aug), ("불변 특징", inv, X_tr)]:
        rmse = [np.sqrt(np.mean((krr(f(Xa), energy(Xa), f(Xt)) - energy(Xt))**2)) for Xt in (X_te, X_te_rot)]
        row.append(f"{name} {rmse[0]:.2f}/{rmse[1]:.2f}")
    print(f"N = {n_tr:3d}  " + "   ".join(row))
```

```output
RMSE (같은 방향 시험셋 / 회전된 시험셋)
N =  25  좌표 1.08/1.39   좌표+증강 1.22/1.30   불변 특징 0.35/0.35
N = 100  좌표 0.89/1.08   좌표+증강 0.82/0.93   불변 특징 0.12/0.12
N = 300  좌표 0.69/1.01   좌표+증강 0.65/0.78   불변 특징 0.08/0.08
```

- **불변 특징은 같은 데이터로 오차가 3–9배 작고**, 회전된 시험셋에서도 오차가 그대로다. 대칭이 구조로 보장되어 있으므로 회전된 시험은 같은 방향의 시험과 같은 문제다.
- **좌표 표현은 회전된 시험셋에서 크게 나빠진다**(N = 300에서 0.69 → 1.01). 실습의 반복 실험에서 이 데이터의 에너지 표준편차가 약 1.0이므로, 회전된 구조에서는 평균값으로 찍는 것과 비슷하거나 그보다 못하다.
- **증강은 격차를 줄이지만 없애지 못한다.** 같은 방향과 회전된 시험의 차이가 줄었지만(0.65/0.78), 불변 특징과의 차이는 크다. 증강은 대칭을 데이터로 **권장**할 뿐이고, 좌표 표현에는 원자 번호를 바꾸는 순열 대칭도 빠져 있다.

::quiz{id="augmentation-vs-equivariance"}

**계산 비용과의 절충.** 이 실험에서는 불변 특징이 계산도 가장 싸다. 일반적으로는 그렇지 않다. 높은 차수의 등변 특징을 쓰는 모델은 같은 크기의 비등변 모델보다 층마다 계산이 많이 든다. 반대로 데이터가 충분히 많으면 비등변 모델과 증강의 격차는 줄어들 수 있다(위 표에서도 증강이 회전 시험의 격차를 줄였다). 그래서 질문은 "등변이냐 아니냐"가 아니라 "주어진 데이터 양과 계산 예산에서 어느 쪽이 같은 정확도에 싸게 도달하는가"다. 이를 답하려면 데이터 크기를 바꿔 가며 학습 곡선을 그리고, 학습·추론 시간을 함께 보고한다.

## 대칭이 깨질 때

**왜 필요한가.** 실제 문제의 대칭은 이상화된 것보다 작은 경우가 많다. 깨진 대칭을 강제하면 모델은 꼭 필요한 정보를 볼 수 없게 된다.

**대칭을 깨는 것들.**

- **외부장**: 중력, 전기장, 자기장은 공간에 특정 방향을 정한다. 남는 대칭은 장의 방향을 축으로 한 회전($SO(2)$)과 평행이동뿐이다.
- **경계 조건**: 벽, 유입구, 유한한 계산 영역은 평행이동 대칭을 깬다. 경계 근처와 내부는 다른 물리를 따른다.
- **결정·이방성 재료**: 연속 회전 대신 격자의 이산 대칭만 남는다.
- **키랄성**: 거울상 분자(광학 이성질체)는 반사로만 서로 바뀐다. 반사까지 포함한 $E(3)$에 불변인 모델(EGNN 포함)은 둘에 같은 스칼라 출력을 낸다. 두 거울상의 성질이 다른 문제(예: 키랄 환경에서의 결합)라면 반사를 뺀 $SE(3)$ 등변 모델이나 반사에 부호가 바뀌는 특징(유사스칼라)이 필요하다.

**처리 방법.** 두 가지가 있다. 하나는 **대칭 군을 줄이는 것**이다. 장의 방향 $\hat{\mathbf{z}}$를 축으로 한 회전에만 불변인 특징(거리와 함께 $z$ 방향 성분의 크기)을 쓴다. 다른 하나는 **대칭을 깨는 원인을 입력으로 넣는 것**이다. 장 벡터 $\mathbf{E}$를 등변 모델의 입력으로 함께 넣으면, 모델은 "계와 장을 **함께** 돌리면 출력도 함께 돈다"는 올바른 대칭을 유지한다. 물리적으로는 이쪽이 더 정직하다. 대칭이 사라진 것이 아니라, 장까지 포함한 전체 계에서는 대칭이 여전히 성립하기 때문이다.

**코드.** cell 2의 에너지에 $z$축 방향의 장이 만드는 항(결합이 $z$축과 평행할수록 에너지가 높다)을 더한다. 이번에는 학습·시험 구조 모두 무작위로 회전시킨다. 완전한 $E(3)$ 불변 특징과, $z$축 회전에만 불변인 특징(쌍마다 거리와 $|\Delta z|$)을 비교한다.

```python {exec="browser" session="geometric"}
def energy_field(X, kappa=1.0):                          # z축 방향의 장: 결합이 z축과 평행할수록 에너지가 높다
    d = (X[:, :, None] - X[:, None])[:, iu[0], iu[1]]    # (B, 6, 3) 쌍별 상대 벡터
    cos2 = d[..., 2]**2 / np.sum(d**2, axis=-1)          # (B, 6) z축과 이루는 각의 cos^2
    return energy(X) + kappa * np.sum(cos2, axis=1)

def inv_z(X):                                            # z축 회전·평행이동·순열에만 불변인 특징 (B, 12)
    d = (X[:, :, None] - X[:, None])[:, iu[0], iu[1]]
    dist, dz = np.linalg.norm(d, axis=-1), np.abs(d[..., 2])
    order = np.argsort(dist, axis=1)                     # 쌍을 거리 순으로 정렬하되 (거리, |dz|) 짝은 유지
    return np.concatenate([np.take_along_axis(dist, order, 1), np.take_along_axis(dz, order, 1)], axis=1)

X_tr, X_te = rotate_all(r.normal(size=(300, 4, 3)), r), rotate_all(r.normal(size=(300, 4, 3)), r)
for name, f in [("E(3) 불변 특징      ", inv), ("z축 회전 불변 특징  ", inv_z)]:
    pred = krr(f(X_tr), energy_field(X_tr), f(X_te))
    print(f"{name} RMSE {np.sqrt(np.mean((pred - energy_field(X_te))**2)):.3f}")
pred = krr(inv(X_tr), energy(X_tr), inv(X_te))
print(f"(참고) 장이 없을 때 E(3) 불변 특징 RMSE {np.sqrt(np.mean((pred - energy(X_te))**2)):.3f}")
```

```output
E(3) 불변 특징       RMSE 1.277
z축 회전 불변 특징   RMSE 0.218
(참고) 장이 없을 때 E(3) 불변 특징 RMSE 0.066
```

$E(3)$ 불변 특징은 장이 없을 때는 오차 0.066으로 정확하지만, 장이 있으면 1.277로 무너진다. 거리만 보는 특징에는 구조가 장에 대해 어느 방향을 향했는지가 들어 있지 않으므로, 아무리 데이터를 늘려도 이 오차는 줄지 않는다. 남은 대칭($z$축 회전)에만 불변인 특징은 오차를 0.218로 줄인다.

::quiz{id="field-breaks-symmetry"}

**연구에서의 의미.** 대칭을 넣기 전에 문제의 대칭을 목록으로 적는다. 연속 대칭인지 이산 대칭인지, 반사가 포함되는지, 경계·외부장·재료가 무엇을 깨는지. 그리고 대칭 제약을 건 모델과 건 제약을 푼(또는 줄인) 모델을 함께 학습해 비교한다. 제약을 건 모델의 오차가 데이터를 늘려도 일정한 바닥에서 멈추면 깨진 대칭을 의심한다.

## 실습

같은 입자 데이터에 대칭을 넣은 모델과 넣지 않은 모델을 학습해 데이터 크기와 회전된 시험셋에서의 성능을 비교한다. 코드는 `labs/scientific-ml/geometric-models/equivariance_experiment.py`다. cell 2의 실험을 데이터 시드 5개, 학습 크기 25–400으로 반복하고 시험 구조를 500개로 늘렸다. 집필 시점의 실행 결과는 다음과 같다.

```output
시드 5개 평균 RMSE (같은 방향 / 회전), 시험 구조 500개, 에너지 표준편차 1.00
N                    좌표             좌표+증강             불변 특징
25         1.02 /  1.38      0.92 /  0.94      0.33 /  0.33
50         0.84 /  1.23      0.81 /  0.83      0.24 /  0.24
100        0.75 /  1.14      0.72 /  0.76      0.18 /  0.18
200        0.78 /  1.37      0.65 /  0.77      0.12 /  0.12
400        0.66 /  1.13      0.58 /  0.66      0.08 /  0.08
```

- **데이터 효율**: 불변 특징은 25개로 학습해도(0.33) 좌표 표현이 400개로 학습한 것(0.66)보다 정확하다.
- **회전 일반화**: 좌표 표현은 회전된 시험셋에서 에너지 표준편차(1.00)보다 큰 오차를 낸다. 같은 방향의 시험셋만 보면 이 실패가 보이지 않는다.
- **증강**은 회전 시험의 오차를 크게 줄이지만, 데이터 효율은 불변 특징을 따라가지 못한다.

과제는 이 비교를 학습되는 등변 모델로 옮기는 것이다. cell 1의 `egnn_layer` 두세 층 뒤에 노드 특징의 합을 읽어 에너지를 내는 모델과, 좌표를 펼쳐 넣는 같은 크기의 MLP를 PyTorch로 학습해(실습 환경의 torch 그룹), 위와 같은 표를 만든다. 학습 시간도 함께 보고한다. 이어서 연습문제로 EGNN의 좌표 갱신을 구현하고 등변성 테스트를 통과시킨다.

::exercise{id="egnn-coordinate-update"}

## 실패 사례

- **대칭이 깨진 문제에 강한 등변 제약을 거는 것.** 외부장·경계·키랄성이 있는 문제에 $E(3)$ 불변 모델을 쓰면, 데이터를 늘려도 줄지 않는 오차 바닥이 생긴다(cell 3의 1.277). 남은 대칭으로 군을 줄이거나, 대칭을 깨는 원인을 입력으로 넣는다.
- **회전된 테스트셋 없이 등변성의 효과를 주장하는 것.** 학습과 같은 방향의 시험셋에서는 비등변 모델의 약점이 드러나지 않는다(cell 2의 좌표 표현). 등변성의 효과를 주장하려면 회전·반사·순열을 적용한 시험셋과 데이터 크기별 학습 곡선을 함께 보고한다.
- **등변성을 수치로 검증하지 않는 것.** 구현 실수 하나로 등변성이 깨져도 학습 손실은 정상처럼 보인다. cell 1처럼 무작위 변환에 대한 검사를 단위 테스트로 둔다. 단, 부동소수점 때문에 `allclose`로 비교한다.
- **반사 대칭을 생각하지 않는 것.** 거울상의 성질이 다른 문제에서 반사까지 포함한 불변 모델을 쓰면 두 거울상을 구별하지 못한다.

## 통과 기준

- 문제 하나의 대칭(연속·이산, 반사 포함 여부, 깨는 요인)을 목록으로 적고 그에 맞는 표현과 모델을 고른 근거를 쓴다.
- 등변 제약의 효과를 데이터 크기별 학습 곡선과 회전된 시험셋 성능으로 보인다.
- 연습문제 `egnn-coordinate-update`의 테스트를 통과한다.

## 참고 문헌

- Batzner 외 (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453. — 구면 조화함수 특징을 쓰는 E(3) 등변 원자 간 퍼텐셜(NequIP)
- Cohen, Welling (2016). Group equivariant convolutional networks. *ICML 2016*, PMLR 48, 2990–2999. — 회전·반사 군에 등변인 합성곱
- Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia (2021). Learning mesh-based simulation with graph networks. *ICLR 2021*. — 메시 위 message passing으로 시뮬레이션을 학습하는 MeshGraphNets
- Satorras, Hoogeboom, Welling (2021). E(n) equivariant graph neural networks. *ICML 2021*, PMLR 139, 9323–9332. — 거리와 상대 벡터로 만드는 E(n) 등변 message passing
