---
id: linear-algebra/svd-and-pca
title: "SVD·PCA·저랭크 근사"
title_en: "SVD, PCA, and Low-Rank Approximation"
tier: core
status: draft
prerequisites: [linear-algebra/eigen-and-symmetric]
---

# SVD·PCA·저랭크 근사

7.5장의 스펙트럴 정리는 대칭행렬에만 적용된다. 그런데 연구에서 가장 자주 만나는 행렬은 대칭이 아니고 정사각도 아니다. 샘플 1000개 × 특징 50개의 데이터 행렬, 출력 512 × 입력 768의 가중치 행렬, 격자점 10만 개 × 시각 200개의 시뮬레이션 snapshot이 그렇다. **특이값 분해**(SVD, singular value decomposition)는 이런 임의의 행렬에도 서로 직교하는 축을 찾아 준다. 모든 행렬은 회전, 축별 늘이기, 회전의 세 단계로 분해된다.

SVD는 이 부의 앞 장들을 하나로 묶는다. rank와 네 부분공간(7.3장), 노름과 pseudoinverse(7.4장), 대칭행렬의 고유분해(7.5장)가 모두 SVD의 성분으로 읽힌다. 그리고 SVD를 자르면 주어진 rank에서 가장 좋은 근사가 되며, 데이터 행렬에 적용하면 PCA가 된다.

## 학습 목표

- SVD를 기하적으로 해석한다
- 저랭크 근사의 최적성을 이해한다
- PCA를 SVD로 구현한다

## SVD

**정의.** 모든 $m \times n$ 실행렬 $\mathbf{A}$는

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top
$$

로 분해된다. $\mathbf{U}$($m \times m$)와 $\mathbf{V}$($n \times n$)는 직교행렬이고, $\boldsymbol{\Sigma}$($m \times n$)는 대각에 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$을 놓고 나머지가 0인 행렬이다. $\sigma_i$를 **특이값**(singular value), $\mathbf{U}$의 열 $\mathbf{u}_i$를 왼쪽 특이벡터, $\mathbf{V}$의 열 $\mathbf{v}_i$를 오른쪽 특이벡터라 한다. 0이 아닌 특이값이 $r$개이면 처음 $r$개 항만 남긴 **thin SVD** $\mathbf{A} = \mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^\top$로 충분하다. 열마다 쓰면

$$
\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i, \qquad \mathbf{A}^\top\mathbf{u}_i = \sigma_i\mathbf{v}_i
$$

이다. 입력 공간의 직교하는 축 $\mathbf{v}_i$가 출력 공간의 직교하는 축 $\mathbf{u}_i$로 가면서 $\sigma_i$배로 늘어난다.

**기하: 회전–스케일–회전.** $\mathbf{A}\mathbf{x} = \mathbf{U}(\boldsymbol{\Sigma}(\mathbf{V}^\top\mathbf{x}))$를 오른쪽부터 읽는다.

1. $\mathbf{V}^\top$: 입력을 회전(또는 반사)해 축 $\mathbf{v}_i$를 좌표축에 맞춘다.
2. $\boldsymbol{\Sigma}$: 좌표축마다 $\sigma_i$배로 늘이거나 줄인다. 차원이 바뀌면 남는 축을 버리거나 0을 채운다.
3. $\mathbf{U}$: 결과를 출력 공간에서 회전해 축을 $\mathbf{u}_i$에 맞춘다.

그래서 단위원(단위구)은 반축이 $\sigma_i\mathbf{u}_i$인 타원(타원체)이 된다. 7.4장의 spectral 노름 $\lVert\mathbf{A}\rVert_2 = \sigma_1$은 "가장 많이 늘어나는 방향의 배율"이다. 아래 위젯에서 세 단계를 하나씩 재생해 본다.

::widget{name="svd-explorer"}

**고유분해와의 관계.** $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$이고 $\mathbf{A}\mathbf{A}^\top = \mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top\mathbf{U}^\top$이다. 두 행렬은 대칭 PSD이므로 스펙트럴 정리가 적용되고, 위 식이 바로 그 고유분해다. 즉 **오른쪽 특이벡터는 $\mathbf{A}^\top\mathbf{A}$의 고유벡터, 특이값은 그 고윳값의 제곱근**이다. SVD의 존재는 이 사실에서 나온다. $\mathbf{A}^\top\mathbf{A}$를 직교 대각화해 $\mathbf{v}_i$와 $\sigma_i$를 얻고, $\sigma_i > 0$인 $i$에 대해 $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$로 두면 이들이 정규직교임을 확인할 수 있다. 실제 계산에서는 $\mathbf{A}^\top\mathbf{A}$를 만들지 않는다. 7.4장에서 본 것처럼 조건수가 제곱되기 때문이다.

**네 부분공간과 pseudoinverse.** 특이벡터는 7.3장의 네 부분공간에 정규직교 basis를 한꺼번에 준다.

| 특이벡터 | span하는 부분공간 |
|---|---|
| $\mathbf{u}_1, \dots, \mathbf{u}_r$ | column space $C(\mathbf{A})$ |
| $\mathbf{u}_{r+1}, \dots, \mathbf{u}_m$ | left null space $N(\mathbf{A}^\top)$ |
| $\mathbf{v}_1, \dots, \mathbf{v}_r$ | row space $C(\mathbf{A}^\top)$ |
| $\mathbf{v}_{r+1}, \dots, \mathbf{v}_n$ | null space $N(\mathbf{A})$ |

rank는 0이 아닌 특이값의 개수이고, 수치적 rank는 허용오차보다 큰 특이값의 개수다(7.3장). 7.4장의 pseudoinverse도 SVD로 한 줄에 쓴다.

$$
\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top.
$$

$\boldsymbol{\Sigma}^+$는 $\boldsymbol{\Sigma}$를 전치하고 0이 아닌 $\sigma_i$를 $1/\sigma_i$로 바꾼 $n \times m$ 행렬이다. column space 성분은 $\sigma_i$로 나눠 되돌리고, 되돌릴 수 없는 방향(left null space)은 버린다. 그 결과는 row space 안에 놓이므로 최소 노름 해가 된다.

```python {exec="browser" session="svd"}
import numpy as np

A = np.array([[3.0, 0.0],
              [4.0, 5.0]])                               # (2, 2)
U, s, Vt = np.linalg.svd(A)                              # (2, 2), (2,), (2, 2)
print("특이값          :", np.round(s, 4))
print("U S V^T = A ?   :", np.allclose(U @ np.diag(s) @ Vt, A))
print("A v1 = s1 u1 ?  :", np.allclose(A @ Vt[0], s[0] * U[:, 0]))
print("sqrt(eig(A^T A)):", np.round(np.sqrt(np.linalg.eigvalsh(A.T @ A))[::-1], 4))

theta = np.linspace(0, 2 * np.pi, 3601)
circle = np.stack([np.cos(theta), np.sin(theta)])        # (2, 3601) 단위원
r = np.linalg.norm(A @ circle, axis=0)                   # 상의 원점 거리
print(f"단위원의 상: 최대 반지름 {r.max():.4f}, 최소 반지름 {r.min():.4f}")
A_pinv = Vt.T @ np.diag(1 / s) @ U.T                     # V S^+ U^T
print("V S^+ U^T = pinv(A) ?", np.allclose(A_pinv, np.linalg.pinv(A)))
```

```output
특이값          : [6.7082 2.2361]
U S V^T = A ?   : True
A v1 = s1 u1 ?  : True
sqrt(eig(A^T A)): [6.7082 2.2361]
단위원의 상: 최대 반지름 6.7082, 최소 반지름 2.2361
V S^+ U^T = pinv(A) ? True
```

NumPy의 `svd`는 $\mathbf{V}$가 아니라 $\mathbf{V}^\top$(`Vt`)를 돌려준다. 그래서 오른쪽 특이벡터 $\mathbf{v}_1$은 `Vt[0]`, 즉 **행**이다. 자주 틀리는 부분이다. 단위원 위 3601개 점의 상에서 원점까지의 최대·최소 거리가 두 특이값 $3\sqrt{5}$와 $\sqrt{5}$와 일치한다.

::quiz{id="svd-geometry"}

## 저랭크 근사

**왜 필요한가.** 7.3장에서 데이터와 가중치가 근사적으로 저랭크인 경우가 많다고 했다. 그렇다면 주어진 rank $k$에서 원래 행렬에 **가장 가까운** 행렬은 무엇이고, 그 근사의 오차는 얼마일까. 이 질문에 정확한 답이 있다는 것이 SVD를 실용적으로 만든다.

**rank-1 행렬의 합.** SVD를 열과 행의 곱으로 펼치면

$$
\mathbf{A} = \sum_{i=1}^r \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top
$$

이다. 각 항 $\mathbf{u}_i\mathbf{v}_i^\top$는 rank 1인 행렬이고, 특이값이 그 항의 크기다. 앞의 $k$개만 남긴 것을 $\mathbf{A}_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$(truncated SVD)라 한다.

**정리(Eckart–Young).** spectral 노름에 대한 부분까지 포함해 Eckart–Young–Mirsky 정리라고도 한다. rank가 $k$ 이하인 모든 행렬 $\mathbf{B}$에 대해

$$
\lVert\mathbf{A} - \mathbf{B}\rVert_F \ge \lVert\mathbf{A} - \mathbf{A}_k\rVert_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}, \qquad
\lVert\mathbf{A} - \mathbf{B}\rVert_2 \ge \lVert\mathbf{A} - \mathbf{A}_k\rVert_2 = \sigma_{k+1}.
$$

즉 truncated SVD는 Frobenius 노름과 spectral 노름 모두에서 **최적의 rank-$k$ 근사**이고, 그 오차는 버린 특이값으로 정확히 계산된다. 탐욕적으로 큰 항부터 고르는 것이 전역 최적이라는 점이 핵심이다.

**에너지 비율.** $\lVert\mathbf{A}\rVert_F^2 = \sum_i \sigma_i^2$이므로, rank-$k$ 근사가 담는 비율

$$
\frac{\sum_{i \le k} \sigma_i^2}{\sum_i \sigma_i^2}
$$

를 흔히 "에너지" 비율이라 부른다. 데이터를 중심화한 경우에는 이 비율이 PCA의 설명 분산 비율과 같다(다음 절).

공간 60점 × 시각 40개의 장(field)을 만들어 확인한다. 세 개의 공간 패턴이 각자 다른 시간 변화를 하고, 작은 잡음이 섞여 있다.

```python {exec="browser" session="svd"}
rng = np.random.default_rng(0)
x, t = np.linspace(0, 1, 60), np.linspace(0, 1, 40)
X = (np.outer(np.sin(np.pi * x), np.exp(-t))
     + 0.5 * np.outer(np.sin(2 * np.pi * x), np.cos(3 * t))
     + 0.2 * np.outer(np.sin(3 * np.pi * x), t)
     + 0.01 * rng.normal(size=(60, 40)))                 # (60, 40) 세 패턴 + 잡음
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("특이값 앞 6개:", np.round(s[:6], 3))
for k in [1, 2, 3, 4]:
    Xk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]              # (60, 40) rank-k 근사
    err = np.linalg.norm(X - Xk)
    G = np.linalg.qr(rng.normal(size=(60, k)))[0]        # 무작위 k차원 부분공간
    err_rand = np.linalg.norm(X - G @ (G.T @ X))
    energy = (s[:k] ** 2).sum() / (s ** 2).sum()
    print(f"k = {k}  오차 {err:.4f}  sqrt(꼬리 s^2 합) {np.sqrt((s[k:] ** 2).sum()):.4f}  "
          f"무작위 부분공간 {err_rand:.4f}  에너지 {energy:.4%}")
```

```output
특이값 앞 6개: [23.271 11.518  0.42   0.137  0.124  0.123]
k = 1  오차 11.5352  sqrt(꼬리 s^2 합) 11.5352  무작위 부분공간 25.8801  에너지 80.2759%
k = 2  오차 0.6251  sqrt(꼬리 s^2 합) 0.6251  무작위 부분공간 24.3924  에너지 99.9421%
k = 3  오차 0.4626  sqrt(꼬리 s^2 합) 0.4626  무작위 부분공간 25.2986  에너지 99.9683%
k = 4  오차 0.4418  sqrt(꼬리 s^2 합) 0.4418  무작위 부분공간 25.1137  에너지 99.9711%
```

근사 오차가 버린 특이값으로 계산한 값과 넷째 자리까지 같다. 같은 rank의 무작위 부분공간에 사영하면 오차가 훨씬 크다. 특이값은 셋째(0.42)까지 뚜렷하고 넷째부터는 0.12–0.14 근처로 평평하다. 이 평평한 바닥이 잡음이다. 셋째 패턴은 에너지로는 0.03%에 불과해 "에너지 99.9%" 같은 기준으로는 버려진다. 에너지 비율만으로 $k$를 고르면 작지만 실제로 있는 구조를 놓칠 수 있으므로, 특이값의 감소 양상을 함께 본다.

::quiz{id="eckart-young-error"}

## PCA

**왜 필요한가.** 특징이 수십·수백 개인 데이터에서 "변동의 대부분이 몇 개의 방향에 몰려 있는가", "그 방향은 무엇인가"를 묻는 것은 탐색적 분석의 출발점이다. 주성분 분석(PCA, principal component analysis)은 이 질문에 답한다. 7.5장에서 공분산 행렬의 고유벡터가 주축이라는 것을 보았다. 이 절은 같은 답을 데이터 행렬의 SVD로 구한다.

**정의.** 데이터 행렬 $\mathbf{X} \in \mathbb{R}^{N \times D}$(행 = 샘플)에서 열 평균을 뺀 **중심화** 행렬 $\mathbf{X}_c$를 만든다. 그 thin SVD가 $\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$이면, 표본 공분산은

$$
\mathbf{C} = \frac{\mathbf{X}_c^\top\mathbf{X}_c}{N - 1} = \mathbf{V}\,\frac{\boldsymbol{\Sigma}^2}{N - 1}\,\mathbf{V}^\top
$$

이다. 이것이 $\mathbf{C}$의 고유분해이므로 다음이 성립한다.

- **주성분 방향**(loading)은 오른쪽 특이벡터 $\mathbf{v}_i$이다.
- $i$번째 방향의 분산은 $\sigma_i^2/(N - 1)$이다.
- 각 샘플의 **주성분 점수**(score)는 $\mathbf{X}_c\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$이다.

**두 관점이 같은 답을 준다.** PCA는 두 가지로 정의할 수 있다.

- **분산 최대화.** 사영한 데이터의 분산 $\mathbf{w}^\top\mathbf{C}\mathbf{w}$를 최대화하는 단위 벡터 $\mathbf{w}$를 찾는다. Rayleigh quotient(7.5장)에 의해 답은 최대 고유벡터 $\mathbf{v}_1$이다.
- **재구성 오차 최소화.** $k$차원 부분공간에 사영했다가 되돌렸을 때의 오차 $\lVert\mathbf{X}_c - \mathbf{X}_c\mathbf{W}\mathbf{W}^\top\rVert_F^2$를 최소화하는 정규직교 $\mathbf{W}$를 찾는다. Eckart–Young에 의해 답은 $\mathbf{V}_k$다.

두 관점이 일치하는 이유는 피타고라스 정리다. 각 점의 제곱 거리는 "부분공간 안의 성분 제곱 + 직교 성분 제곱"으로 나뉘고 그 합은 고정되어 있으므로, 사영된 분산을 최대화하는 것과 남는 오차를 최소화하는 것은 같은 일이다.

```python {exec="browser" session="svd"}
rng = np.random.default_rng(1)
N = 500
z = rng.normal(size=(N, 2))                               # (N, 2) 잠재 요인
W = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 1.0]])          # (2, 3)
D = z @ W + 0.1 * rng.normal(size=(N, 3)) + [1.0, 10.0, -2.0]   # (N, 3)

Dc = D - D.mean(axis=0)                                   # 중심화
U, s, Vt = np.linalg.svd(Dc, full_matrices=False)         # (N, 3), (3,), (3, 3)
var_svd = s ** 2 / (N - 1)
w, Q = np.linalg.eigh(np.cov(D, rowvar=False))            # 오름차순
print("SVD 분산 :", np.round(var_svd, 4))
print("eigh 분산:", np.round(w[::-1], 4))
print("주성분 방향 |cos|:", np.round(np.abs(np.sum(Vt * Q[:, ::-1].T, axis=1)), 6))
print("설명 분산 비율:", np.round(var_svd / var_svd.sum(), 4))
Z = Dc @ Vt.T / np.sqrt(var_svd)                          # (N, 3) whitening
print("whitening 후 공분산 = I ?", np.allclose(np.cov(Z, rowvar=False), np.eye(3)))
```

```output
SVD 분산 : [4.9071 1.7522 0.0097]
eigh 분산: [4.9071 1.7522 0.0097]
주성분 방향 |cos|: [1. 1. 1.]
설명 분산 비율: [0.7358 0.2627 0.0014]
whitening 후 공분산 = I ? True
```

잠재 요인 2개로 만든 3차원 데이터에서, 두 주성분이 분산의 99.86%를 설명하고 셋째 성분은 잡음(분산 $0.1^2 = 0.01$ 근처)만 남는다. SVD와 공분산 고유분해는 같은 분산과 방향을 준다(고유벡터의 부호는 다를 수 있어 $|\cos|$로 비교했다). SVD를 쓰는 이유는 $\mathbf{X}_c^\top\mathbf{X}_c$를 만들지 않아 조건수를 제곱하지 않기 때문이다(7.4장).

**whitening.** 점수를 각 성분의 표준편차로 나누면($\mathbf{Z} = \mathbf{X}_c\mathbf{V}\boldsymbol{\Lambda}^{-1/2}$, $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^2/(N-1)$) 공분산이 단위행렬이 된다. 특징 사이의 상관을 없애고 분산을 1로 맞추는 전처리다. 주의할 점은 분산이 아주 작은 방향(위의 셋째 성분처럼 잡음뿐인 방향)도 1로 늘려 잡음을 크게 키운다는 것이다. 보통 작은 성분은 버리거나 분모에 작은 상수를 더한다.

**중심화.** PCA의 모든 식은 $\mathbf{X}_c$, 즉 평균을 뺀 데이터에 대한 것이다. 평균을 빼지 않으면 무슨 일이 생기는지 같은 데이터로 본다.

```python {exec="browser" session="svd"}
_, s_raw, Vt_raw = np.linalg.svd(D, full_matrices=False)  # 평균을 빼지 않음
mean_dir = D.mean(axis=0) / np.linalg.norm(D.mean(axis=0))   # (3,)
print("|cos(중심화 없는 제1방향, 평균 방향)| =", round(abs(Vt_raw[0] @ mean_dir), 4))
print("|cos(중심화한 제1방향,   평균 방향)| =", round(abs(Vt[0] @ mean_dir), 4))
print("중심화 없이 '분산' 비율:", np.round(s_raw ** 2 / (s_raw ** 2).sum(), 4))
```

```output
|cos(중심화 없는 제1방향, 평균 방향)| = 1.0
|cos(중심화한 제1방향,   평균 방향)| = 0.0657
중심화 없이 '분산' 비율: [0.9459 0.0437 0.0104]
```

중심화하지 않은 첫 방향은 데이터의 변동이 아니라 **원점에서 데이터 구름의 중심을 가리키는 방향**이다. 그리고 그 방향이 "분산"의 95%를 설명하는 것처럼 보인다. 실제 변동의 주축과는 거의 직교한다($|\cos| = 0.066$). 원점에서 먼 데이터일수록 이 착시가 심하다.

::quiz{id="pca-centering"}

## 계산

**비용.** $m \times n$ 행렬($m \ge n$)의 전체 SVD는 대략 $mn^2$에 비례하는 연산이 필요하다. 행렬이 수만 × 수만이면 계산도 메모리도 감당하기 어렵다. 그런데 대부분의 응용은 앞의 $k$개 특이값과 특이벡터만 필요하다.

**truncated SVD.** 앞의 $k$개만 반복법으로 구하는 방법이 있다. 7.5장의 거듭제곱법을 블록으로 확장하거나 Krylov 부분공간 방법(Lanczos 등)을 쓴다. 이런 방법은 행렬–벡터 곱만 필요하므로 희소 행렬이나 명시적으로 저장하지 않은 연산자에도 적용된다. 라이브러리 예로 `scipy.sparse.linalg.svds`가 있다(집필 시점 예시. 인자와 기본 알고리즘은 사용하는 버전의 문서를 확인한다).

**randomized SVD.** 무작위 벡터 몇 개를 행렬에 곱하면, 그 결과는 높은 확률로 앞의 특이벡터 방향을 잘 잡아낸다. 이 관찰을 이용하는 절차는 다음과 같다.

1. $n \times (k + p)$ 무작위 행렬 $\boldsymbol{\Omega}$를 만든다($p$는 5–10 정도의 여유분, oversampling).
2. $\mathbf{Y} = \mathbf{A}\boldsymbol{\Omega}$를 계산한다. 특이값이 천천히 감소하면 $\mathbf{Y} = (\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}\boldsymbol{\Omega}$로 거듭제곱을 $q$번 섞어 큰 특이값 방향을 강조한다.
3. $\mathbf{Y}$의 QR 분해로 정규직교 basis $\mathbf{Q}$를 얻는다. $\mathbf{A} \approx \mathbf{Q}\mathbf{Q}^\top\mathbf{A}$이다.
4. 작은 행렬 $\mathbf{B} = \mathbf{Q}^\top\mathbf{A}$($(k + p) \times n$)의 SVD $\mathbf{B} = \tilde{\mathbf{U}}\boldsymbol{\Sigma}\mathbf{V}^\top$를 구하고, $\mathbf{U} = \mathbf{Q}\tilde{\mathbf{U}}$로 되돌린다.

```python {exec="browser" session="svd"}
rng = np.random.default_rng(2)
m, n, k, p = 1000, 200, 5, 10
Ua = np.linalg.qr(rng.normal(size=(m, n)))[0]             # (m, n)
Va = np.linalg.qr(rng.normal(size=(n, n)))[0]             # (n, n)
sv = 0.7 ** np.arange(n)                                  # 빠르게 감소하는 특이값
M = (Ua * sv) @ Va.T                                      # (m, n)

Omega = rng.normal(size=(n, k + p))                       # (n, k + p) 무작위 시험 벡터
Y = M @ (M.T @ (M @ Omega))                               # (m, k + p) 거듭제곱 1회
Qr = np.linalg.qr(Y)[0]                                   # (m, k + p) 상의 basis
s_small = np.linalg.svd(Qr.T @ M, compute_uv=False)       # (k + p, n) 작은 행렬의 SVD
print("참 특이값       :", np.round(sv[:k], 6))
print("randomized SVD  :", np.round(s_small[:k], 6))
print(f"최대 상대 오차  : {np.max(np.abs(s_small[:k] - sv[:k]) / sv[:k]):.1e}")
```

```output
참 특이값       : [1.     0.7    0.49   0.343  0.2401]
randomized SVD  : [1.     0.7    0.49   0.343  0.2401]
최대 상대 오차  : 3.0e-11
```

큰 SVD는 $15 \times 200$ 행렬의 SVD로 바뀌었다. 이 예는 특이값이 $0.7^i$로 빠르게 감소해 정확도가 매우 높다. 특이값이 천천히 감소하는 행렬에서는 $k + 1$번째 이후 성분이 섞여 들어와 오차가 커지므로, 거듭제곱 횟수 $q$나 여유분 $p$를 늘리고, 정확한 SVD와 비교해 정확도를 확인하는 습관이 필요하다.

## 활용

SVD와 PCA는 "어떤 방향이 중요하고, 어떤 방향은 버려도 되는가"를 정하는 도구로 거의 모든 분야에서 쓰인다.

- **차원 축소.** 주성분 점수 $k$개로 데이터를 요약해 시각화하거나 다음 모델의 입력으로 쓴다. 단, PCA가 고르는 방향은 분산이 큰 방향이지 예측에 유용한 방향이 아니다. 분산이 작은 방향에 타깃과 관련된 신호가 있으면 PCA로 줄이는 순간 사라진다.
- **노이즈 제거.** 신호가 저랭크이고 잡음이 모든 방향에 고르게 퍼져 있으면, 작은 특이값을 버리는 것이 잡음을 줄인다. 앞의 장(field) 예에서 rank 3 근사는 잡음 바닥을 버리고 세 패턴만 남긴다. 몇 개를 남길지는 특이값의 바닥 수준과 평가 데이터로 정한다(실습).
- **모델 압축.** 가중치 $\mathbf{W} \in \mathbb{R}^{m \times n}$을 $\mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$로 바꾸면 파라미터가 $mn$에서 $k(m + n)$으로 준다. 층 하나를 폭 $k$의 두 층으로 바꾸는 것과 같다. Eckart–Young은 가중치 공간에서의 최적성을 말할 뿐, 모델 출력의 정확도가 가장 덜 떨어진다는 보장은 아니다. 압축 뒤에는 반드시 과제 성능을 다시 잰다.
- **표현 분석.** 신경망의 중간 표현(embedding)을 샘플 × 차원 행렬로 모아 특이값 스펙트럼을 보면, 표현이 실제로 몇 개의 방향을 쓰는지(유효 차원) 알 수 있다. 두 모델이나 두 층의 표현을 각각의 주성분 부분공간으로 비교하는 방법도 이 위에 세워진다(18.8장).

:::note[분야 예시]
단일세포 RNA 시퀀싱 분석에서는 세포 수천–수만 개 × 유전자 수천 개의 발현 행렬을 정규화·로그 변환한 뒤 PCA로 수십 차원으로 줄이고, 그 공간에서 이웃 그래프를 만들어 군집화와 2차원 시각화를 한다. 이때 정규화와 로그 변환, 고분산 유전자 선택이 주성분을 크게 바꾼다. 같은 데이터라도 전처리에 따라 "주된 변동"이 세포 유형이 될 수도, 시퀀싱 깊이 같은 기술적 요인이 될 수도 있다. PCA는 분산을 찾을 뿐, 그 분산이 생물학적 의미를 갖는지는 알려 주지 않는다.
:::

## 실습

직접 구현한 SVD 기반 PCA를 다른 경로의 계산(공분산 행렬의 고유분해, `np.linalg.eigh`)과 비교하고, 주성분 수에 따른 재구성 오차를 분석한다. 코드는 `labs/linear-algebra/svd-and-pca/pca_lab.py`다. 1부는 세 성분의 흡광 피크(450·530·620 nm 근처의 가우시안 모양)를 무작위 농도로 섞고 잡음(표준편차 0.01)을 더한 합성 스펙트럼이다. 학습 데이터로 주성분을 구하고, 학습에 쓰지 않은 평가 데이터도 같은 주성분으로 재구성한다. 2부는 단위가 다른 두 특징(온도 K, 압력 Pa)에서 표준화의 효과를 본다. 집필 시점의 실행 결과는 다음과 같다.

```output
[1부] 합성 스펙트럼 200개(학습), 100개(평가), 파장 100개
  SVD 분산과 eigh 고윳값의 최대 상대 차이: 4.6e-14
  주성분 방향 |cos|의 최솟값(앞 3개): 1.000000
   k   설명 분산 누적   학습 RMSE   평가 RMSE
   1       38.0046%     0.12499     0.12686
   2       72.2479%     0.08362     0.08605
   3       99.6259%     0.00971     0.01000
   4       99.6366%     0.00957     0.00995
   6       99.6565%     0.00930     0.00985
  10       99.6930%     0.00879     0.00964
[2부] 온도(K)와 압력(Pa)
  표준화 없음: 제1주성분 [2.e-04 1.e+00]  설명 분산 비율 99.9995%
  표준화 후  : 제1주성분 [0.7071 0.7071]  설명 분산 비율 54.7017%
```

결과를 읽는 법은 세 가지다.

- **두 계산 경로가 반올림 오차 수준에서 일치한다.** 구현을 검증할 때는 이처럼 독립적인 경로로 같은 양을 계산해 비교한다. 성분의 부호는 경로마다 다를 수 있으므로 $|\cos|$로 비교한다.
- **$k = 3$에서 재구성 오차가 잡음 수준(0.01)으로 떨어진다.** 성분 3개로 만든 데이터이므로 예상대로다. $k$를 더 늘리면 학습 RMSE는 계속 줄지만(0.00971 → 0.00879), 평가 RMSE는 거의 줄지 않는다(0.01000 → 0.00964). 늘어난 주성분은 학습 데이터의 잡음을 외운 것이다. 주성분 수는 학습 데이터의 설명 분산이 아니라 **평가 데이터의 재구성 오차**로 고른다.
- **단위가 다른 특징을 그대로 넣으면 PCA는 숫자가 큰 특징을 고른다.** 압력의 표준편차(약 2000 Pa)가 온도(약 5 K)보다 수백 배 커서, 표준화하지 않으면 제1주성분이 사실상 압력 축이고 설명 분산이 99.9995%로 보인다. 표준화하면 두 특징이 같은 무게를 갖고, 설명 분산 54.7%는 두 특징의 상관이 약하다는 것을 정직하게 보여 준다(표준화한 2변수에서 제1성분의 설명 분산은 $(1 + |\rho|)/2$이다).

이어서 두 연습문제를 푼다.

::exercise{id="low-rank-approx"}

::exercise{id="pca-svd"}

## 실패 사례

- **중심화 없이 PCA를 수행하는 것.** 첫 주성분이 데이터의 평균 방향을 가리키고, 그것이 분산의 대부분을 설명하는 것처럼 보인다. `np.linalg.svd(X)`를 바로 부르는 코드는 거의 항상 이 실수다. 라이브러리 PCA는 보통 내부에서 중심화하지만, 직접 SVD를 쓰거나 학습·평가 데이터를 따로 변환할 때는 **학습 데이터의 평균**을 빼야 한다(평가 데이터의 평균을 쓰면 누수다, 15.4장).
- **단위가 다른 특징을 표준화 없이 PCA하는 것.** 주성분이 물리적 중요도가 아니라 단위 선택(Pa냐 kPa냐)을 반영한다. 단위가 다르면 표준화하고, 같은 단위의 측정(같은 스펙트럼의 파장들)이라면 표준화가 오히려 잡음만 있는 채널을 키울 수 있으므로 상황에 맞게 고른다. 어느 쪽을 택했는지 보고한다.
- **`Vt`의 행과 열을 혼동하는 것.** NumPy의 `svd`가 돌려주는 것은 $\mathbf{V}^\top$이므로 주성분 방향은 `Vt[i]`(행)이다. `Vt[:, i]`를 쓰면 shape은 맞아서 오류 없이 엉뚱한 결과가 나온다. 구현을 공분산 고유분해 같은 독립 경로와 비교해 검증한다.

## 통과 기준

- 2 × 2 행렬의 SVD를 회전–스케일–회전으로 해석하고, 특이값이 단위원의 상(타원)의 반축 길이임을 계산으로 확인한다.
- truncated SVD의 오차가 버린 특이값으로 계산한 값과 같음을 확인하고(연습문제 `low-rank-approx`), SVD로 pseudoinverse를 구성해 `np.linalg.pinv`와 비교한다.
- 중심화한 데이터의 SVD로 PCA를 구현해 공분산 고유분해와 비교하고(연습문제 `pca-svd`), 평가 데이터의 재구성 오차로 주성분 수를 정한다.
