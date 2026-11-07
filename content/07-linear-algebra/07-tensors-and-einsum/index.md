---
id: linear-algebra/tensors-and-einsum
title: "텐서와 einsum"
title_en: "Tensors and einsum"
tier: core
status: draft
prerequisites: [linear-algebra/svd-and-pca]
---

# 텐서와 einsum

신경망 코드의 버그 가운데 가장 흔하고 가장 조용한 것은 축(axis)을 잘못 다루는 버그다. shape이 맞으면 코드는 오류 없이 돌아가고, 손실도 내려간다. 다만 batch의 서로 다른 샘플이 섞였거나, 시간 축과 채널 축이 뒤바뀌었거나, 평균을 엉뚱한 축으로 냈을 뿐이다. 이런 버그는 결과가 "조금 나쁜" 형태로만 드러나서 찾기 어렵다.

이 장은 다차원 배열의 연산을 **인덱스로 쓰는 습관**을 기른다. 모든 축에 글자를 붙이고, 어느 축이 남고 어느 축이 합해지는지를 식으로 적으면 연산의 의미가 shape과 함께 고정된다. NumPy·PyTorch·JAX가 모두 지원하는 `einsum`은 이 인덱스 표기를 그대로 코드로 옮긴 함수다. attention, bilinear layer, batch 공분산처럼 행렬 곱 기호로는 한 줄에 쓰기 어려운 연산도 einsum으로는 한 줄이 된다.

## 학습 목표

- 다차원 배열 연산을 인덱스 표기로 표현한다
- einsum으로 복잡한 연산을 작성한다
- shape 대수를 익힌다

## 다차원 배열로서의 텐서

수학과 물리학에서 텐서(tensor)는 좌표를 바꿀 때 정해진 규칙으로 변환되는 다중선형 대상을 뜻한다. 이 장과 머신러닝 코드에서 말하는 텐서는 그보다 좁은 뜻으로, **축이 여러 개인 숫자 배열**이다. 스칼라는 축이 0개, 벡터는 1개, 행렬은 2개인 텐서다. 축의 개수를 차수(order) 또는 rank라고 부르지만, 7.3장에서 본 행렬의 rank와는 다른 개념이므로 이 장에서는 "축의 개수" 또는 "차수"라고 쓴다.

배열 자체는 숫자를 담을 뿐이고 축에 의미가 있지 않다. 축의 의미는 **약속**이다. 이 책은 다음 약어를 쓴다(스타일 가이드).

| 약어 | 의미 | 예 |
|---|---|---|
| `B` | batch | 한 번에 처리하는 샘플 수 |
| `T` | time | 시계열의 시점, 문장의 토큰 위치 |
| `C` | channel | 이미지의 색 채널, 센서 채널 |
| `D` | feature | embedding 차원, 특징 수 |
| `H` | head | multi-head attention의 head 수 |
| `N` | 샘플 수 | 점구름의 점 개수, 측정 횟수 |

같은 `(32, 100, 64)` 배열이라도 `(B, T, D)`로 읽으면 "64차원 특징을 가진 길이 100의 시계열 32개"이고, `(B, N, D)`로 읽으면 "점 100개로 이루어진 점구름 32개"다. 시계열은 시점의 순서가 중요하고 점구름은 점의 순서가 의미가 없다(7.10장). 숫자는 같아도 허용되는 연산이 다르다. 그래서 코드의 모든 텐서 옆에 shape 주석을 달고, 축 이름까지 적는다.

```text
x = encoder(x)      # (B, T, D)
```

라이브러리마다 기본 축 순서가 다르다는 점도 기억한다. 이미지는 PyTorch에서 `(B, C, H, W)`, TensorFlow 기본값과 이미지 파일을 읽어 들인 NumPy 배열에서는 흔히 `(B, H, W, C)`다(여기서 `H`, `W`는 높이와 너비). 데이터를 한 도구에서 다른 도구로 옮길 때 축 순서를 확인하지 않으면 다음 절들의 버그가 생긴다.

:::note[분야 예시]
기상 재분석 자료는 흔히 `(time, level, lat, lon)` 4개의 축을 가진다. 위도·경도 축에서 평균을 내면 지역 평균이 아니라 단순 격자 평균이 되는데, 위도에 따라 격자 한 칸의 면적이 다르기 때문이다(고위도일수록 좁다). 축의 의미를 알면 "어느 축으로, 어떤 가중치로 평균을 내야 하는가"라는 질문이 자연스럽게 따라온다.
:::

## 인덱스 표기와 einsum

**인덱스 표기.** 행렬 곱 $\mathbf{C} = \mathbf{A}\mathbf{B}$를 원소로 쓰면

$$
C_{ik} = \sum_{j} A_{ij} B_{jk}
$$

이다. 이 식에는 세 종류의 인덱스가 있다. $i$와 $k$는 등식 양쪽에 모두 나타나는 **자유 인덱스**(free index)로, 결과의 축이 된다. $j$는 오른쪽에만 나타나고 합해져 사라지는 **더미 인덱스**(dummy index)다. 이렇게 공유된 인덱스를 따라 합해 축을 없애는 연산을 **축약**(contraction)이라 한다.

**아인슈타인 표기법**(Einstein summation convention)은 여기서 한 걸음 더 나간다. "한 항에 두 번 나오는 인덱스는 합한다"고 약속하고 $\sum$ 기호를 생략해 $C_{ik} = A_{ij}B_{jk}$로 쓴다. 물리학에서 오래 쓰여 온 이 표기는 합 기호가 많은 식을 짧게 만든다.

**einsum.** `np.einsum`은 이 약속을 문자열로 받는다. 화살표 왼쪽에는 입력마다 축의 글자를, 오른쪽에는 출력에 남길 축의 글자를 쓴다. 규칙은 두 가지다.

1. 여러 입력에 같은 글자가 있으면 그 축의 같은 위치끼리 곱한다.
2. 출력(화살표 오른쪽)에 없는 글자는 합해서 없앤다.

화살표를 생략하면 NumPy가 한 번만 나온 글자를 알파벳 순서로 출력에 남긴다. 이 암묵 규칙은 읽는 사람이 다시 계산해야 하므로, 이 책에서는 항상 화살표를 쓴다. 익숙한 연산이 einsum으로 어떻게 쓰이는지 확인한다.

```python {exec="browser" session="einsum"}
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 4))          # (I, J)
B = rng.normal(size=(4, 5))          # (J, K)
C = rng.normal(size=(3, 4))          # (I, J)
S = rng.normal(size=(4, 4))          # (J, J)

print("행렬 곱 ij,jk->ik :", np.allclose(np.einsum("ij,jk->ik", A, B), A @ B))
print("전치   ij->ji     :", np.allclose(np.einsum("ij->ji", A), A.T))
print("trace  ii->       :", np.isclose(np.einsum("ii->", S), np.trace(S)))
print("대각   ii->i      :", np.allclose(np.einsum("ii->i", S), np.diag(S)))
print("행 합  ij->i      :", np.allclose(np.einsum("ij->i", A), A.sum(axis=1)))
print("Frobenius ij,ij-> :", np.isclose(np.einsum("ij,ij->", A, C), np.trace(A.T @ C)))
```

```output
행렬 곱 ij,jk->ik : True
전치   ij->ji     : True
trace  ii->       : True
대각   ii->i      : True
행 합  ij->i      : True
Frobenius ij,ij-> : True
```

`"ii->"`처럼 한 입력 안에서 글자가 반복되면 대각 원소만 읽는다. `"ij,ij->"`는 같은 위치의 원소를 곱해 모두 더하므로, 7.4장의 Frobenius 내적 $\langle \mathbf{A}, \mathbf{C} \rangle_F = \operatorname{tr}(\mathbf{A}^\top\mathbf{C})$이다.

::quiz{id="einsum-frobenius"}

**batch 행렬 곱.** 자유 인덱스가 두 입력에 모두 있고 출력에도 남으면 그 축은 "같이 순회하는" batch 축이 된다. $C_{bik} = \sum_j A_{bij} B_{bjk}$는 `"bij,bjk->bik"`로, batch 항목마다 독립된 행렬 곱이다. attention이 정확히 이 구조다. 쿼리 $\mathbf{Q}$와 키 $\mathbf{K}$의 유사도 점수는

$$
s_{bts} = \frac{1}{\sqrt{D}} \sum_{d} Q_{btd} K_{bsd}
$$

이고, softmax로 마지막 축 $s$를 정규화한 가중치 $w_{bts}$로 값 $\mathbf{V}$를 섞는다: $o_{btd} = \sum_s w_{bts} V_{bsd}$. 여기서 $t$는 쿼리의 위치, $s$는 키·값의 위치다(18.4장에서 다시 다룬다).

```python {exec="browser" session="einsum"}
Bn, T, Sn, D = 2, 3, 4, 8
Q = rng.normal(size=(Bn, T, D))                           # (B, T, D)
K = rng.normal(size=(Bn, Sn, D))                          # (B, S, D)
V = rng.normal(size=(Bn, Sn, D))                          # (B, S, D)

scores = np.einsum("btd,bsd->bts", Q, K) / np.sqrt(D)    # (B, T, S)
w = np.exp(scores - scores.max(axis=-1, keepdims=True))
w = w / w.sum(axis=-1, keepdims=True)                     # (B, T, S), 마지막 축 합 1
out = np.einsum("bts,bsd->btd", w, V)                     # (B, T, D)

loop = np.zeros((Bn, T, Sn))
for b in range(Bn):
    for t in range(T):
        for s in range(Sn):
            loop[b, t, s] = Q[b, t] @ K[b, s] / np.sqrt(D)
print("scores", scores.shape, " out", out.shape)
print("loop과 일치:", np.allclose(scores, loop))
print("가중치 행 합:", w.sum(axis=-1)[0])
```

```output
scores (2, 3, 4)  out (2, 3, 8)
loop과 일치: True
가중치 행 합: [1. 1. 1.]
```

einsum 문자열 `"btd,bsd->bts"`를 읽으면 계산의 의미가 드러난다. $b$는 출력에 남으니 batch 항목끼리 섞이지 않고, $d$는 사라지니 특징 축으로 내적하며, $t$와 $s$는 모든 쌍을 만든다. 같은 계산을 `Q @ K.transpose(0, 2, 1)`로도 쓸 수 있지만, 어느 축이 어디로 갔는지는 읽는 사람이 다시 따져야 한다.

**축약 순서와 비용.** 피연산자가 셋 이상이면 결과는 같아도 어떤 순서로 축약하느냐에 따라 비용이 크게 달라진다. $1000 \times 1000$ 행렬 $\mathbf{A}, \mathbf{B}$와 벡터 $\mathbf{x}$에 대해 $(\mathbf{A}\mathbf{B})\mathbf{x}$는 약 $10^9$번, $\mathbf{A}(\mathbf{B}\mathbf{x})$는 약 $2 \times 10^6$번 곱한다. NumPy의 `np.einsum`은 기본값(`optimize=False`)에서 순서를 최적화하지 않는다. 피연산자가 셋 이상이면 `optimize=True`를 주거나 `np.einsum_path`로 축약 순서와 예상 비용을 확인한다. 함수의 기본값은 버전마다 바뀔 수 있으므로 쓰는 버전의 문서에서 확인한다.

::quiz{id="contraction-order"}

## 주요 연산

인덱스로 쓰면 이름이 다른 여러 연산이 같은 틀 안에 들어온다.

| 연산 | 인덱스 표기 | einsum | 결과 shape |
|---|---|---|---|
| 내적 | $\sum_i u_i v_i$ | `"i,i->"` | `()` |
| outer product | $u_i v_j$ | `"i,j->ij"` | `(I, J)` |
| 행렬–벡터 곱 | $\sum_j A_{ij} x_j$ | `"ij,j->i"` | `(I,)` |
| 텐서 축약 | $\sum_{j,k} T_{ijk} M_{jk}$ | `"ijk,jk->i"` | `(I,)` |
| Kronecker product | $A_{ij} B_{kl}$를 $(ik, jl)$로 묶음 | `"ij,kl->ikjl"` 후 reshape | `(IK, JL)` |

outer product $\mathbf{u}\mathbf{v}^\top$는 합하는 축이 없어 축이 늘어나는 연산이다. Kronecker product $\mathbf{A} \otimes \mathbf{B}$는 $\mathbf{A}$의 원소마다 $\mathbf{B}$ 전체를 곱해 블록으로 늘어놓은 행렬로, 4차 텐서 $A_{ij}B_{kl}$을 **행 축 $(i, k)$와 열 축 $(j, l)$로 묶어** 행렬로 펼친 것이다. 2차원 격자 위의 미분 연산자처럼 축마다 독립인 연산을 하나의 행렬로 쓸 때 나타난다(7.9장).

**전치와 reshape는 다르다.** Kronecker product의 einsum이 `"ij,kl->ikjl"`인 이유는 reshape의 동작에 있다. `reshape`는 원소를 메모리에 놓인 순서대로(NumPy 기본은 마지막 축이 가장 빨리 변하는 C 순서) 읽어 새 shape에 다시 채운다. **축을 바꾸지 않는다.** 축의 순서를 바꾸는 것은 `transpose`(또는 `np.moveaxis`)다. 행 축을 $(i, k)$로 묶으려면 reshape 전에 $i, k$가 앞에 나란히 오도록 축을 배치해야 한다.

```python {exec="browser" session="einsum"}
X = np.arange(6).reshape(2, 3)       # (2, 3)
print("X.T =\n", X.T)
print("X.reshape(3, 2) =\n", X.reshape(3, 2))

P = np.array([[1, 10], [100, 1000]])
R = np.array([[1, 2], [3, 4]])
kron = np.einsum("ij,kl->ikjl", P, R).reshape(4, 4)   # (2, 2, 2, 2) → (4, 4)
bad = np.einsum("ij,kl->ijkl", P, R).reshape(4, 4)    # 축 순서를 잘못 둔 경우
print("ikjl → Kronecker 일치:", np.array_equal(kron, np.kron(P, R)))
print("ijkl → Kronecker 일치:", np.array_equal(bad, np.kron(P, R)))
```

```output
X.T =
 [[0 3]
 [1 4]
 [2 5]]
X.reshape(3, 2) =
 [[0 1]
 [2 3]
 [4 5]]
ikjl → Kronecker 일치: True
ijkl → Kronecker 일치: False
```

두 결과 모두 shape은 `(3, 2)`이고 원소도 같은 0–5이지만 배치가 다르다. 전치는 `X[i, j]`를 `X.T[j, i]`로 보내고, reshape는 0, 1, 2, … 순서를 유지한 채 줄을 다시 바꿀 뿐이다. 잘못된 축 순서로 만든 `bad`도 shape `(4, 4)`의 그럴듯한 행렬이고 오류는 나지 않는다.

::quiz{id="reshape-vs-transpose"}

이 차이가 가장 자주 문제가 되는 곳은 multi-head attention의 head 분리다. `(B, T, H*Dh)` 배열을 `(B, H, T, Dh)`로 만들려면 먼저 `reshape(B, T, H, Dh)`로 마지막 축을 쪼갠 뒤 `transpose(0, 2, 1, 3)`로 축을 옮겨야 한다. 바로 `reshape(B, H, T, Dh)`를 하면 shape은 같지만 서로 다른 시점의 특징들이 한 head 안에 섞인다.

## shape 대수

**broadcasting 규칙.** 원소별 연산(`+`, `-`, `*`, 비교 등)에서 두 배열의 shape이 다를 때 NumPy는 다음 규칙으로 shape을 맞춘다.

1. shape을 **오른쪽 끝부터** 나란히 놓는다. 축 개수가 적은 쪽은 왼쪽에 크기 1인 축을 붙인다.
2. 축마다 두 크기가 같거나, 한쪽이 1이어야 한다. 1인 쪽은 다른 쪽 크기만큼 (복사 없이) 늘어난 것처럼 다룬다.
3. 결과 shape은 축마다 큰 쪽 크기다. 1도 아니고 같지도 않은 축이 있으면 오류다.

예를 들어 `(5, 1, 3)`과 `(4, 3)`은 `(5, 1, 3)`과 `(1, 4, 3)`으로 맞춰져 결과가 `(5, 4, 3)`이 된다. 인덱스로 쓰면 broadcasting은 "한쪽 입력에 없는 인덱스는 그 축을 따라 값이 일정하다"는 뜻이다. $d_{nmk} = x_{nk} - y_{mk}$에서 $x$에는 $m$이 없고 $y$에는 $n$이 없다. 코드로는 `x[:, None, :] - y[None, :, :]`이다.

::quiz{id="broadcast-result-shape"}

```python {exec="browser" session="einsum"}
pts = rng.normal(size=(5, 3))                          # (N, D)
ctr = rng.normal(size=(4, 3))                          # (M, D)
diff = pts[:, None, :] - ctr[None, :, :]               # (N, 1, D) - (1, M, D) → (N, M, D)
dist = np.sqrt(np.einsum("nmd,nmd->nm", diff, diff))   # (N, M)
print("dist", dist.shape)

Xb = rng.normal(size=(6, 50, 3))                       # (B, N, D)
Xc = Xb - Xb.mean(axis=1, keepdims=True)               # 샘플 축(N)으로 중심화
cov = np.einsum("bnd,bne->bde", Xc, Xc) / (50 - 1)     # (B, D, D)
ref = np.stack([np.cov(x, rowvar=False) for x in Xb])  # (B, D, D)
print("cov", cov.shape, " np.cov와 일치:", np.allclose(cov, ref))
```

```output
dist (5, 4)
cov (6, 3, 3)  np.cov와 일치: True
```

batch 공분산 $\Sigma_{bde} = \frac{1}{N-1}\sum_n \tilde{X}_{bnd}\tilde{X}_{bne}$에는 같은 feature 축이 출력에 두 번 필요하다. 그래서 두 번째 피연산자의 feature 축에는 다른 글자 $e$를 붙인다. 같은 글자 `d`를 두 번 쓰면(`"bnd,bnd->bd"`) 대각 원소인 분산만 남는다.

`keepdims=True`도 shape 대수의 도구다. `Xb.mean(axis=1)`의 shape은 `(B, D)`라서 `(B, N, D)`에서 바로 빼면 오른쪽 정렬 규칙에 따라 `(B, D)`의 `B`가 `N` 축과 짝지어진다. $B \neq N$이면 오류가 나서 다행이지만, $B = N$이면 오류 없이 엉뚱한 값을 뺀다. `keepdims=True`로 `(B, 1, D)`를 유지하면 이 위험이 사라진다.

**shape 추적 습관.** 축 버그를 줄이는 습관을 정리하면 다음과 같다.

- 모든 텐서 생성·변형 줄에 `# (B, T, D)` 같은 shape 주석을 단다.
- 축을 합하거나 평균 낼 때 `axis=1` 대신 의미를 주석으로 남긴다(`# 시간 축 평균`).
- 테스트에서 축 크기를 모두 다르게 잡는다($B = 2, T = 3, D = 5$). 크기가 같은 축이 있으면 축이 뒤바뀐 버그가 shape 오류로 드러나지 않는다.
- `assert x.shape == (B, T, D)`를 함수 경계에 둔다.

:::note[분야 예시]
여러 배터리 셀의 충방전 곡선을 `(cell, cycle, time, channel)`로 모으면 channel 축에 전압·전류·온도처럼 단위가 다른 측정값이 들어간다. 채널마다 정규화하려면 평균과 표준편차를 `(cell, cycle, time)` 축으로 계산해 shape `(1, 1, 1, channel)`로 남겨야 한다. 실수로 channel 축으로 평균을 내면 전압(V)과 온도(°C)를 더한 값이 나오는데, 이 역시 오류 없이 실행된다.
:::

## 텐서 분해 개요

:::note[심화]
이 절은 처음 읽을 때 건너뛰어도 된다. 텐서 분해는 18.8장의 표현 분석, 모델 압축, 다차원 과학 자료의 저랭크 근사에서 다시 나온다.
:::

7.6장의 SVD는 행렬을 랭크-1 행렬 $\mathbf{u}_r\mathbf{v}_r^\top$의 합으로 쓴다. 3차 이상의 텐서에서 같은 아이디어를 일반화하는 방법은 둘로 갈린다.

**CP 분해**(CANDECOMP/PARAFAC)는 텐서를 랭크-1 텐서(벡터 세 개의 outer product)의 합으로 쓴다.

$$
\mathcal{T}_{ijk} \approx \sum_{r=1}^{R} a_{ir} \, b_{jr} \, c_{kr}.
$$

인자 행렬 $\mathbf{a} \in \mathbb{R}^{I \times R}$, $\mathbf{b} \in \mathbb{R}^{J \times R}$, $\mathbf{c} \in \mathbb{R}^{K \times R}$의 원소 수는 $R(I + J + K)$로, 원래 원소 수 $IJK$보다 훨씬 적다. 인덱스 표기 그대로 einsum `"ir,jr,kr->ijk"`가 CP 텐서를 만든다.

**Tucker 분해**는 작은 핵심 텐서(core tensor) $\mathcal{G}$와 축마다의 인자 행렬로 쓴다.

$$
\mathcal{T}_{ijk} \approx \sum_{p,q,r} G_{pqr} \, a_{ip} \, b_{jq} \, c_{kr}.
$$

축마다 다른 랭크를 쓸 수 있고, 모든 축의 랭크가 같고 $\mathcal{G}$가 대각($G_{pqr} \neq 0$은 $p = q = r$일 때만)인 특수한 경우가 CP 분해다.

텐서를 행렬로 펼쳐(unfolding) 보면 두 분해를 행렬의 언어로 확인할 수 있다. mode-1 펼침은 첫 축을 행으로, 나머지 축을 열로 묶은 $(I, JK)$ 행렬이다. CP 랭크가 $R$인 텐서의 펼침 행렬은 rank가 $R$ 이하다.

```python {exec="browser" session="einsum"}
I, J, Kd, Rk = 10, 20, 30, 2
a = rng.normal(size=(I, Rk))                     # (I, R)
b = rng.normal(size=(J, Rk))                     # (J, R)
c = rng.normal(size=(Kd, Rk))                    # (K, R)
Tn = np.einsum("ir,jr,kr->ijk", a, b, c)         # (I, J, K)

unfold = Tn.reshape(I, J * Kd)                   # mode-1 펼침 (I, J*K)
print("텐서 원소 수    :", Tn.size)
print("CP 파라미터 수  :", Rk * (I + J + Kd))
print("mode-1 펼침 rank:", np.linalg.matrix_rank(unfold))
```

```output
텐서 원소 수    : 6000
CP 파라미터 수  : 120
mode-1 펼침 rank: 2
```

행렬과 다른 점도 있다. 행렬의 최적 저랭크 근사는 SVD를 잘라 얻을 수 있지만(Eckart–Young, 7.6장), 3차 이상의 텐서에서는 일반적으로 그런 닫힌 해가 없고 반복 최적화(예: 교대 최소제곱)로 근사를 찾는다. 텐서의 CP 랭크를 정확히 계산하는 것도 일반적으로 어려운 문제다.

## 실습

attention, bilinear layer, batch 공분산을 einsum 한 줄과 Python loop로 각각 구현하고 결과와 실행 시간을 비교한다. 코드는 `labs/linear-algebra/tensors-and-einsum/einsum_vs_loop.py`다. 집필 시점의 실행 결과는 다음과 같다(시간은 기계마다 다르다).

```output
attention  B=8,T=S=64,D=32  shape (8, 64, 32)  일치 True  einsum    1.78 ms  loop     75.5 ms  (loop/einsum ≈ 42배)
bilinear   B=256,I=J=16,K=8  shape (256, 8)  일치 True  einsum    0.39 ms  loop      3.8 ms  (loop/einsum ≈ 10배)
batch cov  B=32,N=200,D=6    shape (32, 6, 6)  일치 True  einsum    0.63 ms  loop     17.1 ms  (loop/einsum ≈ 27배)
```

결과를 읽는 법은 두 가지다.

- **loop 구현은 정답 확인용이다.** 인덱스 식을 그대로 옮긴 loop는 느리지만 틀리기 어렵다. 새 einsum 식을 쓸 때마다 작은 입력에서 loop와 비교하는 테스트를 남긴다.
- **속도 차이는 연산마다 다르다.** loop 안에서 이미 벡터 연산(`x[b] @ W[k] @ z[b]`)을 쓰는 bilinear는 차이가 작고, 원소마다 Python을 거치는 부분이 많을수록 차이가 커진다. 정확한 배수보다 "loop의 가장 안쪽이 Python인가"가 중요하다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="bilinear-einsum"}

::exercise{id="batch-covariance"}

## 실패 사례

- **reshape와 transpose를 혼동해 데이터를 섞는 것.** `(B, T, D)`를 `(T, B, D)`로 바꾸려고 `reshape`를 쓰거나, multi-head attention에서 head를 나눌 때 transpose를 빠뜨린다. shape이 맞아서 오류 없이 학습이 진행되고, 성능만 조금 나쁘다. 의심되면 작은 입력에 `np.arange`로 값을 채워 각 원소가 어디로 갔는지 출력해 본다.
- **축의 의미를 잃어버리는 것.** `axis=1`로 평균을 냈는데 앞 단계에서 축 순서가 바뀌어 시간 평균이 채널 평균이 된다. 크기가 같은 축($B = T$, 정사각 이미지의 $H = W$)은 축이 바뀌어도 shape 오류가 나지 않으므로 더 위험하다. 테스트에서는 모든 축 크기를 다르게 잡는다.
- **broadcasting이 의도하지 않은 축을 늘리는 것.** 길이 $N$인 예측 `(N,)`과 정답 `(N, 1)`의 차를 구하면 `(N, N)` 행렬이 되고, 그 평균은 의미 없는 값이지만 오류는 나지 않는다. 손실을 계산하기 전에 두 텐서의 shape이 같은지 `assert`한다.
- **batch 전체 통계로 batch 항목을 정규화하는 것.** 센서마다, 샘플마다 따로 내야 할 평균을 batch 전체에서 내면 항목 사이의 차이가 통계에 섞인다(연습문제 `batch-covariance`).

## 통과 기준

- attention 점수, bilinear layer, batch 공분산을 각각 einsum 한 줄과 shape 주석으로 쓰고, 어느 인덱스가 자유 인덱스이고 어느 인덱스가 축약되는지 말로 설명한다.
- `(B, T, D)` 배열에 대해 `reshape`와 `transpose`의 결과가 언제 다른지 작은 예로 보이고, `(5, 1, 3)`과 `(4, 3)` 같은 두 shape의 broadcasting 결과를 손으로 계산한다.
- 연습문제 두 개(`bilinear-einsum`, `batch-covariance`)의 테스트를 통과한다.
