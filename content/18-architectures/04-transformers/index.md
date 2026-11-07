---
id: architectures/transformers
title: "Attention과 Transformer"
title_en: "Attention and Transformers"
tier: core
status: draft
prerequisites: [architectures/inductive-bias, linear-algebra/tensors-and-einsum]
---

# Attention과 Transformer

CNN은 고정된 kernel로 가까운 위치만 섞고(18.2장), RNN은 과거를 고정 크기의 상태로 압축한다(18.3장). 두 구조 모두 "어느 위치의 정보를 얼마나 가져올지"가 입력의 **내용**과 무관하게 구조로 정해져 있다. attention은 이 비율을 입력에 따라 계산한다. 각 위치가 다른 모든 위치와의 유사도를 계산하고, 그 유사도를 가중치로 정보를 모은다. Vaswani 외(2017)의 Transformer는 순환과 합성곱 없이 attention과 위치별 MLP만으로 시퀀스 모델을 만들었고, 이후 언어뿐 아니라 영상·단백질 서열·분자·기상장 등 많은 분야의 기본 구조가 되었다.

이 장은 attention을 NumPy로 직접 구현하며 각 부품이 왜 그렇게 생겼는지 확인한다. $\sqrt{D}$ 스케일, 위치 정보가 없을 때의 순열 등변성, mask, 길이의 제곱으로 커지는 비용을 계산으로 보이고, 마지막에 직접 구현한 블록이 PyTorch 구현과 같은 출력을 내는지 확인한다.

## 학습 목표

- attention을 직접 구현한다
- mask와 위치 인코딩을 다룬다
- 계산 복잡도를 이해한다

## attention

**정의.** 입력 토큰 $T$개의 특징을 행으로 쌓은 $\mathbf{X} \in \mathbb{R}^{T \times D_{\text{model}}}$에서 세 가지 선형 사영을 만든다. **query** $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$는 "무엇을 찾는가", **key** $\mathbf{K} = \mathbf{X}\mathbf{W}_K$는 "무엇을 가지고 있는가", **value** $\mathbf{V} = \mathbf{X}\mathbf{W}_V$는 "가져갈 내용"이다. scaled dot-product attention은

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)\mathbf{V}
$$

이다. $D$는 query·key의 차원이고, softmax는 행마다(key 축으로) 계산한다. 인덱스로 쓰면 7.7장의 $s_{ts} = \sum_d Q_{td}K_{sd}/\sqrt{D}$, $w_{ts} = \operatorname{softmax}_s(s_{ts})$, $o_{td} = \sum_s w_{ts}V_{sd}$다. $t$는 query 위치, $s$는 key·value 위치다.

**softmax의 역할.** softmax는 점수를 0 이상이고 합이 1인 가중치로 바꾼다. 그래서 출력 $\mathbf{o}_t = \sum_s w_{ts}\mathbf{v}_s$는 value 벡터들의 **convex 결합**(7.2장)이다. 출력은 늘 value들이 만드는 convex hull 안에 있고, attention은 새 값을 만들어 내기보다 있는 값을 섞는다. 가중치가 한 key에 몰리면 그 value를 그대로 복사하는 조회(lookup)가 되고, 고르게 퍼지면 평균이 된다. 그 사이를 매끄럽게 오가며 미분 가능하다는 것이 "부드러운 사전 조회"라는 비유의 뜻이다.

**왜 $\sqrt{D}$로 나누는가.** $\mathbf{q}, \mathbf{k}$의 성분이 서로 독립이고 평균 0, 분산 1이면 $\mathbf{q}^\top\mathbf{k} = \sum_{d=1}^{D} q_d k_d$의 분산은 $D$다. $D$가 크면 점수의 폭이 커져 softmax가 한 key로 쏠리고(포화), 포화된 softmax는 gradient가 거의 0이라 학습이 느려진다. $\sqrt{D}$로 나누면 분산이 1 근처로 유지된다.

```python {exec="browser" session="attn"}
import numpy as np

rng = np.random.default_rng(0)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # 최댓값을 빼도 결과는 같고 overflow를 막는다
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):                   # Q (..., T, D), K (..., S, D), V (..., S, Dv)
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])   # (..., T, S)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)     # mask가 False인 칸은 가중치가 정확히 0
    w = softmax(scores)                              # (..., T, S), 행마다 합 1
    return w @ V, w                                  # (..., T, Dv), (..., T, S)

print("   D   Var(q·k)   최대 가중치 평균: 스케일 없음   √D로 나눔")
for D in [4, 64, 512]:
    Q, K = rng.normal(size=(16, D)), rng.normal(size=(16, D))   # (T, D), (S, D)
    raw = Q @ K.T                                    # (T, S)
    print(f"{D:4d}  {raw.var():9.1f}   {softmax(raw).max(-1).mean():22.3f}   {softmax(raw / np.sqrt(D)).max(-1).mean():9.3f}")
```

```output
   D   Var(q·k)   최대 가중치 평균: 스케일 없음   √D로 나눔
   4        3.4                    0.367       0.208
  64       65.9                    0.829       0.254
 512      537.2                    0.925       0.253
```

$D = 512$에서 나누지 않은 점수의 분산은 약 537이고, query마다 가장 큰 가중치의 평균이 0.925로 거의 한 key에 몰린다. $\sqrt{D}$로 나누면 $D$와 무관하게 0.25 근처다. `softmax`에서 행의 최댓값을 빼는 것은 결과를 바꾸지 않으면서 $e^{\text{큰 수}}$의 overflow를 막는 표준 기법이고, mask에서 가릴 칸의 점수를 $-\infty$로 두면 $e^{-\infty} = 0$이라 가중치가 정확히 0이 된다.

아래 위젯에서 온도 $\tau$(표준값 $\sqrt{d_k}$)를 바꾸며 점수 행렬, softmax 가중치, 고른 query의 출력이 어떻게 바뀌는지 본다. causal mask와 PAD mask도 켜고 끌 수 있다.

::widget{name="attention-visualizer"}

::quiz{id="why-scale-sqrt-d"}

## multi-head와 cross-attention

**self-attention과 cross-attention.** $\mathbf{Q}, \mathbf{K}, \mathbf{V}$를 모두 같은 시퀀스 $\mathbf{X}$에서 만들면 **self-attention**이다. 시퀀스 안의 모든 위치 쌍이 서로를 본다. query는 한 시퀀스 $\mathbf{X}$에서, key와 value는 다른 시퀀스 $\mathbf{Z}$에서 만들면 **cross-attention**이다: $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{Z}\mathbf{W}_K$, $\mathbf{V} = \mathbf{Z}\mathbf{W}_V$. 출력의 길이는 query 쪽($T$), 정보의 출처는 key·value 쪽($S$)이고, 두 길이가 달라도 된다. 번역 모델의 decoder가 원문을 참조할 때, 또는 예측하고 싶은 위치들이 관측값들을 참조할 때 쓴다.

:::note[분야 예시]
지상 관측소는 불규칙하게 흩어져 있고 개수도 날마다 다르다. 관측소 $S$곳의 (좌표, 관측값) 임베딩을 key·value로, 예측하고 싶은 격자점 $T$곳의 좌표 임베딩을 query로 두면 cross-attention이 격자점마다 어느 관측소를 얼마나 참고할지 계산한다. 관측소 순서에 무관하고(순열 불변 요약), 관측소 수가 바뀌어도 같은 모델을 쓸 수 있다. 결측 관측소는 padding mask로 가린다.
:::

**multi-head.** attention 한 번은 query마다 하나의 가중치 분포만 만든다. "가까운 위치"와 "같은 종류의 토큰"처럼 서로 다른 기준으로 동시에 정보를 모으려면, $D_{\text{model}}$ 차원을 $H$개의 head로 나눠 head마다 따로 attention을 계산한 뒤 이어 붙인다.

$$
\operatorname{head}_h = \operatorname{Attention}(\mathbf{X}\mathbf{W}_Q^{(h)}, \mathbf{X}\mathbf{W}_K^{(h)}, \mathbf{X}\mathbf{W}_V^{(h)}), \qquad \operatorname{MHA}(\mathbf{X}) = [\operatorname{head}_1, \dots, \operatorname{head}_H]\,\mathbf{W}_O.
$$

head마다 차원은 $D_h = D_{\text{model}}/H$이므로, 사영 행렬 네 개($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$)의 파라미터 수는 $H$와 무관하게 $4D_{\text{model}}^2$(bias 제외)다. 구현에서는 `(T, D)`를 `(T, H, Dh)`로 쪼갠 뒤 `(H, T, Dh)`로 축을 옮긴다. 7.7장에서 본 대로 이 단계에서 transpose 없이 reshape만 하면 shape은 맞지만 서로 다른 위치의 특징이 한 head에 섞인다.

**head의 의미.** 학습된 head들이 서로 다른 패턴(바로 앞 토큰, 같은 단어의 반복 등)을 보이는 경우가 있지만, head 하나가 해석 가능한 기능 하나를 맡는다는 보장은 없다. 같은 기능이 여러 head에 흩어지거나, 한 head가 여러 일을 섞어 하기도 한다.

다음 셀은 multi-head self-attention을 구현하고, 입력 토큰의 순서를 섞었을 때 출력이 어떻게 되는지 본다.

```python {exec="browser" session="attn"}
def mha(X, n_heads=4, mask=None):                    # X (T, D) → (T, D)
    T, D = X.shape
    split = lambda Z: Z.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2)   # (T, D) → (H, T, Dh)
    out, w = attention(split(X @ Wq), split(X @ Wk), split(X @ Wv), mask)     # (H, T, Dh), (H, T, T)
    return out.transpose(1, 0, 2).reshape(T, D) @ Wo, w                       # head를 다시 이어 붙임

def sinusoidal(T, D):                                # 절대 위치 인코딩 (T, D)
    angle = np.arange(T)[:, None] / 10000 ** (np.arange(0, D, 2)[None, :] / D)   # (T, D/2)
    pe = np.zeros((T, D))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe

D, T = 32, 10
Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
X = rng.normal(size=(T, D))                          # 토큰 10개 (T, D)
P = np.eye(T)[rng.permutation(T)]                    # 순서를 섞는 치환 행렬 (T, T)
PE = sinusoidal(T, D)
print("위치 정보 없음  f(PX) = P f(X)            :", np.allclose(mha(P @ X)[0], P @ mha(X)[0]))
print("위치 인코딩 더함 f(PX + PE) = P f(X + PE) :", np.allclose(mha(P @ X + PE)[0], P @ mha(X + PE)[0]))
print("평균 pooling 후 순열 불변                  :", np.allclose(mha(P @ X)[0].mean(0), mha(X)[0].mean(0)))
```

```output
위치 정보 없음  f(PX) = P f(X)            : True
위치 인코딩 더함 f(PX + PE) = P f(X + PE) : False
평균 pooling 후 순열 불변                  : True
```

위치 정보가 없으면 self-attention은 **순열 등변**이다: $f(\mathbf{P}\mathbf{X}) = \mathbf{P}f(\mathbf{X})$. 점수 $\mathbf{q}_t^\top\mathbf{k}_s$는 두 토큰의 내용으로만 정해지고 모든 query 위치에 같은 연산을 적용하기 때문이다. 출력을 평균 pooling하면 불변이 된다. 7.10장의 집합 모델과 같은 대칭이다. 점구름이나 원자 집합처럼 순서가 의미 없는 데이터에는 이 성질이 바로 원하는 귀납적 편향이다. 반면 언어나 시계열에서 위치 정보 없이 쓰면 "개가 사람을 물었다"와 "사람이 개를 물었다"를 구별하지 못한다. 셋째 줄처럼 위치 인코딩을 더하면 이 대칭이 깨진다.

::quiz{id="self-attention-permutation"}

## 위치 정보

순서가 의미를 갖는 데이터에서는 순열 대칭을 의도적으로 깨야 한다. 방법은 크게 셋이다.

**절대 위치 인코딩.** 위치 $p$마다 벡터 $\mathbf{e}_p$를 정해 입력 임베딩에 더한다. Vaswani 외(2017)는 주파수가 다른 사인·코사인으로 만든 고정 벡터 $e_{p,2i} = \sin(p/10000^{2i/D})$, $e_{p,2i+1} = \cos(p/10000^{2i/D})$를 썼다(위 셀의 `sinusoidal`). 위치마다 학습하는 벡터를 쓰기도 하는데, 이 경우 학습 때 본 최대 길이보다 긴 입력에는 쓸 수 없다.

**상대 위치.** 많은 문제에서 중요한 것은 절대 위치가 아니라 두 토큰 사이의 거리다. 점수에 거리 $t - s$에 따른 항을 더하거나, 다음의 회전 방식으로 상대 위치를 점수에 직접 넣는다.

**RoPE.** Su 외(2024)의 rotary position embedding은 query와 key의 성분을 두 개씩 짝지어, 위치 $m$에 비례하는 각 $m\theta_i$만큼 2차원 회전한다. 회전 행렬은 $\mathbf{R}_m^\top\mathbf{R}_n = \mathbf{R}_{n-m}$을 만족하므로(7.10장의 $SO(2)$ 표현)

$$
(\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k}) = \mathbf{q}^\top\mathbf{R}_{n-m}\mathbf{k}
$$

이고, 점수는 상대 위치 $n - m$에만 의존한다. 회전은 벡터의 길이를 바꾸지 않으므로 내용 정보도 보존한다.

```python {exec="browser" session="attn"}
def rope(x, pos, base=10000.0):                      # x (D,), 성분 쌍 (2i, 2i+1)을 pos·θ_i만큼 회전
    theta = pos * base ** (-np.arange(0, len(x), 2) / len(x))   # (D/2,)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

q, k = rng.normal(size=D), rng.normal(size=D)
for m, n in [(3, 1), (10, 8), (52, 50), (52, 49)]:
    print(f"query 위치 {m:2d}, key 위치 {n:2d} (차이 {m - n}): q·k = {rope(q, m) @ rope(k, n):8.4f}")
```

```output
query 위치  3, key 위치  1 (차이 2): q·k =  -1.1311
query 위치 10, key 위치  8 (차이 2): q·k =  -1.1311
query 위치 52, key 위치 50 (차이 2): q·k =  -1.1311
query 위치 52, key 위치 49 (차이 3): q·k =   1.0170
```

위치 (3, 1), (10, 8), (52, 50)은 절대 위치가 모두 다르지만 차이가 2로 같아 점수가 같고, 차이가 3이면 점수가 달라진다.

::quiz{id="rope-relative"}

**과학 데이터의 위치.** 위치가 반드시 정수 인덱스일 필요는 없다. 불규칙한 시계열에서는 실제 관측 시각을, 점구름이나 관측소 데이터에서는 공간 좌표를 위치로 쓴다. 이때 위치 인코딩은 "어떤 대칭을 깨고 어떤 대칭을 남길 것인가"의 선택이다. 절대 좌표를 넣으면 평행이동 대칭이 깨지고, 좌표 차이만 넣으면 평행이동 불변이 남는다(7.10장).

## mask

mask는 attention이 볼 수 있는 (query, key) 쌍을 정한다. 가릴 칸의 점수를 softmax 전에 $-\infty$로 바꾼다.

**causal mask.** 다음 토큰을 예측하는 decoder나 시계열 예측 모델에서 위치 $t$의 출력은 $s \le t$인 위치만 봐야 한다. 점수 행렬의 대각선 위쪽($s > t$)을 가리는 아래삼각 mask다. 학습할 때 이 mask가 없으면 모델이 정답(미래 토큰)을 보고 맞히므로 학습 손실은 매우 낮지만, 실제 생성에서는 쓸모가 없다.

**padding mask.** 길이가 다른 시퀀스를 batch로 묶으려고 끝을 PAD 토큰으로 채웠다면, PAD 위치의 **key**를 모든 query에서 가린다. 가리지 않으면 실제 토큰의 출력에 PAD의 value가 섞이고, 같은 시퀀스의 결과가 batch에 함께 들어간 다른 시퀀스의 길이에 따라 달라진다. PAD 임베딩이 0이어도 점수 0은 softmax에서 0이 아닌 가중치를 받으므로 문제는 사라지지 않는다.

```python {exec="browser" session="attn"}
T = 6
Xp = rng.normal(size=(T, D))
Xp[4:] = 0.0                                         # 마지막 두 칸은 PAD (실제 토큰 4개)
causal = np.tril(np.ones((T, T), dtype=bool))        # query t는 key s ≤ t만 본다 (T, T)
pad = np.ones((T, T), dtype=bool)
pad[:, 4:] = False                                   # 모든 query가 PAD key를 보지 않는다 (T, T)

Xf = Xp.copy()
Xf[2] += 1.0                                         # 시점 2의 토큰만 바꾼다
changed = np.abs(mha(Xf, mask=causal)[0] - mha(Xp, mask=causal)[0]).max(axis=1) > 0
print("causal mask: 시점 2를 바꾸면 출력이 바뀌는 시점:", np.flatnonzero(changed))

Xq = Xp.copy()
Xq[4:] = rng.normal(size=(2, D))                     # PAD 칸에 다른 값이 들어 있다면
for name, m in [("mask 없음    ", None), ("padding mask", pad)]:
    d = np.abs(mha(Xq, mask=m)[0][:4] - mha(Xp, mask=m)[0][:4]).max()
    print(f"{name}: PAD 값을 바꿨을 때 실제 토큰 출력의 최대 변화 =", "0" if d == 0 else f"{d:.3f}")
```

```output
causal mask: 시점 2를 바꾸면 출력이 바뀌는 시점: [2 3 4 5]
mask 없음    : PAD 값을 바꿨을 때 실제 토큰 출력의 최대 변화 = 0.997
padding mask: PAD 값을 바꿨을 때 실제 토큰 출력의 최대 변화 = 0
```

causal mask에서는 시점 2를 바꾸면 시점 2와 그 이후의 출력만 바뀐다. padding mask가 없으면 PAD 칸의 값만 바꿔도 실제 토큰 출력이 최대 1 가까이 변하고, mask가 있으면 정확히 0이다. 두 가지를 덧붙인다. PAD 위치의 **query** 행은 여전히 무언가를 출력하므로 손실과 평가에서 빼야 한다. 그리고 어떤 행의 key가 모두 가려지면 그 행의 softmax에서 `nan`이 나온다. 두 mask를 함께 쓰는 것은 연습문제 `attention-masks`에서 다룬다.

::quiz{id="padding-mask-missing"}

## Transformer 블록

**블록의 구성.** Transformer는 같은 블록을 여러 번 쌓는다. 블록은 두 개의 하위 층으로 이루어진다.

1. **multi-head attention**: 위치 사이의 정보를 섞는다.
2. **position-wise FFN**: 위치마다 독립적으로, 같은 MLP $\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$를 적용한다. 은닉 폭은 흔히 $D_{\text{model}}$의 4배다(원 논문은 512와 2048).

두 하위 층 모두 residual connection(18.2장)과 LayerNorm(17.1장)으로 감싼다. attention은 위치를 섞되 위치마다 선형인 가중합이고, FFN은 위치를 섞지 않되 비선형이다. 둘을 번갈아 쌓아 "어디서 가져올지"와 "가져온 것으로 무엇을 계산할지"를 나눠 맡긴다.

**encoder와 decoder.** encoder 블록은 mask 없는(양방향) self-attention을 쓰고, 입력 전체를 표현으로 바꾼다. decoder 블록은 causal self-attention을 쓰고, encoder–decoder 구조에서는 그 뒤에 encoder 출력을 key·value로 쓰는 cross-attention을 하나 더 둔다. 언어 모델처럼 decoder 블록만 쌓은 구조는 18.5장에서 다룬다.

**pre-norm과 post-norm.** 원래 Transformer는 하위 층의 출력을 residual에 더한 **뒤** 정규화했다: $\mathbf{x} \leftarrow \mathrm{LN}(\mathbf{x} + F(\mathbf{x}))$(post-norm). 요즘 많이 쓰는 pre-norm은 하위 층에 들어가기 **전에** 정규화하고 residual 경로는 그대로 둔다: $\mathbf{x} \leftarrow \mathbf{x} + F(\mathrm{LN}(\mathbf{x}))$. Xiong 외(2020)는 초기화 시점에 post-norm은 출력 근처 층의 gradient가 커서 learning rate warm-up이 필요하고, pre-norm은 gradient가 잘 정돈되어 warm-up 없이도 학습할 수 있음을 분석과 실험으로 보였다. pre-norm에서는 residual 경로에 정규화가 없으므로 블록 출력의 크기가 층을 지날수록 커질 수 있어, 마지막에 LayerNorm을 하나 더 두는 것이 보통이다.

직접 구현한 multi-head attention이 PyTorch의 `nn.MultiheadAttention`과 같은 출력을 내는지 확인한다. PyTorch의 bool mask는 `True`가 "가린다"는 뜻이라, 이 장의 NumPy 코드(`True` = 볼 수 있음)와 반대라는 점에 주의한다.

```python {exec="python"}
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
B, T, D, H = 2, 7, 32, 4
x = torch.randn(B, T, D)                             # (B, T, D)
ref = nn.MultiheadAttention(D, H, batch_first=True)
W_q, W_k, W_v = ref.in_proj_weight.chunk(3)           # 각 (D, D); PyTorch 선형층은 x @ W.T + b
b_q, b_k, b_v = ref.in_proj_bias.chunk(3)

def my_mha(x, causal=False):                         # (B, T, D) → (B, T, D)
    Dh = D // H
    split = lambda z: z.view(B, T, H, Dh).transpose(1, 2)        # (B, H, T, Dh)
    q, k, v = split(x @ W_q.T + b_q), split(x @ W_k.T + b_k), split(x @ W_v.T + b_v)
    s = q @ k.transpose(-1, -2) / Dh ** 0.5                      # (B, H, T, T)
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)     # head를 다시 이어 붙임 (B, T, D)
    return ref.out_proj(o)

future = torch.ones(T, T, dtype=torch.bool).triu(1)  # True = 가린다 (PyTorch의 bool mask 규칙)
for causal in [False, True]:
    y_ref, _ = ref(x, x, x, attn_mask=future if causal else None, need_weights=False)
    print(f"causal={causal!s:5}  nn.MultiheadAttention과 최대 차 < 1e-5:", (my_mha(x, causal) - y_ref).abs().max().item() < 1e-5)
q = k = v = torch.randn(B, H, T, 8)
manual = ((q @ k.transpose(-1, -2)) / 8 ** 0.5).masked_fill(future, float("-inf")).softmax(-1) @ v
print("F.scaled_dot_product_attention(is_causal=True)과 일치:",
      torch.allclose(F.scaled_dot_product_attention(q, k, v, is_causal=True), manual, atol=1e-6))
```

```output
causal=False  nn.MultiheadAttention과 최대 차 < 1e-5: True
causal=True   nn.MultiheadAttention과 최대 차 < 1e-5: True
F.scaled_dot_product_attention(is_causal=True)과 일치: True
```

PyTorch는 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$를 `in_proj_weight` 하나에 세로로 쌓아 두고, 선형층을 `x @ W.T + b`로 계산한다. 이 두 규칙만 맞추면 head 분리·스케일·mask·출력 사영까지 같은 결과가 나온다. 실제 코드에서는 `F.scaled_dot_product_attention`처럼 최적화된 함수를 쓰되, 이렇게 직접 구현한 버전을 작은 입력에서 비교하는 테스트로 남긴다.

## 효율

**길이에 대한 이차 비용.** 길이 $T$인 self-attention은 점수 행렬 $T \times T$를 head마다 만든다. 연산량은 $O(T^2 D)$, 점수 행렬의 메모리는 $O(HT^2)$다. 같은 시퀀스를 RNN은 $O(TD^2)$, 합성곱은 $O(TkD^2)$로 처리하므로, 길이가 특징 차원보다 훨씬 길어지면 attention이 병목이 된다.

```python {exec="browser" session="attn"}
for T_ in [1_024, 8_192, 65_536]:
    gb = T_ ** 2 * 16 * 4 / 1e9                      # head 16개, float32, 한 층·한 시퀀스의 점수 행렬
    print(f"T = {T_:6d}: 점수 행렬 {gb:8.2f} GB")

T = 8
Xs = rng.normal(size=(T, D))                         # (T, D)
full = mha(Xs, mask=np.tril(np.ones((T, T), dtype=bool)))[0]   # 한 번에 계산한 causal 출력 (T, D)
heads = lambda Z: Z.reshape(len(Z), 4, D // 4).transpose(1, 0, 2)   # (t, D) → (H, t, Dh)
K_cache, V_cache, steps = [], [], []
for t in range(T):                                   # 토큰을 하나씩 생성하는 상황
    K_cache.append(Xs[t] @ Wk)                       # 새 토큰의 key·value만 계산해 덧붙인다
    V_cache.append(Xs[t] @ Wv)
    q = heads((Xs[t] @ Wq)[None])                    # (H, 1, Dh)
    o, _ = attention(q, heads(np.array(K_cache)), heads(np.array(V_cache)))   # 과거 전체를 본다
    steps.append(o.transpose(1, 0, 2).reshape(D) @ Wo)
print("KV cache로 한 step씩 = 전체 causal 계산:", np.allclose(np.array(steps), full))
print("cache 크기 (key + value, 원소 수):", 2 * len(K_cache) * D)
```

```output
T =   1024: 점수 행렬     0.07 GB
T =   8192: 점수 행렬     4.29 GB
T =  65536: 점수 행렬   274.88 GB
KV cache로 한 step씩 = 전체 causal 계산: True
cache 크기 (key + value, 원소 수): 512
```

head 16개, float32 기준으로 한 층·한 시퀀스의 점수 행렬은 길이 1024에서 0.07 GB이지만 8192에서 4.29 GB, 65536에서 약 275 GB다. 길이를 8배로 늘리면 64배가 된다. 학습할 때는 역전파를 위해 층마다 이런 값을 저장하므로 실제 메모리 요구는 더 크다.

::quiz{id="score-matrix-memory"}

**efficient attention의 개념.** 이 비용을 줄이는 방법은 크게 셋으로 나뉜다.

- **희소·국소 attention**: 각 query가 가까운 창이나 정해진 일부 key만 본다. 비용은 줄지만 먼 위치 사이의 직접 상호작용을 포기하므로, CNN과 비슷한 국소성 가정을 다시 들여오는 셈이다.
- **저랭크·선형 attention**: softmax를 kernel 함수의 내적으로 근사해 $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ 대신 $\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$ 순서로 계산한다(7.7장의 축약 순서). 길이에 선형이지만 정확한 softmax attention은 아니다.
- **정확하지만 메모리 효율적인 구현**: 점수 행렬을 블록으로 나눠 계산하며 전체를 한꺼번에 저장하지 않는다. Dao 외(2022)의 FlashAttention은 GPU 메모리 계층 사이의 데이터 이동을 줄이도록 블록 계산을 설계했다. 결과는 표준 attention과 같고 메모리는 크게 줄지만, 연산량은 여전히 $T^2$에 비례한다.

**KV cache.** decoder로 토큰을 하나씩 생성할 때, 지난 토큰들의 key·value는 바뀌지 않는다. 그래서 매 step 새 토큰의 query·key·value만 계산하고, key·value는 cache에 덧붙여 재사용한다. 셀의 아래쪽은 이렇게 한 step씩 계산한 결과가 causal mask로 한 번에 계산한 결과와 정확히 같음을 확인한다. 한 step의 attention 비용은 지금까지의 길이 $t$에 비례하고, cache 메모리도 $t$에 비례해 계속 커진다. 고정 크기 상태만 갱신하는 RNN·SSM(18.3장)과의 근본적인 차이다.

::quiz{id="kv-cache-cost"}

:::note[분야 예시]
유전체 서열, 고해상도 기상장, 긴 분자동역학 궤적처럼 과학 데이터의 "길이"는 언어보다 훨씬 긴 경우가 많다. 전 지구를 0.25° 격자로 나누면 격자점이 100만 개를 넘는다. 모든 격자점 쌍에 attention을 거는 것은 비용상 불가능하므로, 패치로 묶어 토큰 수를 줄이거나, 국소 창 attention을 쓰거나, 그래프 구조(18.6장)로 이웃만 보게 한다. 어느 방법이든 "모든 위치가 모든 위치를 본다"는 Transformer의 기본 가정을 문제의 공간 구조에 맞게 줄이는 선택이다.
:::

## 해석의 한계

attention 가중치는 시각화하기 쉬워서 "모델이 어디를 보았는가"의 증거로 자주 쓰인다. 그러나 **attention map은 설명이 아니다.** 이유는 다음과 같다.

- 가중치는 value를 섞는 비율일 뿐이다. 가중치가 커도 그 value 벡터가 작거나 뒤의 사영 $\mathbf{W}_O$에서 상쇄되면 출력에 대한 기여는 작다.
- 한 층의 한 head만 보는 것은 전체 계산의 일부만 보는 것이다. 여러 head와 층, residual 경로, FFN이 출력을 함께 만든다. 몇 층을 지난 뒤의 "토큰 $s$"의 표현에는 이미 다른 토큰의 정보가 섞여 있다.
- Jain과 Wallace(2019)는 여러 NLP 과제에서 attention 가중치가 gradient 기반 중요도와 상관이 낮은 경우가 많고, 매우 다른 attention 분포로도 같은 예측을 낼 수 있음을 보고했다.

입력의 중요도를 주장하려면 그 입력을 바꾸거나 지웠을 때 예측이 어떻게 변하는지 직접 잰다(섭동·ablation). 표현 분석의 방법은 18.8장에서 다룬다.

::quiz{id="attention-not-explanation"}

:::note[분야 예시]
단백질 서열 모델의 attention map에서 잔기 쌍 사이에 큰 가중치가 보이면 이를 3차원 접촉이나 기능 부위의 증거로 읽고 싶어진다. 이런 관찰은 가설의 출발점으로는 유용하지만, 주장으로 쓰려면 알려진 구조 데이터와의 체계적인 비교나 해당 잔기를 바꾼 섭동 실험 같은 독립적인 근거가 필요하다. 가중치가 큰 head를 골라 보여 주는 것만으로는, 다른 head와 층이 무엇을 하는지, 무작위로 초기화한 모델에서도 비슷한 패턴이 보이는지를 알 수 없다.
:::

## 실습

작은 attention 블록을 직접 구현하고 shape·mask·gradient를 테스트로 검증한다. 코드는 `labs/architectures/transformers/attention_block.py`(pre-norm encoder 블록)와 `test_attention_block.py`(테스트 6개)다. 블록은 $\mathbf{h} = \mathbf{x} + \mathrm{MHA}(\mathrm{LN}(\mathbf{x}))$, $\mathbf{y} = \mathbf{h} + \mathrm{FFN}(\mathrm{LN}(\mathbf{h}))$이고, 같은 가중치를 `nn.TransformerEncoderLayer(norm_first=True, dropout=0)`에 복사해 출력을 비교한다. 집필 시점의 실행 결과는 다음과 같다.

```output
블록 파라미터 수: 8544  (PyTorch 구현: 8544)
mask 없음  출력 shape (3, 9, 32)  최대 차 < 1e-5: True
causal     출력 shape (3, 9, 32)  최대 차 < 1e-5: True
```

테스트는 다음을 확인한다.

- **shape**: 출력이 입력과 같은 `(B, T, D)`다.
- **PyTorch 구현과 일치**: mask 없을 때와 causal mask일 때 모두 `nn.TransformerEncoderLayer`와 같다.
- **causal mask**: 위치 4의 입력을 바꾸면 위치 0–3의 출력은 그대로이고 4 이후만 바뀐다.
- **padding mask**: PAD 칸의 값을 크게 바꿔도 실제 토큰의 출력이 변하지 않는다. mask를 빼면 변한다(테스트가 버그를 잡을 수 있는지 확인하는 대조군).
- **gradient**: float64에서 `torch.autograd.gradcheck`로 유한차분과 비교하고, 모든 파라미터에 0이 아닌 gradient가 흐르는지 확인한다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="scaled-dot-product-attention"}

::exercise{id="attention-masks"}

## 실패 사례

- **padding mask 누락.** 오류 없이 실행되고 학습도 되지만, 같은 입력의 결과가 batch 구성에 따라 달라진다. batch 크기 1로 계산한 출력과 PAD가 섞인 batch에서 계산한 출력이 같은지 비교하는 테스트를 둔다. PAD 위치의 출력을 손실에서 빼는 것도 잊지 않는다.
- **attention 가중치를 중요도로 해석하는 것.** 가중치는 섞는 비율이지 기여도나 원인이 아니다. 중요도 주장은 섭동·ablation으로 확인하고, 여러 head와 층, 무작위 초기화 모델과의 비교를 함께 보고한다.
- **causal mask 누락.** 다음 값 예측 모델에서 mask를 빼면 학습 손실이 비현실적으로 낮아진다. 미래 위치 입력을 바꿨을 때 과거 출력이 변하지 않는지 테스트한다.
- **head 분리에서 transpose 누락.** `(B, T, D)`를 바로 `(B, H, T, Dh)`로 reshape하면 위치와 head가 섞인다(7.7장). 직접 구현한 attention은 작은 입력에서 PyTorch 구현과 비교한다.
- **mask 규칙의 반대 혼동.** 라이브러리마다 bool mask의 `True`가 "볼 수 있음"인지 "가림"인지 다르다. 가린 칸의 가중치가 정확히 0인지 출력해 확인한다.

## 통과 기준

- scaled dot-product attention과 multi-head attention을 직접 구현해, 작은 입력에서 PyTorch 구현(`nn.MultiheadAttention` 또는 `nn.TransformerEncoderLayer`)과 같은 출력을 내는 것을 테스트로 보인다.
- causal mask와 padding mask가 각각 무엇을 막는지 섭동 테스트(미래 토큰·PAD 값을 바꿨을 때 출력 불변)로 보인다.
- 사용할 시퀀스 길이에서 attention의 점수 행렬 메모리와 연산량을 계산하고, 필요하면 어떤 efficient attention이나 대안 구조를 쓸지 근거를 댄다.
- 연습문제 두 개(`scaled-dot-product-attention`, `attention-masks`)의 테스트를 통과한다.

## 참고 문헌

- Dao, Fu, Ermon, Rudra, Ré (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS 2022*. — 블록 단위의 정확한 attention 계산
- Jain, Wallace (2019). Attention is not explanation. *NAACL-HLT 2019*, 3543–3556. — attention 가중치 해석의 한계
- Su, Ahmed, Lu, Pan, Bo, Liu (2024). RoFormer: Enhanced Transformer with rotary position embedding. *Neurocomputing*, 568, 127063. — RoPE
- Vaswani et al. (2017). Attention is all you need. *NIPS 30*, 5998–6008. — Transformer
- Xiong et al. (2020). On layer normalization in the Transformer architecture. *ICML 2020*, PMLR 119, 10524–10533. — pre-norm과 post-norm
