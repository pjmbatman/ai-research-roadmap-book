---
id: life-sciences/sequence-models
title: "생물 서열 모델"
title_en: "Biological Sequence Models"
tier: advanced
status: draft
prerequisites: [life-sciences/biology-for-ml, architectures/language-models]
---

# 생물 서열 모델

DNA와 단백질은 문자열이다. 그래서 자연어 처리의 도구, 즉 토큰화·embedding·Transformer·언어모델을 거의 그대로 가져다 쓸 수 있다. 실제로 단백질 서열 수억 개로 학습한 언어모델의 표현에 구조와 기능에 관한 정보가 담긴다는 결과가 나오면서(Rives 외, 2021), 사전학습 서열 모델은 생명과학 ML의 기본 도구가 되었다.

그러나 생물 서열은 자연어 문장과 한 가지 결정적으로 다르다. 서열들은 **진화로 이어진 가족**을 이룬다. 같은 조상에서 갈라진 서열들은 서로 비슷하고 기능도 비슷하다. 이 사실은 서열 모델이 배우는 신호의 원천이면서, 동시에 평가를 망가뜨리는 누수의 원천이다. 이 장은 서열을 표현하고 모델링하는 방법을 정리하고, 사전학습 표현이 "정말로 전이되는지"를 어떻게 확인하는지에 집중한다. Transformer와 언어모델의 구조 자체는 18.4–18.5장에서 다루었으므로 여기서는 생물 서열에 적용할 때 달라지는 판단만 다룬다.

## 학습 목표

- 서열 모델과 사전학습 서열 모델을 사용한다
- 서열 표현의 전이 효과를 평가한다

## 서열의 표현

**왜 필요한가.** 모델은 문자열을 직접 받지 못한다. 서열을 수치로 바꾸는 방법이 모델이 볼 수 있는 정보와 볼 수 없는 정보를 정한다.

**one-hot.** 길이 $L$의 서열을 알파벳 크기 $|A|$(DNA 4, 단백질 20)의 one-hot 벡터 $L$개, 즉 $L \times |A|$ 행렬로 바꾼다. 위치 정보가 그대로 남고 정보 손실이 없다. 대신 서열마다 길이가 달라 고정 길이 입력을 받는 모델에는 자르기나 채우기(padding)가 필요하고, 위치가 한 칸 밀린 두 서열은 전혀 다른 행렬이 된다. CNN·Transformer 같은 서열 모델의 입력으로 쓴다.

**k-mer.** 서열에서 길이 $k$의 모든 부분 문자열(k-mer)을 세어 $|A|^k$ 차원의 개수 벡터를 만든다. 서열 길이와 무관하게 차원이 고정되므로 고전적 모델(16부)에 바로 넣을 수 있다. 대가는 **위치 정보의 손실**이다. $k=1$이면 조성(composition)만 남고, 같은 글자를 다른 순서로 배열한 서열을 구분하지 못한다. $k$를 키우면 순서 정보가 늘지만 차원이 지수적으로 늘고 벡터가 희소해진다.

$$
\mathbf{v}_k(s)[w] = \#\{\, i : s_{i:i+k} = w \,\}, \qquad w \in A^k, \quad \sum_w \mathbf{v}_k(s)[w] = L - k + 1.
$$

여기서 $s_{i:i+k}$는 $i$번째 위치에서 시작하는 길이 $k$의 부분 문자열이다.

```python {exec="browser" session="seq"}
from itertools import product

import numpy as np

DNA = "ACGT"

def one_hot(seq, alphabet=DNA):
    idx = np.array([alphabet.index(c) for c in seq])
    return np.eye(len(alphabet))[idx]                  # (L, |A|)

def kmer_counts(seq, k, alphabet=DNA):
    vocab = {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}
    v = np.zeros(len(vocab))                           # (|A|^k,)
    for s in range(len(seq) - k + 1):
        v[vocab[seq[s:s + k]]] += 1
    return v

a, b = "ACGTACGTTT", "TTACGTACGT"                      # 같은 조각을 다른 순서로 배치
print("one-hot shape:", one_hot(a).shape)
for k in [1, 2, 3]:
    same = np.array_equal(kmer_counts(a, k), kmer_counts(b, k))
    print(f"k={k}: 차원 {4**k:3d}, 두 서열의 k-mer 벡터가 같다: {same}")
print("단백질(20종) 3-mer 차원:", 20**3)
```

```output
one-hot shape: (10, 4)
k=1: 차원   4, 두 서열의 k-mer 벡터가 같다: True
k=2: 차원  16, 두 서열의 k-mer 벡터가 같다: False
k=3: 차원  64, 두 서열의 k-mer 벡터가 같다: False
단백질(20종) 3-mer 차원: 8000
```

두 DNA 서열은 같은 조각을 다른 순서로 배치한 것이다. 1-mer(조성)는 같고 2-mer부터 달라진다. 단백질에서 3-mer만 써도 8,000차원이다.

::quiz{id="kmer-dimension"}

**토큰화.** 언어모델에 넣으려면 서열을 토큰 열로 나눈다(18.5장). 단백질 언어모델은 대개 아미노산 하나를 토큰 하나로 쓴다. 어휘가 20여 개로 작고, 한 잔기의 치환이 기능을 바꿀 수 있으므로 잔기 단위의 해상도가 중요하기 때문이다. DNA 모델에서는 염기 하나, 겹치거나 겹치지 않는 k-mer, BPE처럼 데이터에서 학습한 조각 등 여러 방식이 쓰인다. 토큰 하나가 여러 염기를 묶으면 같은 문맥 길이로 더 긴 서열을 보지만, 한 염기 변이가 토큰 경계를 바꿔 전혀 다른 토큰 열이 될 수 있다.

::quiz{id="composition-only"}

## 서열 모델

**왜 필요한가.** k-mer는 짧은 범위의 패턴만 본다. 단백질의 기능 부위, DNA의 조절 서열은 짧은 모티프(motif)와 모티프 사이의 배치로 이루어지고, 때로는 멀리 떨어진 위치끼리 상호작용한다.

**CNN.** 1차원 합성곱(18.2장)의 필터는 길이 $w$의 모티프 탐지기다. 필터가 서열 위를 미끄러지며 같은 가중치를 쓰므로 모티프가 어디에 있든 찾는다(평행이동 등변성). 층을 쌓거나 dilation을 쓰면 수용 영역이 넓어져 모티프 사이의 배치도 본다. DNA 조절 서열 연구에서 오래 쓰였고, 첫 층 필터를 모티프로 해석하는 관행이 있다.

**Transformer.** self-attention(18.4장)은 모든 위치 쌍을 직접 비교하므로 멀리 떨어진 잔기의 상호작용을 표현하기 좋다. 단백질은 접히면서 서열상 멀리 떨어진 잔기가 공간적으로 가까워지므로 이 성질이 특히 유용하다.

**긴 서열.** attention의 비용은 길이 $L$에 대해 $O(L^2)$다. 단백질은 대개 수백–수천 잔기라 감당할 수 있지만, 유전체의 조절 영역은 훨씬 길다. 대응은 세 가지다. 서열을 window로 잘라 따로 처리하거나, 토큰 하나에 여러 염기를 묶어 길이를 줄이거나, 길이에 선형으로 늘어나는 구조(state space model 등, 18.3장)를 쓴다. 어느 쪽이든 "모델이 실제로 본 문맥의 길이"가 과제에 필요한 거리보다 긴지 확인한다.

:::note[분야 예시]
**유전체학**에서는 DNA 서열로 그 구간의 조절 활성(어떤 단백질이 결합하는가, 주변 유전자가 얼마나 발현되는가)을 예측한다. **단백질 공학**에서는 서열의 한 자리 치환이 효소 활성이나 안정성을 얼마나 바꾸는지 예측한다. **면역학**에서는 항체 서열의 일부 영역으로 결합 대상을 예측한다. 세 과제 모두 입력은 문자열이지만, 필요한 문맥의 길이와 결과를 바꾸는 단위(한 염기, 한 잔기, 한 영역)가 다르다.
:::

## 사전학습 언어모델

**왜 필요한가.** 기능이 실험으로 측정된 서열은 적고, 서열 자체는 데이터베이스에 매우 많다. label 없는 서열로 사전학습한 모델의 표현(embedding)을 label이 적은 과제에 옮겨 쓰면 성능이 오를 수 있다.

**마스크 언어모델 목표.** 18.5장의 언어모델은 다음 토큰을 예측했다. 단백질 언어모델은 흔히 **마스크 언어모델**(masked language model, MLM) 목표를 쓴다. 서열의 일부 위치 집합 $M$을 가리고, 나머지 문맥 $x_{\setminus M}$으로 가린 자리를 맞힌다.

$$
L_{\text{MLM}}(\theta) = -\,\mathbb{E}_{x}\, \frac{1}{|M|} \sum_{i \in M} \log p_\theta\big(x_i \mid x_{\setminus M}\big).
$$

자기회귀 모델과 달리 가린 자리의 **양쪽** 문맥을 본다. 단백질에는 "왼쪽에서 오른쪽"이라는 자연스러운 생성 순서가 없고, 한 잔기의 적합성은 서열 전체에 의존하므로 양방향 문맥이 자연스럽다. 자기지도학습 목표의 일반론은 19.8장에서 다룬다.

**모델이 실제로 배우는 것.** 가린 잔기를 맞히는 가장 좋은 단서는 무엇일까. 같은 가족의 다른 서열들이 그 위치에 어떤 잔기를 가지는가, 즉 **진화 정보**다. 합성 단백질 가족 40개로 확인한다. 가족마다 길이 80의 조상 서열을 만들고, 각 자리를 30% 확률로 무작위 잔기로 바꾸어 구성원 8개를 만든다. 가족마다 기능 label도 정해 두고 실습에서 쓴다. 가족 0–29를 학습 데이터로 보고, 학습에 없던 가족 30–39에서 서열 하나씩 15%를 가린 뒤 세 가지 예측의 손실을 비교한다. 균등 분포, 학습 데이터 전체의 아미노산 빈도, 그리고 같은 가족의 나머지 서열 7개로 만든 위치별 분포(프로필)다.

```python {exec="browser" session="seq"}
AA = "ACDEFGHIKLMNPQRSTVWY"
HYD = np.array([c in "AILMFVW" for c in AA])           # (20,) 소수성 아미노산
rng = np.random.default_rng(0)

def family(label, n=8, L=80, rate=0.3):                # 조상 서열 + 무작위 치환으로 만든 가족
    p = np.where(HYD, 1.0 + 1.0 * label, 1.0)
    anc = rng.choice(20, L, p=p / p.sum())             # (80,) label 1이면 소수성 잔기가 많다
    kids = np.tile(anc, (n, 1))                        # (8, 80)
    hit = rng.random(kids.shape) < rate
    kids[hit] = rng.integers(0, 20, hit.sum())
    return ["".join(AA[i] for i in s) for s in kids]

labels = rng.integers(0, 2, 40)                        # (40,) 가족 40개의 기능 label
seqs = [s for f in labels for s in family(f)]
fam = np.repeat(np.arange(40), 8)                      # (320,) 가족 번호
S = np.array([[AA.index(c) for c in s] for s in seqs])   # (320, 80)
uni = np.bincount(S[fam < 30].ravel(), minlength=20) + 1.0   # 가족 0–29 = 학습 데이터
probs = {"균등": np.full((80, 20), 0.05), "전체 빈도": np.tile(uni / uni.sum(), (80, 1))}
loss = {k: [] for k in [*probs, "가족 프로필"]}
for f in range(30, 40):                                # 학습에 없던 가족의 서열
    query, rest = S[fam == f][0], S[fam == f][1:]      # (80,), (7, 80)
    prof = np.stack([np.bincount(c, minlength=20) for c in rest.T]) + 0.5
    probs["가족 프로필"] = prof / prof.sum(1, keepdims=True)   # (80, 20) 위치별 분포
    mask = rng.random(80) < 0.15                       # 15%를 가린다
    for k, P in probs.items():                         # 가린 자리의 평균 음의 로그우도
        loss[k].append(-np.mean(np.log(P[mask, query[mask]])))
for k, v in loss.items():
    print(f"{k}: 가린 자리 손실 {np.mean(v):.2f} nats")
```

```output
균등: 가린 자리 손실 3.00 nats
전체 빈도: 가린 자리 손실 3.00 nats
가족 프로필: 가린 자리 손실 1.75 nats
```

균등 분포의 손실은 $\ln 20 \approx 3.00$ nats이고, 전체 빈도는 이 합성 데이터에서 조성이 거의 균등해 나아지지 않는다. 가족 프로필은 손실을 1.75 nats로 크게 낮춘다. 사전학습된 언어모델은 프로필을 명시적으로 받지 않지만, 문맥에서 서열이 어느 가족인지 알아보고 그 가족의 위치별 분포를 "기억"하는 방식으로 손실을 줄일 수 있다. 이것이 언어모델 표현에 구조·기능 정보가 담기는 이유의 하나이고, 동시에 **학습 데이터에 가까운 친척이 많은 서열일수록 표현이 좋아진다**는 뜻이다. 그래서 사전학습 모델의 성능은 평가 서열과 사전학습 데이터의 거리에 따라 달라진다(29.5장).

::quiz{id="mlm-objective"}

**embedding의 활용.** 사전학습 모델을 쓰는 방식은 세 가지다.

| 방식 | 하는 일 | 장단점 |
|---|---|---|
| 고정 embedding + 단순 모델 | 잔기별 embedding을 평균 내 서열 벡터를 만들고 선형·트리 모델을 학습 | 계산이 싸고 데이터가 적어도 된다. 표현이 과제에 맞지 않으면 한계가 있다 |
| 미세조정(fine-tuning) | 사전학습 가중치에서 시작해 과제 손실로 전체 또는 일부를 학습 | 성능 여지가 크다. label이 적으면 과적합하고 계산이 비싸다 |
| zero-shot 점수 | 변이 전후 서열의 우도 차이로 변이 효과를 점수화 | label이 필요 없다. 우도와 측정하려는 기능이 일치한다는 보장이 없다 |

사전학습 모델의 가중치와 코드는 대개 공개 저장소로 배포된다. 예를 들어 Rives 외(2021)의 모델은 `github.com/facebookresearch/esm`에 공개되어 있다(집필 시점 예시). 모델 이름, 크기, 사전학습 데이터의 판이 결과에 영향을 주므로 셋 모두 기록한다.

## 진화 정보

**왜 필요한가.** 사전학습 언어모델 이전에도 서열 분석의 가장 강력한 도구는 진화 정보였다. 언어모델이 무엇을 배우는지 이해하려면, 진화 정보를 명시적으로 쓰는 방법을 알아야 한다.

**다중 서열 정렬.** 같은 가족의 서열들을 대응하는 위치끼리 줄 맞춘 것을 **다중 서열 정렬**(multiple sequence alignment, MSA)이라 한다. MSA의 열 하나는 진화 과정에서 그 위치에 어떤 잔기가 허용되었는지 보여 준다. 기능에 중요한 위치는 거의 바뀌지 않고(보존), 덜 중요한 위치는 여러 잔기가 나타난다. 앞 셀의 가족 프로필은 치환만 있는 합성 데이터라 정렬 없이 만들 수 있었지만, 실제 서열에는 삽입·삭제가 있어 정렬이 먼저 필요하다.

**공진화.** 단백질이 접혔을 때 맞닿는 두 잔기는 함께 변한다. 한 쪽이 바뀌면 접촉을 유지하도록 다른 쪽도 바뀐 서열이 자연 선택에서 살아남기 때문이다. 그래서 MSA의 두 열 사이 상관, 예를 들어 상호정보량(14.2장)

$$
I(X_i; X_j) = \sum_{a, b} p_{ij}(a, b) \log \frac{p_{ij}(a, b)}{p_i(a)\, p_j(b)}
$$

이 큰 열 쌍은 3차원에서 가까운 잔기 쌍의 후보가 된다. 여기서 $p_{ij}(a, b)$는 $i$열에 $a$, $j$열에 $b$가 함께 나타나는 비율이다. 30개 열 중 5번과 20번 열만 짝을 이루도록 만든 합성 MSA에서 확인한다.

```python {exec="browser" session="seq"}
N, C = 1000, 30
msa = rng.integers(0, 20, (N, C))                      # (1000, 30) 독립인 열들
partner = rng.permutation(20)                          # 5번 열의 잔기 → 20번 열의 짝 잔기
keep = rng.random(N) < 0.9
msa[keep, 20] = partner[msa[keep, 5]]                  # 90%의 서열에서 짝을 맞춘다

def mutual_info(x, z):
    joint = np.zeros((20, 20))
    np.add.at(joint, (x, z), 1)
    joint /= joint.sum()
    px, pz = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return np.sum(joint[nz] * np.log(joint[nz] / (px @ pz)[nz]))

mi = {(i, j): mutual_info(msa[:, i], msa[:, j]) for i in range(C) for j in range(i + 1, C)}
top = sorted(mi, key=mi.get, reverse=True)[:3]
for p in top:
    print(f"열 {p}: MI {mi[p]:.2f} nats")
print(f"전체 {len(mi)}쌍의 MI 중앙값 {np.median(list(mi.values())):.2f} nats")
```

```output
열 (5, 20): MI 2.54 nats
열 (4, 16): MI 0.26 nats
열 (1, 9): MI 0.25 nats
전체 435쌍의 MI 중앙값 0.20 nats
```

짝지은 열 쌍 (5, 20)의 상호정보량이 2.54 nats로 압도적이다. 그런데 서로 독립으로 만든 나머지 열 쌍도 0이 아니라 중앙값 0.20 nats를 가진다. 서열 1,000개로 $20 \times 20$ 결합 분포를 추정할 때 생기는 양의 편향이다. 실제 MSA에는 두 가지 어려움이 더 있다. 서열들이 계통수(phylogeny)로 묶여 있어 직접 접촉하지 않는 열들도 함께 변하고, A–B, B–C가 접촉하면 A–C도 간접적으로 상관을 보인다. 그래서 실제 공진화 분석은 상관을 그대로 쓰지 않고, 서열 가중치·편향 보정·간접 효과를 분리하는 전역 모델을 쓴다. 이 신호는 29.3장의 구조 예측이 진화 정보를 쓰는 출발점이다.

::quiz{id="coevolution-signal"}

## 실습

사전학습 표현의 효과를 평가하는 실험을 설계한다. 핵심은 **무엇과 비교하고, 어떻게 나누는가**다. 먼저 앞에서 만든 합성 가족 데이터로 분할의 효과를 본다. 기능 label은 가족 단위로 정했고, label 1인 가족은 조상 서열에 소수성 잔기가 많다. 즉 label에는 서열 조성으로 배울 수 있는 진짜 신호가 있다. 두 모델을 비교한다. 2-mer 개수의 1-최근접 이웃(가장 비슷한 서열의 label을 베낀다)과 1-mer 조성의 로지스틱 회귀다. 5-fold 교차검증의 fold는 직접 정한다. 무작위 분할은 서열 하나하나를 무작위로, 가족 분할은 가족 번호로 가족을 통째로 한 fold에 넣는다(15.4장의 group split).

```python {exec="browser" session="seq"}
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

y = labels[fam]                                        # (320,)
X1 = np.stack([kmer_counts(s, 1, AA) for s in seqs])   # (320, 20) 조성
X2 = np.stack([kmer_counts(s, 2, AA) for s in seqs])   # (320, 400)

def nn1(Xtr, ytr, Xte):                                # 가장 가까운 train 서열의 label을 베낀다
    d = (Xte**2).sum(1)[:, None] + (Xtr**2).sum(1)[None, :] - 2 * Xte @ Xtr.T   # (n_te, n_tr)
    return ytr[d.argmin(1)]

def logreg(Xtr, ytr, Xte):
    return LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, ytr).predict(Xte)

def cv_acc(fit_predict, X, folds):                     # folds: (320,) 샘플마다 fold 번호 0–4
    return np.mean([np.mean(fit_predict(X[folds != k], y[folds != k], X[folds == k]) == y[folds == k])
                    for k in range(5)])

random_folds = np.random.default_rng(1).permutation(320) % 5   # 서열 단위로 무작위
family_folds = fam % 5                                         # 가족 단위로 통째로
for name, fp, X in [("1-NN, 2-mer", nn1, X2), ("로지스틱, 조성", logreg, X1)]:
    print(f"{name}: 무작위 분할 {cv_acc(fp, X, random_folds):.2f}   "
          f"가족 분할 {cv_acc(fp, X, family_folds):.2f}")
```

```output
1-NN, 2-mer: 무작위 분할 1.00   가족 분할 0.63
로지스틱, 조성: 무작위 분할 0.84   가족 분할 0.76
```

무작위 분할에서는 1-최근접 이웃이 1.00으로 압도적이다. test 서열마다 같은 가족의 형제 서열이 train에 있어 그 label을 베끼기 때문이다. 가족 단위로 나누면 순위가 뒤집힌다. 1-최근접 이웃은 0.63으로 떨어지고, 조성의 로지스틱 회귀는 0.84에서 0.76으로 덜 떨어진다. 로지스틱 회귀도 조금 떨어지는 것은 무작위 분할에서 같은 가족의 조성을 이미 보았기 때문이다. 처음 보는 가족의 기능을 예측하려면 로지스틱 회귀가 낫다. **무작위 분할은 성능의 크기뿐 아니라 모델의 순위까지 틀리게 알려 준다.** 사전학습 embedding과 one-hot을 비교할 때도 같은 일이 일어난다.

::quiz{id="split-reverses-ranking"}

실제 데이터로 하는 실습은 다음 순서를 따른다. 사전학습 모델을 내려받고 embedding을 계산하는 단계는 GPU와 큰 내려받기가 필요할 수 있다. 도구와 명령은 집필 시점 예시이므로 각 모델 저장소의 안내를 확인한다.

1. **과제와 데이터.** 기능 label이 실험으로 측정된 단백질 데이터셋을 하나 고르고, label의 근거(분석법, 데이터베이스 판)를 기록한다(29.1장).
2. **분할.** 서열 유사도로 군집을 만들고 군집 단위로 train·test를 나눈다(29.5장). 무작위 분할도 함께 만들어 둔다.
3. **표현.** (a) one-hot 또는 k-mer, (b) 사전학습 단백질 언어모델의 평균 embedding을 계산한다.
4. **같은 조건의 비교.** 두 표현에 같은 downstream 모델(예: 로지스틱 회귀 또는 릿지)과 같은 튜닝 예산을 쓴다.
5. **보고.** 네 조합(표현 2 × 분할 2)의 성능을 fold별로 기록하고, 군집 분할에서의 차이를 사전학습의 효과로 보고한다. 가능하면 test 서열과 가장 가까운 사전학습 데이터 서열의 유사도도 함께 보고한다.

TAPE는 단백질 표현 학습을 평가하는 과제 묶음을 만들면서 진화적으로 먼 단백질에 대한 일반화를 시험하도록 분할을 구성했고, 자기지도 사전학습이 대부분의 과제와 모델에서 도움이 되었다고 보고했다(Rao 외, 2019). 자신의 실험도 같은 질문, 즉 "처음 보는 가족에서도 이득이 남는가"에 답해야 한다.

k-mer 표현을 직접 구현한다. 실제 DNA 서열에는 염기를 확정하지 못한 자리가 `N`으로 기록되어 있어, 이 경우를 처리해야 한다.

::exercise{id="kmer-counts"}

## 실패 사례

- **유사 서열이 학습·테스트에 섞인 상태에서 사전학습 효과를 주장하는 것.** 무작위 분할에서는 가장 가까운 친척의 label을 베끼는 모델이 이긴다. 사전학습 표현이 친척을 더 잘 찾아 줄 뿐이라면, 그것은 새 가족에 대한 전이가 아니다.
- **사전학습 데이터와 평가 서열의 중복을 확인하지 않는 것.** 사전학습 모델은 평가 서열이나 그 친척을 이미 보았을 수 있다(15.4장). 우도 기반 zero-shot 점수는 이 영향을 특히 크게 받는다.
- **비교 조건을 맞추지 않는 것.** embedding에는 튜닝한 MLP를, one-hot에는 기본값 로지스틱 회귀를 쓰면 표현의 차이와 모델의 차이가 섞인다.
- **모델이 본 문맥보다 긴 거리의 효과를 주장하는 것.** window로 잘라 학습한 모델은 window 밖의 정보를 쓸 수 없다.
- **MSA의 상관을 그대로 접촉으로 해석하는 것.** 계통 구조와 간접 상관, 표본 크기에서 오는 편향을 보정하지 않으면 거짓 양성이 많다.

## 통과 기준

- 같은 과제를 one-hot(또는 k-mer)과 사전학습 embedding으로 풀어, 무작위 분할과 서열 유사도 기반 군집 분할의 성능을 4개 조합 모두 fold별로 보고한다.
- 군집 분할에서 사전학습 표현의 효과가 남는지(또는 사라지는지)를 근거와 함께 쓰고, 사전학습 데이터와의 중복 점검 결과 또는 점검할 수 없는 이유를 적는다.
- 연습문제 `kmer-counts`의 테스트를 통과한다.

## 참고 문헌

- Rives 외 (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15), e2016239118. — 단백질 서열 2억 5천만 개로 학습한 언어모델의 표현에 구조·기능 정보가 담김
- Rao 외 (2019). Evaluating protein transfer learning with TAPE. *NeurIPS 32*. — 단백질 표현 학습의 평가 과제 묶음과 진화적으로 먼 단백질에 대한 일반화 평가
