---
id: architectures/language-models
title: "언어모델과 LLM의 기초"
title_en: "Language Models and LLM Foundations"
tier: advanced
status: draft
prerequisites: [architectures/transformers]
---

# 언어모델과 LLM의 기초

대규모 언어모델(large language model, LLM)은 이제 문헌을 요약하고, 코드를 쓰고, 실험 기록을 정리하는 데까지 연구 현장에 들어와 있다. 그런데 이 모델이 학습하는 목표는 놀라울 만큼 단순하다. **앞의 토큰들이 주어졌을 때 다음 토큰의 확률분포를 맞히는 것**이다. 이 장은 그 단순한 목표가 어떤 계산으로 이루어지는지, 그리고 그 위에 어떤 후속 학습과 사용 방식이 얹히는지를 차례로 본다.

이 장의 관심은 특정 제품의 사용법이 아니다. 제품과 모델 크기는 몇 달 단위로 바뀐다. 바뀌지 않는 것은 토큰화가 입력을 어떻게 자르는지, perplexity가 무엇을 재는지, 디코딩 설정이 출력을 어떻게 바꾸는지, 그리고 LLM의 출력을 연구 근거로 쓰기 전에 무엇을 확인해야 하는지다. 18.4장의 Transformer가 이 장의 계산 엔진이고, 14.1장의 교차엔트로피가 이 장의 손실이다.

## 학습 목표

- 토큰화·다음 토큰 예측·사전학습·후속 학습으로 이어지는 언어모델의 학습 과정을 설명한다
- perplexity와 디코딩 설정(temperature·top-k)을 계산하고 해석한다
- LLM을 연구 도구와 연구 대상으로 다룰 때의 평가·검증 절차를 세운다

## 토큰화

**왜 필요한가.** 신경망은 문자열을 직접 받지 못한다. 문자열을 정수 열로 바꾸는 규칙이 필요하고, 이 규칙을 **토큰화**(tokenization), 규칙을 구현한 도구를 tokenizer라 한다. 단위를 무엇으로 잡느냐에 따라 문제가 달라진다. 단어 단위로 자르면 사전(vocabulary)이 끝없이 커지고, 학습 때 보지 못한 단어(신조어, 화합물 이름, 유전자 이름)는 모두 "모르는 단어" 하나로 뭉개진다. 문자 단위로 자르면 모르는 단어는 없지만 열이 길어지고, 18.4장에서 본 attention의 비용은 길이의 제곱으로 늘어난다.

**정의.** 하위 단어(subword) 토큰화는 그 사이를 택한다. 자주 나오는 문자열은 한 토큰으로, 드문 문자열은 여러 조각으로 나눈다. 대표적인 방법이 **BPE**(byte pair encoding)다. 원래 데이터 압축 기법이던 것을 기계번역의 열린 어휘 문제에 맞게 고친 것으로(Sennrich 외, 2016), 학습 절차는 다음과 같다.

1. 모든 단어를 문자 열로 쪼개고, 단어 끝 표시(`</w>`)를 붙인다.
2. 말뭉치 전체에서 가장 자주 이웃하는 두 기호 쌍을 찾아 새 기호 하나로 합친다. 이 병합을 규칙 목록에 추가한다.
3. 정해진 횟수(또는 원하는 사전 크기)에 이를 때까지 2를 반복한다.

새 문자열을 토큰화할 때는 문자로 쪼갠 뒤 학습한 병합 규칙을 **배운 순서대로** 적용한다. 같은 말뭉치와 같은 규칙이면 결과는 결정적이다.

**코드.** 화학 단어 몇 개와 연도 몇 개로 이루어진 작은 말뭉치에서 병합 규칙 12개를 배운다.

```python {exec="browser" session="lm"}
from collections import Counter

corpus = ("reaction reactor reactant react reacted "
          "1990 1995 2000 2005 2010 2015 2020 2020 2021").split()
words = Counter(tuple(w) + ("</w>",) for w in corpus)  # 단어(문자 튜플) → 빈도

def merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1]); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)

merges = []
for _ in range(12):
    pairs = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]):
            pairs[a, b] += n
    best = max(pairs, key=lambda p: (pairs[p], p))      # 빈도 최대, 동률은 사전순으로 고정
    merges.append(best)
    words = Counter({merge(w, best): n for w, n in words.items()})
print("병합 순서:", [a + b for a, b in merges])

def tokenize(word):
    w = tuple(word) + ("</w>",)
    for p in merges:
        w = merge(w, p)
    return list(w)

for w in ["reactors", "2020", "2019", "1920"]:
    print(f"{w:>9} → {tokenize(w)}")
```

```output
병합 순서: ['20', 're', 'rea', 'reac', 'react', '5</w>', '0</w>', '99', '20</w>', '2020</w>', '201', '200']
 reactors → ['react', 'o', 'r', 's', '</w>']
     2020 → ['2020</w>']
     2019 → ['201', '9', '</w>']
     1920 → ['1', '9', '20</w>']
```

**직관.** 병합 순서는 말뭉치의 빈도를 그대로 반영한다. 다섯 단어가 공유하는 어간 `react`는 다섯 번째 병합에서 한 토큰이 되고, 처음 보는 `reactors`도 `react` + 나머지 문자로 무리 없이 나뉜다. 모르는 단어가 생기지 않는다는 것이 하위 단어 토큰화의 장점이다.

같은 출력에서 단점도 보인다. 말뭉치에 두 번 나온 `2020`은 한 토큰이지만, `2019`는 `201` + `9`로, `1920`은 `1` + `9` + `20`으로 쪼개진다. **숫자가 자릿값이 아니라 말뭉치 빈도로 잘린다.** 같은 크기의 수가 서로 다른 개수의 토큰이 되고, `20`이라는 토큰이 어떤 수에서는 천의 자리와 백의 자리를, 다른 수에서는 십의 자리와 일의 자리를 뜻한다. 모델은 자릿값을 토큰 경계와 무관하게 다시 배워야 한다.

::quiz{id="bpe-number-split"}

**변형.** 실제 tokenizer는 이 기본형을 몇 가지로 고친다.

- **바이트 수준 BPE.** 문자 대신 UTF-8 바이트 256개에서 시작한다. 어떤 문자열이든 바이트로는 표현되므로 모르는 기호가 원리적으로 생기지 않는다. 한국어 한 글자는 UTF-8로 3바이트라, 한국어 말뭉치가 적은 tokenizer에서는 한 글자가 여러 토큰이 되기도 한다.
- **SentencePiece.** 공백을 다른 문자와 같은 기호로 다뤄, 언어마다 단어를 미리 나누는 전처리 없이 원문에서 바로 하위 단어를 배우는 도구다. BPE와 unigram 언어모델 기반 분할을 모두 지원한다.
- **숫자 규칙.** 위의 문제 때문에 숫자를 한 자리씩 강제로 나누는 규칙을 두는 tokenizer도 있다.

어떤 규칙을 쓰는지는 모델마다 다르다. 쓰려는 모델의 tokenizer로 자기 데이터의 대표 문자열(수치, 단위, 화학식, 유전자 이름)을 직접 토큰화해 보는 것이 확인 절차다.

:::note[분야 예시]
분자를 문자열로 적는 SMILES 표기에서 `Cl`(염소)과 `Br`(브로민)은 두 글자가 한 원자다. 문자 단위로 자르면 `C`(탄소)와 `l`로 나뉘어 원자 경계가 깨진다. 그래서 화학 언어모델은 원자 기호와 결합 기호를 단위로 하는 정규식 tokenizer를 따로 쓰는 경우가 많다. 단백질 서열(아미노산 20종)이나 DNA 서열(염기 4종)은 반대로 문자 하나가 자연스러운 단위지만, 서열이 수천에서 수백만 글자로 길어 k-mer(연속한 k글자)나 BPE로 묶어 길이를 줄인다. 토큰 단위는 도메인의 화학적·생물학적 단위와 맞는지부터 확인한다.
:::

## 자기회귀 언어모델

**정의.** 토큰 열 $x_1, \dots, x_T$의 결합확률은 확률의 곱셈 규칙으로 언제나 다음처럼 분해된다.

$$
p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).
$$

$x_{<t} = (x_1, \dots, x_{t-1})$은 앞선 토큰들이고, $\theta$는 모델 파라미터다. 이 분해 자체는 근사가 아니다. **자기회귀**(autoregressive) 언어모델은 각 조건부분포 $p_\theta(x_t \mid x_{<t})$를 하나의 신경망으로 표현하고, 학습 말뭉치의 음의 로그우도

$$
L(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

를 최소화한다. 이것이 **다음 토큰 예측**(next-token prediction)이다. 각 위치에서 정답 토큰에 대한 교차엔트로피이므로, 14.1장의 결과대로 $L$을 줄이는 것은 데이터 분포와 모델 분포 사이의 KL을 줄이는 것과 같다. 18.4장의 causal mask는 위치 $t$의 예측이 $x_{\ge t}$를 보지 못하게 막아, 길이 $T$인 열 하나로 $T$개의 예측 문제를 한 번에 학습하게 해 준다. 생성 모델로서의 자기회귀 모델은 19.2장에서 다시 다룬다.

**perplexity.** 언어모델의 기본 평가 지표는 손실을 지수로 되돌린 **perplexity**다.

$$
\mathrm{PPL} = \exp\Big(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\Big).
$$

해석은 "모델이 매 위치에서 평균적으로 몇 개의 후보 사이에서 균등하게 망설이는가"다. 사전 크기가 $V$인 균등 분포 모델의 perplexity는 정확히 $V$이고, 정답을 확률 1로 맞히는 모델은 1이다. 자료 생성 과정 자체에 남은 무작위성(엔트로피율)이 perplexity의 하한을 정한다. 아무리 좋은 모델도 데이터에 없는 정보를 만들어 낼 수는 없기 때문이다.

**코드.** 하한을 정확히 아는 자료로 실험한다. DNA 염기 4종(A·C·G·T)으로 이루어진 서열을, 주기 8의 반복 서열에 10% 확률로 돌연변이가 생기는 과정으로 만든다. 즉 $x_t$는 확률 0.9로 $x_{t-8}$을 복사하고, 확률 0.1로 4종 중 하나를 무작위로 고른다. 정답 확률은 반복 위치에서 $0.9 + 0.1/4 = 0.925$, 나머지 세 염기에서 각 $0.025$이므로 하한을 계산할 수 있다. 여기에 앞 $k$개 토큰을 문맥으로 쓰는 계수 기반 n-gram 모델(문맥별 빈도를 세고 add-$\alpha$ 평활화)을 맞춘다.

```python {exec="browser" session="lm"}
import numpy as np

rng = np.random.default_rng(0)
V, PERIOD, P_COPY = 4, 8, 0.9                          # A·C·G·T, 주기 8 반복, 돌연변이 10%
def make_seq(n):
    x = rng.integers(V, size=n)                         # (n,)
    for t in range(PERIOD, n):
        if rng.random() < P_COPY:
            x[t] = x[t - PERIOD]
    return x
train, test = make_seq(20_000), make_seq(5_000)

def ngram_nll(k, alpha=0.1):
    """앞 k개 토큰을 문맥으로 쓰는 계수 모델의 평균 음의 로그우도(nat/토큰)"""
    def ctx_ids(x):                                     # 문맥을 정수 하나로 부호화 (n-k,)
        return sum(x[i:len(x) - k + i] * V ** (k - 1 - i) for i in range(k)) if k else np.zeros(len(x), int)
    counts = np.zeros((V ** k, V))
    np.add.at(counts, (ctx_ids(train), train[k:]), 1)
    prob = (counts + alpha) / (counts + alpha).sum(1, keepdims=True)   # add-α 평활화
    nll = lambda x: -np.log(prob[ctx_ids(x), x[k:]]).mean()
    return nll(train), nll(test)

q = P_COPY + (1 - P_COPY) / V                           # 반복 위치의 정답 확률
floor = -(q * np.log(q) + (V - 1) * (1 - q) / V * np.log((1 - q) / V))
print(f"균등 분포 perplexity = {V},  자료 생성 과정의 하한 = {np.exp(floor):.3f}")
for k in [0, 1, 2, 4, 8]:
    tr, te = ngram_nll(k)
    print(f"문맥 {k}  train ppl {np.exp(tr):6.3f}   test ppl {np.exp(te):6.3f}")
```

```output
균등 분포 perplexity = 4,  자료 생성 과정의 하한 = 1.344
문맥 0  train ppl  3.999   test ppl  4.002
문맥 1  train ppl  3.988   test ppl  4.019
문맥 2  train ppl  3.953   test ppl  4.021
문맥 4  train ppl  3.613   test ppl  4.472
문맥 8  train ppl  1.212   test ppl  3.531
```

결과는 세 가지를 보여 준다.

- **문맥이 의존성에 닿지 못하면 배울 것이 없다.** 문맥 0–2의 test perplexity는 균등 분포의 4와 같다. 이 자료에서 바로 앞 몇 토큰은 다음 토큰에 대한 정보가 없고, 정보는 8칸 뒤에 있다.
- **문맥이 길어지면 계수 모델은 외운다.** 문맥 8에서는 가능한 문맥이 $4^8 = 65{,}536$가지로 학습 토큰 2만 개보다 많다. 대부분의 문맥을 한두 번만 보므로 train perplexity(1.212)는 하한(1.344)보다도 낮아지지만, test에서는 처음 보는 문맥이 대부분이라 3.531에 머문다. **train perplexity가 하한 아래로 내려가면 그것은 성능이 아니라 암기의 증거다.**
- **비교는 test perplexity로 한다.** 문맥 4의 test perplexity(4.472)는 문맥 0보다 나쁘다. 평활화한 희소 통계가 무작위 잡음을 패턴으로 착각한 것이다.

::quiz{id="train-ppl-below-floor"}

계수 모델의 한계는 문맥마다 독립된 표를 두는 데서 온다. 신경망 언어모델은 문맥을 embedding으로 바꾸고 같은 가중치로 모든 문맥을 처리하므로, 보지 못한 문맥에도 일반화할 수 있다. 같은 자료로 문맥 8을 보는 작은 MLP를 PyTorch로 학습한다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
V, PERIOD, P_COPY = 4, 8, 0.9
def make_seq(n):                                        # cell 2와 같은 자료 생성 과정
    x = rng.integers(V, size=n)
    for t in range(PERIOD, n):
        if rng.random() < P_COPY:
            x[t] = x[t - PERIOD]
    return torch.as_tensor(x)
train, test = make_seq(20_000), make_seq(5_000)

def windows(x, k=8):                                    # (n-k, k) 문맥, (n-k,) 다음 토큰
    return x.unfold(0, k, 1)[:-1], x[k:]
Xtr, ytr = windows(train); Xte, yte = windows(test)

model = nn.Sequential(nn.Embedding(V, 8), nn.Flatten(),  # (B, 8) → (B, 8, 8) → (B, 64)
                      nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, V))  # → (B, V) logit
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(1, 1501):
    idx = torch.randint(len(ytr), (256,))
    loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])   # 다음 토큰 예측
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 300, 1500):
        with torch.no_grad():
            ppl = [np.exp(nn.functional.cross_entropy(model(X), y).item()) for X, y in [(Xtr, ytr), (Xte, yte)]]
        print(f"step {step:4d}  train ppl {ppl[0]:.3f}  test ppl {ppl[1]:.3f}")
```

```output
step    1  train ppl 3.862  test ppl 3.837
step  300  train ppl 1.412  test ppl 1.424
step 1500  train ppl 1.382  test ppl 1.439
```

파라미터 약 4,500개짜리 MLP가 300 step 만에 test perplexity 1.424로 하한 1.344에 다가간다. 계수 모델(3.531)과의 차이는 문맥 길이가 아니라 **파라미터 공유**에서 온다. 1500 step에서 train은 더 내려가고 test는 조금 오르기 시작하는 것도 보인다. 실제 LLM은 이 MLP 자리에 18.4장의 Transformer를 두고, 문맥을 수천 토큰 이상으로 늘린 것이다.

**tokenizer가 다르면 perplexity를 직접 비교할 수 없다.** perplexity는 토큰 하나당 평균이다. 같은 텍스트라도 토큰을 크게 자르면 토큰 수가 줄고 토큰 하나가 담는 정보가 늘어 토큰당 perplexity가 커진다. 같은 텍스트 전체의 음의 로그우도 $\mathrm{NLL}_{\text{total}}$은 tokenizer와 무관하게 비교할 수 있는 양이므로, 서로 다른 tokenizer를 쓰는 모델은 문자당·바이트당 값으로 바꿔 비교한다.

$$
\mathrm{PPL}_{\text{char}} = \exp\Big(\frac{\mathrm{NLL}_{\text{total}}}{N_{\text{char}}}\Big) = \mathrm{PPL}_{\text{token}}^{\,N_{\text{token}}/N_{\text{char}}}, \qquad \text{bits per byte} = \frac{\mathrm{NLL}_{\text{total}}}{N_{\text{byte}} \ln 2}.
$$

::quiz{id="tokenizer-ppl-comparison"}

## 사전학습과 후속 학습

다음 토큰 예측으로 대규모 말뭉치를 학습한 모델은 문장을 그럴듯하게 이어 쓰지만, 질문에 답하거나 지시를 따르도록 학습된 것은 아니다. 현재의 LLM은 보통 여러 단계를 거친다. 각 단계는 손실과 데이터가 다르다.

| 단계 | 데이터 | 손실 | 바뀌는 것 |
|---|---|---|---|
| 사전학습(pretraining) | 레이블 없는 대규모 텍스트·코드 | 다음 토큰 예측 | 언어·지식·패턴 전반 |
| instruction tuning(지도 미세조정, SFT) | (지시, 모범 응답) 쌍 | 응답 부분의 다음 토큰 예측 | 지시를 따르는 응답 형식 |
| RLHF | 같은 지시에 대한 응답들의 사람 선호 비교 | 선호를 학습한 보상 모델 점수를 강화학습으로 최대화 | 선호되는 응답 쪽으로 분포 이동 |
| DPO | 선호 응답·비선호 응답 쌍 | 두 응답의 로그확률 차이에 대한 분류형 손실 | RLHF와 같은 목표를 보상 모델 없이 |

- **instruction tuning**은 손실 함수로 보면 사전학습과 같다. 다른 것은 데이터다. 지시와 모범 응답을 이어 붙이고, 보통 응답 부분의 토큰에만 손실을 건다.
- **RLHF**(reinforcement learning from human feedback)는 먼저 사람이 두 응답 중 더 나은 것을 고른 비교 자료로 보상 모델(reward model)을 학습하고, 그 보상을 최대화하도록 언어모델을 강화학습으로 미세조정한다. 이때 원래 모델에서 너무 멀어지지 않도록 KL 벌점을 함께 건다(Ouyang 외, 2022). 강화학습 자체는 20부에서 다룬다.
- **DPO**(direct preference optimization)는 KL 벌점이 붙은 보상 최대화 문제의 해를 언어모델의 확률로 다시 쓰면, 선호 쌍에 대한 단순한 분류 손실로 같은 목표를 최적화할 수 있다는 관찰에서 나왔다. 별도의 보상 모델과 학습 중 샘플링이 필요 없다(Rafailov 외, 2023).

후속 학습은 모델이 **아는 것**보다 **응답하는 방식**을 주로 바꾼다고 이해하는 것이 안전하다. 선호 최적화는 사람이 더 좋다고 판단한 응답 쪽으로 분포를 옮길 뿐이고, 사람이 판단하기 어려운 사실 관계(새 실험 결과, 수치의 정확성)를 보장하지 않는다. 오히려 확신에 찬 문체가 선호되면 틀린 내용도 확신에 찬 문체로 나온다. 사전학습된 모델을 과학 데이터로 이어 학습하는 전이와 foundation model은 19.9장에서 다룬다.

::quiz{id="training-stages"}

## 사용 방식

학습된 LLM을 쓰는 방식은 파라미터를 바꾸는지, 외부 정보를 넣는지에 따라 나뉜다.

**디코딩.** 모델이 내는 것은 다음 토큰에 대한 logit 벡터 $\mathbf{z} \in \mathbb{R}^V$다. 실제 텍스트를 만들려면 여기서 토큰을 하나 골라 문맥에 붙이고 다시 예측하는 일을 반복한다. 고르는 규칙이 출력을 크게 바꾼다. **temperature** $\tau > 0$은 softmax 앞에서 logit을 나눈다.

$$
p_i(\tau) = \frac{\exp(z_i/\tau)}{\sum_{j=1}^{V}\exp(z_j/\tau)}.
$$

$\tau < 1$이면 분포가 뾰족해지고, $\tau \to 0$이면 확률이 가장 큰 토큰만 고르는 greedy 디코딩이 된다. $\tau > 1$이면 평평해지고, $\tau \to \infty$이면 균등 분포가 된다. **top-k 샘플링**은 logit이 큰 $k$개만 남기고 나머지 확률을 0으로 만든 뒤 다시 정규화한다. 비슷한 아이디어로, 누적 확률이 $p$에 이를 때까지의 토큰만 남기는 nucleus(top-p) 샘플링이 있다(Holtzman 외, 2020). 두 방법 모두 확률이 낮은 꼬리 토큰이 가끔 뽑혀 문장이 무너지는 것을 막는다.

```python {exec="browser" session="lm"}
def decode_dist(logits, T=1.0, k=None):
    z = np.asarray(logits, float) / T                   # (V,)
    if k is not None:
        z = np.where(z >= np.sort(z)[-k], z, -np.inf)   # 상위 k개만 남긴다
    p = np.exp(z - z.max())                             # 안정적인 softmax
    return p / p.sum()

logits = np.array([2.0, 1.0, 0.5, -1.0])                # 다음 토큰 후보 4개
H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()      # 엔트로피(nat)
for T, k in [(1.0, None), (0.5, None), (2.0, None), (1.0, 2)]:
    p = decode_dist(logits, T, k)
    print(f"T={T:3.1f} top-k={str(k):4}  p = {np.round(p, 3)}  H = {H(p):.3f}")

p = decode_dist(logits, 1.0)
draws = np.random.default_rng(1).choice(4, size=10_000, p=p)
print("표본 빈도 (T=1)       :", np.round(np.bincount(draws, minlength=4) / 10_000, 3))
```

```output
T=1.0 top-k=None  p = [0.609 0.224 0.136 0.03 ]  H = 1.014
T=0.5 top-k=None  p = [0.842 0.114 0.042 0.002]  H = 0.538
T=2.0 top-k=None  p = [0.434 0.263 0.205 0.097]  H = 1.265
T=1.0 top-k=2     p = [0.731 0.269 0.    0.   ]  H = 0.582
표본 빈도 (T=1)       : [0.608 0.227 0.135 0.031]
```

어느 설정에서도 확률의 **순서**는 바뀌지 않는다. temperature와 top-k는 모델이 이미 매긴 순위를 그대로 두고 분포의 퍼짐만 조절한다. 그래서 temperature를 낮춘다고 모델이 더 정확해지지는 않는다. 같은 답을 더 일관되게 낼 뿐이고, 그 답이 틀렸다면 일관되게 틀린다. 반대로 temperature를 높이면 다양성은 늘지만 근거 없는 꼬리 토큰도 더 자주 뽑힌다. 마지막 줄은 샘플링이 실제로 이 분포를 따른다는 확인이다. 같은 질문을 여러 번 던져 답이 흔들리는지 보는 것은 모델의 불확실성을 엿보는 간단한 방법이지만, 그 흔들림이 정답일 확률과 맞는다는 보장(calibration)은 따로 확인해야 한다(21.1장).

::widget{name="sampling-temperature"}

::quiz{id="temperature-effects"}

**prompting과 in-context learning.** 파라미터를 바꾸지 않고 입력만 설계해 원하는 행동을 끌어내는 것이 prompting이다. 입력 안에 몇 개의 (입력, 출력) 예시를 넣으면 모델이 그 형식을 따라 새 입력에 답하는데, 이를 **in-context learning**이라 한다. gradient 업데이트는 없다. 예시가 바뀌면 출력이 바뀌고, 예시의 순서나 형식처럼 사소해 보이는 차이에도 결과가 달라질 수 있다. 그래서 prompt는 하이퍼파라미터처럼 다루고, 여러 변형에서 결과가 유지되는지 확인한다.

**RAG.** 모델의 파라미터에 담긴 지식은 학습 시점에 고정되어 있고, 출처를 밝히지 못한다. **RAG**(retrieval-augmented generation)는 질문과 관련된 문서를 검색해 입력에 함께 넣고 그 문서를 근거로 답하게 한다(Lewis 외, 2020). 검색은 보통 문서와 질문을 embedding으로 바꿔 가까운 것을 찾는 방식이다(18.8장). RAG는 최신 문헌이나 내부 문서를 쓰게 해 주고 답에 출처를 붙일 수 있게 하지만, 두 단계 각각에서 틀릴 수 있다. 검색이 관련 문서를 놓칠 수 있고, 모델이 검색된 문서에 없는 내용을 덧붙이거나 문서의 내용을 잘못 옮길 수 있다.

**tool use.** 모델이 계산기·코드 실행기·데이터베이스 같은 외부 도구를 호출하는 형식의 텍스트를 내면, 실행 환경이 그 호출을 수행하고 결과를 다시 문맥에 넣는다. 산술이나 단위 변환처럼 토큰 예측으로 하기에 약한 일을 정확한 도구에 맡기는 방식이다. 코딩 에이전트와 MCP 같은 도구 연결의 실제는 4.3장과 4.5장에서 다룬다.

## 평가

LLM 평가는 고전적인 머신러닝 평가보다 어렵다. 세 가지 이유가 있다.

- **오염**(contamination). 사전학습 말뭉치는 웹의 상당 부분을 담고, 공개 벤치마크의 문제와 정답도 웹에 있다. 평가 문항이 학습 데이터에 들어 있었다면 높은 점수는 일반화가 아니라 기억이다. 15.4장의 누수와 같은 문제지만, 학습 데이터가 공개되지 않은 모델이 많아 확인 자체가 어렵다. 공개 이후에 만들어진 문항, 기존 문항의 수치를 바꾼 변형 문항, 비공개 평가 세트가 대응책이다.
- **출력 형식과 채점.** 자유 서술 답은 정답과 글자가 달라도 맞을 수 있고, 같아도 틀린 추론일 수 있다. 다른 LLM을 채점자로 쓰는 방식은 편리하지만, 채점자 자신의 편향(길이가 긴 답, 자기와 비슷한 문체를 선호하는 경향 등)이 들어갈 수 있으므로 사람이 채점한 표본과 일치도를 확인한다.
- **설정 민감도.** prompt 형식, 예시 수, 디코딩 설정이 바뀌면 점수가 바뀐다. 두 모델을 비교할 때는 같은 설정을 쓰고, 설정을 바꾼 변형 여러 개에서 결론이 유지되는지 본다.

perplexity는 토큰 단위의 확률만 재므로 계산이 깔끔하지만, 실제 과제의 정확성과 바로 이어지지는 않는다. 과학 연구에서 쓸 때는 **자기 과제의 정답을 아는 작은 평가 세트**를 직접 만드는 것이 가장 신뢰할 만한 평가다. 공개 벤치마크의 점수는 그 벤치마크의 문항 분포에 대한 정보일 뿐이다.

## 과학 연구에서의 LLM

LLM은 연구에서 두 가지 자리에 놓인다.

**연구 도구로서.** 문헌 검색과 요약(23.1장), 코드 작성과 디버깅(4.3장), 데이터 형식 변환, 실험 기록 정리에 쓴다. 공통 원칙은 하나다. **LLM의 출력은 검증 전까지 가설이다.**

- 인용은 원문을 직접 열어 확인한다. 존재하지 않는 논문, 실제 논문에 없는 수치, 저자·연도가 섞인 인용이 생성될 수 있다.
- 코드는 테스트로 확인한다. 실행된다는 것은 맞다는 뜻이 아니다. 알려진 답이 있는 작은 입력으로 결과를 확인한다.
- 수치와 단위는 원 데이터에서 다시 계산한다. 토큰화에서 본 것처럼 숫자는 언어모델이 다루기 어려운 대상이다.
- 사용한 모델·설정·prompt를 연구 기록에 남긴다(23.7장). 같은 모델 이름도 시점에 따라 다른 모델일 수 있다.

**연구 대상으로서.** 과학 서열(단백질, DNA, 분자 문자열)에 같은 자기회귀 학습을 적용하거나, LLM의 내부 표현에 무엇이 담겼는지 분석한다(18.8장의 probing). 이때는 일반적인 머신러닝 연구의 규칙이 그대로 적용된다. 학습–평가 분할이 누수를 막는지(예: 서열 유사도가 높은 단백질이 train과 test에 갈라져 들어가지 않았는지), baseline과 비교했는지, 대조 실험이 있는지를 확인한다.

:::note[분야 예시]
재료 과학 문헌에서 합성 조건(온도, 시간, 전구체)을 뽑아 데이터베이스를 만드는 일에 LLM을 쓸 수 있다. 사람이 하던 정보 추출을 빠르게 하지만, 단위 누락(°C와 K의 혼동), 표와 본문의 값 불일치, 여러 실험 조건이 한 문단에 섞인 경우의 오추출이 생긴다. 신뢰할 만한 절차는 무작위 표본 수십 건을 사람이 직접 추출해 정답 세트를 만들고, 필드별 정밀도·재현율을 보고한 뒤에 전체에 적용하는 것이다. 오류율을 모르는 자동 추출 데이터로 학습한 모델은 그 오류까지 학습한다.
:::

## 실습

실습은 두 부분이다. 첫째, 본문의 cell 2와 cell 3을 이어서 **작은 언어모델을 학습하고 하한과 비교한다.** 코드는 `labs/architectures/language-models/cells.py`다. 반복 주기(`PERIOD`)를 8에서 12로 바꾸면 문맥 8인 MLP는 어떻게 되는지, 돌연변이 확률을 바꾸면 하한이 어떻게 움직이는지 예측한 뒤 실행해 확인한다. 문맥 길이가 의존성 거리에 닿지 못하는 모델은 아무리 오래 학습해도 균등 분포에서 벗어나지 못한다.

둘째, **도메인 문서로 RAG를 구성해 답변을 검증한다.** 이 부분은 사용할 모델과 검색 도구가 환경마다 다르므로 절차만 적는다.

1. 자기 분야의 문서 10–30개를 고르고, 문서 안에 답이 있는 질문 20개와 정답(문서 위치 포함)을 사람이 먼저 만든다. 문서에 답이 없는 질문도 5개 넣는다.
2. 검색만 따로 평가한다. 각 질문에 대해 정답 문서가 검색 상위 $k$개에 들어오는 비율을 잰다.
3. 검색 결과를 넣은 생성 답변을 평가한다. 답이 정답과 맞는지, 인용한 문단이 그 답을 실제로 뒷받침하는지를 따로 채점한다. 답이 없는 질문에 "문서에 없다"고 답하는 비율도 잰다.
4. 검색 없이 같은 질문을 던진 결과와 비교해, RAG가 무엇을 고쳤고 무엇을 고치지 못했는지 정리한다.

이어서 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="sequence-perplexity"}

## 실패 사례

- **LLM 출력을 사실 확인 없이 연구 근거로 쓰는 것.** 그럴듯한 인용, 정확해 보이는 수치, 실행되는 코드는 모두 확인 대상이다. 다음 토큰 예측은 "그럴듯한 이어 쓰기"를 학습한 것이지 "참인 문장"을 학습한 것이 아니다.
- **tokenizer가 다른 모델의 perplexity를 그대로 비교하는 것.** 토큰당 값은 토큰 크기에 따라 달라진다. 문자당·바이트당 값으로 바꾸거나 같은 tokenizer를 쓴다.
- **train perplexity나 오염된 벤치마크 점수를 일반화 성능으로 보고하는 것.** 하한 아래의 train perplexity는 암기의 신호이고, 학습 데이터에 들어 있었을 수 있는 평가 문항의 점수는 기억을 잰 것일 수 있다.
- **temperature를 0으로 두면 정확해진다고 생각하는 것.** temperature는 순위를 바꾸지 않는다. 일관성과 정확성은 다르다.

## 통과 기준

- 작은 말뭉치에서 BPE 병합을 직접 수행하고, 수치·화학식 같은 도메인 문자열이 어떻게 토큰화되는지 보인다.
- 자료 생성 과정의 하한과 비교해 n-gram 모델과 신경망 언어모델의 train·test perplexity를 해석하고, 연습문제 `sequence-perplexity`의 테스트를 통과한다.
- 사전학습·instruction tuning·RLHF·DPO의 데이터와 손실을 구분해 설명하고, 자기 연구에서 LLM 출력을 검증하는 절차(정답 세트, 인용 확인, 설정 기록)를 쓴다.

## 참고 문헌

- Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL 2016 (Volume 1: Long Papers)*, 1715–1725. — 하위 단어 단위와 BPE 병합 학습
- Holtzman, Buys, Du, Forbes, Choi (2020). The Curious Case of Neural Text Degeneration. *ICLR 2020*. — 최대화 기반 디코딩의 반복 문제와 nucleus 샘플링
- Ouyang 외 (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. — 지시 미세조정과 RLHF의 절차
- Rafailov, Sharma, Mitchell, Manning, Ermon, Finn (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*. — 보상 모델 없는 선호 최적화
- Lewis 외 (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. — 검색과 생성의 결합
