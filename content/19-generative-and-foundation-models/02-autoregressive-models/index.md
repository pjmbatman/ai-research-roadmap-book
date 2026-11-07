---
id: generative-and-foundation-models/autoregressive-models
title: "자기회귀 모델"
title_en: "Autoregressive Models"
tier: core
status: draft
prerequisites: [generative-and-foundation-models/generative-modeling]
---

# 자기회귀 모델

고차원 분포 $p(\mathbf{x})$를 직접 모델링하기는 어렵다. 그런데 확률의 곱셈 규칙을 반복하면 결합분포는 **1차원 조건부분포들의 곱**으로 정확히 쪼개진다. 1차원 조건부분포는 분류(이산값)나 회귀(연속값) 모델로 다룰 수 있다. 자기회귀 모델(autoregressive model)은 이 분해를 그대로 모델로 삼는다. 근사가 들어가는 곳은 각 조건부분포를 신경망으로 표현하는 부분뿐이므로, 이 계열은 19.1장의 표에서 **정확한 likelihood**를 주는 몇 안 되는 계열이다. 언어모델(18.5장)이 가장 유명한 예이고, 이미지·음성·분자 서열·시계열에도 같은 원리가 쓰인다.

대가도 분명하다. 생성할 때는 한 차원(토큰)씩 순서대로 뽑아야 하고, 학습 때와 생성 때 모델이 보는 입력이 다르다. 이 장은 분해, 그것을 구현하는 구조, 샘플링 전략, 연속값으로의 확장을 차례로 다루고, 마지막으로 학습과 생성의 조건 차이가 만드는 실패를 직접 재현한다.

## 학습 목표

- chain rule 분해로 분포를 모델링한다
- 샘플링 전략을 이해한다
- teacher forcing과 노출 편향(exposure bias)을 진단한다

## chain rule 분해

**정의.** $d$차원 확률변수 $\mathbf{x} = (x_1, \dots, x_d)$의 결합분포는 조건부확률의 정의(11.3장)를 반복 적용해

$$
p(\mathbf{x}) = p(x_1)\,p(x_2 \mid x_1)\,p(x_3 \mid x_1, x_2)\cdots p(x_d \mid x_1, \dots, x_{d-1}) = \prod_{i=1}^{d} p(x_i \mid \mathbf{x}_{<i})
$$

로 쓸 수 있다. 여기서 $\mathbf{x}_{<i} = (x_1, \dots, x_{i-1})$이고 $\mathbf{x}_{<1}$은 빈 조건이다. 이 식은 **근사가 아니라 항등식**이고, 어떤 순서로 변수를 나열해도 성립한다. 자기회귀 모델은 각 조건부분포를 파라미터 $\theta$를 공유하는 하나의 신경망으로 표현한다.

$$
\log p_\theta(\mathbf{x}) = \sum_{i=1}^{d} \log p_\theta(x_i \mid \mathbf{x}_{<i}).
$$

log-likelihood가 조건부 로그확률의 합이므로 정확히 계산된다. 이산 토큰이면 각 항이 분류의 교차엔트로피이고, 최대우도 학습은 "다음 토큰 맞히기"의 교차엔트로피 합을 줄이는 것과 같다.

**코드로 확인.** 이진 변수 3개의 결합분포를 무작위로 정하고, 두 가지 순서로 조건부분포를 계산해 곱한다.

```python {exec="browser" session="ar"}
import numpy as np
from itertools import product

rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)    # (2, 2, 2) 이진 변수 3개의 결합분포

def chain_rule_prob(x, order):
    """p(x) = prod_i p(x_{o_i} | x_{o_1}, ..., x_{o_{i-1}})"""
    prob, fixed = 1.0, {}
    for i in order:
        def marg(assign):                         # 고정된 변수만 맞추고 나머지는 더한다
            return sum(P[s] for s in product([0, 1], repeat=3)
                       if all(s[j] == v for j, v in assign.items()))
        prob *= marg({**fixed, i: x[i]}) / marg(fixed)
        fixed[i] = x[i]
    return prob

for order in [(0, 1, 2), (2, 0, 1)]:
    err = max(abs(chain_rule_prob(x, order) - P[x]) for x in product([0, 1], repeat=3))
    print(f"순서 {order}: 최대 오차 < 1e-12: {err < 1e-12}")
x = (1, 0, 1)
terms = [P[1].sum(), P[1, 0].sum() / P[1].sum(), P[1, 0, 1] / P[1, 0].sum()]
print("log p(x) =", np.round(np.log(P[x]), 4), "= 조건부 로그의 합", np.round(np.sum(np.log(terms)), 4))
```

```output
순서 (0, 1, 2): 최대 오차 < 1e-12: True
순서 (2, 0, 1): 최대 오차 < 1e-12: True
log p(x) = -1.185 = 조건부 로그의 합 -1.185
```

두 순서 모두 결합분포를 반올림 오차 수준으로 복원한다. 분해 자체에는 정보 손실이 없다.

::quiz{id="ar-loglik-sum"}

**순서의 선택.** 분해는 어떤 순서로도 정확하지만, **모델링 난이도**는 순서마다 다르다. 시간에 따라 인과적으로 생성되는 시계열이나 문장처럼 자연스러운 순서가 있으면 그 순서가 조건부분포를 단순하게 만든다. 이미지에는 자연스러운 순서가 없어 보통 래스터 순서(왼쪽 위에서 오른쪽 아래로)를 쓴다. Pixel RNN(van den Oord, Kalchbrenner, Kavukcuoglu, 2016)이 픽셀을 이 순서로 하나씩 예측한 대표적인 예다. 분자 그래프나 3차원 구조처럼 순서가 정의되지 않는 데이터는 순서를 정하는 규칙 자체가 설계 대상이 된다.

**파라미터 수.** 조건부분포를 표로 저장하면 $i$번째 이진 변수의 조건부분포는 $2^{i-1}$개의 숫자가 필요하고, 합은 $1 + 2 + \cdots + 2^{d-1} = 2^d - 1$로 결합분포 표 전체와 같다. 분해만으로 계산량이 줄지는 않는다. 줄이는 방법은 두 가지다. 조건을 최근 $k$개로 자르는 Markov 가정($n$-gram, AR($k$))은 파라미터를 크게 줄이지만 분해가 더 이상 정확하지 않다. 신경망으로 모든 조건부분포를 하나의 함수 $f_\theta(\mathbf{x}_{<i})$로 공유하는 방법은 전체 이력을 조건으로 쓰면서도 파라미터 수를 고정한다. 현대 자기회귀 모델은 후자다.

## 구조

**왜 필요한가.** 조건부분포 $d$개를 신경망 $d$개로 따로 계산하면 학습이 $d$배 느리다. 필요한 것은 입력 $\mathbf{x}$ 전체를 한 번 통과시켜 $d$개의 조건부분포를 **동시에** 내되, $i$번째 출력이 $\mathbf{x}_{<i}$에만 의존하도록 보장하는 구조다. 이 성질을 인과성(causality) 또는 자기회귀 성질이라 한다.

**구현 방법.** 공통 아이디어는 가중치나 연결에 **mask**를 씌워 미래 입력으로 가는 경로를 끊는 것이다.

- **masked linear(MADE 류)**: 가중치 행렬의 대각선과 그 위를 0으로 만든다.
- **masked convolution(PixelCNN 류)**: convolution 커널에서 현재 픽셀과 그 이후 픽셀에 해당하는 위치를 0으로 만든다(18.2장).
- **causal Transformer**: attention 점수 행렬에서 미래 위치를 $-\infty$로 가려, softmax 뒤의 가중치를 0으로 만든다(18.4장).

인과성은 **Jacobian으로 확인**할 수 있다. $i$번째 출력을 $j \geq i$인 입력으로 미분한 값이 모두 0이어야 하므로, Jacobian은 대각선을 포함한 위쪽이 0인 하삼각행렬이다. 아래 코드는 대각선 아래만 남긴 가중치로 만든 작은 층의 Jacobian을 유한차분으로 계산한다.

```python {exec="browser" session="ar"}
d = 5
mask = np.tril(np.ones((d, d)), k=-1)             # (d, d) 대각선 아래만 1: t는 0..t-1만 본다
W = rng.normal(size=(d, d)) * mask                # (d, d) 가려진 가중치

def logits(x):                                    # x: (d,) -> (d,) 모든 위치의 조건부 logit을 한 번에
    return W @ np.tanh(x) + 0.1

x0 = rng.normal(size=d)
J = np.zeros((d, d))                              # J[t, s] = d logit_t / d x_s (유한차분)
for s in range(d):
    e = np.zeros(d); e[s] = 1e-6
    J[:, s] = (logits(x0 + e) - logits(x0)) / 1e-6
print("출력 t가 입력 s >= t에 의존하지 않음:", np.all(np.abs(np.triu(J)) < 1e-12))
print("의존 패턴 (1 = 의존):")
print((np.abs(J) > 1e-8).astype(int))
```

```output
출력 t가 입력 s >= t에 의존하지 않음: True
의존 패턴 (1 = 의존):
[[0 0 0 0 0]
 [1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [1 1 1 1 0]]
```

**teacher forcing.** 학습할 때는 데이터 $\mathbf{x}$ 전체를 넣고, 위치마다 **실제 데이터의 이전 값**을 조건으로 다음 값의 로그확률을 계산한다. 이 방식을 teacher forcing이라 한다. mask 덕분에 $d$개 위치의 손실이 순전파 한 번으로 모두 계산되므로 학습은 병렬이다. 반면 생성할 때는 $x_1$을 뽑고, 그것을 넣어 $x_2$를 뽑는 식으로 **자기 출력을 다음 입력으로** 써야 하므로 순전파를 $d$번 해야 한다. 학습은 빠르고 생성은 느리다는 비대칭이 이 계열의 특징이다.

::quiz{id="causal-mask-parallel"}

:::note[분야 예시]
단백질 서열은 아미노산 20종의 문자열이므로 언어모델과 같은 방식으로 자기회귀 모델을 학습할 수 있다. 학습된 모델의 log-likelihood $\log p_\theta(\mathbf{x})$는 서열마다 정확히 계산되므로, 돌연변이 서열과 원래 서열의 log-likelihood 차이를 "이 변이가 진화적으로 그럴듯한가"의 점수로 쓰는 연구가 있다. 정확한 likelihood가 생성뿐 아니라 **평가 도구**로 쓰이는 예다.
:::

## 샘플링

**왜 필요한가.** 학습된 조건부분포 $p_\theta(x_i \mid \mathbf{x}_{<i})$에서 그대로 뽑는 것(ancestral sampling)이 모델 분포에서 뽑는 정확한 방법이다. 그런데 실제로는 분포를 의도적으로 바꿔서 뽑는 경우가 많다. 모델 분포의 긴 꼬리에 있는 낮은 확률 토큰들이 누적되면 생성 결과의 품질을 떨어뜨리기 때문이다.

**정의.** 다음 토큰의 logit 벡터를 $\mathbf{z}$라 하자.

- **greedy**: 매 step 가장 확률이 큰 토큰 $\arg\max_i z_i$를 고른다. 결정론적이다.
- **온도**(temperature) $T$: $p_i \propto \exp(z_i / T)$. $T < 1$이면 분포가 뾰족해지고 $T \to 0$이면 greedy가 된다. $T > 1$이면 평평해지고 $T \to \infty$이면 균등분포가 된다.
- **top-k**: 확률 상위 $k$개만 남기고 다시 정규화한다.
- **nucleus(top-p)**: 확률이 큰 순서로 누적확률이 처음 $p$ 이상이 되는 가장 작은 집합만 남긴다(Holtzman et al., 2020). 분포가 뾰족한 step에서는 후보가 적고, 평평한 step에서는 많다. top-k의 $k$가 고정된 것과 다르다.

**코드로 확인.** 10개 토큰의 분포에 다섯 가지 전략을 적용하고, 남는 후보 수와 엔트로피를 비교한다.

```python {exec="browser" session="ar"}
def filtered(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)                        # 확률이 큰 순서
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:                         # 누적확률이 top_p에 처음 도달할 때까지
        n_keep = np.searchsorted(np.cumsum(p[order]), top_p) + 1
        keep[order[n_keep:]] = False
    p = np.where(keep, p, 0.0)
    return p / p.sum()

lg = np.log(np.array([0.40, 0.20, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01, 0.01]))
def entropy(p):
    return -np.sum(p[p > 0] * np.log(p[p > 0]))
for name, kw in [("T=1", {}), ("T=0.5", {"temperature": 0.5}), ("T=2", {"temperature": 2.0}),
                 ("top-k=3", {"top_k": 3}), ("top-p=0.9", {"top_p": 0.9})]:
    p = filtered(lg, **kw)
    print(f"{name:10s} 후보 {np.sum(p > 0):2d}개  최고 확률 {p.max():.3f}  엔트로피 {entropy(p):.3f}")
```

```output
T=1        후보 10개  최고 확률 0.400  엔트로피 1.790
T=0.5      후보 10개  최고 확률 0.697  엔트로피 1.007
T=2        후보 10개  최고 확률 0.230  엔트로피 2.153
top-k=3    후보  3개  최고 확률 0.556  엔트로피 0.981
top-p=0.9  후보  6개  최고 확률 0.440  엔트로피 1.514
```

$T = 0.5$는 확률을 제곱한 뒤 정규화한 것과 같아 최고 확률이 0.40에서 0.70으로 커진다. top-p = 0.9는 누적확률 0.91에 도달하는 6개를 남긴다. 이 분포에서는 top-k = 3보다 후보가 많지만, 더 뾰족한 분포였다면 후보가 1–2개로 줄었을 것이다.

::quiz{id="nucleus-size"}

**연구에서의 의미.** 온도·top-k·top-p로 바꾼 분포는 **더 이상 학습한 모델의 분포가 아니다.** 품질이 좋아 보이는 표본을 얻는 대가로 다양성을 잃고, log-likelihood로 평가한 모델과 실제로 표본을 뽑는 분포가 달라진다. Holtzman et al.(2020)은 확률을 최대화하는 디코딩(greedy, beam search)이 반복적이고 단조로운 텍스트를 만든다는 것을 보이고, 꼬리를 잘라 내는 nucleus sampling을 제안했다. 과학 데이터를 생성할 때도 같은 선택이 나타난다. 새로운 분자 후보를 넓게 탐색하려면 높은 온도가, 가장 그럴듯한 후보 몇 개를 원하면 낮은 온도가 맞다. 어느 쪽이든 논문에는 샘플링 설정을 함께 보고해야 표본의 다양성 통계를 해석할 수 있다.

## 연속값의 자기회귀

**왜 필요한가.** 측정 시계열, 파형, 좌표처럼 각 차원이 연속값이면 조건부분포가 범주형이 아니다. 두 가지 방법이 있다.

**분포 head.** 신경망이 조건부분포의 파라미터를 출력한다. 가장 단순한 것은 가우시안 head $p_\theta(x_i \mid \mathbf{x}_{<i}) = \mathcal{N}\big(\mu_\theta(\mathbf{x}_{<i}), \sigma_\theta^2(\mathbf{x}_{<i})\big)$이고, 조건부분포가 여러 봉우리를 가지면 혼합 가우시안 head를 쓴다(12.3장의 분포를 출력하는 모델). 가우시안 head의 음의 로그우도는

$$
-\log p_\theta(x_i \mid \mathbf{x}_{<i}) = \frac{(x_i - \mu_\theta)^2}{2\sigma_\theta^2} + \frac{1}{2}\log(2\pi\sigma_\theta^2)
$$

이고, $\sigma$를 고정하면 제곱오차 회귀와 같다. 고전적인 AR($p$) 모형(11.7장)은 $\mu$가 직전 $p$개 값의 선형함수이고 $\sigma$가 상수인 특수한 경우다.

**이산화.** 값의 범위를 $B$개 구간으로 나누고 범주형 분포로 다룬다. 봉우리가 여럿인 조건부분포나 비대칭 분포를 모양 가정 없이 표현할 수 있다는 장점이 있고, 구간 수만큼 출력이 커지고 이웃 구간이 가깝다는 사실을 모델이 따로 배워야 한다는 단점이 있다. 이산화한 모델의 NLL은 구간의 **확률**이고 가우시안 head의 NLL은 **밀도**이므로, 두 값을 비교하려면 구간 폭 $\Delta$에 대해 $\log\Delta$만큼 보정해야 한다(밀도 × 폭 ≈ 확률).

**코드로 확인.** 진폭이 스스로 제한되는 비선형 진동(극한 순환, limit cycle)에 잡음이 더해진 시계열을 만든다. 직전 두 값을 입력으로 하는 가우시안 head 모델 두 개를 teacher forcing(최소제곱 = 고정 분산 가우시안의 최대우도)으로 맞춘다. 하나는 선형 AR(2), 다른 하나는 참 과정의 세제곱 항을 포함한 모델이다.

```python {exec="browser" session="ar"}
def simulate(n, R, rng, noise=0.02):              # 참 과정: 진폭이 스스로 제한되는 비선형 진동
    x = np.zeros((R, n)); x[:, 0], x[:, 1] = 1.0, 0.95
    for t in range(2, n):
        a, b = x[:, t - 1], x[:, t - 2]
        x[:, t] = 1.9 * a - 0.95 * b + 0.15 * (a - a**3) + noise * rng.normal(size=R)
    return x                                      # (R, n)

def feats(a, b, cubic):                           # 입력: 직전 두 값 (teacher forcing)
    return np.stack([a, b] + ([a**3] if cubic else []), axis=-1)

x_tr, x_te = simulate(3000, 1, rng)[0], simulate(1000, 1, rng)[0]
models = {}
for name, cubic in [("선형 AR(2)", False), ("AR(2)+세제곱", True)]:
    w = np.linalg.lstsq(feats(x_tr[1:-1], x_tr[:-2], cubic), x_tr[2:], rcond=None)[0]
    sig = (x_tr[2:] - feats(x_tr[1:-1], x_tr[:-2], cubic) @ w).std()   # head의 표준편차
    r = x_te[2:] - feats(x_te[1:-1], x_te[:-2], cubic) @ w
    nll = np.mean(0.5 * r**2 / sig**2 + 0.5 * np.log(2 * np.pi * sig**2))
    models[name] = (w, sig, cubic)
    print(f"{name:12s} 계수 {np.round(w, 3)}  1-step RMSE {np.sqrt(np.mean(r**2)):.4f}  test NLL/step {nll:.3f}")
```

```output
선형 AR(2)     계수 [ 1.858 -0.864]  1-step RMSE 0.0312  test NLL/step -2.048
AR(2)+세제곱    계수 [ 2.046 -0.947 -0.148]  1-step RMSE 0.0200  test NLL/step -2.491
```

두 모델 모두 1-step RMSE가 0.02–0.03으로, 신호의 진폭(약 1)에 비해 작다. 세제곱 항이 있는 모델은 참 잡음 0.02를 그대로 복원하고, test NLL도 step당 0.44 nats 낮다. 이 숫자들은 test 데이터에서 정확히 계산한 log-likelihood다.

:::note[분야 예시]
기상·기후 분야의 데이터 기반 예보 모델은 현재 대기 상태에서 일정 시간 뒤의 상태를 예측하는 함수를 학습하고, 그 출력을 다시 입력으로 넣어 며칠 뒤까지 굴린다(rollout). 구조적으로 자기회귀 모델이다. 한 step 오차가 작아도 수십 step을 굴리면 오차가 쌓이므로, 이런 모델은 1-step 오차와 함께 긴 rollout의 안정성과 통계(에너지 스펙트럼, 극값 분포)를 따로 평가한다. 다음 절의 실습이 그 이유를 보여 준다.
:::

## 실습

앞 절의 두 모델로 시계열을 **자유 생성**(free-running generation)한다. 같은 초기값에서 출발해 매 step 가우시안 head에서 표본을 뽑고 그 값을 다음 입력으로 넣는다. 샘플링 온도 $\tau$는 head의 표준편차를 $\tau\sigma$로 바꾼다. 비교 지표는 두 가지다. $|x|$의 95% 분위수(진폭 분포)와, 같은 시각에서 200개 경로 사이의 표준편차(다양성)다.

```python {exec="browser" session="ar"}
def generate(w, sig, cubic, R=200, H=400, tau=1.0, seed=3):
    g = np.random.default_rng(seed)
    xs = np.zeros((R, H)); xs[:, 0], xs[:, 1] = 1.0, 0.95
    for t in range(2, H):                         # 자기 출력을 다음 입력으로 쓴다
        xs[:, t] = feats(xs[:, t - 1], xs[:, t - 2], cubic) @ w + tau * sig * g.normal(size=R)
    return xs                                     # (R, H)

q95 = lambda xs: np.quantile(np.abs(xs[:, 200:]), 0.95)
print(f"참 과정        |x|의 95% 분위수 {q95(simulate(400, 200, np.random.default_rng(3))):.3f}")
for name, (w, sig, cubic) in models.items():
    print(f"{name:12s}  |x|의 95% 분위수 {q95(generate(w, sig, cubic)):.3f}")
w, sig, cubic = models["AR(2)+세제곱"]
for tau in [0.5, 1.0, 2.0]:
    xs = generate(w, sig, cubic, tau=tau)
    print(f"온도 {tau}: t=100에서 경로 간 표준편차 {xs[:, 100].std():.3f}  |x| 95% 분위수 {q95(xs):.3f}")
```

```output
참 과정        |x|의 95% 분위수 1.007
선형 AR(2)      |x|의 95% 분위수 1.554
AR(2)+세제곱     |x|의 95% 분위수 1.004
온도 0.5: t=100에서 경로 간 표준편차 0.066  |x| 95% 분위수 0.921
온도 1.0: t=100에서 경로 간 표준편차 0.365  |x| 95% 분위수 1.004
온도 2.0: t=100에서 경로 간 표준편차 0.731  |x| 95% 분위수 1.136
```

결과를 읽는 법은 세 가지다.

- **teacher forcing 지표는 순위를 맞게 매겼지만 실패의 모양을 보여 주지 않았다.** 선형 AR(2)의 1-step RMSE는 0.03으로 작아 보였지만, 자유 생성한 경로의 진폭 95% 분위수는 1.55로 참 과정(1.01)보다 55% 크다. 선형 모델에는 진폭을 되돌리는 항이 없어서, 생성 중에 자기 표본이 만든 작은 편차가 쌓여 데이터에 없던 큰 진폭으로 흘러간다.
- **올바른 구조의 모델은 진폭 분포를 재현한다**(1.004 대 1.007). 1-step 성능이 좋다는 것과 생성 분포가 맞는다는 것은 별개의 확인이다.
- **온도는 다양성과 충실도를 맞바꾼다.** 온도 0.5에서는 경로들이 거의 같은 위상을 유지해(표준편차 0.07) 다양성이 사라지고, 온도 2에서는 진폭 분포가 참 과정보다 넓어진다(1.14). 온도 1이 학습한 분포 그대로다.

이어서 연습문제로 온도와 nucleus 필터를 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="top-p-filter"}

## 실패 사례

- **학습과 생성의 조건 차이(exposure bias)를 무시하는 것.** teacher forcing으로 학습한 모델은 언제나 **실제 데이터의 이력**을 조건으로만 학습했다. 생성할 때는 자기 표본이 만든 이력을 조건으로 받는데, 그 이력은 작은 오차 때문에 학습 데이터의 분포에서 조금씩 벗어나고, 모델은 그런 입력에서의 행동을 배운 적이 없다. 이 불일치를 노출 편향이라 부른다. 증상은 실습에서 본 것처럼 1-step 지표는 좋은데 rollout 통계가 틀리는 것이다. 진단하려면 반드시 자유 생성한 결과를 데이터와 같은 통계로 비교한다. 완화 방법으로는 학습 중 일부 step에서 모델 자신의 출력을 입력으로 쓰는 scheduled sampling(Bengio et al., 2015), 여러 step을 굴린 뒤 손실을 거는 multi-step 학습, 입력에 잡음을 더하는 학습이 쓰인다.
- **샘플링 설정을 바꾼 표본으로 모델의 분포를 평가하는 것.** 낮은 온도·top-k·top-p로 뽑은 표본은 다양성이 줄어든 별개의 분포다. 그 표본으로 "모델이 데이터의 다양성을 재현한다/못한다"고 결론 내리면 모델이 아니라 디코딩 설정을 평가한 것이다.
- **이산화한 모델과 연속 밀도 모델의 NLL을 그대로 비교하는 것.** 하나는 확률, 하나는 밀도다. 구간 폭 보정 $\log\Delta$ 없이는 숫자의 크기가 의미를 갖지 않는다.
- **mask를 한 칸 어긋나게 거는 것.** 출력 $i$가 입력 $x_i$ 자신을 보면 학습 손실은 0에 가깝게 떨어지지만 생성은 완전히 실패한다. 학습 손실이 의심스럽게 낮으면 Jacobian의 대각선이 0인지부터 확인한다.

## 통과 기준

- 이산 분포에서 chain rule 분해로 계산한 확률이 결합분포와 일치함을 보이고, 자기회귀 모델의 log-likelihood가 조건부 로그확률의 합인 이유를 설명한다.
- 연속 시계열에 가우시안 head 자기회귀 모델을 맞춰 test log-likelihood를 계산하고, 자유 생성한 경로의 통계를 데이터와 비교해 teacher forcing 지표만으로는 보이지 않는 차이를 보고한다.
- 연습문제 `top-p-filter`의 테스트를 통과한다.

## 참고 문헌

- van den Oord, Kalchbrenner, Kavukcuoglu (2016). Pixel recurrent neural networks. *ICML 2016*, PMLR 48, 1747–1756. — 픽셀을 순서대로 예측하는 이미지 자기회귀 모델
- Holtzman, Buys, Du, Forbes, Choi (2020). The curious case of neural text degeneration. *ICLR 2020*. arXiv:1904.09751. — 최대 확률 디코딩의 반복 문제와 nucleus sampling
- Bengio, Vinyals, Jaitly, Shazeer (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NeurIPS 28*, 1171–1179. — 학습 중 모델 출력을 입력으로 섞어 학습–생성 불일치를 줄이는 방법
