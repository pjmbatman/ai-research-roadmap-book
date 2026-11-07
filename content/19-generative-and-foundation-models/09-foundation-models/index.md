---
id: generative-and-foundation-models/foundation-models
title: "사전학습·전이·Foundation Model"
title_en: "Pretraining, Transfer, and Foundation Models"
tier: advanced
status: draft
prerequisites: [generative-and-foundation-models/self-supervised-learning, information-and-learning-theory/modern-generalization]
---

# 사전학습·전이·Foundation Model

"○○ 분야의 foundation model"이라는 이름을 단 모델이 과학 분야마다 나오고 있다. 이름이 같다고 같은 것을 약속하지는 않는다. 어떤 모델은 수십 가지 하위 과제에서 전이 효과가 검증되었고, 어떤 모델은 큰 데이터로 학습했다는 것 외에 확인된 것이 없다. 연구자는 두 가지 입장에서 이 문제를 만난다. 남이 만든 사전학습 모델을 자기 문제에 **쓸지, 어떻게 쓸지** 정해야 하고, 자기 모델을 foundation model이라 **주장할 때** 무엇을 보여야 하는지 알아야 한다.

이 장은 19.8장의 자기지도학습과 14.5장의 스케일링 법칙 위에서, 사전학습한 모델을 다른 과제로 옮기는 **전이**(transfer)의 방법과 평가를 다룬다. 파라미터와 메모리를 계산해 전이 방법의 비용을 비교하고, LoRA의 저랭크 갱신을 NumPy로 적합해 보고, 원천 과제와 닮지 않은 과제에서 전이가 오히려 해가 되는 **부정적 전이**를 만들어 본다. 마지막으로 평가 자료가 사전학습 자료와 겹치는지 점검하는 방법을 서열 자료로 확인한다.

## 학습 목표

- foundation model의 조건(넓은 사전학습, 여러 하위 과제로의 적응, 검증된 전이 범위)을 설명하고 단지 큰 모델과 구분한다
- full fine-tuning·PEFT(LoRA·adapter)·고정 특징·prompting 중 전이 방법을 데이터 크기와 자원에 맞게 선택한다
- 전이 효과와 부정적 전이를 대조 실험과 데이터 중복 점검으로 평가한다

## foundation model이란

**정의.** Bommasani 외(2021)는 넓은 데이터로 (대개 대규모 자기지도학습으로) 학습되어, 미세조정 같은 **적응**(adaptation)을 거쳐 폭넓은 하위 과제에 쓰일 수 있는 모델을 foundation model이라 불렀다. 이름에는 두 가지가 담겨 있다. 많은 응용이 그 위에 세워지는 **토대**라는 점, 그리고 토대 자체는 **미완성**이어서 적응 없이는 쓸 수 없다는 점이다. 저자들은 동시에 하나의 모델이 많은 응용의 공통 토대가 되면(동질화, homogenization) 그 모델의 결함이 모든 하위 모델로 물려진다는 위험도 강조했다.

**조건으로 다시 쓰기.** 이 정의를 검증할 수 있는 조건으로 바꾸면 다음과 같다.

| 조건 | 확인할 것 |
|---|---|
| 넓은 사전학습 | 사전학습 자료가 어떤 출처·modality·조건을 포함하는가. 그 범위가 문서로 공개되어 있는가 |
| 적응 가능성 | 사전학습 목적과 다른 하위 과제에 적은 레이블·적은 계산으로 적응되는가 |
| 검증된 전이 범위 | 어떤 하위 과제들에서, 어떤 baseline과 비교해, 얼마나 이득이 있었는가 |
| 실패 범위 | 어디서 전이가 실패하거나 해가 되는가 |

**큰 모델과의 구분.** 파라미터가 많다는 것은 이 조건 중 어느 것도 보장하지 않는다. 한 데이터셋의 한 과제로 학습한 큰 모델은 큰 모델일 뿐이다. 반대로 비교적 작은 모델이라도 넓은 자료로 사전학습되어 여러 과제에서 전이 효과가 검증되었다면 위의 조건을 더 잘 만족한다. foundation model이라는 주장은 크기가 아니라 **전이의 범위와 그 검증**에 대한 주장이다.

::quiz{id="foundation-vs-large"}

## 데이터 통합

**왜 어려운가.** 넓은 사전학습이란 결국 서로 다른 곳에서 온 자료를 하나의 학습에 넣는 일이다. 과학 자료는 출처마다 다르다.

- **modality**: 같은 대상을 영상·스펙트럼·서열·시계열·텍스트로 기록한다. 한 모델에 넣으려면 각 modality를 공통의 token이나 embedding으로 바꾸는 encoder가 필요하다.
- **단위와 척도**: 같은 물리량을 다른 단위로(K와 °C, eV와 kJ/mol), 다른 척도로(선형과 로그) 기록한다. 단위를 명시적으로 통일하고, 변수 이름에 단위를 남긴다.
- **프로토콜**: 측정 장비, 실험 절차, 계산 방법(예: 시뮬레이션의 근사 수준)이 다르면 같은 대상도 다른 값을 낸다. 이 차이가 대상의 차이보다 크면, 모델은 대상보다 **출처**를 먼저 배운다. 19.8장의 shortcut이 출처 단위로 나타나는 것이다.
- **비중**: 한 출처가 자료의 대부분을 차지하면 사전학습은 그 출처의 분포에 맞춰진다. 출처별 표본 수와 가중치를 기록하고, 필요하면 출처별로 재표집한다.

**연구에서의 의미.** 통합 자료의 품질은 사전학습 모델의 상한이다. 출처·단위·프로토콜·비중을 **메타데이터로 남겨** 두어야 나중에 전이가 실패했을 때 원인을 추적할 수 있고, 평가 자료가 사전학습 자료와 겹치는지도 점검할 수 있다(이 장의 평가 절).

:::note[분야 예시]
재료 물성 데이터베이스를 여러 개 합쳐 사전학습할 때, 데이터베이스마다 전자구조 계산의 근사 방법과 수렴 설정이 다를 수 있다. 같은 결정 구조의 형성 에너지가 데이터베이스에 따라 체계적으로 다르면, 모델은 구조가 아니라 "어느 데이터베이스에서 왔는가"를 예측하는 쪽으로 기운다. 출처를 입력에 명시하고, 출처를 바꿔 끼운 예측을 비교해 이 효과의 크기를 잰다.
:::

## 스케일링

**데이터·모델·계산의 배분.** 14.5장에서 본 스케일링 법칙은 모델 크기 $N_{\text{param}}$, 데이터 크기 $D$, 계산량 $C \approx 6N_{\text{param}}D$ 사이의 배분을 정하는 도구다. 예를 들어 파라미터 3억 3천만 개의 모델을 token 100억 개로 한 번 학습하는 데는 대략 $6 \times 3.35\times10^{8} \times 10^{10} \approx 2\times10^{19}$ FLOP이 든다. 이 계산은 같은 예산으로 더 작은 모델을 더 많은 자료로 학습하는 것과 비교할 때 출발점이 된다.

**과학 데이터의 사정.** 언어·이미지의 스케일링 연구는 자료가 사실상 무한한 상황을 가정하는 경우가 많다. 과학 분야에서는 자료가 먼저 바닥난다. 같은 자료를 여러 epoch 반복하면 새 자료만큼의 효과를 기대하기 어렵고, 시뮬레이션으로 자료를 늘리면 시뮬레이터의 근사가 모델에 들어간다. 그래서 과학 분야의 사전학습에서는 모델을 키우는 것보다 **자료의 다양성과 품질**을 늘리는 것이 더 큰 효과를 내는 경우가 많다. 14.5장의 경고도 그대로 적용된다. 작은 범위에서 맞춘 스케일링 곡선을 멀리 외삽하지 않고, 바닥(줄일 수 없는 오차)을 함께 추정한다.

## 전이 방법

**무엇을 학습하는가.** 전이 방법은 사전학습한 파라미터 중 무엇을 고정하고 무엇을 학습하는지로 나뉜다.

| 방법 | 학습하는 것 | 장점 | 약점 |
|---|---|---|---|
| full fine-tuning | 모든 파라미터 | 적응 능력이 가장 크다 | 메모리·저장 비용, 작은 자료에서 과적합, 사전학습 특징의 왜곡 |
| PEFT: adapter | 층 사이에 넣은 작은 병목 모듈 | 과제마다 작은 모듈만 저장 | 추론 시 추가 연산 |
| PEFT: LoRA | 가중치 행렬의 저랭크 갱신 $\mathbf{B}\mathbf{A}$ | 학습 파라미터가 매우 적고, 병합하면 추론 비용이 그대로 | rank가 표현할 수 있는 변화만 |
| 고정 특징 | 출력 head만 | 가장 싸고 안정적 | 사전학습 특징에 없는 정보는 얻지 못함 |
| prompting | 없음(입력만 설계) | 학습이 필요 없음 | 언어모델류에 한정, 설정에 민감(18.5장) |

**adapter.** Houlsby 외(2019)는 Transformer의 각 층에 차원을 줄였다 다시 늘리는 작은 병목 모듈을 넣고 그것만 학습해, BERT를 텍스트 분류 과제들로 전이한 실험에서, GLUE 벤치마크 기준으로 과제당 약 3.6%의 파라미터를 더하는 것만으로 full fine-tuning과 0.8% 이내의 성능을 보고했다.

**LoRA.** Hu 외(2022)는 사전학습 가중치 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$를 고정하고, 갱신을 저랭크 행렬의 곱으로 제한했다.

$$
\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}, \qquad \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r},\ \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}},\ r \ll \min(d_{\text{in}}, d_{\text{out}}).
$$

$\alpha$는 갱신의 크기를 정하는 상수다. $\mathbf{B}$를 0으로 초기화하므로 학습은 사전학습 모델과 정확히 같은 함수에서 시작한다. 학습할 파라미터는 $r(d_{\text{in}} + d_{\text{out}})$개로, 행렬 전체 $d_{\text{in}}d_{\text{out}}$개보다 훨씬 적다. 저자들은 GPT-3 175B의 full fine-tuning과 비교해 학습 파라미터를 약 1만 분의 1로, GPU 메모리를 약 3분의 1로 줄이면서 비슷하거나 더 나은 성능을 보고했다. 학습이 끝나면 $\mathbf{B}\mathbf{A}$를 $\mathbf{W}_0$에 더해 병합할 수 있으므로 추론 지연이 늘지 않는다.

**비용 계산.** 가상의 Transformer 구성(층 24개, 폭 1024, 어휘 32,000)에서 파라미터 수와, Adam으로 학습할 때 필요한 메모리를 센다. 층마다 attention 투영 4개와 폭 $4d$의 MLP가 있으면 파라미터는 대략 $12Ld^2$이고, 여기에 embedding $Vd$가 더해진다. LoRA는 attention의 Q·V 투영에만 rank 8로 건다.

```python {exec="browser" session="fm9"}
import numpy as np

L, d, V, r = 24, 1024, 32000, 8                    # 가상의 Transformer 구성: 층, 폭, 어휘, LoRA rank
attn = 4 * (d * d + d)                             # Q, K, V, O 투영 (가중치 + bias)
mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)        # 폭 4d의 MLP 두 층
norm = 2 * 2 * d                                   # LayerNorm 2개 (scale, shift)
total = L * (attn + mlp + norm) + V * d            # + token embedding
lora = L * 2 * r * (d + d)                         # Q, V 투영에만 rank r 갱신 B A: (d×r) + (r×d)
print(f"전체 파라미터       {total / 1e6:8.1f} M   (12·L·d² 근사 {12 * L * d * d / 1e6:.1f} M + embedding {V * d / 1e6:.1f} M)")
print(f"LoRA 학습 파라미터  {lora / 1e6:8.3f} M   (전체의 {100 * lora / total:.3f}%)")
GB = 1024**3
full_ft = total * (4 + 4 + 8)                      # fp32 가중치 + gradient + Adam 1·2차 모멘트 (byte)
lora_ft = total * 4 + lora * (4 + 4 + 8)           # 고정 가중치 + LoRA 파라미터의 학습 상태
print(f"학습 상태 메모리(activation 제외)  full {full_ft / GB:.1f} GB,  LoRA {lora_ft / GB:.2f} GB")
```

```output
전체 파라미터          335.1 M   (12·L·d² 근사 302.0 M + embedding 32.8 M)
LoRA 학습 파라미터     0.786 M   (전체의 0.235%)
학습 상태 메모리(activation 제외)  full 5.0 GB,  LoRA 1.26 GB
```

full fine-tuning은 fp32에서 파라미터마다 가중치·gradient·Adam의 두 모멘트로 16 byte가 들어, activation을 빼고도 5.0 GB가 필요하다. LoRA는 고정 가중치(4 byte)와 전체의 0.235%에 불과한 학습 파라미터의 상태만 있으면 되므로 1.26 GB다. 실제 메모리는 activation, 정밀도(bf16, 8-bit 최적화기), 구현에 따라 크게 달라지므로, 자기 환경에서 측정한 값으로 계획한다(22.5장).

**LoRA를 NumPy로.** 고정 가중치 $\mathbf{W}_0$(64 × 64)에서 출발해, 과제가 요구하는 변화가 rank 2의 큰 성분과 작은 full-rank 성분으로 이루어진 경우를 만든다. 출력 잡음의 분산은 0.25다. 갱신을 rank $r$로 제한한 최소제곱(reduced-rank 회귀)이 LoRA의 최적해에 해당한다. 학습 표본 수를 바꿔 시험 MSE를 잰다.

```python {exec="browser" session="fm9"}
rng = np.random.default_rng(0)
d_in, d_out = 64, 64
W0 = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)                # 사전학습된 고정 가중치
dW_true = rng.normal(size=(d_out, 2)) @ rng.normal(size=(2, d_in)) / np.sqrt(d_in)   # 과제의 변화: rank 2
dW_true += 0.02 * rng.normal(size=(d_out, d_in))                   # + 작은 full-rank 성분

def task(n):                                                       # 입력 (n, d_in), 출력 (n, d_out)
    X = rng.normal(size=(n, d_in))
    return X, X @ (W0 + dW_true).T + 0.5 * rng.normal(size=(n, d_out))

def fit_delta(X, Y, rank=None):                                    # W0는 고정, 잔차에 ΔW를 맞춘다
    C = np.linalg.lstsq(X, Y - X @ W0.T, rcond=None)[0]            # 최소제곱 ΔWᵀ (d_in, d_out)
    if rank is None:
        return C.T                                                 # full fine-tuning
    _, _, Vt = np.linalg.svd(X @ C, full_matrices=False)           # reduced-rank 회귀
    return (C @ Vt[:rank].T @ Vt[:rank]).T                         # rank-r ΔW = B A

X_te, Y_te = task(5000)
print("학습 표본   rank 1   rank 2   rank 8   full(4096개)   ΔW=0(사전학습 그대로)")
for n in [100, 2000]:
    X, Y = task(n)
    err = lambda dW: np.mean((Y_te - X_te @ (W0 + dW).T) ** 2)
    print(f"{n:6d}    " + "  ".join(f"{err(fit_delta(X, Y, r)):6.3f}" for r in [1, 2, 8]) +
          f"     {err(fit_delta(X, Y)):6.3f}          {err(0):6.3f}")

U, s, Vt = np.linalg.svd(fit_delta(X, Y, 2))                       # ΔW = B A로 인수분해
B, A = U[:, :2] * s[:2], Vt[:2]                                    # (d_out, 2), (2, d_in): 2·(64+64) = 256개
x = rng.normal(size=d_in)
print("병합 W0 + BA의 출력 = 두 경로 출력:", np.allclose((W0 + B @ A) @ x, W0 @ x + B @ (A @ x)))
```

```output
학습 표본   rank 1   rank 2   rank 8   full(4096개)   ΔW=0(사전학습 그대로)
   100     1.040   0.296   0.399      0.688           2.502
  2000     1.024   0.275   0.270      0.258           2.502
병합 W0 + BA의 출력 = 두 경로 출력: True
```

- **사전학습 그대로(2.502)보다는 어떤 적응이든 낫다.** 이 과제는 사전학습 가중치와 다르다.
- **표본이 100개일 때는 rank 2 LoRA(0.296)가 full fine-tuning(0.688)보다 훨씬 낫다.** 4,096개의 갱신을 100개 표본으로 맞추면 잡음까지 맞춘다. 저랭크 제한은 이 과제에서 올바른 귀납적 편향이고, 정규화처럼 작동한다. rank 1은 필요한 변화를 담지 못하고(1.040), rank 8은 필요 이상의 자유도로 조금 나빠진다(0.399).
- **표본이 2,000개면 full fine-tuning(0.258)이 가장 낫다.** 작은 full-rank 성분까지 맞출 자료가 있기 때문이다. 이때 잡음 바닥은 0.25이므로 모든 방법이 바닥 가까이에 있다.
- **병합은 정확하다.** $\mathbf{W}_0 + \mathbf{B}\mathbf{A}$ 하나로 계산한 출력과, 고정 경로와 저랭크 경로를 따로 계산해 더한 출력이 같다. rank 2 갱신은 $2 \times (64 + 64) = 256$개의 수로 저장된다.

::widget{name="lora-rank"}

::quiz{id="lora-param-count"}

## 도메인 적응과 부정적 전이

**도메인 적응.** 사전학습 자료(원천 도메인)와 하위 과제의 자료(목표 도메인)는 분포가 다르다(21.4장의 분포 이동). 이 차이를 줄이는 대표적인 방법은 두 가지다.

- **도메인 자료로 사전학습을 이어 하기**: 레이블 없는 목표 도메인 자료로 같은 자기지도 목적함수를 조금 더 학습한 뒤 미세조정한다. Gururangan 외(2020)는 언어모델에서 목표 도메인과 과제의 자료로 두 번째 사전학습 단계를 거치면 여러 도메인의 과제에서 성능이 오른다고 보고했다.
- **미세조정 절차의 조정**: Kumar 외(2022)는 사전학습 특징이 좋고 분포 이동이 클 때, 무작위로 초기화한 head와 함께 전체를 미세조정하면 사전학습 특징이 왜곡되어 분포 밖(OOD) 정확도가 linear probe보다 나빠질 수 있음을 보이고, 먼저 head만 학습한 뒤(linear probe) 전체를 미세조정하는 LP-FT를 제안했다.

**부정적 전이.** 사전학습이 하위 과제 성능을 처음부터 학습한 것보다 **나쁘게** 만드는 경우를 부정적 전이(negative transfer)라 한다. 전이는 "원천에서 배운 것이 목표에서도 쓸모 있다"는 가정에 기대므로, 그 가정이 틀리면 사전학습은 잘못된 출발점이 된다. 선형 회귀의 장난감 문제로 확인한다. 원천 과제에서 배운 가중치 $\mathbf{w}_{\text{src}}$가 있고, 목표 과제의 가중치는 그와 닮았거나(작은 차이) 무관하다. 세 방법을 같은 정규화 강도로 비교한다. 처음부터(0을 향해 ridge), 미세조정(원천 가중치를 향해 ridge, 즉 원천에서 출발해 크게 벗어나지 않게), 고정 특징($\mathbf{x}^\top\mathbf{w}_{\text{src}}$ 하나 위의 선형 head)이다.

```python {exec="browser" session="fm9"}
p = 50
w_src = rng.normal(size=p)                                         # 원천 과제에서 배운 가중치 (p,)
targets = {"닮은 과제": w_src + 0.3 * rng.normal(size=p),          # 목표 = 원천 + 작은 차이
           "무관한 과제": rng.normal(size=p)}                       # 목표가 원천과 독립

def fit(X, y, prior, lam=20.0):                                    # min ‖y - Xw‖² + λ‖w - prior‖²
    return prior + np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ (y - X @ prior))

def r2(w, w_t, X_te):
    y_te = X_te @ w_t + rng.normal(size=len(X_te))
    return 1 - np.mean((y_te - X_te @ w) ** 2) / np.var(y_te)

X_te = rng.normal(size=(4000, p))
for name, w_t in targets.items():
    print(f"[{name}]  학습 표본   처음부터   미세조정(원천에서 출발)   고정 특징 x·w_src + 선형 head")
    for n in [10, 40, 200]:
        res = []
        for rep in range(20):                                      # 학습 표본을 바꿔 20번 평균
            X = rng.normal(size=(n, p)); y = X @ w_t + rng.normal(size=n)
            f = X @ w_src; a = f @ y / (f @ f)                     # 1차원 고정 특징 위의 선형 head
            res.append([r2(fit(X, y, np.zeros(p)), w_t, X_te), r2(fit(X, y, w_src), w_t, X_te), r2(a * w_src, w_t, X_te)])
        print(f"{'':12}{n:5d}      " + "        ".join(f"{v:7.3f}" for v in np.mean(res, 0)))
```

```output
[닮은 과제]  학습 표본   처음부터   미세조정(원천에서 출발)   고정 특징 x·w_src + 선형 head
               10        0.189          0.884          0.860
               40        0.580          0.926          0.871
              200        0.957          0.971          0.871
[무관한 과제]  학습 표본   처음부터   미세조정(원천에서 출발)   고정 특징 x·w_src + 선형 head
               10        0.137         -0.685         -0.125
               40        0.587          0.121         -0.013
              200        0.957          0.938         -0.004
```

- **닮은 과제에서는 전이가 강력하다.** 표본 10개로 미세조정은 $R^2 = 0.884$, 처음부터는 0.189다. 고정 특징도 0.860으로 좋지만, 표본이 늘어도 0.871에서 멈춘다. 원천과 목표의 차이를 표현할 수 없기 때문이다.
- **무관한 과제에서는 전이가 해가 된다.** 표본 10개의 미세조정은 $R^2 = -0.685$로, 처음부터(0.137)보다도, 아무것도 예측하지 않는 것(0)보다도 나쁘다. 잘못된 출발점으로 끌어당기는 정규화가 자료의 신호를 이긴 것이다. 표본이 200개로 늘면 자료가 출발점의 영향을 이겨 0.938로 회복하지만, 여전히 처음부터(0.957)보다 낮다. 고정 특징은 어떤 표본 수에서도 0 근처다.

**진단.** 부정적 전이는 **처음부터 학습한 기준선과 같은 예산으로 비교할 때만** 보인다. 전이한 모델의 점수만 보고하면 0.938은 좋은 결과처럼 보인다. 그래서 전이 실험에는 항상 다음을 둔다.

1. 같은 구조·같은 튜닝 예산의 처음부터 학습한 기준선
2. 목표 레이블 수를 바꾼 학습 곡선(이득이 어느 표본 수에서 사라지거나 뒤집히는가)
3. 원천과 목표의 유사성에 대한 독립적인 근거(같은 물리량인가, 같은 측정 방식인가, 표현 공간에서 분포가 겹치는가)

::quiz{id="negative-transfer"}

:::note[분야 예시]
전지구 재분석 자료로 사전학습한 기상 모델을 한 지역의 관측소 자료로 미세조정하는 상황을 생각한다. 지역의 기후가 사전학습 자료에 충분히 담겨 있다면 적은 관측으로도 큰 이득을 얻는다. 그러나 사전학습 자료의 해상도로는 보이지 않는 지형 효과(산악 강수, 해안 국지풍)가 목표 과제의 핵심이라면, 사전학습 모델의 특징은 그 효과를 담지 못하고 오히려 거친 해상도의 패턴으로 예측을 끌어당길 수 있다. 같은 관측 자료로 처음부터 학습한 작은 모델과 비교해야 어느 경우인지 알 수 있다.
:::

## 평가

**task suite.** foundation model의 주장은 여러 하위 과제에 대한 주장이므로 평가도 과제 묶음(task suite)으로 한다. 과제는 사전학습 목적과 가까운 것과 먼 것, 레이블이 많은 것과 적은 것, 원천 분포 안과 밖을 고루 포함해야 한다. 결과는 과제마다 전이 방법·레이블 수·기준선과 함께 표로 보고하고, 과제 사이의 평균 하나로 요약하지 않는다(12.5장의 여러 데이터셋 비교).

**데이터 중복 검증.** 사전학습 자료는 크고 출처가 넓어서, 하위 과제의 시험 표본이나 그와 거의 같은 표본이 들어 있기 쉽다. 그러면 시험 점수는 일반화가 아니라 기억을 잰다(18.5장의 오염, 19.7장의 기억). **정확히 같은** 표본만 찾는 중복 검사는 부족하다. 단백질 서열을 흉내 낸 자료로 확인한다. 사전학습 서열 2,000개 중 100개를 몇 자리만 바꾼 변이체와, 새로 만든 서열 100개로 시험 자료를 만든다. 변이체는 원본의 성질을 물려받는다고 둔다. 겹침은 길이 5의 부분 서열($k$-mer) 집합의 Jaccard 유사도로 찾는다.

```python {exec="browser" session="fm9"}
from scipy import sparse

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))                         # 아미노산 20종
corpus = rng.integers(20, size=(2000, 60))                           # 사전학습 서열 2000개, 길이 60
labels = rng.integers(2, size=2000)                                  # 서열마다의 가상 성질 (0/1)
near = corpus[:100].copy()                                           # 사전학습 서열 100개를 조금 바꾼 변이체
near[np.arange(100)[:, None], rng.integers(60, size=(100, 3))] = rng.integers(20, size=(100, 3))
test = np.vstack([near, rng.integers(20, size=(100, 60))])           # 시험 200개: 변이체 100 + 새 서열 100
y_test = np.r_[labels[:100], rng.integers(2, size=100)]              # 변이체는 원본의 성질을 물려받는다

def kmer_matrix(S, k=5):                                             # 서열 × k-mer 존재 여부 (희소 행렬)
    codes = sum(S[:, i:S.shape[1] - k + 1 + i] * 20 ** i for i in range(k))   # (n, 60-k+1)
    rows = np.repeat(np.arange(len(S)), codes.shape[1])
    M = sparse.csr_matrix((np.ones(codes.size), (rows, codes.ravel())), shape=(len(S), 20**k))
    M.data[:] = 1.0                                                  # 같은 k-mer가 여러 번 나와도 1
    return M

Kc, Kt = kmer_matrix(corpus), kmer_matrix(test)
inter = (Kt @ Kc.T).toarray()                                        # 공유 k-mer 수 (200, 2000)
jac = inter / (np.asarray(Kt.sum(1)) + np.asarray(Kc.sum(1)).T - inter)
corpus_str = {"".join(AA[c]) for c in corpus}
exact = sum("".join(AA[t]) in corpus_str for t in test)                # 문자열이 완전히 같은 서열
flag = jac.max(1) > 0.3
print(f"정확히 같은 서열: {exact}개    k-mer Jaccard > 0.3: {flag.sum()}개 (변이체 {flag[:100].sum()}, 새 서열 {flag[100:].sum()})")
pred = labels[jac.argmax(1)]                                         # "모델": 가장 닮은 사전학습 서열의 성질을 답한다
print(f"정확도  전체 {np.mean(pred == y_test):.2f}   변이체 {np.mean(pred[:100] == y_test[:100]):.2f}   새 서열 {np.mean(pred[100:] == y_test[100:]):.2f}")
```

```output
정확히 같은 서열: 0개    k-mer Jaccard > 0.3: 100개 (변이체 100, 새 서열 0)
정확도  전체 0.77   변이체 1.00   새 서열 0.54
```

- **정확한 일치 검사는 겹침을 하나도 찾지 못한다.** 모든 변이체가 원본과 몇 자리씩 다르기 때문이다.
- **$k$-mer 유사도는 변이체 100개를 모두 찾고, 새 서열은 하나도 잘못 잡지 않는다.**
- **가장 닮은 사전학습 서열의 성질을 그대로 답하는 "모델"이 전체 정확도 0.77을 받는다.** 그러나 이것은 변이체에서의 1.00(기억)과 새 서열에서의 0.54(우연 수준)의 평균이다. 시험 자료를 겹침 여부로 나누어 보고하지 않으면, 기억을 일반화로 착각한다.

:::note[분야 예시]
단백질 구조·기능 예측에서는 시험 서열이 학습 서열과 일정 수준 이상의 서열 동일성을 갖지 않도록 군집 단위로 분할하는 관행이 있다. 사전학습 모델을 평가할 때는 이 기준을 하위 과제의 학습 자료뿐 아니라 **사전학습 자료**에도 적용해야 한다. 사전학습 자료가 공개되지 않은 모델이라면 이 검증을 할 수 없다는 사실 자체를 평가의 한계로 보고한다.
:::

**작은 데이터에서의 전이 효과.** 과학 연구에서 전이가 가장 필요한 곳은 레이블이 적은 곳이다. 그래서 평가의 중심은 큰 벤치마크의 최고 점수가 아니라, 레이블 수를 줄여 가며 그린 **학습 곡선**이다. 19.8장의 표처럼, 처음부터 학습한 기준선과의 차이가 어느 레이블 수에서 얼마인지가 전이의 실질적인 가치다.

## 실습

감쇠 진동 신호의 각진동수를 맞히도록 레이블이 많은 원천 영역에서 MLP를 사전학습하고, 다른 센서로 측정한 목표 영역(기준선 drift, 이득 차이, 더 큰 잡음)으로 전이한다. 목표 레이블 수를 바꿔 가며 처음부터 학습, 고정 특징(head만 학습), LoRA(두 은닉층에 rank 4, head 학습), full 미세조정을 같은 step 수로 비교한다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import copy

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
t_s = torch.linspace(0, 1, 64)                                     # 시간 (T,), 단위 s
def domain(n, drift):                                              # 감쇠 진동 신호와 정규화한 ω. drift: 목표 영역의 센서 특성
    om, ga, ph = 2 * np.pi * (3 + 5 * torch.rand(n)), 0.5 + 3.5 * torch.rand(n), 2 * np.pi * torch.rand(n)
    x = torch.exp(-ga[:, None] * t_s) * torch.cos(om[:, None] * t_s + ph[:, None]) + 0.1 * torch.randn(n, 64)
    if drift:                                                      # 다른 센서: 기준선 drift, 이득 차이, 더 큰 잡음
        x = 0.6 * x + torch.randn(n, 1) * t_s + 0.2 * torch.randn(n, 64)
    return x, ((om - 2 * np.pi * 5.5) / (2 * np.pi * 1.5))[:, None]

class LoRA(nn.Module):                                             # y = W0 x + (α/r)·B A x, W0 고정
    def __init__(self, base, r=4, alpha=4):
        super().__init__(); self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B = 0: 시작점은 사전학습 모델 그대로
        base.requires_grad_(False)
    def forward(self, x):
        return self.base(x) + self.s * (x @ self.A.T) @ self.B.T

def mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

def train(net, x, y, steps, lr=1e-3):
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        idx = torch.randint(len(x), (min(128, len(x)),))
        loss = ((net(x[idx]) - y[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return net

pre = train(mlp(), *domain(20000, drift=False), steps=3000)        # 원천 영역에서 사전학습 (레이블 많음)
x_te, y_te = domain(2000, drift=True)
def r2(net):
    with torch.no_grad():
        return 1 - ((net(x_te) - y_te) ** 2).mean().item() / y_te.var().item()

def variant(kind):
    net = copy.deepcopy(pre)
    if kind == "고정 특징": net[:4].requires_grad_(False)        # 마지막 선형 head만 학습
    if kind == "LoRA": net[0], net[2] = LoRA(net[0]), LoRA(net[2])  # 은닉층은 저랭크 갱신만, head는 학습
    return mlp() if kind == "처음부터" else net

kinds = ["처음부터", "고정 특징", "LoRA", "full"]
n_train = [sum(p.numel() for p in variant(k).parameters() if p.requires_grad) for k in kinds]

print(f"사전학습 모델을 그대로 목표 영역에 쓰면 R² = {r2(pre):.3f}")
print("목표 레이블   처음부터   고정 특징   LoRA(r=4)   full 미세조정")
print("학습 파라미터 " + "  ".join(f"{k:>9,}" for k in n_train))
for n in [32, 128, 1024]:
    x, y = domain(n, drift=True)
    row = [r2(train(variant(k), x, y, steps=600)) for k in kinds]
    print(f"{n:8d}    " + "   ".join(f"{v:8.3f}" for v in row))
```

```output
사전학습 모델을 그대로 목표 영역에 쓰면 R² = 0.640
목표 레이블   처음부터   고정 특징   LoRA(r=4)   full 미세조정
학습 파라미터    82,689        257      3,585     82,689
      32       0.021      0.685      0.781      0.674
     128       0.631      0.795      0.871      0.859
    1024       0.887      0.830      0.943      0.947
```

- **사전학습 모델을 그대로 쓰면 0.640이다.** 분포 이동으로 성능이 떨어졌지만, 쓸모없는 수준은 아니다. 원천과 목표가 닮은 경우다.
- **레이블 32개에서 LoRA(0.781)가 가장 낫다.** full 미세조정(0.674)은 8만 개의 파라미터를 32개 표본으로 움직이며 사전학습 특징을 흔들고, 고정 특징(0.685)은 센서 차이를 흡수할 수 없다. 처음부터는 거의 아무것도 배우지 못한다(0.021).
- **레이블 1,024개에서는 LoRA(0.943)와 full(0.947)이 비슷하고, 처음부터(0.887)는 여전히 뒤처진다.** 고정 특징은 0.830에서 멈춘다. LoRA는 학습 파라미터가 full의 4% 남짓(3,585개 대 82,689개)이면서 거의 같은 성능을 낸다.

이 결과는 한 시드, 한 쌍의 도메인에서 얻은 것이다. 자기 문제에서는 다음을 수행한다.

1. 공개된 사전학습 모델 하나를 고른다. 사전학습 자료의 출처와 범위가 문서로 공개된 모델을 고르고, 그 문서를 보고서에 인용한다. 모델 이름·버전·내려받은 날짜를 기록한다(집필 시점의 모델 목록은 빠르게 바뀌므로 이 책은 특정 모델을 권하지 않는다).
2. 자기 자료에서 목표 레이블 수를 3단계 이상으로 바꿔, 고정 특징·LoRA·full 미세조정·처음부터 학습을 같은 탐색 예산으로 비교한다. 시드 3개 이상으로 반복해 변동을 붙인다.
3. 사전학습 자료와 자기 시험 자료의 겹침을 점검한다. 자료 형식에 맞는 근사 중복 기준(서열 동일성, 영상의 지각적 해시, 분자의 지문 유사도 등)을 쓴다. 점검할 수 없으면 그 사실을 적는다.
4. 처음부터 학습한 기준선보다 나쁜 설정이 있는지 찾고, 있다면 원천과 목표의 차이로 설명을 시도한다.

마지막으로 LoRA의 순전파·병합·파라미터 수를 구현한다. 테스트가 바로 채점한다.

::exercise{id="lora-merge"}

## 실패 사례

- **큰 모델이라는 이유만으로 foundation model이라 부르는 것.** 크기는 전이 범위에 대해 아무것도 말하지 않는다. 사전학습 자료의 범위, 적응한 하위 과제의 목록, 각 과제의 기준선 비교, 실패한 과제를 함께 보여야 한다.
- **처음부터 학습한 기준선 없이 전이 결과를 보고하는 것.** 부정적 전이는 기준선과 비교해야만 보인다. 이 장의 무관한 과제에서 미세조정의 $R^2$ 0.938은 그 자체로는 좋아 보였지만 처음부터(0.957)보다 낮았다.
- **정확한 일치만으로 데이터 중복을 점검하는 것.** 몇 자리만 다른 변이체, 해상도만 바꾼 영상, 단위만 바꾼 표는 정확한 일치 검사를 통과한다. 근사 중복 기준으로 점검하고, 시험 결과를 겹침 여부로 나누어 보고한다.
- **전이 방법마다 다른 예산을 쓰는 것.** full 미세조정에 학습률 탐색을 여러 번 하고 LoRA에는 기본값만 쓰면, 차이는 방법이 아니라 튜닝에서 온다(12.5장).
- **자료가 적은데 full 미세조정부터 하는 것.** 레이블이 적을 때 full 미세조정은 사전학습 특징을 흔들어 고정 특징이나 LoRA보다 나쁠 수 있다. 싼 방법(고정 특징)에서 시작해 이득이 있을 때만 학습하는 파라미터를 늘린다.

## 통과 기준

- 어떤 모델의 foundation model 주장을 사전학습 자료의 범위, 적응한 하위 과제, 기준선 비교, 데이터 중복 점검의 네 항목으로 평가한 표를 작성한다.
- 자기 자료에서 목표 레이블 수 3단계 이상에 대해 고정 특징·LoRA·full 미세조정·처음부터 학습을 같은 예산으로 비교하고, 부정적 전이가 있는지 판정한다.
- 연습문제 `lora-merge`의 테스트를 통과한다.

## 참고 문헌

- Bommasani 외 (2021). On the opportunities and risks of foundation models. arXiv:2108.07258. — foundation model이라는 용어의 정의, 동질화의 위험
- Houlsby 외 (2019). Parameter-efficient transfer learning for NLP. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2790–2799. — adapter 모듈
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations* (ICLR 2022). arXiv:2106.09685. — 저랭크 갱신으로 학습 파라미터를 줄이는 미세조정
- Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith (2020). Don't stop pretraining: Adapt language models to domains and tasks. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 8342–8360. — 도메인·과제 자료로 이어 하는 사전학습
- Kumar, Raghunathan, Jones, Ma, Liang (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. *ICLR 2022*. arXiv:2202.10054. — 미세조정의 특징 왜곡과 LP-FT
