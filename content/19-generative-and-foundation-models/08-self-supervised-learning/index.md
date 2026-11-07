---
id: generative-and-foundation-models/self-supervised-learning
title: "자기지도학습"
title_en: "Self-Supervised Learning"
tier: core
status: draft
prerequisites: [architectures/inductive-bias, information-and-learning-theory/mutual-information]
---

# 자기지도학습

과학 데이터의 대부분에는 레이블이 없다. 망원경은 밤마다 영상을 쌓고, 서열 데이터베이스는 계속 커지고, 시뮬레이션은 끝없이 출력을 낸다. 반면 사람이 확인한 레이블(세포 유형, 결함 여부, 측정한 물성)은 비싸고 적다. **자기지도학습**(self-supervised learning, SSL)은 레이블 없는 데이터에서 데이터 자신이 만든 과제를 풀며 표현을 학습하고, 그 표현을 적은 레이블로 하위 과제에 옮겨 쓰는 방법이다.

이 장은 SSL의 방법 계열을 하나의 질문으로 묶는다. "같은 대상의 서로 다른 모습은 같은 표현으로 보내되, 모든 것을 하나의 표현으로 보내는 손쉬운 해답(붕괴)은 어떻게 막는가." 그리고 SSL에서 가장 흔한 두 함정, 즉 손실은 낮지만 쓸모없는 표현(붕괴와 shortcut)과 공정하지 않은 비교를 계산으로 확인한다. 14.2장에서 본 InfoNCE의 정보 이론적 해석, 18.1장의 귀납적 편향, 18.8장의 probe가 이 장의 바탕이다.

## 학습 목표

- 주요 자기지도학습 계열(masked reconstruction·contrastive·non-contrastive·predictive)의 목적함수와 붕괴를 막는 장치를 설명한다
- pretext 과제와 증강을 설계하고, 증강이 표현에 넣는 불변성과 shortcut의 위험을 판단한다
- 사전학습 효과를 무작위 초기화·같은 데이터·같은 downstream 예산과 비교해 공정하게 평가한다

## 방법 계열

**왜 필요한가.** 레이블 없이 표현을 학습하려면 레이블 대신 맞힐 무언가가 필요하다. 이렇게 데이터에서 자동으로 만든 과제를 **pretext 과제**라 한다. 계열은 pretext 과제가 무엇을 맞히게 하는지로 나뉜다.

| 계열 | pretext 과제 | 붕괴를 막는 장치 | 대표 예 |
|---|---|---|---|
| masked reconstruction | 입력의 가린 부분을 **입력 공간에서** 복원 | 입력을 복원해야 하므로 붕괴할 수 없다 | 가린 단어 예측, MAE |
| contrastive | 같은 대상의 두 view를 다른 대상들 사이에서 골라내기 | 다른 대상(negative)과 멀어지게 하는 항 | SimCLR(InfoNCE) |
| non-contrastive | 두 view의 표현을 서로 맞추기(negative 없음) | stop-gradient, 이동평균 target, 분산·공분산 정규화 | BYOL, SimSiam, VICReg |
| predictive (JEPA류) | 보이는 부분의 표현으로 가린 부분의 **표현**을 예측 | 이동평균 target 등 non-contrastive와 비슷한 장치 | I-JEPA |

**masked reconstruction.** 입력의 일부를 가리고 나머지로 복원하게 한다. 언어모델의 가린 토큰 예측이 대표적이고, 이미지에서는 MAE(masked autoencoder)가 패치의 75%처럼 큰 비율을 무작위로 가리고 보이는 패치만 encoder에 넣어 가린 픽셀을 복원한다(He 외, 2022). 가장 단순한 경우를 생각하면 의미가 분명해진다. 선형 autoencoder로 입력을 복원하는 문제의 최적 encoder는 PCA의 주성분 공간이다. 재구성 계열은 "입력의 분산을 잘 설명하는 방향"을 배우는 쪽으로 치우치고, 그 방향이 하위 과제에 중요한지는 보장하지 않는다(18.8장의 PCA 예시가 같은 현상이다).

**contrastive.** 한 표본에 무작위 증강을 두 번 적용해 view $\mathbf{x}_i, \mathbf{x}_i'$를 만들고, encoder 출력 $\mathbf{z} = g(f(\mathbf{x}))$를 단위 구 위로 정규화한다. batch의 $N$개 표본에서 나온 $2N$개 view 중 $\mathbf{z}_i$의 짝 $\mathbf{z}_i'$를 나머지 $2N - 1$개 사이에서 골라내는 분류 손실이 SimCLR의 NT-Xent다(Chen 외, 2020).

$$
\ell_i = -\log\frac{\exp(\mathbf{z}_i^\top\mathbf{z}_i'/\tau)}{\sum_{k \ne i}\exp(\mathbf{z}_i^\top\mathbf{z}_k/\tau)}.
$$

$\tau$는 temperature, 합은 자기 자신을 뺀 $2N - 1$개 view에 대한 것이다. 14.2장의 InfoNCE를 대칭으로 쓴 것이다. Wang과 Isola(2020)는 negative가 많을 때 이 손실이 두 항으로 나뉜다는 것을 보였다. 짝끼리 가깝게 하는 **정렬**(alignment) $\mathbb{E}\lVert \mathbf{z} - \mathbf{z}' \rVert^2$와, 표현을 구 위에 고르게 퍼뜨리는 **균일성**(uniformity) $\log\mathbb{E}\,e^{-2\lVert \mathbf{z}_i - \mathbf{z}_j \rVert^2}$이다. 둘 다 작을수록 좋다. 세 가지 표현에서 이 양들을 계산한다.

```python {exec="browser" session="ssl"}
import numpy as np

rng = np.random.default_rng(0)
normalize = lambda z: z / np.linalg.norm(z, axis=1, keepdims=True)

def nt_xent(z1, z2, tau):                          # 두 view의 embedding (N, D), 2N-1개 중 짝 고르기
    z = normalize(np.vstack([z1, z2]))             # (2N, D), 단위 구 위로
    S = z @ z.T / tau                              # 코사인 유사도 / τ, (2N, 2N)
    np.fill_diagonal(S, -np.inf)                   # 자기 자신은 후보에서 뺀다
    N = len(z1)
    pos = np.r_[np.arange(N, 2 * N), np.arange(N)] # i의 짝: 다른 view의 같은 표본
    lse = np.log(np.exp(S - S.max(1, keepdims=True)).sum(1)) + S.max(1)
    return np.mean(lse - S[np.arange(2 * N), pos])

def align_uniform(z1, z2):                         # Wang·Isola의 두 성질 (작을수록 좋다)
    z1, z2 = normalize(z1), normalize(z2)
    z = np.vstack([z1, z2]); d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    iu = np.triu_indices(len(z), 1)
    return np.mean(((z1 - z2) ** 2).sum(1)), np.round(np.log(np.mean(np.exp(-2 * d2[iu]))), 3) + 0.0

N, D = 256, 16
base = rng.normal(size=(N, D))                     # 표본마다 고유한 방향
cases = {"좋은 표현": (base + 0.2 * rng.normal(size=(N, D)), base + 0.2 * rng.normal(size=(N, D))),
         "짝을 모름": (rng.normal(size=(N, D)), rng.normal(size=(N, D))),
         "완전 붕괴": (np.ones((N, D)) + 1e-3 * rng.normal(size=(N, D)), np.ones((N, D)) + 1e-3 * rng.normal(size=(N, D)))}
print(f"log(2N-1) = {np.log(2 * N - 1):.3f}")
print("표현          NT-Xent(τ=0.1)  NT-Xent(τ=0.5)   정렬     균일성")
for name, (z1, z2) in cases.items():
    a, u = align_uniform(z1, z2)
    print(f"{name:10}     {nt_xent(z1, z2, 0.1):7.3f}         {nt_xent(z1, z2, 0.5):7.3f}      {a:6.3f}   {u:6.3f}")
```

```output
log(2N-1) = 6.236
표현          NT-Xent(τ=0.1)  NT-Xent(τ=0.5)   정렬     균일성
좋은 표현            0.441           4.456       0.084   -3.458
짝을 모름            8.738           6.319       1.959   -3.513
완전 붕괴            6.236           6.236       0.000    0.000
```

- **좋은 표현은 정렬과 균일성이 모두 좋고 손실이 작다.** $\tau = 0.1$에서 0.441이다.
- **완전히 붕괴한 표현은 정렬이 완벽하지만(0.000) 균일성이 최악(0.000)이다.** 모든 유사도가 같으므로 손실은 정확히 $\log(2N - 1) = 6.236$이 되고, temperature와 무관하다. contrastive 손실은 균일성 항 덕분에 붕괴를 벌한다.
- **짝을 모르는 표현은 $\tau = 0.1$에서 붕괴보다도 나쁘다(8.738).** 작은 temperature는 유사도 차이를 증폭하므로, 우연히 가까운 negative에 확신을 갖고 틀리는 표현을 크게 벌한다. temperature는 손실이 어려운 negative에 얼마나 집중하는지를 정하는 하이퍼파라미터다.

::quiz{id="collapse-loss-value"}

**non-contrastive.** negative를 쓰지 않고 두 view의 표현을 맞추기만 하면, 모든 입력을 같은 벡터로 보내는 상수 함수가 손실 0을 달성한다. 그런데도 이 계열이 동작하는 것은 붕괴를 막는 장치가 있기 때문이다. BYOL은 online 신경망이 target 신경망의 표현을 예측하게 하고, target은 online의 느린 이동평균으로만 갱신한다(Grill 외, 2020). SimSiam은 이동평균 없이도 한쪽 가지의 gradient를 끊는 **stop-gradient**가 붕괴를 막는 핵심이라는 것을 실험으로 보였다(Chen, He, 2021). VICReg은 구조 장치 대신 손실에 항을 더한다. 차원별 표준편차가 임계값 아래로 내려가면 벌하는 분산 항과, 차원 사이의 공분산을 0으로 미는 공분산 항이다(Bardes 외, 2022).

**predictive(JEPA류).** 재구성 계열처럼 가린 부분을 예측하되, 픽셀이 아니라 **표현**을 예측한다. I-JEPA는 한 이미지 안에서 보이는 문맥 블록의 표현으로 가린 목표 블록들의 표현을 예측하며, 손으로 설계한 증강에 의존하지 않는다(Assran 외, 2023). 입력의 세세한 잡음까지 복원할 필요가 없다는 것이 재구성 계열과의 차이이고, 표현 공간의 목표가 붕괴할 수 있다는 점은 non-contrastive 계열과 같다.

## 설계 요소

**증강은 불변성의 선언이다.** contrastive와 non-contrastive 계열에서 두 view를 같은 표현으로 보내라는 것은, 증강이 바꾸는 성질을 **표현이 무시하라**는 뜻이다. 회전을 증강으로 쓰면 회전 불변 표현을, 색 변화를 쓰면 색에 둔감한 표현을 얻는다. 18.1장에서 대칭성을 구조에 넣는 것을 귀납적 편향이라 했는데, 증강은 같은 가정을 데이터 쪽에서 넣는 방법이다. SimCLR의 실험에서도 증강의 **조합**이 표현의 품질을 크게 좌우했다(Chen 외, 2020).

따라서 증강을 고르는 질문은 "하위 과제가 이 성질에 의존하는가"다. 의존한다면 그 증강은 표현에서 필요한 정보를 지운다.

:::note[분야 예시]
세포 현미경 영상에서 회전과 뒤집기는 대개 안전한 증강이다. 세포의 방향은 생물학적 의미가 없는 경우가 많기 때문이다. 반면 기상 장(field) 자료에서 회전은 안전하지 않다. 북반구와 남반구, 위도에 따라 달라지는 Coriolis 효과 때문에 회전한 기상 장은 물리적으로 다른 상황이다. 분광 자료에서 봉우리 위치를 옮기는 증강은 화학 정보를 지운다. 이미지 분야의 증강 목록을 과학 데이터에 그대로 가져오지 않는다.
:::

**positive와 negative.** positive 쌍은 "같다고 간주할 것", negative는 "구별할 것"을 정한다. 같은 batch의 다른 표본을 negative로 쓰면, 같은 클래스의 표본도 negative가 되어 서로 밀어내는 **false negative**가 생긴다. batch가 클수록 negative가 많아져 contrastive 학습이 유리하다는 것도 SimCLR의 관찰이다. 과학 데이터에서는 positive를 자연스럽게 정의할 수 있는 경우가 많다. 같은 세포의 두 측정 modality, 같은 물리계의 다른 시점, 같은 분자의 다른 conformation이 그 예다(14.2장의 분야 예시).

**마스킹 전략.** 재구성·예측 계열에서는 무엇을 얼마나 가리는지가 pretext 과제의 난이도를 정한다. 가린 부분이 이웃의 단순한 보간으로 복원된다면, 모델은 국소적인 보간만 배운다. MAE가 높은 마스킹 비율을 쓰고, I-JEPA가 충분히 큰 목표 블록과 공간적으로 퍼진 문맥을 강조하는 이유다. 시계열에서는 개별 시점보다 연속 구간을, 공간장에서는 점보다 덩어리를 가리는 편이 의미 있는 과제가 된다.

**projection head.** SimCLR은 encoder 출력 $f(\mathbf{x})$ 위에 작은 비선형 신경망 $g$를 두고 손실은 $g(f(\mathbf{x}))$에 걸되, 하위 과제에는 $f(\mathbf{x})$를 쓰는 것이 표현의 품질을 크게 높인다고 보고했다. 손실이 요구하는 불변성 때문에 $g$의 출력에서는 정보가 지워지지만, $f$의 출력에는 남아 있을 수 있다. 그래서 평가할 때 **어느 층의 표현을 쓰는지** 밝혀야 한다.

## 실패 양상

**표현 붕괴(representation collapse).** 두 종류가 있다.

- **완전 붕괴**: 모든 입력이 같은 벡터(또는 거의 같은 벡터)로 간다. non-contrastive 손실에서 장치가 빠지면 생긴다.
- **차원 붕괴**(dimensional collapse): 표현이 전체 공간이 아니라 저차원 부분공간에만 퍼진다. Jing 외(2022)는 negative가 있는 contrastive 학습에서도 차원 붕괴가 일어나며, 특징 방향을 따라 강한 증강과 저랭크 해를 향하는 암묵적 정규화가 원인이라고 분석했다.

붕괴를 진단하는 지표를 세 표현에 계산한다. **유효 rank**는 중심화한 표현 행렬의 특이값을 확률처럼 정규화해 엔트로피의 지수를 취한 것으로, 표현이 실제로 퍼진 차원 수를 잰다. **분산 벌점**은 VICReg의 분산 항, 즉 차원별 표준편차가 1보다 작은 만큼의 평균이다.

```python {exec="browser" session="ssl"}
def eff_rank(z):                                   # 특이값 분포의 엔트로피로 잰 유효 차원
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = s / s.sum()
    return np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))

def vicreg_var(z, gamma=1.0):                      # 차원별 표준편차가 gamma 아래로 내려간 만큼의 벌점
    return np.mean(np.maximum(0, gamma - np.sqrt(z.var(0) + 1e-4)))

low = rng.normal(size=(N, 3)) @ rng.normal(size=(3, D))   # 3차원 부분공간에만 퍼진 표현
noise = lambda s: s * rng.normal(size=(N, D))
emb = {"건강한 표현": (base + noise(0.2), base + noise(0.2)),
       "차원 붕괴(3차원)": (low + noise(0.05), low + noise(0.05)),
       "완전 붕괴": (np.ones((N, D)) + noise(1e-3), np.ones((N, D)) + noise(1e-3))}
print("표현                정렬 MSE   유효 rank   분산 벌점   NT-Xent(τ=0.1)")
for name, (z1, z2) in emb.items():
    print(f"{name:14}     {np.mean((z1 - z2) ** 2):7.4f}    {eff_rank(z1):6.2f}     {vicreg_var(z1):6.3f}      {nt_xent(z1, z2, 0.1):6.3f}")
```

```output
표현                정렬 MSE   유효 rank   분산 벌점   NT-Xent(τ=0.1)
건강한 표현              0.0811     15.88      0.014       0.432
차원 붕괴(3차원)          0.0051      3.70      0.095       3.374
완전 붕괴               0.0000     15.89      0.990       6.236
```

- **정렬 MSE(non-contrastive 계열의 기본 손실)는 붕괴할수록 좋아진다.** 건강한 표현 0.0811, 차원 붕괴 0.0051, 완전 붕괴 0.0000이다. 붕괴를 막는 장치 없이 이 손실만 줄이면 학습은 붕괴로 간다.
- **유효 rank는 차원 붕괴를 잡지만(3.70) 완전 붕괴는 놓친다(15.89).** 유효 rank는 크기를 보지 않으므로, 완전 붕괴한 표현에 남은 $10^{-3}$ 크기의 잡음을 "16차원에 고르게 퍼진 표현"으로 읽는다. 크기를 보는 분산 벌점(0.990)이 완전 붕괴를 잡는다. 두 지표를 함께 본다.
- **contrastive 손실은 두 종류의 붕괴를 모두 벌한다.** 3.374, 6.236으로 건강한 표현(0.432)보다 크다. 다만 차원 붕괴를 완전히 막지는 못한다는 것이 앞의 Jing 외(2022)의 관찰이다.

::widget{name="contrastive-collapse"}

::quiz{id="effective-rank-blind"}

**shortcut learning.** pretext 과제를 의도하지 않은 쉬운 단서로 풀 수 있으면, 모델은 그 단서를 배운다. 두 view가 공유하는 단서가 의미 있는 내용이 아니라 장비 ID, 촬영 배치의 흔적, 파일 압축 패턴, 영상 가장자리의 표지처럼 표본마다 고유한 **표지**(tag)이면, 표지만 읽어도 짝을 맞힐 수 있다. 의미 있는 특징 4개(레이블에 따라 이동)와 표본마다 고유한 표지 4개로 이루어진 자료에서, 표지만 읽는 encoder와 의미만 읽는 encoder를 비교한다. 증강은 의미 특징에 잡음을 더하고, 표지는 그대로 두거나(약한 증강) 매번 다시 뽑는다(표지를 겨냥한 증강).

```python {exec="browser" session="ssl"}
from sklearn.linear_model import LogisticRegression

n = 1000
y = rng.integers(2, size=n)                        # 하위 과제의 레이블 (n,)
sem = (2 * y[:, None] - 1) * 0.8 + rng.normal(size=(n, 4))   # 의미 있는 특징: 레이블에 따라 이동 (n, 4)
tag = rng.normal(size=(n, 4))                      # 표본마다 고유한 표지(장비 ID, 배치 흔적 등) (n, 4)

def two_views(resample_tag):                       # 증강: 의미 특징에 잡음, 표지는 그대로 두거나 다시 뽑는다
    v = [np.hstack([sem + 0.7 * rng.normal(size=sem.shape),
                    rng.normal(size=tag.shape) if resample_tag else tag]) for _ in range(2)]
    return v                                        # 두 view, 각각 (n, 8)

read_tag, read_sem = (lambda v: v[:, 4:]), (lambda v: v[:, :4])   # 표지만 읽는 encoder, 의미만 읽는 encoder
probe = lambda Z: LogisticRegression().fit(Z[:500], y[:500]).score(Z[500:], y[500:])
print("증강              encoder       NT-Xent(τ=0.1)   선형 probe 정확도")
for aug, resample in [("표지 유지", False), ("표지 무작위화", True)]:
    v1, v2 = two_views(resample)
    for enc_name, enc in [("표지만 읽음", read_tag), ("의미만 읽음", read_sem)]:
        print(f"{aug:12}   {enc_name:10}     {nt_xent(enc(v1)[:256], enc(v2)[:256], 0.1):6.3f}           {probe(enc(v1)):.3f}")
```

```output
증강              encoder       NT-Xent(τ=0.1)   선형 probe 정확도
표지 유지          표지만 읽음          2.586           0.524
표지 유지          의미만 읽음          5.936           0.900
표지 무작위화        표지만 읽음         12.162           0.468
표지 무작위화        의미만 읽음          5.522           0.890
```

**표지를 그대로 두면 표지만 읽는 encoder의 손실이 더 낮다(2.586 대 5.936).** 그런데 그 표현으로 하위 과제를 풀면 정확도가 우연 수준(0.524)이다. 손실만 보면 shortcut을 배운 encoder가 더 좋아 보인다. 증강이 표지를 무작위화하면 표지로는 짝을 맞힐 수 없어 손실이 12.162로 치솟고, 의미를 읽는 encoder가 가장 낮은 손실(5.522)을 갖게 된다. **pretext 손실은 표현의 품질을 재지 않는다.** 증강은 모델이 써서는 안 되는 단서를 지우도록 설계해야 하고, 표현의 품질은 하위 과제로 확인해야 한다.

**pretext–downstream 간극.** pretext 과제가 요구하는 것과 하위 과제가 요구하는 것이 다르면, pretext 성능이 좋아져도 하위 성능은 좋아지지 않는다. 증강이 하위 과제에 필요한 성질을 지우는 경우(위의 기상 장 회전), 재구성 과제가 하위 과제와 무관한 고분산 세부를 복원하는 데 용량을 쓰는 경우가 그 예다. pretext 손실과 하위 성능을 학습 중에 함께 기록해, 둘이 함께 움직이는지 확인한다.

## 평가

**평가 프로토콜.** 사전학습한 표현을 평가하는 방식은 하위 과제에 얼마나 많은 것을 새로 학습하는지에 따라 나뉜다.

| 프로토콜 | 새로 학습하는 것 | 재는 것 |
|---|---|---|
| linear probe | 고정된 표현 위의 선형 모델 | 정보가 선형으로 읽히는가(18.8장) |
| $k$-NN | 없음(학습 표본과의 거리) | 표현 공간의 이웃 구조 |
| fine-tuning | encoder 전체와 head | 좋은 초기값으로서의 가치 |
| few-shot | 아주 적은 레이블로 위의 방식 중 하나 | 레이블 효율 |

linear probe는 표현 자체를 가장 직접 재지만, 18.8장에서 본 것처럼 무작위 초기화 망도 높은 점수를 받을 수 있다. fine-tuning은 실제 사용에 가깝지만 결과가 학습률·epoch 같은 하이퍼파라미터와 튜닝 예산에 민감하다. 어느 쪽이든 **대조군과 같은 조건**으로 비교해야 의미가 있다.

**공정한 비교의 세 조건.**

1. **무작위 초기화 기준선**: 같은 구조를 사전학습 없이 같은 방식으로 평가한 점수. 구조만으로 얻는 몫이다.
2. **같은 데이터**: 사전학습 자료가 하위 과제의 시험 자료를 포함하지 않아야 한다. 레이블 없이 쓴 것이라도 시험 표본을 사전학습에서 봤다면 그 표본의 표현은 유리하다. 같은 환자·같은 분자·같은 관측 영역 단위로 분할을 맞추고(15.4장), 거의 같은 중복 표본도 찾아 뺀다.
3. **같은 downstream 예산**: probe나 fine-tuning의 하이퍼파라미터 탐색 횟수와 학습 step을 모든 방법에 같게 준다(12.5장). 사전학습 모델에만 더 많은 튜닝을 하면 그 차이가 사전학습의 효과로 보인다.

**레이블 수에 따른 효과.** 사전학습의 효과는 하위 과제의 레이블 수에 따라 달라진다. 잠재 요인 5개가 100차원으로 관측되고 레이블이 그중 두 요인에 의존하는 자료에서, 레이블 없는 자료 5,000개로 PCA(선형 autoencoder)를 사전학습하고, 입력 그대로·같은 크기의 무작위 투영과 비교한다. 세 표현 모두 같은 logistic regression probe와 같은 정규화 강도를 쓰고, 레이블 표본을 바꿔 10번 반복한 평균 정확도를 보고한다.

```python {exec="browser" session="ssl"}
d, k = 100, 5
A = rng.normal(size=(k, d))                        # 잠재 요인 5개 → 관측 100차원
def make(n):                                       # 관측 x (n, d)와 레이블 y (n,)
    h = rng.normal(size=(n, k))
    return h @ A + 3.0 * rng.normal(size=(n, d)), (h[:, 0] + 0.5 * h[:, 1] > 0).astype(int)

X_unl, _ = make(5000)                              # 레이블 없는 사전학습 자료
_, _, Vt = np.linalg.svd(X_unl - X_unl.mean(0), full_matrices=False)
encoders = {"입력 그대로(100)": lambda X: X,
            "무작위 투영(5)": lambda X, R=rng.normal(size=(d, k)): X @ R,
            "사전학습 PCA(5)": lambda X: (X - X_unl.mean(0)) @ Vt[:k].T}   # 선형 autoencoder = PCA
X_te, y_te = make(2000)
print("레이블 수   " + "   ".join(f"{name:>14}" for name in encoders))
for n_lab in [10, 30, 100, 1000]:
    accs = {name: [] for name in encoders}
    for rep in range(10):                          # 레이블 표본을 바꿔 10번 반복
        X_tr, y_tr = make(n_lab)
        for name, enc in encoders.items():         # 모두 같은 probe, 같은 정규화 강도
            accs[name].append(LogisticRegression(C=1.0, max_iter=1000).fit(enc(X_tr), y_tr).score(enc(X_te), y_te))
    print(f"{n_lab:6d}     " + "   ".join(f"{np.mean(a):14.3f}" for a in accs.values()))
```

```output
레이블 수      입력 그대로(100)        무작위 투영(5)      사전학습 PCA(5)
    10              0.662            0.547            0.730
    30              0.791            0.616            0.848
   100              0.830            0.663            0.883
  1000              0.873            0.688            0.895
```

- **레이블이 10개일 때 사전학습의 이득이 가장 크다(0.662 → 0.730).** 레이블이 1,000개면 그 차이는 0.022로 줄어든다. 레이블이 많으면 하위 모델이 입력에서 직접 필요한 방향을 찾을 수 있기 때문이다. "사전학습이 성능을 높였다"는 주장은 **어떤 레이블 수에서**인지와 함께 말해야 한다.
- **같은 크기(5차원)의 무작위 투영은 크게 뒤처진다.** 5차원으로 줄인 것 자체가 아니라 **어떤 5차원인가**가 중요하다는 뜻이다. 이 비교가 없으면 "차원 축소가 도움이 되었다"와 "사전학습이 도움이 되었다"를 구별할 수 없다.

::quiz{id="label-budget"}

**연구에서의 의미.** 자기 연구에서 SSL을 쓸지 정할 때 가장 유용한 실험은 위 표를 자기 자료로 만드는 것이다. 레이블 수를 몇 단계로 바꿔 가며 사전학습 모델, 무작위 초기화, 입력에 직접 학습한 baseline을 같은 예산으로 비교한다. 레이블이 충분히 많은 영역에서 이득이 사라진다면, 사전학습의 비용(계산, 구현의 복잡성)을 정당화하기 어렵다.

## 실습

감쇠 진동 신호(18.8장과 같은 형태)에서 contrastive 학습으로 encoder를 사전학습하고, 각진동수 $\omega$와 감쇠율 $\gamma$를 linear probe로 읽는다. 증강은 물리적으로 의미 없는 성질을 바꾼다. 같은 $(\omega, \gamma)$로 위상, 진폭(0.5–1.5배), 잡음만 다시 뽑아 두 view를 만든다. 대조군은 같은 구조의 무작위 초기화 encoder와, negative 없이 정렬 손실만 쓴 encoder(붕괴 방지 장치 없음)다. probe는 projection head 앞의 backbone 출력(128차원)에 건다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)
rng = np.random.default_rng(0)
t_s = torch.linspace(0, 1, 64)                                     # 시간 (T,), 단위 s
def signals(omega, gamma):                                         # 같은 (ω, γ)의 새 view: 위상·진폭·잡음은 무작위
    B = len(omega)
    phi, amp = torch.rand(B, 1) * 2 * np.pi, 0.5 + torch.rand(B, 1)
    return amp * torch.exp(-gamma[:, None] * t_s) * torch.cos(omega[:, None] * t_s + phi) + 0.1 * torch.randn(B, 64)
draw = lambda B: (2 * np.pi * (3 + 5 * torch.rand(B)), 0.5 + 3.5 * torch.rand(B))   # ω (rad/s), γ (1/s)

def nt_xent(z1, z2, tau=0.2):                                      # (B, D) 두 개 → 스칼라
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    S = z @ z.T / tau - 1e9 * torch.eye(len(z))                    # 자기 자신 제외
    B = len(z1)
    return F.cross_entropy(S, torch.cat([torch.arange(B, 2 * B), torch.arange(B)]))

def encoder():                                                     # backbone(평가 대상)과 projection head
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()), nn.Linear(128, 64)

def train(loss_name, steps=1500):
    f, g = encoder(); opt = torch.optim.Adam([*f.parameters(), *g.parameters()], lr=1e-3)
    for _ in range(steps):
        om, ga = draw(256)
        z1, z2 = g(f(signals(om, ga))), g(f(signals(om, ga)))      # 같은 계열의 두 view (256, 64)
        loss = nt_xent(z1, z2) if loss_name == "contrastive" else ((z1 - z2) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return f

def eff_rank(z):                                                   # cell 2와 같은 유효 rank
    s = torch.linalg.svdvals(z - z.mean(0)); p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()

def probe_r2(Z, y, n_tr):                                          # ridge probe, 앞 n_tr개로 학습, 마지막 1000개로 시험
    Z = (Z - Z[:n_tr].mean(0)) / (Z[:n_tr].std(0) + 1e-6)
    w = torch.linalg.solve(Z[:n_tr].T @ Z[:n_tr] + 1e-2 * n_tr * torch.eye(Z.shape[1]), Z[:n_tr].T @ (y[:n_tr] - y[:n_tr].mean()))
    pred = Z[-1000:] @ w + y[:n_tr].mean()
    return (1 - ((y[-1000:] - pred) ** 2).mean() / y[-1000:].var()).item()

om, ga = draw(3000); X = signals(om, ga)                           # 평가용 자료 (3000, 64)
models = {"입력 그대로": None, "무작위 초기화": encoder()[0], "대조 학습": train("contrastive"), "정렬만(비대조)": train("align")}
print("표현              평균 std   유효 rank   ω probe R² (레이블 100)   (레이블 1000)   γ probe R² (1000)")
with torch.no_grad():
    for name, enc in models.items():
        Z = X if enc is None else enc(X)
        print(f"{name:12}     {Z.std(0).mean().item():7.3f}   {eff_rank(Z):7.2f}        {probe_r2(Z, om, 100):7.3f}              {probe_r2(Z, om, 1000):7.3f}        {probe_r2(Z, ga, 1000):7.3f}")
```

```output
표현              평균 std   유효 rank   ω probe R² (레이블 100)   (레이블 1000)   γ probe R² (1000)
입력 그대로             0.367     36.56         -1.382               -0.082         -0.049
무작위 초기화            0.043     79.61         -0.937                0.756          0.540
대조 학습              0.083     67.78          0.723                0.919          0.855
정렬만(비대조)           0.000      1.88          0.001                0.001         -0.004
```

- **대조 학습은 레이블이 적을 때 큰 차이를 만든다.** 레이블 100개로 $\omega$를 읽을 때 무작위 초기화는 $R^2 = -0.937$(128차원 특징에 비해 레이블이 너무 적어 과적합한다), 대조 학습은 0.723이다. 레이블 1,000개에서는 0.756 대 0.919로 차이가 줄어든다.
- **$\gamma$도 좋아졌다(0.540 → 0.855).** 증강이 위상·진폭·잡음만 바꾸고 $\gamma$는 보존했으므로, 짝을 맞히려면 $\gamma$를 표현에 담아야 한다. 증강이 보존하는 성질이 표현에 남는다.
- **정렬만 쓴 encoder는 완전히 붕괴했다.** 평균 표준편차 0.000, 유효 rank 1.88이고, 어떤 probe도 아무것도 읽지 못한다. 학습 손실은 0에 가까웠을 것이므로, 손실 곡선만 보면 학습이 아주 잘된 것처럼 보인다.

이어서 다음을 수행한다.

1. 증강에서 위상 무작위화를 빼고 같은 실험을 반복한다. probe 성능이 어떻게 바뀌는지, 표현이 위상을 담게 되는지(위상의 $\cos\phi$, $\sin\phi$를 probe로 읽어 본다) 확인한다.
2. 정렬만 쓴 encoder에 VICReg의 분산 항(차원별 표준편차의 hinge)을 더해 붕괴가 사라지는지 본다. 평균 표준편차와 유효 rank를 함께 기록한다.
3. 레이블 수를 30, 100, 300, 1000으로 바꿔 사전학습·무작위 초기화·입력 그대로의 probe 곡선을 그리고, 시드 3개로 반복해 변동을 붙인다.
4. 사전학습 자료와 probe 시험 자료가 같은 $(\omega, \gamma)$ 계열을 공유하지 않는지 확인한다. 위 코드는 매번 새로 뽑으므로 겹치지 않는다. 실제 자료에서는 이것이 저절로 보장되지 않는다.

마지막으로 NT-Xent 손실을 NumPy로 구현한다. 테스트가 바로 채점한다.

::exercise{id="nt-xent"}

## 실패 사례

- **사전학습 데이터와 평가 데이터가 겹치는 것.** 레이블 없이 썼더라도 시험 표본을 사전학습에서 봤다면 그 표본의 표현은 유리하다. 웹·공개 데이터베이스에서 모은 사전학습 자료에는 공개 벤치마크의 시험 표본이 섞여 있기 쉽다. 같은 단위(환자·분자·관측 영역)로 분할하고, 거의 같은 중복까지 찾아 뺀다.
- **downstream 튜닝 예산이 다른 비교.** 사전학습 모델의 fine-tuning에는 학습률을 여러 개 시도하고 무작위 초기화 baseline에는 기본값 하나만 쓰면, 차이의 일부는 튜닝의 효과다. 같은 탐색 공간, 같은 시도 횟수, 같은 학습 step을 쓴다.
- **pretext 손실을 표현 품질로 읽는 것.** 붕괴한 표현은 정렬 손실이 0이고, shortcut을 배운 표현은 contrastive 손실이 더 낮았다. 손실 곡선과 함께 하위 과제 probe와 붕괴 지표(표준편차, 유효 rank)를 학습 중에 기록한다.
- **이미지 증강을 과학 데이터에 그대로 쓰는 것.** 증강은 "이 성질은 중요하지 않다"는 선언이다. 하위 과제가 그 성질에 의존하면 사전학습이 필요한 정보를 지운다.

## 통과 기준

- 자기지도 사전학습의 효과를 무작위 초기화 기준선, 겹치지 않는 사전학습·평가 분할, 같은 downstream 예산의 통제된 비교로 보이고, 레이블 수 두 단계 이상에서 결과를 보고한다.
- 사전학습한 표현의 붕괴 여부를 차원별 표준편차와 유효 rank로 확인하고, 증강이 보존·제거하는 성질을 하위 과제와 연결해 설명한다.
- 연습문제 `nt-xent`의 테스트를 통과한다.

## 참고 문헌

- Chen, Kornblith, Norouzi, Hinton (2020). A simple framework for contrastive learning of visual representations. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 1597–1607. — SimCLR, 증강 조합·비선형 projection head·큰 batch의 효과
- Wang, Isola (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 9929–9939. — contrastive 손실의 정렬·균일성 분해
- Grill 외 (2020). Bootstrap your own latent: A new approach to self-supervised learning. arXiv:2006.07733. — negative 없이 이동평균 target을 예측하는 BYOL
- Chen, He (2021). Exploring simple Siamese representation learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR 2021), 15750–15758. — stop-gradient가 붕괴를 막는다는 SimSiam의 관찰
- Bardes, Ponce, LeCun (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. *International Conference on Learning Representations* (ICLR 2022). arXiv:2105.04906. — 분산·공분산 정규화로 붕괴 방지
- Jing, Vincent, LeCun, Tian (2022). Understanding dimensional collapse in contrastive self-supervised learning. *ICLR 2022*. arXiv:2110.09348. — contrastive 학습의 차원 붕괴와 원인
- He, Chen, Xie, Li, Dollár, Girshick (2022). Masked autoencoders are scalable vision learners. *CVPR 2022*, 16000–16009. — 높은 비율로 가린 패치를 복원하는 MAE
- Assran 외 (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR 2023*, 15619–15629. — 표현 공간에서 가린 블록을 예측하는 I-JEPA
