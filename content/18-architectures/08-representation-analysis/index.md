---
id: architectures/representation-analysis
title: "표현 분석"
title_en: "Analyzing Learned Representations"
tier: advanced
status: draft
prerequisites: [architectures/inductive-bias, classical-ml/unsupervised-learning]
---

# 표현 분석

학습된 신경망의 중간층은 입력을 수백, 수천 차원의 벡터로 바꾼다. 과학 연구에서는 이 벡터가 무엇을 담고 있는지 묻고 싶을 때가 많다. 물성 예측 모델의 중간층이 결정 구조의 대칭을 알고 있는가, 기후 모델의 표현이 계절을 구별하는가, 단백질 언어모델의 embedding이 접힘 구조를 반영하는가. 이런 질문에 답하는 도구가 **표현 분석**(representation analysis)이다.

이 장의 도구는 세 가지다. 표현에서 어떤 양을 읽어 낼 수 있는지 재는 **probe**, 두 표현이 얼마나 비슷한지 재는 **유사도 지표**, 그리고 표현 공간의 이웃을 직접 확인하는 **검색**이다. 세 도구 모두 숫자를 하나 내놓는데, 그 숫자는 대조 실험 없이는 거의 해석할 수 없다. 이 장의 실험은 학습하지 않은 무작위 신경망조차 이 지표들에서 높은 점수를 받는다는 데서 출발한다. 16.5장에서 본 t-SNE·UMAP의 해석 한계도 여기서 다시 등장한다.

## 학습 목표

- linear probe·CKA·최근접 이웃 검색으로 학습된 표현을 정량적으로 분석한다
- 무작위 초기화 망·섞은 정답 같은 대조 실험과 함께 분석 결과를 해석하고, 저차원 시각화의 한계를 설명한다

## probing

**왜 필요한가.** 모델의 출력만 보면 모델이 **무엇을 근거로** 그 출력을 냈는지 알 수 없다. 중간층의 표현에 어떤 정보가 있는지 재면, 모델 내부에서 정보가 어디서 생기고 어디서 사라지는지 추적할 수 있다.

**정의.** 층 $\ell$의 표현 $\mathbf{z}^{(\ell)}(\mathbf{x}) \in \mathbb{R}^{D}$를 고정하고, 그 위에 작은 모델 $g$를 학습해 어떤 양 $y$(레이블, 물리량)를 예측하게 한다. 이 작은 모델을 **probe**라 한다. 원래 모델의 파라미터는 건드리지 않는다. 가장 많이 쓰는 것은 선형 모델인 **linear probe**다(Alain, Bengio, 2016).

$$
\hat{y} = \mathbf{w}^\top \mathbf{z}^{(\ell)}(\mathbf{x}) + b, \qquad
\min_{\mathbf{w}, b} \sum_{i \in \text{train}} \big(y_i - \hat{y}_i\big)^2 + \lambda \lVert \mathbf{w} \rVert^2.
$$

probe의 성능은 **held-out 자료**에서 잰다. 회귀면 $R^2$, 분류면 정확도다. "층 $\ell$의 표현에서 $y$를 선형으로 읽어 낼 수 있다"가 probe 결과의 정확한 의미다.

**코드.** 감쇠 진동 신호 $x(t) = e^{-\gamma t}\cos(\omega t + \phi) + \varepsilon$를 자료로 쓴다. 각진동수 $\omega$와 감쇠율 $\gamma$가 우리가 찾을 물리량이고, 위상 $\phi$는 무작위다. 세 가지 표현에 같은 ridge probe를 적용한다. 신호 자체(64시점), 그리고 **학습하지 않은** 무작위 ReLU 층 1개와 2개를 지난 표현이다. 마지막 열은 $\omega$를 무작위로 섞은 정답으로 같은 probe를 학습한 대조군이다.

```python {exec="browser" session="rep"}
import numpy as np

rng = np.random.default_rng(0)
N, T = 2000, 64
t_s = np.linspace(0, 1, T)                              # 시간 (T,), 단위 s
omega = rng.uniform(2 * np.pi * 3, 2 * np.pi * 8, N)    # 각진동수 (N,), 단위 rad/s
gamma = rng.uniform(0.5, 4.0, N)                        # 감쇠율 (N,), 단위 1/s
phi = rng.uniform(0, 2 * np.pi, N)
X = np.exp(-gamma[:, None] * t_s) * np.cos(omega[:, None] * t_s + phi[:, None])
X += 0.1 * rng.normal(size=(N, T))                      # 감쇠 진동 신호 (N, T)
tr, te = np.arange(1500), np.arange(1500, N)

def probe_r2(Z, y, lam=1e-2):                           # 표준화 + ridge, test R²
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zs = (Z - mu) / sd
    w = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * len(tr) * np.eye(Z.shape[1]), Zs[tr].T @ (y[tr] - y[tr].mean()))
    pred = Zs[te] @ w + y[tr].mean()
    return 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()

W1 = rng.normal(size=(T, 256)) / np.sqrt(T)             # 학습하지 않은 무작위 층
W2 = rng.normal(size=(256, 256)) / np.sqrt(256)
H1 = np.maximum(0, X @ W1)                              # (N, 256)
H2 = np.maximum(0, H1 @ W2)                             # (N, 256)
print("표현                  ω probe R²   γ probe R²   ω 섞은 정답 R²")
for name, Z in [("입력 신호 (64)", X), ("무작위 층 1 (256)", H1), ("무작위 층 2 (256)", H2)]:
    print(f"{name:18}   {probe_r2(Z, omega):8.3f}   {probe_r2(Z, gamma):9.3f}   {probe_r2(Z, rng.permutation(omega)):10.3f}")
```

```output
표현                  ω probe R²   γ probe R²   ω 섞은 정답 R²
입력 신호 (64)             -0.062      -0.050       -0.044
무작위 층 1 (256)           0.968       0.905       -0.109
무작위 층 2 (256)           0.942       0.885       -0.081
```

**직관.** 결과는 두 방향으로 놀랍다.

- **입력 신호에는 $\omega$와 $\gamma$의 정보가 다 들어 있지만, 선형으로는 읽히지 않는다.** 위상 $\phi$가 무작위라서 같은 $\omega$라도 신호의 부호와 모양이 제각각이고, 시점 값들의 어떤 선형 결합도 $\omega$와 상관이 없다. $R^2$가 0 근처(음수는 평균 예측보다 나쁘다는 뜻)인 것은 정보가 **없어서**가 아니라 **선형으로 꺼낼 수 없어서**다.
- **학습하지 않은 무작위 층 하나를 지나면 $R^2$가 0.968이 된다.** 무작위 ReLU 특징 256개는 입력의 비선형 함수를 많이 만들어 내고, 그중 일부의 선형 결합이 $\omega$를 잘 근사한다. 이것은 무작위 특징으로 커널 회귀를 근사하는 것과 같은 원리다.

섞은 정답에 대한 $R^2$는 모두 0 아래다. 이 probe가 held-out 자료에서 우연한 패턴을 성능으로 착각하지 않는다는 확인이다.

**연구에서의 의미.** "학습된 모델의 층 2에서 $\omega$의 probe $R^2$가 0.95였다"는 보고는 그 자체로는 아무것도 말하지 않는다. 같은 구조의 **무작위 초기화 망**이 이미 0.94를 낸다면, 학습이 $\omega$의 표현에 기여한 몫은 작다. probe 결과는 항상 다음 대조군과 함께 보고한다.

- **무작위 초기화한 같은 구조의 망**: 구조만으로 얻는 점수.
- **입력 자체**: 아무 처리도 하지 않았을 때의 점수.
- **섞은 정답**(control task): probe가 표현이 아니라 probe 자신의 용량으로 외운 점수. Hewitt, Liang(2019)은 언어 표현의 probe에서, 실제 과제의 정확도와 이런 대조 과제의 정확도의 차이를 **selectivity**라 부르고, MLP probe는 대조 과제도 잘 맞혀 selectivity가 낮은 반면 linear probe는 높다는 것을 보였다. probe의 용량이 클수록 "표현에 정보가 있다"와 "probe가 외웠다"를 구별하기 어렵다.

::quiz{id="probe-random-baseline"}

**probe가 말하지 않는 것.** probe의 높은 성능은 정보가 표현에 **있다**는 뜻이지, 모델이 그 정보를 **사용한다**는 뜻이 아니다. 무작위 층의 표현에서 $\omega$를 읽을 수 있다고 해서 무작위 층이 $\omega$를 "계산한" 것은 아니다. 모델이 그 정보를 쓰는지 알려면 개입이 필요하다. 표현에서 그 방향을 제거하거나 바꾼 뒤 모델의 출력이 달라지는지 확인한다. 이런 개입 기반 분석은 21.5장(해석가능성)에서 다룬다.

::quiz{id="probe-not-usage"}

## 표현 유사도

**왜 필요한가.** 두 모델(또는 한 모델의 두 층)이 같은 것을 배웠는지 묻고 싶을 때가 있다. 시드만 다른 두 모델이 같은 표현에 이르는가, 사전학습 모델의 어느 층이 과학 데이터에 가장 가까운 표현을 갖는가. 그런데 표현을 좌표별로 비교할 수는 없다. 뉴런의 순서를 바꾸거나 표현 공간을 회전해도 모델의 기능은 그대로이기 때문이다. 비교 지표는 이런 변환에 둔감해야 한다.

**정의.** 같은 입력 $N$개에 대한 두 표현 행렬 $\mathbf{A} \in \mathbb{R}^{N \times D_1}$, $\mathbf{B} \in \mathbb{R}^{N \times D_2}$를 열마다 중심화한 뒤, **linear CKA**(centered kernel alignment)는

$$
\mathrm{CKA}(\mathbf{A}, \mathbf{B}) = \frac{\lVert \mathbf{A}^\top\mathbf{B} \rVert_F^2}{\lVert \mathbf{A}^\top\mathbf{A} \rVert_F\,\lVert \mathbf{B}^\top\mathbf{B} \rVert_F}
$$

로 정의한다(Kornblith 외, 2019). $\lVert \cdot \rVert_F$는 Frobenius 노름이다. 같은 값을 표본 사이의 유사도 행렬(Gram 행렬) $\mathbf{K} = \mathbf{A}\mathbf{A}^\top$, $\mathbf{L} = \mathbf{B}\mathbf{B}^\top$로 쓸 수도 있다. 이 형태는 "두 표현이 표본 쌍 사이의 유사도 구조를 얼마나 비슷하게 매기는가"를 잰다는 것을 보여 준다. 값은 0과 1 사이다. 차원 $D_1$, $D_2$가 달라도 계산할 수 있다.

CKA는 **직교 변환**과 **등방 스케일**(전체에 같은 상수를 곱하는 것)에 불변이다. 일반적인 가역 선형변환에는 불변이 아니다. 이전에 쓰이던 CCA(canonical correlation analysis) 계열 지표, 예컨대 SVCCA(Raghu 외, 2017)는 가역 선형변환에 불변이 되도록 설계되었는데, Kornblith 외(2019)는 표현의 차원이 표본 수보다 크면 그런 불변성 때문에 지표가 서로 다른 표현을 구별하지 못할 수 있다고 지적했다.

```python {exec="browser" session="rep"}
def linear_cka(A, B):                                   # (N, D1), (N, D2) → 0과 1 사이
    A = A - A.mean(0); B = B - B.mean(0)                # 열 중심화
    return np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))

n = 500                                                 # 계산량을 줄이려 표본 500개로 비교
A1, A2 = H1[:n], H2[:n]                                 # (n, 256)
Q, _ = np.linalg.qr(rng.normal(size=(256, 256)))        # 무작위 직교행렬
M = rng.normal(size=(256, 256))                         # 무작위 가역 행렬
print(f"CKA(H1, H1 Q)       = {linear_cka(A1, A1 @ Q):.3f}   (직교 변환)")
print(f"CKA(H1, 3 H1)       = {linear_cka(A1, 3 * A1):.3f}   (등방 스케일)")
print(f"CKA(H1, H1 M)       = {linear_cka(A1, A1 @ M):.3f}   (일반 가역 선형변환)")
print(f"CKA(H1, H2)         = {linear_cka(A1, A2):.3f}   (같은 망의 층 1, 2)")

W1b = rng.normal(size=(T, 256)) / np.sqrt(T)            # 다른 시드의 무작위 층
print(f"CKA(H1, 다른 망 H1) = {linear_cka(A1, np.maximum(0, X[:n] @ W1b)):.3f}   (학습 없는 두 망)")
noise_a, noise_b = rng.normal(size=(n, 256)), rng.normal(size=(n, 256))   # 서로 무관한 두 표현
big = 10 * rng.normal(size=(n, 1))                      # 두 표현이 공유하는 큰 분산 한 방향
print(f"CKA(무관, 무관)     = {linear_cka(noise_a, noise_b):.3f}")
print(f"CKA(무관+공통축, 무관+공통축) = {linear_cka(np.hstack([noise_a, big]), np.hstack([noise_b, big])):.3f}")
```

```output
CKA(H1, H1 Q)       = 1.000   (직교 변환)
CKA(H1, 3 H1)       = 1.000   (등방 스케일)
CKA(H1, H1 M)       = 0.978   (일반 가역 선형변환)
CKA(H1, H2)         = 0.935   (같은 망의 층 1, 2)
CKA(H1, 다른 망 H1) = 0.941   (학습 없는 두 망)
CKA(무관, 무관)     = 0.337
CKA(무관+공통축, 무관+공통축) = 0.973
```

처음 세 줄이 불변성의 정의를 확인한다. 나머지 줄은 해석의 함정을 보여 준다.

- **학습하지 않은 두 망도 CKA가 0.941이다.** 같은 입력에서 출발한 ReLU 특징은 입력의 큰 구조(신호의 크기, 감쇠 정도)를 공유하므로, 무작위 가중치끼리도 표본 사이의 유사도 구조가 비슷해진다. "두 학습된 모델의 CKA가 0.9로 높다"는 결과는 무작위 망끼리의 CKA와 비교해야 의미가 있다.
- **서로 무관한 표현도 0이 아니다(0.337).** 여기서는 계산량을 줄이려 표본 500개만 썼는데, 표본 수가 차원(256)에 비해 충분히 크지 않으면 우연한 상관이 크게 남는다. 같은 계산을 표본 2,000개로 하면 약 0.11로 줄어든다. 같은 표본 수에서 잰 이 값이 "아무 관계 없음"의 기준선이다.
- **공유하는 큰 분산 방향 하나가 CKA를 지배한다.** 나머지 256차원이 서로 무관해도, 분산이 큰 한 방향을 공유하면 CKA는 0.973이 된다. CKA는 분산이 큰 방향에 가중치를 많이 둔다. 표현에 배치 효과나 측정 장비 같은 큰 공통 요인이 섞여 있으면, CKA는 그 요인의 공유를 표현의 유사도로 보고한다.

::quiz{id="cka-invariances"}

:::note[분야 예시]
여러 실험실에서 측정한 현미경 영상으로 학습한 두 모델의 표현을 CKA로 비교한다고 하자. 두 모델 모두 영상의 밝기(실험실마다 다른 조명)를 강하게 표현한다면, 생물학적 특징에 대한 표현이 전혀 달라도 CKA는 높게 나올 수 있다. 같은 분석을 밝기를 정규화한 입력이나 실험실별로 중심화한 표현에서 반복하고, 실험실 레이블을 probe로 읽어 얼마나 강하게 표현되는지도 함께 보고한다.
:::

## 검색과 최근접 이웃

**왜 필요한가.** probe와 CKA는 요약 수치다. 표현을 가장 직접적으로 점검하는 방법은 표현 공간에서 가까운 표본들이 실제로 비슷한지 눈으로, 그리고 수치로 확인하는 것이다. embedding 기반 검색은 18.5장의 RAG, 과학 문헌 검색, 분자·재료 데이터베이스의 유사 구조 찾기에서 그대로 쓰인다.

**방법.** 질의 표본마다 표현 공간에서 가장 가까운 학습 표본을 찾고, 둘의 물리량이 얼마나 다른지 잰다. 이웃의 물리량이 비슷하면 표현이 그 물리량을 거리 구조에 담고 있다는 뜻이다. 비교 기준은 무작위로 고른 이웃이다.

```python {exec="browser" session="rep"}
def nn_error(E, y):                                     # test 질의마다 train에서 최근접 이웃, |Δy| 평균
    d2 = (E[te] ** 2).sum(1)[:, None] - 2 * E[te] @ E[tr].T + (E[tr] ** 2).sum(1)[None]
    return np.abs(y[te] - y[tr][np.argmin(d2, axis=1)]).mean()

standardize = lambda Z: (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-8)
temp_mk = rng.normal(298_000, 2_000, size=(N, 1))      # 무관한 메타데이터: 센서 온도 (N, 1), 단위 mK
rand_nn = np.abs(omega[te] - omega[tr][rng.integers(len(tr), size=len(te))]).mean()
print(f"무작위 이웃                  |Δω| = {rand_nn:5.2f} rad/s")
for name, E in [("입력 신호", X), ("무작위 층 1", H1),
                ("층 1 + 센서 온도(mK)", np.hstack([H1, temp_mk])),
                ("위를 열마다 표준화", standardize(np.hstack([H1, temp_mk])))]:
    print(f"{name:20}  |Δω| = {nn_error(E, omega):5.2f} rad/s   |Δγ| = {nn_error(E, gamma):4.2f} 1/s")
```

```output
무작위 이웃                  |Δω| = 10.32 rad/s
입력 신호                 |Δω| =  0.54 rad/s   |Δγ| = 0.35 1/s
무작위 층 1               |Δω| =  0.64 rad/s   |Δγ| = 0.36 1/s
층 1 + 센서 온도(mK)       |Δω| = 10.25 rad/s   |Δγ| = 1.14 1/s
위를 열마다 표준화            |Δω| =  0.64 rad/s   |Δγ| = 0.36 1/s
```

세 가지를 읽는다.

- **probe가 실패한 입력 신호에서도 최근접 이웃은 잘 맞는다(0.54 rad/s).** 최근접 이웃은 비선형 판단이다. 같은 $\omega$·$\gamma$·$\phi$를 가진 신호는 서로 가깝고, 학습 표본 1,500개 중에는 위상까지 비슷한 이웃이 있다. probe와 검색은 서로 다른 질문("선형으로 읽히는가"와 "가까운 것끼리 비슷한가")에 답한다.
- **무관한 열 하나가 검색을 망친다.** 센서 온도를 mK 단위로 붙이면 그 열의 수치 범위(수천)가 다른 256개 열(1 안팎)을 압도해, 이웃이 센서 온도가 비슷한 표본이 된다. $\omega$ 오차는 무작위 이웃 수준(10.25 rad/s)으로 돌아간다.
- **열마다 표준화하면 회복된다.** 그러나 이것은 그 열이 257개 중 하나로 희석되었기 때문일 뿐이다. 거리 기반 분석에서는 어떤 특징이 거리를 지배하는지 먼저 확인하고, 분석 목적과 무관한 메타데이터는 표현에서 뺀다.

::quiz{id="nn-unscaled-feature"}

## 시각화

**왜 필요한가.** 수백 차원의 표현을 2차원 산점도로 그려 군집을 확인하는 것은 표현 분석에서 가장 흔한 그림이다. 그러나 2차원 그림은 정보의 대부분을 버린 결과다. 그림에서 보이지 않는 구조가 표현에 있을 수 있고, 그림에서 보이는 구조가 표현에 없을 수도 있다.

**PCA의 경우.** PCA(7.6장)로 투영하면 분산이 큰 방향 두 개만 남는다. 분산이 크다는 것과 우리가 관심 있는 구별에 중요하다는 것은 다른 이야기다. 무작위 층 1의 표현에서 "빠른 진동인가"($\omega$가 중앙값보다 큰가)를 판별하는 선형 분류기를, 주성분 $k$개만 남긴 표현에서 학습해 본다.

```python {exec="browser" session="rep"}
fast = (omega > np.median(omega)).astype(float)         # 이진 레이블: 빠른 진동인가 (N,)
Zc = H1 - H1[tr].mean(0)
U, S, Vt = np.linalg.svd(Zc[tr], full_matrices=False)
evr = S**2 / (S**2).sum()                               # 설명 분산 비율

def probe_acc(Z):                                       # 최소제곱 선형 분류기, test 정확도
    Zb = np.hstack([Z, np.ones((N, 1))])
    w = np.linalg.lstsq(Zb[tr], 2 * fast[tr] - 1, rcond=None)[0]
    return ((Zb[te] @ w > 0) == fast[te]).mean()

for k in [2, 10, 50, 256]:
    print(f"PC {k:3d}개  설명 분산 {evr[:k].sum():.3f}   빠른 진동 판별 정확도 {probe_acc(Zc @ Vt[:k].T):.3f}")
```

```output
PC   2개  설명 분산 0.293   빠른 진동 판별 정확도 0.434
PC  10개  설명 분산 0.691   빠른 진동 판별 정확도 0.888
PC  50개  설명 분산 0.939   빠른 진동 판별 정확도 0.980
PC 256개  설명 분산 1.000   빠른 진동 판별 정확도 0.994
```

주성분 2개로 그린 산점도에서는 빠른 진동과 느린 진동이 완전히 섞여 보일 것이다. 두 주성분으로 학습한 분류기의 정확도 0.434는 우연 수준이다(학습 자료에서 조금 더 많던 쪽을 주로 고르는데, test 자료에서는 그쪽이 44%다). 그런데 같은 표현 전체로는 0.994로 거의 완벽하게 구별된다. 이 표현에서 분산이 큰 방향은 신호의 감쇠와 크기를 따라가고, 진동수를 구별하는 정보는 분산이 작은 방향에 흩어져 있다. **2차원 PCA 그림에서 군집이 보이지 않는다는 것은 표현에 그 정보가 없다는 증거가 아니다.**

**t-SNE와 UMAP의 경우.** 16.5장에서 본 비선형 투영은 반대 방향의 함정을 가진다. 이웃 관계를 보존하는 데 집중하므로 군집은 잘 드러나지만, 군집의 크기와 군집 사이의 거리는 의미가 없을 수 있고, 하이퍼파라미터(perplexity 등)에 따라 무작위 잡음에서도 군집처럼 보이는 덩어리가 생길 수 있다(Wattenberg 외, 2016). 그림에서 본 군집은 원래 공간에서의 정량 분석(probe, 최근접 이웃, 군집 지표)으로 확인한다.

::quiz{id="pca-explained-variance"}

:::note[분야 예시]
단일세포 RNA 시퀀싱 연구에서는 세포마다의 발현 벡터를 UMAP으로 2차원에 그리고, 덩어리마다 세포 유형을 붙이는 일이 흔하다. 같은 그림이 하이퍼파라미터나 무작위 시드에 따라 군집의 배치와 크기가 달라질 수 있으므로, 군집 사이의 거리로 세포 유형의 "가까움"을 주장하지 않는다. 세포 유형 판별은 원래 발현 공간의 군집이나 표지 유전자 발현으로 확인하고, 2차원 그림은 그 결과를 보여 주는 용도로만 쓴다.
:::

## 실습

학습된 모델의 층별 표현에 linear probe를 적용해, 물리량이 어디서 인코딩되는지 분석한다. 아래는 신호에서 $\omega$만 예측하도록 학습한 2층 MLP의 각 층에 같은 ridge probe를 적용한 것이다. **대조군은 학습 전(무작위 초기화) 같은 망**이다. $\gamma$는 학습 목표가 아니다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
N, T = 2000, 64
t_s = np.linspace(0, 1, T)                              # 앞 셀과 같은 감쇠 진동 자료
omega = rng.uniform(2 * np.pi * 3, 2 * np.pi * 8, N); gamma = rng.uniform(0.5, 4.0, N)
phi = rng.uniform(0, 2 * np.pi, N)
X = np.exp(-gamma[:, None] * t_s) * np.cos(omega[:, None] * t_s + phi[:, None]) + 0.1 * rng.normal(size=(N, T))
tr, te = np.arange(1500), np.arange(1500, N)

def probe_r2(Z, y, lam=1e-2):                           # 앞 셀과 같은 ridge probe
    Zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-8)
    w = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * len(tr) * np.eye(Z.shape[1]), Zs[tr].T @ (y[tr] - y[tr].mean()))
    return 1 - ((y[te] - Zs[te] @ w - y[tr].mean()) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()

l1, l2 = nn.Sequential(nn.Linear(T, 128), nn.ReLU()), nn.Sequential(nn.Linear(128, 128), nn.ReLU())
net = nn.Sequential(l1, l2, nn.Linear(128, 1))         # (B, 64) → (B, 1): ω만 학습한다
xt = torch.tensor(X, dtype=torch.float32)               # (N, T)
yt = torch.tensor((omega - omega[tr].mean()) / omega[tr].std(), dtype=torch.float32)[:, None]

def layer_reps():                                       # 층 1, 2의 표현 (N, 128) 두 개
    with torch.no_grad():
        h1 = l1(xt); return [h1.numpy(), l2(h1).numpy()]

before = layer_reps()                                   # 학습 전(무작위 초기화) 같은 망
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(3000):
    idx = torch.as_tensor(rng.choice(tr, 128))
    loss = nn.functional.mse_loss(net(xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
after = layer_reps()
print("층    학습 전 ω R²  학습 후 ω R²   학습 전 γ R²  학습 후 γ R²")
for i in range(2):
    print(f"층 {i + 1}     {probe_r2(before[i], omega):6.3f}       {probe_r2(after[i], omega):6.3f}"
          f"        {probe_r2(before[i], gamma):6.3f}       {probe_r2(after[i], gamma):6.3f}")
```

```output
층    학습 전 ω R²  학습 후 ω R²   학습 전 γ R²  학습 후 γ R²
층 1      0.916        0.983         0.876        0.900
층 2      0.885        0.992         0.835        0.887
```

대조군과 함께 읽으면 결론이 바뀐다. 학습 후 층 2의 $\omega$ probe $R^2$ 0.992만 보면 "학습이 $\omega$를 표현으로 만들었다"고 말하고 싶어진다. 그러나 학습 전 같은 망이 이미 0.885다. 학습이 더한 몫은 그 차이(층 2에서 0.885 → 0.992)이고, 학습 목표에 가까운 층 2에서 차이가 더 크다. 학습 목표가 아닌 $\gamma$도 조금 올랐다(0.835 → 0.887). 이 자료에서 $\omega$를 정확히 추정하려면 감쇠의 영향을 구별해야 하므로, $\gamma$ 정보가 함께 남는 것은 자연스럽다. 그러나 이것도 모델이 $\gamma$를 **쓴다**는 증거는 아니다.

자기 모델로 다음을 수행한다.

1. 층마다 probe 성능을 학습된 망·무작위 초기화 망·입력 자체에 대해 표로 만든다.
2. 섞은 정답으로 같은 probe를 학습해, probe가 외워서 내는 점수를 확인한다. 선형 probe와 작은 MLP probe를 모두 써서 selectivity를 비교한다.
3. 관심 있는 층 하나에서 CKA와 최근접 이웃 오차를 함께 보고한다. 무작위 망끼리의 CKA를 기준선으로 쓴다.
4. 2차원 그림을 그렸다면, 그림에서 받은 인상을 원래 공간의 probe나 최근접 이웃으로 확인한다.

이어서 CKA를 Gram 행렬 형태로 구현한다. 테스트가 바로 채점한다.

::exercise{id="cka-gram"}

## 실패 사례

- **probe의 높은 성능을 모델이 그 정보를 '사용한다'는 증거로 보는 것.** probe는 정보가 표현에서 읽힌다는 것만 보여 준다. 무작위 망의 표현에서도 같은 정보가 읽힐 수 있고, 읽힌다고 해서 모델의 출력이 그 정보에 의존하는 것도 아니다. 사용 여부는 개입 실험으로 확인한다.
- **대조군 없이 probe·CKA 수치를 보고하는 것.** 무작위 초기화 망, 입력 자체, 섞은 정답, 서로 무관한 표현의 CKA가 각 지표의 기준선이다. 기준선 없이 "0.9는 높다"고 말할 수 없다.
- **용량이 큰 probe로 "정보가 있다"고 주장하는 것.** MLP probe는 표현이 아니라 probe 자신이 과제를 풀 수 있다. probe의 용량을 키울수록 selectivity를 함께 확인한다.
- **2차원 투영에서 군집이 없다고(또는 있다고) 결론 내리는 것.** PCA 두 축은 분산이 큰 방향일 뿐이고, t-SNE·UMAP의 군집 크기와 거리는 의미가 없을 수 있다. 원래 공간에서 정량적으로 확인한다.

## 통과 기준

- 학습된 모델의 층별 표현에 linear probe를 적용하고, 무작위 초기화 망·입력·섞은 정답 대조군과 함께 표로 보고한다.
- 두 표현의 CKA를 계산하고, 무작위 망끼리의 CKA와 큰 공통 분산 방향의 영향을 고려해 해석한다.
- 최근접 이웃 검색과 2차원 투영 결과를 원래 공간의 정량 지표로 확인하고, 연습문제 `cka-gram`의 테스트를 통과한다.

## 참고 문헌

- Alain, Bengio (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644. — 층별 linear probe
- Hewitt, Liang (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP 2019*. — 대조 과제와 selectivity
- Kornblith, Norouzi, Lee, Hinton (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*, PMLR 97, 3519–3529. — CKA와 표현 유사도 지표의 불변성
- Raghu, Gilmer, Yosinski, Sohl-Dickstein (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. *NIPS 2017*. — CCA 기반 표현 비교
- Wattenberg, Viégas, Johnson (2016). How to Use t-SNE Effectively. *Distill*. doi:10.23915/distill.00002 — t-SNE 그림을 읽을 때의 함정
