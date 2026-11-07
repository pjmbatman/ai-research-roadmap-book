---
id: classical-ml/unsupervised-learning
title: "비지도학습과 표현 탐색"
title_en: "Unsupervised Learning and Exploration"
tier: core
status: draft
prerequisites: [linear-algebra/svd-and-pca]
---

# 비지도학습과 표현 탐색

새 데이터를 받으면 label을 붙이기 전에 먼저 구조를 본다. 측정값이 몇 개의 방향에 몰려 있는가, 샘플이 몇 개의 무리로 나뉘는가, 다른 것과 동떨어진 샘플이 있는가. 비지도학습(unsupervised learning)은 이런 질문에 답하는 도구 모음이다. 차원 축소, 군집(clustering), 밀도 추정, 이상 탐지가 여기에 속한다.

이 도구들은 답을 **항상** 낸다는 점에서 위험하다. k-means에 $k = 3$을 주면 구조가 없는 데이터에서도 군집 세 개를 돌려주고, t-SNE는 거의 어떤 데이터에서도 보기 좋게 분리된 섬들을 그린다. 이 장은 각 방법이 무엇을 최적화하는지 정리하고, 그 결과를 독립적인 기준으로 검증하는 습관을 만드는 데 초점을 둔다.

## 학습 목표

- 차원 축소·군집·밀도추정·이상탐지를 사용한다
- 시각화 방법의 해석 한계를 안다
- 군집 수와 군집 결과를 독립적인 기준으로 검증한다

## 차원 축소

**왜 필요한가.** 특징이 수십 개를 넘으면 산점도로 볼 수 없고, 거리 기반 방법은 차원의 저주(16.3장)로 약해진다. 차원 축소는 데이터를 적은 수의 좌표로 요약한다. 그런데 "좋은 요약"의 기준은 방법마다 다르다. 7.6장의 PCA는 **분산**을 최대로 보존하는 직교 방향을 찾는다. 이 절은 다른 기준을 쓰는 두 방법, kernel PCA와 ICA를 PCA와 비교한다.

**kernel PCA.** PCA는 선형 사영이라 데이터가 곡면 위에 놓이면 펴지 못한다. kernel PCA는 특징 사상 $\phi(\mathbf{x})$로 보낸 공간에서 PCA를 하되, $\phi$를 직접 계산하지 않고 커널 $k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top\phi(\mathbf{x}')$로 만든 $N \times N$ Gram 행렬을 중심화해 고유분해한다(Schölkopf 외, 1998). 커널 트릭은 16.3장에서 다룬다. 대가는 두 가지다. 계산이 샘플 수 $N$에 대해 $O(N^3)$로 커지고, 커널과 그 폭이 결과를 크게 바꾸는데 그것을 고를 객관적 기준이 비지도 상황에는 없다.

**ICA.** 독립성분분석(ICA, independent component analysis)은 관측이 서로 **독립인** 원천 신호의 선형 혼합이라고 가정한다.

$$
\mathbf{x} = \mathbf{A}\mathbf{s}, \qquad s_1, \dots, s_D \text{ 서로 독립}.
$$

$\mathbf{x}$는 센서 $D$개의 관측, $\mathbf{s}$는 원천, $\mathbf{A}$는 알 수 없는 혼합 행렬이다. PCA가 만드는 성분은 **무상관**(공분산 0)일 뿐 독립은 아니다. 무상관은 2차 통계량만의 조건이라, 무상관인 성분을 아무렇게나 회전해도 여전히 무상관이다. 그래서 PCA는 원천을 하나로 정하지 못한다. ICA는 3차 이상의 통계량, 즉 분포의 **비가우스성**(non-Gaussianity)을 최대화하는 회전을 찾아 이 모호함을 없앤다. 독립인 신호의 합은 중심극한정리에 따라 각 신호보다 가우스에 가까워지므로, 가장 덜 가우스적인 방향이 원천 하나에 해당한다는 생각이다(Hyvärinen, Oja, 2000). 같은 이유로 원천 중 가우스 분포를 따르는 것이 둘 이상이면 ICA도 그것들을 분리하지 못한다. 그리고 원천의 순서·부호·크기는 정할 수 없다.

사인파와 사각파를 센서 두 개로 섞은 신호에서 확인한다. 각 원천과 가장 닮은 성분의 상관계수 절댓값을 비교한다.

```python {exec="browser" session="unsup"}
import warnings
import numpy as np
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

t_s = np.linspace(0, 8, 2000)                                   # (T,) 시간 [s]
S = np.c_[np.sin(2 * np.pi * 1.0 * t_s),                       # 원천 1: 사인파
          np.sign(np.sin(2 * np.pi * 0.3 * t_s))]              # 원천 2: 사각파, (T, 2)
A = np.array([[1.0, 0.6], [0.5, 1.0]])                         # 혼합 행렬 (센서 2개)
X = S @ A.T                                                    # (T, 2) 관측

def match(Z):                                                  # 각 원천과 가장 닮은 성분의 |상관|
    C = np.abs(np.corrcoef(S.T, Z.T)[:2, 2:])                  # (원천, 성분)
    return C.max(axis=1).round(2)

Z_pca = PCA(n_components=2).fit_transform(X)                   # (T, 2)
Z_ica = FastICA(n_components=2, random_state=0).fit_transform(X)
print("관측 센서    |상관|:", match(X))
print("PCA 성분     |상관|:", match(Z_pca))
print("ICA 성분     |상관|:", match(Z_ica))
```

```output
관측 센서    |상관|: [0.74 0.94]
PCA 성분     |상관|: [0.87 0.83]
ICA 성분     |상관|: [1. 1.]
```

PCA 성분은 여전히 두 원천이 섞여 있다(상관 0.87, 0.83). PCA는 분산이 큰 방향을 찾았을 뿐이고, 그 방향이 원천과 일치할 이유가 없다. ICA 성분은 원천과의 상관이 1.00으로 두 원천을 되찾았다. 다만 이것은 가정(선형 혼합, 독립, 비가우스)이 정확히 맞는 합성 예다. 실제 데이터에서 ICA 성분이 "물리적 원천"이라는 주장은 가정을 따로 확인해야 한다.

::quiz{id="pca-uncorrelated-not-independent"}

**연구에서의 의미.** 세 방법은 서로 다른 질문에 답한다. PCA는 "변동이 어디에 몰려 있는가", kernel PCA는 "비선형 변환 뒤 변동이 어디에 몰려 있는가", ICA는 "독립인 원천으로 분해할 수 있는가"다. 뇌파(EEG)에서 눈 깜박임 성분을 분리하는 것처럼 혼합 모델이 물리적으로 그럴듯할 때 ICA가 강하다. 차원 축소의 결과를 다음 모델의 입력으로 쓸 때는, 7.6장에서 본 것처럼 분산이 큰 방향이 예측에 유용한 방향이라는 보장이 없다는 점을 기억한다.

## 군집

**왜 필요한가.** 군집은 비슷한 샘플을 묶는다. 재료의 미세구조 유형, 기상 패턴의 유형, 세포의 유형처럼 범주를 미리 모르는 상태에서 범주의 후보를 찾을 때 쓴다. 대표적인 방법 네 가지는 "비슷하다"와 "무리"를 다르게 정의한다.

| 방법 | 무엇을 최적화하는가 | 잘 찾는 모양 | 정해야 하는 것 |
|---|---|---|---|
| k-means | 군집 내 제곱합 $\sum_i \lVert\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\rVert^2$ | 크기가 비슷한 둥근 무리 | $k$, 초기값 |
| GMM | 가우스 혼합의 likelihood (EM) | 타원형, 크기가 다른 무리 | 성분 수, 공분산 형태 |
| 계층적 군집 | 가까운 무리부터 병합 (linkage 기준) | linkage에 따라 다름 | linkage, 자를 높이 |
| DBSCAN | 밀도로 연결된 영역 | 임의 모양, 잡음 점 분리 | 반경 `eps`, 최소 이웃 수 |

**k-means**는 각 점을 가장 가까운 중심에 배정하고, 중심을 배정된 점들의 평균으로 옮기는 두 단계를 반복한다(Lloyd 알고리즘). 두 단계 모두 군집 내 제곱합을 늘리지 않으므로 반복은 수렴하지만, 전역 최솟값이 아니라 초기값에 따른 국소해에 멈출 수 있다. 그래서 라이브러리는 여러 초기값(`n_init`)으로 돌려 가장 좋은 것을 고른다. **GMM**(Gaussian mixture model)은 데이터가 $k$개 가우스 분포의 혼합에서 나왔다고 보고 EM 알고리즘으로 likelihood를 최대화한다. 각 점에 확률적인 소속(soft assignment)을 주고, 공분산을 군집마다 다르게 둘 수 있다. **계층적 군집**은 모든 점을 따로 두고 시작해 가장 가까운 두 무리를 반복해서 합친다. 무리 사이 거리를 최근접 쌍(single linkage)으로 재면 길게 이어진 사슬 모양도 한 군집이 되고, Ward 기준으로 재면 k-means와 비슷한 둥근 군집이 된다. **DBSCAN**은 반경 `eps` 안에 이웃이 충분히 많은 점을 핵심점으로 보고, 핵심점끼리 이어진 영역을 군집으로 삼는다(Ester 외, 1996). 어느 영역에도 속하지 않는 점은 잡음으로 남긴다. 군집 수를 정하지 않아도 되는 대신 `eps`가 사실상 같은 역할을 한다.

**거리와 스케일.** 네 방법은 모두 거리(또는 그에 준하는 것)에 의존한다. 그래서 특징의 단위가 군집 결과를 정한다. 온도로 구별되는 세 집단에, 집단과 무관하게 흔들리는 압력을 함께 넣고 k-means를 돌린다. 같은 데이터를 압력 단위만 Pa와 kPa로 바꾸고, 표준화한 경우와 비교한다. 참 집단과의 일치도는 ARI(adjusted Rand index)로 잰다. 1이면 완전히 일치하고, 무작위 배정이면 0 근처다.

```python {exec="browser" session="unsup"}
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(3)
label = np.repeat([0, 1, 2], 100)                              # 참 집단 (N,)
temp_k = np.array([300.0, 310.0, 320.0])[label] + rng.normal(0, 2, 300)    # 온도 [K]
press_pa = 1e5 + rng.normal(0, 3000, 300)                      # 압력 [Pa]: 집단과 무관한 변동
F = np.c_[temp_k, press_pa]                                    # (N, 2)

for name, Fx in [("K, Pa", F), ("K, kPa", F / [1, 1000]), ("표준화", StandardScaler().fit_transform(F))]:
    pred = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Fx)   # (N,)
    print(f"{name:7s} ARI = {adjusted_rand_score(label, pred):.2f}")
```

```output
K, Pa   ARI = 0.00
K, kPa  ARI = 0.99
표준화     ARI = 0.37
```

압력을 Pa로 쓰면 압력의 표준편차(3000)가 온도 차이(10–20)를 압도해, k-means는 압력을 세 구간으로 자르고 참 집단과 전혀 맞지 않는다(ARI 0.00). kPa로 바꾸기만 해도 거의 완벽하게 맞는다(0.99). 표준화는 두 특징에 같은 무게를 주므로, 집단과 무관한 압력의 변동이 온도와 같은 비중으로 들어와 중간 결과(0.37)가 된다. **표준화는 만능이 아니다.** 군집에 어떤 특징이 얼마나 중요해야 하는지는 데이터가 아니라 연구 질문이 정한다. 단위를 바꿔도, 특징을 빼거나 넣어도 결과가 유지되는지 확인하고, 어떤 스케일을 썼는지 보고한다.

::quiz{id="units-change-clusters"}

:::note[분야 예시]
기상학에서는 매일의 대기압 패턴(예: 500 hPa 지위고도장)을 PCA로 줄인 뒤 k-means로 묶어 몇 개의 **기상 체제**(weather regime)로 분류하는 방법이 오래 쓰여 왔다. 군집 수는 분석가가 정하고, 몇 개로 나누느냐에 따라 체제의 모양이 달라진다. 그래서 군집이 실제로 반복되는 순환 패턴인지, 연속적인 변동을 임의로 자른 것인지를 놓고 군집 수의 안정성과 물리적 해석을 함께 검토한다.
:::

**군집 수 고르기.** k-means의 군집 내 제곱합은 $k$를 늘리면 항상 줄어들어 $k = N$에서 0이 된다. 그래서 제곱합 자체로는 $k$를 고를 수 없고, 다음 같은 기준을 쓴다.

- **elbow**: 제곱합 곡선이 꺾이는 지점. 꺾임이 분명하지 않은 경우가 많아 주관적이다.
- **silhouette**: 점 $i$에서 같은 군집 점들까지의 평균 거리 $a_i$와, 가장 가까운 다른 군집까지의 평균 거리 $b_i$로 $s_i = (b_i - a_i)/\max(a_i, b_i)$를 계산해 평균한다(Rousseeuw, 1987). 1에 가까우면 군집이 촘촘하고 서로 떨어져 있다.
- **BIC**: GMM처럼 확률모형이면 $\mathrm{BIC} = -2\log\hat{L} + p\log N$이 가장 작은 성분 수를 고른다. $p$는 파라미터 수다.

이 기준들이 "구조가 없다"고 말할 수 있는지 확인한다. 둥근 덩어리 세 개와, 같은 넓이의 균일 분포에서 각각 기준을 계산한다.

```python {exec="browser" session="unsup"}
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

data = {"덩어리 3개": np.r_[rng.normal([0, 0], 0.5, (100, 2)),
                            rng.normal([3, 0], 0.5, (100, 2)),
                            rng.normal([0, 3], 0.5, (100, 2))],
        "균일 분포": rng.uniform(0, 4, (300, 2))}              # 각각 (300, 2)
for name, D in data.items():
    sil = [silhouette_score(D, KMeans(k, n_init=10, random_state=0).fit_predict(D)) for k in range(2, 7)]
    bic = [GaussianMixture(k, random_state=0).fit(D).bic(D) for k in range(1, 7)]
    print(f"{name}: silhouette k=2..6 {np.round(sil, 2)}  최대 k={2 + int(np.argmax(sil))}"
          f" | GMM BIC 최소 k={1 + int(np.argmin(bic))}")
```

```output
덩어리 3개: silhouette k=2..6 [0.55 0.7  0.58 0.44 0.34]  최대 k=3 | GMM BIC 최소 k=3
균일 분포: silhouette k=2..6 [0.37 0.4  0.43 0.4  0.4 ]  최대 k=4 | GMM BIC 최소 k=3
```

덩어리 세 개에서는 두 기준이 모두 $k = 3$을 고르고, silhouette이 0.70으로 다른 $k$보다 뚜렷이 높다. 균일 분포에서도 두 기준은 답을 낸다. silhouette은 $k = 4$에서 0.43으로 "적당한" 값이고, BIC는 가우스 성분 3개를 고른다. 균일 분포에는 무리가 없지만, 정사각형 밀도를 가우스 하나보다 가우스 셋으로 더 잘 근사할 수 있기 때문이다. **BIC가 고르는 것은 밀도를 근사하는 데 필요한 성분 수이지, 실제로 떨어진 무리의 수가 아니다.** silhouette 값 0.4–0.5는 구조가 있다는 증거가 되지 못하고, 곡선이 평평하다는 사실이 더 중요한 정보다.

::quiz{id="silhouette-on-uniform"}

$k$를 고르는 더 믿을 만한 방법은 기준 하나의 최댓값이 아니라 **안정성**을 보는 것이다. 데이터를 부트스트랩으로 다시 뽑거나 절반으로 나누어 군집을 반복하고, 같은 점들이 계속 같은 군집에 묶이는지(ARI)를 잰다. 구조가 없는 데이터에서 $k$개로 자른 경계는 재표본마다 크게 움직인다.

## 밀도 추정과 이상 탐지

**왜 필요한가.** 이상 탐지(anomaly detection)는 "정상"에서 벗어난 샘플을 찾는다. 센서 고장, 불량 셀, 측정 오류, 드물지만 흥미로운 현상을 찾을 때 쓴다. 이상은 보통 드물고 종류가 다양해서 label을 모으기 어렵다. 그래서 정상 데이터의 분포를 배운 뒤, 그 분포에서 일어나기 어려운 샘플에 높은 점수를 주는 방식이 기본이다.

방법은 "정상"을 표현하는 방식에 따라 나뉜다.

- **밀도 추정.** 커널 밀도 추정(KDE, kernel density estimation)은 각 학습 점에 폭 $h$인 커널을 얹어 더한다: $\hat{p}(\mathbf{x}) = \frac{1}{N}\sum_i K_h(\mathbf{x} - \mathbf{x}_i)$. 밀도 $\hat{p}$가 낮은 점이 이상이다. 폭 $h$(bandwidth)가 결과를 정하고, 차원이 높아지면 필요한 샘플이 급격히 늘어 실용성이 떨어진다.
- **분리 기반.** isolation forest는 무작위 특징과 무작위 분할점으로 데이터를 반복해서 자르는 트리를 여러 개 만든다(Liu 외, 2008). 드물고 동떨어진 점은 적은 분할로 혼자 남으므로, 트리에서의 평균 경로 길이가 짧을수록 이상 점수가 높다. 밀도를 직접 추정하지 않아 고차원에서도 비교적 빠르다.
- **경계 기반.** one-class SVM은 정상 데이터를 감싸는 경계를 커널 공간에서 학습한다. 경계 밖이 이상이다.
- **재구성 기반.** PCA나 autoencoder로 정상 데이터를 저차원으로 압축했다 되돌리고, 재구성 오차 $\lVert\mathbf{x} - \hat{\mathbf{x}}\rVert^2$가 큰 점을 이상으로 본다. 정상 데이터가 저차원 구조(상관) 위에 놓여 있을 때, 그 구조를 깨는 이상을 잘 찾는다.

흔한 실수는 특징마다 따로 임계값을 거는 것이다. 두 센서가 강하게 상관된(상관계수 0.95) 정상 데이터에, 각 센서 값은 정상 범위 안이지만 상관을 거스르는 점 10개를 섞는다. 축별 z-점수, PCA 재구성 오차, isolation forest의 탐지 성능을 ROC-AUC로 비교한다. 모든 방법은 정상 데이터로만 적합한다.

```python {exec="browser" session="unsup"}
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

cov = np.array([[1.0, 0.95], [0.95, 1.0]])
normal = rng.multivariate_normal([0, 0], cov, 500)             # (500, 2) 강하게 상관된 정상 데이터
u = rng.uniform(0.7, 1.5, 10) * rng.choice([-1, 1], 10)
odd = np.c_[u, -u]                                             # (10, 2) 상관을 거스르는 점들
Xa = np.r_[normal, odd]                                        # (510, 2)
is_odd = np.r_[np.zeros(500), np.ones(10)]

mu, sd = normal.mean(0), normal.std(0)
score_z = np.abs((Xa - mu) / sd).max(axis=1)                   # 축별 z-점수의 최댓값
pc = PCA(1).fit(normal)                                        # 정상 데이터로만 적합
score_pca = np.sum((Xa - pc.inverse_transform(pc.transform(Xa))) ** 2, axis=1)
score_if = -IsolationForest(random_state=0).fit(normal).score_samples(Xa)
for name, s in [("축별 z-점수", score_z), ("PCA 재구성 오차", score_pca), ("isolation forest", score_if)]:
    print(f"{name:16s} AUC = {roc_auc_score(is_odd, s):.3f}")
```

```output
축별 z-점수          AUC = 0.712
PCA 재구성 오차       AUC = 1.000
isolation forest AUC = 0.975
```

축별 z-점수는 이상 점을 정상 점과 잘 구별하지 못한다(AUC 0.712). 각 좌표가 $\pm 1.5$ 안에 있어 어느 센서 하나만 보면 평범하기 때문이다. PCA 재구성 오차는 정상 데이터의 주축(대각선)에서 벗어난 거리를 재므로 10개를 모두 가장 높은 점수로 올린다(AUC 1.000). isolation forest도 높은 성능을 낸다(0.975). 같은 원리로, 상관된 가우스 데이터에서는 Mahalanobis 거리 $\sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$가 공분산 구조를 반영한 이상 점수다.

::quiz{id="mahalanobis-anticorrelated"}

**연구에서의 의미.** 이상 탐지의 성능은 "어떤 이상을 찾고 싶은가"에 달려 있다. 재구성 기반 방법은 상관 구조를 깨는 이상에 강하지만, 주축 방향으로 멀리 나간 점(두 센서가 함께 크게 오른 점)은 재구성이 잘 되어 놓친다. 그래서 재구성 오차와 주성분 점수 자체의 크기를 함께 보는 경우가 많다. 그리고 이상 탐지기의 임계값을 정하거나 성능을 보고하려면 결국 **label이 붙은 이상 사례**가 몇 개라도 필요하다. 위 예의 AUC도 이상 label이 있었기에 계산할 수 있었다.

:::note[분야 예시]
배터리 제조에서는 셀마다 화성(formation) 공정의 전압·용량 곡선이 기록된다. 대부분의 셀은 거의 같은 곡선을 그리므로, 정상 셀들의 곡선을 PCA로 요약하고 재구성 오차가 큰 셀을 검사 대상으로 올리는 방식이 가능하다. 이때 "정상" 학습 데이터에 불량 셀이 섞여 있으면 그 패턴까지 정상으로 배운다. 학습 데이터의 구성과 오염 가능성을 함께 기록한다.
:::

## 시각화

**왜 필요한가.** 고차원 데이터를 2차원 그림으로 보고 싶을 때, PCA의 처음 두 성분은 분산이 큰 방향만 보여 주므로 비선형 구조나 작은 무리를 놓친다. t-SNE와 UMAP은 **이웃 관계**를 보존하는 2차원 좌표를 찾는 비선형 방법이다.

**t-SNE.** 원공간에서 점 $i$를 중심으로 한 가우스 커널로 "점 $j$가 $i$의 이웃일 확률" $p_{j|i}$를 정의하고, 2차원 좌표에서는 꼬리가 두꺼운 Student $t$ 분포로 같은 확률 $q_{ij}$를 정의한 뒤, 두 분포의 KL divergence를 최소화하도록 좌표를 옮긴다(van der Maaten, Hinton, 2008). 점마다 커널의 폭은 **perplexity**로 정한다. perplexity는 대략 "각 점이 고려하는 유효 이웃 수"이고, 원 논문은 SNE에서 전형적인 값이 5–50이라고 소개한다. 결과가 이 값에 따라 달라지므로 여러 값으로 그려 비교한다. 커널 폭을 점마다 이웃 수에 맞춰 바꾸기 때문에, 촘촘한 군집과 퍼진 군집이 2차원에서 비슷한 크기로 그려진다. 2차원의 두꺼운 꼬리는 서로 다른 군집을 밀어내 섬처럼 떨어뜨린다.

**UMAP.** 원공간에서 $k$-최근접 이웃 그래프를 만들고, 그 그래프 구조를 보존하는 저차원 좌표를 최적화한다(McInnes 외, 2018). `n_neighbors`는 국소 구조와 전역 구조 사이의 균형을, `min_dist`는 2차원에서 점들이 얼마나 촘촘히 뭉칠 수 있는지를 정한다. t-SNE보다 빠르고 큰 데이터에 쓰기 쉬워 널리 쓰이지만, 해석의 한계는 t-SNE와 같은 종류다.

두 방법이 보존하려는 것은 **이웃이 누구인가**다. 군집의 크기와 군집 사이의 거리는 최적화 목표에 들어 있지 않다. 크기가 10배 다른 두 군집 A, B와 멀리 떨어진 군집 C를 5차원에 만들고, 원공간과 t-SNE 좌표에서 군집의 퍼짐 비와 군집 사이 거리 비를 비교한다.

```python {exec="browser" session="unsup"}
from sklearn.manifold import TSNE

blobs = np.r_[rng.normal(0, 0.2, (100, 5)),                    # 촘촘한 군집 A
              rng.normal(0, 2.0, (100, 5)) + [6, 0, 0, 0, 0],  # 10배 퍼진 군집 B
              rng.normal(0, 0.2, (100, 5)) + [40, 0, 0, 0, 0]] # 멀리 떨어진 군집 C, (300, 5)
E = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(blobs)

def spread(Z, i):
    return np.linalg.norm(Z[i] - Z[i].mean(0), axis=1).mean()

def gap(Z, i, j):
    return np.linalg.norm(Z[i].mean(0) - Z[j].mean(0))

a, b, c = np.arange(0, 100), np.arange(100, 200), np.arange(200, 300)
for name, Z in [("원공간", blobs), ("t-SNE", E)]:
    print(f"{name:5s}  퍼짐 B/A = {spread(Z, b) / spread(Z, a):3.0f}배   거리 AC/AB = {gap(Z, a, c) / gap(Z, a, b):3.0f}배")
```

```output
원공간    퍼짐 B/A =  10배   거리 AC/AB =   7배
t-SNE  퍼짐 B/A =   1배   거리 AC/AB =   3배
```

원공간에서 B는 A보다 10배 퍼져 있지만 t-SNE 그림에서는 두 군집이 거의 같은 크기다. 원공간에서 C는 B보다 7배 멀리 있지만 그림에서는 3배 정도로 줄었다. 그림만 보고 "B는 A만큼 균질하다", "C는 B보다 조금 더 다르다"고 읽으면 둘 다 틀린다. Wattenberg 외(2016)는 같은 종류의 예를 여러 개 들어, perplexity에 따라 그림이 크게 바뀌고 구조가 없는 잡음에서도 무리처럼 보이는 모양이 생길 수 있음을 보였다.

::quiz{id="tsne-what-to-read"}

t-SNE·UMAP 그림에서 비교적 믿을 수 있는 것과 없는 것을 정리하면 다음과 같다.

| 읽어도 되는 것 (확인 후) | 읽으면 안 되는 것 |
|---|---|
| 가까이 붙은 점들이 원공간에서도 이웃이다 | 군집의 크기·밀도 |
| 어떤 label이 한 섬에 모여 있다 (label을 색으로 칠해 확인) | 군집 사이의 거리와 배치 |
| 여러 perplexity·seed에서 반복되는 분리 | 한 설정에서만 보이는 분리 |

그림은 가설을 만드는 데 쓰고, 가설은 원공간의 거리나 별도의 정량 지표로 확인한다. 2차원 좌표를 군집화나 예측의 입력으로 쓰는 것도 피한다. t-SNE에는 새 점을 변환하는 함수가 없고(전체를 다시 최적화해야 한다), 2차원으로 줄이는 과정에서 거리 정보가 왜곡되기 때문이다.

## 군집의 의미

**통계적 군집과 물리적 집단은 다르다.** 군집 알고리즘이 찾는 것은 선택한 특징과 거리에서 서로 가까운 점들의 무리다. 그것이 실제로 서로 다른 물리적 집단(다른 상, 다른 세포 유형, 다른 고장 모드)인지는 별개의 질문이다. 둘이 어긋나는 대표적인 경우는 세 가지다.

- **연속 변동을 자른 것.** 온도에 따라 연속적으로 변하는 물성을 k-means로 셋으로 나누면 "저온·중온·고온 군집"이 생기지만, 집단이 셋인 것이 아니라 연속체를 셋으로 자른 것이다. 군집 경계에서 밀도가 실제로 낮아지는지 확인한다.
- **측정 조건을 찾은 것.** 장비·배치(batch)·측정 날짜에 따른 체계적 차이가 가장 큰 변동이면, 군집은 시료가 아니라 측정 조건을 반영한다. 군집과 메타데이터(장비, 날짜, 작업자)의 대응을 먼저 본다.
- **전처리가 만든 것.** 정규화, 특징 선택, 차원 축소의 선택이 군집을 바꾼다. 앞의 단위 예에서 Pa와 kPa는 완전히 다른 군집을 냈다.

그래서 군집은 **독립적인 근거**로 검증한다. 군집에 쓰지 않은 변수(조성, 공정 조건, 다른 측정)가 군집에 따라 달라지는가, 재표본과 다른 알고리즘에서 같은 군집이 다시 나오는가, 새로 모은 데이터에서도 같은 구조가 보이는가, 그리고 도메인 지식으로 군집의 차이를 설명할 수 있는가를 차례로 묻는다.

:::note[분야 예시]
재료 과학에서 고처리량 계산이나 실험으로 얻은 수천 개 조성의 X선 회절 패턴을 군집으로 묶어 상(phase) 영역을 찾는 연구가 있다. 회절 패턴이 비슷한 조성이 한 군집이 되지만, 두 상이 섞인 영역은 연속적인 혼합 비율을 가지므로 군집 경계가 실제 상 경계와 어긋날 수 있다. 그래서 군집 결과를 결정 구조 데이터베이스의 참조 패턴과 대조하고, 상 경계 근처는 추가 측정으로 확인한다.
:::

## 실습

2차원 embedding에서 보이는 분리가 실제 하위 과제의 일반화와 연결되는지 검증한다. 코드는 `labs/classical-ml/unsupervised-learning/embedding_check.py`다. 상세 목차는 UMAP을 제안했지만 실습 환경에 umap-learn이 없어 같은 계열의 t-SNE를 쓴다(설치했다면 한 줄을 바꿔 같은 검증을 반복할 수 있다). 합성 데이터에서 측정 장비 3대가 특징 20개 중 18개에 큰 오프셋을 더하고, 예측하려는 label은 오프셋이 없는 특징 두 개의 약한 조합에만 들어 있다. 질문은 셋이다. 그림의 군집은 무엇에 대응하는가, 그 좌표로 label을 예측하면 새 장비에 일반화되는가, 장비 효과를 지우면 그림이 어떻게 바뀌는가. 집필 시점의 실행 결과는 다음과 같다.

```output
샘플 600개, 특징 20개, 장비 3대, label 양성 비율 0.50

1. 그림의 군집은 무엇에 대응하는가 (silhouette, 2차원 embedding 기준)
   장비로 묶을 때 +0.84   label로 묶을 때 +0.01

2. label 예측 정확도 (5-fold 층화 CV / 장비 단위 leave-one-group-out)
   kNN, t-SNE 2차원 좌표         층화 CV 0.780   새 장비 0.540
   kNN, 원래 특징 20개            층화 CV 0.800   새 장비 0.630
   로지스틱 회귀, 원래 특징 20개        층화 CV 0.815   새 장비 0.725
   (다수 클래스만 예측할 때 0.502)

3. 장비 오프셋을 장비별 중심화로 지운 뒤 다시 그린 embedding
   장비로 묶을 때 -0.01   label로 묶을 때 +0.01
```

결과를 읽는 법은 다음과 같다.

- **그림의 뚜렷한 섬 세 개는 장비다.** 장비 기준 silhouette이 0.84로 매우 높고 label 기준은 0.01이다. 그림만 보고 "데이터에 세 유형이 있다"고 해석했다면 측정 조건을 시료의 성질로 착각한 것이다.
- **같은 장비 안에서는 그림 좌표로도 label을 꽤 맞힌다(0.780).** 그러나 처음 보는 장비로 일반화하면 0.540으로 거의 우연 수준이다. 장비마다 섬이 따로 있어 좌표의 의미가 장비에 묶여 있기 때문이다. 원래 특징으로 학습한 로지스틱 회귀는 새 장비에서 0.725를 유지한다. 시각적 분리의 선명도와 하위 과제의 일반화는 별개다. 그리고 평가 분할이 "같은 장비"인지 "새 장비"인지가 결론을 바꾼다(15.4장, 16.6장).
- **장비 효과를 지워도 label은 그림에 드러나지 않는다.** label 정보가 특징 20개 중 2개의 약한 조합에만 있으므로, 이웃 관계를 보존하는 embedding의 주된 구조가 되지 못한다. 그림에 보이지 않는다고 예측 가능한 신호가 없는 것도 아니다.

이어서 연습문제를 푼다. k-means의 Lloyd 반복을 직접 구현하고, 초기값에 따라 다른 국소해에 멈추는 것을 확인한다.

::exercise{id="kmeans-lloyd"}

## 실패 사례

- **t-SNE·UMAP 그림의 군집 크기와 군집 사이 거리를 해석하는 것.** 이 장의 예에서 10배 퍼진 군집이 같은 크기로, 7배 먼 군집이 3배 거리로 그려졌다. 그림은 가설을 만드는 데만 쓰고, 여러 perplexity·seed에서 반복되는지와 원공간의 정량 지표로 확인한다.
- **알고리즘이 돌려준 $k$개 군집을 집단이 $k$개라는 증거로 쓰는 것.** 균일 분포에서도 silhouette은 0.4대의 값을, GMM의 BIC는 성분 3개를 골랐다. 기준 곡선의 모양과 재표본 안정성을 함께 보고, 군집에 쓰지 않은 변수로 검증한다.
- **단위나 전처리를 바꾸면 사라지는 군집을 보고하는 것.** Pa를 kPa로 바꾸자 ARI가 0.00에서 0.99로 바뀌었다. 스케일 선택을 기록하고, 선택을 바꿔도 결론이 유지되는지 확인한다.
- **군집과 측정 조건의 대응을 확인하지 않는 것.** 실습에서 가장 선명한 구조는 장비였다. 군집을 해석하기 전에 장비·날짜·배치 같은 메타데이터와의 교차표를 먼저 만든다.
- **이상 탐지기를 오염된 "정상" 데이터로 학습하는 것.** 학습 데이터에 섞인 이상은 정상으로 학습된다. 정상 데이터의 출처와 선별 기준을 기록한다.

## 통과 기준

- 자신의 데이터에서 군집 수를 두 가지 이상의 기준과 재표본 안정성으로 고르고, 군집에 쓰지 않은 변수나 메타데이터로 군집의 의미를 검증한 결과를 제시한다.
- t-SNE 또는 UMAP 그림에서 주장하는 구조를 원공간의 정량 지표(silhouette, 하위 과제 성능 등)로 확인하고, 그림에서 읽으면 안 되는 것을 구분해 설명한다.
- 연습문제 `kmeans-lloyd`의 테스트를 통과한다.

## 참고 문헌

- Rousseeuw (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. — silhouette 지표
- Ester, Kriegel, Sander, Xu (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD-96*, 226–231. — DBSCAN
- Schölkopf, Smola, Müller (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5), 1299–1319. — kernel PCA
- Hyvärinen, Oja (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430. — ICA와 비가우스성 기준, FastICA
- Liu, Ting, Zhou (2008). Isolation forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM 2008)*, 413–422. — isolation forest
- van der Maaten, Hinton (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605. — t-SNE
- Wattenberg, Viégas, Johnson (2016). How to use t-SNE effectively. *Distill*. doi:10.23915/distill.00002 — perplexity에 따른 변화, 군집 크기·거리 해석의 함정
- McInnes, Healy, Melville (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426. — UMAP
