---
id: chemistry-and-materials/property-prediction
title: "물성 예측"
title_en: "Property Prediction"
tier: core
status: draft
prerequisites: [chemistry-and-materials/molecular-representations, reliability/uncertainty-quantification]
---

# 물성 예측

분자나 재료를 입력으로 받아 용해도, 흡수 파장, 형성 에너지, 독성 같은 수치를 예측하는 것이 물성 예측(property prediction)이다. 회귀 문제이므로 16장과 17장의 모델이 그대로 쓰이고, 이 장은 그 모델을 다시 가르치지 않는다. 대신 화학·재료에서 물성 예측이 **왜 평가가 어려운가**를 다룬다. 핵심은 하나다. 모델을 쓰는 상황은 거의 언제나 "학습에 있던 것과 다른 화학 구조"를 예측하는 일인데, 무작위로 나눈 시험 데이터는 이 상황을 재현하지 못한다.

15.4장에서 분할과 누수의 일반 원리를, 21장에서 불확실성을 다뤘다. 이 장은 그 원리를 분자 데이터에 적용한다. scaffold 분할이 왜 무작위 분할보다 정직한 수치를 주는지, 측정 잡음이 어떻게 오차 바닥을 정하는지, 적용 영역(applicability domain)을 어떻게 수치로 보고하는지를 합성 데이터로 직접 확인한다. 합성 데이터를 쓰는 이유는 참값과 데이터 생성 규칙을 알아야 "모델이 틀린 것"과 "평가가 속은 것"을 구분할 수 있기 때문이다.

## 학습 목표

- 분자·재료 물성 예측 모델을 학습하고 평가한다
- 화학 공간에서의 일반화를 검증한다

## 데이터

**왜 필요한가.** 모델의 성능 상한은 레이블의 품질이 정한다. 레이블이 잡음투성이면 완벽한 모델도 그 잡음만큼 틀린다. 그래서 데이터의 출처와 잡음을 먼저 이해해야 한다.

**두 종류의 레이블.** 화학·재료 물성 데이터는 크게 두 출처에서 온다.

- **실험 데이터**: 실제로 측정한 값. 참값에 가깝지만 측정 조건(온도, 용매, 장비, 실험실)에 따라 흩어지고, 서로 다른 조건의 값을 한 데이터셋에 섞으면 조건이 숨은 변수가 된다. 데이터 양도 대개 적다.
- **계산 데이터**: 밀도범함수 이론(DFT) 같은 양자화학 계산으로 얻은 값. 많이 만들 수 있고 조건이 일관되지만, 계산 수준(함수, 기저 집합)마다 체계적으로 다른 값을 내고, 실험 참값과도 체계적 차이가 있다. 계산 수준을 섞으면 그 차이가 잡음처럼 들어온다(28.5장).

**측정 잡음이 정하는 오차 바닥.** 레이블에 표준편차 $\sigma$의 잡음이 있으면, 참값을 정확히 아는 모델조차 잡음 때문에 틀린다. 정규 잡음에서 평균절대오차(MAE)의 하한은 $\sigma\sqrt{2/\pi}$이고, 결정계수 $R^2$의 상한은 1보다 작다. 이 바닥 아래로 내려간 성능은 모델이 좋은 것이 아니라 시험 레이블의 잡음에 과적합한 것이다.

```python {exec="browser" session="pp"}
import numpy as np

rng = np.random.default_rng(0)
y_true = rng.normal(-3.0, 1.5, size=20000)                 # 참 물성(예: log 용해도) (N,)
for sigma in [0.2, 0.5, 0.8]:
    y_lab = y_true + sigma * rng.normal(size=y_true.size)    # 실험 레이블 = 참값 + 측정 잡음
    mae_perfect = np.abs(y_true - y_lab).mean()              # 참값을 정확히 아는 모델의 MAE
    r2_ceiling = 1 - ((y_lab - y_true) ** 2).mean() / y_lab.var()
    print(f"잡음 σ = {sigma:.1f}  완벽한 모델의 MAE = {mae_perfect:.3f}"
          f" (σ√(2/π) = {sigma * np.sqrt(2 / np.pi):.3f})  R² 상한 = {r2_ceiling:.3f}")
```

```output
잡음 σ = 0.2  완벽한 모델의 MAE = 0.160 (σ√(2/π) = 0.160)  R² 상한 = 0.982
잡음 σ = 0.5  완벽한 모델의 MAE = 0.398 (σ√(2/π) = 0.399)  R² 상한 = 0.900
잡음 σ = 0.8  완벽한 모델의 MAE = 0.637 (σ√(2/π) = 0.638)  R² 상한 = 0.778
```

실험 용해도(log 단위)의 재현 표준편차는 흔히 0.5–0.7 수준으로 보고된다. 그런 데이터에서 MAE 0.3을 주장하는 모델은 측정 잡음의 바닥(약 0.4–0.56)보다 낮으므로, 수치 자체를 의심해야 한다. 중복 측정의 흩어짐으로 $\sigma$를 추정해 두면 "더 줄일 수 있는 오차"와 "레이블 잡음"을 나눌 수 있다.

::quiz{id="noise-floor"}

## 분할

**왜 필요한가.** 모델을 쓰는 목적은 아직 만들지 않았거나 측정하지 않은 분자의 물성을 예측하는 것이다. 그런 분자는 학습 데이터와 구조가 다르다. 평가도 이 상황을 재현해야 한다. 무작위 분할은 그러지 못한다. 분자 데이터는 비슷한 구조가 무리 지어 있어서, 무작위로 뽑은 test 분자는 train에 거의 같은 친척을 두고 있을 때가 많다.

**골격.** 15.4장에서 본 group split을 분자에 적용한 것이 **scaffold 분할**이다. 분자의 골격(scaffold)은 곁사슬을 떼고 고리와 이를 잇는 뼈대만 남긴 구조다(Bemis, Murcko, 1996). 같은 골격의 분자들은 구조와 물성이 비슷하므로, 골격을 통째로 한쪽에만 두면 모델은 "처음 보는 골격"에 대해 평가받는다. 이것이 신약·소재 탐색에서 "새로운 화학 구조로의 일반화"를 재는 표준 방법이다.

아래 코드는 골격 계열(family) 구조가 있는 합성 화학 공간을 만든다. 골격 계열마다 공유하는 지문 비트가 있고, 물성은 계열 효과, 골격 고유 효과, 치환기 비트의 기여, 잡음의 합이다. 같은 골격 분자끼리는 Tanimoto 유사도가 높고 다른 골격과는 낮게 나오는지 확인한다.

```python {exec="browser" session="pp"}
n_bits, n_fam, n_scaf, n_mol = 256, 12, 150, 1200
fam_bits = [rng.choice(n_bits, 10, replace=False) for _ in range(n_fam)]   # 골격 계열마다 공유 비트
fam_of = rng.choice(n_fam, n_scaf, p=np.arange(n_fam, 0, -1) / np.arange(n_fam, 0, -1).sum())
scaf_bits = [np.union1d(fam_bits[fam_of[k]], rng.choice(n_bits, 6, replace=False)) for k in range(n_scaf)]
sub_bits = [rng.choice(n_bits, 4, replace=False) for _ in range(50)]       # 치환기 50종
w = rng.normal(0, 0.15, n_bits)                                            # 비트마다 기여
fam_eff = rng.normal(0, 1.0, n_fam)                                        # 계열 효과
scaf_eff = rng.normal(0, 0.3, n_scaf)                                      # 골격 고유 효과
p_scaf = 1 / np.arange(1, n_scaf + 1) ** 0.8                               # 골격 크기가 치우친 분포
scaf = rng.choice(n_scaf, n_mol, p=p_scaf / p_scaf.sum())                  # (N,)
X = np.zeros((n_mol, n_bits), dtype=bool)                                  # 지문 (N, n_bits)
for i, s in enumerate(scaf):
    X[i, scaf_bits[s]] = True
    for t in rng.choice(50, rng.integers(1, 5), replace=False):
        X[i, sub_bits[t]] = True
y = X @ w + fam_eff[fam_of[scaf]] + scaf_eff[scaf] + 0.2 * rng.normal(size=n_mol)   # 물성 (N,)

def tanimoto(A, B):                                                        # (n, d), (m, d) → (n, m)
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter)

S = tanimoto(X, X)
same = scaf[:, None] == scaf[None, :]
off = ~np.eye(n_mol, dtype=bool)
print(f"분자 {n_mol}개, 골격 {len(np.unique(scaf))}개, 가장 큰 골격 3개의 크기 {np.sort(np.bincount(scaf))[::-1][:3]}")
print(f"평균 Tanimoto: 같은 골격 {S[same & off].mean():.3f}, 다른 골격 {S[~same].mean():.3f}")
```

```output
분자 1200개, 골격 143개, 가장 큰 골격 3개의 크기 [124  78  45]
평균 Tanimoto: 같은 골격 0.509, 다른 골격 0.086
```

**무작위 분할과 scaffold 분할의 차이.** 같은 데이터에 두 분할을 적용하고, Tanimoto 커널 Gaussian process로 예측한다. 함께 평균값만 내는 baseline(train 평균으로 모두 예측)과 test 분자의 최근접 train 유사도도 본다.

```python {exec="browser" session="pp"}
def fit_predict(tr, te, lam=0.1):                          # Tanimoto 커널 GP의 사후 평균·표준편차
    K = tanimoto(X[tr], X[tr]) + lam * np.eye(len(tr))     # (n_tr, n_tr)
    Ks = tanimoto(X[te], X[tr])                            # (n_te, n_tr)
    mu = y[tr].mean()
    mean = mu + Ks @ np.linalg.solve(K, y[tr] - mu)
    var = 1.0 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
    return mean, np.sqrt(np.clip(var, 1e-12, None)) * y[tr].std()

perm = rng.permutation(n_mol)
splits = {"무작위": (perm[240:], perm[:240])}
test_scafs, n_te = [], 0
for s in rng.permutation(n_scaf):                          # 골격 단위로 test를 채운다
    if n_te < 240:
        test_scafs.append(s)
        n_te += (scaf == s).sum()
is_te = np.isin(scaf, test_scafs)
splits["scaffold"] = (np.flatnonzero(~is_te), np.flatnonzero(is_te))
for name, (tr, te) in splits.items():
    mean, sd = fit_predict(tr, te)
    nn_sim = tanimoto(X[te], X[tr]).max(1)                 # test 분자의 최근접 train 유사도
    print(f"{name:8s} train {len(tr)} test {len(te)}  MAE {np.abs(mean - y[te]).mean():.3f}"
          f"  평균 예측 MAE {np.abs(y[tr].mean() - y[te]).mean():.3f}  최근접 유사도 중앙값 {np.median(nn_sim):.3f}")
```

```output
무작위      train 960 test 240  MAE 0.226  평균 예측 MAE 0.724  최근접 유사도 중앙값 0.645
scaffold train 943 test 257  MAE 0.464  평균 예측 MAE 0.719  최근접 유사도 중앙값 0.472
```

scaffold 분할의 MAE(0.464)가 무작위 분할(0.226)의 두 배다. 두 수치 모두 같은 모델, 같은 데이터에서 나왔다. 차이는 test 분자가 train과 얼마나 비슷한가뿐이다. 무작위 분할에서는 test 분자의 최근접 train 유사도 중앙값이 0.645로, 같은 골격의 친척이 train에 있다는 뜻이다. scaffold 분할에서는 0.472로 떨어진다. **무작위 분할의 0.226을 "새로운 분자에 대한 성능"으로 보고하면 실제 쓰임새에서의 오차를 절반으로 과소평가한다.** 어느 분할이든 평균 예측 baseline(약 0.72)보다는 나으므로 모델이 구조에서 무언가를 배우기는 했다.

::quiz{id="random-split-optimistic"}

**분할의 위계.** 분할은 재려는 일반화의 종류를 정한다. 주장에 맞는 분할을 고른다.

| 분할 | 재는 일반화 | 쓰는 때 |
|---|---|---|
| 무작위 | 같은 분포 안의 보간 | 데이터가 균질하고 조건이 같을 때(드물다) |
| scaffold | 새로운 골격 | 새 화학 계열로의 확장 |
| 군집(유사도) | 유사도 임계 밖의 구조 | scaffold가 없거나 재료일 때 |
| 시간 | 미래의 발견 | 연대가 있는 데이터, 전향적 예측 추정 |

시간 분할(time-split)은 데이터에 측정·등록 연대가 있을 때, 과거로 학습해 미래를 맞히는 전향적(prospective) 성능을 가장 비슷하게 추정한다고 보고된다(Sheridan, 2013). scaffold 분할조차 낙관적일 수 있는데, test 골격을 무작위로 고르면 train 골격과 가까운 것이 섞이기 때문이다. 유사도 기준으로 군집을 만들어 먼 군집을 통째로 빼면 더 엄격하다.

## 모델

**baseline부터.** 16장의 원칙대로 가장 단순한 모델부터 올린다. 지문이나 기술자에 Ridge·랜덤 포레스트·Gaussian process를 붙인 모델이 수백–수천 개 데이터에서 강한 baseline이다. 데이터가 수천 개를 넘고 물성이 국소 환경을 넘는 상호작용에 달려 있으면 그래프 신경망(18.6장)으로 간다. 물성이 3차원 배치에 직접 달려 있고 좌표를 믿을 수 있으면 등변 모델(24.8장)이다.

**사전학습 모델.** 레이블이 적은 과제에서는 레이블 없는 대규모 분자(또는 다른 물성)로 사전학습한 표현을 전이하는 방법이 쓰인다. 자기지도 사전학습(19.8장)과 전이의 함정(19.9장의 부정적 전이)이 그대로 적용된다. 사전학습 데이터가 평가 데이터와 겹치면 성능이 부풀려지므로, 사전학습 코퍼스와 test 분자의 중복을 확인한다(28.5장).

**핵심은 모델이 아니라 비교 조건.** 어떤 모델을 쓰든 비교는 같은 분할, 같은 특징, 같은 하이퍼파라미터 탐색 예산에서 한다. 벤치마크 논문은 양자역학·생물물리 데이터에서 학습 알고리즘보다 특징의 선택이 더 중요할 수 있다고 보고했다(Wu 외, 2018). 새 모델이 좋아 보일 때, 그 이득이 모델에서 온 것인지 특징이나 분할의 차이에서 온 것인지 먼저 가른다.

## 불확실성

**왜 필요한가.** 예측값 하나만으로는 그 예측을 믿어야 할지 알 수 없다. 모델을 후보 선별에 쓰려면 "이 예측은 믿을 만한가"를 수치로 함께 내야 한다. 21장에서 불확실성의 종류(aleatoric·epistemic)와 추정 방법(앙상블, 근사 베이지안)을 다뤘다. 여기서는 그것을 화학 공간의 외삽 진단에 쓴다.

**적용 영역.** 모델이 믿을 만한 입력의 범위를 **적용 영역**(applicability domain)이라 한다. 화학·독성학의 QSAR에서는 이것을 모델과 함께 보고하는 관행이 있다(21.6장). 가장 단순한 적용 영역 지표는 **최근접 train 유사도**다. test 분자가 가장 가까운 train 분자와 얼마나 비슷한가로, 유사도가 낮으면 외삽이다. 모델이 내는 예측 표준편차도 적용 영역 지표가 된다.

```python {exec="browser" session="pp"}
from scipy.stats import spearmanr

tr, te = splits["scaffold"]
mean, sd = fit_predict(tr, te)
err = np.abs(mean - y[te])
nn_sim = tanimoto(X[te], X[tr]).max(1)
bins = np.digitize(nn_sim, [0.4, 0.5])                    # 0: < 0.4, 1: 0.4–0.5, 2: ≥ 0.5
for b, label in enumerate(["< 0.4  ", "0.4–0.5", "≥ 0.5  "]):
    print(f"최근접 유사도 {label}  분자 {(bins == b).sum():3d}  MAE {err[bins == b].mean():.3f}")
print(f"Spearman(예측 표준편차, |오차|) = {spearmanr(sd, err)[0]:.3f}")
keep = sd <= np.quantile(sd, 0.5)
print(f"불확실성 하위 50%만 남기면 MAE {err[keep].mean():.3f} (전체 {err.mean():.3f})")
```

```output
최근접 유사도 < 0.4    분자  49  MAE 0.554
최근접 유사도 0.4–0.5  분자 119  MAE 0.481
최근접 유사도 ≥ 0.5    분자  89  MAE 0.392
Spearman(예측 표준편차, |오차|) = 0.389
불확실성 하위 50%만 남기면 MAE 0.366 (전체 0.464)
```

최근접 train 유사도가 낮은 분자일수록 오차가 크다(0.554 → 0.481 → 0.392). 예측 표준편차와 실제 오차의 Spearman 상관은 0.389로, 모델의 불확실성이 오차를 완벽히는 아니어도 어느 정도 가린다. 불확실성이 낮은 절반만 남기면 MAE가 0.464에서 0.366으로 떨어진다. **이것이 불확실성 기반 후보 선별이다.** 모든 예측을 쓰는 대신, 모델이 자신 있는 예측만 골라 실험 자원을 아낀다. 단, 이 선별은 포기한 절반에 대해서는 답을 주지 않는다는 비용을 치른다(21.6장의 선택적 예측).

:::note[분야 예시]
신약 개발의 용해도·독성 예측에서는 모델이 자신 없어 하는 분자를 먼저 실험으로 측정해 데이터에 추가하는 능동 학습(active learning, 25.3장)이 흔하다. 모델의 불확실성이 높은 분자가 "측정하면 가장 많이 배우는 분자"라는 가정이다. 재료 탐색에서도 Gaussian process의 예측 분산을 그대로 베이지안 최적화(28.4장)의 탐색 신호로 쓴다. 두 경우 모두 불확실성이 실제 오차와 상관이 있어야 작동하므로, 배포 전에 이 상관을 scaffold 분할에서 확인한다.
:::

::quiz{id="applicability-domain"}

## 실습

같은 합성 화학 공간에서 무작위·scaffold·계열(군집) 세 분할의 성능 차이와 불확실성–오차 관계를 5개 시드로 측정한다. 코드는 `labs/chemistry-and-materials/property-prediction/split_comparison.py`다. 계열 분할은 골격보다 더 거친 단위(골격 계열)로 나눠, 아예 처음 보는 화학 계열에 대한 일반화를 잰다. 집필 시점의 실행 결과는 다음과 같다.

```output
시드 0–4 평균 ± 표준편차
분할         test 수            MAE      평균 예측 MAE      최근접 유사도   Spearman(sd,|e|)
무작위           240   0.240 ± 0.016   0.954 ± 0.061        0.649      0.246 ± 0.039
scaffold      247   0.439 ± 0.117   0.949 ± 0.153        0.466      0.113 ± 0.185
계열            294   1.132 ± 0.366   1.200 ± 0.305        0.239      0.081 ± 0.117
```

- **분할이 거칠수록 오차가 커지고 흩어진다.** 무작위 0.240, scaffold 0.439, 계열 1.132로 커지고, 표준편차도 함께 커진다. 계열 분할에서는 test가 어느 계열이냐에 따라 성능이 크게 달라져, 한 번의 수치를 믿기 어렵다.
- **계열 분할에서는 모델이 평균 예측 baseline(1.200)을 거의 못 넘는다(1.132).** 처음 보는 계열에서는 모델이 배운 구조–물성 관계가 통하지 않아, 그냥 평균을 내는 것과 비슷해진다. 이것이 외삽의 현실이다.
- **불확실성–오차 상관은 분할이 거칠수록 약해진다.** 모델의 불확실성은 학습 분포 근처에서 잘 작동하고, 멀리 외삽할수록 그 자체가 믿을 수 없다. "불확실성이 오차를 가린다"는 주장도 분할에 따라 검증해야 한다.

실습에서는 시드를 더 늘려 보고, 계열 수나 골격 크기 분포를 바꿔 학습 곡선과 상관이 어떻게 변하는지 확인한다. 이어서 scaffold 분할을 직접 구현한다.

::exercise{id="scaffold-split"}

## 실패 사례

- **무작위 분할 성능으로 새로운 화학 구조에 대한 일반화를 주장하는 것.** 무작위 분할의 test 분자는 train에 친척이 있어, 실제 쓰임새보다 낙관적인 수치를 준다. 주장에 맞는 분할(scaffold·군집·시간)로 다시 잰다.
- **측정 잡음의 바닥 아래 성능을 보고하는 것.** 레이블 잡음 $\sigma$가 정하는 MAE 하한보다 낮은 수치는, 모델이 아니라 시험 레이블에 과적합했다는 신호다.
- **서로 다른 측정·계산 조건의 레이블을 섞고 조건을 무시하는 것.** 조건이 숨은 변수가 되어, 모델이 물성이 아니라 "어느 데이터셋에서 왔는가"를 배운다.
- **적용 영역 없이 점 예측만 보고하는 것.** 외삽 분자에 대한 예측을 보간 분자와 똑같이 신뢰하면, 선별 결과가 실제보다 좋아 보인다. 최근접 유사도나 예측 불확실성을 함께 보고한다.
- **불확실성과 오차의 상관을 확인하지 않고 불확실성 기반 선별을 쓰는 것.** 상관이 없으면 선별이 무작위와 다를 바 없다. scaffold 분할에서 먼저 확인한다.

## 통과 기준

- 한 데이터셋에 무작위 분할과 화학 공간 기반 분할(scaffold 또는 군집)을 모두 적용해 성능 차이를 수치로 보고한다.
- test 분자의 최근접 train 유사도 분포나 예측 불확실성으로 적용 영역을 함께 보고한다.
- 연습문제 `scaffold-split`의 테스트를 통과한다.

## 참고 문헌

- Bemis, Murcko (1996). The Properties of Known Drugs. 1. Molecular Frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887–2893. — 분자 골격(scaffold) 정의
- Sheridan (2013). Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction. *Journal of Chemical Information and Modeling*, 53(4), 783–790. — 시간 분할과 전향적 예측 성능 추정
- Wu, Ramsundar, Feinberg, Gomes, Geniesse, Pappu, Leswing, Pande (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9, 513–530. — 분자 물성 벤치마크와 분할
