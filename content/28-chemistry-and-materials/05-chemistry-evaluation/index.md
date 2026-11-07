---
id: chemistry-and-materials/chemistry-evaluation
title: "화학·재료 데이터와 평가의 함정"
title_en: "Data and Evaluation Pitfalls in Chemistry and Materials"
tier: core
status: draft
prerequisites: [chemistry-and-materials/property-prediction]
---

# 화학·재료 데이터와 평가의 함정

화학·재료 분야의 논문은 흔히 "우리 모델이 벤치마크에서 최고 성능"이라고 보고한다. 이 장은 그 수치를 비판적으로 읽는 법을 다룬다. 화학·재료 데이터에는 평가를 조용히 망가뜨리는 특성이 몇 가지 있다. 같은 분자가 여러 번 들어오고, 비슷한 구조가 train과 test에 갈라지며, 서로 다른 계산 수준의 값이 섞이고, 벤치마크가 포화되어 암기만으로도 높은 점수가 나온다. 이런 문제를 진단하지 못하면, 벤치마크 1위가 실제 쓸모와 무관할 수 있다.

15.4장의 분할과 누수, 21장의 불확실성을 화학 데이터에 적용한다. 방법을 다시 가르치지 않고, 화학·재료 벤치마크를 읽고 자기 데이터를 정제하는 판단을 다룬다. 참 규칙을 아는 합성 데이터로, "성능이 좋아 보이는 것"이 어디서 오는지 분해한다.

## 학습 목표

- 화학·재료 데이터의 품질 문제를 진단한다
- 벤치마크 결과를 비판적으로 해석한다

## 데이터 품질

**왜 필요한가.** 모든 평가는 데이터가 참값을 담는다고 가정한다. 화학 데이터에서 이 가정은 여러 이유로 깨진다. 같은 분자가 다른 표기로 중복되고, 측정 조건이 섞이며, 계산 수준이 다르다. 데이터를 정제하지 않으면 모델이 물성이 아니라 이 잡동사니를 배운다.

**중복의 두 종류.** 28.1장에서 본 대로 같은 분자가 여러 SMILES로 적힐 수 있다. 데이터셋을 합치면 **정확한 중복**(표준화하면 같은 분자)과 **근접 중복**(골격이 같거나 치환기 하나만 다른 분자)이 생긴다. 둘을 탐지하는 절차는 다르다. 정확한 중복은 표준화한 표현을 해시로 비교하고, 근접 중복은 분자 유사도(Tanimoto)로 찾는다.

```python {exec="browser" session="eval"}
import numpy as np

rng = np.random.default_rng(0)
L = 64
n_core = 120
cores = (rng.random((n_core, L)) < 0.12).astype(bool)      # 서로 다른 핵심 구조 120개

def make_dataset(n):
    X, origin = [], []
    for _ in range(n):
        c = rng.integers(n_core)
        x = cores[c].copy()
        for t in rng.choice(L, rng.integers(0, 3), replace=False):   # 치환기 0–2개 추가
            x[t] = True
        X.append(x); origin.append(c)
    return np.array(X), np.array(origin)

X, origin = make_dataset(1500)
keys = {x.tobytes(): i for i, x in enumerate(X)}
n_exact = len(X) - len(keys)                                 # 비트가 완전히 같은 중복
print(f"분자 {len(X)}개, 서로 다른 비트 벡터 {len(keys)}개 → 정확한 중복 {n_exact}개")

def tanimoto(A, B):
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)

S = tanimoto(X, X); np.fill_diagonal(S, 0)
near = (S >= 0.9).any(1)                                     # Tanimoto ≥ 0.9인 짝이 있는 분자
print(f"근접 중복(Tanimoto ≥ 0.9인 짝이 있는 분자): {near.sum()}개 ({near.mean():.0%})")
print(f"같은 핵심 구조에서 나온 분자 쌍의 평균 유사도: {S[origin[:, None] == origin[None]].mean():.3f}")
```

```output
분자 1500개, 서로 다른 비트 벡터 1003개 → 정확한 중복 497개
근접 중복(Tanimoto ≥ 0.9인 짝이 있는 분자): 805개 (54%)
같은 핵심 구조에서 나온 분자 쌍의 평균 유사도: 0.761
```

1500개 중 정확한 중복이 497개, 근접 중복까지 세면 절반이 넘는다. 실제 공개 데이터셋에서도 이런 중복은 흔하다. 중복을 그대로 두고 무작위로 분할하면, 같은 분자나 그 친척이 train과 test에 함께 들어간다.

**측정 조건과 계산 수준.** 실험값은 온도·용매·장비에 따라 다르고, 계산값은 함수·기저 집합에 따라 다르다(28.2장). 이 조건을 기록하지 않고 섞으면 조건이 숨은 변수가 된다. 아래에서 이것이 평가에 어떻게 나타나는지 본다.

## 누수

**근접 중복이 만드는 누수.** 15.4장에서 누수는 "평가가 실제 쓰임새보다 좋게 나오게 하는 정보의 샘" 이라고 정의했다. 화학에서 가장 흔한 누수는 근접 중복이 분할 간에 갈라지는 것이다. test 분자가 train에 거의 같은 친척을 두면, 그 친척의 레이블만 외워도 test를 맞힌다. 이것을 가장 적나라하게 보이는 모델이 **1-NN 암기 baseline**이다. 가장 가까운 train 분자의 레이블을 그대로 답으로 내는 모델이다.

```python {exec="browser" session="eval"}
w = rng.normal(0, 1.0, L)
y = X.astype(float) @ w + 0.1 * rng.normal(size=len(X))      # 물성 (N,)

def knn_predict(tr, te):                                     # 가장 가까운 train 분자(들)의 평균(암기 baseline)
    Sij = tanimoto(X[te], X[tr])
    mx = Sij.max(1)                                           # 최근접 유사도
    co_max = Sij >= mx[:, None] - 1e-9                        # 동점인 최근접 이웃 모두(순서 무관)
    return (co_max * y[tr][None]).sum(1) / co_max.sum(1), mx

perm = rng.permutation(len(X))
tr, te = perm[500:], perm[:500]
pred, nn_sim = knn_predict(tr, te)
leaked = nn_sim >= 0.9                                       # test 분자가 train에 근접 중복을 둔 경우
mae_all = np.abs(pred - y[te]).mean()
mae_clean = np.abs(pred - y[te])[~leaked].mean()
print(f"무작위 분할, 1-NN 암기 baseline (물성 표준편차 {y.std():.2f})")
print(f"  전체 test MAE            {mae_all:.3f}")
print(f"  근접 중복 test 비율       {leaked.mean():.0%}")
print(f"  근접 중복 제거 후 MAE     {mae_clean:.3f}  (누수가 성능을 {mae_clean / mae_all:.1f}배 좋아 보이게 했다)")
```

```output
무작위 분할, 1-NN 암기 baseline (물성 표준편차 2.28)
  전체 test MAE            0.518
  근접 중복 test 비율       52%
  근접 중복 제거 후 MAE     0.829  (누수가 성능을 1.6배 좋아 보이게 했다)
```

물성의 표준편차가 2.28인데 아무것도 학습하지 않는 암기 baseline이 전체 test에서 MAE 0.518을 낸다. 인상적으로 보이지만, test의 52%가 train에 근접 중복을 둔 덕이다. 근접 중복을 뺀 나머지에서는 MAE가 0.829로 올라간다. 누수가 성능을 1.6배 좋아 보이게 한 것이다. 사전학습 모델을 쓸 때는 **사전학습 코퍼스와 test의 중복**도 같은 누수를 만든다(19.9장). 큰 코퍼스로 사전학습한 모델이 벤치마크 test 분자를 이미 본 적이 있으면, 그 성능은 일반화가 아니라 기억이다.

::quiz{id="near-duplicate-leakage"}

## 벤치마크

**암기 baseline을 함께 보고한다.** 벤치마크 점수가 높을 때, 그것이 일반화인지 암기인지 가르는 가장 간단한 방법은 1-NN 암기 baseline을 함께 돌리는 것이다. 이 baseline이 강하면 train과 test가 너무 비슷하다는 신호다(Wallach, Heifets, 2018). 그리고 근접 중복을 제거하면 암기 baseline은 크게 흔들리지만 진짜로 일반화하는 모델은 거의 영향받지 않는다.

```python {exec="browser" session="eval"}
from numpy.linalg import lstsq

def ridge_fit(tr, te, lam=1.0):
    A = np.c_[X[tr].astype(float), np.ones(len(tr))]
    beta = np.linalg.solve(A.T @ A + lam * np.eye(L + 1), A.T @ y[tr])
    return np.c_[X[te].astype(float), np.ones(len(te))] @ beta

for name, keep in [("원본(중복 포함)", np.ones(len(te), bool)), ("근접 중복 제거 후", ~leaked)]:
    idx = te[keep]
    nn_pred, _ = knn_predict(tr, idx)
    ridge_pred = ridge_fit(tr, idx)
    print(f"{name:16s}  1-NN 암기 MAE {np.abs(nn_pred - y[idx]).mean():.3f}   Ridge 모델 MAE {np.abs(ridge_pred - y[idx]).mean():.3f}")
```

```output
원본(중복 포함)         1-NN 암기 MAE 0.518   Ridge 모델 MAE 0.091
근접 중복 제거 후        1-NN 암기 MAE 0.829   Ridge 모델 MAE 0.093
```

중복을 제거하면 암기 baseline은 0.518에서 0.829로 뛰지만, 구조–물성 관계를 배운 Ridge 모델은 0.091에서 0.093으로 거의 그대로다. **진짜 일반화하는 모델은 중복 제거에 둔감하다.** 그래서 벤치마크는 (1) 암기 baseline을 함께 보고하고, (2) 근접 중복을 제거한 뒤 평가하고, (3) 모델과 암기 baseline의 격차가 중복 제거 뒤에도 유지되는지 봐야 한다. 포화된 벤치마크(여러 모델이 비슷한 높은 점수에 몰린 경우)에서는 이 격차가 특히 중요하다. 점수 차이가 암기 여부의 차이일 수 있기 때문이다.

**지표와 오차 막대.** 두 모델의 점수 차이를 보고할 때는 그 차이가 우연의 범위를 넘는지 확인해야 한다. test 집합이 유한하므로 MAE에는 표본 오차가 있다. 부트스트랩(12장)으로 MAE 차이의 신뢰구간을 구한다.

```python {exec="browser" session="eval"}
def bootstrap_mae_diff(err_a, err_b, n_boot=2000):
    rg = np.random.default_rng(1)
    diffs = []
    idx = np.arange(len(err_a))
    for _ in range(n_boot):
        b = rg.choice(idx, len(idx), replace=True)
        diffs.append(err_a[b].mean() - err_b[b].mean())
    return np.percentile(diffs, [2.5, 97.5])

clean = te[~leaked]
err_nn = np.abs(knn_predict(tr, clean)[0] - y[clean])
err_ridge = np.abs(ridge_fit(tr, clean) - y[clean])
lo, hi = bootstrap_mae_diff(err_nn, err_ridge)
print(f"근접 중복 제거 test에서 MAE 차이(1-NN − Ridge): {err_nn.mean() - err_ridge.mean():+.3f}")
print(f"  95% 부트스트랩 구간 [{lo:+.3f}, {hi:+.3f}]  → 0을 포함하지 않음: {not (lo <= 0 <= hi)}")
```

```output
근접 중복 제거 test에서 MAE 차이(1-NN − Ridge): +0.736
  95% 부트스트랩 구간 [+0.655, +0.817]  → 0을 포함하지 않음: True
```

여기서는 구간이 0을 포함하지 않으므로 Ridge가 암기 baseline보다 유의하게 낫다. 반대로 두 최신 모델의 MAE 차이가 0.01인데 구간이 0을 넉넉히 포함한다면, "우리 모델이 더 낫다"는 주장은 근거가 약하다. 포화된 벤치마크에서 소수점 셋째 자리로 순위를 다투는 논문은 대개 이 점검을 생략한다. 오차 막대 없는 순위표는 읽는 사람이 채워 넣어야 한다.

::quiz{id="error-bars"}

**계산 수준의 혼합.** 데이터를 섞을 때 가장 조용한 함정은 서로 다른 계산 수준을 한 데이터셋에 넣는 것이다. 두 수준 사이에 체계적 차이(offset)가 있으면, 모델은 물성이 아니라 "이 분자가 어느 수준에서 계산됐는가"를 추측하려 한다. 그런데 추론 시점에는 그 정보가 없다.

```python {exec="browser" session="eval"}
level = rng.integers(0, 2, len(X))                           # 0, 1 두 계산 수준
offset = np.array([0.0, 3.0])                                # 수준마다 체계적 차이
y_mixed = X.astype(float) @ w + offset[level] + 0.1 * rng.normal(size=len(X))
tr2, te2 = perm[500:], perm[:500]
# (a) 수준을 무시하고 섞어 학습·평가
beta = np.linalg.solve(np.c_[X[tr2].astype(float), np.ones(len(tr2))].T @ np.c_[X[tr2].astype(float), np.ones(len(tr2))]
                       + np.eye(L + 1), np.c_[X[tr2].astype(float), np.ones(len(tr2))].T @ y_mixed[tr2])
pred_mixed = np.c_[X[te2].astype(float), np.ones(len(te2))] @ beta
mae_mixed = np.abs(pred_mixed - y_mixed[te2]).mean()
# (b) 한 수준만 써서 학습·평가
m0 = level == 0
tr0 = np.array([i for i in tr2 if m0[i]]); te0 = np.array([i for i in te2 if m0[i]])
A0 = np.c_[X[tr0].astype(float), np.ones(len(tr0))]
beta0 = np.linalg.solve(A0.T @ A0 + np.eye(L + 1), A0.T @ y_mixed[tr0])
mae_single = np.abs(np.c_[X[te0].astype(float), np.ones(len(te0))] @ beta0 - y_mixed[te0]).mean()
print(f"두 계산 수준(체계적 차이 {offset[1] - offset[0]:.1f})을 섞어 학습·평가한 MAE: {mae_mixed:.3f}")
print(f"한 수준만 써서 학습·평가한 MAE:                 {mae_single:.3f}")
print(f"섞었을 때 늘어난 오차의 상당 부분은 물리가 아니라 계산 수준 차이에서 온다")
```

```output
두 계산 수준(체계적 차이 3.0)을 섞어 학습·평가한 MAE: 1.498
한 수준만 써서 학습·평가한 MAE:                 0.098
섞었을 때 늘어난 오차의 상당 부분은 물리가 아니라 계산 수준 차이에서 온다
```

같은 모델, 같은 분자인데 두 계산 수준을 섞으면 MAE가 1.498로, 한 수준만 쓸 때(0.098)의 15배다. 모델이 추론 시점에 알 수 없는 "계산 수준"을 추측해야 하기 때문이다. 벤치마크가 이렇게 섞인 데이터로 만들어졌다면, 모든 모델의 오차에 이 혼합 성분이 섞여 있어 모델 간 비교가 흐려진다. 데이터 출처를 기록하고, 가능하면 한 수준 안에서 평가하거나 수준을 명시적 특징으로 넣는다.

:::note[분야 예시]
재료 물성 벤치마크는 여러 데이터베이스의 DFT 계산을 모아 만드는 경우가 많은데, 데이터베이스마다 함수·설정이 달라 같은 물질에도 다른 값을 준다. 이것을 무시하고 합치면 모델이 물성 대신 출처를 배운다. 신약 활성 데이터에서도 측정 분석법(assay)이 다른 값을 한 레이블로 합치면 같은 문제가 생긴다. Matbench 같은 벤치마크 모음이 과제별로 데이터 출처와 분할을 고정해 두는 이유가 이것이다(Dunn 외, 2020).
:::

## 실제 활용과의 간극

벤치마크 점수가 높아도 실제 연구에서 쓸모가 없을 수 있다. 벤치마크는 "주어진 test 분포에서의 평균 오차"를 재지만, 실제 쓰임새는 대개 "새로운 화학 구조에서 유망한 후보를 고르는" 일이다. 둘 사이의 간극을 메우는 지표는 평균 오차가 아니라 **후보 선별 성공률**이다. 상위 $k$개 예측 중 실제로 목표를 만족한 비율(hit rate), 상위 후보의 농축(enrichment), 그리고 그 후보가 실험으로 검증됐는지다(28.4장). 벤치마크 MAE가 0.01 낮은 모델이 선별 성공률은 같거나 낮을 수 있다. 쓰임새에 맞는 지표로 평가하고, 가능하면 전향적(prospective) 검증, 즉 모델이 고른 후보를 실제로 합성·측정해 확인한 결과를 보고한다.

## 실습

공개 데이터셋 하나(또는 합성 데이터)에서 중복·유사 구조를 탐지하고, 정제 전후의 벤치마크 성능 차이를 측정한다. 본문 코드가 전체 흐름을 보여 준다. 실습에서는 다음을 바꿔 본다.

- **근접 중복 임계(Tanimoto)**를 0.7–0.99로 바꾸며, 임계에 따라 "누수"로 판정되는 test 비율과 정제 후 MAE가 어떻게 변하는지 본다. 임계 선택이 곧 "무엇을 같은 분자로 볼 것인가"의 정의다.
- **계산 수준 offset**을 0에서 키우며, 섞인 데이터의 MAE가 어디까지 커지는지 본다.
- **두 모델의 MAE 차이**에 부트스트랩 구간을 붙여, test 크기를 줄이면 구간이 어떻게 넓어지는지 확인한다.

연습문제로 근접 중복에 의한 분할 누수를 탐지하는 함수를 구현한다.

::exercise{id="leakage-audit"}

## 실패 사례

- **서로 다른 계산 수준의 데이터를 그대로 섞는 것.** 체계적 차이가 잡음처럼 들어와, 모델이 물성 대신 출처를 배운다. 출처를 기록하고 한 수준 안에서 평가하거나 수준을 특징으로 넣는다.
- **근접 중복을 제거하지 않고 무작위 분할로 평가하는 것.** 암기만으로 test를 맞혀 성능이 부풀려진다. 근접 중복을 제거하고, 암기 baseline을 함께 보고한다.
- **사전학습 코퍼스와 벤치마크 test의 중복을 확인하지 않는 것.** 모델이 이미 본 분자를 "예측"한다. 코퍼스와 test의 중복을 점검한다.
- **오차 막대 없이 소수점 순위로 모델을 비교하는 것.** test 표본 오차 안의 차이는 우연일 수 있다. MAE 차이에 신뢰구간을 붙인다.
- **벤치마크 MAE로 실제 선별 성공을 주장하는 것.** 평균 오차와 후보 선별 성공률은 다르다. 쓰임새에 맞는 지표와 전향적 검증으로 보고한다.

## 통과 기준

- 데이터셋에서 정확한 중복과 근접 중복을 탐지하고, 정제 전후의 성능 차이를 수치로 보고한다.
- 분할 근거(무작위·scaffold·군집·시간), 암기 baseline, 모델 간 차이의 오차 막대를 포함한 평가 보고서를 쓴다.
- 연습문제 `leakage-audit`의 테스트를 통과한다.

## 참고 문헌

- Wallach, Heifets (2018). Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization. *Journal of Chemical Information and Modeling*, 58(5), 916–932. — train/test 중복이 암기를 보상하는 벤치마크 편향(AVE)
- Dunn, Wang, Ganose, Dopp, Jain (2020). Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials*, 6, 138. — 재료 물성 벤치마크의 과제별 데이터·분할 고정
- Sheridan (2013). Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction. *Journal of Chemical Information and Modeling*, 53(4), 783–790. — 전향적 예측 성능 추정
