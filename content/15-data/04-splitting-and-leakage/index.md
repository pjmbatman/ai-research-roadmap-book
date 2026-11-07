---
id: data/splitting-and-leakage
title: "데이터 분할과 누수 방지"
title_en: "Data Splitting and Leakage Prevention"
tier: core
status: draft
prerequisites: [data/preprocessing-and-eda]
---

# 데이터 분할과 누수 방지

test 성능은 "이 모델이 아직 보지 못한 데이터에서 얼마나 잘 하는가"를 재는 숫자다. 그런데 test 데이터에 대한 정보가 어떤 경로로든 학습 과정에 들어가면, 이 숫자는 아무것도 재지 못한다. 이런 정보의 유입을 **누수**(leakage)라 한다. 누수는 코드 오류처럼 프로그램을 멈추지 않는다. 오히려 성능을 좋게 보이게 하므로 누구도 의심하지 않는다. 머신러닝을 쓰는 여러 과학 분야의 논문을 조사한 연구는 누수가 17개 분야 294편 이상의 논문에 영향을 주었다고 보고했다(Kapoor·Narayanan, 2023).

이 장은 누수를 막는 두 가지 일을 다룬다. 첫째, **분할을 주장과 맞춘다.** test는 논문이 주장하려는 일반화 대상(새 셀, 새 환자, 미래 시점, 새 지역)을 흉내 내야 한다. 둘째, **test의 정보가 학습에 들어가는 경로를 모두 막는다.** 전처리 통계, 특징 선택, 하이퍼파라미터 선택, 중복 샘플, 사전학습 데이터가 모두 그런 경로가 될 수 있다. 15.1장의 독립성의 단위와 15.3장의 "통계량은 train에서만"이 여기서 하나의 체계가 된다.

## 학습 목표

- 일반화 대상에 맞는 분할을 설계한다
- 누수의 유형을 식별하고 방지한다
- test set을 위생적으로 관리한다

## 분할 전략

**왜 필요한가.** 같은 데이터, 같은 모델이라도 어떻게 나누느냐에 따라 test 성능이 크게 달라진다. 그리고 어떤 분할이 옳은지는 데이터가 아니라 **주장이** 정한다. "이미 측정한 셀의 다음 사이클"을 예측하는 모델과 "아직 만들지 않은 셀"을 예측하는 모델은 다른 분할로 평가해야 한다.

**네 가지 분할.**

| 분할 | 방법 | 대응하는 주장 |
|---|---|---|
| random split | 샘플을 무작위로 나눈다 | 같은 분포에서 독립으로 뽑은 새 샘플. 샘플이 정말 독립일 때만 |
| group split | 그룹(셀·환자·분자 골격)을 통째로 한쪽에 둔다 | 처음 보는 그룹 |
| time-based split | 어떤 시점 이전은 train, 이후는 test | 미래 시점 |
| stratified split | label이나 조건의 비율을 양쪽에서 맞춘다 | 다른 분할과 함께 써서 희귀한 class·조건이 한쪽에 몰리지 않게 한다 |

이 밖에 **공간 분할**(spatial blocking)은 지도를 구역으로 나누어 구역 단위로 train과 test를 정한다. 가까운 지점의 값은 비슷하므로, 무작위 분할에서는 test 지점 바로 옆의 지점이 train에 있다. 생태·지구과학에서는 시간·공간·계층·계통 구조가 있는 데이터를 무작위로 교차검증하면 예측 오차를 심하게 과소평가한다는 점이 체계적으로 정리되어 있다(Roberts 외, 2017).

**직관.** 그룹마다 공유하는 효과 $u_g$가 있는 데이터 $y = f(x) + u_g + \varepsilon$에서, 무작위 분할의 test 샘플은 대개 같은 그룹의 train 샘플을 이웃으로 가진다. 최근접 이웃 모델은 그 이웃의 $u_g$를 베껴 와 test를 잘 맞힌다. 그룹 분할에서는 test 그룹의 $u_g$를 한 번도 보지 못했으므로 그만큼 오차가 커진다. 그 오차가 처음 보는 그룹에 대한 정직한 성능이다. 아래 위젯에서 그룹 효과의 크기 $\sigma_g$를 바꾸며 두 분할의 차이를 확인한다.

::widget{name="split-leakage"}

같은 실험을 scikit-learn으로 한다. 그룹 16개에 샘플이 15개씩 있다.

```python {exec="browser" session="split"}
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
G, PER = 16, 15                                               # 그룹 16개 × 샘플 15개
groups = np.repeat(np.arange(G), PER)                         # (240,)
grid = np.stack(np.meshgrid(np.arange(4), np.arange(4)), -1).reshape(G, 2)
centers = 1.2 * grid - 1.8 + rng.uniform(-0.25, 0.25, (G, 2))  # (16, 2) 그룹마다 다른 위치
X = centers[groups] + rng.normal(0, 0.33, size=(G * PER, 2))  # (240, 2)
u = rng.normal(0, 1.0, size=G)                                # (16,) 그룹이 공유하는 효과
y = np.sin(0.8 * X[:, 0]) + 0.3 * X[:, 1] + u[groups] + rng.normal(0, 0.15, G * PER)

knn = KNeighborsRegressor(n_neighbors=5)
rmse = "neg_root_mean_squared_error"
s_rand = cross_val_score(knn, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring=rmse)
s_group = cross_val_score(knn, X, y, groups=groups, cv=GroupKFold(4), scoring=rmse)
print(f"무작위 5-fold: RMSE {-s_rand.mean():.2f}")
print(f"그룹 4-fold  : RMSE {-s_group.mean():.2f}")
```

```output
무작위 5-fold: RMSE 0.75
그룹 4-fold  : RMSE 1.40
```

같은 모델, 같은 데이터인데 무작위 분할의 RMSE(0.75)가 그룹 분할(1.40)의 절반 남짓이다. "새 그룹에서도 통한다"고 주장하면서 0.75를 보고하면 성능을 두 배 가까이 부풀린 셈이다. scikit-learn에는 `GroupKFold` 외에도 그룹을 하나씩 빼는 `LeaveOneGroupOut`, 시간 순서를 지키는 `TimeSeriesSplit`, 그룹과 label 비율을 함께 고려하는 `StratifiedGroupKFold`가 있다.

::quiz{id="split-matches-claim"}

## 독립성의 단위

**왜 필요한가.** 그룹 분할을 하려면 무엇이 그룹인지 정해야 한다. 15.1장에서 본 독립성의 단위가 그 답이다. 그러나 실제 데이터에는 단위가 여러 겹으로 겹쳐 있다.

**겹쳐 있는 단위.**

| 단위 | 예 | 같은 단위 안에서 공유되는 것 |
|---|---|---|
| subject | 환자, 실험 동물, 참가자 | 생리적 특성, 생활 습관 |
| cell | 배터리 셀, 세포주 | 제조 편차, 유전적 배경 |
| device | 측정 장비, 센서 | 보정 오차, 잡음 특성 |
| batch | 제조 lot, 실험 날짜, 시약 batch | 공정 조건, 환경 |
| 실험실 | 연구 기관, 병원 | 프로토콜, 장비, 대상 집단 |

배터리 데이터에서 셀 단위로 나누었더라도, 같은 lot의 셀이 양쪽에 있으면 lot 효과는 여전히 샐 수 있다. 어느 단위까지 나누어야 하는지는 주장의 범위가 정한다. "같은 공정으로 만든 새 셀"이면 셀 단위로 충분하고, "다른 제조사의 셀"이면 제조사 단위로 나누어야 한다. 단위가 너무 커서 그룹 수가 적으면(예: 실험실 3곳) fold마다 성능이 크게 흔들리므로, 평균과 함께 fold별 결과를 모두 보고한다.

:::note[분야 예시]
신약 개발의 분자 데이터에서는 같은 골격(scaffold)을 공유하는 분자들이 비슷한 성질을 가진다. 무작위 분할의 test 분자는 대개 train에 거의 같은 골격의 친척이 있다. 그래서 골격 단위로 나누는 scaffold split이 "새로운 화학 구조"에 대한 일반화를 평가하는 표준적인 방법으로 쓰인다. 단백질에서는 서열 유사도로 군집을 만들어 군집 단위로 나눈다. 의료 영상에서는 한 환자의 여러 영상, 같은 병원의 여러 환자가 겹친 단위가 된다.
:::

## 누수의 유형

**정의.** 누수는 예측 시점에 정당하게 이용할 수 없는 정보, 특히 평가 대상(test)의 정보가 모델의 학습이나 입력에 들어가는 것이다(Kaufman 외, 2012). 경로에 따라 여러 유형이 있다.

| 유형 | 예 | 방지 |
|---|---|---|
| 같은 단위의 분리 | 같은 셀의 사이클이 양쪽에 | 그룹 분할 |
| 중첩 window·인접 시점 | stride 1 window를 무작위로 나눔 | 시간 분할 + gap |
| 미래 정보 | 중심 이동 평균, 사후에 기록된 값을 입력으로 | 예측 시점 기준으로 특징 계산(1.3장) |
| 전처리 | 전체 데이터로 정규화·결측 대체 | train에서만 fit |
| feature selection | 전체 데이터의 label로 특징 선택 | pipeline 안에서 선택 |
| 튜닝 | test 점수로 하이퍼파라미터 선택 | validation 분리 |
| 중복 샘플 | 같은 측정이 두 번 기록, 사전학습 데이터와 겹침 | 정규화 후 해시·유사도 점검 |

**중첩 window와 인접 시점.** 시계열 모델은 흔히 긴 신호를 길이 $W$의 window로 잘라 샘플을 만든다. stride를 1로 하면 이웃한 window는 $W-1$개의 값을 공유한다. 이 window들을 무작위로 섞어 나누면, test window마다 한 칸 옆의 거의 같은 window가 train에 있다. 정답도 마찬가지로 거의 같다. 30 step 입력으로 10 step 뒤를 예측하는 문제에서 비교한다.

```python {exec="browser" session="split"}
x = np.zeros(3000)
for k in range(1, 3000):                                      # 정상(stationary)인 AR(1) 신호
    x[k] = 0.97 * x[k - 1] + rng.normal(0, 0.25)
W, H = 30, 10                                                 # 입력 30 step, 10 step 뒤를 예측
starts = np.arange(len(x) - W - H)                            # (2960,) stride 1로 겹치는 window
Xw = np.stack([x[s:s + W] for s in starts])                   # (2960, 30)
yw = x[starts + W + H - 1]                                    # (2960,)

def rmse_of(tr, te):                                          # 1-최근접 이웃 예측의 RMSE
    a, b = Xw[te], Xw[tr]                                     # (n_te, 30), (n_tr, 30)
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T   # (n_te, n_tr)
    pred = yw[tr][d2.argmin(axis=1)]                          # 가장 가까운 train window의 정답
    return np.sqrt(np.mean((pred - yw[te]) ** 2))

perm = rng.permutation(len(starts))
cut = int(0.8 * len(starts))
print(f"무작위 분할        : RMSE {rmse_of(perm[:cut], perm[cut:]):.2f}")
gap = W + H                                                   # 경계에서 겹치는 window를 버린다
print(f"시간 분할 + gap {gap}: RMSE {rmse_of(starts[:cut - gap], starts[cut:]):.2f}")
```

```output
무작위 분할        : RMSE 0.41
시간 분할 + gap 40: RMSE 0.83
```

무작위 분할의 RMSE(0.41)는 시간 분할(0.83)의 절반이다. 1-최근접 이웃 모델은 test window와 한 칸 어긋난 train window를 찾아 그 정답을 그대로 내놓았다. 이 모델은 미래를 예측하는 법이 아니라 이웃 window를 찾는 법을 배웠다. 시간 분할에서도 경계 근처의 window는 서로 값을 공유하므로, 경계에서 $W + H$ 칸 안쪽의 train window를 버리는 **gap**을 둔다.

::quiz{id="window-gap"}

:::caution[함정]
"시간 순서로 나누었으니 안전하다"고 생각하기 쉽다. 그러나 여러 entity(셀, 관측소)가 섞인 데이터를 한 시점으로 자르면, 같은 entity의 과거가 train에, 미래가 test에 있다. "이미 본 셀의 미래"에 대한 주장이라면 맞는 분할이지만, "새 셀"에 대한 주장이라면 시간과 그룹을 함께 나누어야 한다.
:::

::quiz{id="leakage-types"}

## fit은 학습 데이터에서만

**왜 필요한가.** 데이터에서 무언가를 추정하는 모든 단계는 학습이다. 정규화의 평균과 표준편차, 결측 대체에 쓰는 중앙값, PCA의 축, 특징 선택의 순위, 이상치 판정의 임계값이 모두 그렇다. 이 추정에 test가 들어가면 test의 정보가 모델에 들어간다. 15.3장에서는 정규화 통계가 분포 이동을 가리는 것을 보았다. 특징 선택의 누수는 훨씬 극적이다. 특징 5000개가 모두 순수한 잡음이고 label도 무작위인 데이터에서 비교한다.

```python {exec="browser" session="split"}
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

Xn = rng.normal(size=(100, 5000))                             # (100, 5000) 순수한 잡음 특징
yn = rng.integers(0, 2, size=100)                             # (100,) 특징과 무관한 label
cv = StratifiedKFold(5, shuffle=True, random_state=0)

X_sel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)    # 전체 데이터의 label로 선택
leaky = cross_val_score(LogisticRegression(), X_sel, yn, cv=cv).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
honest = cross_val_score(pipe, Xn, yn, cv=cv).mean()         # fold마다 train에서만 선택
print(f"선택 후 CV (누수)   : 정확도 {leaky:.2f}")
print(f"pipeline 안에서 선택: 정확도 {honest:.2f}")
```

```output
선택 후 CV (누수)   : 정확도 0.86
pipeline 안에서 선택: 정확도 0.32
```

전체 데이터의 label로 특징 20개를 고르고 교차검증하면 정확도가 0.86이다. 잡음뿐인 데이터에서 "예측 모델"을 만들어 낸 것이다. 5000개 특징 중 우연히 label과 상관된 것을 골랐고, 그 우연한 상관은 선택에 쓰인 샘플(test fold 포함)에서만 성립한다. 선택을 pipeline 안에 넣어 fold마다 train에서만 하면 정확도는 우연 수준으로 내려간다. 여기서는 0.32로 0.5보다도 낮게 나왔는데, 샘플 100개의 교차검증 추정이 이 정도로 흔들린다는 뜻이지 모델이 반대로 예측하는 법을 배웠다는 뜻은 아니다.

::quiz{id="feature-selection-leak"}

**pipeline 패턴.** 추정 단계를 모델과 함께 하나의 객체로 묶으면, 교차검증 도구가 fold마다 train에서 fit하고 test에 transform하는 일을 대신 해 준다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
scores = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5))
```

PyTorch로 직접 학습 루프를 쓸 때도 원칙은 같다. 정규화 통계는 train `DataLoader`의 데이터로만 계산해 저장하고(체크포인트에 함께), validation·test·배포 시에 그 값을 불러 쓴다.

## validation과 test의 역할

**왜 필요한가.** 모델을 만드는 과정은 선택의 연속이다. 구조, 학습률, 정규화 강도, epoch 수, 특징 조합을 고른다. 고를 때 쓴 데이터의 점수는 고른 모델에 유리하게 치우친다. 여러 후보 중 가장 좋은 것을 고르면, 그 점수에는 실력과 함께 운이 들어 있기 때문이다.

**역할 분리.**

- **train**: 파라미터를 학습한다.
- **validation**: 모델·하이퍼파라미터를 고르고, early stopping 시점을 정한다. train 안의 교차검증으로 대신할 수 있다.
- **test**: 모든 선택이 끝난 최종 모델을 **한 번** 평가해 보고한다.

test를 보고 무언가를 바꾸면 test는 validation이 된다. 효과를 계산으로 확인한다. label이 무작위인 문제에서 무작위 모델 500개를 같은 집합으로 채점해 가장 좋은 것을 고른다.

```python {exec="browser" session="split"}
X_val, y_val = rng.normal(size=(200, 10)), rng.integers(0, 2, 200)   # label은 무작위
X_new, y_new = rng.normal(size=(200, 10)), rng.integers(0, 2, 200)   # 한 번도 안 본 데이터
best_acc, best_w = 0.0, None
for _ in range(500):                                          # 무작위 모델 500개 중 가장 좋은 것
    w = rng.normal(size=10)                                   # (10,)
    acc = np.mean((X_val @ w > 0) == y_val)
    if acc > best_acc:
        best_acc, best_w = acc, w
print(f"고르는 데 쓴 집합의 정확도: {best_acc:.3f}")
print(f"새 데이터에서의 정확도   : {np.mean((X_new @ best_w > 0) == y_new):.3f}")
```

```output
고르는 데 쓴 집합의 정확도: 0.570
새 데이터에서의 정확도   : 0.450
```

고르는 데 쓴 집합에서 0.570이던 모델이 새 데이터에서는 0.450이다. label이 무작위이므로 어떤 모델도 0.5를 넘을 수 없다. 0.57은 500번 중 가장 운이 좋았던 결과일 뿐이다. 하이퍼파라미터를 수십 번 바꾸며 test 점수를 확인하는 것도 같은 일이다.

::quiz{id="best-of-many"}

**test set 사용 기록.** test의 위생을 지키는 가장 실용적인 방법은 기록이다. test를 평가할 때마다 날짜, 모델 버전(커밋), 평가 결과, 평가 이유를 한 줄씩 남긴다. 기록이 한 줄을 넘으면, 그 이후의 test 점수는 몇 번째 사용인지 함께 보고한다. 분할 자체도 파일(split manifest)로 고정해 실험마다 같은 test를 쓰게 한다(15.5장).

::quiz{id="test-reuse"}

## 사전학습 데이터와의 중복

**왜 필요한가.** 사전학습(pretraining)된 모델을 downstream 과제에 쓰면, 사전학습 데이터는 학습 데이터의 일부다. downstream의 test 샘플이 사전학습 데이터에 이미 있었다면, 모델은 그 샘플을 이미 보았다. 공개 데이터를 대규모로 모은 사전학습 데이터는 공개 benchmark와 겹치기 쉽다.

**점검 방법.** 세 단계로 점검한다.

1. **정확한 중복**: 표기를 정규화(대소문자, 공백, 단위, 분자라면 표준 표기)한 뒤 해시로 비교한다.
2. **근접 중복**: 서열 동일성, 분자 유사도, 영상 지각 해시처럼 분야에 맞는 유사도로 가까운 샘플을 찾는다.
3. **분리 보고**: 겹치는 test 샘플과 나머지의 성능을 나누어 보고한다.

```python {exec="browser" session="split"}
import hashlib

AA = list("ACDEFGHIKLMNPQRSTVWY")
pretrain = ["".join(rng.choice(AA, 30)) for _ in range(2000)]        # 사전학습 서열
test = ["".join(rng.choice(AA, 30)) for _ in range(180)]             # 새 서열 180개
test += pretrain[:15]                                                 # 그대로 복사된 15개
test += [" " + s.lower() for s in pretrain[100:105]]                  # 표기만 다른 5개

def key(seq):
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()  # 정규화 후 해시

exact = len(set(test) & set(pretrain))
normalized = len({key(s) for s in test} & {key(s) for s in pretrain})
print("문자열 그대로 비교한 중복:", exact)
print("정규화 후 해시로 찾은 중복:", normalized)
```

```output
문자열 그대로 비교한 중복: 15
정규화 후 해시로 찾은 중복: 20
```

문자열 그대로 비교하면 15개만 찾고, 표기를 정규화하면 앞뒤 공백과 소문자로 기록된 5개를 더 찾는다. 한두 자리만 다른 서열은 이 방법으로도 찾지 못하므로 유사도 기준이 따로 필요하다. 사전학습 데이터 목록을 공개하지 않은 모델은 이 점검을 할 수 없다. 그 경우 결과의 한계로 명시한다.

::quiz{id="pretraining-overlap"}

## 외부 검증

**왜 필요한가.** 분할을 아무리 잘 설계해도, 같은 실험실·같은 장비·같은 기간에 모은 데이터 안에서의 분할이다. 공유된 조건이 모든 샘플에 들어 있으면 어떤 분할로도 그 조건의 영향을 분리하지 못한다.

**정의.** 외부 검증(external validation)은 학습 데이터를 만든 과정과 독립적으로 수집한 데이터로 평가하는 것이다. 세 방향이 있다.

- **새 도메인**: 다른 실험실, 다른 병원, 다른 제조사의 데이터
- **새 조건**: 학습 범위 밖의 온도·속도·농도
- **새 시점**: 모델을 만든 뒤에 수집한 데이터

외부 검증의 성능은 대개 내부 test보다 낮다. 이 차이가 크다는 것은 실패가 아니라 정보다. 모델이 학습 데이터에 공유된 조건에 얼마나 기대고 있었는지를 보여 준다. 외부 데이터를 구할 수 없으면 가장 가까운 대안으로 "가장 바깥 단위로 하나씩 빼는" 분할(실험실 하나 빼기, 기간 하나 빼기)을 쓰고, 그 한계를 쓴다.

:::note[분야 예시]
기후·생태 모델에서는 한 지역에서 학습한 모델을 멀리 떨어진 지역에 적용해 공간 외삽 능력을 본다. 임상 예측 모델은 개발한 병원과 다른 병원의 환자에서 성능을 확인해야 진료에 쓸 근거가 된다. 배터리 수명 예측은 학습에 쓰지 않은 충전 프로토콜에서의 성능으로 새 조건에 대한 일반화를 확인한다.
:::

## 실습

같은 데이터(자신의 연구 데이터, 또는 그룹 구조가 있는 공개 데이터)에 random split과 group split을 적용해 성능 차이를 측정한다.

1. 독립성의 단위를 정하고 그 근거를 쓴다(15.1장의 데이터 사전).
2. 같은 모델, 같은 하이퍼파라미터로 random K-fold와 group K-fold의 성능을 fold별로 기록한다. 시계열이면 시간 분할(gap 포함)도 추가한다.
3. 두 성능의 차이를 해석한다. 차이가 크면 그룹 효과가 크다는 뜻이고, 주장에 맞는 쪽의 숫자만 보고한다.
4. 아래 **누수 감사 체크리스트**를 자신의 파이프라인에 적용하고 항목마다 근거를 쓴다.

| 점검 항목 | 확인 방법 |
|---|---|
| 분할이 주장하는 일반화 대상과 일치하는가 | 주장 문장과 분할 단위를 나란히 쓴다 |
| 같은 독립 단위가 train과 test에 동시에 있는가 | 그룹 ID의 교집합이 공집합인지 코드로 확인 |
| 시계열 window가 경계에서 겹치는가 | gap 크기와 계산 근거 |
| 예측 시점 이후의 정보를 쓰는 특징이 있는가 | 특징마다 계산에 쓰는 시점 범위 |
| 데이터에서 추정하는 모든 단계가 train에서만 fit되는가 | pipeline 구성 또는 통계량 저장 위치 |
| 하이퍼파라미터 선택에 test를 썼는가 | test 사용 기록 |
| train·test·사전학습 데이터 사이에 중복이 있는가 | 정규화 해시와 유사도 점검 결과 |

그룹 분할의 핵심 함수를 직접 구현한다.

::exercise{id="group-kfold"}

## 실패 사례

- **같은 셀의 다른 사이클이 학습·테스트에 섞이는 것.** 모델은 셀의 고유한 특성을 외워 test를 맞힌다. "새 셀"에 대한 주장과 "이미 본 셀"에 대한 성능이 어긋난다. 셀 ID의 교집합을 코드로 확인한다.
- **test 성능을 보고 하이퍼파라미터를 고르는 것.** 고른 점수는 운이 섞여 부풀려진다. 선택은 validation으로, test는 한 번만.
- **전처리·특징 선택을 분할 전에 하는 것.** 잡음에서도 높은 성능이 나온다. 모든 추정 단계를 pipeline 안에 넣는다.
- **stride 1 window를 무작위로 나누는 것.** test window의 거의 복사본이 train에 있다. 시간 분할과 gap을 쓴다.
- **그룹 수가 적은데 평균 성능만 보고하는 것.** fold 하나가 결과를 좌우할 수 있다. fold별 결과와 그룹 수를 함께 보고한다.

## 통과 기준

- 자신의 분할이 주장하는 일반화 대상과 일치함을 독립성의 단위로 논증하고, random split과 group(또는 시간) split의 성능 차이를 fold별로 기록한다.
- 누수 감사 체크리스트의 모든 항목에 근거를 쓰고, 그룹 교집합·pipeline 구성을 코드로 확인한다.
- 연습문제 `group-kfold`의 테스트를 통과한다.

## 참고 문헌

- Kapoor, Narayanan (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. — 여러 과학 분야의 누수 사례 조사와 누수 유형 분류
- Kaufman, Rosset, Perlich, Stitelman (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15. — 누수의 정의와 학습–예측 분리 원칙
- Roberts 외 (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. — 구조가 있는 데이터의 block 교차검증
