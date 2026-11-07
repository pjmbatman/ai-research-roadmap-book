---
id: classical-ml/trees-and-ensembles
title: "트리와 앙상블"
title_en: "Trees and Ensembles"
tier: core
status: draft
prerequisites: [classical-ml/baselines]
---

# 트리와 앙상블

실험 기록, 공정 조건, 환자 기록, 관측소 자료처럼 행이 샘플이고 열이 서로 다른 종류의 측정인 **표 데이터**(tabular data)는 과학 연구에서 가장 흔한 데이터 형태다. 이런 데이터에서 가장 강한 baseline은 대개 신경망이 아니라 결정 트리의 앙상블, 특히 잘 튜닝된 gradient boosting이다. 트리는 특징의 단위나 단조 변환에 무관하고, 범주형 변수와 문턱 효과를 자연스럽게 다루고, 무관한 특징에 비교적 강하다.

이 장은 결정 트리 하나에서 시작해, 트리를 평균해 분산을 줄이는 bagging과 랜덤 포레스트, 트리를 순차적으로 더해 편향을 줄이는 gradient boosting으로 나아간다. 그리고 트리 모델과 함께 거의 항상 보고되는 **특징 중요도**가 어떻게 틀린 결론으로 이어지는지 본다. 16.1장의 관점에서, 이 장의 목표는 표 데이터 문제에서 새 방법이 넘어야 할 공정하고 강한 기준선을 세우는 것이다.

## 학습 목표

- 결정 트리·랜덤 포레스트·gradient boosting을 사용한다
- 표 데이터에서 강한 baseline을 만든다
- 특징 중요도를 시험 데이터에서 계산하고, 그 편향과 한계를 설명한다

## 결정 트리

**정의.** 회귀 트리는 입력 공간을 축에 평행한 직사각형 영역들로 나누고, 영역마다 학습 출력의 평균을 예측으로 쓴다. 영역은 이진 분할을 반복해 만든다. 각 단계에서 특징 $j$와 문턱 $s$를 골라 현재 영역을 $\{x_j \le s\}$와 $\{x_j > s\}$로 나누는데, 나눈 뒤의 제곱오차 합이 가장 작아지는 $(j, s)$를 고른다.

$$
\min_{j,\, s} \Big[ \sum_{i:\, x_{ij} \le s} (y_i - \bar y_L)^2 + \sum_{i:\, x_{ij} > s} (y_i - \bar y_R)^2 \Big].
$$

$\bar y_L, \bar y_R$은 두 영역의 학습 출력 평균이다. 분류 트리는 제곱오차 대신 **불순도**(impurity)를 줄인다. 클래스 비율이 $p_k$인 영역의 Gini 불순도는 $1 - \sum_k p_k^2$, 엔트로피는 $-\sum_k p_k \log p_k$다. 한 클래스만 남으면 둘 다 0이다. 가능한 모든 분할 중 지금 당장 가장 좋은 것을 고르는 **탐욕적**(greedy) 방법이므로 전체 최적의 트리를 보장하지는 않는다.

**직관과 성질.** 트리의 예측은 계단 함수다. 이 구조에서 세 가지 성질이 나온다.

- **단조 변환에 무관하다.** 분할은 값의 **순서**만 쓰므로, 특징을 표준화하거나 로그를 씌워도 같은 트리가 나온다. 16.2·16.3장에서 거듭 문제가 된 특징 스케일이 트리에는 문제가 되지 않는다.
- **상호작용과 문턱을 자연스럽게 표현한다.** "온도가 520–600 °C이고 시간이 3시간 이상이면 강도가 뛴다" 같은 조건은 분할 세 번으로 표현된다. 선형 모델은 이를 위해 특징을 직접 만들어야 한다.
- **외삽하지 않는다.** 학습 범위 밖의 입력은 가장자리 영역에 떨어져 그 영역의 평균, 즉 상수를 예측한다. 추세를 연장하지 못한다.

**깊이와 과적합.** 분할을 계속하면 결국 잎(leaf)마다 학습 점 하나가 남아 학습 오차가 0이 된다. 시효 조건과 조성으로 합금의 항복강도를 예측하는 가상의 표 데이터로 확인한다. 특징은 시효 온도·시간, Cu·Mg 함량, 공정 라인(범주형, 코드 0–4), 그리고 출력과 **무관한** 연속 특징 하나와 로트 번호(값의 종류가 1000가지)다. 강도에는 조성의 선형 효과, 최적 시효 조건에서의 문턱 효과, 시간의 매끄러운 효과, 라인별 작은 차이가 섞여 있고, 측정 잡음의 표준편차는 10 MPa다.

```python {exec="browser" session="tree"}
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def make_alloy(n, rng):                                    # 가상의 합금 시편 표 데이터
    temp_c = rng.uniform(450, 650, n)                      # 시효 온도 [°C]
    time_h = rng.uniform(0.5, 12, n)                       # 시효 시간 [h]
    cu = rng.uniform(0, 0.05, n)                           # Cu 질량분율
    mg = rng.uniform(0, 0.02, n)                           # Mg 질량분율
    line = rng.integers(0, 5, n)                           # 공정 라인 (범주형, 코드 0–4)
    noise_x = rng.normal(size=n)                           # 출력과 무관한 연속 특징
    lot_id = rng.integers(0, 1000, n)                      # 출력과 무관한 로트 번호(값의 종류가 많다)
    peak = (temp_c > 520) & (temp_c < 600) & (time_h > 3)  # 최적 시효 조건에서 강도가 뛴다
    y = (200 + 3000 * cu + 5000 * mg + 40 * peak + 15 * np.sin(time_h)
         + np.array([0, 10, -5, 15, 0])[line] + rng.normal(0, 10, n))   # 항복강도 [MPa]
    X = np.column_stack([temp_c, time_h, cu, mg, line, noise_x, lot_id])  # (n, 7)
    return X, y

names = ["temp_c", "time_h", "cu", "mg", "line", "noise_x", "lot_id"]
rng = np.random.default_rng(0)
X_tr, y_tr = make_alloy(1000, rng)
X_te, y_te = make_alloy(1000, rng)
rmse = lambda m, X, y: np.sqrt(np.mean((m.predict(X) - y) ** 2))
for depth in (1, 3, 6, 10, None):
    tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"깊이 {str(depth):>4s}: 잎 {tree.get_n_leaves():4d}개  학습 RMSE {rmse(tree, X_tr, y_tr):5.1f}  시험 RMSE {rmse(tree, X_te, y_te):5.1f} MPa")
```

```output
깊이    1: 잎    2개  학습 RMSE  43.9  시험 RMSE  42.6 MPa
깊이    3: 잎    8개  학습 RMSE  29.2  시험 RMSE  31.0 MPa
깊이    6: 잎   63개  학습 RMSE  19.2  시험 RMSE  25.4 MPa
깊이   10: 잎  489개  학습 RMSE   6.8  시험 RMSE  28.1 MPa
깊이 None: 잎 1000개  학습 RMSE   0.0  시험 RMSE  28.5 MPa
```

깊이 6에서 시험 오차가 가장 작고(25.4 MPa), 그보다 깊으면 학습 오차는 계속 줄지만 시험 오차는 늘어난다. 제한 없이 키운 트리는 잎이 학습 점 수와 같은 1000개이고 학습 오차가 0이다. 잡음까지 외운 것이다. 깊이, 잎의 최소 샘플 수(`min_samples_leaf`), 분할의 최소 개선량은 모두 트리의 복잡도를 정하는 하이퍼파라미터이고 교차검증으로 고른다. 그러나 가장 좋은 단일 트리(25.4 MPa)도 잡음 수준(10 MPa)과는 거리가 멀다. 트리 하나는 **분산이 크다.** 데이터가 조금만 바뀌어도 위쪽 분할이 바뀌고, 그러면 아래 구조 전체가 바뀐다.

::quiz{id="tree-no-scaling"}

## bagging과 랜덤 포레스트

**왜 필요한가.** 분산이 큰 추정량을 여러 개 평균하면 분산이 준다. 같은 데이터에서 트리를 여러 개 만들려면 데이터를 흔들어야 한다. **bagging**(bootstrap aggregating)은 학습 데이터에서 $N$개를 복원추출한 부트스트랩 표본마다 깊은 트리를 하나씩 키우고, 예측을 평균한다.

**분산 감소의 한계.** 분산이 $\sigma^2$이고 서로의 상관계수가 $\rho$인 예측 $B$개를 평균하면 분산은

$$
\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^B \hat f_b\Big) = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2
$$

이다. $B$를 늘리면 둘째 항은 사라지지만 첫째 항 $\rho\sigma^2$는 남는다. 트리들이 같은 강한 특징으로 맨 위를 나누면 서로 비슷해져 $\rho$가 크고, 평균의 이득이 작다. **랜덤 포레스트**(random forest)는 분할마다 특징의 무작위 부분집합(`max_features`)만 후보로 허용해 트리 사이의 상관을 낮춘다(Breiman, 2001). 각 트리는 조금 나빠지지만(편향이 약간 늘지만) 평균의 분산이 더 크게 준다.

**OOB 오차.** 부트스트랩 표본은 복원추출이라, 특정 샘플이 한 표본에 한 번도 뽑히지 않을 확률은 $(1 - 1/N)^N \to e^{-1} \approx 0.368$이다. 각 샘플은 약 37%의 트리에서 학습에 쓰이지 않았으므로, 그 트리들의 예측 평균으로 샘플마다 "보지 않은 데이터에서의 예측"을 만들 수 있다. 이것이 **OOB**(out-of-bag) 예측이고, 별도의 검증 세트 없이 일반화 오차를 추정한다.

```python {exec="browser" session="tree"}
from sklearn.ensemble import RandomForestRegressor

for label, mf in [("bagging (특징 전부)", 1.0), ("랜덤 포레스트 (특징 1/3)", 0.33)]:
    rf = RandomForestRegressor(n_estimators=300, max_features=mf, oob_score=True, random_state=0, n_jobs=1)
    rf.fit(X_tr, y_tr)
    oob_rmse = np.sqrt(np.mean((rf.oob_prediction_ - y_tr) ** 2))
    print(f"{label}: OOB RMSE {oob_rmse:5.1f}  시험 RMSE {rmse(rf, X_te, y_te):5.1f} MPa")
```

```output
bagging (특징 전부): OOB RMSE  20.8  시험 RMSE  19.8 MPa
랜덤 포레스트 (특징 1/3): OOB RMSE  20.7  시험 RMSE  20.0 MPa
```

트리 300개를 평균하자 시험 오차가 단일 트리의 25–28 MPa에서 약 20 MPa로 줄었다. OOB RMSE(20.7–20.8)는 시험 RMSE(19.8–20.0)와 가깝다. OOB 추정은 학습 데이터만으로 얻었다는 점이 중요하다. 이 데이터에서는 특징 전부를 쓰는 bagging과 1/3만 쓰는 랜덤 포레스트가 비슷하다. 강한 특징(Cu 함량)이 하나뿐이라 트리 사이의 상관이 원래 크지 않기 때문이다. 강하게 상관된 특징이 많을수록 `max_features`를 줄이는 효과가 크다. 랜덤 포레스트는 기본 설정에서도 좋은 성능을 내는 경우가 많아, 튜닝 없이 바로 세울 수 있는 baseline으로 특히 쓸모 있다.

:::note[분야 예시]
원격탐사에서 위성 영상의 화소를 토지 피복(숲, 농지, 도시, 수면)으로 분류하는 문제에는 랜덤 포레스트가 오래전부터 널리 쓰인다. 입력은 여러 분광 밴드의 반사율과 지형 정보 같은 표 데이터이고, 밴드마다 척도가 다르며, 학습 라벨은 현장 조사로 얻어 수가 많지 않다. 스케일 처리 없이 바로 쓸 수 있고 OOB 오차로 별도 검증 세트 없이 성능을 가늠할 수 있다는 점이 이런 환경에 맞는다. 다만 이웃 화소들은 서로 비슷해서, 무작위 분할의 OOB·시험 오차는 새 지역에 대한 성능을 과대평가한다. 공간 블록 단위로 분할한다(15.4장).
:::

## gradient boosting

**왜 필요한가.** bagging은 깊은(편향이 작고 분산이 큰) 트리를 평균해 분산을 줄인다. boosting은 반대로 얕은(편향이 크고 분산이 작은) 트리를 **순차적으로 더해** 편향을 줄인다. 각 트리는 지금까지의 모델이 틀린 부분을 고친다.

**함수 공간에서의 gradient descent.** 손실 $\sum_i L(y_i, F(\mathbf{x}_i))$를 함수 $F$에 대해 최소화한다고 하자. 파라미터 대신 학습 점에서의 함수값 $F(\mathbf{x}_i)$를 변수로 보면, 음의 gradient는

$$
r_{im} = -\frac{\partial L(y_i, F)}{\partial F}\bigg|_{F = F_{m-1}(\mathbf{x}_i)}
$$

이다. gradient boosting은 이 음의 gradient $r_{im}$에 회귀 트리 $h_m$을 맞추고, 작은 걸음으로 더한다(Friedman, 2001).

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, h_m(\mathbf{x}).
$$

$\nu \in (0, 1]$은 학습률(shrinkage)이다. 제곱 손실 $L = \frac{1}{2}(y - F)^2$이면 $r_{im} = y_i - F_{m-1}(\mathbf{x}_i)$, 즉 **잔차**다. "잔차에 트리를 맞추고 조금 더한다"를 반복하는 것이 제곱 손실의 gradient boosting이다. 다른 손실(절댓값, Huber, 분위수, 로지스틱)도 음의 gradient만 바꾸면 같은 틀로 다룬다. 10장의 gradient descent를 파라미터 공간이 아니라 함수 공간에서 하는 셈이다. 직접 구현해 본다.

```python {exec="browser" session="tree"}
lr = 0.1                                                   # 학습률(한 걸음의 크기)
F = np.full(len(y_tr), y_tr.mean())                        # 초기 예측: 평균, (N,)
F_te = np.full(len(y_te), y_tr.mean())
for m in range(1, 301):
    h = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr - F)   # 잔차 = 음의 gradient
    F += lr * h.predict(X_tr)
    F_te += lr * h.predict(X_te)
    if m in (1, 10, 100, 300):
        print(f"단계 {m:3d}: 학습 RMSE {np.sqrt(np.mean((F - y_tr) ** 2)):3.0f}  시험 RMSE {np.sqrt(np.mean((F_te - y_te) ** 2)):3.0f} MPa")
```

```output
단계   1: 학습 RMSE  53  시험 RMSE  52 MPa
단계  10: 학습 RMSE  31  시험 RMSE  32 MPa
단계 100: 학습 RMSE   9  시험 RMSE  13 MPa
단계 300: 학습 RMSE   6  시험 RMSE  13 MPa
```

첫 단계에서는 평균에서 크게 움직이지 못하고(학습률 0.1), 단계가 쌓이며 학습·시험 오차가 함께 줄어 100단계에서 시험 RMSE 13 MPa에 이른다. 랜덤 포레스트(20 MPa)보다 훨씬 낫다. 이 데이터의 강도는 효과들의 합에 가까워, 얕은 트리를 더해 가는 boosting의 구조와 잘 맞는다. 100단계 이후에는 학습 오차만 계속 줄고 시험 오차는 거의 그대로다. 트리를 더 더하면 결국 과적합한다.

::quiz{id="boosting-residuals"}

**하이퍼파라미터.** gradient boosting은 랜덤 포레스트보다 튜닝에 민감하다. 핵심은 다음과 같다.

- **학습률과 트리 수.** 학습률을 낮추면 같은 성능에 더 많은 트리가 필요하지만 대개 일반화가 좋아진다. 트리 수는 직접 고르는 대신 검증 오차가 더 이상 줄지 않으면 멈추는 **early stopping**으로 정한다.
- **트리의 크기.** 깊이 3–8 정도의 얕은 트리를 쓴다. 깊이는 표현할 수 있는 상호작용의 차수를 정한다.
- **무작위성.** 각 단계에서 학습 데이터의 일부만 쓰는 subsampling(stochastic gradient boosting)과 특징 부분집합은 분산을 줄인다.
- **정규화.** 잎의 최소 샘플 수, 잎 값에 대한 L2 벌점 등.

::widget{name="tree-boosting-overfit"}

**구현.** 실무에서는 대규모 데이터에 최적화된 라이브러리를 쓴다. XGBoost는 2차 미분 정보와 정규화된 목적함수, 희소 데이터 처리를 갖춘 확장 가능한 구현이다(Chen & Guestrin, 2016). LightGBM은 특징값을 히스토그램 구간으로 묶어 분할 탐색을 빠르게 하고, CatBoost는 범주형 변수 처리를 강조한다. scikit-learn의 `HistGradientBoostingRegressor`도 히스토그램 기반이며 결측값과 범주형 특징을 직접 지원한다. 라이브러리마다 기본값과 이름이 다르므로, 버전과 설정을 기록하고 공식 문서에서 확인한다(집필 시점 기준의 설명이다).

## 표 데이터에서의 위치

**딥러닝 대비 강점.** 45개의 표 데이터셋으로 한 대규모 벤치마크(Grinsztajn 외, 2022)는 중간 크기(약 1만 개)의 표 데이터에서 트리 기반 모델이 많은 하이퍼파라미터 탐색 예산을 준 신경망보다 여전히 앞선다고 보고했다. 저자들은 그 이유로 세 가지 귀납적 편향을 들었다. 트리는 무관한 특징에 강하고, 특징의 축 방향(각 열이 서로 다른 물리량이라는 사실)을 보존하며, 문턱처럼 불규칙한 함수를 쉽게 학습한다. 신경망은 매끄러운 함수를 선호하고, 입력을 회전시켜도 같은 방식으로 학습하므로 열마다 의미가 다르다는 정보를 쓰지 않는다. 같은 데이터에서 세 가지 강한 baseline을 비교한다.

```python {exec="browser" session="tree"}
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "Ridge (16.2장)": make_pipeline(StandardScaler(), RidgeCV()),
    "랜덤 포레스트": RandomForestRegressor(n_estimators=200, max_features=0.33, random_state=0, n_jobs=1),
    "gradient boosting": GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=3,
                                                   n_iter_no_change=20, validation_fraction=0.2, random_state=0),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    print(f"시험 RMSE {rmse(model, X_te, y_te):3.0f} MPa  {name}")
print("early stopping이 트리 2000개 중 500개 전에 멈췄는가:", models["gradient boosting"].n_estimators_ < 500)
```

```output
시험 RMSE  24 MPa  Ridge (16.2장)
시험 RMSE  20 MPa  랜덤 포레스트
시험 RMSE  14 MPa  gradient boosting
early stopping이 트리 2000개 중 500개 전에 멈췄는가: True
```

표준화와 정규화 강도 선택까지 마친 Ridge는 문턱 효과와 시간의 비선형 효과를 표현하지 못해 24 MPa에 머문다. 랜덤 포레스트가 20 MPa, early stopping을 건 gradient boosting이 14 MPa다. 학습 데이터의 20%를 떼어 내 검증 오차가 20단계 동안 개선되지 않으면 멈추게 했고, 트리 2000개 중 500개도 쓰기 전에 멈췄다. 표 데이터 문제에서 새 방법을 제안한다면, 이 gradient boosting이 넘어야 할 기준이다.

**범주형 변수.** 공정 라인, 시료 공급처, 장비 번호 같은 범주형 변수를 다루는 방법은 세 가지다.

- **정수 코드(ordinal encoding).** 이 장의 예처럼 범주를 0, 1, 2…로 둔다. 트리는 코드의 순서를 따라 나누므로, 순서에 의미가 없으면 같은 범주 묶음을 만들기 위해 분할이 여러 번 필요하다. 그래도 트리는 대개 잘 작동한다.
- **원-핫 인코딩.** 범주마다 0/1 열 하나. 선형 모델에는 표준이지만, 범주가 수백 개이면 열이 너무 많아지고 트리의 각 분할이 범주 하나만 떼어 내게 된다.
- **타깃 인코딩.** 범주를 그 범주의 학습 출력 평균으로 바꾼다. 강력하지만 **누수 위험이 크다.** 자기 자신의 출력이 들어간 평균을 특징으로 쓰면 학습 점수가 부풀려진다. 교차검증 fold 안에서만 계산하거나, 순서 기반 통계를 쓰는 구현(CatBoost 등)을 쓴다(15.4장).

범주의 종류가 매우 많은 변수(시료 ID, 로트 번호)는 특히 조심한다. 출력과 무관해도 학습 데이터를 외우는 데 쓰일 수 있고, 다음 절에서 보듯 중요도를 왜곡한다. 그런 변수가 같은 단위(같은 로트의 시편들)를 묶는 역할이라면, 특징이 아니라 group 분할의 단위로 써야 할 수도 있다.

## 특징 중요도

**왜 필요한가.** 트리 앙상블은 수백 개의 트리로 이루어져 직접 읽을 수 없다. 그래서 "어느 특징이 예측에 중요한가"를 요약하는 특징 중요도가 거의 항상 보고된다. 문제는 흔히 쓰는 두 방법이 서로 다른 것을 재고, 그중 하나는 체계적으로 편향되어 있다는 점이다.

**impurity 기반 중요도**(mean decrease in impurity, MDI)는 각 특징으로 분할할 때 줄어든 불순도(회귀에서는 제곱오차)를 모든 트리에서 합한 값이다. scikit-learn의 `feature_importances_`가 이것이다. 계산이 공짜지만 두 가지 결함이 있다.

- **학습 데이터에서 계산한다.** 깊은 트리가 잡음을 외우기 위해 한 분할도 불순도를 줄인 것으로 계산된다.
- **분할 후보가 많은 특징을 선호한다.** 연속 특징이나 범주가 많은 특징은 시도할 수 있는 문턱이 많아서, 우연히 불순도를 줄이는 분할을 찾을 기회가 많다(Strobl 외, 2007).

**permutation importance**는 학습한 모델을 고정하고, **시험 데이터**에서 한 특징의 값만 무작위로 섞었을 때 성능이 얼마나 떨어지는지 잰다. 섞인 특징은 출력과의 관계가 끊어지므로, 모델이 그 특징에 의존했다면 성능이 떨어진다.

```python {exec="browser" session="tree"}
from sklearn.inspection import permutation_importance

rf = models["랜덤 포레스트"]
perm = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=1)
print("특징      impurity 중요도   permutation 중요도(시험 R² 감소)")
for j, name in enumerate(names):
    print(f"{name:8s}  {rf.feature_importances_[j]:.2f}              {round(perm.importances_mean[j], 2) + 0.0:+.2f}")
```

```output
특징      impurity 중요도   permutation 중요도(시험 R² 감소)
temp_c    0.09              +0.10
time_h    0.06              +0.03
cu        0.49              +1.01
mg        0.26              +0.42
line      0.02              +0.00
noise_x   0.04              +0.00
lot_id    0.04              +0.00
```

출력과 **무관한** noise_x와 lot_id의 impurity 중요도(각 0.04)는 실제로 효과가 있는 공정 라인(0.02)의 두 배이고, 시효 시간(0.06)에 가깝다. 깊은 트리들이 이 두 특징으로 학습 데이터의 잡음을 나눈 결과다. 시험 데이터에서 잰 permutation importance는 두 특징 모두 0이다. 섞어도 시험 성능이 변하지 않으므로 모델이 이 특징들에서 일반화되는 정보를 얻지 않았다는 뜻이다. Cu 함량의 permutation importance가 1을 넘는 것은, 이 특징을 섞으면 예측이 평균 baseline보다도 나빠져 $R^2$가 음수로 떨어지기 때문이다. 라인의 효과는 실제로 있지만 잡음에 비해 작아(라인 간 차이 최대 20 MPa, 출력 분산의 약 2%) 이 크기의 데이터에서는 두 방법 모두 거의 0으로 잰다.

::quiz{id="impurity-importance-bias"}

**permutation importance의 한계.** 시험 데이터에서 잰다고 해서 모든 문제가 풀리지는 않는다.

- **상관된 특징.** 두 특징이 같은 정보를 담고 있으면, 하나를 섞어도 모델이 다른 하나에 기대어 성능이 거의 떨어지지 않는다. 둘 다 중요도가 낮게 나온다. 또한 섞은 값과 나머지 특징의 조합은 실제로 존재하지 않는 입력(예: 낮은 온도와 그 온도에서는 불가능한 상 분율)을 만들어, 모델이 학습하지 않은 영역에서 평가하게 된다.
- **모델에 대한 설명이지 세계에 대한 설명이 아니다.** 중요도는 "이 모델이 예측에 이 특징을 얼마나 쓰는가"다. 모델이 쓰는 특징이 원인이라는 보장은 없다. 원인과 상관된 특징, 측정 장비나 배치를 드러내는 특징(누수)도 높은 중요도를 받는다.

따라서 중요도는 가설을 만드는 도구이지 인과의 증거가 아니다. 어떤 공정 변수를 바꾸면 강도가 오르는지 알려면 그 변수를 실제로 바꾸는 실험이나 인과추론(16.8장)이 필요하다.

:::note[분야 예시]
임상 표 데이터로 질병 위험을 예측한 모델에서 "검사 X의 중요도가 가장 높다"는 결과는 흔히 "X가 질병의 원인이다"로 오해된다. 그러나 의사가 이미 질병을 의심한 환자에게만 검사 X를 처방한다면, "검사 X를 했는가" 또는 그 결과값은 질병의 원인이 아니라 의사의 판단을 드러내는 신호다. 모델은 이 신호를 잘 쓰고 중요도도 높게 나오지만, 검사를 늘린다고 질병이 생기지는 않는다. 같은 구조가 공학 데이터의 "추가 검사를 받은 시편" 같은 특징에서도 나타난다.
:::

## 실습

표 형식 공학 데이터에서 gradient boosting을 튜닝하고 신경망과 공정하게 비교한다. 코드는 `labs/classical-ml/trees-and-ensembles/tabular_benchmark.py`다. 앞의 합금 데이터(학습 2000개, 시험 2000개)에서 gradient boosting과 MLP(scikit-learn의 `MLPRegressor`)에 **같은 예산**을 준다. 둘 다 학습 데이터 안의 3-fold 교차검증으로 무작위 탐색 설정 15개를 시도하고, 가장 좋은 설정을 학습 데이터 전체로 다시 학습한 뒤 시험 데이터에서 한 번만 평가한다. MLP에는 입력과 출력의 표준화를 넣어 신경망에 불리하지 않게 했다. 두 모델의 차이는 시험 샘플별 제곱오차의 짝지은 부트스트랩으로 판정한다(16.1장). 집필 시점의 실행 결과는 다음과 같다.

```output
Ridge (튜닝: RidgeCV)         시험 RMSE  23.2 MPa
gradient boosting  설정 15개 탐색  CV RMSE  12.3  시험 RMSE  12.2 MPa  (   77 s)
    고른 설정: learning_rate=0.0647, max_depth=2, min_samples_leaf=5, subsample=0.7
MLP                설정 15개 탐색  CV RMSE  17.0  시험 RMSE  16.6 MPa  (   42 s)
    고른 설정: alpha=8.35e-05, hidden_layer_sizes=(32,), learning_rate_init=0.00846
MSE 차이 (MLP − GB) +127.0 MPa², 95% 부트스트랩 CI [+111.9, +141.4]
```

괄호 안의 시간은 BLAS·OpenMP 스레드 하나로 실행한 집필 시점 값이고 기계마다 다르다. 결과는 세 가지로 읽는다.

- **판정.** 같은 예산에서 gradient boosting(12.2 MPa)이 MLP(16.6 MPa)보다 낫고, 짝지은 MSE 차이의 95% 구간 [+111.9, +141.4] MPa²가 0을 포함하지 않는다. 잡음의 표준편차가 10 MPa이므로 gradient boosting은 잡음 수준에 가깝다. 정규화 선형 모델(23.2 MPa)은 둘 모두에 크게 진다.
- **교차검증 점수와 시험 점수.** 두 모델 모두 CV RMSE와 시험 RMSE가 가깝다(12.3 대 12.2, 17.0 대 16.6). 탐색에 쓴 CV 점수가 크게 낙관적이지 않았다는 뜻이지만, 보고할 숫자는 탐색에 쓰지 않은 시험 점수다.
- **범위.** 이 결과는 문턱 효과가 섞인 가상 데이터 하나에서 얻은 것이다. "boosting이 신경망보다 낫다"는 일반 결론이 아니라, 이 문제에서 신경망을 제안하려면 넘어야 할 기준이 무엇인지를 보여 준다. 결론의 범위를 넓히려면 여러 데이터셋에서의 비교(12.5장)가 필요하다.

다음을 직접 해 본다.

1. 탐색 예산을 설정 5개와 30개로 바꿔 두 모델의 시험 RMSE가 예산에 따라 어떻게 변하는지 기록한다. 예산이 적을 때 어느 모델이 더 손해를 보는가.
2. 학습 데이터를 500개로 줄여 같은 비교를 반복한다. 데이터 크기에 따라 결론이 바뀌는가.
3. lot_id를 특징에서 빼고 다시 학습해 시험 성능이 변하는지 확인한다. impurity 중요도만 보고 lot_id를 "중요한 특징"으로 남겨 두었다면 무엇을 놓쳤을지 적는다.

이어서 두 연습문제를 푼다. 첫 문제는 회귀 트리의 최적 분할을 NumPy로 찾고, 두 번째 문제는 permutation importance를 직접 구현한다.

::exercise{id="best-split"}

::exercise{id="permutation-importance"}

## 실패 사례

- **impurity 기반 중요도를 인과적 중요도로 해석하는 것.** impurity 중요도는 학습 데이터에서 계산되고 값의 종류가 많은 특징으로 편향된다. 무관한 로트 번호가 실제 공정 변수보다 중요하게 나올 수 있다(앞의 실험). 시험 데이터에서 permutation importance를 계산하고, 그것도 "모델이 쓰는 정보"일 뿐 원인이 아니라는 점을 명시한다.
- **기본 설정의 boosting과 튜닝한 신경망을 비교하는 것.** gradient boosting은 학습률·트리 크기·early stopping에 민감하다. 같은 탐색 예산을 준다(16.1장).
- **트리 모델로 외삽하는 것.** 학습 범위 밖에서 트리 앙상블은 상수를 예측한다. 더 높은 온도, 더 긴 시간, 새 조성으로의 예측이 목적이면 외삽을 따로 검증하고, 필요하면 선형 항이나 물리식과 결합한다.
- **타깃 인코딩을 전체 데이터로 계산하는 것.** 시험 샘플의 출력이 특징에 섞여 들어간다. fold 안에서만 계산한다.
- **같은 로트·같은 셀의 샘플이 학습과 시험에 섞인 채로 OOB 오차를 믿는 것.** OOB는 샘플 단위의 무작위 분할과 같다. 독립 단위가 따로 있으면 OOB도 낙관적이다.

## 통과 기준

- 표 데이터 문제에서 early stopping과 교차검증으로 튜닝한 gradient boosting baseline을 제시하고, 같은 예산의 다른 모델(정규화 선형 모델, 랜덤 포레스트, 신경망)과 짝지은 비교로 차이를 보고한다.
- 같은 모델에서 impurity 중요도와 시험 데이터의 permutation importance를 함께 계산하고, 두 결과가 다른 특징에 대해 그 이유를 설명한다.
- 연습문제 두 개(`best-split`, `permutation-importance`)의 테스트를 통과한다.

## 참고 문헌

- Breiman (2001). Random forests. *Machine Learning*, 45(1), 5–32. — 무작위 특징 부분집합으로 트리 사이의 상관을 낮추는 랜덤 포레스트와 OOB 추정
- Friedman (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. — 함수 공간의 steepest descent로서의 gradient boosting
- Chen, Guestrin (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. — 확장 가능한 tree boosting 구현
- Strobl, Boulesteix, Zeileis, Hothorn (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8, 25. — 측정 척도와 범주 수에 따른 랜덤 포레스트 중요도의 편향
- Grinsztajn, Oyallon, Varoquaux (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*. — 45개 표 데이터셋에서 트리 기반 모델과 신경망의 비교와 귀납적 편향 분석
