---
id: classical-ml/model-selection
title: "모델 선택과 평가"
title_en: "Model Selection and Evaluation"
tier: core
status: draft
prerequisites: [classical-ml/baselines, statistical-inference/comparing-models]
---

# 모델 선택과 평가

모델을 만드는 일의 절반은 평가다. 어떤 데이터로 검증하는가, 어떤 숫자로 성능을 재는가, 여러 후보 중 하나를 고른 뒤 그 성능을 어떻게 보고하는가. 이 세 가지 중 하나라도 문제 구조와 맞지 않으면, 모델 자체가 아무리 좋아도 보고된 숫자는 실제 성능을 말해 주지 않는다.

12.5장은 이미 얻은 점수들을 통계적으로 비교하는 법을 다뤘다. 이 장은 그 앞 단계, 즉 점수를 **만드는** 절차를 다룬다. 교차검증을 데이터의 독립 단위에 맞추고, 연구 질문에 맞는 metric과 threshold를 고르고, 하이퍼파라미터 탐색에 쓴 점수와 최종 보고에 쓰는 점수를 분리한다.

## 학습 목표

- 교차검증을 문제 구조에 맞게 사용한다
- 적절한 metric을 고른다
- 모델 선택과 최종 보고를 분리한다

## 교차검증

**왜 필요한가.** 데이터를 학습·시험으로 한 번 나누면 시험 점수가 그 분할의 운에 좌우되고, 데이터가 적으면 시험에 떼어 둘 여유도 없다. 교차검증(cross-validation, CV)은 데이터를 $k$개 fold로 나누어 각 fold를 한 번씩 시험에 쓰고 점수를 평균한다. 모든 샘플이 한 번씩 시험에 쓰이므로 데이터를 효율적으로 쓴다.

**정의.** fold 배정이 무엇을 시험하는지를 정한다.

| 방식 | fold를 나누는 단위 | 추정하는 것 |
|---|---|---|
| $k$-fold | 샘플 (무작위) | 같은 분포에서 새로 뽑은 **샘플**에 대한 성능 |
| stratified $k$-fold | 샘플, 클래스 비율 유지 | 위와 같음. 희귀 클래스가 fold마다 고르게 들어감 |
| group $k$-fold | 그룹(셀·환자·장비·실험 배치) | 처음 보는 **그룹**에 대한 성능 |
| time-series CV | 시간 순서 (과거로 학습, 미래로 시험) | **미래** 시점에 대한 성능 (16.7장) |

$k$-fold는 샘플이 서로 독립이라고 가정한다. 같은 셀에서 여러 번 측정했거나 같은 환자에게서 여러 영상을 얻었다면, 같은 그룹의 샘플이 학습과 시험에 나뉘어 들어가 모델이 그룹의 "얼굴"을 외워 시험을 맞힐 수 있다. 이것이 15.4장의 누수(leakage)이고, 해결책은 **연구가 일반화하려는 단위로 fold를 나누는 것**이다.

배터리 셀 30개에서 셀마다 20번 측정한 데이터로 확인한다. 목표값에는 운전 조건으로 설명되지 않는 셀 고유의 편차가 있고, 특징에는 셀을 식별할 수 있는 값(예: 초기 저항)이 들어 있다. 초기 저항은 목표값과 아무 관계가 없지만 셀마다 고유하다. $k$-NN 회귀로 샘플 단위 $k$-fold, 셀 단위 group $k$-fold, 그리고 처음 보는 셀 200개(독립 데이터)에서의 $R^2$를 비교한다.

```python {exec="browser" session="msel"}
import warnings
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def make_cells(rng, n_cells, per_cell=20):                     # 셀마다 측정 20개
    g = np.repeat(np.arange(n_cells), per_cell)                # (N,) 셀 번호
    x = rng.uniform(0, 1, (len(g), 3))                          # (N, 3) 운전 조건
    tag = rng.normal(0, 1, (n_cells, 2))[g]                    # 셀을 식별하는 특징(예: 초기 저항) (N, 2)
    offset = rng.normal(0, 0.5, n_cells)[g]                    # 특징으로 설명되지 않는 셀 편차
    y = np.sin(3 * x[:, 0]) + x[:, 1] + offset + rng.normal(0, 0.1, len(g))
    return np.c_[x, tag], y, g

rng = np.random.default_rng(0)
X, y, g = make_cells(rng, n_cells=30)
X_new, y_new, _ = make_cells(rng, n_cells=200)                 # 처음 보는 셀 200개 (독립 데이터)
group_folds = [(np.flatnonzero(g % 5 != k), np.flatnonzero(g % 5 == k)) for k in range(5)]   # 셀 단위 5-fold
for name, cols in [("조건 + 셀 식별 특징", slice(None)), ("조건만", slice(0, 3))]:
    knn = KNeighborsRegressor(n_neighbors=3)
    r2_kf = cross_val_score(knn, X[:, cols], y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    r2_gkf = cross_val_score(knn, X[:, cols], y, cv=group_folds).mean()
    r2_new = knn.fit(X[:, cols], y).score(X_new[:, cols], y_new)
    print(f"{name:12s} R²  k-fold {r2_kf:5.2f}   group k-fold {r2_gkf:5.2f}   새 셀 {r2_new:5.2f}")
```

```output
조건 + 셀 식별 특징 R²  k-fold  0.63   group k-fold -0.05   새 셀 -0.30
조건만          R²  k-fold  0.26   group k-fold  0.20   새 셀  0.22
```

셀 식별 특징을 넣은 모델은 샘플 단위 $k$-fold에서 $R^2 = 0.63$으로 좋아 보인다. 그러나 셀 단위로 나누면 $-0.05$이고, 처음 보는 셀에서는 $-0.30$으로 평균만 예측하는 것보다 나쁘다. 학습 셀의 측정이 시험 fold에 섞여 있으면, 초기 저항이 같은 이웃(같은 셀의 다른 측정)을 찾아 그 셀의 편차를 그대로 가져오기 때문이다. 운전 조건만 쓰는 모델은 세 숫자가 비슷하고(0.26, 0.20, 0.22), 처음 보는 셀에서는 이쪽이 낫다. **샘플 단위 CV로 모델을 골랐다면 틀린 모델을 골랐을 것이다.** 분할 방식은 성능 숫자만이 아니라 모델 선택을 바꾼다.

라이브러리에서는 `GroupKFold`처럼 그룹 번호를 받는 분할기를 쓴다. 위 코드는 버전에 따라 fold 배정이 달라지지 않도록 셀 번호로 fold를 직접 정했다.

::quiz{id="kfold-vs-group"}

**nested CV의 필요성.** CV 점수로 하이퍼파라미터나 모델을 **고르면**, 고른 후보의 CV 점수는 더 이상 공정한 성능 추정이 아니다. 여러 후보 중 가장 높은 점수를 골랐으므로, 그 점수에는 "그 fold들에서 운이 좋았던 정도"가 섞여 있다. 12.4장의 다중 비교와 같은 선택 편향이다. 해결책은 선택 절차 전체를 바깥 CV로 한 번 더 감싸는 **nested CV**다.

1. 바깥 CV가 데이터를 $k_\mathrm{out}$개 fold로 나눈다.
2. 바깥 fold마다, 학습 부분 안에서만 안쪽 CV로 하이퍼파라미터를 고르고 그 학습 부분 전체로 다시 학습한다.
3. 바깥 시험 fold에서 평가한다. 바깥 시험 fold는 선택에 한 번도 쓰이지 않았다.

nested CV가 추정하는 것은 특정 하이퍼파라미터 설정의 성능이 아니라 **"이 탐색 절차로 모델을 고르면 얼마나 좋은 모델이 나오는가"**다. 바깥 fold마다 다른 설정이 골라질 수 있고, 그것이 정상이다. 최종 모델은 전체 데이터로 같은 탐색을 한 번 더 해서 만든다. 데이터가 충분하다면 nested CV 대신 학습·검증·시험의 세 부분으로 나누고 시험 세트를 마지막에 한 번만 쓰는 방법도 같은 원리다(Varma, Simon, 2006; Cawley, Talbot, 2010).

::widget{name="split-leakage"}

## 회귀 metric

**왜 필요한가.** metric은 "무엇을 틀리면 얼마나 나쁜가"를 숫자로 정한 것이다. 같은 예측이라도 metric에 따라 좋아 보이기도 나빠 보이기도 하고, 단위와 해석이 다르다. 12.3장에서 본 것처럼 손실함수가 고르는 대표값(평균·중앙값)도 metric마다 다르다.

**정의.** 참값 $y_i$, 예측 $\hat{y}_i$, 샘플 $N$개에 대해

$$
\mathrm{MAE} = \frac{1}{N}\sum_i |y_i - \hat{y}_i|, \qquad
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}, \qquad
\mathrm{MAPE} = \frac{100\%}{N}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|,
$$

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}.
$$

- **MAE·RMSE**는 목표값과 같은 단위를 가진다(°C, mAh). RMSE는 큰 오차에 더 큰 벌점을 주므로, 가끔 크게 틀리는 모델에서 MAE보다 훨씬 커진다. 둘의 비가 크면 오차 분포의 꼬리가 길다는 신호다.
- **MAPE**는 단위가 없어 서로 다른 크기의 대상을 비교할 때 편해 보이지만, 참값이 0에 가까우면 폭발하고, 0을 어디에 두느냐(°C냐 K냐)에 따라 값이 바뀐다. 상대오차는 비율 척도(절대 0이 의미 있는 양: 질량, 용량, 농도)에서만 의미가 있다.
- **$R^2$**는 "평균만 예측하는 baseline보다 제곱오차를 얼마나 줄였는가"다. 분모가 참값의 분산이므로, 같은 오차라도 참값의 범위가 넓은 데이터에서는 높게, 좁은 데이터에서는 낮게 나온다. 시험 데이터에서는 음수가 될 수 있다(앞 셀의 $-0.30$).

같은 오차(모든 점에서 0.5 °C)를 단위와 범위만 바꿔 평가한다.

```python {exec="browser" session="msel"}
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

temp_c = np.array([0.2, 1.0, 5.0, 12.0, 20.0, 30.0])           # 참 온도 [°C]
pred_c = temp_c + 0.5                                          # 모든 점에서 0.5 °C 높게 예측
for unit, yt, yp in [("°C", temp_c, pred_c), ("K", temp_c + 273.15, pred_c + 273.15)]:
    print(f"단위 {unit:2s}  MAE {mean_absolute_error(yt, yp):.2f}   MAPE {100 * mean_absolute_percentage_error(yt, yp):6.2f}%")

narrow_c = np.linspace(19, 21, 50)                             # 좁은 범위의 참값
wide_c = np.linspace(0, 40, 50)                                # 넓은 범위의 참값
noise = np.where(np.arange(50) % 2 == 0, 0.5, -0.5)            # 같은 크기(±0.5 °C)의 오차
for name, yt in [("19–21 °C", narrow_c), ("0–40 °C", wide_c)]:
    print(f"범위 {name:9s} MAE {mean_absolute_error(yt, yt + noise):.2f}   R² {r2_score(yt, yt + noise):.3f}")
```

```output
단위 °C  MAE 0.50   MAPE  53.06%
단위 K   MAE 0.50   MAPE   0.18%
범위 19–21 °C  MAE 0.50   R² 0.279
범위 0–40 °C   MAE 0.50   R² 0.998
```

MAE는 단위를 바꿔도 0.50 °C(= 0.50 K)로 같지만, MAPE는 °C에서 53%, K에서 0.18%다. 0.2 °C에서 0.7 °C로 예측한 한 점이 °C 기준 상대오차 250%를 만들기 때문이다. 온도처럼 원점이 임의인 양에 MAPE를 쓰면 숫자가 단위 선택을 반영할 뿐이다. $R^2$는 MAE가 같은데도 참값 범위가 19–21 °C이면 0.279, 0–40 °C이면 0.998이다. $R^2$가 낮다고 오차가 큰 것이 아니고, 높다고 오차가 작은 것도 아니다. 그래서 회귀 결과는 **단위가 있는 오차(MAE·RMSE)**를 먼저 보고하고, 그 오차가 응용에서 허용되는 수준인지(예: 센서 정확도 ±0.5 °C와 비교) 함께 쓴다. 조건에 따라 오차가 다르면 평균 하나보다 조건별 오차를 보고한다(뒤의 "진단" 절).

::quiz{id="mape-near-zero"}

## 분류 metric

**왜 필요한가.** 분류에서 가장 직관적인 metric인 정확도(accuracy)는 클래스가 불균형하면 거의 아무 정보도 주지 않는다. 불량률 1%인 공정에서 "모두 정상"이라고 답하는 분류기의 정확도는 99%다.

**정의.** 양성 판정과 실제 양성으로 혼동행렬(confusion matrix)의 네 칸 TP·FP·FN·TN을 만든다.

$$
\text{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}, \qquad
\text{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.
$$

precision은 "양성이라고 한 것 중 진짜 양성의 비율", recall은 "진짜 양성 중 찾아낸 비율"이다. 둘 다 threshold에 따라 바뀐다. threshold를 하나로 정하지 않고 분류기의 **순위 매기는 능력**을 보는 metric이 두 가지 있다.

- **ROC-AUC**: threshold를 바꾸며 그린 (FPR, TPR) 곡선 아래 넓이. 무작위로 고른 양성 하나가 무작위로 고른 음성 하나보다 높은 점수를 받을 확률과 같다. 클래스 비율에 거의 영향을 받지 않는다.
- **PR-AUC**(average precision): threshold를 바꾸며 그린 (recall, precision) 곡선 아래 넓이. 무작위 분류기의 값이 양성 비율과 같아서, 양성이 드물면 낮아진다. 희귀한 양성을 찾는 문제에서는 이것이 실제 사용 상황(경보 중 진짜 비율)을 더 잘 반영한다(Saito, Rehmsmeier, 2015).

같은 점수 분포(양성은 음성보다 평균이 2만큼 높음)를 가진 분류기를 양성 비율 50%와 1%에서 평가한다. 확률은 Bayes 규칙으로 계산한 보정된(calibrated) 확률이고, 판정 threshold는 0.5다.

```python {exec="browser" session="msel"}
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

def make_scores(rng, n, prior):                                # 양성 비율 prior, 점수 분포는 고정
    yy = (rng.uniform(size=n) < prior).astype(int)             # (n,)
    s = rng.normal(0, 1, n) + 2.0 * yy                         # 양성은 평균 2만큼 높다
    lr = norm.pdf(s, 2, 1) / norm.pdf(s, 0, 1)                 # 우도비
    p = prior * lr / (prior * lr + 1 - prior)                  # 보정된(calibrated) 양성 확률
    return yy, s, p

for prior in (0.5, 0.01):
    yy, s, p = make_scores(rng, 20000, prior)
    pred = (p >= 0.5).astype(int)
    print(f"양성 {prior:4.0%}  accuracy: 전부 음성 {accuracy_score(yy, 0 * yy):.3f}, 분류기 {accuracy_score(yy, pred):.3f}")
    print(f"           precision {precision_score(yy, pred):.2f}  recall {recall_score(yy, pred):.2f}  F1 {f1_score(yy, pred):.2f}"
          f"   ROC-AUC {roc_auc_score(yy, s):.3f}  PR-AUC {average_precision_score(yy, s):.3f}")
```

```output
양성  50%  accuracy: 전부 음성 0.500, 분류기 0.843
           precision 0.85  recall 0.84  F1 0.84   ROC-AUC 0.922  PR-AUC 0.923
양성   1%  accuracy: 전부 음성 0.991, 분류기 0.992
           precision 0.74  recall 0.14  F1 0.23   ROC-AUC 0.932  PR-AUC 0.351
```

양성 1%에서 "전부 음성"의 정확도는 0.991이고, 분류기의 정확도 0.992와 거의 같다. 정확도만 보면 분류기가 쓸모없어 보이지만, ROC-AUC는 두 경우 모두 0.92–0.93이다. 점수 분포가 같으므로 순위 매기는 능력은 같다(두 값의 차이는 양성이 약 200개뿐인 데서 오는 표본 변동이다). 반면 PR-AUC는 0.923에서 0.351로 떨어진다. 양성이 드물면 높은 점수를 받은 샘플 중에도 음성이 많이 섞이기 때문이다. threshold 0.5에서 recall은 0.14로, 불량의 86%를 놓친다. **어떤 metric도 단독으로 충분하지 않다.** 불균형 문제에서는 양성 비율을 함께 쓰고, ROC-AUC와 PR-AUC, 그리고 실제 운용할 threshold에서의 precision·recall을 함께 보고한다.

::quiz{id="accuracy-imbalanced"}

:::note[분야 예시]
천문학의 과도 현상(transient) 탐지에서는 망원경 영상의 차분에서 나온 후보 수천 개 중 실제 천체 현상은 소수이고 대부분은 장비나 처리 과정의 허상이다. 분류기의 정확도는 거의 항상 높게 나오므로, 연구자들은 "진짜 현상을 몇 % 놓치는가"(recall)와 "사람이 확인할 후보 중 허상이 얼마나 섞이는가"(precision)를 함께 보고 운용 threshold를 정한다. 놓친 현상은 다시 관측할 수 없는 경우가 많아 recall 쪽에 무게를 둔다.
:::

## threshold 선택

**왜 필요한가.** 0.5는 기본값일 뿐 근거가 있는 값이 아니다. 불량을 놓치는 비용과 정상 제품을 재검사하는 비용이 다르면, 최적의 threshold도 0.5가 아니다.

**정의.** 샘플이 양성일 확률이 $p$일 때, 양성으로 판정하면 기대 비용은 $(1-p)\,c_\mathrm{FP}$(음성인데 양성이라 함), 음성으로 판정하면 $p\,c_\mathrm{FN}$(양성을 놓침)이다. 기대 비용이 작은 쪽을 고르면(Elkan, 2001), 양성으로 판정하는 조건은

$$
p\,c_\mathrm{FN} > (1-p)\,c_\mathrm{FP} \iff p > \frac{c_\mathrm{FP}}{c_\mathrm{FP} + c_\mathrm{FN}}
$$

이다. 맞힌 경우의 비용은 0으로 두었다. 불량을 놓치는 비용이 재검사 비용의 50배이면 threshold는 $1/51 \approx 0.020$이다.

**calibration과의 관계.** 이 식은 $p$가 **보정된 확률**일 때만 맞다. 즉 모델이 0.02라고 말한 샘플들 중 실제로 약 2%가 양성이어야 한다. 많은 분류기의 점수는 순위는 맞아도 확률로서는 틀리다(예: 클래스 비율을 맞춰 재표본화해 학습한 모델은 양성 확률을 크게 과대평가한다). 그럴 때는 두 가지 방법이 있다. 점수를 보정한 뒤(calibration, 21.2장) 이론식을 쓰거나, **검증 데이터에서 직접 비용을 최소화하는 threshold**를 고른다. 후자는 보정을 가정하지 않지만 검증 데이터가 충분해야 한다.

```python {exec="browser" session="msel"}
c_fn, c_fp = 50.0, 1.0                                         # 불량을 놓친 비용, 정상을 재검사한 비용
y_val, _, p_val = make_scores(rng, 20000, 0.01)                # 검증 데이터
y_te, _, p_te = make_scores(rng, 20000, 0.01)                  # 시험 데이터

def cost_per_1000(yy, p, thr):
    pred = p >= thr
    return 1000 * (c_fn * np.sum(~pred & (yy == 1)) + c_fp * np.sum(pred & (yy == 0))) / len(yy)

grid = np.linspace(0.001, 0.5, 500)
thr_val = grid[np.argmin([cost_per_1000(y_val, p_val, t) for t in grid])]
for name, thr in [("0.5 (기본값)", 0.5), ("이론값 c_fp/(c_fp+c_fn)", c_fp / (c_fp + c_fn)), ("검증 데이터에서 고름", thr_val)]:
    print(f"{name:24s} threshold {thr:.3f}   시험 데이터 비용 {cost_per_1000(y_te, p_te, thr):6.1f} / 1000개")
```

```output
0.5 (기본값)                threshold 0.500   시험 데이터 비용  425.8 / 1000개
이론값 c_fp/(c_fp+c_fn)     threshold 0.020   시험 데이터 비용  217.8 / 1000개
검증 데이터에서 고름              threshold 0.017   시험 데이터 비용  223.1 / 1000개
```

시험 데이터 1000개당 비용은 threshold 0.5에서 425.8, 이론값 0.020에서 217.8로 절반 가까이 준다. 검증 데이터에서 고른 0.017도 비슷한 비용(223.1)을 낸다. 이 예의 확률은 정확히 보정되어 있으므로 두 방법이 일치한다. threshold는 **검증 데이터에서** 고르고 시험 데이터에서는 평가만 했다는 점에 주의한다. 시험 데이터로 threshold를 고르면 그것도 선택 편향이다.

::quiz{id="cost-threshold-value"}

## 하이퍼파라미터 탐색

**왜 필요한가.** 정규화 강도, 트리 깊이, 커널 폭 같은 하이퍼파라미터는 학습으로 정해지지 않고 검증 점수로 골라야 한다. 어떤 방식으로 후보를 정하는가와, 고른 점수를 어떻게 다루는가가 문제다.

**탐색 방식.**

- **grid search**: 각 하이퍼파라미터의 후보 값을 정하고 모든 조합을 시험한다. 차원이 늘면 조합 수가 곱으로 늘고, 중요한 하이퍼파라미터가 몇 개뿐이면 나머지 축의 시도가 낭비된다.
- **random search**: 범위 안에서 무작위로 조합을 뽑는다. 같은 예산에서 중요한 축마다 더 많은 서로 다른 값을 시험하므로, 중요한 하이퍼파라미터가 소수일 때 grid보다 효율적이다(Bergstra, Bengio, 2012).
- **베이지안 최적화**: 지금까지의 (설정, 점수)로 점수의 대리 모델(흔히 Gaussian process, 13.7장)을 세우고, 다음에 시험할 설정을 그 모델이 유망하다고 보는 곳에서 고른다(Snoek 외, 2012). 한 번 학습이 비쌀 때 시도 횟수를 줄인다.

**탐색 예산의 공정성.** 새 방법은 수백 번 튜닝하고 baseline은 기본값으로 두면, 차이의 일부는 방법이 아니라 튜닝 노력이다(16.1장). 비교하는 방법마다 같은 탐색 방식과 같은 예산(시도 횟수나 계산 시간)을 쓰고, 그 예산을 보고한다.

**선택 편향.** 탐색은 많은 후보 중 CV 점수가 가장 높은 것을 고른다. 후보가 많고 데이터가 적을수록, 가장 높은 점수는 실제 성능보다 운의 몫이 커진다. 극단적인 경우로 확인한다. label이 특징과 **무관한** 데이터(참 정확도 0.5)에서 특징 50개 중 하나를 고르는 "하이퍼파라미터"를 5-fold CV로 탐색한다. 설정마다의 모델은 그 특징 하나로 두 클래스의 평균 중 가까운 쪽을 예측하는 분류기다. 데이터셋 200개에서 탐색 최고 점수와 nested CV 추정값을 비교한다.

```python {exec="browser" session="msel"}
def feature_acc(Xtr, ytr, Xte, yte):                           # 특징 하나씩 쓰는 최근접 평균 분류기들
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)      # (D,), (D,)
    pred = np.abs(Xte - m1) < np.abs(Xte - m0)                 # (n_te, D) 설정(특징)마다 예측
    return (pred == yte[:, None]).mean(0)                      # (D,) 설정마다 정확도

def cv_acc(Xd, yd, k=5):                                       # 설정마다 k-fold CV 정확도, (D,)
    fold = np.arange(len(yd)) % k
    return np.mean([feature_acc(Xd[fold != f], yd[fold != f], Xd[fold == f], yd[fold == f])
                    for f in range(k)], axis=0)

flat, nested = [], []
for d in range(200):                                           # label이 특징과 무관한 데이터셋 200개
    r = np.random.default_rng(100 + d)
    Xn, yn = r.normal(size=(40, 50)), r.permutation(np.repeat([0, 1], 20))   # (40, 50), 참 정확도 0.5
    flat.append(cv_acc(Xn, yn).max())                          # 설정 50개 중 CV 점수 최고값
    outer = np.arange(40) % 5
    nested.append(np.mean([feature_acc(Xn[outer != f], yn[outer != f], Xn[outer == f], yn[outer == f])[
        np.argmax(cv_acc(Xn[outer != f], yn[outer != f]))]    # 안쪽 CV로 고른 설정을 바깥 fold에서 평가
        for f in range(5)]))
print(f"탐색에 쓴 CV 최고 점수: 평균 {np.mean(flat):.3f}   0.6 이상인 데이터셋 {np.mean(np.array(flat) >= 0.6):.0%}")
print(f"nested CV 추정값:       평균 {np.mean(nested):.3f}   0.6 이상인 데이터셋 {np.mean(np.array(nested) >= 0.6):.0%}")
```

```output
탐색에 쓴 CV 최고 점수: 평균 0.695   0.6 이상인 데이터셋 100%
nested CV 추정값:       평균 0.490   0.6 이상인 데이터셋 17%
```

신호가 전혀 없는데도 탐색에 쓴 CV 최고 점수는 평균 0.695이고, 200개 데이터셋 **모두**에서 0.6을 넘는다. 이 숫자를 성능으로 보고했다면 모든 데이터셋에서 "의미 있는 예측"을 주장했을 것이다. nested CV는 평균 0.490으로 참값 0.5 근처를 추정한다. 개별 데이터셋에서는 nested 추정값도 흔들려 17%에서 0.6을 넘는다. 데이터가 40개뿐이라 어떤 추정이든 분산이 크다는 뜻이고, 12.5장에서 본 대로 점수 하나가 아니라 그 불확실성을 함께 보고해야 한다.

::quiz{id="best-score-is-optimistic"}

현실의 탐색은 이 예보다 덜 극단적이지만 같은 방향으로 편향된다. 편향의 크기는 후보 수, 데이터 크기, 점수의 분산에 따라 달라지므로 미리 알 수 없다. Varma와 Simon(2006)은 신호가 없는 데이터에서 CV로 튜닝한 분류기의 CV 오차가 독립 데이터에서의 오차보다 크게 낙관적이고, nested CV는 거의 편향이 없음을 시뮬레이션으로 보였다. Cawley와 Talbot(2010)은 모델 선택 기준의 **분산**이 선택 과정의 과적합을 낳으며, 그로 인한 성능 손실과 평가 편향이 학습 알고리즘 사이의 차이와 비슷한 크기일 수 있다고 보고했다.

::quiz{id="nested-cv-estimates-procedure"}

## 진단

평균 점수 하나는 모델이 **왜** 그 성능인지, 어디서 실패하는지 말해 주지 않는다. 다음 도구로 성능을 분해한다.

- **학습 곡선**(learning curve): 학습 데이터 크기를 늘려 가며 학습 점수와 검증 점수를 그린다. 두 곡선 사이가 크게 벌어져 있으면 분산(과적합)이 문제이고 데이터를 늘리면 나아질 가능성이 있다. 두 곡선이 함께 낮은 값에 수렴하면 편향이 문제이고 데이터를 늘려도 나아지지 않는다. 더 유연한 모델이나 더 좋은 특징이 필요하다.
- **검증 곡선**(validation curve): 하이퍼파라미터 하나(정규화 강도, 깊이)를 바꿔 가며 두 점수를 그린다. 최적점 주변이 평평하면 선택이 결과에 크게 중요하지 않고, 뾰족하면 선택이 불안정하다는 뜻이다.
- **잔차 분석**: 회귀에서 잔차 $y_i - \hat{y}_i$를 예측값, 각 특징, 시간에 대해 그린다. 패턴이 보이면 모델이 놓친 구조가 있다(12.6장의 잔차 진단).
- **조건별 오차**: 온도 구간별, 장비별, 재료 계열별로 오차를 나누어 본다. 평균 오차가 좋아도 특정 조건에서 체계적으로 실패할 수 있고, 과학 연구에서는 그 조건이 가장 흥미로운 경우(극한 조건, 희귀 재료)인 경우가 많다.

:::note[분야 예시]
기후 모델의 통계적 후처리(downscaling)를 평가할 때는 전체 RMSE 하나보다 계절별·지역별 오차와 극값(상위 1% 강수)의 오차를 따로 본다. 평균적인 날의 오차가 작아도 홍수를 일으키는 극한 강수를 체계적으로 과소평가하면, 응용(방재 설계)에서는 쓸 수 없는 모델이다. 어떤 조건에서의 오차가 응용에 중요한지 평가 설계 단계에서 정한다.
:::

## 실습

같은 모델을 nested CV와 단순 CV로 평가해 성능 추정의 낙관 편향을 측정한다. 코드는 `labs/classical-ml/model-selection/nested_vs_flat.py`다. 특징 10개 중 2개에만 비선형(원 모양) 신호가 있는 이진 분류 데이터를 크기 $n$으로 30번 뽑고, RBF 커널 SVM의 하이퍼파라미터 격자 25개($C$ 5개 × $\gamma$ 5개)를 탐색한다. 데이터셋마다 (1) 탐색에 쓴 CV 최고 점수(`best_score_`), (2) nested CV 추정값, (3) 전체 데이터로 탐색해 고른 최종 모델을 같은 분포의 독립 데이터 2만 개에서 잰 참 정확도를 구해, (1)과 (2)가 (3)을 얼마나 잘 맞히는지 본다. 집필 시점의 실행 결과는 다음과 같다.

```output
데이터셋 30개씩, SVC(RBF) 격자 25개, 5-fold (nested는 5 × 5-fold)
  n    단순 CV   nested CV   참 정확도 | 단순 CV - 참   nested - 참   (데이터셋별 sd)
  60   0.659     0.612      0.594   |   +0.064        +0.017      (0.072, 0.096)
 120   0.669     0.651      0.659   |   +0.010        -0.007      (0.048, 0.074)
 240   0.723     0.711      0.727   |   -0.004        -0.016      (0.045, 0.038)
```

결과를 읽는 법은 네 가지다.

- **데이터가 작을 때 단순 CV는 낙관적이다.** $n = 60$에서 탐색 최고 점수는 참 정확도보다 평균 0.064 높았다. 참 정확도 0.594는 우연(0.5)보다 조금 나은 수준인데, 단순 CV는 0.659를 보고한다. nested CV의 평균 편향은 0.017로 훨씬 작다.
- **편향은 데이터가 늘수록 줄어든다.** $n = 120, 240$에서는 단순 CV의 평균 편향이 0.01 이하다. 후보가 25개로 같아도, 데이터가 많으면 후보마다의 CV 점수가 덜 흔들려 "가장 운 좋은 후보"의 몫이 작아진다. 편향의 크기는 후보 수와 데이터 크기에 달려 있어 미리 알 수 없으므로, 작은 데이터에서 많은 후보를 탐색할수록 nested CV가 중요하다.
- **nested CV는 약간 비관적일 수 있다.** $n = 240$에서 nested 추정값은 참값보다 0.016 낮다. 바깥 fold마다 데이터의 80%로만 학습하는데, 이 문제는 데이터가 늘수록 성능이 계속 오르는 구간(학습 곡선이 아직 오르는 중, 0.659 → 0.727)에 있기 때문이다. 이 편향은 보수적인 방향이다.
- **데이터셋 하나에서의 오차는 편향보다 분산이 크다.** 괄호 안의 데이터셋별 표준편차(0.04–0.10)는 평균 편향보다 크다. 어떤 추정 방법을 쓰든 데이터셋 하나의 점수는 참값에서 수 %p 벗어날 수 있다. nested CV는 체계적인 낙관을 없애 줄 뿐이고, 불확실성은 12.5장의 방법으로 따로 보고해야 한다.

이어서 두 연습문제를 푼다. 첫째는 비용을 최소화하는 threshold를 검증 데이터에서 고르고, 둘째는 ridge 회귀의 정규화 강도 선택을 nested CV로 감싼다.

::exercise{id="min-cost-threshold"}

::exercise{id="nested-cv-ridge"}

## 실패 사례

- **불균형 데이터에 정확도를 쓰는 것.** 양성 1%에서 "전부 음성"이 정확도 0.991을 낸다. 양성 비율과 함께 precision·recall, PR-AUC를 보고하고, 운용 threshold를 비용에 근거해 정한다.
- **모델 선택에 쓴 점수를 최종 성능으로 보고하는 것.** 신호가 없는 데이터에서도 탐색 최고 점수는 평균 0.695였다. 선택에 쓰지 않은 시험 데이터나 nested CV로 최종 성능을 보고하고, 탐색한 후보 수를 함께 쓴다.
- **그룹 구조가 있는 데이터에 샘플 단위 $k$-fold를 쓰는 것.** 셀 단위 데이터에서 $k$-fold는 $R^2 = 0.63$을, 처음 보는 셀에서의 실제 성능은 $-0.30$을 보였고, 모델 선택까지 뒤집혔다. 일반화하려는 단위로 fold를 나눈다.
- **시험 데이터로 threshold·전처리·특징을 고르는 것.** 하이퍼파라미터가 아니어도 시험 데이터를 보고 내린 결정은 모두 선택 편향을 만든다. 모든 결정은 학습·검증 데이터 안에서 내리고 시험 데이터는 마지막에 한 번 쓴다.
- **원점이 임의인 양에 MAPE를 쓰는 것.** °C와 K에서 같은 오차의 MAPE가 300배 다르다. 단위가 있는 오차를 먼저 보고한다.

## 통과 기준

- 자신의 연구 문제에서 일반화 대상(새 샘플·새 그룹·새 시점)을 밝히고, 그에 맞는 분할 방식과 metric을 근거와 함께 고른다.
- 모델 선택에 사용한 데이터(또는 안쪽 CV)와 최종 성능 보고에 사용한 데이터(시험 세트 또는 바깥 CV)를 명확히 구별한 평가 보고서를 쓰고, 탐색한 후보 수와 예산을 적는다.
- 연습문제 두 개(`min-cost-threshold`, `nested-cv-ridge`)의 테스트를 통과한다.

## 참고 문헌

- Elkan (2001). The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)*, 973–978. — 기대 비용을 최소화하는 결정과 비용 민감 학습
- Varma, Simon (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91. — CV로 튜닝한 점수의 편향과 nested CV
- Cawley, Talbot (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107. — 모델 선택 기준의 분산이 낳는 과적합과 평가 편향
- Bergstra, Bengio (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305. — random search가 grid search보다 효율적인 이유
- Snoek, Larochelle, Adams (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems 25*, 2951–2959. — Gaussian process 기반 하이퍼파라미터 탐색
- Saito, Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432. — 불균형 데이터에서 PR 곡선과 ROC 곡선
