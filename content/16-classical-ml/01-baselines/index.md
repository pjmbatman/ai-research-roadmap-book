---
id: classical-ml/baselines
title: "기준 모델의 철학"
title_en: "The Philosophy of Baselines"
tier: core
status: draft
prerequisites: [data/splitting-and-leakage]
---

# 기준 모델의 철학

"새 모델의 오차가 0.03이다"라는 문장은 그 자체로는 아무것도 말하지 않는다. 0.03이 좋은 숫자인지 알려면 비교 대상이 필요하다. 어제 값을 그대로 오늘의 예측으로 내놓는 규칙이 0.02를 낸다면, 새 모델은 문제를 푼 것이 아니라 문제를 더 어렵게 푼 것이다. 결과 숫자에 의미를 주는 비교 대상이 **baseline**(기준 모델)이다.

이 장은 baseline을 "논문 표의 맨 윗줄을 채우는 형식"이 아니라 **문제의 난이도를 재는 측정 도구**로 다룬다. 평균·최근값·추세·물리식·룩업 테이블 같은 단순 baseline을 설계하고, 그것을 새 모델과 같은 예산으로 튜닝해 강한 baseline으로 만들고, 새 모델이 정말 이겼는지 짝지은 비교로 판정한다. 마지막으로 baseline이 이기는 결과를 실패가 아니라 정보로 읽는 법을 다룬다. 16장의 나머지(선형·커널·트리 모델)는 모두 이 장의 의미에서 "강한 baseline"의 후보다.

## 학습 목표

- 문제에 맞는 단순 baseline을 설계한다
- baseline이 연구 주장에 주는 의미를 이해한다
- 새 모델이 가장 강한 baseline을 이기는지 짝지은 비교로 판정한다

## 왜 baseline인가

**왜 필요한가.** 성능 숫자는 모델의 실력과 문제의 난이도가 섞인 값이다. 같은 MAE 5 mAh라도, 용량이 거의 변하지 않는 구간을 예측하는 문제에서는 형편없는 값이고, 급격히 열화하는 구간을 예측하는 문제에서는 뛰어난 값이다. 둘을 분리하려면 같은 데이터·같은 분할·같은 metric에서 "아무 학습 없이도 얻을 수 있는 성능"을 함께 재야 한다.

**정의.** baseline은 새 방법과 **같은 입력 정보, 같은 분할, 같은 metric**으로 평가하는 비교 기준 모델이다. 세 조건 중 하나라도 다르면 비교가 아니다. 예를 들어 baseline만 무작위 분할로, 새 모델은 group 분할(15.4장)로 평가하면 숫자 차이의 대부분은 분할의 차이다.

baseline은 연구에서 두 가지 역할을 한다.

- **개선의 기준점.** 새 방법의 기여는 절대 점수가 아니라 baseline 대비 개선량이다. 이를 정규화한 값이 **skill score**다. 오차 metric $E$(MAE, RMSE 등)에 대해

$$
\mathrm{SS} = 1 - \frac{E_{\text{model}}}{E_{\text{baseline}}}
$$

로 정의한다. $\mathrm{SS} = 1$이면 오차가 0, $\mathrm{SS} = 0$이면 baseline과 같고, $\mathrm{SS} < 0$이면 baseline보다 나쁘다. MSE를 오차로 쓰고 학습 평균을 baseline으로 쓰면 SS는 결정계수 $R^2$와 같은 꼴이 된다. 즉 $R^2$는 "평균 baseline 대비 skill score"다. 이 사실이 이 장 뒤에서 중요해진다.

- **문제 난이도의 측정.** 단순 baseline이 이미 오차를 거의 0으로 만든다면 그 문제는 쉽거나, 평가 설계가 정보를 흘리고 있다(누수, 15.4장). 반대로 모든 baseline이 평균과 비슷하다면 입력에 출력에 대한 정보가 거의 없을 수 있다. 어느 쪽이든 모델을 키우기 전에 알아야 할 정보다.

:::note[분야 예시]
기상 예보에서는 baseline이 제도화되어 있다. 내일 날씨가 오늘과 같다고 예측하는 **persistence**와, 그 날짜의 장기 평균값을 내놓는 **climatology**(기후값)가 표준 기준이고, 예보 성능은 흔히 이 기준 대비 skill score로 보고한다. 짧은 예보 기간에서는 persistence가, 긴 예보 기간에서는 climatology가 이기기 어려운 기준이 된다. 예보 기간에 따라 강한 baseline이 바뀐다는 점이 이 장의 핵심 메시지와 같다.
:::

## 단순 baseline

단순 baseline은 학습을 거의 하지 않거나 파라미터가 몇 개뿐인 예측 규칙이다. 문제의 구조에 따라 자연스러운 후보가 다르다.

| baseline | 예측 | 무엇을 재는가 |
|---|---|---|
| **평균** | 학습 데이터의 출력 평균(분류라면 최빈 클래스) | 입력을 전혀 쓰지 않을 때의 오차 |
| **최근값**(persistence) | 마지막으로 관측한 값 | 시간 상관만으로 얻는 성능 |
| **선형 추세** | 최근 구간에 맞춘 직선을 연장 | 국소적인 변화율만으로 얻는 성능 |
| **물리식** | 지배 방정식·경험식에 파라미터 몇 개를 맞춘 값 | 도메인 지식만으로 얻는 성능 |
| **룩업 테이블** | 비슷한 조건(같은 사이클, 같은 조성군)의 학습 데이터 평균 | 조건별 평균만으로 얻는 성능 |

이 표의 각 줄은 "그 정보만으로 얼마나 갈 수 있는가"를 잰다. 그래서 baseline 다섯 개를 비교하면 성능 숫자 다섯 개 이상의 정보, 즉 **문제의 어느 구조가 예측력을 주는지**를 얻는다.

배터리 셀의 용량 열화 예측으로 확인한다. 셀마다 용량이 사이클 수 $n$에 따라

$$
C(n) = c_0 - a\sqrt{n} - b\,n + \varepsilon
$$

으로 줄어드는 가상 데이터를 만든다. $\sqrt{n}$ 항은 전극 표면의 SEI 층 성장처럼 확산이 지배하는 열화를, 선형 항은 사이클마다 일정하게 쌓이는 손실을 흉내 낸 것이다. 셀마다 $a, b, c_0$가 다르다. 학습 셀 40개와 시험 셀 20개로 **셀 단위 분할**을 하고, 시험 셀의 처음 200사이클을 보고 201–400사이클의 용량을 예측한다. 물리식 baseline은 $\sqrt{n}$ 항만 아는 단순한 형태 $c_0 - a\sqrt{n}$으로 둔다.

```python {exec="browser" session="base"}
import numpy as np

rng = np.random.default_rng(0)
cyc = np.arange(1, 401)                                     # 사이클 1–400, (T,)
def make_cells(n, rng, fade=1.0):                           # 셀 n개의 용량 곡선, (n, T)
    a = fade * rng.uniform(0.003, 0.007, (n, 1))            # √n 항 (SEI 성장)
    b = fade * rng.uniform(0.0, 3e-4, (n, 1))               # 선형 항
    c0 = 2.0 + rng.normal(0, 0.01, (n, 1))                  # 초기 용량 [Ah]
    return c0 - a * np.sqrt(cyc) - b * cyc + rng.normal(0, 0.002, (n, len(cyc)))

cap_tr, cap_te = make_cells(40, rng), make_cells(20, rng)  # 셀 단위 분할: 학습 40, 시험 20
obs, fut = cyc <= 200, cyc > 200                            # 200사이클까지 보고 이후를 예측

def per_cell_fit(cap, basis, rows):                         # 셀마다 최소제곱 → 미래 구간 예측
    B = np.column_stack(basis(cyc))                         # (T, P)
    coef = np.linalg.lstsq(B[rows], cap[:, rows].T, rcond=None)[0]   # (P, n)
    return (B[fut] @ coef).T                                # (n, 200)

pred = {
    "평균":          np.full((20, 200), cap_tr[:, fut].mean()),
    "최근값":        np.repeat(cap_te[:, obs][:, -10:].mean(1, keepdims=True), 200, 1),
    "선형 추세":     per_cell_fit(cap_te, lambda n: (np.ones_like(n), n), (cyc > 100) & obs),
    "물리식 c0-a√n": per_cell_fit(cap_te, lambda n: (np.ones_like(n), np.sqrt(n)), obs),
    "룩업 테이블":   np.tile(cap_tr[:, fut].mean(0), (20, 1)),
}
mae_mah = {k: 1000 * np.abs(p - cap_te[:, fut]).mean(1) for k, p in pred.items()}   # 셀별 MAE, (20,)
for k, v in mae_mah.items():
    print(f"MAE {v.mean():5.1f} mAh  {k}")
```

```output
MAE  37.3 mAh  평균
MAE  33.7 mAh  최근값
MAE   6.5 mAh  선형 추세
MAE  10.4 mAh  물리식 c0-a√n
MAE  34.7 mAh  룩업 테이블
```

결과는 문제의 구조를 드러낸다. 평균·최근값·룩업 테이블은 모두 33–37 mAh로 비슷하다. 셀마다 초기 용량과 열화 속도가 달라서, 셀 정보를 쓰지 않는 평균·룩업 테이블은 셀 간 차이를 놓친다. 최근값은 셀 정보를 쓰지만 앞으로 계속 줄어든다는 사실을 무시한다. 셀 자신의 최근 기울기를 쓰는 선형 추세(6.5 mAh)가 압도적으로 낫다. 이 문제에서 예측력의 대부분은 **셀마다 다른 열화 속도**에 있다는 뜻이다. 반쪽짜리 물리식(10.4 mAh)은 선형 추세보다 나쁘다. 물리식이 맞는 구조를 담고 있어도 항이 빠지면 외삽에서 체계적으로 틀린다.

::quiz{id="persistence-for-short-horizon"}

**연구에서의 의미.** 단순 baseline의 성능은 "이 문제는 어려운가"에 대한 첫 번째 측정값이다. 새 모델의 결과를 보기 전에 이 표를 먼저 만들어 두면, 새 모델의 숫자를 읽을 기준이 생긴다. 또한 baseline 사이의 순위 자체가 데이터에 대한 가설을 준다. 이 예에서는 "셀별 기울기가 중요하다"는 가설이고, 이는 새 모델에 어떤 입력을 줘야 하는지를 알려 준다.

## 강한 baseline

**왜 필요한가.** 단순 baseline을 이기는 것은 필요조건일 뿐이다. 새 방법이 수십 번의 하이퍼파라미터 탐색을 거쳤는데 baseline은 기본값 그대로라면, 두 숫자의 차이에는 방법의 차이와 **튜닝 노력의 차이**가 섞인다. 정보검색 분야에서는 신경망 모델이 약하거나 튜닝되지 않은 baseline 대비 개선을 보고하는 관행이 비판받았고(Lin, 2018), 같은 문제는 어느 분야에서나 생길 수 있다.

**정의.** **강한 baseline**은 (1) 문제 구조를 반영해 잘 설계되고, (2) 새 방법과 **같은 튜닝 예산**으로 같은 검증 데이터에서 튜닝된 비교 모델이다. 튜닝 예산은 시도한 설정의 수, 계산 시간, 검증 데이터의 사용 방식을 모두 포함한다. 시험 데이터는 어느 쪽도 튜닝에 쓰지 않는다.

앞의 baseline들을 강하게 만든다. 선형 추세의 창 길이(최근 몇 사이클로 직선을 맞출지)는 하이퍼파라미터이므로 **학습 셀에서** 고른다. 이 baseline은 다른 셀에서 아무것도 배우지 않으므로, 학습 셀은 순수한 검증 데이터 역할을 한다. 물리식에는 빠진 선형 항을 넣고, 룩업 테이블은 절대 용량 대신 **열화량**을 표로 만들어 셀의 현재값에 더한다.

```python {exec="browser" session="base"}
def trend(cap, w):                                          # 마지막 w 사이클로 선형 추세
    return per_cell_fit(cap, lambda n: (np.ones_like(n), n), (cyc > 200 - w) & obs)
def physics2(cap):                                          # c0 - a√n - b n
    return per_cell_fit(cap, lambda n: (np.ones_like(n), np.sqrt(n), n), obs)
def lookup_anchored(cap):                                   # 학습 셀의 평균 열화량 + 현재값
    drop = cap_tr[:, fut].mean(0) - cap_tr[:, obs][:, -10:].mean()
    return cap[:, obs][:, -10:].mean(1, keepdims=True) + drop

mae = lambda p, cap: 1000 * np.abs(p - cap[:, fut]).mean(1)  # 셀별 MAE [mAh]
tr_mae = {w: mae(trend(cap_tr, w), cap_tr).mean() for w in (25, 50, 100, 200)}
print("창 길이별 학습 셀 MAE:", "  ".join(f"w={w} {v:.1f}" for w, v in tr_mae.items()), "mAh")
w_best = min(tr_mae, key=tr_mae.get)                        # 창 길이는 학습 셀로 고른다
strong = {f"선형 추세 w={w_best}": trend(cap_te, w_best), "물리식 + 선형항": physics2(cap_te),
          "룩업 + 현재값": lookup_anchored(cap_te)}
for k, p in strong.items():
    mae_mah[k] = mae(p, cap_te)
    print(f"시험 MAE {mae_mah[k].mean():5.1f} mAh  {k}")
```

```output
창 길이별 학습 셀 MAE: w=25 5.1  w=50 4.3  w=100 6.1  w=200 17.8 mAh
시험 MAE   5.3 mAh  선형 추세 w=50
시험 MAE   1.7 mAh  물리식 + 선형항
시험 MAE  10.3 mAh  룩업 + 현재값
```

세 baseline이 모두 좋아졌다. 창 길이를 학습 셀로 고른 선형 추세는 6.5에서 5.3 mAh로, 열화량을 쓴 룩업 테이블은 34.7에서 10.3 mAh로 내려갔다. 선형 항을 넣은 물리식은 1.7 mAh로 단순 baseline 중 가장 좋던 값보다 네 배 가까이 낮다. 이 데이터는 이 식으로 만들었으니 놀랄 일은 아니지만, 실제 문제에서도 도메인의 경험식은 이런 위치에 있는 경우가 많다. 적은 파라미터로 외삽까지 버틴다.

강한 baseline을 만드는 원칙을 정리한다.

- **튜닝은 같은 예산, 같은 검증 데이터로.** 새 방법에 설정 50개를 시도했다면 baseline에도 비슷한 수의 설정을 시도한다. 하이퍼파라미터 탐색 방법은 16.6장에서 다룬다.
- **고전 모델을 후보에 넣는다.** 표 데이터라면 잘 튜닝된 gradient boosting(16.4장), 매끄러운 저차원 문제라면 정규화된 선형 모델(16.2장)이나 커널 방법(16.3장)이 흔히 강한 baseline이다.
- **도메인 지식을 baseline에 쓴다.** 물리식, 경험식, 전문가가 쓰는 규칙은 데이터 없이도 얻는 정보다. 새 모델이 이 정보를 이기지 못하면 데이터에서 새로 배운 것이 없다는 뜻이다.
- **baseline의 구현과 설정을 보고한다.** 독자가 baseline이 공정하게 튜닝되었는지 판단할 수 있어야 한다.

::quiz{id="fair-tuning-budget"}

:::note[분야 예시]
재료·화학의 물성 예측에서 자연스러운 룩업 테이블은 "같은 조성군 또는 가장 비슷한 학습 화합물의 측정값"이다. 이 baseline이 강하다면 시험 화합물이 학습 화합물과 너무 비슷하다는 신호일 수 있다. 무작위 분할 대신 화합물 골격(scaffold)이나 조성군 단위로 분할(15.4장)하면 이 baseline의 성능이 크게 떨어지고, 그때 비로소 새 모델이 "새로운 화합물"로 일반화하는지 잴 수 있다.
:::

## baseline이 이기는 경우

이제 "새 모델"을 넣는다. 학습 셀 40개로, 처음 200사이클의 곡선(10사이클 간격 20점)에서 뒤 200사이클의 곡선을 직접 예측하는 ridge 회귀(16.2장)를 학습한다. 정규화 강도는 학습 데이터 안에서 고른다. 그리고 앞 셀들에서 가장 좋은 baseline과 **시험 셀마다** 오차를 짝지어 비교한다(12.5장의 짝지은 비교). 셀이 독립 단위이므로 셀별 MAE 차이를 부트스트랩한다.

```python {exec="browser" session="base"}
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=np.logspace(-6, 0, 13))              # 앞 200사이클 → 뒤 200사이클
model.fit(cap_tr[:, obs][:, ::10], cap_tr[:, fut])          # (40, 20) → (40, 200)
mae_mah["Ridge"] = mae(model.predict(cap_te[:, obs][:, ::10]), cap_te)
best = min((k for k in mae_mah if k != "Ridge"), key=lambda k: mae_mah[k].mean())
d = mae_mah["Ridge"] - mae_mah[best]                        # 셀별 짝지은 차이, (20,)
boot = np.random.default_rng(1).choice(d, (5000, d.size)).mean(1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Ridge MAE {mae_mah['Ridge'].mean():.1f} mAh, 최강 baseline({best}) {mae_mah[best].mean():.1f} mAh")
print(f"차이 {d.mean():+.1f} mAh, 95% CI [{lo:+.1f}, {hi:+.1f}], Ridge가 이긴 셀 {np.sum(d < 0)}/20")
print(f"평균 baseline 대비 skill {1 - mae_mah['Ridge'].mean() / mae_mah['평균'].mean():.2f}")
print(f"최강 baseline 대비 skill {1 - mae_mah['Ridge'].mean() / mae_mah[best].mean():.2f}")
```

```output
Ridge MAE 3.2 mAh, 최강 baseline(물리식 + 선형항) 1.7 mAh
차이 +1.5 mAh, 95% CI [+0.8, +2.2], Ridge가 이긴 셀 2/20
평균 baseline 대비 skill 0.91
최강 baseline 대비 skill -0.84
```

같은 모델에 대해 두 문장을 쓸 수 있다.

- "제안한 모델은 평균 baseline 대비 skill score 0.91을 달성했다." — 사실이다. MSE로 쟀다면 $R^2$가 높다고 보고했을 것이다.
- "제안한 모델은 가장 강한 baseline보다 셀당 1.5 mAh 나쁘다(95% CI [+0.8, +2.2]). 20개 셀 중 2개에서만 이겼다." — 이것도 사실이다.

첫 문장만 쓴 논문은 거짓말을 하지 않고도 독자를 오도한다. 평균 baseline은 거의 모든 모델이 이기는 기준이라, 그 대비 skill score는 문제에 대한 정보도 모델에 대한 정보도 거의 주지 않는다. 판정은 **가장 강한 baseline**에 대해, **짝지은 차이의 신뢰구간**으로 한다. 이 예에서는 구간 전체가 0보다 커서 "baseline이 이긴다"고 판정할 수 있다. 구간이 0을 포함했다면 "차이를 판정할 수 없다"가 정직한 결론이다. 평균값의 대소만 보고 "이겼다"고 쓰지 않는다.

::quiz{id="paired-ci-verdict"}

**baseline이 이기는 것은 실패가 아니라 신호다.** 그 신호가 가리키는 곳은 세 군데다.

- **문제 정의.** 예측 기간이 짧아 persistence가 거의 완벽하다면, 그 예측 기간은 연구할 가치가 적다. 더 긴 기간, 더 어려운 조건으로 문제를 다시 정의한다. 이 예에서는 "고온처럼 학습 셀과 다른 조건의 셀"이나 "200사이클보다 적게 보고 예측"이 더 의미 있는 문제일 수 있다.
- **데이터.** 새 모델이 baseline이 쓰는 정보를 받지 못하고 있을 수 있다. 이 예의 ridge 모델은 다른 셀들에서 "곡선 모양 → 미래" 관계를 배우지만, 물리식 baseline은 열화의 함수 형태라는 강한 사전 지식을 쓴다. 이 지식을 모델 구조에 넣는 것이 과학 머신러닝(24장)의 출발점이다.
- **평가.** metric이 쉬운 부분에 지배되거나(예: 대부분 정상 상태인 시계열의 전체 평균 오차), 분할이 쉬운 일반화만 재고 있을 수 있다. 연구 주장과 맞는 metric과 분할인지 다시 본다(15.4장, 16.6장).

:::note[분야 예시]
생명과학의 유전자 발현 예측에서도 같은 구조가 나타난다. 유전자마다 평균 발현량이 크게 다르기 때문에 "각 유전자의 학습 데이터 평균"이라는 룩업 테이블 baseline만으로도 전체 상관계수가 높게 나온다. 새 모델이 조건(약물 처리, 유전자 교란)에 따른 **변화**를 예측하는지 보려면, 이 룩업 테이블 대비 성능이나 변화량 자체의 예측 성능을 보고해야 한다. 앞의 룩업 테이블을 절대 용량에서 열화량으로 바꾼 것과 같은 생각이다.
:::

## 실습

같은 셀 단위 분할과 같은 metric(셀별 MAE)에서 단순 baseline 다섯 개를 비교하는 표는 앞의 첫 번째 셀이 만들었다. 다음을 순서대로 진행해 자신의 baseline 표를 완성한다.

1. 세 번째 셀의 짝지은 차이 `d`를 처음 5개 셀(`d[:5]`)만으로 다시 부트스트랩한다. 시험 셀이 5개뿐일 때 신뢰구간이 얼마나 넓어지는지, 판정이 바뀌는지 기록한다.
2. `make_cells`에 `fade=1.8`을 주어 열화가 빠른 "고온" 시험 셀을 만들고, ridge 모델과 세 강한 baseline을 다시 비교한다. 학습 셀의 평균 열화량을 쓰는 룩업 테이블은 이 조건에서 어떻게 되는가. 셀 자신의 곡선만 쓰는 baseline과 학습 셀에서 배운 모델 중 무엇이 조건 변화에 강한지 설명한다.
3. 자신의 연구 문제에 대해 위의 다섯 종류 baseline 표를 채우고, 각 baseline이 "무엇을 재는지"를 한 줄씩 적는다.

이어서 두 연습문제를 푼다. 첫 문제는 시계열의 단순 baseline 네 개를 구현하고, 두 번째 문제는 짝지은 부트스트랩으로 새 모델이 baseline을 이기는지 판정하는 함수를 만든다.

::exercise{id="forecast-baselines"}

::exercise{id="beats-baseline"}

## 실패 사례

- **약한 baseline만 골라 비교하는 것.** 평균이나 기본 설정의 선형 회귀만 비교 대상으로 두면 거의 모든 모델이 크게 이긴다. 높은 $R^2$는 평균 baseline 대비 skill score일 뿐이다. 문제 구조에 맞는 단순 baseline(시계열이면 persistence, 물리 문제면 물리식)과 잘 튜닝된 고전 모델을 함께 둔다.
- **baseline을 튜닝하지 않는 것.** 새 방법에는 탐색 예산을 쓰고 baseline은 기본값으로 두면, 차이는 방법이 아니라 노력의 차이를 잰다. 같은 예산·같은 검증 데이터로 튜닝하고, 그 설정을 보고한다.
- **평균값의 대소만으로 "이겼다"고 쓰는 것.** 시험 단위가 적으면 평균 차이는 운으로 뒤집힌다. 독립 단위(셀·환자·지역)로 짝지은 차이의 신뢰구간을 보고한다.
- **baseline만 다른 조건에서 평가하는 것.** 다른 분할, 다른 전처리, 다른 시험 기간으로 잰 baseline 숫자를 다른 논문에서 가져와 표에 넣으면 비교가 아니다. 가능하면 같은 파이프라인에서 직접 다시 돌린다.

## 통과 기준

- 자신의 연구 문제에 대해 같은 분할·같은 metric으로 단순 baseline 다섯 개와 강한 baseline 하나 이상의 표를 만들고, 각 baseline이 무엇을 재는지 설명한다.
- 새 모델(또는 가장 복잡한 후보)과 가장 강한 baseline의 짝지은 차이를 독립 단위의 신뢰구간으로 보고하고, "이긴다·진다·판정 불가" 중 하나로 판정한다.
- 연습문제 두 개(`forecast-baselines`, `beats-baseline`)의 테스트를 통과한다.

## 참고 문헌

- Lin (2018). The neural hype and comparisons against weak baselines. *ACM SIGIR Forum*, 52(2). — 정보검색 분야에서 약하거나 튜닝되지 않은 baseline 대비 개선을 보고하는 관행에 대한 비판
