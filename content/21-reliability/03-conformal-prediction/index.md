---
id: reliability/conformal-prediction
title: "Conformal Prediction"
title_en: "Conformal Prediction"
tier: advanced
status: draft
prerequisites: [reliability/evaluating-predictive-distributions]
---

# Conformal Prediction

21.1장의 앙상블과 분포 head는 모델 가정이 맞을 때 좋은 불확실성을 준다. 그런데 가정이 맞는지 알 수 없을 때는 어떻게 할까. 21.2장에서 보았듯 보정이 틀린 불확실성은 흔하고, 그 사실은 평가해 보기 전에는 모른다. conformal prediction은 질문을 바꾼다. "모델이 맞는가"를 묻지 않고, **보정 데이터에서 모델이 얼마나 틀렸는지를 세어** 그만큼 구간을 넓힌다. 그 결과 모델이 무엇이든, 잡음 분포가 무엇이든, 유한한 표본에서 정확한 포함 확률을 보장하는 구간을 얻는다.

공짜는 없다. 보장에는 단 하나의 가정, **교환가능성**이 필요하고, 보장의 종류는 대개 기대하는 것보다 약하다. 이 장은 보장이 정확히 무엇을 말하는지, 언제 성립하고 언제 깨지는지를 시뮬레이션으로 확인한다.

## 학습 목표

- conformal prediction의 보장과 가정을 이해한다
- split conformal을 구현한다

## 기본 아이디어

**왜 필요한가.** 과학·공학에서 예측 구간은 안전 여유, 실험 계획, 품질 판정에 쓰인다. 이때 필요한 것은 "모델이 맞다면 90%"가 아니라 "모델이 틀려도 90%"인 구간이다. 12.4장의 신뢰구간이 정규성 같은 가정에 기대는 것과 달리, conformal prediction은 데이터의 순서 대칭성 하나에서 보장을 끌어낸다.

**교환가능성.** 확률변수 $Z_1, \dots, Z_{n+1}$의 결합분포가 순서를 어떻게 바꿔도 같으면 교환가능(exchangeable)하다고 한다. i.i.d.이면 교환가능하다. 교환가능성은 i.i.d.보다 조금 약하지만 핵심은 같다. **어느 하나가 특별하지 않다.** 여기서 $Z_i = (X_i, Y_i)$는 보정 데이터 $n$개와 아직 $Y$를 모르는 시험점 하나다.

**비순응 점수.** 각 점이 모델과 얼마나 "맞지 않는지"를 재는 함수 $s(x, y)$를 정한다. 가장 단순한 것은 절댓값 잔차 $s(x, y) = \lvert y - \hat\mu(x)\rvert$다. 점수 함수는 무엇이든 되지만, 보정 데이터를 보기 **전에** 정해야 한다(학습 데이터로 만든 모델은 괜찮다).

**순위 논증.** 시험점까지 $n+1$개 점수 $S_1, \dots, S_{n+1}$이 교환가능하므로, 시험 점수 $S_{n+1}$의 순위는 $1, \dots, n+1$에 고르게 분포한다(동점이 없다고 하자). 그러므로 보정 점수를 정렬해 $k$번째로 작은 값을 $\hat q$라 하면

$$
\Pr\big[S_{n+1} \le \hat q\big] \ge \frac{k}{n+1}.
$$

$k = \lceil (n+1)(1-\alpha) \rceil$로 고르면 우변이 $1 - \alpha$ 이상이 된다. 시험점의 예측 구간을 $C(x) = \{y : s(x, y) \le \hat q\}$로 정의하면, 절댓값 점수에서는 $[\hat\mu(x) - \hat q,\ \hat\mu(x) + \hat q]$이고

$$
1 - \alpha \ \le\ \Pr\big[Y_{n+1} \in C(X_{n+1})\big] \ \le\ 1 - \alpha + \frac{1}{n+1}
$$

이다(Vovk 외, 2005; Lei 외, 2018). 오른쪽 상한은 점수에 동점이 없을 때 성립한다. 이 부등식에는 모델의 정확도도, 잡음의 분포도 들어 있지 않다. 이것을 **분포 무관**(distribution-free) 보장이라 한다.

**보장의 정확한 의미.** 확률 $\Pr$은 보정 데이터 $n$개와 시험점을 **모두** 무작위로 볼 때의 확률이다. 따라서 두 가지를 말하지 않는다.

- 이미 뽑은 보정 세트 하나로 만든 구간이 시험 데이터의 90% 이상을 덮는다는 것. 보정 세트마다 포함률이 흔들린다(다음 절에서 계산한다).
- 특정 입력 $x$에서 90% 이상을 덮는다는 것. 이것은 **조건부 보장**이고, split conformal이 주는 것은 입력에 대해 평균 낸 **주변 보장**이다.

::quiz{id="marginal-not-conditional"}

## split conformal

**절차.** 데이터를 학습용과 보정용으로 나눈다. 이 분할이 split conformal의 이름이다.

1. 학습 데이터로 아무 모델 $\hat\mu$를 학습한다.
2. 보정 데이터 $(x_i, y_i)_{i=1}^{n}$로 점수 $s_i = \lvert y_i - \hat\mu(x_i)\rvert$를 계산한다.
3. $k = \lceil (n+1)(1-\alpha) \rceil$, $\hat q$ = 정렬한 점수의 $k$번째 값. $k > n$이면 $\hat q = \infty$.
4. 새 입력의 구간은 $\hat\mu(x) \pm \hat q$.

모델을 한 번만 학습하므로 계산이 싸다. 대가는 데이터의 일부를 보정에 떼어 놓는다는 것이다. 모든 데이터를 쓰는 full conformal, 교차검증처럼 돌려 쓰는 방법도 있지만 계산이 훨씬 비싸다(Lei 외, 2018).

잡음이 $x$에 따라 커지는 데이터 $y = \sin 2x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, (0.05 + 0.2x)^2)$에서 5차 다항식을 모델로 써 본다.

```python {exec="browser" session="cp"}
import numpy as np

rng = np.random.default_rng(0)
f_true = lambda x: np.sin(2 * x)
sd_true = lambda x: 0.05 + 0.2 * x              # 잡음이 x에 따라 커진다

def sample(n, lo=0.0, hi=3.0):
    x = rng.uniform(lo, hi, n)
    return x, f_true(x) + rng.normal(size=n) * sd_true(x)

x_tr, y_tr = sample(500)                        # 학습용
coef = np.polyfit(x_tr, y_tr, 5)                # 아무 모델이나 된다: 여기서는 5차 다항식
mu = lambda x: np.polyval(coef, x)

alpha = 0.1
x_cal, y_cal = sample(500)                      # 보정용: 학습에 쓰지 않은 데이터
scores = np.abs(y_cal - mu(x_cal))              # (n,) 비순응 점수
n = len(scores)
k = int(np.ceil((n + 1) * (1 - alpha)))         # 보정 점수 중 k번째로 작은 값
q_hat = np.sort(scores)[k - 1]

x_te, y_te = sample(20000)
covered = np.abs(y_te - mu(x_te)) <= q_hat
print(f"n = {n}, k = {k}, q_hat = {q_hat:.3f}")
print(f"시험 포함률 {covered.mean():.3f}  (목표 ≥ {1 - alpha})   구간 폭 {2 * q_hat:.3f}")
```

```output
n = 500, k = 451, q_hat = 0.640
시험 포함률 0.890  (목표 ≥ 0.9)   구간 폭 1.280
```

시험 포함률이 0.890으로 목표 0.9보다 **조금 낮다**. 보장이 깨진 것이 아니다. 보장은 보정 세트에 대한 평균이고, 이번 보정 세트는 조금 운이 나빴다. 이것을 확인하려면 보정 세트를 여러 번 새로 뽑아야 한다.

**포함률의 분포.** 점수가 연속이면 보정 세트 하나로 만든 구간의 포함률(시험 데이터가 무한할 때)은 정확히 $\mathrm{Beta}(k,\ n+1-k)$ 분포를 따른다는 것이 알려져 있다. 평균은 $k/(n+1) \ge 1-\alpha$이고, 흔들림은 $n$이 클수록 작다. 보정 세트를 1000번 새로 뽑아, 각 구간의 참 포함률을 계산해 이론과 비교한다.

```python {exec="browser" session="cp"}
from scipy.stats import beta, norm

x_grid = rng.uniform(0, 3, 2000)                # 참 포함률을 계산할 x 표본
bias, sd = f_true(x_grid) - mu(x_grid), sd_true(x_grid)
def true_coverage(q):                           # P(|Y - mu(X)| <= q): 시험 데이터가 무한할 때의 포함률
    return np.mean(norm.cdf((q - bias) / sd) - norm.cdf((-q - bias) / sd))

for n in (19, 500):
    k = int(np.ceil((n + 1) * (1 - alpha)))
    cov = np.array([true_coverage(np.sort(np.abs(y - mu(x)))[k - 1]) for x, y in (sample(n) for _ in range(1000))])
    b = beta(k, n + 1 - k)                      # 이론: 포함률 ~ Beta(k, n + 1 - k)
    print(f"n = {n:3d}  평균 {cov.mean():.3f} (이론 {b.mean():.3f})  sd {cov.std():.3f} (이론 {b.std():.3f})"
          f"  0.85 미만인 보정 세트 {np.mean(cov < 0.85):.1%}")
```

```output
n =  19  평균 0.901 (이론 0.900)  sd 0.065 (이론 0.065)  0.85 미만인 보정 세트 18.5%
n = 500  평균 0.901 (이론 0.900)  sd 0.013 (이론 0.013)  0.85 미만인 보정 세트 0.0%
```

평균 포함률은 두 경우 모두 0.90 근처로 이론과 맞다. 차이는 흔들림이다. 보정 데이터가 19개이면 포함률의 표준편차가 0.065이고, 보정 세트의 약 18%에서 포함률이 0.85 미만이다. 500개면 표준편차가 0.013으로 줄어 0.85 미만인 경우가 사라진다. **보정 데이터의 크기는 평균 보장이 아니라 개별 보장의 안정성을 정한다.** 실무에서는 수백 개 이상을 권하고, 보고할 때 $n$을 함께 쓴다.

::quiz{id="conformal-rank"}

**$n$이 너무 작으면.** $k > n$이 되면 유한한 구간으로는 보장할 수 없다. $\alpha = 0.1$이면 $n \ge 9$, $\alpha = 0.01$이면 $n \ge 99$가 필요하다. 이때 `np.quantile(scores, 1 - alpha)`로 대신하면 보장이 사라진다. 이 함수는 $n$을 기준으로 보간하므로 순위가 한 칸 모자라기 때문이다.

::quiz{id="too-few-calibration"}

**분류.** 분류에서도 같다. 점수로 $s(x, y) = 1 - \hat p_y(x)$(참 class의 예측 확률이 낮을수록 큰 점수)를 쓰면, 구간 대신 **예측 집합** $C(x) = \{y : 1 - \hat p_y(x) \le \hat q\}$를 얻는다. 모델이 확신하는 입력에서는 class 하나, 헷갈리는 입력에서는 여러 class가 들어간다. 집합의 크기 자체가 불확실성의 표시가 된다.

:::note[분야 예시]
신약 개발에서는 물성 예측 모델에 conformal 예측을 붙여 "이 분자의 용해도는 90% 확률로 이 구간"이라는 형태로 보고하는 연구가 많다. 모델의 내부가 무엇이든(그래프 신경망, 랜덤 포레스트) 같은 절차로 구간을 붙일 수 있다는 점이 장점이다. 단, 보정 분자와 새 분자가 같은 화학 공간에서 왔다는 교환가능성 가정이 필요하다. 새로운 골격의 분자로 가면 보장이 사라진다(15.4장의 scaffold split과 같은 문제).
:::

## 변형

**조건부 coverage의 한계.** 위 데이터에서 절댓값 점수의 구간은 폭이 어디서나 같다. 잡음이 작은 왼쪽에서는 지나치게 넓고, 큰 오른쪽에서는 지나치게 좁을 것이다. 분포 가정 없이 유한 표본에서 모든 $x$에 대한 조건부 보장을 주면서 쓸모 있는(무한하지 않은) 구간을 만드는 것은 일반적으로 불가능하다는 결과가 알려져 있다. 그래서 변형들은 조건부 보장 대신 **조건부로 더 잘 맞는 구간**을 목표로 한다. 핵심은 점수 함수를 바꾸는 것이다.

- **정규화(적응적) 점수**: $s(x, y) = \lvert y - \hat\mu(x)\rvert / \hat\sigma(x)$. $\hat\sigma(x)$는 학습 데이터로 따로 추정한 잡음 크기(21.1장의 분포 head나 잔차 회귀)다. 구간은 $\hat\mu(x) \pm \hat q\,\hat\sigma(x)$로, 잡음이 큰 곳에서 넓어진다.
- **CQR**(conformalized quantile regression): 분위수 회귀(12.3장)로 $\hat q_{\alpha/2}(x)$, $\hat q_{1-\alpha/2}(x)$를 학습하고, 점수를 $s = \max\{\hat q_{\alpha/2}(x) - y,\ y - \hat q_{1-\alpha/2}(x)\}$로 둔다. 관측이 분위수 구간 밖으로 나간 거리(안이면 음수)다. 구간은 $[\hat q_{\alpha/2}(x) - \hat q,\ \hat q_{1-\alpha/2}(x) + \hat q]$다(Romano 외, 2019). 분위수 회귀가 이미 이분산을 반영하므로 conformal 단계는 그 구간을 조금 넓히거나 좁혀 포함률만 맞춘다.
- **분류의 적응적 집합**: class 확률을 큰 순서로 더해 가며 참 class까지의 누적 확률을 점수로 쓰면, 어려운 입력일수록 집합이 커지는 경향이 더 뚜렷해진다.

어떤 점수를 쓰든 주변 보장은 똑같이 성립한다. 점수가 좋을수록 구간이 좁고 조건별로 고르게 맞는다. 절댓값 점수와 정규화 점수를 같은 보정 데이터로 비교한다.

```python {exec="browser" session="cp"}
r_tr = y_tr - mu(x_tr)
lv = np.polyfit(x_tr, np.log(r_tr ** 2) + 1.27, 2)        # 학습 데이터로 log 분산 회귀 (21.1장)
sig_hat = lambda x: np.exp(0.5 * np.polyval(lv, x))

def split_conformal(score_cal, alpha=0.1):
    k = int(np.ceil((len(score_cal) + 1) * (1 - alpha)))
    return np.sort(score_cal)[k - 1]

q_abs = split_conformal(np.abs(y_cal - mu(x_cal)))                    # 폭이 어디서나 같은 구간
q_norm = split_conformal(np.abs(y_cal - mu(x_cal)) / sig_hat(x_cal))  # 폭이 sig_hat(x)에 비례하는 구간
half = {"절댓값 점수": np.full_like(x_te, q_abs), "정규화 점수": q_norm * sig_hat(x_te)}
bins = [(0, 1), (1, 2), (2, 3)]
print("점수          전체    x∈[0,1) [1,2)  [2,3)   평균 폭")
for name, h in half.items():
    inside = np.abs(y_te - mu(x_te)) <= h
    by_bin = "  ".join(f"{inside[(x_te >= a) & (x_te < b)].mean():.3f}" for a, b in bins)
    print(f"{name}  {inside.mean():.3f}   {by_bin}   {2 * h.mean():.3f}")
```

```output
점수          전체    x∈[0,1) [1,2)  [2,3)   평균 폭
절댓값 점수  0.890   0.998  0.923  0.753   1.280
정규화 점수  0.897   0.873  0.895  0.921   1.201
```

두 점수 모두 전체 포함률은 0.89–0.90이다. 절댓값 점수는 $x \in [0, 1)$에서 0.998, $[2, 3)$에서 0.753이다. 잡음이 큰 곳의 관측 네 개 중 하나가 구간 밖이다. 정규화 점수는 세 구간을 0.873–0.921로 좁혔고 평균 폭도 1.280에서 1.201로 줄었다. 그래도 완벽하지는 않다. $\hat\sigma(x)$가 2차 다항식으로 근사한 추정이라 $[0, 1)$에서 조금 어긋난다. 조건부로 얼마나 맞는지는 점수 함수, 즉 그 안에 든 모델의 품질이 정한다.

::quiz{id="normalized-score-effect"}

**연구에서의 의미.** conformal을 붙였다고 불확실성 모델링이 필요 없어지는 것은 아니다. 역할이 나뉜다. 모델(분포 head, 분위수 회귀, 앙상블)은 구간의 **모양**을 정하고, conformal 단계는 **보장**을 붙인다. 구간이 쓸모 있으려면 모양이 좋아야 하고, 보장이 있으려면 conformal이 필요하다. 21.2장의 평가 표에 conformal 구간의 조건별 coverage도 함께 넣는다.

## 시계열과 분포 이동

**교환가능성이 깨질 때.** 보장의 유일한 가정이 교환가능성이므로, 그것이 깨지는 곳에서 보장도 깨진다. 흔한 경우는 다음과 같다.

- **시계열**: 시간적으로 상관된 데이터는 교환가능하지 않다. 오늘의 오차는 어제의 오차와 닮았다.
- **공변량 이동**: 시험 입력의 분포가 보정 때와 다르다. 조건부 포함률이 고르지 않으면 주변 포함률이 바뀐다.
- **조건부 이동(drift)**: 장비 노화·교체로 $p(y \mid x)$ 자체가 바뀐다.

**가중 conformal.** 공변량 이동에서는 $p(y \mid x)$가 같고 $p(x)$만 바뀐다. 가능도 비 $w(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$를 알면, 보정 점수에 $w(x_i)$로 가중치를 주어 시험 분포에서 본 것처럼 분위수를 계산할 수 있다(Tibshirani 외, 2019). 시험점 자신의 가중치 $w(x)$는 무한대 점수에 둔다. 가중치가 시험점마다 다르므로 $\hat q$도 시험점마다 다르다. 시험 입력이 잡음 큰 오른쪽으로 쏠린 경우를 본다. 가능도 비 $w(x) = 4(x/3)^3$은 이 시뮬레이션에서 알고 있다고 가정한다.

```python {exec="browser" session="cp"}
x_sh = 3 * rng.beta(4, 1, 20000)                  # 시험 입력이 잡음 큰 오른쪽으로 쏠린다
y_sh = f_true(x_sh) + rng.normal(size=20000) * sd_true(x_sh)
w = lambda x: 4 * (x / 3) ** 3                    # 가능도 비 p_test(x) / p_train(x) (알고 있다고 가정)

s_cal = np.abs(y_cal - mu(x_cal))
order = np.argsort(s_cal)
s_sorted, cw = s_cal[order], np.cumsum(w(x_cal)[order])   # 정렬한 점수와 누적 가중치
W = cw[-1]
need = (1 - alpha) * (W + w(x_sh))                # 시험점 자신의 가중치는 +inf 점수에 둔다
idx = np.searchsorted(cw, need)                   # 누적 가중치가 처음으로 need 이상이 되는 위치
q_w = np.where(idx < len(s_cal), s_sorted[np.minimum(idx, len(s_cal) - 1)], np.inf)

r_sh = np.abs(y_sh - mu(x_sh))
print(f"이동 없는 시험 포함률          {np.mean(np.abs(y_te - mu(x_te)) <= q_abs):.3f}")
print(f"이동 후, 보통 split conformal  {np.mean(r_sh <= q_abs):.3f}   폭 {2 * q_abs:.3f}")
print(f"이동 후, 가중 conformal        {np.mean(r_sh <= q_w):.3f}   평균 폭 {2 * q_w.mean():.3f}")
```

```output
이동 없는 시험 포함률          0.890
이동 후, 보통 split conformal  0.774   폭 1.280
이동 후, 가중 conformal        0.914   평균 폭 1.850
```

보통 split conformal의 포함률은 0.890에서 0.774로 떨어진다. 잡음이 큰 영역에서 구간이 좁았던 조건부 실패가, 그 영역이 많아지자 주변 실패로 드러난 것이다. 가중 conformal은 오른쪽 점수에 더 큰 가중치를 주어 구간을 넓히고(평균 폭 1.850) 포함률을 0.914로 되돌린다. 실제로는 $w(x)$를 모르므로 추정해야 한다. 학습 입력과 시험 입력을 구분하는 분류기로 추정하는 방법을 21.4장에서 다룬다. 추정한 가중치가 틀리면 보장도 근사가 된다.

**적응적 conformal.** 시간에 따라 분포가 변하는 순차 데이터에서는 보장 대상을 바꾼다. Gibbs & Candès(2021)의 **ACI**(adaptive conformal inference)는 매 시점의 구간에 쓰는 유의수준 $\alpha_t$를 온라인으로 고친다.

$$
\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t).
$$

$\mathrm{err}_t$는 시점 $t$의 관측이 구간 밖이면 1, 안이면 0이고, $\gamma$는 걸음 크기다. 구간을 놓치면 $\alpha_t$가 줄어 다음 구간이 넓어지고, 계속 맞히면 조금씩 좁아진다. 이 방법은 데이터 생성 과정에 대한 가정 없이 **장기 평균** 포함률이 $1 - \alpha$에 수렴함을 보장한다. 개별 시점의 보장은 아니다. 분포가 갑자기 바뀌면 그 직후 얼마 동안은 구간이 틀리고, $\gamma$가 그 적응 속도를 정한다.

:::note[분야 예시]
전력 수요·풍력 발전량·하천 유량 예측처럼 운영 중에 매일 구간을 내는 문제에서는 계절과 설비 변화로 분포가 계속 바뀐다. 이런 곳에서는 한 번 보정한 구간을 계속 쓰기보다, 최근 오차로 구간을 계속 갱신하는 적응적 방법이 쓰인다. 배터리 상태 추정에서도 셀이 노화하면 측정 잡음과 모델 오차가 함께 변하므로, 출고 시점의 보정이 수명 말기까지 유효하다고 가정할 수 없다.
:::

::quiz{id="exchangeability-breaks"}

## 실습

split conformal과 CQR을 구현하고, 분포 이동 상황에서 포함률이 어떻게 무너지는지 확인한다. 코드는 `labs/reliability/conformal-prediction/conformal_shift.py`다. 모델은 scikit-learn의 gradient boosting(점 예측은 제곱손실, 분위수는 quantile 손실)이고, 학습·보정 데이터는 각각 1000개다. 시험 상황은 세 가지다. 이동 없음, 시험 입력이 잡음 큰 오른쪽으로 쏠린 공변량 이동, 잡음이 1.5배가 된 조건부 이동이다. 마지막으로 잡음 배율이 1에서 2로 서서히 커지는 순차 데이터에서 고정 split conformal과 ACI($\gamma = 0.01$)를 비교한다. 집필 시점의 실행 결과는 다음과 같다(시드 0).

```output
목표 포함률 0.90.  각 칸: 전체 포함률 / x ∈ [2, 3]의 포함률 / 평균 폭
이동 없음     split(절댓값) 0.888/0.750/1.30   CQR 0.898/0.883/1.22   분위수 회귀만 0.864/0.865/1.14
공변량 이동    split(절댓값) 0.757/0.720/1.30   CQR 0.877/0.866/1.69   분위수 회귀만 0.858/0.848/1.61
조건부 이동    split(절댓값) 0.771/0.553/1.30   CQR 0.742/0.698/1.21   분위수 회귀만 0.695/0.675/1.13

순차 데이터의 구간별 포함률 (잡음 배율 1 → 2)
t ∈ [0, 1000)  고정 split 0.843   ACI 0.897
t ∈ [1000, 2000)  고정 split 0.773   ACI 0.896
t ∈ [2000, 3000)  고정 split 0.716   ACI 0.898
전체          고정 split 0.777   ACI 0.897
```

결과를 네 가지로 읽는다.

- **분위수 회귀만으로는 보장이 없다.** 이동이 없어도 포함률이 0.864로 목표보다 낮다. 분위수 회귀가 학습 데이터에 과적합해 구간이 좁아진 것이다. CQR이 보정 데이터로 구간을 넓혀 0.898로 맞춘다.
- **이동이 없으면 CQR이 조건부로 더 고르다.** 두 conformal 방법 모두 전체로는 목표 근처지만, 잡음이 큰 $x \in [2, 3]$에서 split(절댓값)은 0.750, CQR은 0.883이다. 평균 폭도 CQR이 더 좁다(1.22 대 1.30).
- **공변량 이동에서는 조건부로 고른 방법이 덜 무너진다.** split(절댓값)은 0.757로 크게 떨어지지만 CQR은 0.877로 덜 떨어진다. 구간이 입력에 맞춰 넓어지는 방법은 입력 분포가 바뀌어도 덜 영향받는다. 그래도 보장은 없다.
- **조건부 이동에서는 모든 방법이 무너진다.** 보정 데이터가 대표하지 못하는 변화이므로 어떤 점수 함수로도 고칠 수 없다. 순차 데이터에서 고정 split의 포함률은 0.843 → 0.773 → 0.716으로 계속 떨어지고, ACI는 구간을 넓혀 가며 각 구간에서 0.90 근처를 유지한다.

이어서 연습문제로 split conformal의 핵심인 순위 계산을 직접 구현한다.

::exercise{id="split-conformal"}

## 실패 사례

- **주변 coverage 보장을 조건부 보장으로 오해하는 것.** "이 환자의 예측 구간은 90% 확률로 맞다"는 말은 split conformal이 보장하지 않는다. 보장하는 것은 환자들에 대해 평균한 포함률이다. 집단별·조건별 포함률은 21.2장의 방법으로 따로 확인한다.
- **교환가능성을 확인하지 않고 보장을 주장하는 것.** 보정 데이터를 다른 시기·장비·집단에서 가져오거나, 시계열을 무작위로 섞어 보정하면 보장의 근거가 사라진다. 보정 데이터가 어디서 왔는지 쓰고, 시험 상황과 같은 분포인지 점검한다(21.4장).
- **보정 데이터를 학습이나 모델 선택에 함께 쓰는 것.** 점수 함수는 보정 데이터와 독립으로 정해야 한다. 같은 데이터로 하이퍼파라미터를 고르고 보정하면 점수가 작게 나와 포함률이 떨어진다.
- **보정 데이터가 너무 적은 것.** 평균 보장은 성립해도 개별 보정 세트의 포함률이 크게 흔들린다. $n$과 함께 그 흔들림을 보고한다.
- **`np.quantile(scores, 1 - alpha)`로 $\hat q$를 구하는 것.** 순위가 한 칸 모자라 보장이 깨진다. $\lceil (n+1)(1-\alpha) \rceil$번째 순서통계량을 쓴다.

## 통과 기준

- 자신의 데이터에서 conformal 구간의 보장이 성립하는 조건(교환가능성의 근거, 보정 데이터의 출처와 크기)을 명시한다.
- 보정 세트를 바꿔 가며(또는 이론 분포로) 주변 포함률이 $1 - \alpha$ 이상임을 확인하고, 조건별 포함률과 분포 이동 상황의 포함률을 따로 보고한다.
- 연습문제 `split-conformal`의 테스트를 통과한다.

## 참고 문헌

- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*. Springer. — conformal prediction의 원전, 교환가능성 아래의 보장
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111. — 회귀의 full·split conformal과 유한 표본 주변 보장
- Romano, Patterson, Candès (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. — CQR
- Tibshirani, Barber, Candès, Ramdas (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. — 가능도 비로 가중한 conformal
- Gibbs, Candès (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. — ACI, 장기 평균 포함률 보장
