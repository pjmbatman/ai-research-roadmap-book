---
id: information-and-learning-theory/generalization-basics
title: "일반화의 기초"
title_en: "Foundations of Generalization"
tier: core
status: draft
prerequisites: [probability/expectation-and-limits]
---

# 일반화의 기초

학습 손실이 0에 가까운 모델이 새 데이터에서 엉망인 일은 흔하다. 반대로 학습 손실이 꽤 남아 있는 모델이 새 데이터에서 더 잘 맞기도 한다. 학습의 목적은 가진 데이터를 맞추는 것이 아니라 **아직 보지 않은 데이터**에서 잘하는 것이고, 이 둘 사이의 관계를 다루는 것이 일반화(generalization) 이론이다.

이 장은 일반화를 말하는 기본 언어를 세운다. 학습 손실과 진짜 목표를 구분하는 경험 위험과 모집단 위험, 오차가 어디서 오는지 나누는 오차 분해와 bias–variance 분해, 모델 집합의 "용량"이 일반화에 주는 영향을 재는 VC 차원과 Rademacher 복잡도의 직관, 그리고 가정 없이는 어떤 일반화도 없다는 no free lunch 정리다. 마지막 절은 이 이론들이 실제 딥러닝 실험과 어디서 어긋나는지 정리하고, 그 간극을 14.5장으로 넘긴다.

## 학습 목표

- 경험 위험과 모집단 위험을 구분한다
- 오차를 분해한다
- 용량과 일반화 bound의 의미를 안다

## 위험

**왜 필요한가.** "모델이 좋다"는 말을 정확히 하려면, 무엇에 대한 어떤 평균에서 좋은지 정해야 한다. 학습 손실은 우리가 계산할 수 있는 양이고, 우리가 원하는 양은 아니다. 둘을 다른 이름으로 불러야 혼동하지 않는다.

**정의.** 입력–출력 쌍 $(x, y)$가 미지의 분포 $P$에서 나온다고 하자. 예측 함수 $h$와 손실 $\ell$에 대해

$$
R(h) = \mathbb{E}_{(x, y) \sim P}\big[\ell(h(x), y)\big], \qquad
\hat{R}_N(h) = \frac{1}{N}\sum_{i=1}^{N}\ell(h(x_i), y_i)
$$

를 각각 **모집단 위험**(population risk, 기대 위험)과 **경험 위험**(empirical risk)이라 한다. 우리가 원하는 것은 $R(h)$를 작게 하는 것이지만, $P$를 모르므로 학습 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$에서 계산한 $\hat{R}_N(h)$만 쓸 수 있다. 정해진 함수 집합(가설 집합) $\mathcal{H}$ 안에서 경험 위험을 최소화하는 $\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{R}_N(h)$를 고르는 원리를 **경험 위험 최소화**(empirical risk minimization, ERM)라 한다. 10장의 학습 손실 $L(\theta)$가 바로 $\hat{R}_N$이고, 14.3장에서 본 대로 손실 $\ell$은 대개 NLL이다.

**고정된 $h$와 고른 $\hat{h}$는 다르다.** 데이터를 보기 전에 정한 함수 $h$에 대해서는 $\hat{R}_N(h)$가 독립 표본의 평균이므로 $\mathbb{E}[\hat{R}_N(h)] = R(h)$, 즉 불편 추정량이다(12.1장). 대수의 법칙(11.5장)에 따라 $N$이 커지면 $R(h)$로 수렴한다. 그러나 $\hat{h}$는 **같은 데이터로 골랐다.** 데이터의 우연한 잡음에 잘 맞는 함수가 골라지므로, 그 함수의 경험 위험은 모집단 위험보다 체계적으로 작다. 사인 곡선에 잡음 표준편차 0.3을 더한 데이터 15개로 두 경우를 500번 반복해 비교한다.

```python {exec="browser" session="gen"}
import numpy as np

def f_true(x):
    return np.sin(2 * np.pi * x)

def sample(rng, n, noise=0.3):
    x = rng.uniform(0, 1, n)
    return x, f_true(x) + noise * rng.normal(size=n)

rng = np.random.default_rng(0)
x_te, y_te = sample(rng, 100_000)                  # 모집단 위험을 근사할 큰 시험 집합
fixed = np.array([0.0, 6.0, -18.0, 12.0])          # 데이터를 보기 전에 정한 3차식 계수
emp_fixed, emp_erm, pop_erm = [], [], []
for _ in range(500):
    x, y = sample(rng, 15)
    emp_fixed.append(np.mean((np.polyval(fixed[::-1], x) - y) ** 2))
    c = np.polyfit(x, y, 7)                        # ERM: 7차 다항식 최소제곱
    emp_erm.append(np.mean((np.polyval(c, x) - y) ** 2))
    pop_erm.append(np.mean((np.polyval(c, x_te) - y_te) ** 2))
print(f"고정한 모델  경험 위험 평균 {np.mean(emp_fixed):.3f}   모집단 위험 {np.mean((np.polyval(fixed[::-1], x_te) - y_te) ** 2):.3f}")
print(f"ERM 모델     경험 위험 평균 {np.mean(emp_erm):.3f}   모집단 위험 중앙값 {np.median(pop_erm):.3f}")
```

```output
고정한 모델  경험 위험 평균 0.180   모집단 위험 0.180
ERM 모델     경험 위험 평균 0.042   모집단 위험 중앙값 0.629
```

고정한 3차식은 경험 위험의 평균이 모집단 위험과 같다(0.180). ERM으로 고른 7차 다항식은 경험 위험이 0.042로, 잡음의 분산 $0.3^2 = 0.09$보다도 작다. 잡음까지 맞췄다는 뜻이고, 모집단 위험은 그 15배다. 두 위험의 차이 $R(\hat h) - \hat{R}_N(\hat h)$를 **일반화 간극**(generalization gap)이라 한다. 일반화 이론의 중심 질문은 이 간극이 얼마나 클 수 있는가이다. 검증·시험 집합을 학습 데이터와 따로 두는 이유(15.4장)도 여기에 있다. 학습에 쓰지 않은 데이터에서 잰 위험만이 고른 모델에 대해 불편이다.

::quiz{id="training-risk-optimism"}

## 오차 분해

**왜 필요한가.** 모델의 시험 오차가 크다는 것만으로는 무엇을 고쳐야 할지 알 수 없다. 모델을 키워야 하는지, 데이터를 더 모아야 하는지, 학습을 더 오래 해야 하는지는 오차가 **어디서** 왔는지에 달려 있다.

**정의.** 가능한 모든 함수 중 가장 좋은 위험(Bayes 위험)을 $R^\star$, 가설 집합 $\mathcal{H}$ 안에서 가장 좋은 함수를 $h^\star_{\mathcal{H}} = \arg\min_{h \in \mathcal{H}} R(h)$, ERM의 정확한 해를 $\hat{h}$, 실제 optimizer가 유한한 시간에 도달한 함수를 $\tilde{h}$라 하자. 그러면 $\tilde{h}$의 초과 위험은 세 항으로 나뉜다(Bottou & Bousquet, 2008).

$$
R(\tilde{h}) - R^\star =
\underbrace{R(h^\star_{\mathcal{H}}) - R^\star}_{\text{근사 오차}} +
\underbrace{R(\hat{h}) - R(h^\star_{\mathcal{H}})}_{\text{추정 오차}} +
\underbrace{R(\tilde{h}) - R(\hat{h})}_{\text{최적화 오차}}.
$$

- **근사 오차**(approximation error): $\mathcal{H}$가 참 관계를 표현하지 못해서 생긴다. 직선으로 사인 곡선을 맞추면 데이터가 무한해도 남는다. $\mathcal{H}$를 키워야 줄어든다(7.2장의 표현력).
- **추정 오차**(estimation error): 유한한 데이터로 $\mathcal{H}$ 안의 최선을 찾아야 해서 생긴다. 데이터가 늘면 줄고, $\mathcal{H}$가 크면 커진다.
- **최적화 오차**(optimization error): 학습 손실을 정확히 최소화하지 못해서 생긴다. 비볼록 문제나 이른 종료에서 생긴다(10장). 경우에 따라 음수일 수도 있다. 정확한 ERM 해가 잡음을 과하게 맞추는 상황에서는, 덜 최적화된 해가 오히려 모집단 위험이 작다(10.3장의 early stopping).

여기에 $R^\star$ 자체가 있다. 측정 잡음처럼 어떤 모델도 줄일 수 없는 오차로, 12.3장의 aleatoric 불확실성에 해당한다. 제곱손실에서는 $R^\star$가 잡음의 분산이다.

::quiz{id="error-decomposition"}

**연구에서의 의미.** 세 항은 서로 다른 처방을 가리킨다. 학습 오차와 시험 오차가 모두 높으면 근사 오차나 최적화 오차를 의심한다(모델 확대, 학습 설정 점검, 17.9장). 학습 오차는 낮은데 시험 오차가 높으면 추정 오차를 의심한다(데이터 추가, 정규화, 모델 축소). 학습 오차가 잡음 수준 $R^\star$보다 낮아졌다면 잡음을 외우기 시작했다는 신호다.

## bias–variance

**왜 필요한가.** 오차 분해의 근사 오차와 추정 오차는 제곱손실에서 계산 가능한 두 양, bias와 variance로 구체화된다. 둘을 직접 재 보면 모델 복잡도가 일반화에 어떻게 작용하는지 숫자로 볼 수 있다.

**분해.** $y = f(x) + \varepsilon$, $\mathbb{E}[\varepsilon] = 0$, $\mathrm{Var}(\varepsilon) = \sigma^2$이라 하자. 학습 데이터 $\mathcal{D}$가 무작위이므로 학습된 예측 $\hat{f}_{\mathcal{D}}(x)$도 확률변수다. 새 관측 $y$에 대한 기대 제곱오차는

$$
\mathbb{E}_{\mathcal{D}, \varepsilon}\big[(\hat{f}_{\mathcal{D}}(x) - y)^2\big] =
\underbrace{\big(\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)] - f(x)\big)^2}_{\text{bias}^2} +
\underbrace{\mathrm{Var}_{\mathcal{D}}\big[\hat{f}_{\mathcal{D}}(x)\big]}_{\text{variance}} +
\underbrace{\sigma^2}_{\text{잡음}}
$$

이다. 12.1장의 추정량 MSE 분해 $\mathrm{MSE} = \text{bias}^2 + \text{variance}$를, 숫자 하나가 아니라 **함수**를 추정하는 경우로 옮긴 것이다(Geman, Bienenstock, Doursat, 1992). 교차항이 사라지는 이유도 같다. bias는 "같은 실험을 무한히 반복해 평균한 모델"이 참 함수에서 벗어난 정도, variance는 데이터가 바뀔 때마다 모델이 흔들리는 정도다.

사인 곡선에서 표본 20개를 뽑아 다항식을 맞추는 실험을 400번 반복하고, 평가 지점 50곳에서 평균한 bias², variance를 차수별로 잰다. 마지막 열은 각 반복에서 새로 뽑은 관측으로 직접 잰 시험 MSE다.

```python {exec="browser" session="gen"}
x_grid = np.linspace(0.05, 0.95, 50)               # 평가 지점, (M,)
noise = 0.3
print("차수   bias²    variance  noise    합      시험 MSE")
for deg in (1, 3, 5, 6):
    preds, test = [], []
    for seed in range(400):
        r = np.random.default_rng(seed)
        x, y = sample(r, 20, noise)
        c = np.polyfit(x, y, deg)
        preds.append(np.polyval(c, x_grid))        # (M,)
        y_new = f_true(x_grid) + noise * r.normal(size=50)
        test.append(np.mean((preds[-1] - y_new) ** 2))
    P = np.array(preds)                            # (R, M)
    bias2 = np.mean((P.mean(0) - f_true(x_grid)) ** 2)
    var = np.mean(P.var(0))
    print(f"{deg:3d}   {bias2:.4f}   {var:.4f}    {noise**2:.4f}   {bias2 + var + noise**2:.4f}   {np.mean(test):.4f}")
```

```output
차수   bias²    variance  noise    합      시험 MSE
  1   0.1560   0.0324    0.0900   0.2785   0.2778
  3   0.0030   0.0224    0.0900   0.1154   0.1143
  5   0.0005   0.1401    0.0900   0.2305   0.2313
  6   0.0008   0.7075    0.0900   0.7983   0.8034
```

세 항의 합이 직접 잰 시험 MSE와 (시뮬레이션 오차 안에서) 같다. 1차식은 bias가 지배적이다(0.156). 3차식에서 bias가 거의 사라지고 시험 MSE가 가장 작다. 5차와 6차에서는 bias가 더 줄지만 variance가 빠르게 커져(0.140, 0.708) 시험 MSE가 다시 커진다. 이것이 고전적인 **U자 곡선**과 bias–variance trade-off다.

::quiz{id="bias-variance-degree"}

**학습 곡선.** 같은 모델 복잡도에서 데이터 크기 $N$을 바꾸면 무엇이 변하는가. 학습 오차와 검증 오차를 $N$의 함수로 그린 것이 **학습 곡선**(learning curve)이다. 3차와 7차 다항식으로 비교한다(검증 오차는 반복 200번의 중앙값).

```python {exec="browser" session="gen"}
print("   N    3차: 학습   검증     7차: 학습   검증")
for n in (15, 30, 60, 120, 480):
    row = []
    for deg in (3, 7):
        tr, va = [], []
        for seed in range(200):
            r = np.random.default_rng(seed)
            x, y = sample(r, n)
            c = np.polyfit(x, y, deg)
            tr.append(np.mean((np.polyval(c, x) - y) ** 2))
            va.append(np.mean((np.polyval(c, x_te[:2000]) - y_te[:2000]) ** 2))
        row += [np.mean(tr), np.median(va)]
    print(f"{n:4d}      {row[0]:.3f}   {row[1]:.3f}        {row[2]:.3f}   {row[3]:.3f}")
```

```output
   N    3차: 학습   검증     7차: 학습   검증
  15      0.067   0.126        0.039   0.479
  30      0.080   0.107        0.065   0.133
  60      0.087   0.102        0.077   0.106
 120      0.091   0.099        0.084   0.098
 480      0.093   0.097        0.088   0.093
```

세 가지를 읽는다.

- **학습 오차는 $N$과 함께 커지고, 검증 오차는 줄어든다.** 데이터가 적을수록 잡음까지 맞추기 쉽기 때문이다. 두 곡선은 위아래에서 같은 값으로 다가간다.
- **두 곡선이 다가가는 값은 잡음 0.09에 그 모델족의 근사 오차를 더한 것이다.** 3차식은 사인 곡선을 정확히 표현하지 못해 $N = 480$에서도 0.097 근처에 머문다. 7차식은 근사 오차가 더 작아 0.093으로 내려간다.
- **어느 모델이 나은지는 $N$에 달려 있다.** $N = 15$에서는 3차식이 크게 낫고(0.126 대 0.479), $N = 120$ 이상에서는 7차식이 조금 낫다. "복잡한 모델은 과적합한다"가 아니라 "복잡한 모델은 더 많은 데이터가 필요하다"가 정확한 표현이다.

학습 곡선은 "데이터를 더 모으면 나아질까"라는 실험 설계 질문에 답한다. 두 곡선 사이가 아직 벌어져 있으면 데이터가 도움이 되고(추정 오차가 지배), 둘이 이미 붙어 있는데 둘 다 높으면 모델이나 특징을 바꿔야 한다(근사 오차가 지배).

**고전적 그림의 한계.** U자 곡선은 이 장의 다항식처럼 파라미터 수가 데이터 수보다 적은 영역의 그림이다. 파라미터가 데이터보다 훨씬 많은 현대 신경망에서는 복잡도를 더 키우면 시험 오차가 **다시 내려가는** 현상(double descent)이 관찰된다. 또 "복잡도"를 파라미터 수로 재는 것 자체가 부정확할 수 있다. 이것이 14.5장의 주제다.

:::note[분야 예시]
기후 모델의 통계적 후처리(downscaling)에서 관측소 하나의 기록이 수십 년, 즉 독립 표본으로 보면 많지 않은 경우가 흔하다. 같은 예측 변수로 선형 회귀와 큰 신경망을 비교하면, 짧은 기록에서는 선형 회귀가 이기고 긴 기록에서는 신경망이 이기는 일이 생긴다. 어느 모델이 "더 좋은가"는 데이터 크기와 함께 말해야 하고, 학습 곡선이 그 근거가 된다.
:::

## 용량과 bound

**왜 필요한가.** 일반화 간극을 실험으로 재지 않고 미리 보장할 수 있을까. 그러려면 "가설 집합이 얼마나 많은 것을 맞출 수 있는가", 즉 **용량**(capacity)을 수로 재야 한다. 용량이 클수록 우연한 잡음에도 잘 맞는 함수가 집합 안에 있을 가능성이 커지고, 간극도 커질 수 있다.

**출발점: 함수 하나.** 손실이 $[0, 1]$ 범위에 있고 $h$를 데이터를 보기 전에 정했다면, Hoeffding 부등식(11.5장)에서

$$
P\big(\lvert \hat{R}_N(h) - R(h) \rvert > \epsilon\big) \le 2e^{-2N\epsilon^2}
$$

이다. 확률 $1 - \delta$ 이상으로 $\lvert \hat{R}_N(h) - R(h) \rvert \le \sqrt{\log(2/\delta)/(2N)}$이다. 시험 집합에서 잰 오류율의 신뢰 폭이 이것이다.

**유한한 집합: union bound.** $\mathcal{H}$에 함수가 $\lvert\mathcal{H}\rvert$개 있고 그중 하나를 데이터로 고른다면, 모든 함수에 대해 동시에 성립해야 하므로 각 사건의 확률을 더한다(union bound). 그러면 확률 $1 - \delta$ 이상으로 모든 $h \in \mathcal{H}$에 대해

$$
R(h) \le \hat{R}_N(h) + \sqrt{\frac{\log\lvert\mathcal{H}\rvert + \log(2/\delta)}{2N}}
$$

이다. 이 식에 핵심이 다 들어 있다. 간극은 $1/\sqrt{N}$으로 줄고, 집합이 클수록($\log\lvert\mathcal{H}\rvert$) 커진다. 같은 정확도 $\epsilon$을 보장하는 데 필요한 표본 수, 즉 **표본 복잡도**(sample complexity)는 $N \gtrsim (\log\lvert\mathcal{H}\rvert + \log(1/\delta))/\epsilon^2$이다. "확률 $1 - \delta$ 이상으로(probably), 오차 $\epsilon$ 이내(approximately correct)"의 보장을 다항식 크기의 표본으로 주는 학습을 **PAC 학습**(probably approximately correct)이라 한다(Shalev-Shwartz & Ben-David, 2014).

::quiz{id="hoeffding-sample-size"}

**VC 차원.** 실제 가설 집합(모든 직선, 모든 신경망)은 무한하므로 $\log\lvert\mathcal{H}\rvert$를 쓸 수 없다. 대신 "$N$개의 점에 레이블을 어떻게 붙여도 맞출 수 있는가"를 센다. 점 $N$개의 가능한 모든 $2^N$가지 이진 레이블을 $\mathcal{H}$가 모두 구현하면 $\mathcal{H}$가 그 점들을 **산산조각낸다**(shatter)고 하고, 산산조각낼 수 있는 점 집합의 최대 크기를 **VC 차원** $d_{\mathrm{VC}}$라 한다. 평면의 직선 분류기(bias 포함)는 일반적인 위치의 점 3개는 산산조각내지만 4개는 못 하므로 $d_{\mathrm{VC}} = 3$이고, 일반적으로 $\mathbb{R}^d$의 선형 분류기는 $d + 1$이다. VC bound는 $\log\lvert\mathcal{H}\rvert$ 자리에 대략 $d_{\mathrm{VC}}\log(N/d_{\mathrm{VC}})$가 들어간 꼴이다. VC 차원이 유한하면 PAC 학습이 가능하다.

**Rademacher 복잡도.** VC 차원은 데이터 분포와 무관한 최악의 경우를 본다. 실제 데이터에서 용량을 재는 방법으로 **Rademacher 복잡도**가 있다. 무작위 부호 $\sigma_i \in \{-1, +1\}$(동전 던지기)를 레이블로 주었을 때, 함수 집합이 그 잡음과 얼마나 상관될 수 있는지를 잰다.

$$
\hat{\mathfrak{R}}_N(\mathcal{F}) = \mathbb{E}_{\sigma}\left[\sup_{f \in \mathcal{F}} \frac{1}{N}\sum_{i=1}^{N}\sigma_i f(x_i)\right].
$$

함수 집합이 어떤 무작위 레이블에도 잘 맞으면 값이 크다. 손실 함수족의 Rademacher 복잡도로 $R \le \hat{R}_N + 2\mathfrak{R}_N + O(\sqrt{\log(1/\delta)/N})$ 꼴의 bound를 얻는다. 직관은 단순하다. **무작위 레이블을 잘 맞출 수 있는 모델은, 진짜 레이블을 잘 맞췄다는 사실로부터 아무것도 보장받지 못한다.**

이 직관을 직접 본다. 입력과 무관한 무작위 레이블 $\pm 1$을 표본 100개에 붙이고, 특징 수 $p$를 늘리며 선형 모델(최소 노름 최소제곱 해의 부호)로 맞춘다.

```python {exec="browser" session="gen"}
rng = np.random.default_rng(1)
n = 100
labels = rng.choice([-1.0, 1.0], size=n)           # 입력과 무관한 레이블, (N,)
X_all = rng.normal(size=(n + 1000, 200))           # 특징 200개까지, (N + 1000, 200)
print("특징 수 p   학습 정확도   시험 정확도")
for p in (5, 20, 50, 90, 100, 150):
    X, X_te = X_all[:n, :p], X_all[n:, :p]
    w = np.linalg.pinv(X) @ labels                 # 최소 노름 최소제곱, (p,)
    acc_tr = np.mean(np.sign(X @ w) == labels)
    y_te_rand = rng.choice([-1.0, 1.0], size=1000)
    acc_te = np.mean(np.sign(X_te @ w) == y_te_rand)
    print(f"{p:6d}        {acc_tr:.2f}          {acc_te:.2f}")
```

```output
특징 수 p   학습 정확도   시험 정확도
     5        0.53          0.49
    20        0.68          0.49
    50        0.81          0.50
    90        1.00          0.52
   100        1.00          0.46
   150        1.00          0.50
```

특징이 5개일 때는 무작위 레이블을 거의 맞추지 못한다(0.53). 특징 수가 표본 수에 가까워지면 아무 의미 없는 레이블을 100% 맞춘다. 시험 정확도는 언제나 0.5 근처다. 학습 정확도 100%라는 같은 숫자가, 용량이 작은 모델에서는 "패턴을 찾았다"는 증거이고 용량이 큰 모델에서는 아무 증거도 아니다. 용량을 모르면 학습 성능을 해석할 수 없다.

::quiz{id="vc-dimension-plane"}

## 귀납적 편향과 정규화

**No free lunch.** 지금까지의 bound는 "$\mathcal{H}$가 작으면 일반화가 보장된다"고 말한다. 그렇다면 왜 $\mathcal{H}$를 크게 잡으면 안 되는가. **no free lunch 정리**(Wolpert, 1996)가 근본적인 답을 준다. 가능한 모든 목표 함수에 대해 균등하게 평균하면, 학습 데이터 밖의 점에서 모든 학습 알고리즘의 기대 성능이 같다. 무작위로 추측하는 알고리즘과도 같다. 어떤 알고리즘이 어떤 문제에서 잘한다면, 그만큼 다른 문제에서 못한다.

이 정리의 실용적인 의미는 비관이 아니다. **일반화는 언제나 가정에서 온다.** 학습 데이터 밖에서 무엇을 예측할지는 데이터만으로 정해지지 않고, 알고리즘이 어떤 함수를 다른 함수보다 선호하는지, 즉 **귀납적 편향**(inductive bias)이 정한다. 매끄러운 함수를 선호하는 것, 작은 계수를 선호하는 것, 이동에 불변인 특징을 선호하는 것(CNN, 18.2장), 물리 법칙을 만족하는 해를 선호하는 것(24장)이 모두 귀납적 편향이다. 귀납적 편향이 실제 문제의 구조와 맞으면 일반화하고, 맞지 않으면 일반화하지 못한다.

::quiz{id="no-free-lunch"}

**정규화는 귀납적 편향의 표현이다.** 12.2장에서 L2 정규화가 가우시안 사전분포의 MAP와 같다는 것을, 10.2장에서 정규화 항이 노름 제약과 동치라는 것을 보았다. 이 장의 언어로는, 정규화는 가설 집합을 노름이 작은 함수로 **실질적으로 줄여** 용량을 낮추는 일이다. 노름이 $B$ 이하인 선형 함수 집합 $\{x \mapsto \mathbf{w}^\top x : \lVert\mathbf{w}\rVert \le B\}$의 Rademacher 복잡도는 입력 노름이 $C$ 이하일 때 $BC/\sqrt{N}$ 이하로, 파라미터 수와 무관하다. 파라미터 수가 아니라 **노름**이 용량을 정할 수 있다는 이 관찰이 14.5장에서 과매개변수화 모델의 일반화를 설명하는 출발점이 된다.

:::note[분야 예시]
결정 구조에서 물성을 예측하는 재료 과학 문제에서, 원자 배치를 회전·평행이동해도 에너지는 같아야 한다. 이 대칭을 모델 구조에 넣으면(24.8장) 회전시킨 구조를 따로 학습할 필요가 없어 같은 데이터로 더 잘 일반화한다. 반대로 실제로는 방향에 따라 달라지는 양(예: 특정 결정 방향의 전도도)에 회전 불변성을 강제하면, 근사 오차가 데이터와 무관하게 남는다. 귀납적 편향은 도메인 지식이 맞을 때만 이득이다.
:::

## 이론과 실험의 간극

**bound는 대부분 vacuous하다.** 위의 bound에 실제 숫자를 넣으면 쓸모없는 값이 나오는 경우가 많다. 파라미터가 수백만 개인 신경망의 VC 차원은 학습 데이터 수보다 크고, bound가 보장하는 시험 오류율이 1을 넘는다(vacuous). Zhang 외(2017)는 이미지 분류에 쓰는 표준 신경망이 무작위 레이블을 100% 맞출 수 있음을 보였다. 이 신경망의 Rademacher 복잡도는 최댓값에 가깝다는 뜻이고, 따라서 용량 기반 bound로는 같은 신경망이 진짜 레이블에서 잘 일반화하는 이유를 설명할 수 없다. 위 cell 4가 그 축소판이다.

**bound가 필요로 하는 가정.**

- **i.i.d.**: 학습과 시험 데이터가 같은 분포에서 독립적으로 나와야 한다. 시계열, 공간적으로 상관된 측정, 같은 환자·같은 시료에서 나온 반복 측정은 이 가정을 깬다(15.4장의 데이터 누수, 12.6장의 pseudoreplication).
- **고정된 가설 집합**: 데이터를 본 뒤 모델 구조나 하이퍼파라미터를 바꾸면, 실제로 탐색한 집합은 처음 정한 $\mathcal{H}$보다 크다.
- **시험 데이터의 일회성**: 시험 집합으로 모델을 여러 번 고르면, 시험 오류율은 더 이상 고른 모델의 불편 추정이 아니다.

마지막 가정이 깨지는 경우를 계산해 본다. 참 오류율이 모두 0.30인 모델 100개를 시험 데이터 500개로 평가하고 가장 좋은 것을 고른다.

```python {exec="browser" session="gen"}
rng = np.random.default_rng(2)
true_err, n_test, k = 0.30, 500, 100               # 모든 모델의 참 오류율이 0.30
best = []
for _ in range(2000):
    errs = rng.binomial(n_test, true_err, size=k) / n_test   # 모델 k개의 시험 오류율, (k,)
    best.append(errs.min())
delta = 0.05
eps_one = np.sqrt(np.log(2 / delta) / (2 * n_test))          # Hoeffding, 모델 1개
eps_k = np.sqrt(np.log(2 * k / delta) / (2 * n_test))        # union bound, 모델 k개
print(f"고른 모델의 시험 오류율 평균 {np.mean(best):.3f}  (참값 {true_err})")
print(f"Hoeffding 폭: 모델 1개 ±{eps_one:.3f},  모델 {k}개 ±{eps_k:.3f}")
print(f"참값 - 고른 값 > 모델 1개 폭인 비율 {np.mean(true_err - np.array(best) > eps_one):.2f}")
```

```output
고른 모델의 시험 오류율 평균 0.249  (참값 0.3)
Hoeffding 폭: 모델 1개 ±0.061,  모델 100개 ±0.091
참값 - 고른 값 > 모델 1개 폭인 비율 0.12
```

모든 모델이 똑같이 0.30인데, 고른 모델은 평균 0.249로 보고된다. 모델 하나에 대한 95% 신뢰 폭(±0.061)을 그대로 쓰면, 12%의 경우 참값이 그 폭 밖에 있다. 명목상 5%여야 할 실패율이 두 배를 넘는다. 고른 모델 수를 반영한 union bound의 폭(±0.091)이어야 이 낙관을 덮는다. 하이퍼파라미터 탐색(22.3장)이나 여러 아키텍처 비교가 같은 시험 집합을 반복해서 쓰면 정확히 이 일이 생긴다.

**이 장의 이론이 주는 것.** bound가 숫자로 맞지 않더라도, 이 장의 틀은 실험을 설계하고 해석하는 데 그대로 쓰인다. 학습 오차와 시험 오차를 구분하고, 오차의 출처를 나누고, 데이터 크기를 바꿔 학습 곡선을 그리고, 시험 데이터를 한 번만 쓰고, 모델 선택에 쓴 데이터와 최종 보고에 쓴 데이터를 분리하는 것이다. 과매개변수화 모델에서 고전적 그림이 어떻게 바뀌는지는 14.5장에서 다룬다.

## 실습

다항식 회귀에서 차수와 데이터 크기를 바꾸어 학습·검증 오차 곡선을 그리고 오차 분해로 해석한다. 코드는 `labs/information-and-learning-theory/generalization-basics/`에 두고, 그림은 `outputs/`에 저장한다.

1. 본문의 사인 곡선 데이터(잡음 표준편차 0.3)에서 차수 0–10, 데이터 크기 $N \in \{15, 30, 60, 120, 480\}$의 격자를 만든다. 각 칸에서 학습 MSE와 검증 MSE(반복 100번 이상의 중앙값)를 계산한다.
2. $N$마다 차수에 대한 검증 오차 곡선을 그리고, 최적 차수가 $N$과 함께 어떻게 이동하는지 기록한다.
3. 차수 3과 7에 대해 학습 곡선을 그리고, 두 곡선이 다가가는 값을 잡음 분산(0.09)과 각 차수의 근사 오차(큰 $N$에서 잰 bias²)의 합과 비교한다.
4. bias–variance 분해(cell 2)를 $N = 20$과 $N = 200$에서 반복해, 데이터가 늘면 bias와 variance 중 무엇이 주로 줄어드는지 확인한다.
5. 결과를 "이 문제에서 다음에 할 일은 데이터 추가인가, 모델 변경인가"라는 질문에 대한 답으로 한 문단 쓴다.

이어서 두 연습문제를 푼다.

::exercise{id="bias-variance-decomposition"}

::exercise{id="hoeffding-bound"}

## 실패 사례

- **낮은 training loss를 좋은 모델의 증거로 보는 것.** 학습 손실은 고른 모델에 대해 낙관적인 추정이다(본문 cell 1에서 경험 위험 0.042, 모집단 위험 0.629). 용량이 충분히 크면 무작위 레이블도 100% 맞춘다. 학습에 쓰지 않은 데이터에서 잰 위험만 일반화의 근거가 된다.
- **같은 시험 집합으로 여러 번 고르고 그 시험 성능을 보고하는 것.** 시험 집합이 사실상 학습(선택)에 쓰여 낙관적으로 편향된다. 검증 집합으로 고르고, 시험 집합은 최종 보고에 한 번만 쓴다.
- **i.i.d.가 아닌 데이터를 무작위로 나누어 일반화를 주장하는 것.** 같은 시료·같은 기간의 측정이 학습과 시험에 나뉘어 들어가면, 시험 성능은 새 시료·새 기간에 대한 일반화를 말해 주지 않는다.
- **"모델이 크면 과적합한다"를 데이터 크기와 무관한 규칙으로 쓰는 것.** 학습 곡선에서 보았듯 같은 모델이 작은 $N$에서는 지고 큰 $N$에서는 이긴다.

## 통과 기준

- 낮은 training loss가 좋은 일반화를 보장하지 않는 이유를 세 관점에서 설명한다. 데이터(경험 위험의 낙관 편향, i.i.d. 가정), 모델(용량과 무작위 레이블), 평가(시험 집합의 반복 사용).
- 주어진 학습 곡선이나 bias–variance 표를 보고, 근사·추정·최적화 오차 중 무엇이 지배적인지와 다음에 할 일을 말한다.
- 유한 가설 집합의 Hoeffding–union bound로 필요한 표본 수를 계산하고, 이런 bound가 큰 신경망에서 vacuous해지는 이유를 설명한다.
- 연습문제 두 개(`bias-variance-decomposition`, `hoeffding-bound`)의 테스트를 통과한다.

## 참고 문헌

- Geman, Bienenstock, Doursat (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1–58. — 함수 추정에서의 bias–variance 분해
- Wolpert (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation*, 8(7), 1341–1390. — no free lunch 정리
- Bottou, Bousquet (2008). The tradeoffs of large scale learning. *Advances in Neural Information Processing Systems 20 (NIPS 2007)*, 161–168. — 근사·추정·최적화 오차의 분해
- Shalev-Shwartz, Ben-David (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press. — ERM, PAC 학습, VC 차원, Rademacher 복잡도의 표준 교과서
- Zhang, Bengio, Hardt, Recht, Vinyals (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*. arXiv:1611.03530. — 신경망이 무작위 레이블을 맞춘다는 실험과 용량 기반 설명의 한계
