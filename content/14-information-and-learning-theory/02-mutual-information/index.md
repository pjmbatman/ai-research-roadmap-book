---
id: information-and-learning-theory/mutual-information
title: "상호정보량과 표현"
title_en: "Mutual Information and Representations"
tier: advanced
status: draft
prerequisites: [information-and-learning-theory/entropy-and-kl]
---

# 상호정보량과 표현

두 변수가 "관련이 있다"는 말을 수로 쓰는 가장 흔한 방법은 상관계수다. 그러나 상관계수는 **선형** 관계만 본다. $Y = X^2$처럼 $X$를 알면 $Y$가 완전히 정해지는데도 상관계수가 0인 경우가 있다. **상호정보량**(mutual information, MI)은 관계의 모양을 가정하지 않고 "한 변수를 알면 다른 변수의 불확실성이 얼마나 줄어드는가"를 잰다. 그래서 특징 선택, 신경 부호화 분석, 그리고 자기지도 표현학습(19.8장)의 목적함수 설계에 두루 쓰인다.

문제는 추정이다. MI는 정의는 깔끔하지만 데이터에서 값을 구하기가 아주 어렵다. 히스토그램으로 추정하면 bin 수에 따라 값이 몇 배씩 달라지고, 신경망으로 하한을 추정하면 batch 크기가 값의 상한을 정한다. 이 장은 MI의 정의와 성질, 추정량의 편향, 그리고 contrastive learning과 information bottleneck이 MI로 어떻게 설명되는지, 그 설명이 어디까지 맞는지를 다룬다.

## 학습 목표

- 상호정보량을 정의하고 성질을 사용한다
- 상호정보량 추정량의 편향과 한계를 진단한다
- 표현학습 목적함수와 정보량의 관계를 이해한다

## 상호정보량

**왜 필요한가.** 측정 변수 수백 개 중 목표와 관련 있는 변수를 고르거나, 학습된 표현이 입력의 어떤 정보를 담는지 묻거나, 두 센서 신호가 같은 현상을 보는지 판단할 때, 선형 관계만 보는 상관계수로는 놓치는 관계가 있다.

**정의.** 두 확률변수 $X$, $Y$의 상호정보량은 결합분포와, 둘이 독립이라고 가정한 분포 사이의 KL divergence다(14.1장).

$$
I(X; Y) = D_{\mathrm{KL}}\big(p(x, y) \,\|\, p(x)\,p(y)\big) = \sum_{x, y} p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)}.
$$

엔트로피로 쓰면

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)
$$

이다. 여기서 $H(X \mid Y) = -\sum_{x,y} p(x,y)\log p(x \mid y)$는 $Y$를 안 뒤에 남는 $X$의 불확실성(조건부 엔트로피)이다. 첫 번째 꼴이 해석을 준다. **$Y$를 관측해서 줄어든 $X$의 불확실성**이 MI다.

**성질.**

- $I(X; Y) \ge 0$이고, 등호는 $X$와 $Y$가 **독립**일 때만 성립한다(KL의 비음수성). 상관계수 0은 독립을 뜻하지 않지만, MI 0은 독립을 뜻한다.
- 대칭이다: $I(X; Y) = I(Y; X)$. KL은 비대칭이지만, MI는 두 변수를 바꿔도 같은 두 분포를 비교한다.
- 이산 변수에서는 $I(X; Y) \le \min(H(X), H(Y))$다. $Y$가 $X$의 함수이면 $H(Y \mid X) = 0$이므로 $I(X; Y) = H(Y)$다.
- 연속 변수에서도 KL로 정의하므로 단위를 바꿔도 변하지 않는다. 미분 엔트로피와 달리 음수가 되지 않는다. 상관계수 $\rho$인 이변량 가우시안에서는 $I = -\frac12\log(1 - \rho^2)$이다.

$X \in \{-1, 0, 1\}$이 같은 확률로 나오고 $Y = X^2$인 경우를 결합확률 표로 계산한다.

```python {exec="browser" session="mi"}
import numpy as np

def H(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))                  # nat

def mutual_info(pxy):                              # pxy: 결합확률 표, (|X|, |Y|)
    return H(pxy.sum(1)) + H(pxy.sum(0)) - H(pxy.ravel())

xs = np.array([-1, 0, 1])                          # X는 세 값 중 하나, 같은 확률
ys = np.array([0, 1])
pxy = np.array([[1 / 3 * (y == x**2) for y in ys] for x in xs])   # Y = X², (3, 2)
EX = np.sum(pxy * xs[:, None]); EY = np.sum(pxy * ys[None, :])
cov = np.sum(pxy * xs[:, None] * ys[None, :]) - EX * EY
print(f"Y = X²       Cov(X, Y) = {abs(cov):.3f}   I(X; Y) = {mutual_info(pxy):.4f} nat   H(Y) = {H(pxy.sum(0)):.4f}")
indep = np.outer([0.2, 0.3, 0.5], [0.6, 0.4])      # 독립: p(x, y) = p(x) p(y)
print(f"독립         I(X; Y) = {abs(mutual_info(indep)):.4f} nat")
```

```output
Y = X²       Cov(X, Y) = 0.000   I(X; Y) = 0.6365 nat   H(Y) = 0.6365
독립         I(X; Y) = 0.0000 nat
```

공분산은 0인데 MI는 $H(Y)$와 같다. $Y$가 $X$의 함수여서 $X$를 알면 $Y$의 불확실성이 모두 사라지기 때문이다.

::quiz{id="zero-correlation-mi"}

**Data processing inequality.** $X \to Y \to Z$가 Markov 사슬이면, 즉 $Z$가 $Y$만 보고 만들어지면(결정적 함수 $Z = g(Y)$이든 확률적 처리이든)

$$
I(X; Z) \le I(X; Y)
$$

이다. **처리는 정보를 늘리지 못한다.** 측정 $Y$를 어떻게 가공해도, 가공된 결과가 원래 대상 $X$에 대해 담는 정보는 측정 자체보다 많아질 수 없다. 네 상태 $X$를 잡음 섞인 센서 $Y$로 읽고, 그 결과를 다시 두 묶음으로 합치거나 무작위 채널에 통과시켜 확인한다.

```python {exec="browser" session="mi"}
rng = np.random.default_rng(0)
px = np.full(4, 0.25)                              # X: 네 가지 상태, (4,)
noisy = 0.7 * np.eye(4) + 0.1                      # p(y | x): 70% 정답 + 잡음, 행 합 1, (4, 4)
pxy = px[:, None] * noisy                          # p(x, y), (4, 4)
merge = np.array([[1, 0], [1, 0], [0, 1], [0, 1]]) # Z = g(Y): 네 상태를 두 묶음으로, (4, 2)
pxz = pxy @ merge                                  # p(x, z), (4, 2)
rand = rng.dirichlet(np.ones(3), size=4)           # 임의의 확률적 처리 p(w | y), (4, 3)
pxw = pxy @ rand                                   # p(x, w), (4, 3)
print(f"I(X; Y) = {mutual_info(pxy):.4f}")
print(f"I(X; Z) = {mutual_info(pxz):.4f}   (Z = Y를 두 묶음으로 합침)")
print(f"I(X; W) = {mutual_info(pxw):.4f}   (W = Y에 무작위 채널 적용)")
```

```output
I(X; Y) = 0.4460
I(X; Z) = 0.1361   (Z = Y를 두 묶음으로 합침)
I(X; W) = 0.0394   (W = Y에 무작위 채널 적용)
```

어떤 처리든 $I(X; Y) = 0.446$ nat을 넘지 못한다. 신경망의 층을 거칠 때도 마찬가지다. 입력 $X$에서 층 $T_1, T_2, \dots$로 가는 사슬에서 $I(Y; T_k)$는 $k$가 커질수록 줄거나 같다($Y$는 레이블). 그래도 깊은 층의 표현이 유용한 이유는 정보를 **만들기** 때문이 아니라, 필요한 정보를 선형 분류기 같은 단순한 읽기 장치로 꺼낼 수 있는 꼴로 **바꾸기** 때문이다. MI는 정보가 어떤 꼴로 담겼는지는 재지 않는다.

::quiz{id="data-processing"}

:::note[분야 예시]
신경과학에서는 자극 $S$와 뉴런의 스파이크 반응 $R$ 사이의 $I(S; R)$로 뉴런이 자극에 대해 얼마나 많은 정보를 전달하는지 잰다. 스파이크 열을 시간 bin으로 나누어 단어(word)로 만든 뒤 히스토그램으로 확률을 추정하는 방법이 오래 쓰였는데, 바로 다음 절의 편향 문제가 이 분야에서 집중적으로 연구되었다(Paninski, 2003). 기후 과학에서도 두 지역의 기후 지수 사이의 비선형 연관을 MI로 재어 원격 상관(teleconnection) 네트워크를 만드는 연구가 있다. 두 경우 모두 추정량의 편향을 보정하지 않으면 데이터가 적은 쪽에서 "관계가 더 강하다"는 잘못된 결론이 나온다.
:::

## 추정의 어려움

**왜 필요한가.** 실제로 가진 것은 분포가 아니라 표본이다. MI를 쓰려면 표본에서 추정해야 하고, 추정값이 참값과 얼마나 다를 수 있는지 알아야 결론을 믿을 수 있다.

**히스토그램 추정량.** 가장 직접적인 방법은 연속 변수를 bin으로 나누어 결합 히스토그램을 만들고, 그 상대도수를 확률로 보고 앞의 식에 넣는 것이다(plug-in 추정량). 이 추정량에는 방향이 반대인 두 편향이 있다.

- **bin이 너무 굵으면 아래로 편향된다.** 연속 변수를 굵게 양자화하는 것은 하나의 처리이므로, data processing inequality에 따라 양자화된 변수의 MI는 원래 MI보다 작거나 같다.
- **bin이 너무 가늘면 위로 편향된다.** 칸이 많고 칸당 표본이 적으면, 우연한 몰림이 의존성처럼 보인다. 칸당 표본이 충분할 때의 근사 편향은 $(B_x - 1)(B_y - 1)/(2N)$ 정도다($B_x, B_y$는 bin 수, $N$은 표본 수). 독립인 변수에서도 MI가 양수로 나온다.

상관 0.6인 가우시안 쌍(참 MI 0.223 nat)과 독립인 쌍(참 MI 0)에서, 표본 수와 bin 수를 바꾸며 20번 반복한 평균을 본다.

```python {exec="browser" session="mi"}
def mi_hist(x, y, bins):
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    return mutual_info(counts / counts.sum())

rho = 0.6
true_mi = -0.5 * np.log(1 - rho**2)                # 이변량 가우시안의 MI [nat]
print(f"참값: 상관 0.6 → {true_mi:.3f} nat,  독립 → 0")
print("   N   bins   상관 0.6    독립")
for n in (200, 2000):
    for bins in (5, 20, 50):
        est = []
        for seed in range(20):
            r = np.random.default_rng(seed)
            x, e = r.normal(size=(2, n))           # (N,), (N,)
            y = rho * x + np.sqrt(1 - rho**2) * e
            est.append((mi_hist(x, y, bins), mi_hist(x, e, bins)))
        m = np.mean(est, axis=0)
        print(f"{n:5d}  {bins:4d}   {m[0]:.3f}      {m[1]:.3f}")
```

```output
참값: 상관 0.6 → 0.223 nat,  독립 → 0
   N   bins   상관 0.6    독립
  200     5   0.207      0.040
  200    20   0.753      0.627
  200    50   1.854      1.790
 2000     5   0.153      0.004
 2000    20   0.269      0.074
 2000    50   0.517      0.360
```

결과를 읽는 법은 세 가지다.

- **같은 데이터에서 bin 수만 바꿔도 추정값이 약 9배 달라진다**($N = 200$에서 0.207 → 1.854). bin 50개, 표본 200개면 칸이 2500개라 대부분의 칸이 비어 있고, 독립인 변수의 MI도 1.79 nat으로 나온다.
- **표본이 많고 bin이 굵을 때는 아래로 편향된다**($N = 2000$, bin 5에서 0.153 < 0.223). 독립인 쌍의 편향이 0.004로 거의 없으므로, 이 차이는 양자화로 잃은 정보다.
- **$N = 200$, bin 5의 0.207이 참값에 가장 가까운 것은 우연이다.** 아래로 가는 양자화 편향과 위로 가는 표본 편향이 상쇄된 결과다. 독립인 쌍에 같은 설정을 쓰면 0.040이 나오는 것이 그 증거다. 참값을 모르는 실제 문제에서는 이런 상쇄를 믿을 수 없다.

::quiz{id="histogram-bias"}

실무에서 쓰는 대응은 세 가지다. 첫째, 독립인 기준선을 함께 계산한다. 한 변수의 순서를 무작위로 섞은(permutation) 데이터에서 같은 추정량을 돌리면, 그 값이 "관계가 없을 때 이 추정량이 내는 값"이다. 둘째, 같은 bin 설정과 같은 $N$에서만 MI를 비교한다. 셋째, bin에 덜 민감한 추정량을 쓴다. 최근접 이웃 거리에 기반한 KSG 추정량(Kraskov 외, 2004)이 대표적이고, 집필 시점 예시로 scikit-learn의 `mutual_info_regression`이 이 계열의 추정량을 구현한다(사용 전에 문서에서 현재 구현을 확인한다).

**고차원과 변분 하한.** 표현학습에서 MI를 재고 싶은 변수는 수백 차원이다. 히스토그램은 차원마다 bin이 곱해지므로 쓸 수 없다. 그래서 신경망 critic $f(x, y)$를 학습해 MI의 **하한**을 최대화하는 방법이 쓰인다. 두 가지가 대표적이다.

- **MINE**(Belghazi 외, 2018): KL의 Donsker–Varadhan 표현 $I(X; Y) \ge \mathbb{E}_{p(x,y)}[f] - \log\mathbb{E}_{p(x)p(y)}[e^{f}]$를 critic에 대해 최대화한다. 두 번째 항을 미니배치로 추정하면 편향이 생기고, 분산이 크다.
- **InfoNCE**(van den Oord 외, 2018): 짝지어진 표본 $(x_i, y_i)$ $K$쌍으로 만든 다음 양을 최대화한다.

$$
I(X; Y) \ \ge\ I_{\text{NCE}} = \mathbb{E}\left[\frac{1}{K}\sum_{i=1}^{K}\log\frac{e^{f(x_i, y_i)}}{\frac{1}{K}\sum_{j=1}^{K} e^{f(x_i, y_j)}}\right].
$$

$x_i$마다 진짜 짝 $y_i$를 나머지 $K - 1$개의 가짜 짝 $y_j$ 사이에서 골라내는 $K$-분류 문제의 로그 확률이다. 이 하한에는 구조적인 천장이 있다. 로그 안의 분수는 분모의 합에 분자 자신이 들어 있으므로 $K$를 넘을 수 없고, 따라서 $I_{\text{NCE}} \le \log K$다(Poole 외, 2019). 참 MI가 $\log K$보다 크면 critic이 아무리 좋아도 InfoNCE는 참값에 닿지 못한다. 성분마다 상관 0.9인 10차원 가우시안 쌍(참 MI 8.30 nat)에 **최적 critic** $f(x, y) = \log p(y \mid x) - \log p(y)$를 그대로 넣고 batch 크기 $K$를 바꿔 본다.

```python {exec="browser" session="mi"}
rho, d = 0.9, 10                                   # 성분마다 상관 0.9인 10차원 쌍
true_mi = -0.5 * d * np.log(1 - rho**2)            # 성분별 MI의 합

def critic(x, y):                                  # 최적 critic log p(y|x) - log p(y) (상수 제외), (K, K)
    sq = (y**2).sum(1)[None, :] - 2 * rho * x @ y.T + rho**2 * (x**2).sum(1)[:, None]   # ‖y_j - ρx_i‖²
    return -sq / (2 * (1 - rho**2)) + (y**2).sum(1)[None, :] / 2

r = np.random.default_rng(1)
print(f"참 MI = {true_mi:.3f} nat")
for K in (4, 16, 64, 256, 1024):
    vals = []
    for _ in range(10):
        x = r.normal(size=(K, d))                  # (K, d)
        y = rho * x + np.sqrt(1 - rho**2) * r.normal(size=(K, d))
        S = critic(x, y)                           # S[i, j] = f(x_i, y_j)
        lse = np.log(np.mean(np.exp(S - S.max(1, keepdims=True)), axis=1)) + S.max(1)
        vals.append(np.mean(np.diag(S) - lse))     # mean_i [f(x_i,y_i) - log(1/K Σ_j e^f(x_i,y_j))]
    print(f"K = {K:4d}   InfoNCE = {np.mean(vals):.3f}   log K = {np.log(K):.3f}")
```

```output
참 MI = 8.304 nat
K =    4   InfoNCE = 1.386   log K = 1.386
K =   16   InfoNCE = 2.766   log K = 2.773
K =   64   InfoNCE = 4.078   log K = 4.159
K =  256   InfoNCE = 5.343   log K = 5.545
K = 1024   InfoNCE = 6.379   log K = 6.931
```

critic이 완벽한데도 InfoNCE는 $\log K$를 따라 올라갈 뿐이고, $K = 1024$에서도 참값 8.30에 한참 못 미친다. $K$가 작으면 진짜 짝을 고르는 문제가 너무 쉬워서 $\log K$에 붙고, $K$가 커지면 비로소 critic의 품질과 참 MI가 값에 영향을 준다. 참 MI 8.3 nat에 다가가려면 $K$가 $e^{8.3} \approx 4000$보다 훨씬 커야 한다.

::quiz{id="infonce-ceiling"}

이 천장은 InfoNCE만의 문제가 아니다. McAllester와 Stratos(2020)는 분포에 대한 가정 없이 표본 $N$개로 얻는 높은 신뢰도의 MI 하한은 $O(\log N)$을 넘을 수 없음을 보였다. MI가 큰 상황에서 "MI를 정확히 추정했다"는 주장은 그 자체로 의심해야 한다는 뜻이다.

## 정보 병목

**아이디어.** 좋은 표현 $T$는 입력 $X$의 모든 것을 담는 것이 아니라, 예측하려는 $Y$에 관련된 것만 담아야 한다는 직관이 있다. information bottleneck(IB)은 이를 최적화 문제로 쓴다(Tishby, Pereira, Bialek, 1999).

$$
\min_{p(t \mid x)} \ I(X; T) - \beta\, I(T; Y).
$$

$I(X; T)$는 표현이 입력에서 가져온 정보량(압축의 정도), $I(T; Y)$는 표현이 목표에 대해 담은 정보량, $\beta > 0$은 둘 사이의 trade-off를 정한다. $X \to T$ 사슬에서 data processing inequality에 따라 $I(T; Y) \le I(X; Y)$이므로, 표현은 $X$가 가진 $Y$에 대한 정보를 넘을 수 없다. IB의 해는 그 한계 안에서 불필요한 정보를 버린 표현이다.

**딥러닝의 IB 이론과 논쟁.** Shwartz-Ziv와 Tishby(2017)는 작은 신경망의 층별 $(I(X; T), I(T; Y))$를 학습 중에 추적해, 학습이 먼저 $I(T; Y)$를 늘리는 단계와 이후 $I(X; T)$를 줄이는 **압축** 단계로 나뉘고, 압축이 일반화와 관련된다고 주장했다. 이 주장은 반박을 받았다. Saxe 외(2018)는 압축 단계가 tanh처럼 양쪽으로 포화하는 활성화에서는 나타나지만 ReLU에서는 나타나지 않고, 압축과 일반화 사이의 인과관계도 확인되지 않는다고 보고했다. 그리고 이런 결과가 결정적 신경망에서 유한한 MI를 계산하기 위해 필요한 가정(활성화의 binning)에 의존한다고 지적했다.

이 논쟁의 핵심은 앞 절과 같은 추정 문제다. 연속 입력을 받는 **결정적** 신경망에서는 $T = f(X)$가 $X$의 함수이므로, $f$가 단사이면 $I(X; T)$는 무한대이거나 정의되지 않는다. 유한한 값을 얻으려면 활성화를 bin으로 나누거나 잡음을 더해야 하고, 그렇게 얻은 "압축"은 신경망의 성질이 아니라 binning의 성질일 수 있다. tanh가 포화하면 많은 입력이 같은 bin에 몰려 히스토그램 MI가 줄어드는 것이 그 예다.

::quiz{id="deterministic-ib"}

**연구에서의 의미.** IB는 표현학습을 생각하는 유용한 틀이고, 변분 근사로 바꾼 목적함수(잡음을 명시적으로 넣은 확률적 encoder)는 실제로 쓰인다. 하지만 "신경망은 학습 중에 입력 정보를 압축하므로 일반화한다"는 문장을 확립된 사실로 인용하면 안 된다. 결정적 신경망의 MI를 보고하는 논문을 읽을 때는 MI를 어떻게 정의하고 추정했는지(bin 크기, 잡음 수준)부터 확인한다.

**contrastive learning과 MI.** 자기지도학습의 contrastive 목적함수(19.8장)는 같은 대상의 두 view $v_1, v_2$(예: 같은 이미지의 다른 증강, 같은 분자의 두 표현)를 encoder로 보낸 표현 $z_1, z_2$에 대해 InfoNCE를 최대화한다. 그래서 흔히 "두 view 사이의 MI를 최대화한다"고 설명한다. 이 해석은 방향만 맞다.

- InfoNCE는 MI의 하한이고, batch 크기 $K$가 정하는 천장 $\log K$가 있다. 학습이 진행되며 손실이 $\log K$ 근처의 값에 도달했다면 MI에 대해서는 그 이상 말할 수 없다.
- MI는 encoder를 가역 변환으로 바꿔도 변하지 않는다. 그런데 같은 MI를 갖는 표현이라도 downstream 과제에 쓰기 좋은 정도는 크게 다를 수 있다. contrastive learning이 좋은 표현을 만드는 이유를 MI만으로는 설명할 수 없고, encoder 구조와 증강의 선택(귀납적 편향)이 큰 역할을 한다.

:::note[분야 예시]
단일세포 생물학에서 같은 세포의 RNA 측정과 단백질 측정처럼 서로 다른 modality를 contrastive 목적함수로 정렬하는 연구가 있다. 두 측정이 같은 세포 상태를 본다는 가정 아래 두 표현 사이의 InfoNCE를 최대화하는 것이다. 이때 손실 값으로 "두 modality가 공유하는 정보가 몇 nat이다"라고 해석하면 앞의 천장 문제에 걸린다. 손실은 학습 신호로만 쓰고, 표현의 품질은 세포 유형 분류 같은 별도의 평가로 확인한다.
:::

## 실습

InfoNCE 하한을 직접 **학습**해 MI를 추정하고, batch 크기에 따른 천장 효과를 확인한다. 본문의 cell 4는 최적 critic을 알고 있다고 가정했다. 실습에서는 critic을 학습한다.

1. 상관 $\rho$인 $d$차원 가우시안 쌍(cell 4와 같은 데이터)에서 critic을 $f(x, y) = x^\top \mathbf{W} y + y^\top \mathbf{V} y$ 꼴로 둔다. 이 데이터의 최적 critic은 이 꼴에 $\lVert x \rVert^2$ 항을 더한 것인데, 그 항은 행($x_i$)마다 상수여서 InfoNCE에서 약분된다. 따라서 이 critic족 안에 최적 critic이 있다.
2. InfoNCE의 $\mathbf{W}$에 대한 gradient를 손으로 유도해(행마다 softmax의 cross-entropy와 같은 꼴이다) NumPy로 경사상승을 구현한다. 코드는 `labs/information-and-learning-theory/mutual-information/`에 둔다.
3. $d \in \{1, 5, 10\}$, $K \in \{16, 64, 256\}$에서 학습된 InfoNCE를 참 MI, $\log K$와 함께 표로 만든다. 참 MI가 $\log K$보다 작을 때와 클 때 결과가 어떻게 다른지 설명한다.
4. 같은 데이터에서 히스토그램 추정량($d = 1$만 가능)과 비교하고, 고차원에서 히스토그램을 쓸 수 없는 이유를 칸의 수로 설명한다.

이어서 두 연습문제를 푼다. 첫째는 결합 도수표에서 plug-in MI를 계산하고 편향을 확인하고, 둘째는 점수 행렬에서 InfoNCE를 계산해 천장을 확인한다.

::exercise{id="plugin-mi"}

::exercise{id="infonce-bound"}

## 실패 사례

- **InfoNCE 값을 MI의 정확한 추정으로 해석하는 것.** InfoNCE는 하한이고 $\log K$를 넘지 못한다. "batch 크기를 늘렸더니 MI가 증가했다"는 관찰은 표현이 좋아졌다는 증거가 아니라 천장이 올라간 결과일 수 있다. 같은 $K$에서만 비교하고, 값이 $\log K$에 가까우면 천장에 걸렸다고 본다.
- **히스토그램 MI를 bin 설정과 표본 수를 밝히지 않고 보고하거나 비교하는 것.** 같은 데이터에서도 bin 수에 따라 값이 몇 배씩 달라진다. 표본 수가 다른 두 조건의 MI를 비교하면 표본이 적은 쪽이 체계적으로 크게 나온다.
- **상관계수 0을 "관계없음"으로 보고하는 것.** 비선형 관계는 상관계수에 잡히지 않는다. 변수 선택에서 상관계수만으로 걸러 내면 $Y = X^2$ 같은 변수를 버린다.
- **결정적 신경망의 층별 MI를 추정 방법과 분리해 해석하는 것.** binning이나 잡음 수준에 따라 "압축"이 나타나기도 하고 사라지기도 한다.

## 통과 기준

- contrastive learning의 InfoNCE 목적함수를 MI 하한으로 설명하고, 그 하한이 batch 크기 $K$에 대해 $\log K$를 넘지 못하는 이유를 식으로 보인다.
- 히스토그램 MI 추정량이 bin 수와 표본 수에 따라 어느 방향으로 편향되는지 설명하고, permutation 기준선으로 편향을 진단한다.
- data processing inequality를 써서 "처리된 표현이 원래 입력보다 목표에 대한 정보를 더 담을 수 없다"를 설명한다.
- 연습문제 두 개(`plugin-mi`, `infonce-bound`)의 테스트를 통과한다.

## 참고 문헌

- Tishby, Pereira, Bialek (1999). The information bottleneck method. *Proceedings of the 37th Annual Allerton Conference on Communication, Control and Computing*, 368–377. arXiv:physics/0004057. — information bottleneck의 정식화
- Paninski (2003). Estimation of entropy and mutual information. *Neural Computation*, 15(6), 1191–1253. — 이산화 추정량의 편향과 한계
- Kraskov, Stögbauer, Grassberger (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138. — 최근접 이웃 기반 MI 추정량(KSG)
- van den Oord, Li, Vinyals (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748. — InfoNCE 목적함수
- Belghazi 외 (2018). Mutual information neural estimation. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 531–540. — Donsker–Varadhan 하한을 쓰는 신경망 MI 추정(MINE)
- Poole, Ozair, van den Oord, Alemi, Tucker (2019). On variational bounds of mutual information. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 5171–5180. — 변분 MI 하한들의 편향·분산과 InfoNCE의 $\log K$ 천장
- McAllester, Stratos (2020). Formal limitations on the measurement of mutual information. *Proceedings of AISTATS 2020*, PMLR 108, 875–884. — 분포 가정 없는 MI 하한은 $O(\log N)$을 넘지 못함
- Shwartz-Ziv, Tishby (2017). Opening the black box of deep neural networks via information. arXiv:1703.00810. — 정보 평면과 압축 단계 주장
- Saxe 외 (2018). On the information bottleneck theory of deep learning. *ICLR 2018*. — 압축 단계가 활성화와 MI 추정 방식에 의존한다는 반론
