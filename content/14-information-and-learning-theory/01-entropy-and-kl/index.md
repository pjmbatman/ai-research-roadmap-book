---
id: information-and-learning-theory/entropy-and-kl
title: "엔트로피·교차엔트로피·KL"
title_en: "Entropy, Cross-Entropy, and KL Divergence"
tier: core
status: draft
prerequisites: [probability/distributions]
---

# 엔트로피·교차엔트로피·KL

분류 모델을 학습할 때 쓰는 cross-entropy, VAE의 KL 항, 확산 모델과 GAN의 목적함수, 변분추론의 ELBO는 모두 "두 확률분포가 얼마나 다른가"를 재는 양이다. 이 양들은 정보이론의 세 개념, **엔트로피**, **교차엔트로피**, **KL divergence**에서 나온다. 12.2장에서 cross-entropy 손실이 categorical 분포의 음의 로그우도(NLL)와 같다는 것을 보았다. 이 장은 같은 양을 정보이론 쪽에서 다시 본다. 그러면 "왜 이 손실인가"뿐 아니라 "이 손실을 최소화한 모델이 어떤 성격을 갖는가"까지 예측할 수 있다.

특히 KL divergence는 거리처럼 보이지만 **대칭이 아니다.** $D_{\mathrm{KL}}(p \,\|\, q)$와 $D_{\mathrm{KL}}(q \,\|\, p)$를 최소화한 근사분포는 서로 아주 다르게 생긴다. 어느 방향을 쓰는지가 생성모델이 다양한 샘플을 내는지, 변분추론이 불확실성을 과소평가하는지를 정한다. 이 장의 마지막 절은 KL이 아예 쓸모없어지는 경우와, 그때 쓰는 다른 divergence를 다룬다.

## 학습 목표

- 엔트로피와 KL을 정의하고 해석한다
- forward KL과 reverse KL이 만드는 근사의 차이를 예측한다
- 분포 간 거리 척도를 비교한다

## 엔트로피

**왜 필요한가.** 관측 하나가 "얼마나 놀라운가"를 수로 잴 수 있어야 정보를 비교하고 압축하고 손실로 쓸 수 있다. 동전 던지기의 결과는 주사위 결과보다 덜 놀랍고, 앞면이 99% 나오는 동전의 결과는 그보다도 덜 놀랍다.

**정의.** 확률 $p(x)$로 일어나는 사건의 **정보량**(information content, surprisal)을 $-\log p(x)$로 정의한다. 확률이 1이면 0이고, 확률이 작을수록 커진다. 독립 사건 두 개의 확률은 곱해지므로 정보량은 더해진다. 이산 확률변수 $X$의 **엔트로피**(entropy)는 정보량의 기댓값이다.

$$
H(X) = H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x \sim p}\big[-\log p(x)\big].
$$

$p(x) = 0$인 항은 $0 \log 0 = 0$으로 둔다($t \log t \to 0$, $t \to 0^+$). 로그의 밑이 2이면 단위는 bit, 자연로그이면 nat이다. 1 nat $= 1/\ln 2 \approx 1.443$ bit다. 이 책은 손실과 맞추기 위해 따로 밝히지 않으면 자연로그(nat)를 쓴다.

엔트로피는 $0 \le H(p) \le \log K$를 만족한다($K$는 가능한 결과의 수). 한 결과만 확률 1이면 0, 모든 결과가 같은 확률이면 최대 $\log K$다. 즉 엔트로피는 분포의 **불확실성**, 결과를 미리 알 수 없는 정도다.

**직관: 부호 길이.** 엔트로피에는 구체적인 의미가 있다. 결과를 0과 1의 부호(code)로 보내야 한다면 자주 나오는 결과에 짧은 부호를, 드문 결과에 긴 부호를 주는 것이 유리하다. Shannon의 원천 부호화 정리(source coding theorem)에 따르면, 부호 길이의 기댓값은 $H(p)$ bit보다 짧을 수 없고, 결과 $x$에 길이 $\lceil -\log_2 p(x) \rceil$의 부호를 주면 $H(p) + 1$ bit 미만을 달성한다(Shannon, 1948; Cover & Thomas, 2006). 세 분포에서 확인한다.

```python {exec="browser" session="info"}
import numpy as np

def entropy_bits(p):
    p = p[p > 0]                                   # 0 log 0 = 0
    return -np.sum(p * np.log2(p))

dists = {
    "균등 8개": np.full(8, 1 / 8),
    "2의 거듭제곱": np.array([0.5, 0.25, 0.125, 0.125]),
    "일반": np.array([0.4, 0.3, 0.2, 0.1]),
}
for name, p in dists.items():
    code_len = np.ceil(-np.log2(p))                # Shannon 부호: 결과마다 ⌈-log2 p⌉ 비트
    L = np.sum(p * code_len)
    print(f"H = {entropy_bits(p):.3f} bit   평균 부호 길이 L = {L:.3f} bit   ({name})")
```

```output
H = 3.000 bit   평균 부호 길이 L = 3.000 bit   (균등 8개)
H = 1.750 bit   평균 부호 길이 L = 1.750 bit   (2의 거듭제곱)
H = 1.846 bit   평균 부호 길이 L = 2.400 bit   (일반)
```

8가지 결과가 같은 확률이면 3 bit가 필요하다. 확률이 모두 2의 거듭제곱이면 부호 길이가 엔트로피와 정확히 같고, 그렇지 않으면 $H \le L < H + 1$ 범위에 있다. 엔트로피는 "이 분포에서 나온 결과를 전달하는 데 평균적으로 필요한 최소 정보량"이다.

::quiz{id="uniform-entropy-bits"}

**미분 엔트로피.** 연속 확률변수는 밀도 $p(x)$로 같은 식을 쓴다.

$$
h(p) = -\int p(x)\log p(x)\,dx.
$$

이를 **미분 엔트로피**(differential entropy)라 한다. 이름은 비슷하지만 이산 엔트로피와 성질이 다르다. 밀도는 1보다 클 수 있으므로 $h$는 **음수가 될 수 있다.** 또 변수의 단위를 바꾸면 값이 바뀐다. $Y = cX$이면 $h(Y) = h(X) + \log\lvert c \rvert$다. 가우시안 $\mathcal{N}(\mu, \sigma^2)$의 미분 엔트로피는 $\frac12\log(2\pi e\sigma^2)$이다. 표준편차 2 mm인 측정 잡음을 m와 mm로 각각 표현해 계산한다.

```python {exec="browser" session="info"}
rng = np.random.default_rng(0)
x_m = rng.normal(0.0, 0.002, size=200_000)         # 위치 측정 잡음 [m], σ = 2 mm

def gauss_h_nats(sigma):                           # N(μ, σ²)의 미분 엔트로피 ½ log(2πeσ²)
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

for unit, scale in (("m", 1.0), ("mm", 1e3)):
    x = x_m * scale
    mc = -np.mean(-0.5 * np.log(2 * np.pi * x.std() ** 2) - x**2 / (2 * x.std() ** 2))
    print(f"단위 {unit:2s}  h = {gauss_h_nats(x.std()):7.3f} nat   (Monte Carlo {mc:7.3f})")
print(f"차이 = log(1000) = {np.log(1000):.3f} nat")
```

```output
단위 m   h =  -4.794 nat   (Monte Carlo  -4.794)
단위 mm  h =   2.113 nat   (Monte Carlo   2.113)
차이 = log(1000) = 6.908 nat
```

같은 잡음인데 m 단위에서는 $-4.79$ nat, mm 단위에서는 $2.11$ nat이다. 미분 엔트로피 하나만으로는 "불확실성이 크다·작다"를 말할 수 없고, 같은 단위의 두 값의 **차이**만 의미가 있다. 12.2장에서 연속 변수의 NLL이 단위에 따라 바뀌고 음수도 될 수 있다고 한 것과 같은 현상이다. 연속 변수의 NLL 평균은 미분 엔트로피의 추정이기 때문이다. 다행히 다음 절의 KL은 단위를 바꿔도 변하지 않는다. 밀도 비 $p/q$에서 단위가 약분되기 때문이다.

::quiz{id="differential-entropy-facts"}

:::note[분야 예시]
통계역학의 Gibbs 엔트로피 $S = -k_B\sum_i p_i \ln p_i$는 정보 엔트로피에 Boltzmann 상수를 곱한 것이다. 연속 위상공간에서 고전 엔트로피를 정의할 때 위상공간 부피를 Planck 상수 같은 기준 부피로 나누어 무차원으로 만드는 것은, 미분 엔트로피가 단위에 따라 바뀌는 문제를 기준 단위로 고정해 해결하는 것이다. 10.2장에서 본 대로, 평균 에너지 제약 아래에서 이 엔트로피를 최대화하면 Boltzmann 분포가 나온다.
:::

**연구에서의 의미.** 엔트로피는 분류 모델의 예측 불확실성을 요약하는 데 자주 쓰인다(예측분포 $\boldsymbol\pi$의 엔트로피가 크면 모델이 헷갈린다는 뜻). 능동 학습(25.3장)에서 다음에 측정할 샘플을 고를 때 쓰는 기준도 엔트로피에서 출발한다. 다만 예측 엔트로피는 모델이 **스스로 보고한** 불확실성이다. 과신하는 모델의 엔트로피는 낮게 나온다. 이것이 21장에서 calibration을 따로 검사하는 이유다.

## 교차엔트로피와 KL

**왜 필요한가.** 실제 데이터 분포 $p$를 모르는 채로 모델 $q$를 쓴다고 하자. $q$가 틀린 만큼 얼마나 손해를 보는지 재는 양이 필요하다. 학습은 이 손해를 줄이는 일이다.

**정의.** $p$에서 나온 결과를 $q$에 맞춘 부호로 보내면 평균 부호 길이는 $-\log q(x)$의 $p$-기댓값이 된다. 이것이 **교차엔트로피**(cross-entropy)다.

$$
H(p, q) = -\sum_x p(x)\log q(x) = \mathbb{E}_{x \sim p}\big[-\log q(x)\big].
$$

$p$에 맞춘 최적 부호보다 얼마나 더 긴지가 **KL divergence**(Kullback–Leibler divergence, 상대 엔트로피)다(Kullback & Leibler, 1951).

$$
D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p).
$$

연속 변수에서는 합을 적분으로 바꾼다. 이때 $H(p, q)$와 $H(p)$는 각각 단위에 따라 바뀌지만, 차이인 KL은 바뀌지 않는다.

**성질.**

- **비음수성**(Gibbs 부등식): $D_{\mathrm{KL}}(p \,\|\, q) \ge 0$이고, 등호는 $p = q$일 때만 성립한다. $-\log$가 볼록 함수이므로 Jensen 부등식(11.5장)에서 $\mathbb{E}_p[-\log(q/p)] \ge -\log\mathbb{E}_p[q/p] = -\log 1 = 0$이다.
- **비대칭성**: 일반적으로 $D_{\mathrm{KL}}(p \,\|\, q) \neq D_{\mathrm{KL}}(q \,\|\, p)$이다. 삼각부등식도 성립하지 않는다. 그래서 KL은 거리(metric)가 아니라 **divergence**라 부른다.
- **지지집합 조건**: $p(x) > 0$인데 $q(x) = 0$인 $x$가 하나라도 있으면 $D_{\mathrm{KL}}(p \,\|\, q) = \infty$다. 반대 방향은 유한할 수 있다.

```python {exec="browser" session="info"}
def kl(p, q):
    m = p > 0                                      # 0 log(0/q) = 0
    if np.any(q[m] == 0):
        return np.inf                              # p > 0인데 q = 0인 곳이 있으면 무한대
    return np.sum(p[m] * np.log(p[m] / q[m]))      # nat

def cross_entropy(p, q):
    m = p > 0
    return -np.sum(p[m] * np.log(q[m]))

p = np.array([0.70, 0.20, 0.10])
q = np.array([0.40, 0.40, 0.20])
H_p = cross_entropy(p, p)
print(f"H(p) = {H_p:.4f}   H(p,q) = {cross_entropy(p, q):.4f}   H(p) + KL(p||q) = {H_p + kl(p, q):.4f}")
print(f"KL(p||q) = {kl(p, q):.4f}   KL(q||p) = {kl(q, p):.4f}")
r = np.array([0.5, 0.5, 0.0])                      # 세 번째 결과를 불가능하다고 보는 모델
print(f"KL(p||r) = {kl(p, r)}   KL(r||p) = {kl(r, p):.4f}")
```

```output
H(p) = 0.8018   H(p,q) = 0.9856   H(p) + KL(p||q) = 0.9856
KL(p||q) = 0.1838   KL(q||p) = 0.1920
KL(p||r) = inf   KL(r||p) = 0.2899
```

마지막 줄이 지지집합 조건이다. 모델 $r$은 실제로 10% 확률로 일어나는 결과를 "불가능"이라고 말하므로 $D_{\mathrm{KL}}(p \,\|\, r)$은 무한대다. 12.2장에서 3번 중 3번 성공한 데이터의 Bernoulli MLE가 다음 실패 하나에 NLL 무한대를 받는다고 한 것이 같은 현상이다. 반대 방향 $D_{\mathrm{KL}}(r \,\|\, p)$는 0.29로 유한하다.

**교차엔트로피 최소화 = KL 최소화.** 학습에서 $p$는 고정된 데이터 분포이고 바꿀 수 있는 것은 모델 $q_\theta$뿐이다. $H(p)$는 $\theta$와 무관한 상수이므로

$$
\arg\min_\theta H(p, q_\theta) = \arg\min_\theta \big[H(p) + D_{\mathrm{KL}}(p \,\|\, q_\theta)\big] = \arg\min_\theta D_{\mathrm{KL}}(p \,\|\, q_\theta)
$$

이다. 그리고 $p$를 학습 데이터의 경험분포 $\hat{p}$로 바꾸면 $H(\hat{p}, q_\theta) = -\frac{1}{N}\sum_i \log q_\theta(x_i)$, 즉 평균 NLL이다. 정리하면 **cross-entropy 손실 최소화 = 평균 NLL 최소화(MLE) = 데이터 분포에서 모델로의 KL 최소화**다. 세 이름이 같은 최적화 문제를 가리킨다. 다만 손실의 **값**은 KL이 아니다. cross-entropy의 최솟값은 0이 아니라 $H(p)$이고, 레이블에 잡음이 있으면 $H(p) > 0$이므로 학습 손실이 0까지 내려가지 않는 것이 정상이다. 14.3장은 이 관계를 바탕으로 목적함수를 설계한다.

::quiz{id="ce-equals-kl"}

## forward KL과 reverse KL

**왜 필요한가.** 복잡한 분포 $p$를 다루기 쉬운 분포족(예: 가우시안) 안의 $q$로 근사하는 일은 변분추론(13.6장)과 생성모델(19장)에서 반복해서 나온다. KL이 비대칭이므로 어느 쪽을 최소화할지 골라야 하고, 이 선택이 근사의 성격을 정한다.

**정의.** 근사분포 $q$를 고를 때

- **forward KL**: $\min_q D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_p[\log p - \log q]$. 기댓값을 참 분포 $p$로 잰다. MLE가 이 방향이다.
- **reverse KL**: $\min_q D_{\mathrm{KL}}(q \,\|\, p) = \mathbb{E}_q[\log q - \log p]$. 기댓값을 근사분포 $q$로 잰다. 변분추론이 이 방향이다(14.3장의 ELBO).

**직관.** forward KL에서는 $p(x) > 0$인데 $q(x) \approx 0$인 곳에서 $\log(p/q)$가 폭발한다. 그래서 $q$는 $p$가 질량을 둔 **모든 곳을 덮으려** 한다(mass-covering, 평균 맞추기). reverse KL에서는 $q(x) > 0$인데 $p(x) \approx 0$인 곳이 폭발한다. 그래서 $q$는 $p$가 없는 곳에 질량을 두지 않으려 하고, $p$의 한 봉우리 안에 **숨는다**(mode-seeking, 0 강제). 봉우리 두 개인 $p$에 가우시안 하나를 두 방향으로 맞춰 확인한다. 평균과 표준편차의 격자에서 각 KL을 수치적분으로 계산하고 최솟점을 찾는다.

```python {exec="browser" session="info"}
x = np.linspace(-8, 8, 4001)                       # 적분 격자, (G,)
dx = x[1] - x[0]
def normal_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

p = 0.5 * normal_pdf(x, -2.0, 0.6) + 0.5 * normal_pdf(x, 2.0, 0.6)   # 참 분포, (G,)
mus, sigmas = np.linspace(-3, 3, 121), np.linspace(0.2, 3.0, 57)
best = {"forward KL(p||q)": (np.inf,), "reverse KL(q||p)": (np.inf,)}
for mu in mus:
    for s in sigmas:
        q = normal_pdf(x, mu, s)
        fwd = np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300))) * dx
        rev = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
        for key, val in (("forward KL(p||q)", fwd), ("reverse KL(q||p)", rev)):
            if val < best[key][0]:
                best[key] = (val, mu, s)
for key, (val, mu, s) in best.items():
    print(f"{key}  최소 {val:.3f} nat  →  q = N({mu:+.2f}, {s:.2f}²)")
print(f"p의 평균 {np.sum(x * p) * dx:+.2f}, 표준편차 {np.sqrt(np.sum(x**2 * p) * dx):.2f}")
```

```output
forward KL(p||q)  최소 0.555 nat  →  q = N(+0.00, 2.10²)
reverse KL(q||p)  최소 0.692 nat  →  q = N(-2.00, 0.60²)
p의 평균 +0.00, 표준편차 2.09
```

forward KL의 최적 가우시안은 평균 0, 표준편차 2.1로, $p$의 평균·표준편차와 (격자 간격 안에서) 같다. 가우시안처럼 지수족인 $q$에서 forward KL 최소화는 **모멘트 맞추기**가 된다(12.2장에서 본 지수족 MLE의 성질: 충분통계량의 기댓값을 맞춘다). 그 결과 $q$는 두 봉우리를 모두 덮지만, 실제로 거의 관측되지 않는 0 근처에 가장 큰 밀도를 둔다. reverse KL의 최적 가우시안은 왼쪽 봉우리 하나에 정확히 맞는다. 오른쪽 봉우리 $\mathcal{N}(+2, 0.6^2)$도 같은 값을 주는 최솟점이다(대칭). 이때의 KL 0.692는 거의 $\log 2 \approx 0.693$으로, 질량의 절반을 무시한 대가다.

::quiz{id="reverse-kl-mode"}

**연구에서의 의미.** 두 성격은 실제 방법의 알려진 성질과 이어진다.

- MLE로 학습한 생성모델(forward KL)은 데이터의 모든 모드를 덮으려 하므로 샘플이 다양하지만, 모델 용량이 부족하면 모드 사이의 "있을 법하지 않은" 영역에도 확률을 준다. 19장의 생성모델 평가에서 다양성과 품질의 trade-off로 다시 나온다.
- 변분추론(reverse KL)은 사후분포의 한 모드에 맞추는 경향이 있고, 사후분포의 폭을 **과소평가**하기 쉽다. 근사 사후분포로 만든 신뢰구간이 실제보다 좁을 수 있다는 뜻이다(13.6장).

:::note[분야 예시]
분자 구조의 형태(conformation)는 에너지 지형의 여러 골짜기에 흩어져 있다. 분자 동역학 데이터로 볼츠만 분포를 근사하는 생성모델을 reverse KL(에너지 함수 $-\log p$만 알고 데이터 없이 학습하는 경우)로 학습하면 한두 개의 골짜기에만 샘플이 몰리는 mode collapse가 생기기 쉽다. forward KL(시뮬레이션 데이터로 MLE)은 데이터가 덮은 골짜기를 모두 덮지만, 데이터가 방문하지 못한 골짜기는 여전히 놓친다. 어느 방향이 무엇을 놓치는지 알면 결과를 해석할 때 무엇을 추가로 확인할지 정할 수 있다.
:::

## 다른 divergence

**왜 필요한가.** KL은 지지집합이 겹치지 않으면 무한대가 되고, 무한대는 "얼마나 다른가"에 대해 아무 정보도 주지 않는다. 고차원 데이터가 저차원 다양체 위에 놓여 있으면(7.8장) 모델 분포와 데이터 분포의 지지집합이 거의 겹치지 않는 일이 흔하다. 그래서 다른 척도가 필요하다.

**Jensen–Shannon divergence.** 두 분포의 중간 $m = \frac12(p + q)$를 기준으로 KL을 잰다(Lin, 1991).

$$
\mathrm{JS}(p, q) = \tfrac12 D_{\mathrm{KL}}(p \,\|\, m) + \tfrac12 D_{\mathrm{KL}}(q \,\|\, m).
$$

$m$은 $p$나 $q$가 양수인 곳에서 항상 양수이므로 JS는 늘 유한하고, 대칭이며, $0 \le \mathrm{JS} \le \log 2$다. 그 제곱근은 거리(metric)의 조건을 만족한다. 원래의 GAN 목적함수는 판별자가 최적일 때 JS를 최소화하는 것으로 해석된다(19.4장).

**f-divergence.** KL과 JS는 더 큰 가족의 일원이다. 볼록 함수 $f$로 $f(1) = 0$을 만족하면

$$
D_f(p \,\|\, q) = \mathbb{E}_{x \sim q}\Big[f\Big(\frac{p(x)}{q(x)}\Big)\Big]
$$

로 정의한다. $f(t) = t\log t$이면 $D_{\mathrm{KL}}(p \,\|\, q)$, $f(t) = -\log t$이면 $D_{\mathrm{KL}}(q \,\|\, p)$, $f(t) = \frac12\lvert t - 1 \rvert$이면 total variation 거리다. 모두 밀도의 **비** $p/q$만 보므로, 지지집합이 겹치지 않으면 두 분포가 얼마나 떨어져 있는지 구별하지 못한다.

**Wasserstein 거리.** 비 대신 **위치**를 본다. $p$의 질량을 $q$ 모양으로 옮기는 데 드는 최소 비용(옮긴 질량 × 이동 거리)이 1-Wasserstein 거리 $W_1$이고, earth mover's distance라고도 한다. 질량을 얼마나 멀리 옮겨야 하는지를 재므로 지지집합이 겹치지 않아도 의미 있는 값을 준다. 같은 모양의 분포를 점점 멀리 옮기며 세 척도를 비교한다.

```python {exec="browser" session="info"}
from scipy.stats import wasserstein_distance

support = np.arange(10)                            # 위치 0..9
p = np.zeros(10); p[[1, 2]] = 0.5                  # 질량이 1, 2에 있다
for shift in (2, 4, 6):
    q = np.roll(p, shift)                          # 같은 모양을 shift칸 옮긴다
    m = 0.5 * (p + q)
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    w1 = wasserstein_distance(support, support, p, q)
    print(f"이동 {shift}칸  KL(p||q) = {kl(p, q)}   JS = {js:.4f} (log 2 = {np.log(2):.4f})   W1 = {w1:.1f}")
```

```output
이동 2칸  KL(p||q) = inf   JS = 0.6931 (log 2 = 0.6931)   W1 = 2.0
이동 4칸  KL(p||q) = inf   JS = 0.6931 (log 2 = 0.6931)   W1 = 4.0
이동 6칸  KL(p||q) = inf   JS = 0.6931 (log 2 = 0.6931)   W1 = 6.0
```

KL은 언제나 무한대, JS는 언제나 최댓값 $\log 2$에 붙어 있다. 두 척도 모두 "2칸 떨어진 분포"와 "6칸 떨어진 분포"를 구별하지 못한다. 이것을 모델의 손실로 쓰면 모델을 데이터 쪽으로 옮겨도 손실이 줄지 않으므로, 어느 쪽으로 움직여야 할지 알려 주는 gradient가 없다. $W_1$은 이동 거리에 비례해 커진다. Wasserstein GAN은 이 성질을 근거로 JS 대신 $W_1$을 목적함수로 쓰자고 제안했다(Arjovsky 외, 2017).

::quiz{id="disjoint-support"}

척도를 고를 때 물을 것은 세 가지다. 지지집합이 겹치는가(겹치지 않으면 f-divergence는 정보가 없다). 표본만으로 추정할 수 있는가(KL은 밀도 비가 필요하고, $W_1$은 1차원에서는 정렬로 계산되지만 고차원에서는 추정이 어렵다). 그리고 **무엇이 다를 때 크게 벌하고 싶은가**(KL은 확률이 0에 가까운 곳의 차이를, $W_1$은 위치의 차이를 크게 본다).

| 척도 | 대칭 | 범위 | 지지집합이 겹치지 않을 때 | 대표 용도 |
|---|---|---|---|---|
| forward KL | 아니오 | $[0, \infty]$ | $\infty$ | MLE, cross-entropy 손실 |
| reverse KL | 아니오 | $[0, \infty]$ | $\infty$ | 변분추론 |
| JS | 예 | $[0, \log 2]$ | $\log 2$ (상수) | 원래의 GAN |
| $W_1$ | 예 | $[0, \infty)$ | 이동 거리에 비례 | WGAN, 분포 비교 지표 |

## 실습

쌍봉 분포 $p$에 가우시안 $q$를 forward KL과 reverse KL로 맞춘 결과를 그림으로 비교한다. 본문의 격자 탐색 코드(cell 4)를 시작점으로 삼아 다음을 수행한다.

1. $p$, forward KL 최적 $q$, reverse KL 최적 $q$ 세 밀도를 한 그림에 그린다(`matplotlib`, 결과는 `labs/information-and-learning-theory/entropy-and-kl/outputs/`에 저장한다).
2. 두 봉우리 사이 거리를 $\pm 0.5$에서 $\pm 3$까지 바꾸며 두 최적 $q$의 표준편차를 기록한다. 봉우리가 가까우면 두 방향의 답이 비슷해지고, 멀어질수록 갈라지는 것을 확인한다.
3. reverse KL의 격자 탐색을 $\mu = +2$ 근처에서 시작하는 경사하강으로 바꾸면 오른쪽 봉우리로, $\mu = -2$ 근처에서 시작하면 왼쪽 봉우리로 가는 것을 확인한다. reverse KL 최소화는 비볼록 문제이고 초기값에 따라 다른 모드에 붙는다.

이어서 두 연습문제를 푼다. 첫째는 0과 무한대의 규약까지 포함해 이산 KL과 cross-entropy를 구현하고, 둘째는 두 가우시안 사이 KL의 닫힌 형태를 구현해 비대칭성을 확인한다.

::exercise{id="discrete-kl"}

::exercise{id="gaussian-kl"}

## 실패 사례

- **KL을 대칭인 거리로 다루는 것.** "모델 A와 B의 분포 차이는 KL 0.3"처럼 방향을 밝히지 않고 보고하거나, $D_{\mathrm{KL}}(p \,\|\, q)$와 $D_{\mathrm{KL}}(q \,\|\, p)$를 바꿔 써도 된다고 가정한다. 두 값은 다르고(본문 예에서 0.184와 0.192), 최소화한 결과는 질적으로 다르다(평균 맞추기와 모드 찾기). 논문과 코드에서 어느 방향인지 항상 확인한다.
- **cross-entropy 손실이 0이 아니라고 학습이 덜 됐다고 판단하는 것.** cross-entropy의 최솟값은 데이터의 엔트로피 $H(p)$다. 레이블이 본질적으로 불확실하면(같은 입력에 다른 레이블이 붙는 측정) 손실은 0으로 갈 수 없고, 0에 가까워지면 오히려 잡음을 외운 것이다.
- **미분 엔트로피나 연속 NLL의 절댓값을 다른 단위·다른 전처리 사이에서 비교하는 것.** 단위를 바꾸면 값이 상수만큼 이동한다. 같은 단위·같은 정규화에서만 비교한다.
- **확률 0을 출력하는 모델로 KL이나 NLL을 계산하는 것.** $q = 0$인 곳에 관측이 하나라도 있으면 값이 무한대(또는 구현에 따라 `nan`)가 된다. softmax 출력처럼 정의상 양수인 분포를 쓰거나, 로그 공간에서 계산한다(`log_softmax`).

## 통과 기준

- cross-entropy 최소화가 KL 최소화, 그리고 MLE와 같은 이유를 $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$로 설명하고, 손실의 최솟값이 0이 아닐 수 있는 이유를 말한다.
- 쌍봉 분포에 대한 forward·reverse KL 근사를 그려, 각각이 mass-covering·mode-seeking인 이유를 기댓값을 재는 분포의 차이로 설명한다.
- 지지집합이 겹치지 않는 두 분포에서 KL·JS·$W_1$이 어떻게 행동하는지 예를 들어 설명한다.
- 연습문제 두 개(`discrete-kl`, `gaussian-kl`)의 테스트를 통과한다.

## 참고 문헌

- Shannon (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423; 27(4), 623–656. — 엔트로피의 정의와 원천 부호화 정리
- Kullback, Leibler (1951). On information and sufficiency. *The Annals of Mathematical Statistics*, 22(1), 79–86. — KL divergence(상대 엔트로피)의 도입
- Lin (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151. — 절대연속 조건이 필요 없는 divergence, Jensen–Shannon divergence
- Arjovsky, Chintala, Bottou (2017). Wasserstein generative adversarial networks. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 214–223. — 지지집합이 겹치지 않을 때 JS와 Wasserstein 거리의 차이
- Cover, Thomas (2006). *Elements of Information Theory* (2nd ed.). Wiley. — 엔트로피·KL·미분 엔트로피의 표준 교과서
