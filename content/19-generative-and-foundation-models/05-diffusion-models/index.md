---
id: generative-and-foundation-models/diffusion-models
title: "확산 모델과 score 기반 모델"
title_en: "Diffusion and Score-Based Models"
tier: core
status: draft
prerequisites: [generative-and-foundation-models/generative-modeling, probability/stochastic-processes]
---

# 확산 모델과 score 기반 모델

데이터에 잡음을 조금씩 더하면 어떤 분포든 결국 표준정규분포가 된다. 이 과정은 학습할 것이 없는 고정된 확률과정이다. 확산 모델(diffusion model)은 그 과정을 **거꾸로 되돌리는 법**을 배운다. 잡음에서 출발해 잡음을 조금씩 걷어 내면 데이터 분포의 표본이 된다. 한 번에 복잡한 변환을 배우는 GAN·flow와 달리, 각 step은 "약간 흐려진 데이터에서 잡음을 조금 덜어 내는" 단순한 회귀 문제다. 이 단순함 덕분에 학습이 안정적이고, 이미지·분자 구조·기후장 같은 고차원 과학 데이터의 생성에 널리 쓰인다.

이 장은 forward 과정과 그 닫힌 형태, 잡음 제거 목적함수와 그 최적해가 **score** $\nabla_\mathbf{x}\log p_t(\mathbf{x})$라는 사실, 샘플링 방법과 step 수의 관계, 조건부 생성을 차례로 다룬다. 학습된 신경망 대신 **정확한 score**를 계산할 수 있는 1차원 혼합분포를 쓰면, 샘플러 자체의 오차와 신경망의 오차를 분리해서 볼 수 있다. 11.7장의 SDE와 Ornstein–Uhlenbeck 과정이 이 장의 수학적 배경이다.

## 학습 목표

- forward·reverse 과정을 이해한다
- denoising 목적함수를 유도한다
- 샘플링 방법을 비교한다

## forward 과정

**정의.** 시각 $t = 1, \dots, T$마다 잡음 스케줄 $\beta_t \in (0, 1)$을 정하고, 데이터 $x_0 \sim p_{\text{data}}$에서 출발해

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t \mathbf{I}\big)
$$

로 잡음을 더한다. 신호를 $\sqrt{1 - \beta_t}$배로 줄이고 분산 $\beta_t$의 잡음을 더하므로, 분산이 1인 데이터는 계속 분산 1을 유지한다(variance preserving). 이 장의 코드는 $T = 1000$, $\beta_t$를 $10^{-4}$에서 $0.02$까지 선형으로 늘리는 스케줄을 쓴다.

**닫힌 형태의 주변분포.** 가우시안 잡음을 거듭 더한 결과는 다시 가우시안이므로, 중간 step을 거치지 않고 $x_0$에서 바로 $x_t$를 뽑을 수 있다. $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$라 하면

$$
q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t)\mathbf{I}\big), \qquad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon},\ \ \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

이다. $\bar\alpha_t$는 $t$와 함께 1에서 0으로 줄어든다. $\bar\alpha_T \approx 0$이면 $x_T$는 데이터와 거의 무관한 $\mathcal{N}(\mathbf{0}, \mathbf{I})$이다. 이 성질 덕분에 학습할 때 임의의 $t$를 골라 $x_t$를 한 번에 만들 수 있다. 데이터 분포 전체에 대한 주변분포 $p_t(x) = \int q(x_t \mid x_0)\,p_{\text{data}}(x_0)\,dx_0$는 데이터 분포를 $\sqrt{\bar\alpha_t}$배로 줄인 뒤 분산 $1 - \bar\alpha_t$의 가우시안과 convolution한 것이다. 데이터가 가우시안 혼합이면 $p_t$도 가우시안 혼합이고, 성분 평균은 $\sqrt{\bar\alpha_t}\mu_k$, 분산은 $\bar\alpha_t\sigma_k^2 + 1 - \bar\alpha_t$다.

**코드로 확인.** 혼합분포 $0.35\,\mathcal{N}(-1.5, 0.25^2) + 0.65\,\mathcal{N}(1.2, 0.35^2)$에서 표본 2만 개를 뽑아 1000 step 동안 한 step씩 잡음을 더하고, 몇몇 시각에서 표본의 평균·분산을 닫힌 형태의 $p_t$와 비교한다.

```python {exec="browser" session="diff"}
import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)                # (T,) 선형 잡음 스케줄, t = 1..T
abar = np.cumprod(1 - betas)                       # (T,) abar_t = prod_{s<=t} (1 - beta_s)
PI, MU, SD = np.array([0.35, 0.65]), np.array([-1.5, 1.2]), np.array([0.25, 0.35])

def sample_x0(n, rng):                             # 데이터: 1차원 가우시안 혼합
    k = (rng.random(n) < PI[1]).astype(int)
    return MU[k] + SD[k] * rng.normal(size=n)

def comps(t):                                      # p_t도 혼합: 성분 평균·분산 (t = 0이면 데이터)
    a = 1.0 if t == 0 else abar[t - 1]
    return np.sqrt(a) * MU, a * SD**2 + (1 - a)

rng = np.random.default_rng(0)
x = sample_x0(20000, rng)                          # (20000,)
for t in range(1, T + 1):                          # q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t)
    x = np.sqrt(1 - betas[t - 1]) * x + np.sqrt(betas[t - 1]) * rng.normal(size=x.size)
    if t in (100, 300, 600, 1000):
        m, v = comps(t)
        mean = PI @ m
        var = PI @ (v + m**2) - mean**2
        print(f"t={t:4d}  abar={abar[t - 1]:.4f}  시뮬레이션 평균 {x.mean():6.3f} 분산 {x.var():.3f}"
              f"  |  닫힌 형태 평균 {mean:6.3f} 분산 {var:.3f}")
```

```output
t= 100  abar=0.8970  시뮬레이션 평균  0.230 분산 1.694  |  닫힌 형태 평균  0.242 분산 1.682
t= 300  abar=0.3964  시뮬레이션 평균  0.150 분산 1.311  |  닫힌 형태 평균  0.161 분산 1.301
t= 600  abar=0.0259  시뮬레이션 평균  0.035 분산 1.019  |  닫힌 형태 평균  0.041 분산 1.020
t=1000  abar=0.0000  시뮬레이션 평균  0.005 분산 0.990  |  닫힌 형태 평균  0.002 분산 1.000
```

한 step씩 시뮬레이션한 결과와 닫힌 형태가 표본 오차 안에서 일치한다. $t = 600$에서 이미 $\bar\alpha_t = 0.026$으로 신호가 거의 사라지고, $t = 1000$에서는 평균 0, 분산 1인 표준정규분포에 도달한다. 데이터의 분산(약 1.7)이 1이 아니어도 끝은 같다. 11.7장에서 본 것처럼 이 과정은 OU 과정의 이산화이고, 어떤 초기 분포에서 출발해도 정상분포 $\mathcal{N}(0, 1)$로 수렴한다.

::quiz{id="forward-closed-form"}

## reverse 과정과 학습

**reverse 과정.** 생성은 $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에서 출발해 $p_\theta(x_{t-1} \mid x_t)$를 따라 $x_0$까지 내려가는 것이다. $\beta_t$가 작으면 참 역방향 조건부분포 $q(x_{t-1} \mid x_t)$도 거의 가우시안이므로, $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\boldsymbol{\mu}_\theta(x_t, t), \sigma_t^2\mathbf{I}\big)$로 모델링한다. 문제는 평균 $\boldsymbol{\mu}_\theta$를 어떻게 학습하느냐다.

**$x_0$를 알면 역방향은 닫힌 꼴이다.** 베이즈 정리로 $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$이고, 두 가우시안의 곱이므로

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\tilde{\boldsymbol{\mu}}_t(x_t, x_0),\ \tilde\beta_t\mathbf{I}\big), \quad
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}x_t, \quad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t
$$

이다(연습문제에서 유도한다). 생성할 때는 $x_0$를 모르므로, 신경망이 $x_t$에서 $x_0$를 **예측**하고 그 예측을 $\tilde{\boldsymbol{\mu}}_t$에 넣는다.

**ELBO에서 denoising 손실로.** 확산 모델은 $x_1, \dots, x_T$를 잠재변수로 갖는 계층적 VAE(19.3장)로 볼 수 있다. encoder가 학습 없이 고정된 forward 과정 $q$이고 decoder가 reverse 과정 $p_\theta$다. 그 ELBO를 풀어 쓰면 step마다 $\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)$의 합이 되고, 두 가우시안의 분산이 같으면 이 KL은 평균의 제곱 차이에 비례한다. $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\varepsilon}$로 $x_0$를 $\boldsymbol{\varepsilon}$으로 바꿔 쓰면, 결국 신경망 $\boldsymbol{\varepsilon}_\theta(x_t, t)$가 섞인 잡음 $\boldsymbol{\varepsilon}$을 맞히는 회귀가 된다. Ho, Jain, Abbeel(2020)은 step별 가중치를 버린 단순한 손실

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{1, T\},\ x_0,\ \boldsymbol{\varepsilon}}\Big[\big\lVert \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\varepsilon},\ t\big)\big\rVert^2\Big]
$$

이 표본 품질에 더 좋다고 보고했고, 이것이 표준적인 학습 목적함수가 되었다. 가중치를 버렸으므로 이 손실은 엄밀히는 ELBO가 아니라 **가중 ELBO**다.

**최적해는 조건부 기댓값이다.** 제곱오차를 최소화하는 함수는 조건부 평균이므로(12.3장) 최적의 잡음 예측은

$$
\boldsymbol{\varepsilon}^\star(x_t, t) = \mathbb{E}[\boldsymbol{\varepsilon} \mid x_t]
$$

이다. 같은 $x_t$가 서로 다른 $(x_0, \boldsymbol{\varepsilon})$ 쌍에서 나올 수 있으므로, 최적해에서도 손실은 0이 되지 않는다. 다음 절에서 보듯 이 조건부 기댓값은 score로 정확히 쓸 수 있다. 아래 코드는 $t = 300$에서 정확한 $\boldsymbol{\varepsilon}^\star$와 그것을 $c$배 한 예측의 손실을 비교한다.

```python {exec="browser" session="diff"}
def score(x, t):                                   # d/dx log p_t(x), 혼합의 닫힌 형태
    m, v = comps(t)
    logw = np.log(PI) - 0.5 * (x[:, None] - m) ** 2 / v - 0.5 * np.log(v)    # (n, 2)
    r = np.exp(logw - logw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)                   # 성분 책임도
    return (r * (-(x[:, None] - m) / v)).sum(1)

def eps_star(x, t):                                # E[eps | x_t] = -sqrt(1 - abar_t) * score
    return -np.sqrt(1 - abar[t - 1]) * score(x, t)

t = 300
x0 = sample_x0(200000, rng)
eps = rng.normal(size=x0.size)
xt = np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps    # 닫힌 형태로 한 번에
for c in [0.0, 0.8, 1.0, 1.2]:
    print(f"eps_hat = {c} * eps*:  denoising 손실 E(eps - eps_hat)^2 = {np.mean((eps - c * eps_star(xt, t)) ** 2):.4f}")
x0_hat = (xt - np.sqrt(1 - abar[t - 1]) * eps_star(xt, t)) / np.sqrt(abar[t - 1])   # Tweedie 공식
print(f"x0 예측의 제곱오차 {np.mean((x0 - x0_hat) ** 2):.4f}  vs  x0 = x_t / sqrt(abar) {np.mean((x0 - xt / np.sqrt(abar[t - 1])) ** 2):.4f}")
```

```output
eps_hat = 0.0 * eps*:  denoising 손실 E(eps - eps_hat)^2 = 1.0026
eps_hat = 0.8 * eps*:  denoising 손실 E(eps - eps_hat)^2 = 0.4923
eps_hat = 1.0 * eps*:  denoising 손실 E(eps - eps_hat)^2 = 0.4713
eps_hat = 1.2 * eps*:  denoising 손실 E(eps - eps_hat)^2 = 0.4927
x0 예측의 제곱오차 0.7175  vs  x0 = x_t / sqrt(abar) 1.5266
```

$c = 1$, 즉 정확한 $\boldsymbol{\varepsilon}^\star$에서 손실이 가장 작고(0.471), 그래도 0보다 훨씬 크다. 예측하지 않은 경우($c = 0$)의 손실은 잡음의 분산 1이다. 마지막 줄은 같은 예측을 $x_0$로 환산한 것이다. $\hat{x}_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}^\star)/\sqrt{\bar\alpha_t}$의 제곱오차는 잡음을 무시하고 크기만 되돌린 $x_t/\sqrt{\bar\alpha_t}$의 절반 이하다.

::quiz{id="optimal-denoiser"}

**ε · $x_0$ · v 예측.** $x_t$, $x_0$, $\boldsymbol{\varepsilon}$ 사이에는 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\varepsilon}$라는 선형 관계가 있으므로, 셋 중 무엇을 예측하든 서로 바꿔 쓸 수 있다. 신경망이 무엇을 출력하게 할지(parameterization)는 학습의 수치적 성질을 바꾼다.

- **ε 예측**: 출력의 크기가 모든 $t$에서 비슷하다(표준정규). 가장 널리 쓰인다. 다만 $\bar\alpha_t \to 0$인 큰 $t$에서 $\hat{x}_0$로 환산하면 $1/\sqrt{\bar\alpha_t}$가 오차를 크게 키운다.
- **$x_0$ 예측**: 출력이 데이터 공간에 있어 값의 범위 제약을 걸기 쉽다. 작은 $t$에서는 거의 항등함수를 배우는 셈이라 비효율적이다.
- **v 예측**: $\mathbf{v} = \sqrt{\bar\alpha_t}\,\boldsymbol{\varepsilon} - \sqrt{1 - \bar\alpha_t}\,x_0$를 예측한다. $t$의 양 끝에서 모두 수치적으로 안정적인 절충이다.

## score 관점

**score와 Tweedie 공식.** 밀도 $p$의 score는 로그밀도의 기울기 $\nabla_\mathbf{x}\log p(\mathbf{x})$다. 정규화 상수가 미분으로 사라지므로, 에너지 기반 모델처럼 정규화 상수를 모르는 분포에서도 정의된다. 가우시안 잡음으로 흐려진 분포 $p_t$에서는 다음 관계(Tweedie 공식)가 성립한다.

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1 - \bar\alpha_t)\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}, \qquad
\boldsymbol{\varepsilon}^\star(x_t, t) = \mathbb{E}[\boldsymbol{\varepsilon} \mid x_t] = -\sqrt{1 - \bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t).
$$

즉 **잡음 예측을 학습하는 것은 각 시각의 score를 학습하는 것과 같다.** 앞 셀의 `eps_star`가 바로 이 식으로 계산한 것이다. 혼합분포의 score는 성분별 score $-(x - m_k)/v_k$를 성분의 사후확률(책임도)로 가중평균한 것이다.

**score matching.** score를 직접 회귀하려면 참 score를 알아야 하는데, 데이터만으로는 모른다. Hyvärinen(2005)은 $\mathbb{E}\lVert \mathbf{s}_\theta(\mathbf{x}) - \nabla\log p(\mathbf{x})\rVert^2$를 부분적분해 참 score 없이 계산할 수 있는 목적함수로 바꿀 수 있음을 보였다(score matching). 다만 그 식에는 $\mathbf{s}_\theta$의 Jacobian 대각합이 들어가 고차원에서 비싸다. Vincent(2011)는 데이터에 잡음을 더한 분포의 score는 **잡음을 되돌리는 방향** $-(x_t - \sqrt{\bar\alpha_t}x_0)/(1 - \bar\alpha_t)$를 회귀해서 배울 수 있음을 보였다(denoising score matching). 이 회귀 목표를 $\boldsymbol{\varepsilon}$으로 쓰면 $-\boldsymbol{\varepsilon}/\sqrt{1 - \bar\alpha_t}$이므로, 앞 절의 $L_{\text{simple}}$이 곧 denoising score matching이다. 잡음 크기마다 score를 배워 Langevin 동역학으로 표본을 만드는 score 기반 모델과, 변분 하한에서 출발한 DDPM이 같은 대상을 학습한다는 것이 Ho et al.(2020)의 관찰이다.

**SDE·ODE 표현.** step 간격을 0으로 보내면 forward 과정은 11.7장의 variance preserving SDE

$$
d\mathbf{x} = -\tfrac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{W}
$$

가 된다. 이 SDE를 시간 역방향으로 푸는 SDE와, 같은 주변분포 $p_t$를 갖는 결정론적 ODE(probability flow ODE)가 존재하고, 둘 다 drift에 score가 들어간다(Song et al., 2021).

$$
\text{reverse SDE: } d\mathbf{x} = \big[-\tfrac{1}{2}\beta(t)\mathbf{x} - \beta(t)\nabla_\mathbf{x}\log p_t(\mathbf{x})\big]dt + \sqrt{\beta(t)}\,d\bar{\mathbf{W}}, \qquad
\text{ODE: } \frac{d\mathbf{x}}{dt} = -\tfrac{1}{2}\beta(t)\big[\mathbf{x} + \nabla_\mathbf{x}\log p_t(\mathbf{x})\big].
$$

$dt$는 음수(시간을 거꾸로)이고 $\bar{\mathbf{W}}$는 역방향 Brown 운동이다. ODE는 잡음 $\mathbf{x}_T$와 데이터 $\mathbf{x}_0$ 사이의 **가역 사상**이므로 19.4장의 연속 flow와 같고, 변수변환 공식으로 정확한 likelihood를 계산할 수 있다(19.1장 표의 "ODE로 계산"). 이 장의 이산 스케줄은 연속 스케줄 $\beta(t) = 0.1 + 19.9\,t$($t \in [0, 1]$)를 1000 step으로 나눈 것과 같다.

아래 위젯은 같은 형태의 1차원 혼합분포에 이 연속 스케줄을 적용한다. forward 과정에서 입자들이 표준정규분포로 퍼지는 모습과, 정확한 score를 쓴 reverse SDE·ODE로 되돌아오는 모습을 비교하고, step 수를 줄였을 때 무엇이 틀어지는지 확인한다.

::widget{name="diffusion-process"}

::quiz{id="eps-score-relation"}

## 샘플링

**ancestral sampling (DDPM).** 학습한 $\boldsymbol{\varepsilon}_\theta$로 $\hat{x}_0$를 만들고, $q(x_{t-1} \mid x_t, \hat{x}_0)$에서 뽑는 과정을 $t = T$부터 1까지 반복한다. step마다 새 잡음이 들어가므로 확률적이고, reverse SDE의 이산화에 해당한다.

**DDIM.** Song, Meng, Ermon(2021)은 같은 학습 결과($\boldsymbol{\varepsilon}_\theta$)를 그대로 쓰면서 주변분포 $q(x_t \mid x_0)$는 같고 step 간 잡음은 다른 과정들을 정의했다. 한 step의 갱신은

$$
x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\,\boldsymbol{\varepsilon}_\theta(x_t, t) + \sigma\,\mathbf{z}
$$

이고($t' < t$), $\sigma^2 = \eta^2\,\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}\big(1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}\big)$이다. $\eta = 1$이면 $\sigma^2$이 DDPM의 사후분산 $\tilde\beta$와 같아 DDPM이 되고, $\eta = 0$이면 잡음 없는 결정론적 샘플러가 된다. 결정론적 DDIM은 probability flow ODE의 이산화로 볼 수 있다. 저자들은 DDIM이 DDPM보다 벽시계 시간 기준 10–50배 빠르게 고품질 표본을 만든다고 보고했다. step 수를 줄이는 것은 $T$개 시각 중 일부 $t$만 골라 그 사이를 한 번에 건너뛰는 것이다.

**ODE solver.** 결정론적 샘플러는 결국 ODE를 푸는 것이므로, 9장의 수치해법이 그대로 적용된다. Euler법(DDIM에 해당)보다 차수가 높은 solver를 쓰면 같은 오차를 더 적은 함수 평가(신경망 호출)로 얻을 수 있다. 확산 모델의 샘플링 비용은 신경망 호출 횟수가 정하므로, step 수가 곧 비용이다.

**코드로 확인.** 정확한 $\boldsymbol{\varepsilon}^\star$를 쓰면 남는 오차는 이산화(step 수)와 유한한 표본 수뿐이다. step 수를 바꿔 가며 표본 1만 개를 만들고, 참 분포와의 Wasserstein-1 거리(W1, 1차원에서는 두 분위수 함수 사이의 넓이)와 왼쪽 모드의 비율을 잰다.

```python {exec="browser" session="diff"}
from scipy.stats import wasserstein_distance

ref = sample_x0(200000, np.random.default_rng(1))  # 비교용 참 분포 표본

def sampler(n_steps, eta, n=10000, seed=2, ab_sampler=None):
    """eta = 1: DDPM(확률적), eta = 0: DDIM(결정론적). T개 시각 중 n_steps개만 쓴다."""
    ab_s = abar if ab_sampler is None else ab_sampler   # 샘플러가 믿는 abar
    g = np.random.default_rng(seed)
    ts = np.linspace(T, 1, n_steps).round().astype(int)
    x = g.normal(size=n)                           # x_T ~ N(0, 1)
    for i, t in enumerate(ts):
        a_t = ab_s[t - 1]
        a_prev = ab_s[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0
        e = eps_star(x, t)                         # '학습된 신경망' 자리에 정확한 eps*
        x0_hat = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
        sig = eta * np.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev - sig**2) * e + sig * g.normal(size=n)
    return x

def report(name, s):
    print(f"{name:22s} W1 {wasserstein_distance(s, ref):.3f}  왼쪽 모드 비율 {np.mean(s < 0):.3f}  (참 0.350)")

for n_steps in [1000, 100, 20, 5]:
    report(f"DDPM {n_steps:4d} step", sampler(n_steps, eta=1.0))
```

```output
DDPM 1000 step         W1 0.004  왼쪽 모드 비율 0.350  (참 0.350)
DDPM  100 step         W1 0.032  왼쪽 모드 비율 0.341  (참 0.350)
DDPM   20 step         W1 0.085  왼쪽 모드 비율 0.335  (참 0.350)
DDPM    5 step         W1 0.400  왼쪽 모드 비율 0.319  (참 0.350)
```

1000 step에서 W1은 0.004로 표본 1만 개의 통계적 오차 수준이다. step을 줄이면 오차가 커지고, 5 step에서는 W1이 0.40으로 분포가 크게 틀어진다. 같은 score를 썼으므로 이 차이는 전부 이산화 오차다. 결정론적 DDIM과 비교한다.

```python {exec="browser" session="diff"}
for n_steps in [100, 20, 5]:
    report(f"DDIM {n_steps:4d} step", sampler(n_steps, eta=0.0))
abar_b = np.cumprod(1 - np.linspace(1e-4, 0.01, T))     # 샘플러만 다른 스케줄(beta 끝값 절반)을 믿는다
report("DDPM 1000 step, 불일치", sampler(1000, eta=1.0, ab_sampler=abar_b))
report("DDIM   20 step, 불일치", sampler(20, eta=0.0, ab_sampler=abar_b))
```

```output
DDIM  100 step         W1 0.024  왼쪽 모드 비율 0.342  (참 0.350)
DDIM   20 step         W1 0.068  왼쪽 모드 비율 0.337  (참 0.350)
DDIM    5 step         W1 0.336  왼쪽 모드 비율 0.334  (참 0.350)
DDPM 1000 step, 불일치    W1 0.123  왼쪽 모드 비율 0.342  (참 0.350)
DDIM   20 step, 불일치    W1 0.268  왼쪽 모드 비율 0.342  (참 0.350)
```

같은 step 수에서 DDIM이 DDPM보다 W1이 작다(100 step 0.024 대 0.032, 20 step 0.068 대 0.085). 확률적 샘플러는 step마다 잡음을 새로 넣는 대신 앞 step의 오차를 일부 지우는 효과가 있어 step이 충분히 많으면 유리할 수 있지만, step이 적을 때는 결정론적 ODE 이산화가 대체로 효율적이다. 마지막 두 줄은 실패 사례에서 다룬다.

::quiz{id="steps-vs-quality"}

## 조건부 생성

**조건부 score.** 조건 $\mathbf{c}$가 있으면 $p_t(\mathbf{x} \mid \mathbf{c})$의 score가 필요하다. 베이즈 정리로

$$
\nabla_\mathbf{x}\log p_t(\mathbf{x} \mid \mathbf{c}) = \nabla_\mathbf{x}\log p_t(\mathbf{x}) + \nabla_\mathbf{x}\log p_t(\mathbf{c} \mid \mathbf{x})
$$

이다. 둘째 항은 잡음 섞인 $\mathbf{x}$를 받는 분류기의 기울기다. 이 항에 가중치를 키워 곱하면(classifier guidance) 조건을 더 강하게 반영한 표본이 나온다. 그러려면 잡음 수준마다 동작하는 분류기를 따로 학습해야 한다.

**classifier-free guidance.** Ho & Salimans(2022)는 분류기 없이 같은 효과를 내는 방법을 제안했다. 하나의 신경망을 학습하면서 일정 확률로 조건을 빈 값으로 바꿔, 조건부 예측 $\boldsymbol{\varepsilon}_\theta(\mathbf{x}, t, \mathbf{c})$와 무조건 예측 $\boldsymbol{\varepsilon}_\theta(\mathbf{x}, t, \varnothing)$를 모두 배운다. 샘플링에서는 둘을 외삽한다.

$$
\tilde{\boldsymbol{\varepsilon}} = (1 + w)\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}, t, \mathbf{c}) - w\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}, t, \varnothing).
$$

score로 바꾸면 $\nabla\log p_t(\mathbf{x}) + (1 + w)\nabla\log p_t(\mathbf{c} \mid \mathbf{x})$에 해당하므로, 암묵적인 분류기의 기울기를 $(1 + w)$배로 키운 것이다. $w = 0$이면 보통의 조건부 샘플링이다. 정확한 조건부·무조건 score로 확인한다. 조건은 오른쪽 성분이고, 참 조건부분포는 $\mathcal{N}(1.2, 0.35^2)$이다.

```python {exec="browser" session="diff"}
def eps_cond(x, t, k):                             # 조건 c = 성분 k일 때 p_t(x | c)는 가우시안
    m, v = comps(t)
    return -np.sqrt(1 - abar[t - 1]) * (-(x - m[k]) / v[k])

def guided_ddim(k, w, n_steps=50, n=10000, seed=3):
    g = np.random.default_rng(seed)
    ts = np.linspace(T, 1, n_steps).round().astype(int)
    x = g.normal(size=n)
    for i, t in enumerate(ts):
        e = (1 + w) * eps_cond(x, t, k) - w * eps_star(x, t)    # guidance 가중치 w
        a_t, a_prev = abar[t - 1], (abar[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0)
        x0_hat = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev) * e
    return x

print("조건 c = 오른쪽 성분, 참 조건부분포 N(1.2, 0.35^2)")
for w in [0.0, 1.0, 3.0]:
    s = guided_ddim(1, w)
    print(f"w = {w}: 평균 {s.mean():.3f}  표준편차 {s.std():.3f}  x < 0.7인 비율 {np.mean(s < 0.7):.3f}  (참 조건부분포 0.076)")
```

```output
조건 c = 오른쪽 성분, 참 조건부분포 N(1.2, 0.35^2)
w = 0.0: 평균 1.198  표준편차 0.325  x < 0.7인 비율 0.064  (참 조건부분포 0.076)
w = 1.0: 평균 1.359  표준편차 0.243  x < 0.7인 비율 0.000  (참 조건부분포 0.076)
w = 3.0: 평균 1.543  표준편차 0.197  x < 0.7인 비율 0.000  (참 조건부분포 0.076)
```

$w = 0$에서는 참 조건부분포에 가까운 표본이 나온다(50 step DDIM이라 표준편차가 0.325로 약간 좁다). $w$를 키우면 평균이 다른 성분에서 **멀어지는** 쪽(1.36, 1.54)으로 밀리고 표준편차가 줄어든다. 참 조건부분포에서는 7.6%가 0.7보다 작아야 하는데, $w = 1$부터는 그런 표본이 하나도 없다. guidance는 "조건에 확실히 속하는" 표본을 얻는 대가로 다양성을 버리고, 결과 분포는 더 이상 $p(\mathbf{x} \mid \mathbf{c})$가 아니다. 19.2장의 샘플링 온도와 같은 성격의 손잡이다.

::quiz{id="guidance-weight"}

## 과학 데이터에의 적용

**시계열과 물리장.** 확산 모델은 데이터의 형태를 가리지 않는다. 신경망 $\boldsymbol{\varepsilon}_\theta$의 구조만 데이터에 맞추면 된다. 격자 위의 물리장(온도·강수·유속)은 이미지와 같은 convolution 구조로, 시계열은 시간 축 convolution이나 Transformer로, 분자·입자 시스템은 회전·평행이동에 대한 대칭성을 지키는 그래프 신경망(18.6장)으로 다룬다. 표본 하나하나가 독립적으로 뽑히므로, 같은 조건에서 여러 가능한 결과의 **앙상블**을 만드는 데 맞다.

**역문제.** 관측 $\mathbf{y} = \mathcal{A}(\mathbf{x}) + \text{잡음}$에서 원래 신호 $\mathbf{x}$를 복원하는 역문제에서는 사후분포 $p(\mathbf{x} \mid \mathbf{y}) \propto p(\mathbf{x})\,p(\mathbf{y} \mid \mathbf{x})$에서 표본을 뽑고 싶다. 조건부 score의 분해를 그대로 쓰면

$$
\nabla_\mathbf{x}\log p_t(\mathbf{x} \mid \mathbf{y}) = \underbrace{\nabla_\mathbf{x}\log p_t(\mathbf{x})}_{\text{학습한 사전분포}} + \nabla_\mathbf{x}\log p_t(\mathbf{y} \mid \mathbf{x})
$$

이다. 사전분포 쪽은 관측과 무관하게 데이터로 한 번 학습한 확산 모델이 주고, 관측 모형 $p(\mathbf{y} \mid \mathbf{x})$는 물리(측정 연산자 $\mathcal{A}$와 잡음 모형)가 준다. 같은 사전분포를 여러 종류의 관측에 다시 쓸 수 있다는 것이 장점이다. 다만 $p_t(\mathbf{y} \mid \mathbf{x}_t)$는 잡음 섞인 $\mathbf{x}_t$에 대한 우도라서 정확히 계산할 수 없고, 여러 근사법이 제안되어 있다. 근사가 들어가므로 이렇게 얻은 표본이 정확한 사후분포의 표본이라고 가정하면 안 된다.

:::note[분야 예시]
기상·기후 분야에서는 거친 해상도의 수치모델 출력을 조건으로 고해상도 강수장을 생성하거나, 결정론적 예보 하나를 조건으로 여러 앙상블 멤버를 생성하는 데 확산 모델이 쓰인다. 같은 조건에서 뽑은 표본들의 퍼짐이 예보 불확실성을 나타내도록 하는 것이 목적이므로, 평가도 표본 하나의 정확도보다 앙상블의 calibration(21장)과 극한값 통계가 중심이 된다.
:::

:::note[분야 예시]
의료·과학 영상의 재구성(예: 측정 횟수를 줄인 MRI, 희소한 각도의 단층 촬영)은 $\mathbf{y} = \mathcal{A}\mathbf{x} + \text{잡음}$ 꼴의 선형 역문제다. 영상 데이터로 학습한 확산 모델을 사전분포로 두고 위의 분해로 측정과 일관된 영상을 생성하면, 한 번의 측정에 대해 여러 재구성을 뽑아 영역별 불확실성을 볼 수 있다. 사전분포가 학습 데이터에 없던 구조(예: 드문 병변)를 지우거나 그럴듯한 가짜 구조를 만들 위험이 있어, 재구성 결과를 진단·측정에 쓰려면 그 위험을 따로 검증해야 한다.
:::

## 실습

지금까지는 정확한 score를 썼다. 이제 같은 데이터에서 작은 MLP로 $\boldsymbol{\varepsilon}_\theta(x_t, t)$를 학습하고(입력은 $x_t$와 시각의 사인·코사인 특징), 같은 DDPM 1000 step 샘플러로 표본을 만든다. 학습 손실은 앞에서 유도한 $L_{\text{simple}}$이다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행되고, 필요한 정의를 모두 셀 안에 다시 쓴다.

```python {exec="python"}
import numpy as np
import torch
from torch import nn
from scipy.stats import wasserstein_distance

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
abar = torch.cumprod(1 - betas, 0)                 # (T,)
def sample_x0(n):                                  # 0.35 N(-1.5, 0.25^2) + 0.65 N(1.2, 0.35^2)
    right = torch.rand(n) < 0.65
    return torch.where(right, 1.2 + 0.35 * torch.randn(n), -1.5 + 0.25 * torch.randn(n))
net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
feat = lambda x, t: torch.stack([x, torch.sin(torch.pi * t / T), torch.cos(torch.pi * t / T)], 1)  # (B, 3)
opt = torch.optim.Adam(net.parameters(), 2e-3)
for step in range(2000):
    x0, t, e = sample_x0(512), torch.randint(1, T + 1, (512,)), torch.randn(512)
    xt = abar[t - 1].sqrt() * x0 + (1 - abar[t - 1]).sqrt() * e   # 닫힌 형태의 forward
    loss = ((net(feat(xt, t)).squeeze(1) - e) ** 2).mean()        # denoising 손실
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():                              # DDPM ancestral sampling, 1000 step
    x = torch.randn(10000)
    for t in range(T, 0, -1):
        e = net(feat(x, torch.full_like(x, t))).squeeze(1)
        x = (x - betas[t - 1] / (1 - abar[t - 1]).sqrt() * e) / (1 - betas[t - 1]).sqrt()
        if t > 1:                                  # 사후분포 분산 beta_t (1 - abar_{t-1}) / (1 - abar_t)
            x = x + (betas[t - 1] * (1 - abar[t - 2]) / (1 - abar[t - 1])).sqrt() * torch.randn_like(x)
print(f"마지막 batch의 denoising 손실 {loss.item():.3f}")
print(f"학습된 eps, DDPM 1000 step: W1 {wasserstein_distance(x.numpy(), sample_x0(200000).numpy()):.3f}"
      f"  왼쪽 모드 비율 {(x < 0).float().mean():.3f}  (참 0.350)")
```

```output
마지막 batch의 denoising 손실 0.317
학습된 eps, DDPM 1000 step: W1 0.066  왼쪽 모드 비율 0.361  (참 0.350)
```

정확한 $\boldsymbol{\varepsilon}^\star$로 1000 step을 돌린 결과(셀 3, W1 0.004)와 비교하면 W1이 0.066으로 커졌고 왼쪽 모드 비율도 0.361로 조금 틀어졌다. 샘플러와 step 수가 같으므로 이 차이는 신경망의 근사 오차다. denoising 손실 0.317은 셀 2의 0.471과 직접 비교할 수 없다. 셀 2는 $t = 300$ 하나의 값이고 이 손실은 모든 $t$의 평균이며, 최적해에서도 0이 아닌 하한이 $t$마다 다르다. **손실 값만으로는 남은 오차가 신경망 때문인지 피할 수 없는 부분인지 알 수 없다.** 정확한 score를 아는 장난감 문제가 유용한 이유다.

이어서 다음을 직접 해 본다.

1. 셀 3·4의 샘플러에서 `eps_star`를 바꿔 가며 step 수 10, 50, 200에서 DDPM과 DDIM의 W1을 표로 만든다.
2. 19.1장의 2차원 원형 데이터로 같은 실험을 한다. 성분이 8개인 가우시안 혼합이므로 `comps`와 `score`를 2차원으로 고치면 정확한 score를 그대로 쓸 수 있다.
3. 연습문제에서 forward 과정의 사후분포 $q(x_{t-1} \mid x_t, x_0)$를 구현한다. reverse 과정 한 step의 평균과 분산이 여기서 나온다.

::exercise{id="ddpm-posterior"}

## 실패 사례

- **학습 시와 샘플링 시 잡음 스케줄 불일치.** 신경망은 학습한 스케줄의 시각 $t$에서 $\boldsymbol{\varepsilon}$을 예측하도록 배웠다. 샘플러가 다른 $\bar\alpha_t$를 쓰면, 신경망이 예측한 잡음의 크기와 샘플러가 빼는 잡음의 크기가 어긋난다. 셀 4의 마지막 두 줄은 score는 정확하고 샘플러만 $\beta$의 끝값이 절반인 스케줄을 믿은 경우로, 1000 step에서도 W1이 0.123으로 정상(0.004)의 30배다. 다른 사람의 checkpoint를 쓰거나 샘플러 라이브러리를 바꿀 때 스케줄의 종류·$T$·시각의 0/1 기준을 함께 확인한다.
- **step 수를 밝히지 않고 표본 품질을 비교하는 것.** 같은 모델도 5 step과 1000 step의 결과가 크게 다르다(셀 3). 모델 비교에는 샘플러 종류와 step 수(또는 신경망 호출 횟수)를 고정하고 보고한다.
- **guidance를 켠 표본으로 조건부분포를 평가하는 것.** $w > 0$이면 표본은 $p(\mathbf{x} \mid \mathbf{c})$보다 좁고 치우친 분포에서 나온다(셀 5). 앙상블의 퍼짐을 불확실성으로 해석하려는 과학 응용에서는 $w$가 불확실성을 직접 줄인다는 점을 기억한다.
- **denoising 손실로 모델을 고르는 것.** 손실에는 $t$마다 다른 피할 수 없는 하한이 섞여 있고, $t$를 고르는 분포와 가중치에 따라 값이 달라진다. 표본 기반 지표나 같은 조건의 ELBO로 비교한다.
- **역문제의 근사 사후 표본을 정확한 사후분포로 해석하는 것.** 관측 우도 항의 근사 때문에 표본의 퍼짐이 실제 사후 불확실성과 다를 수 있다. 참 사후분포를 아는 합성 문제에서 calibration을 먼저 확인한다.

## 통과 기준

- forward 과정의 닫힌 형태 $q(x_t \mid x_0)$를 유도하고 한 step씩 시뮬레이션한 결과와 비교한다.
- denoising 목적함수의 최적해가 $\boldsymbol{\varepsilon}^\star = -\sqrt{1 - \bar\alpha_t}\,\nabla\log p_t$임을 설명하고, DDPM과 DDIM 샘플러를 구현해 각 단계를 수식과 대응시킨다.
- 정확한 score와 학습한 신경망으로 step 수에 따른 표본 품질(W1 등)을 비교하는 표를 만들고, 이산화 오차와 신경망 오차를 구분해 해석한다.
- 연습문제 `ddpm-posterior`의 테스트를 통과한다.

## 참고 문헌

- Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML 2015*, PMLR 37, 2256–2265. — 구조를 지우는 forward 과정과 학습된 reverse 과정
- Ho, Jain, Abbeel (2020). Denoising diffusion probabilistic models. *NeurIPS 33*, 6840–6851. arXiv:2006.11239. — 가중 변분 하한, denoising score matching과의 연결, ε 예측
- Song, Meng, Ermon (2021). Denoising diffusion implicit models. *ICLR 2021*. arXiv:2010.02502. — 같은 학습 결과로 쓰는 결정론적·빠른 샘플러
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*. arXiv:2011.13456. — forward·reverse SDE와 probability flow ODE
- Hyvärinen (2005). Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 695–709. — score matching
- Vincent (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 1661–1674. — denoising score matching
- Ho, Salimans (2022). Classifier-free diffusion guidance. arXiv:2207.12598 (짧은 판: NeurIPS 2021 Workshop on Deep Generative Models). — 분류기 없는 guidance
