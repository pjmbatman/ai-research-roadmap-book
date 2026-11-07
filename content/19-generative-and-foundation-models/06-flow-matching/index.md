---
id: generative-and-foundation-models/flow-matching
title: "Flow Matching과 연속시간 생성모델"
title_en: "Flow Matching and Continuous-Time Generative Models"
tier: advanced
status: draft
prerequisites: [generative-and-foundation-models/diffusion-models, numerical-methods/ode-solvers]
---

# Flow Matching과 연속시간 생성모델

확산 모델(19.5장)은 잡음을 조금씩 걷어 내며 데이터를 만든다. 같은 생성 과정을 다르게 볼 수도 있다. 잡음 분포의 각 점이 시간에 따라 움직여 데이터 분포의 점에 도착한다고 보고, 그 움직임을 정하는 **속도장**(velocity field)을 학습하는 것이다. 생성은 이 속도장으로 정의된 ODE를 푸는 일이 되고, 9.6장의 ODE 수치해법이 그대로 샘플러가 된다.

이 관점은 오래전부터 있었지만, 속도장을 학습하려면 학습 중에 ODE를 풀어야 해서 비쌌다. **flow matching**은 이 문제를 회귀 문제로 바꾼다. 데이터 점과 잡음 점을 잇는 단순한 경로를 미리 정하고, 그 경로의 속도를 신경망이 맞히게 한다. 이 장은 그 목적함수를 유도하고, 1차원 장난감 문제에서 참 속도장을 닫힌 형태로 계산해 학습 결과와 비교한다. 그 비교로 생성 품질의 오차가 **모델**에서 오는지 **ODE solver**에서 오는지 나누는 법을 익히고, 확산 모델이 같은 틀의 한 선택이라는 것을 계산으로 확인한다.

## 학습 목표

- 속도장과 ODE로 분포를 옮기는 연속시간 생성모델을 설명하고, flow matching 목적함수를 조건부 경로로부터 유도한다
- 가우시안 경로에서 flow matching과 확산 모델이 서로 바꿀 수 있는 양을 학습한다는 것을 보이고, 둘의 실제 차이를 설명한다
- ODE step 수에 따른 샘플 품질을 측정해 solver 오차와 모델 오차를 구분하고, rectified flow의 효과와 한계를 평가한다

## 연속 정규화 흐름

**왜 필요한가.** 19.4장의 normalizing flow는 가역 변환을 층으로 쌓고 각 층의 Jacobian 행렬식을 계산해 정확한 likelihood를 얻는다. 그 대가로 층의 구조가 크게 제한된다. 변환을 이산적인 층 대신 연속적인 시간의 흐름으로 쓰면, 구조 제약 없이 아무 신경망이나 속도장으로 쓸 수 있다.

**정의.** 시간 $t \in [0, 1]$에서 속도장 $v_\theta(\mathbf{x}, t)$가 정의하는 ODE

$$
\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}, t), \qquad \mathbf{x}(0) \sim p_0
$$

를 생각한다. $p_0$는 표준 가우시안 같은 단순한 분포다. 각 점을 시간 1까지 흘려보낸 $\mathbf{x}(1)$의 분포를 $p_1$이라 하고, 이것을 데이터 분포에 맞추는 것이 목표다. 이 모델을 **연속 정규화 흐름**(continuous normalizing flow, CNF)이라 한다(Chen 외, 2018). 중간 시각의 분포 $p_t$들의 모임을 **확률 경로**(probability path)라 한다. 이 장은 $t = 0$을 잡음, $t = 1$을 데이터로 둔다. 19.5장의 확산 모델은 보통 반대 방향($t = 0$이 데이터)으로 시간을 쓰므로, 두 장의 식을 비교할 때 $t \leftrightarrow 1 - t$로 바꿔 읽는다.

**직관.** 잉크 방울이 섞인 물의 흐름을 생각한다. 물의 속도장이 주어지면 각 잉크 입자가 어디로 갈지 정해지고, 입자들의 밀도도 함께 정해진다. 흐름이 모이는 곳(속도장의 발산이 음수인 곳)에서는 밀도가 높아지고, 퍼지는 곳에서는 낮아진다. 이 관계를 식으로 쓴 것이 연속 방정식(continuity equation) $\partial_t p_t + \nabla \cdot (p_t v) = 0$이고, 한 입자를 따라가며 보면

$$
\frac{d}{dt}\log p_t\big(\mathbf{x}(t)\big) = -\nabla \cdot v\big(\mathbf{x}(t), t\big)
$$

가 된다. 19.4장의 변수변환 공식에서 Jacobian 행렬식의 로그가 했던 일을, 연속 시간에서는 발산(Jacobian의 대각합)의 시간 적분이 한다. 이것을 **순간 변수변환 공식**(instantaneous change of variables)이라 한다.

**코드.** $\mathcal{N}(0, 1)$을 $\mathcal{N}(2, 0.5^2)$로 옮기는 속도장은 닫힌 형태로 쓸 수 있다. $\mathbf{x}_t = (1 - t + ts)\mathbf{x}_0 + t\mu$를 시간으로 미분해 $\mathbf{x}_0$를 $\mathbf{x}_t$로 바꿔 쓰면 된다. 점들을 Euler 방법(9.6장)으로 옮기면서, 같은 루프에서 발산을 적분해 각 점의 로그 밀도를 추적한다.

```python {exec="browser" session="fm"}
import numpy as np

rng = np.random.default_rng(0)
mu, s = 2.0, 0.5                                   # 목표 분포 N(2, 0.5²)

def v(x, t):                                       # N(0,1)을 N(mu, s²)로 옮기는 속도장
    return mu + (s - 1) * (x - t * mu) / (1 - t + t * s)

x = rng.normal(size=5000)                          # x_0 ~ N(0, 1), (N,)
logp = -0.5 * x**2 - 0.5 * np.log(2 * np.pi)       # log p_0(x_0), (N,)
n_steps = 100
dt = 1.0 / n_steps
for k in range(n_steps):
    t = k * dt
    logp -= (s - 1) / (1 - t + t * s) * dt         # d log p / dt = -∂v/∂x
    x = x + v(x, t) * dt                           # Euler step
exact = -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
print(f"x_1 평균 {x.mean():.3f}, 표준편차 {x.std():.3f}  (목표 {mu}, {s})")
print(f"log p_1 오차 최대 {np.abs(logp - exact).max():.4f}  (Euler {n_steps} step)")
```

```output
x_1 평균 1.998, 표준편차 0.498  (목표 2.0, 0.5)
log p_1 오차 최대 0.0025  (Euler 100 step)
```

평균과 표준편차가 목표와 다른 것은 5,000개 표본의 표집 잡음뿐이다. 이 속도장은 각 점을 **직선**으로, 일정한 속력으로 옮기므로 Euler 방법이 점의 위치를 정확히 따라간다. 반면 로그 밀도는 $-\int_0^1 \partial_x v\,dt = -\log s$라는 적분을 Euler 합으로 근사하므로 0.0025의 오차가 남는다. 경로가 직선이면 적은 step으로도 정확하다는 이 관찰이 뒤의 rectified flow로 이어진다.

**연구에서의 의미.** CNF는 likelihood를 계산할 수 있는 생성모델이지만, 최대가능도로 학습하려면 학습 단계마다 ODE와 발산 적분을 함께 풀어야 한다. 고차원에서 발산(Jacobian의 대각합)을 정확히 계산하는 비용도 크다. 학습 비용이 이 계열의 걸림돌이었고, flow matching은 이 비용을 없애는 방법이다. 학습이 끝난 모델의 likelihood가 필요하면 여전히 위와 같은 적분을 해야 한다는 점은 기억해 둔다.

## flow matching

**왜 필요한가.** 속도장을 학습하는 가장 직접적인 방법은 원하는 확률 경로 $p_t$와 그 경로를 만드는 속도장 $u_t$를 알고, $v_\theta$를 $u_t$에 회귀시키는 것이다.

$$
L_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\ \mathbf{x} \sim p_t}\big\lVert v_\theta(\mathbf{x}, t) - u_t(\mathbf{x}) \big\rVert^2.
$$

문제는 데이터 분포를 모르므로 $p_t$도 $u_t$도 모른다는 것이다.

**조건부 경로.** Lipman 외(2023)의 해법은 **데이터 점 하나에 조건을 건** 경로를 쓰는 것이다. 잡음 $\mathbf{x}_0 \sim p_0$와 데이터 $\mathbf{x}_1 \sim p_{\text{data}}$를 뽑아 직선으로 잇는다.

$$
\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\mathbf{x}_1, \qquad u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0.
$$

짝 하나의 속도는 알고 있다. 모든 짝의 경로를 모으면 주변 확률 경로 $p_t$가 생기고, 그 경로를 만드는 주변 속도장은 **같은 점을 지나는 조건부 속도들의 평균**이다.

$$
u_t(\mathbf{x}) = \mathbb{E}\big[\mathbf{x}_1 - \mathbf{x}_0 \,\big|\, \mathbf{x}_t = \mathbf{x}\big].
$$

**조건부 목적함수.** 그래서 주변 속도 대신 조건부 속도에 회귀한다.

$$
L_{\text{CFM}}(\theta) = \mathbb{E}_{t,\ \mathbf{x}_0,\ \mathbf{x}_1}\big\lVert v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \big\rVert^2.
$$

이 손실은 표본만으로 계산되고 ODE를 풀 필요가 없다(simulation-free). 그런데 이것을 최소화해도 되는 이유가 있다. 제곱을 전개하면 $\theta$에 의존하는 항은 $\lVert v_\theta \rVert^2$와 교차항 $-2\,v_\theta \cdot (\mathbf{x}_1 - \mathbf{x}_0)$뿐이고, 교차항의 기댓값은 $\mathbf{x}_t$에 조건을 걸어 먼저 평균하면 $-2\,v_\theta \cdot u_t(\mathbf{x}_t)$가 된다. 그래서

$$
L_{\text{CFM}}(\theta) = L_{\text{FM}}(\theta) + \underbrace{\mathbb{E}_t\,\mathbb{E}_{\mathbf{x}_t}\operatorname{tr}\operatorname{Cov}\big[\mathbf{x}_1 - \mathbf{x}_0 \,\big|\, \mathbf{x}_t\big]}_{\theta\text{와 무관한 상수}}
$$

이고, 두 손실의 gradient는 같다. 교차항을 정리하는 데 쓴 것은 반복 기댓값의 법칙(11.3장)이고, 결과는 "제곱 손실을 최소화하는 예측은 조건부 기댓값"이라는 익숙한 사실의 한 사례다. 같은 논리로 비슷한 목적함수가 rectified flow(Liu 외, 2023)와 stochastic interpolant(Albergo, Vanden-Eijnden, 2023)라는 이름으로 거의 같은 시기에 제안되었다.

**코드.** 데이터 분포가 혼합 가우시안 $0.3\,\mathcal{N}(-2, 0.3^2) + 0.7\,\mathcal{N}(2, 0.3^2)$이면, 성분마다 $(\mathbf{x}_t, \mathbf{x}_1 - \mathbf{x}_0)$가 결합 가우시안이므로 참 주변 속도장 $v^\star = u_t$를 닫힌 형태로 쓸 수 있다. 신경망 대신 $(x, t)$의 무작위 Fourier 특징 200개 위의 선형 모델을 $v_\theta$로 쓰면, CFM 손실의 최소화가 ridge 회귀 한 번으로 끝난다.

```python {exec="browser" session="fm"}
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(1)
pis, ms, ss = np.array([0.3, 0.7]), np.array([-2.0, 2.0]), np.array([0.3, 0.3])   # 데이터: 혼합 가우시안
def sample_data(n):
    k = rng.choice(2, size=n, p=pis)
    return ms[k] + ss[k] * rng.normal(size=n)

def v_star(x, t):                                  # 참 주변 속도장 E[x1 - x0 | x_t = x] (닫힌 형태)
    var = (1 - t) ** 2 + t**2 * ss[:, None] ** 2   # 성분별 x_t의 분산 (2, N)
    logw = np.log(pis)[:, None] - 0.5 * (x - t * ms[:, None]) ** 2 / var - 0.5 * np.log(var)
    w = np.exp(logw - logw.max(0)); w /= w.sum(0)  # 사후 성분 확률 (2, N)
    return (w * (ms[:, None] + (t * ss[:, None] ** 2 - (1 - t)) / var * (x - t * ms[:, None]))).sum(0)

W_rf, b_rf = rng.normal(size=(2, 200)) * [[1.5], [3.0]], rng.uniform(0, 2 * np.pi, 200)
feat = lambda x, t: np.cos(np.stack([x, t * np.ones_like(x)], 1) @ W_rf + b_rf)   # (N, 200) 무작위 특징

def cfm_data(n, x1=None, x0=None):                 # x_t = (1-t)x0 + t x1, 목표 u = x1 - x0
    x0 = rng.normal(size=n) if x0 is None else x0
    x1 = sample_data(n) if x1 is None else x1
    t = rng.uniform(size=n)
    return (1 - t) * x0 + t * x1, t, x1 - x0

def fit(xt, t, u, lam=1e-3):                       # CFM 손실의 최소화 = ridge 회귀의 해
    F = feat(xt, t)
    return np.linalg.solve(F.T @ F + lam * len(u) * np.eye(F.shape[1]), F.T @ u)

theta = fit(*cfm_data(20000))
v_theta = lambda x, t: feat(x, t) @ theta
xt_te, t_te, u_te = cfm_data(20000)                # test용 (x_t, t, u)
loss = lambda f: np.mean((f(xt_te, t_te) - u_te) ** 2)
print(f"CFM 손실 (test)   v = 0: {loss(lambda x, t: 0 * x):.3f}   학습한 v_θ: {loss(v_theta):.3f}   참 v*: {loss(v_star):.3f}")
print(f"E|v_θ - v*|² = {np.mean((v_theta(xt_te, t_te) - v_star(xt_te, t_te)) ** 2):.3f}")
```

```output
CFM 손실 (test)   v = 0: 5.171   학습한 v_θ: 2.256   참 v*: 2.214
E|v_θ - v*|² = 0.032
```

**참 속도장의 CFM 손실도 0이 아니라 2.214다.** 이 값이 위 식의 상수항, 즉 같은 $\mathbf{x}_t$를 지나는 여러 짝의 속도가 서로 다른 데서 오는 분산이다. 학습한 $v_\theta$의 손실 2.256에서 이 바닥을 빼면 0.042이고, 직접 잰 $\mathbb{E}\lVert v_\theta - v^\star \rVert^2 = 0.032$와 같은 크기다(두 값은 서로 다른 무작위 표본에서 잰 추정이라 정확히 같지는 않다). 손실 값 자체로는 모델이 얼마나 좋은지 알 수 없고, 바닥을 알 때만 해석할 수 있다는 뜻이다. 실제 문제에서는 바닥을 모르므로, CFM 손실은 같은 경로·같은 데이터에서 모델끼리 비교하는 데만 쓴다.

::quiz{id="cfm-loss-floor"}

**연구에서의 의미.** flow matching의 학습은 회귀다. 학습 루프는 확산 모델과 거의 같고(잡음을 뽑고, 섞고, 목표를 맞힌다), 경로와 목표만 다르다. 데이터를 조건 $c$와 함께 쓰면($v_\theta(\mathbf{x}, t, c)$) 조건부 생성이 되고, 19.5장의 classifier-free guidance도 속도장에 같은 방식으로 적용할 수 있다.

:::note[분야 예시]
기후 모델의 고해상도 강수장을 저해상도 조건에서 생성하는 다운스케일링(27.3장)은 조건부 생성 문제다. 속도장 $v_\theta(\mathbf{x}, t, c)$의 조건 $c$에 저해상도 장을 넣고, 같은 조건에서 여러 번 ODE를 풀면 고해상도 장의 앙상블을 얻는다. 이때 표본 하나가 그럴듯한지보다, 앙상블이 강수 극값의 빈도와 공간 상관을 맞히는지를 19.7장의 방식으로 평가해야 한다.
:::

## rectified flow와 최적 수송

**왜 필요한가.** 생성 비용은 ODE를 풀 때 속도장을 몇 번 계산하는가(function evaluation 수, NFE)로 정해진다. 경로가 곧을수록 적은 step으로 정확하게 풀 수 있다. 앞 절의 조건부 경로는 직선이지만, 그렇다고 **주변** 흐름의 경로가 직선인 것은 아니다.

**왜 굽는가.** $\mathbf{x}_0$와 $\mathbf{x}_1$을 독립으로 뽑으면 서로 다른 짝의 직선들이 교차한다. 교차점에서 주변 속도장은 여러 방향의 평균을 내므로, 실제 ODE 궤적은 어느 짝의 직선도 따라가지 않고 굽는다. 위 실험에서 참 속도장으로도 손실 바닥이 2.214였던 것이 바로 이 교차의 양이다.

**solver 오차와 모델 오차.** 같은 $\mathbf{x}_0$ 2,000개를 참 속도장 $v^\star$와 학습한 $v_\theta$로 각각 Euler step 수를 바꿔 풀고, 참 속도장을 1,000 step으로 푼 결과와의 1차원 Wasserstein 거리 $W_1$(14.1장)을 잰다. $v^\star$ 열의 오차는 순수한 solver 오차이고, $v_\theta$ 열은 solver 오차와 모델 오차가 더해진 것이다.

```python {exec="browser" session="fm"}
def euler(vf, x0, n_steps):                        # dx/dt = v(x, t), t: 0 → 1
    x, dt = x0.copy(), 1.0 / n_steps
    for k in range(n_steps):
        x = x + vf(x, np.full_like(x, k * dt)) * dt
    return x

x0 = rng.normal(size=2000)                         # 모든 설정에 같은 x0 (2000,)
ref = euler(v_star, x0, 1000)                      # 기준: 참 속도장을 아주 잘게 푼 결과
print("step   W1(참 v*)   W1(학습한 v_θ)")
for n in [1, 2, 4, 8, 32, 128]:
    print(f"{n:4d}     {wasserstein_distance(euler(v_star, x0, n), ref):.3f}       {wasserstein_distance(euler(v_theta, x0, n), ref):.3f}")
```

```output
step   W1(참 v*)   W1(학습한 v_θ)
   1     1.667       1.546
   2     0.627       0.769
   4     0.283       0.365
   8     0.131       0.194
  32     0.037       0.087
 128     0.007       0.067
```

두 영역이 보인다.

- **step이 적을 때(1–8)는 solver 오차가 지배한다.** 참 속도장도 1 step에서는 $W_1 = 1.667$로 형편없다. 이 영역에서 학습한 모델이 나쁘다고 결론 내리면 안 된다.
- **step이 많을 때(32 이상)는 모델 오차가 남는다.** 참 속도장은 0.007까지 내려가지만, 학습한 속도장은 0.067 근처에서 멈춘다. step을 더 늘려도 이 차이는 줄지 않는다.

ODE 생성모델의 품질을 보고할 때는 이렇게 step 수를 늘려 품질이 수렴하는지 확인하고, 수렴한 값을 모델의 품질로, 그보다 적은 step의 값을 비용–품질 trade-off로 따로 보고한다.

::quiz{id="solver-vs-model-error"}

**rectified flow.** Liu 외(2023)는 경로를 곧게 펴는 절차를 제안했다. 먼저 독립 짝으로 흐름을 학습한다. 그 흐름으로 새 잡음 $\mathbf{z}_0$를 풀어 $\mathbf{z}_1$을 얻으면, 이번에는 $(\mathbf{z}_0, \mathbf{z}_1)$이 **결정적인 짝**(coupling)이다. 이 짝들로 같은 직선 경로의 flow matching을 다시 학습한다. 이 절차를 **reflow**라 한다. ODE 궤적은 서로 교차하지 않으므로 새 짝의 직선들도 (흐름이 정확하다면) 덜 교차하고, 그만큼 주변 경로가 곧아진다. 저자들은 reflow가 볼록 비용에 대한 수송 비용을 늘리지 않는다는 것을 보였다.

```python {exec="browser" session="fm"}
z0 = rng.normal(size=12000)                        # 새 잡음 (12000,)
z1 = euler(v_theta, z0, 32)                        # 학습한 흐름이 정한 짝 z0 → z1 (결정적 coupling)
tr, te = slice(0, 8000), slice(8000, None)
theta_r = fit(*cfm_data(8000, x1=z1[tr], x0=z0[tr]))   # 같은 직선 경로, 짝만 바꾼다
v_reflow = lambda x, t: feat(x, t) @ theta_r
xt_q, t_q, u_q = cfm_data(4000, x1=z1[te], x0=z0[te])
print(f"CFM 손실: 독립 짝 {loss(v_theta):.3f}  →  reflow 짝 {np.mean((v_reflow(xt_q, t_q) - u_q) ** 2):.3f}")
print("step   W1(1차 흐름 v_θ)   W1(reflow)")
for n in [1, 2, 4, 32]:
    print(f"{n:4d}       {wasserstein_distance(euler(v_theta, x0, n), ref):.3f}            {wasserstein_distance(euler(v_reflow, x0, n), ref):.3f}")
```

```output
CFM 손실: 독립 짝 2.256  →  reflow 짝 0.011
step   W1(1차 흐름 v_θ)   W1(reflow)
   1       1.546            0.226
   2       0.769            0.192
   4       0.365            0.154
  32       0.087            0.123
```

**CFM 손실이 2.256에서 0.011로 떨어진다.** 짝을 바꾸자 같은 $\mathbf{x}_t$를 지나는 짝들의 속도가 거의 하나로 정해져, 손실의 바닥이 사라졌다. 경로가 곧아졌다는 직접적인 증거다. 그 결과 **1 step 생성의 $W_1$이 1.546에서 0.226으로** 줄었다.

그러나 공짜는 아니다. 32 step에서는 reflow(0.123)가 1차 흐름(0.087)보다 나쁘다. reflow는 1차 흐름이 만든 표본을 정답으로 삼으므로 1차 흐름의 모델 오차를 물려받고, 다시 학습하면서 생기는 회귀 오차를 더한다. 이 예에서 reflow 짝의 속도장은 두 성분 사이에서 급하게 바뀌어(잡음 쪽의 이웃한 두 점이 서로 다른 성분으로 간다) 매끄러운 특징 200개로는 맞히기 어렵다. reflow는 **많은 step의 품질을 약간 내주고 적은 step의 품질을 크게 얻는** 거래다.

**최적 수송과의 관계.** 1차원에서 ODE 궤적은 서로 추월할 수 없으므로 흐름이 만드는 짝은 순서를 보존하고, 순서를 보존하는 짝은 볼록 비용(예: $\lvert x_1 - x_0 \rvert^2$)에 대한 최적 수송(optimal transport) 짝이다. 그래서 1차원의 reflow 짝은 곧바로 최적 수송 짝이고, 그 짝의 직선들은 교차하지 않는다. 위에서 손실 바닥이 거의 0이 된 이유다. 고차원에서는 한 번의 reflow로 최적 수송 짝이 되지 않으며, 경로를 곧게 하는 것과 최적 수송을 푸는 것은 일반적으로 다른 문제다.

::widget{name="flow-paths"}

## 확산 모델과의 관계

**같은 틀.** 19.5장의 확산 모델이 만드는 중간 분포도 $\mathbf{x}_t = \alpha_t\,\mathbf{x}_1 + \sigma_t\,\boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 꼴의 **가우시안 경로**다. flow matching의 직선 경로는 $\alpha_t = t$, $\sigma_t = 1 - t$인 특수한 경우다. 가우시안 경로에서는 네 가지 예측 대상이 서로 바뀐다.

| 예측 대상 | 의미 | 다른 것으로 |
|---|---|---|
| $\hat{\boldsymbol{\varepsilon}} = \mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{x}_t]$ | 섞인 잡음 (19.5장의 ε-예측) | $\hat{\boldsymbol{\varepsilon}} = -\sigma_t\,\nabla\log p_t(\mathbf{x}_t)$ |
| $\hat{\mathbf{x}}_1 = \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t]$ | 깨끗한 데이터 (denoiser) | $\hat{\mathbf{x}}_1 = (\mathbf{x}_t - \sigma_t\hat{\boldsymbol{\varepsilon}})/\alpha_t$ |
| $\nabla\log p_t(\mathbf{x}_t)$ | score | Tweedie 공식으로 $\hat{\boldsymbol{\varepsilon}}$와 연결 |
| $v = \mathbb{E}[\dot\alpha_t\mathbf{x}_1 + \dot\sigma_t\boldsymbol{\varepsilon} \mid \mathbf{x}_t]$ | 속도 | $v = \dot\alpha_t\hat{\mathbf{x}}_1 + \dot\sigma_t\hat{\boldsymbol{\varepsilon}}$ |

마지막 줄에 앞의 두 줄을 넣으면 score로 쓴 속도가 나온다.

$$
v(\mathbf{x}, t) = \frac{\dot\alpha_t}{\alpha_t}\,\mathbf{x} + \Big(\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2 - \dot\sigma_t\sigma_t\Big)\nabla\log p_t(\mathbf{x}).
$$

$\dot\alpha_t, \dot\sigma_t$는 시간 미분이다. 이 속도장의 ODE가 19.5장에서 본 확산 모델의 probability flow ODE와 같은 주변 분포를 만든다. 혼합 가우시안에서 score를 닫힌 형태로 계산해 확인한다.

```python {exec="browser" session="fm"}
def score(x, t):                                   # ∇x log p_t(x), 혼합 가우시안의 닫힌 형태
    var = (1 - t) ** 2 + t**2 * ss[:, None] ** 2
    logw = np.log(pis)[:, None] - 0.5 * (x - t * ms[:, None]) ** 2 / var - 0.5 * np.log(var)
    w = np.exp(logw - logw.max(0)); w /= w.sum(0)
    return (w * (-(x - t * ms[:, None]) / var)).sum(0)

xg = np.linspace(-4, 4, 201)                       # 격자 (201,)
for t in [0.1, 0.5, 0.9]:
    a, sig, da, dsig = t, 1 - t, 1.0, -1.0         # x_t = a·x1 + sig·ε 와 시간 미분
    sc = score(xg, t)
    v_from_score = da / a * xg + (da / a * sig**2 - dsig * sig) * sc
    eps_hat = -sig * sc                            # 확산 모델의 ε-예측 E[ε | x_t]
    x1_hat = (xg - sig * eps_hat) / a              # denoiser E[x1 | x_t]
    ok1 = np.abs(v_from_score - v_star(xg, t)).max() < 1e-8
    ok2 = np.abs((x1_hat - xg) / (1 - t) - v_star(xg, t)).max() < 1e-8
    print(f"t = {t}:  score로 만든 v = v*: {ok1}   (x̂1 - x)/(1-t) = v*: {ok2}")
```

```output
t = 0.1:  score로 만든 v = v*: True   (x̂1 - x)/(1-t) = v*: True
t = 0.5:  score로 만든 v = v*: True   (x̂1 - x)/(1-t) = v*: True
t = 0.9:  score로 만든 v = v*: True   (x̂1 - x)/(1-t) = v*: True
```

직선 경로의 속도는 "지금 위치에서 예측한 깨끗한 데이터 쪽으로, 남은 시간 $1 - t$ 안에 도착하는 속도" $(\hat{\mathbf{x}}_1 - \mathbf{x})/(1 - t)$다. 완벽하게 학습된 ε-예측 확산 모델과 완벽하게 학습된 flow matching 모델은 같은 경로를 쓰는 한 **같은 정보**를 담는다.

::quiz{id="velocity-from-eps"}

**다른 선택.** 그래서 두 계열의 실제 차이는 다음 선택에 있다.

- **경로**($\alpha_t, \sigma_t$): 직선, 확산 모델의 분산 보존(VP) 경로, 삼각함수 경로 등. 경로가 바뀌면 같은 데이터에서도 속도장의 곡률과 필요한 step 수가 바뀐다.
- **예측 대상과 시간별 가중치**: ε, $\mathbf{x}_1$, $v$ 중 무엇을 신경망 출력으로 둘지는 같은 손실을 시간마다 다르게 가중하는 것과 같다. 완벽한 모델에서는 차이가 없지만, 용량이 유한한 모델에서는 어느 시간 구간의 오차를 줄일지가 달라진다.
- **샘플러**: 확산 모델은 확률적 SDE 샘플러와 결정적 ODE 샘플러를 모두 쓰고, flow matching은 보통 ODE를 쓴다. ODE 샘플러에는 9.6장의 고차 방법과 적응적 step을 그대로 쓸 수 있다.
- **비가우시안 $p_0$와 짝 설계**: flow matching의 틀은 잡음 분포가 가우시안이 아니어도, 짝이 독립이 아니어도(reflow, 두 데이터 분포 사이의 변환) 그대로 쓸 수 있다. 이것이 확산 모델 틀보다 넓은 부분이다.

"flow matching이 확산 모델보다 낫다"는 말은 이 선택들의 특정 조합에 대한 실험 결과이지, 틀 자체의 우열이 아니다.

## 실습

같은 2차원 데이터(원 위의 가우시안 8개)에, 같은 MLP와 같은 학습 예산으로 두 경로를 학습한다. 하나는 직선 경로, 다른 하나는 확산 모델의 분산 보존 경로와 비슷한 삼각함수 경로($\alpha_t = \sin(\pi t/2)$, $\sigma_t = \cos(\pi t/2)$)다. 품질은 무작위 방향 64개로 투영한 1차원 $W_1$의 평균(sliced $W_1$)으로 잰다(로컬 실습 환경이 필요하다).

```python {exec="python"}
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
rng = np.random.default_rng(0)
ang = 2 * np.pi * np.arange(8) / 8
centers = 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1)           # 원 위의 가우시안 8개 (8, 2)
data = lambda n: torch.tensor(centers[rng.integers(8, size=n)] + 0.15 * rng.normal(size=(n, 2)), dtype=torch.float32)
half_pi = np.pi / 2
paths = {                                                          # x_t = a(t)·x1 + s(t)·ε 와 시간 미분
    "직선": (lambda t: t, lambda t: 1 - t, lambda t: 1 + 0 * t, lambda t: -1 + 0 * t),
    "삼각(VP류)": (lambda t: torch.sin(half_pi * t), lambda t: torch.cos(half_pi * t),
                 lambda t: half_pi * torch.cos(half_pi * t), lambda t: -half_pi * torch.sin(half_pi * t)),
}
proj = torch.nn.functional.normalize(torch.randn(2, 64), dim=0)    # 무작위 방향 64개 (2, 64)
sliced_w1 = lambda a, b: (torch.sort(a @ proj, 0)[0] - torch.sort(b @ proj, 0)[0]).abs().mean().item()
ref = data(4000)
print(f"기준: 데이터 두 표본의 sliced W1 = {sliced_w1(data(4000), ref):.3f}")
print("경로        끝 손실   1 step  2 step  4 step  8 step  32 step")
for name, (a, s, da, ds) in paths.items():
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt, tail = torch.optim.Adam(net.parameters(), lr=2e-3), []
    for step in range(3000):
        x1, eps, t = data(256), torch.randn(256, 2), torch.rand(256, 1)    # (B, 2), (B, 2), (B, 1)
        xt, u = a(t) * x1 + s(t) * eps, da(t) * x1 + ds(t) * eps          # 조건부 경로와 목표 속도
        loss = ((net(torch.cat([xt, t], 1)) - u) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tail.append(loss.item())
    w1 = []
    with torch.no_grad():
        for n in [1, 2, 4, 8, 32]:
            x = torch.randn(4000, 2, generator=torch.Generator().manual_seed(1))
            for k in range(n):
                x = x + net(torch.cat([x, torch.full((4000, 1), k / n)], 1)) / n
            w1.append(sliced_w1(x, ref))
    print(f"{name:10} {np.mean(tail[-300:]):7.3f}  " + "  ".join(f"{w:6.3f}" for w in w1))
```

```output
기준: 데이터 두 표본의 sliced W1 = 0.031
경로        끝 손실   1 step  2 step  4 step  8 step  32 step
직선           1.937   1.099   0.549   0.178   0.093   0.068
삼각(VP류)      3.107   0.516   0.199   0.126   0.106   0.100
```

세 가지를 읽는다.

- **끝 손실(1.937과 3.107)은 비교할 수 없다.** 두 경로는 목표 속도의 크기와 손실 바닥이 다르다. 손실이 낮다고 직선 경로가 더 잘 학습된 것이 아니다.
- **적은 step에서는 이 실험의 삼각 경로가 더 낫다(1 step 0.516 대 1.099).** 직선 **조건부** 경로가 곧은 **주변** 경로를 보장하지 않는다는 앞 절의 내용과 맞는다. 독립 짝을 쓰는 한, 경로의 모양만으로 step 효율을 예측할 수 없다.
- **32 step에서는 직선 경로가 낫다(0.068 대 0.100).** 둘 다 데이터 두 표본 사이의 값 0.031보다 크므로, 이 영역의 차이는 모델 오차의 차이다. 한 번의 시드로 얻은 결과이므로 결론을 내리려면 12.5장의 방식으로 시드를 바꿔 반복해야 한다.

이어서 다음을 수행한다.

1. 시드 5개로 위 실험을 반복하고, step 수별 sliced $W_1$의 평균과 표준편차를 표로 만든다.
2. 19.5장에서 구현한 확산 모델(ε-예측, 같은 MLP)을 같은 데이터에 학습해, 같은 NFE에서의 품질과 학습 손실 곡선의 흔들림을 비교한다. ε-예측을 위 표의 관계로 속도로 바꾸면 같은 Euler 샘플러를 쓸 수 있다.
3. 직선 경로 모델에 reflow를 한 번 적용하고, 1·2·4 step 품질이 얼마나 바뀌는지, 32 step 품질은 얼마나 잃는지 잰다.

마지막으로 조건부 경로의 표본과 목표를 만드는 함수, ε-예측을 속도로 바꾸는 함수를 구현한다. 테스트가 바로 채점한다.

::exercise{id="cfm-target"}

## 실패 사례

- **ODE solver 오차를 모델 오차로 오해하는 것.** step이 적은 설정에서 나쁜 품질을 보고 모델 구조나 학습을 바꾸는 경우가 흔하다. step을 늘려 품질이 수렴하는지 먼저 확인한다. 참 속도장으로도 1 step의 $W_1$은 1.667이었다. 반대로 많은 step에서도 남는 차이는 solver를 바꿔도 줄지 않는다.
- **CFM 손실 값을 모델 품질로 읽는 것.** 참 속도장에서도 손실은 0이 아니고(2.214), 경로가 다르면 손실의 크기 자체가 다르다. 손실은 같은 경로·같은 데이터에서의 상대 비교에만 쓰고, 품질은 생성된 표본으로 잰다(19.7장).
- **"직선 경로이므로 1 step이면 충분하다"고 가정하는 것.** 직선인 것은 짝 하나의 조건부 경로다. 독립 짝에서 주변 경로는 굽고, 곧게 하려면 reflow 같은 추가 절차가 필요하며, 그 절차에는 품질 비용이 따른다.
- **시간 방향을 섞는 것.** 이 장은 $t = 0$이 잡음, 19.5장의 확산 모델은 $t = 0$이 데이터인 관례를 쓴다. 공개 구현을 가져다 쓸 때 시간 방향과 $\alpha_t, \sigma_t$의 정의를 확인하지 않으면 학습과 샘플링이 서로 다른 경로를 가정하게 된다.

## 통과 기준

- 조건부 flow matching 목적함수 $L_{\text{CFM}}$이 $L_{\text{FM}}$과 상수만큼 다르다는 것을 유도하고, 그 상수가 손실의 바닥이 되는 이유를 설명한다.
- 가우시안 경로에서 ε-예측·score·속도 사이의 변환식을 쓰고, 연습문제 `cfm-target`의 테스트를 통과한다.
- 학습한 생성모델의 품질을 step 수별로 측정해 solver 오차가 사라지는 지점을 찾고, 그 지점의 값을 모델의 품질로 보고한다.

## 참고 문헌

- Chen, Rubanova, Bettencourt, Duvenaud (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems 31* (NeurIPS 2018). arXiv:1806.07366. — 연속 정규화 흐름과 순간 변수변환 공식
- Lipman, Chen, Ben-Hamu, Nickel, Le (2023). Flow matching for generative modeling. *International Conference on Learning Representations* (ICLR 2023). arXiv:2210.02747. — 조건부 확률 경로의 속도장 회귀, 확산 경로를 포함하는 가우시안 경로 족
- Liu, Gong, Liu (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR 2023*. arXiv:2209.03003. — 직선 경로와 reflow, 수송 비용이 늘지 않는 결정적 짝
- Albergo, Vanden-Eijnden (2023). Building normalizing flows with stochastic interpolants. *ICLR 2023*. arXiv:2209.15571. — 두 분포 사이의 보간으로 속도장을 이차 손실로 학습
