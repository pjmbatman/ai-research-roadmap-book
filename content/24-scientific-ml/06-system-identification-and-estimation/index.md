---
id: scientific-ml/system-identification-and-estimation
title: "시스템 식별과 상태 추정"
title_en: "System Identification and State Estimation"
tier: core
status: draft
prerequisites: [scientific-ml/physics-and-learning, probability/multivariate-gaussian]
---

# 시스템 식별과 상태 추정

동역학 모델을 데이터와 맞추는 일에는 두 방향이 있다. 하나는 측정으로 모델의 **파라미터나 식 자체**를 정하는 **시스템 식별**(system identification)이고, 다른 하나는 모델을 믿고 측정에서 직접 보이지 않는 **상태**를 실시간으로 추정하는 **상태 추정**(state estimation)이다. 반응 속도 상수를 실험으로 정하는 것은 앞의 것이고, 배터리의 충전 상태나 위성의 위치를 잡음 섞인 센서로 따라가는 것은 뒤의 것이다.

두 문제 모두 먼저 물어야 할 질문이 같다. **이 측정이 그 양에 대한 정보를 담고 있는가?** 담고 있지 않으면 어떤 알고리즘도 답을 만들어 낼 수 없고, 알고리즘은 그래도 숫자를 내놓는다. 이 장은 그 질문(식별 가능성과 관측 가능성)에서 시작해 희소 회귀로 방정식을 찾는 방법, 칼만 필터 계열과 particle filter로 상태를 추정하는 방법으로 나아간다.

## 학습 목표

- 식별 가능성과 관측 가능성을 판단한다
- 희소 회귀로 데이터에서 동역학 방정식을 식별한다
- 필터로 상태를 추정한다

## 파라미터 식별

**왜 필요한가.** 추정한 파라미터를 "이 재료의 열전도율은 얼마다"처럼 물리적으로 해석하려면, 그 값이 데이터로 정해져야 한다. 데이터가 파라미터의 어떤 조합만 정하고 개별 값은 정하지 못하는데도 최적화는 늘 어떤 값을 내놓는다. 그 값은 초기값과 최적화기의 습관이 정한 것이다.

**정의.** 모델 출력을 $y(t; \boldsymbol{\theta})$라 하자. 잡음 없는 측정을 원하는 만큼 할 수 있을 때 $\boldsymbol{\theta} \neq \boldsymbol{\theta}'$이면 $y(\cdot; \boldsymbol{\theta}) \neq y(\cdot; \boldsymbol{\theta}')$인 성질을 **구조적 식별 가능성**(structural identifiability)이라 한다. 모델 식 자체의 성질이다. 실제 측정은 유한하고 잡음이 있으므로, 주어진 실험 설계에서 파라미터의 불확실성이 쓸모 있을 만큼 작은지를 따로 묻는다. 이것이 **실용적 식별 가능성**(practical identifiability)이다.

**민감도.** 측정 시각 $t_1, \dots, t_T$에서 출력이 파라미터에 얼마나 민감한지를 모은 **민감도 행렬**

$$
S_{ij} = \frac{\partial y(t_i; \boldsymbol{\theta})}{\partial \log \theta_j}
$$

이 두 질문에 답한다. 로그 파라미터로 미분하면 단위가 다른 파라미터를 상대 변화로 비교할 수 있다. 측정 잡음이 표준편차 $\sigma$의 독립 가우시안이면 Fisher 정보는 $\mathbf{F} = \mathbf{S}^\top\mathbf{S}/\sigma^2$이고, 최우추정량의 공분산은 근사적으로 $\mathbf{F}^{-1}$이다(12.2장). $\mathbf{S}$의 열이 일차종속이면($\mathbf{S}$의 특이값 하나가 0) 그 방향으로 파라미터를 움직여도 출력이 변하지 않는다. 국소적으로 구조적 식별이 불가능하다는 뜻이다. 특이값이 0은 아니지만 작으면 그 방향의 분산 $\sigma^2/s_{\min}^2$이 커져 실용적 식별이 어렵다.

**예: 저항을 받는 낙하.** 질량 $m$인 물체가 속도에 비례하는 저항 $c v$를 받으며 떨어지면 속도는

$$
v(t) = \frac{m g}{c}\left(1 - e^{-c t / m}\right) = v_\infty\left(1 - e^{-t/\tau}\right), \qquad v_\infty = \frac{mg}{c},\ \tau = \frac{m}{c}
$$

이다. $m$과 $c$는 비율 $m/c$로만 나타나므로 속도 측정으로는 따로 정할 수 없다. 같은 식을 종단속도 $v_\infty$와 시상수 $\tau$로 쓰면 두 파라미터는 구조적으로 식별 가능하다. 그러나 측정이 초기 1초뿐이면 $v(t) \approx (v_\infty/\tau)\,t = g\,t$라서 비율만 보인다.

```python {exec="browser" session="sysid"}
import numpy as np

G_MS2 = 9.81
def v_drag(t_s, m_kg, c_kgps):                     # 선형 저항 낙하: v = (m g / c)(1 - exp(-c t / m))
    return m_kg * G_MS2 / c_kgps * (1 - np.exp(-c_kgps * t_s / m_kg))
def v_two(t_s, v_inf_mps, tau_s):                  # 같은 식을 (종단속도, 시상수)로 쓴 것
    return v_inf_mps * (1 - np.exp(-t_s / tau_s))

def sens(f, theta, t_s, h=1e-6):                   # 로그 파라미터에 대한 민감도 행렬 (T, P)
    cols = []
    for j in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[j] *= np.exp(h); dn[j] *= np.exp(-h)
        cols.append((f(t_s, *up) - f(t_s, *dn)) / (2 * h))
    return np.column_stack(cols)

t_long_s = np.linspace(0.2, 10, 50)
sv = np.linalg.svd(sens(v_drag, np.array([2.0, 0.5]), t_long_s), compute_uv=False)
print("(m, c): 특이값 비 s_min/s_max < 1e-6:", sv[1] / sv[0] < 1e-6)
sigma_mps = 0.05                                   # 측정 잡음 표준편차
for name, t_s in [("0–1 s ", np.linspace(0.02, 1, 50)), ("0–10 s", t_long_s)]:
    S = sens(v_two, np.array([39.24, 4.0]), t_s)   # (50, 2) v_inf = m g / c, tau = m / c
    cov = sigma_mps**2 * np.linalg.inv(S.T @ S)    # 로그 파라미터의 근사 공분산
    sd = np.sqrt(np.diag(cov))
    print(f"관측 {name}: 상대오차 v_inf {sd[0]:.3f}, tau {sd[1]:.3f}, 상관 {cov[0, 1] / sd[0] / sd[1]:.3f}")
```

```output
(m, c): 특이값 비 s_min/s_max < 1e-6: True
관측 0–1 s : 상대오차 v_inf 0.052, tau 0.057, 상관 1.000
관측 0–10 s: 상대오차 v_inf 0.001, tau 0.002, 상관 0.937
```

$(m, c)$의 민감도 행렬은 두 열이 정확히 반대 방향이라 특이값 비가 사실상 0이다. $(v_\infty, \tau)$로 바꾸어도 0–1 s 측정에서는 두 파라미터의 상관이 1.000이고 상대오차가 5% 남짓이다. 같은 측정 잡음으로 0–10 s를 재면 상대오차가 0.1–0.2%로 줄고 상관도 0.937로 내려간다. 모델은 같고 **실험 설계**만 바뀌었다. Fisher 정보의 행렬식이나 최소 고윳값을 최대화하도록 측정 시각·조건을 고르는 것이 최적 실험 설계의 출발점이다(25.3장).

::quiz{id="structural-vs-practical"}

식별 불가능을 발견했을 때의 처방은 세 가지다. 식별 가능한 조합으로 **재매개화**하고(위의 $m/c$), 다른 측정으로 **정보를 더하거나**(질량을 저울로 따로 잰다), 사전분포로 **정보를 명시적으로 넣는다**(24.7장). 어느 경우에도 식별되지 않는 개별 파라미터 값을 해석하지 않는다. 민감도 행렬은 국소 선형화에 기댄 판단이므로, 비선형성이 강하면 프로파일 우도(profile likelihood)로 한 파라미터씩 고정해 가며 우도가 얼마나 평평한지를 확인한다(Raue 외, 2009).

**희소 회귀로 방정식 식별.** 지금까지는 식의 구조를 알고 계수만 정했다. 구조를 모를 때는 후보 항을 넉넉히 늘어놓고 데이터가 필요한 항만 고르게 할 수 있다. 상태 $\mathbf{x}(t) \in \mathbb{R}^d$의 측정을 행으로 쌓은 $\mathbf{X} \in \mathbb{R}^{T \times d}$, 시간미분 추정 $\dot{\mathbf{X}}$, 후보 함수(상수, 1차·2차 단항식, 삼각함수 등)를 열로 계산한 라이브러리 $\mathbf{\Theta}(\mathbf{X}) \in \mathbb{R}^{T \times p}$로

$$
\dot{\mathbf{X}} \approx \mathbf{\Theta}(\mathbf{X})\,\mathbf{\Xi}, \qquad \mathbf{\Xi} \in \mathbb{R}^{p \times d}\ \text{는 희소}
$$

를 푼다. 물리 법칙은 대개 항 몇 개로 쓰인다는 가정을 희소성으로 넣은 것이다. Brunton, Proctor, Kutz(2016)가 SINDy(sparse identification of nonlinear dynamics)라는 이름으로 제안했다. 희소 해를 찾는 단순하고 널리 쓰이는 방법이 **순차 임계 최소제곱**(STLSQ, sequentially thresholded least squares)이다. 최소제곱을 푼 뒤 절댓값이 임계값 $\lambda$보다 작은 계수를 0으로 만들고, 남은 항만으로 다시 최소제곱을 푸는 일을 반복한다. 16.2장의 Lasso가 $\ell_1$ 벌점으로 희소성을 유도하는 것과 같은 목적을 더 단순한 절차로 이룬다.

Lorenz 계 $\dot{x} = 10(y - x)$, $\dot{y} = x(28 - z) - y$, $\dot{z} = xy - \tfrac{8}{3}z$의 궤적 5000점을 RK4(9.6장)로 만들고, 중심차분으로 미분을 추정해 2차까지의 단항식 10개 중에서 항을 고른다.

```python {exec="browser" session="sysid"}
def lorenz(s):                                     # s: (3,) 상태 (x, y, z)
    x, y, z = s
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])

dt = 0.002
traj = np.empty((5000, 3)); traj[0] = [-8.0, 7.0, 27.0]
for i in range(4999):                              # 고전 4차 Runge–Kutta
    s = traj[i]; k1 = lorenz(s); k2 = lorenz(s + dt / 2 * k1); k3 = lorenz(s + dt / 2 * k2)
    traj[i + 1] = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + lorenz(s + dt * k3))

def library(X):                                    # (T, 3) → (T, 10): 2차까지의 단항식
    x, y, z = X.T
    return np.column_stack([np.ones(len(X)), x, y, z, x*x, x*y, x*z, y*y, y*z, z*z])
names = ["1", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]

def stlsq(Theta, dX, thresh, iters=10):           # 순차 임계 최소제곱(STLSQ)
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0] # (10, 3) 계수
    for _ in range(iters):
        small = np.abs(Xi) < thresh
        Xi[small] = 0.0
        for k in range(dX.shape[1]):               # 남은 항만으로 다시 최소제곱
            Xi[~small[:, k], k] = np.linalg.lstsq(Theta[:, ~small[:, k]], dX[:, k], rcond=None)[0]
    return Xi

Xi0 = stlsq(library(traj), np.gradient(traj, dt, axis=0), thresh=0.5)
for k, lhs in enumerate(["dx/dt", "dy/dt", "dz/dt"]):
    print(lhs, "=", " ".join(f"{Xi0[j, k]:+.3f} {names[j]}" for j in np.flatnonzero(Xi0[:, k])))
```

```output
dx/dt = -9.999 x +9.999 y
dy/dt = +27.993 x -0.999 y -1.000 xz
dz/dt = -2.666 z +1.000 xy
```

10 × 3 = 30개의 후보 계수 중에서 참 항 7개만 남았고, 계수도 소수 셋째 자리까지 거의 맞는다. 남은 오차(예: 27.993)는 중심차분의 절단 오차에서 온다.

**잡음이 문제다.** 측정에 잡음이 있으면 미분 추정이 잡음을 증폭한다. 간격 $\Delta t$의 중심차분은 잡음 표준편차 $\sigma$를 대략 $\sigma/(\sqrt{2}\,\Delta t)$로 키운다. 같은 궤적에 잡음을 더해 본다.

```python {exec="browser" session="sysid"}
rng = np.random.default_rng(0)
true_terms = Xi0 != 0                              # (10, 3) 잡음 없을 때 찾은 7개 항
for noise in [0.1, 0.5, 1.0]:
    X_noisy = traj + noise * rng.normal(size=traj.shape)
    Xi = stlsq(library(X_noisy), np.gradient(X_noisy, dt, axis=0), thresh=0.5)
    missed, spurious = np.sum(true_terms & (Xi == 0)), np.sum(~true_terms & (Xi != 0))
    print(f"잡음 sd {noise:.1f}: 놓친 항 {missed}, 가짜 항 {spurious}, dx/dt의 y 계수 {Xi[2, 0]:.2f} (참 10)")
```

```output
잡음 sd 0.1: 놓친 항 0, 가짜 항 0, dx/dt의 y 계수 9.98 (참 10)
잡음 sd 0.5: 놓친 항 1, 가짜 항 0, dx/dt의 y 계수 9.81 (참 10)
잡음 sd 1.0: 놓친 항 1, 가짜 항 0, dx/dt의 y 계수 9.23 (참 10)
```

잡음 sd 0.1(상태 크기의 1% 미만)에서는 구조가 그대로 복원되지만, 0.5부터는 $\dot{y}$ 식의 $-y$ 항을 놓친다. 놓친 항의 몫은 $x$의 계수가 대신 흡수한다(28에서 약 25로). 잡음이 라이브러리 $\mathbf{\Theta}(\mathbf{X})$ 자체에도 들어가 계수가 0 쪽으로 줄어드는 효과(errors-in-variables)도 함께 작동한다. 실제 연구에서는 미분 전에 잡음을 줄이는 평활, 미분을 피하는 적분·약형식(weak form), 임계값과 라이브러리의 체계적 선택이 필요하다. 이 주제는 25.2장에서 방정식 발견의 관점으로 다시 다룬다.

:::note[분야 예시]
- **생화학 반응망**: 효소 반응 모델의 속도 상수 수십 개를 일부 종의 농도 시계열만으로 추정하는 일이 흔하다. 관측하지 않은 종의 농도에만 들어가는 상수는 대개 조합으로만 식별되고, 프로파일 우도가 평평하게 나온다. 이때 보고할 것은 개별 상수가 아니라 식별 가능한 조합과, 그것을 정하기 위해 추가로 측정할 종이다.
- **구조 동역학**: 진동 실험에서 질량·감쇠·강성 행렬을 식별할 때, 가진 주파수가 고유진동수 근처를 지나지 않으면 감쇠가 거의 보이지 않는다. 같은 모델이라도 가진 신호의 설계가 실용적 식별 가능성을 정한다.
:::

## 관측 가능성

**왜 필요한가.** 상태 추정에서도 같은 질문이 있다. 측정하는 것이 상태의 일부뿐일 때, 시간에 따른 측정을 모으면 나머지 상태를 알 수 있는가?

**정의.** 선형 시스템 $\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k$, $\mathbf{y}_k = \mathbf{H}\mathbf{x}_k$에서 $\mathbf{y}_0, \dots, \mathbf{y}_{n-1}$을 쌓으면 $\mathbf{x}_0$에 대한 선형 방정식이 된다.

$$
\begin{bmatrix} \mathbf{y}_0 \\ \mathbf{y}_1 \\ \vdots \\ \mathbf{y}_{n-1} \end{bmatrix}
= \underbrace{\begin{bmatrix} \mathbf{H} \\ \mathbf{H}\mathbf{F} \\ \vdots \\ \mathbf{H}\mathbf{F}^{n-1} \end{bmatrix}}_{\mathcal{O}} \mathbf{x}_0.
$$

$n$은 상태의 차원이다. 관측 가능성 행렬 $\mathcal{O}$의 rank가 $n$이면 측정으로 초기 상태를 유일하게 정할 수 있고, 시스템은 **관측 가능**(observable)하다. rank가 $n$보다 작으면 $\mathcal{O}$의 영공간 방향(7.3장)의 상태 성분은 측정에 전혀 나타나지 않는다. ($n$개보다 많은 측정을 쌓아도 Cayley–Hamilton 정리 때문에 rank는 늘지 않는다.)

**직관.** 등속 운동 $\mathbf{x} = (\text{위치}, \text{속도})$, $\mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$에서 위치를 재면($\mathbf{H} = [1\ 0]$) $\mathcal{O} = \begin{bmatrix} 1 & 0 \\ 1 & \Delta t \end{bmatrix}$로 rank 2다. 위치의 변화에서 속도가 보인다. 속도만 재면($\mathbf{H} = [0\ 1]$) $\mathcal{O} = \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix}$로 rank 1이다. 속도를 아무리 오래 재도 출발 위치는 알 수 없다. 측정된 속도를 적분하면 위치의 **변화**만 알 수 있다.

::quiz{id="observability-rank"}

비선형 시스템에서는 관측 가능성이 상태에 따라 달라진다. 이 장의 진자 예처럼 추의 수평 위치 $L\sin\theta$를 재면, 한 순간의 측정으로는 $\theta$와 $\pi - \theta$를 구별할 수 없다. 두 각도에서의 진동 주기와 속도 변화가 다르므로 시간에 따른 측정을 모으면 구별되지만, 그 정보가 쌓이기까지 필터는 틀린 쪽에 머물 수 있다.

**연구에서의 의미.** 필터의 공분산이 줄지 않거나 추정이 표류하면 알고리즘보다 관측 가능성을 먼저 의심한다. 배터리에서 충전 상태(SOC)를 단자 전압으로 추정하는 것은 개방회로 전압 곡선의 기울기 덕분인데, 기울기가 거의 0인 평탄 구간에서는 전압이 SOC에 대해 정보를 거의 주지 않는다. 이런 구간에서는 필터 설계보다 측정(전류 적산의 정확도, 다른 센서)이 성능을 정한다.

## Kalman filter 계열

**칼만 필터.** 16.7장의 스칼라 칼만 필터를 상태 벡터로 일반화한다. 선형 가우시안 상태공간 모델 $\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{w}_k$, $\mathbf{y}_k = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k$($\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$, $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$)에서 사후분포 $\mathcal{N}(\mathbf{m}_k, \mathbf{P}_k)$를 두 단계로 갱신한다(Kalman, 1960).

$$
\text{예측:}\quad \mathbf{m}^-_k = \mathbf{F}\mathbf{m}_{k-1}, \qquad \mathbf{P}^-_k = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^\top + \mathbf{Q}
$$

$$
\text{갱신:}\quad \mathbf{S}_k = \mathbf{H}\mathbf{P}^-_k\mathbf{H}^\top + \mathbf{R}, \quad \mathbf{K}_k = \mathbf{P}^-_k\mathbf{H}^\top\mathbf{S}_k^{-1}, \quad \mathbf{m}_k = \mathbf{m}^-_k + \mathbf{K}_k(\mathbf{y}_k - \mathbf{H}\mathbf{m}^-_k), \quad \mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}^-_k
$$

$\mathbf{y}_k - \mathbf{H}\mathbf{m}^-_k$를 **혁신**(innovation), $\mathbf{S}_k$를 그 공분산, $\mathbf{K}_k$를 **칼만 이득**이라 한다. 갱신 단계는 11.4장의 가우시안 조건부분포 공식을 $(\mathbf{x}_k, \mathbf{y}_k)$의 결합분포에 적용한 것과 같다. 이득은 예측의 불확실성 $\mathbf{P}^-_k$와 측정 잡음 $\mathbf{R}$의 상대적 크기로 "측정을 얼마나 믿을지"를 정한다.

::quiz{id="kalman-gain-limits"}

**확장 칼만 필터(EKF).** 동역학 $\mathbf{x}_k = f(\mathbf{x}_{k-1}) + \mathbf{w}_k$나 관측 $\mathbf{y}_k = h(\mathbf{x}_k) + \mathbf{v}_k$가 비선형이면, 현재 추정값에서 Jacobian $\mathbf{F}_k = \partial f/\partial \mathbf{x}$, $\mathbf{H}_k = \partial h/\partial \mathbf{x}$로 선형화해 같은 식을 쓴다. 평균은 비선형 함수 $f$, $h$로 그대로 밀고, 공분산만 선형화로 전파한다. 선형화가 잘 맞는 범위(추정 불확실성이 작고 비선형성이 약한 범위)에서는 정확하고 빠르지만, 불확실성이 커서 선형화 지점이 틀리면 공분산이 실제 오차보다 작게 계산되어 **과신**한다.

감쇠 진자 $\ddot\theta = -(g/L)\sin\theta - b\dot\theta$의 각도를, 추의 수평 위치 $L\sin\theta$(잡음 sd 0.1 m)만 재서 추정한다. 진폭이 2 rad이라 $\sin\theta$가 단조가 아닌 범위($|\theta| > \pi/2$)까지 흔들린다. 초기 추정이 맞는 경우($\theta = 2.0$)와, 측정으로는 구별되지 않는 거울상 쪽으로 틀린 경우($\theta = 1.0$, $\sin 1.0 \approx 0.84$, $\sin 2.0 \approx 0.91$)를 비교한다. 초기 공분산은 두 경우 모두 단위행렬이다.

```python {exec="browser" session="sysid"}
G_L, B, DT_S, L_M = 9.81, 0.1, 0.02, 1.0          # g/L (1/s^2), 감쇠 (1/s), 시간 간격 (s), 진자 길이 (m)
def step(s):                                       # s: (..., 2) = (각도 rad, 각속도 rad/s), 반암시적 Euler
    w = s[..., 1] + DT_S * (-G_L * np.sin(s[..., 0]) - B * s[..., 1])
    return np.stack([s[..., 0] + DT_S * w, w], axis=-1)
def jac_f(s):                                      # step의 Jacobian (2, 2)
    a = -G_L * np.cos(s[0]) * DT_S
    return np.array([[1 + DT_S * a, DT_S * (1 - B * DT_S)], [a, 1 - B * DT_S]])
h = lambda s: L_M * np.sin(s[..., 0])              # 관측: 추의 수평 위치 (m)

rng = np.random.default_rng(3)
q_sd, r_sd = np.array([1e-3, 1e-2]), 0.1           # 과정 잡음 sd (rad, rad/s), 관측 잡음 sd (m)
x_true = np.empty((400, 2)); x_true[0] = [2.0, 0.0] # 진폭 2 rad: sin θ가 단조가 아닌 범위까지 흔들린다
for t in range(399):
    x_true[t + 1] = step(x_true[t]) + q_sd * rng.normal(size=2)
y_obs = h(x_true) + r_sd * rng.normal(size=400)    # (400,)

def ekf(m, P, Q=np.diag(q_sd**2), R=r_sd**2):
    out_m, out_sd = [], []
    for y in y_obs:
        F = jac_f(m); m = step(m); P = F @ P @ F.T + Q                 # 예측
        H = np.array([L_M * np.cos(m[0]), 0.0])                        # 관측 Jacobian (2,)
        K = P @ H / (H @ P @ H + R)                                    # 칼만 이득 (2,)
        m = m + K * (y - h(m)); P = P - np.outer(K, H @ P)             # 갱신
        out_m.append(m[0]); out_sd.append(np.sqrt(P[0, 0]))
    return np.array(out_m), np.array(out_sd)

def report(name, mean, sd):
    err = mean - x_true[:, 0]
    rmse = lambda e: np.sqrt(np.mean(e**2))
    print(f"{name}: RMSE 처음 2 s {rmse(err[:100]):.3f}, 나머지 {rmse(err[100:]):.3f} rad | "
          f"95% 구간 포함률 처음 2 s {np.mean(np.abs(err[:100]) < 1.96 * sd[:100]):.2f}")
report("EKF 초기 추정 θ=2.0", *ekf(np.array([2.0, 0.0]), np.eye(2)))
report("EKF 초기 추정 θ=1.0", *ekf(np.array([1.0, 0.0]), np.eye(2)))
```

```output
EKF 초기 추정 θ=2.0: RMSE 처음 2 s 0.089, 나머지 0.024 rad | 95% 구간 포함률 처음 2 s 0.97
EKF 초기 추정 θ=1.0: RMSE 처음 2 s 0.410, 나머지 0.055 rad | 95% 구간 포함률 처음 2 s 0.07
```

초기 추정이 맞으면 EKF는 정확하고(나머지 구간 0.024 rad) 구간도 잘 맞는다. 거울상 쪽에서 출발하면 처음 2 s 동안 오차가 0.410 rad인데 95% 구간 포함률은 0.07이다. 틀린 쪽에서 선형화한 공분산을 믿고 **틀린 답을 확신**한 것이다. 시간이 지나 동역학이 두 가설을 구별해 주면 참 궤적으로 돌아온다.

**UKF.** unscented Kalman filter는 Jacobian 대신 평균 주위에 대칭으로 놓은 소수의 점(sigma point, $d$차원 상태에서 보통 $2d + 1$개)을 비선형 함수에 통과시키고, 결과 점들의 가중 평균과 공분산으로 분포를 근사한다(Julier, Uhlmann, 2004). Jacobian을 구할 필요가 없고 2차 효과를 일부 반영하지만, 여전히 사후분포를 가우시안 하나로 표현하므로 위와 같은 두 갈래(다봉) 상황은 다루지 못한다.

**잡음 공분산 튜닝.** $\mathbf{Q}$와 $\mathbf{R}$은 필터의 성능을 크게 좌우하지만 대개 정확히 모른다. $\mathbf{R}$은 센서 사양이나 정지 상태 측정으로 정할 수 있고, $\mathbf{Q}$는 모델이 놓친 모든 것을 대신하는 손잡이다. $\mathbf{Q}$가 너무 작으면 필터가 모델을 과신해 측정을 무시하다가 발산하고, 너무 크면 측정 잡음을 그대로 따라간다. 필터가 일관된지는 혁신으로 검사한다. 모델이 맞으면 정규화 혁신 제곱 $\epsilon_k = \mathbf{e}_k^\top\mathbf{S}_k^{-1}\mathbf{e}_k$($\mathbf{e}_k$는 혁신)는 자유도가 측정 차원인 $\chi^2$ 분포를 따르므로, 그 평균이 측정 차원과 크게 다르면 공분산 설정이 틀린 것이다. 혁신의 로그우도 $\sum_k \log\mathcal{N}(\mathbf{e}_k; \mathbf{0}, \mathbf{S}_k)$를 최대화해 $\mathbf{Q}$, $\mathbf{R}$을 추정할 수도 있다.

::widget{name="kalman-1d"}

## particle filter

**왜 필요한가.** 사후분포가 여러 봉우리를 가지거나 심하게 비대칭이면 가우시안 하나로는 표현할 수 없다. **particle filter**는 사후분포를 가중치가 있는 표본(입자) $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^n$으로 표현한다. Gordon, Salmond, Smith(1993)의 bootstrap filter가 가장 기본적인 형태다.

**정의.** 매 시점에 세 단계를 반복한다.

1. **예측**: 각 입자를 동역학 $\mathbf{x}^{(i)}_k \sim p(\mathbf{x}_k \mid \mathbf{x}^{(i)}_{k-1})$로 민다(잡음 포함).
2. **가중**: 새 측정의 우도로 가중치를 매긴다. $w^{(i)} \propto p(\mathbf{y}_k \mid \mathbf{x}^{(i)}_k)$, 합이 1이 되게 정규화.
3. **재표집**: 가중치에 비례해 입자를 복원 추출한다. 가중치가 작은 입자는 사라지고 큰 입자는 복제된다.

선형화도, 가우시안 가정도 없다. 비용은 입자 수 $n$에 비례한다(13.5장의 MCMC처럼 표본으로 분포를 표현하지만, 시간에 따라 순차적으로 갱신한다는 점이 다르다).

```python {exec="browser" session="sysid"}
def particle_filter(m0, sd0, n=2000, seed=0):
    r = np.random.default_rng(seed)
    p = m0 + sd0 * r.normal(size=(n, 2))           # (n, 2) 입자
    out_m, out_sd = [], []
    for y in y_obs:
        p = step(p) + q_sd * r.normal(size=(n, 2))                     # 예측: 입자를 동역학으로 민다
        logw = -0.5 * (y - h(p))**2 / r_sd**2                          # 가중치 ∝ 우도
        w = np.exp(logw - logw.max()); w /= w.sum()
        mu = w @ p[:, 0]
        out_m.append(mu); out_sd.append(np.sqrt(w @ (p[:, 0] - mu)**2))
        p = p[r.choice(n, size=n, p=w)]                                # 재표집
    return np.array(out_m), np.array(out_sd)
report("PF  초기 추정 θ=2.0", *particle_filter(np.array([2.0, 0.0]), 1.0))
report("PF  초기 추정 θ=1.0", *particle_filter(np.array([1.0, 0.0]), 1.0))
```

```output
PF  초기 추정 θ=2.0: RMSE 처음 2 s 0.143, 나머지 0.028 rad | 95% 구간 포함률 처음 2 s 0.94
PF  초기 추정 θ=1.0: RMSE 처음 2 s 0.168, 나머지 0.035 rad | 95% 구간 포함률 처음 2 s 0.87
```

거울상 쪽에서 출발해도 처음 2 s의 오차가 0.168 rad로 EKF(0.410)보다 작고, 포함률이 0.87로 훨씬 정직하다. 초기 분포가 넓어($\mathrm{sd} = 1$) 참 각도 근처에도 입자가 있었고, 동역학이 두 가설을 구별해 주는 동안 두 무리의 입자가 함께 살아남았기 때문이다. 초기 추정이 맞는 경우에는 EKF가 더 정확하다. 사후분포가 거의 가우시안일 때 입자 2000개는 EKF의 해석적 갱신보다 표본 잡음만큼 손해를 본다.

**가중치 퇴화.** 측정이 정확하거나 상태 차원이 높으면 우도가 좁아져 입자 대부분의 가중치가 0에 가까워진다. 유효 표본 수(effective sample size)

$$
n_{\text{eff}} = \frac{1}{\sum_i (w^{(i)})^2}
$$

가 이 정도를 잰다. 모든 가중치가 같으면 $n_{\text{eff}} = n$, 한 입자가 모든 가중치를 가지면 1이다. 실무에서는 $n_{\text{eff}}$가 $n/2$ 아래로 떨어질 때만 재표집하기도 한다. 상태 차원이 커지면 필요한 입자 수가 매우 빠르게 늘어나므로, 수십 차원 이상에서는 bootstrap filter를 그대로 쓰기 어렵다. 날씨 예보처럼 상태가 수백만 차원인 자료 동화(data assimilation)에서는 앙상블 칼만 필터나 변분 방법을 쓴다.

## 학습과 결합

필터는 모델 $f$, $h$와 잡음 공분산을 입력으로 받는다. 이것들을 데이터로 학습하면 시스템 식별과 상태 추정이 하나의 문제가 된다.

- **학습된 동역학 + 필터.** 물리 모델을 모르거나 일부만 알 때, $f$를 신경망 $f_\phi$나 하이브리드 모델(24.5장)로 바꾸고 EKF에 쓴다. EKF에 필요한 Jacobian $\partial f_\phi/\partial \mathbf{x}$는 자동미분(8.4장)으로 구한다. 학습된 동역학이 학습 데이터 분포 밖의 상태에서 엉뚱하게 행동하면 필터도 함께 틀린다.
- **미분가능 필터(differentiable filter).** 칼만 필터의 예측·갱신은 미분 가능한 연산이므로, 혁신의 로그우도 $\sum_k \log \mathcal{N}(\mathbf{e}_k; \mathbf{0}, \mathbf{S}_k)$를 손실로 두고 $f_\phi$, $\mathbf{Q}$, $\mathbf{R}$을 역전파로 함께 학습할 수 있다. 숨은 상태의 정답 없이 측정만으로 학습한다는 점이 장점이다. particle filter는 재표집이 이산적인 선택이라 그대로는 미분할 수 없어 별도의 근사가 필요하다.
- **식별 문제는 사라지지 않는다.** $f_\phi$와 $\mathbf{Q}$를 함께 학습하면, 모델이 못 맞춘 부분을 큰 $\mathbf{Q}$가 흡수할 수 있다. 24.5장의 $\boldsymbol{\theta}$–$\delta$ 혼동과 같은 구조다. 학습된 잡음 공분산이 센서 사양이나 물리적 예상과 맞는지 확인한다.

## 실습

잡음이 있는 측정으로 EKF와 particle filter를 구현해 상태 추정 정확도와 불확실성을 비교한다. 코드는 `labs/scientific-ml/system-identification-and-estimation/filter_compare.py`다. 본문 cell 4–5의 진자 문제를 참 궤적 10개로 반복하고, 입자 수 200과 2000을 비교한다. 집필 시점의 실행 결과는 다음과 같다.

```output
방법            초기θ | RMSE 처음2s 나머지 | 포함률 처음2s 나머지   (참 궤적 10개 평균)
EKF            2.0 |      0.076  0.029 |       0.95  0.95
PF n=200       2.0 |      0.157  0.103 |       0.54  0.41
PF n=2000      2.0 |      0.122  0.031 |       0.87  0.91
EKF            1.0 |      0.422  0.071 |       0.13  0.77
PF n=200       1.0 |      0.206  0.117 |       0.41  0.42
PF n=2000      1.0 |      0.171  0.032 |       0.93  0.91
```

- **초기 추정이 맞으면 EKF가 가장 좋다.** 오차도 작고 95% 구간 포함률도 0.95로 정확하다.
- **초기 추정이 거울상 쪽이면 EKF는 처음 2 s 동안 틀린 답을 확신한다**(포함률 0.13). 나머지 구간 평균 포함률도 0.77로, 일부 궤적에서는 오래 회복하지 못했다.
- **입자가 너무 적으면 particle filter가 무너진다.** 입자 200개에서는 가중치 퇴화 때문에 포함률이 0.4대로 떨어진다. 2000개에서는 두 초기 조건 모두 포함률 0.9 안팎을 유지한다. particle filter의 장점은 입자 수가 충분할 때만 나타난다.

과제는 이 비교를 확장하는 것이다. 관측 잡음 sd를 0.02와 0.3으로 바꾸고, particle filter에 $n_{\text{eff}} < n/2$일 때만 재표집하는 규칙을 넣어 정확도·포함률·$n_{\text{eff}}$의 시간 변화를 보고한다. 어느 조건에서 EKF로 충분하고 어느 조건에서 particle filter가 필요한지 한 문단으로 정리한다.

이어서 두 연습문제를 푼다. STLSQ로 감쇠 진동자의 방정식을 복원하고, 칼만 필터의 갱신 단계를 행렬로 구현한다.

::exercise{id="stlsq"}

::exercise{id="kalman-update"}

## 실패 사례

- **식별 불가능한 파라미터를 추정하고 값을 해석하는 것.** 낙하 예의 $m$과 $c$처럼 조합으로만 출력에 나타나는 파라미터는 최적화가 어떤 값을 내놓든 그 값에 물리적 의미가 없다. 추정 전에 민감도 행렬의 특이값이나 프로파일 우도로 식별 가능한 파라미터 집합을 정하고, 나머지는 고정하거나 재매개화한다.
- **짧은 실험으로 실용적 식별을 확인하지 않는 것.** 구조적으로 식별 가능해도 측정 구간이 짧으면 파라미터가 강하게 상관된다(cell 1의 상관 1.000). 추정값과 함께 공분산(또는 상관)을 보고한다.
- **잡음 섞인 데이터의 수치 미분으로 희소 회귀를 하고 결과 식을 그대로 믿는 것.** 잡음이 커지면 참 항이 빠지고 다른 항의 계수가 그 몫을 흡수한다(cell 3). 잡음 수준을 바꿔 가며 식이 안정적으로 유지되는지 확인한다.
- **EKF의 공분산을 불확실성으로 그대로 보고하는 것.** 선형화가 틀린 구간에서 EKF 공분산은 실제 오차보다 훨씬 작다(cell 4의 포함률 0.07). 모의 데이터에서 포함률을 확인하고, 실제 데이터에서는 혁신의 일관성을 검사한다.

## 통과 기준

- 모델 하나의 파라미터에 대해 민감도 행렬의 특이값(또는 프로파일 우도)으로 식별 가능한 파라미터 집합을 정하고, 그 집합만 추정해 공분산과 함께 보고한다.
- 같은 측정으로 EKF와 particle filter를 실행해 RMSE와 95% 구간 포함률을 비교하고, 어느 필터를 고를지 근거를 쓴다.
- 연습문제 `stlsq`, `kalman-update`의 테스트를 통과한다.

## 참고 문헌

- Brunton, Proctor, Kutz (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS*, 113(15), 3932–3937. — 후보 함수 라이브러리와 희소 회귀로 동역학을 찾는 SINDy
- Gordon, Salmond, Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107–113. — bootstrap filter(particle filter의 기본형)
- Julier, Uhlmann (2004). Unscented filtering and nonlinear estimation. *Proceedings of the IEEE*, 92(3), 401–422. — unscented 변환과 UKF
- Kalman (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. — 칼만 필터
- Raue 외 (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics*, 25(15), 1923–1929. — 프로파일 우도로 보는 구조적·실용적 식별 가능성
