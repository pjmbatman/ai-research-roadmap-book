---
id: scientific-ml/neural-operators
title: "Neural Operator와 동역학 학습"
title_en: "Neural Operators and Learning Dynamics"
tier: advanced
status: draft
prerequisites: [scientific-ml/physics-and-learning, linear-algebra/function-spaces]
---

# Neural Operator와 동역학 학습

24.2장의 PINN과 24.3장의 neural ODE는 주어진 조건 하나에 대한 해를 다룬다. 초기 조건이나 계수 분포가 바뀌면 다시 학습하거나 다시 적분해야 한다. 설계 탐색, 불확실성 전파, 기상 예보처럼 수많은 조건에서 해를 빠르게 내야 하는 문제에서는 **조건에서 해로 가는 사상 자체**를 배우고 싶다. 그런데 조건과 해는 모두 함수다. 초기 온도 분포, 투수율 장, 바람장은 공간의 함수이고, 컴퓨터에서는 어떤 격자 위의 값으로만 보인다.

neural operator는 이 사상을 함수에서 함수로 가는 연산자로 학습한다. 핵심 요구는 학습된 모델이 특정 격자가 아니라 연속 대상에 묶이는 것, 즉 다른 해상도·다른 센서 배치로 샘플링한 같은 함수에 같은 답을 주는 것이다. 이 장은 7.9장의 함수공간과 연산자 언어 위에서 문제를 정의하고, DeepONet과 FNO의 핵심 구조를 NumPy로 구현해 해상도 불변성이 무엇을 보장하고 무엇을 보장하지 않는지 확인한다. 이어서 PyTorch FNO로 비선형 반응–확산 방정식의 한 step 연산자를 배워 해상도 변화와 장기 rollout에서 평가하고, rollout의 폭주를 물리 제약으로 막는 방법을 본다.

## 학습 목표

- 함수→함수 매핑으로 문제를 정의한다
- 주요 neural operator를 구현한다
- 해상도·형상 일반화와 장기 rollout을 평가한다

## operator learning

**정의.** 입력 함수 $a \in \mathcal{A}$(초기 조건, 계수 장, 경계값, 강제항)를 출력 함수 $u \in \mathcal{U}$(나중 시각의 해, 정상 상태 해)로 보내는 연산자

$$
\mathcal{G}: \mathcal{A} \to \mathcal{U}, \qquad u = \mathcal{G}(a)
$$

를 입력–출력 함수 쌍 $\{(a_j, u_j)\}_{j=1}^{n}$으로 근사하는 문제를 **연산자 학습**(operator learning)이라 한다. $\mathcal{A}$, $\mathcal{U}$는 함수공간(7.9장)이다. 예를 들어 1차원 열방정식 $u_t = u_{xx}$에서 $\mathcal{G}$는 초기 온도 $a(x) = u(x, 0)$를 시각 $T$의 온도 $u(x, T)$로 보낸다. 24.1장의 분류로는 대리모델 문제이고, 입력과 출력이 함수라는 점이 다르다.

**이산화와의 관계.** 데이터는 언제나 격자 위의 값 $a(x_1), \dots, a(x_N)$으로 주어진다. 그래서 $\mathbb{R}^N \to \mathbb{R}^N$ 사상을 MLP로 배우는 것도 가능해 보인다. 그러나 그 사상은 격자 $N$에 묶인다. 해상도가 다른 시뮬레이션, 센서 배치가 다른 장비, 적응적 메시로 만든 데이터에는 쓸 수 없고, 격자를 바꿀 때마다 다시 학습해야 한다. neural operator가 요구하는 성질은 **이산화 불변성**(discretization invariance)이다. 파라미터가 연속 대상에 정의되어 있어, 입력을 어떤 격자로 샘플링하든 같은 연산자를 근사해야 한다(Kovachki 외, 2023). 7.9장의 표현을 빌리면, 학습되는 것은 행렬이 아니라 연속 연산자이고 행렬은 그 그림자다.

이 장의 NumPy 셀은 두 정답 연산자를 쓴다. 주기 영역 $[0, 1)$에서 입력 함수를 Fourier 계수로 정의하면 어떤 격자에서도 정확히 샘플링할 수 있고, 두 연산자의 정확한 출력도 계수로 계산된다.

- **열방정식** $u_t = u_{xx}$, $T = 0.002$: 파수 $k$ 성분에 $e^{-(2\pi k)^2 T}$를 곱한다. 고주파일수록 빨리 사라진다.
- **이류** $u_t + u_x = 0$, 이동 거리 0.3: 파수 $k$ 성분에 $e^{-2\pi i k \cdot 0.3}$를 곱한다. 모든 성분의 크기를 보존하고 위상만 바꾼다.

```python {exec="browser" session="operator"}
import numpy as np

rng = np.random.default_rng(0)
K_IN = 12                                               # 입력 함수가 가진 최고 파수

def sample_functions(n, K=K_IN, decay=1.5):            # 함수 n개를 Fourier 계수로 정의, (n, K) 복소수
    k = np.arange(1, K + 1)
    return (rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))) / k**decay

def on_grid(coef, N):                                   # 계수 → 격자 [0, 1)의 N점 값, (n, N)
    k = np.arange(1, coef.shape[1] + 1)
    x = np.arange(N) / N
    return 2 * np.real(coef @ np.exp(2j * np.pi * np.outer(k, x)))

def heat(coef, T=0.002):                                # u_t = u_xx를 T까지: 모드마다 exp(-(2πk)² T)
    k = np.arange(1, coef.shape[1] + 1)
    return coef * np.exp(-(2 * np.pi * k) ** 2 * T)

def advect(coef, shift=0.3):                            # u_t + u_x = 0을 0.3까지: 모드마다 위상 회전
    k = np.arange(1, coef.shape[1] + 1)
    return coef * np.exp(-2j * np.pi * k * shift)

def rel_err(pred, true):                                # 상대 L2 오차
    return np.linalg.norm(pred - true) / np.linalg.norm(true)

fmt = lambda e: "< 1e-12" if e < 1e-12 else f"{e:.1e}"  # 반올림 수준의 값은 판정으로
a_train, a_test = sample_functions(200), sample_functions(50)   # 학습 입력 200개, 시험 입력 50개
print("격자 64점의 입력 배열:", on_grid(a_train, 64).shape, " 격자 256점:", on_grid(a_train, 256).shape)
print("같은 함수, x = 0.25의 값 (64점, 256점):", on_grid(a_train[:1], 64)[0, 16].round(6), on_grid(a_train[:1], 256)[0, 64].round(6))
```

```output
격자 64점의 입력 배열: (200, 64)  격자 256점: (200, 256)
같은 함수, x = 0.25의 값 (64점, 256점): -2.953492 -2.953492
```

같은 함수를 64점과 256점에서 샘플링하면 배열의 크기는 다르지만, 겹치는 점 $x = 0.25$의 값은 같다. 데이터는 격자 값이지만 대상은 함수다.

::quiz{id="operator-problem"}

## DeepONet

**branch와 trunk.** 연산자의 보편 근사 정리(Chen & Chen, 1995)는 연속 연산자를 다음 꼴로 근사할 수 있음을 보였다. DeepONet(Lu 외, 2021)은 이 꼴을 두 신경망으로 구현한다.

$$
\mathcal{G}(a)(y) \approx \sum_{k=1}^{p} b_k\big(a(x_1), \dots, a(x_m)\big)\; t_k(y).
$$

- **branch 망** $\mathbf{b}$는 입력 함수를 고정된 **센서** $x_1, \dots, x_m$에서 잰 값을 받아 계수 $p$개를 낸다.
- **trunk 망** $\mathbf{t}$는 출력 위치 $y$를 받아 basis 함수 값 $p$개를 낸다.

출력은 두 벡터의 내적이다. trunk가 출력 함수의 basis를 배우고, branch가 입력에 따라 그 basis의 계수를 정한다고 볼 수 있다. trunk는 $y$의 연속 함수이므로 **출력은 어떤 위치에서든** 평가할 수 있다. 반면 branch는 센서 $m$개의 값 벡터를 받으므로 **입력 이산화는 고정**된다.

구조만 보이기 위해 trunk를 고정된 Fourier basis($p = 25$)로, branch를 선형 사상(최소제곱)으로 둔 가장 단순한 DeepONet을 열방정식에 맞춘다. 실제 DeepONet은 둘 다 학습되는 비선형 신경망이다. 이 연산자는 선형이므로 이 단순한 형태로도 정확히 표현된다.

```python {exec="browser" session="operator"}
m, p = 32, 25                                           # 센서 수, basis 함수 수
sensors = np.arange(m) / m                              # 고정된 센서 위치, (32,)

def trunk(y):                                           # 출력 위치 y에서의 basis 값, (len(y), p)
    k = np.arange(1, (p - 1) // 2 + 1)
    return np.hstack([np.ones((len(y), 1)), np.cos(2 * np.pi * np.outer(y, k)), np.sin(2 * np.pi * np.outer(y, k))])

y_fit = np.arange(512) / 512
coef = np.linalg.lstsq(trunk(y_fit), on_grid(heat(a_train), 512).T, rcond=None)[0].T   # 출력 함수의 basis 계수, (200, p)
B = np.linalg.lstsq(on_grid(a_train, m), coef, rcond=None)[0]   # 선형 branch: 센서 값 → 계수, (32, p)

def deeponet(u_at_sensors, y):                          # G(u)(y) = Σ_k branch_k(u) · trunk_k(y), (n, len(y))
    return (u_at_sensors @ B) @ trunk(y).T

y_query = rng.uniform(0, 1, 1000)                       # 학습에 쓰지 않은 임의의 출력 위치
true = 2 * np.real(heat(a_test) @ np.exp(2j * np.pi * np.outer(np.arange(1, K_IN + 1), y_query)))
x50 = np.arange(50) / 50
u50 = on_grid(a_test, 50)                               # 다른 장비: 50점 격자로 측정한 입력
shifted = 2 * np.real(a_test @ np.exp(2j * np.pi * np.outer(np.arange(1, K_IN + 1), sensors + 0.5 / m)))
interp = np.array([np.interp(sensors, x50, u, period=1.0) for u in u50])
print(f"정해진 센서 값, 임의 위치 1000곳     상대 오차 {fmt(rel_err(deeponet(on_grid(a_test, m), y_query), true))}")
print(f"센서가 반 칸 어긋난 입력             상대 오차 {rel_err(deeponet(shifted, y_query), true):.1e}")
print(f"50점 입력을 센서 위치로 선형 보간    상대 오차 {rel_err(deeponet(interp, y_query), true):.1e}")
```

```output
정해진 센서 값, 임의 위치 1000곳     상대 오차 < 1e-12
센서가 반 칸 어긋난 입력             상대 오차 1.1e-01
50점 입력을 센서 위치로 선형 보간    상대 오차 2.4e-03
```

학습 센서의 값을 넣으면 학습에 쓰지 않은 임의의 위치 1000곳에서 반올림 수준으로 맞는다. 출력 쪽 이산화에는 자유롭다. 그러나 다른 장비가 같은 함수를 센서에서 반 칸 어긋난 위치에서 쟀다면, 값 32개를 그대로 넣었을 때 상대 오차가 0.11이다. 값의 개수가 같아도 위치가 다르면 다른 입력이다. 다른 격자의 측정은 먼저 센서 위치로 보간해야 하고(50점 입력의 선형 보간에서 오차 $2.4 \times 10^{-3}$), 보간 오차가 예측 오차에 더해진다.

::quiz{id="deeponet-sensors"}

## FNO

**Fourier 공간의 적분 커널.** neural operator의 한 층은 일반적으로

$$
v_{\ell+1}(x) = \sigma\Big( W v_\ell(x) + \int_{\Omega} \kappa_\phi(x, y)\, v_\ell(y)\, dy \Big)
$$

꼴이다. $v_\ell(x) \in \mathbb{R}^{C}$는 채널 $C$개짜리 은닉 함수, $W$는 점마다 적용하는 선형 사상, $\kappa_\phi$는 학습되는 적분 커널, $\sigma$는 활성화다. 적분이 함수 전체를 섞고, 커널이 연속 위치 $(x, y)$의 함수이므로 격자와 무관하게 정의된다. 문제는 적분의 비용이다. $N$점에서 계산하면 $O(N^2)$이다.

Fourier neural operator(FNO, Li 외, 2021)는 커널을 이동 불변 $\kappa(x, y) = \kappa(x - y)$로 제한한다. 그러면 적분은 합성곱이 되고, 합성곱 정리에 따라 Fourier 공간에서는 모드마다 곱하는 연산이 된다.

$$
\int \kappa(x - y)\, v(y)\, dy = \mathcal{F}^{-1}\big[\, R_\phi(k) \cdot \mathcal{F}[v]\,(k) \,\big]\,(x), \qquad |k| < K.
$$

$R_\phi(k)$는 파수 $k$마다 학습되는 $C \times C$ 복소 행렬이고, $K$개의 낮은 모드만 남긴다(모드 절단). FFT를 쓰면 비용이 $O(N \log N)$이다. 파라미터가 격자 칸이 아니라 **파수**에 붙어 있다는 점이 핵심이다. 7.9장에서 미분이 Fourier 계수에 $ik$를 곱하는 연산이라 어떤 $N$에도 그대로 쓰였던 것과 같은 원리다.

스펙트럼 합성곱 층을 NumPy로 구현하고, 같은 무작위 가중치를 32점부터 256점까지의 격자에 적용해 겹치는 점의 결과를 비교한다. 비교 대상은 보통의 CNN 층처럼 격자 칸 단위로 정의된 3점 합성곱이다.

```python {exec="browser" session="operator"}
def spectral_conv(u, w):                                # u: (n, N) 격자 값, w: (K,) 복소 가중치 → (n, N)
    N, K = u.shape[1], w.size
    u_hat = np.fft.rfft(u, axis=1) / N                  # 격자 크기와 무관한 Fourier 계수, (n, N//2+1)
    out_hat = np.zeros_like(u_hat)
    out_hat[:, 1:K + 1] = w * u_hat[:, 1:K + 1]         # 낮은 K개 모드에만 가중치, 나머지는 버림
    return np.fft.irfft(out_hat * N, n=N, axis=1)

def stencil_conv(u, kernel=(0.25, 0.5, 0.25)):         # 격자 칸 단위의 3점 합성곱 (보통의 CNN 층)
    return kernel[0] * np.roll(u, 1, axis=1) + kernel[1] * u + kernel[2] * np.roll(u, -1, axis=1)

w = rng.normal(size=16) + 1j * rng.normal(size=16)      # 무작위 스펙트럼 가중치, 모드 16개
f = a_train[:5]                                         # 함수 5개
ref_s, ref_c = spectral_conv(on_grid(f, 32), w), stencil_conv(on_grid(f, 32))
for N in [64, 128, 256]:
    step = N // 32                                      # 32점 격자와 겹치는 점만 비교
    d_s = np.abs(spectral_conv(on_grid(f, N), w)[:, ::step] - ref_s).max()
    d_c = np.abs(stencil_conv(on_grid(f, N))[:, ::step] - ref_c).max()
    print(f"N = {N:3d}  32점 결과와의 최대 차이: 스펙트럼 합성곱 {'< 1e-12' if d_s < 1e-12 else f'{d_s:.1e}'}"
          f"   3점 합성곱 {d_c:.2f}")
```

```output
N =  64  32점 결과와의 최대 차이: 스펙트럼 합성곱 < 1e-12   3점 합성곱 0.23
N = 128  32점 결과와의 최대 차이: 스펙트럼 합성곱 < 1e-12   3점 합성곱 0.31
N = 256  32점 결과와의 최대 차이: 스펙트럼 합성곱 < 1e-12   3점 합성곱 0.32
```

스펙트럼 합성곱은 모든 해상도에서 32점 결과와 반올림 수준으로 같다. Fourier 계수를 $N$으로 정규화하면 같은 함수의 계수가 격자와 무관해지기 때문이다. 3점 합성곱은 격자가 바뀌면 결과가 0.2–0.3 달라진다. 커널이 격자 칸에 묶여 있어, 격자가 촘촘해지면 물리적으로 더 좁은 영역을 평균하는 다른 연산자가 된다.

::quiz{id="fno-resolution"}

**데이터로 학습하기.** 은닉 채널이 하나인 스펙트럼 층은 모드마다 복소수 하나 $w_k$를 곱하는 선형 연산자이므로, 데이터에서 모드별 최소제곱으로 정확히 학습할 수 있다. 해상도 64의 데이터로 모드 12개의 가중치를 학습하고, 해상도 64와 256에서 두 종류의 입력으로 평가한다. 하나는 학습 분포와 같은 매끄러운 함수(파수 12까지), 다른 하나는 파수 30까지 고주파 성분이 큰 함수다.

```python {exec="browser" session="operator"}
def fit_spectral(u_in, u_out, K):                       # 모드마다 최소제곱: w_k = Σ out·conj(in) / Σ |in|²
    N = u_in.shape[1]
    a, b = np.fft.rfft(u_in, axis=1)[:, 1:K + 1] / N, np.fft.rfft(u_out, axis=1)[:, 1:K + 1] / N
    return (b * a.conj()).sum(0) / (np.abs(a) ** 2).sum(0)

a_rough = sample_functions(50, K=30, decay=1.0)         # 파수 30까지, 고주파가 더 큰 함수
for name, op in [("열방정식", heat), ("이류    ", advect)]:
    w = fit_spectral(on_grid(a_train, 64), on_grid(op(a_train), 64), K=K_IN)  # 해상도 64에서 모드 12개 학습
    msg = []
    for N in [64, 256]:
        smooth = rel_err(spectral_conv(on_grid(a_test, N), w), on_grid(op(a_test), N))
        rough = rel_err(spectral_conv(on_grid(a_rough, N), w), on_grid(op(a_rough), N))
        msg.append(f"N={N:3d}: 매끄러운 입력 {fmt(smooth)}, 고주파 입력 {rough:.1e}")
    print(f"{name} | " + " | ".join(msg))
```

```output
열방정식 | N= 64: 매끄러운 입력 < 1e-12, 고주파 입력 1.1e-07 | N=256: 매끄러운 입력 < 1e-12, 고주파 입력 1.1e-07
이류     | N= 64: 매끄러운 입력 < 1e-12, 고주파 입력 1.7e-01 | N=256: 매끄러운 입력 < 1e-12, 고주파 입력 1.7e-01
```

결과는 세 가지를 말한다.

- **해상도 불변성은 성립한다.** 해상도 64에서 학습한 가중치가 256에서도 같은 오차를 낸다.
- **매끄러운 입력에서는 두 연산자 모두 정확하다.** 정답 연산자가 모드마다 곱하는 꼴이라 이 층이 정확히 표현한다.
- **고주파 입력에서는 연산자에 따라 갈린다.** 학습된 층은 파수 13 이상의 성분을 버린다. 열방정식은 그 성분을 $e^{-(2\pi \cdot 13)^2 \cdot 0.002} \approx 2 \times 10^{-6}$ 이하로 스스로 지우므로 버려도 오차가 $10^{-7}$이다. 이류는 모든 성분을 보존하므로 버린 만큼이 그대로 오차(0.17)가 된다. 이 오차는 해상도를 올려도 줄지 않는다.

**해상도 불변성은 정확도를 보장하지 않는다.** 모드 절단 아래의 성분에 대해서만 같은 연산자를 정의할 뿐이다. 고주파를 보존하거나 만들어 내는 연산자(이류, 충격파를 만드는 Burgers 방정식, 난류)에서는 모드 수 $K$가 정확도의 상한을 정한다. 또 학습 데이터에 없는 파수(여기서는 13 이상)의 가중치는 데이터가 정하지 못한다. 그래서 모드 수를 입력이 가진 파수에 맞췄다.

::quiz{id="truncation-advection"}

실제 FNO는 이 층을 여러 개 쌓고 층마다 점별 선형 사상 $W$와 비선형 활성화를 더해 비선형 연산자를 표현한다. 입력을 채널 $C$개로 올리는 lifting 층과 출력으로 내리는 projection 층이 앞뒤에 붙는다. 점별 사상과 활성화는 고주파 성분을 만들어 낼 수 있으므로 신경망 전체는 모드 $K$ 이상의 출력도 낸다. 그러나 그 고주파 성분은 격자에 따라 달라질 수 있어, 해상도 불변성이 근사적으로만 성립한다.

## 다른 계열

FNO의 Fourier 변환은 균일한 주기 격자를 전제한다. 다른 설정을 위한 계열이 있다.

- **kernel integral 계열.** 적분 커널 $\kappa_\phi(x, y)$를 신경망으로 두고, 적분을 점들 사이의 가중합(근방 점만, 또는 저랭크 근사)으로 계산한다. 불규칙한 점 집합에 쓸 수 있지만 비용이 크다.
- **graph 기반 operator.** 메시의 노드를 그래프 노드로, 인접 관계를 간선으로 두고 message passing(18장)으로 국소 상호작용을 반복한다. 비정형 메시와 복잡한 형상에 자연스럽고, 유한요소·유한체적 시뮬레이터의 데이터와 잘 맞는다. 대칭을 반영한 설계는 24.8장에서 다룬다.
- **Transformer 기반 operator.** 점들을 token으로 보고 attention으로 전역 상호작용을 계산한다(18장). attention은 적분 커널을 데이터에 따라 정하는 것으로 해석할 수 있다. 점 수가 많으면 attention의 비용을 줄이는 근사가 필요하다.

어느 계열이든 같은 질문으로 평가한다. 입력·출력을 어떤 이산화로 받는가, 학습 해상도·형상 밖에서 무엇이 보장되는가, 계산 비용은 점 수에 따라 어떻게 늘어나는가.

## 일반화

neural operator 논문에서 "일반화"는 세 가지를 뜻할 수 있고, 각각 다른 평가가 필요하다.

- **해상도 일반화.** 학습 해상도보다 높거나 낮은 해상도의 입력에서 같은 정확도를 내는가. 셀 3–4가 보인 것처럼 구조적으로 보장되는 부분(모드 절단 아래)과 아닌 부분(절단 위 성분, 비선형 층이 만드는 고주파)을 구분한다. 평가할 때는 **해상도마다 그 해상도에서 수렴한 정답**과 비교해야 한다. 저해상도 정답을 보간해 고해상도 정답으로 쓰면 비교 자체가 오차를 가린다.
- **mesh 일반화.** 균일 격자로 학습한 모델을 비균일 메시, 적응적으로 세분한 메시에 쓸 수 있는가. FNO는 균일 격자를 전제하므로 보간을 거쳐야 하고, DeepONet의 branch는 센서 위치가 고정되어 있다(셀 2). graph·kernel 계열이 이 경우에 더 자연스럽다.
- **형상 일반화.** 학습에 없던 영역 모양(다른 날개 단면, 다른 전극 형상)에서도 맞는가. 형상이 바뀌면 경계 조건과 해의 구조가 바뀌므로, 입력 분포 밖으로의 외삽이다. 형상을 바꾼 별도 시험 세트로 평가하고, 학습 형상과의 거리(형상 파라미터 범위)를 함께 보고한다.

:::note[분야 예시]
기상 분야의 데이터 기반 예보 모델은 재분석 자료로 "현재 대기 상태 → 몇 시간 뒤 상태" 연산자를 학습한다. 구면 위의 데이터이고 격자가 극지방에서 촘촘해지므로, 평면 FFT 대신 구면 조화함수(spherical harmonics) 기반 연산이나 그래프 기반 모델이 쓰인다. 평가는 한 step 정확도보다 며칠에 걸친 rollout의 정확도와 물리적 일관성(에너지 스펙트럼이 지나치게 매끄러워지지 않는가)이 중심이다(27장).
:::

## 시간 동역학

**one-step operator와 trajectory operator.** 시간에 따라 변하는 문제를 연산자로 배우는 방법은 두 가지다.

- **one-step operator** $\mathcal{G}_{\Delta t}: u(\cdot, t) \mapsto u(\cdot, t + \Delta t)$를 배우고, 반복 적용(autoregressive rollout)으로 긴 시간을 예측한다. 학습 데이터를 많이 쓸 수 있고 시간 길이에 제한이 없지만, 오차가 step마다 쌓인다.
- **trajectory operator** $\mathcal{G}: u(\cdot, 0) \mapsto u(\cdot, t)$, $t \in [0, T]$를 한 번에 배운다(시간을 입력 좌표로 넣거나, 시공간 전체를 출력으로). 오차가 쌓이지 않지만 $T$ 이후는 예측할 수 없고, 출력이 커서 학습이 비싸다.

PyTorch로 FNO를 만들어 Allen–Cahn 방정식 $u_t = \nu u_{xx} + u - u^3$($\nu = 5 \times 10^{-4}$, 주기 영역)의 one-step operator($\Delta t = 0.1$)를 배운다. Allen–Cahn은 상분리(phase separation)를 묘사하는 반응–확산 방정식으로, 해가 $\pm 1$의 두 상으로 갈라지며 경계가 얇아진다. 학습 데이터는 해상도 64의 궤적 200개, 각 10 step(시간 0–1)이다. 평가는 학습하지 않은 초기 조건에서 해상도 64와 128로 한 step 오차와 40 step(시간 4) rollout 오차를 잰다. 128 해상도의 정답은 같은 solver를 128점에서 돌려 만든다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다(집필 환경 CPU 1스레드에서 약 20초).

```python {exec="python"}
import numpy as np
import torch

torch.manual_seed(0); torch.set_num_threads(1)
gen = np.random.default_rng(1)

def allen_cahn(u, n_sub=10, dt=0.01, nu=5e-4):          # u_t = ν u_xx + u - u³, 주기 [0, 1), 반암시적 스펙트럼 step
    k = 2 * np.pi * np.fft.rfftfreq(u.shape[-1], d=1.0 / u.shape[-1])
    for _ in range(n_sub):                              # 한 연산자 step = 시간 0.1
        u = np.fft.irfft((np.fft.rfft(u) + dt * np.fft.rfft(u - u**3)) / (1 + dt * nu * k**2), n=u.shape[-1])
    return u

def trajectories(n, N, steps):                          # 무작위 매끄러운 초기 조건에서 궤적, (n, steps+1, N)
    x, kk = np.arange(N) / N, np.arange(1, 6)
    c = gen.normal(size=(n, 5)) / kk; s = gen.normal(size=(n, 5)) / kk
    u = [0.5 * (c @ np.cos(2 * np.pi * np.outer(kk, x)) + s @ np.sin(2 * np.pi * np.outer(kk, x)))]
    for _ in range(steps):
        u.append(allen_cahn(u[-1]))
    return torch.tensor(np.stack(u, 1), dtype=torch.float32)

class SpectralConv(torch.nn.Module):                    # 낮은 modes개 Fourier 모드에 채널 혼합 가중치
    def __init__(self, ch, modes):
        super().__init__(); self.modes = modes
        self.w = torch.nn.Parameter(torch.randn(ch, ch, modes, dtype=torch.cfloat) / ch)
    def forward(self, v):                               # v: (B, C, N)
        v_hat = torch.fft.rfft(v, norm="forward")[..., :self.modes]          # (B, C, modes)
        out = torch.zeros(v.shape[0], v.shape[1], v.shape[-1] // 2 + 1, dtype=torch.cfloat)
        out[..., :self.modes] = torch.einsum("bik,iok->bok", v_hat, self.w)
        return torch.fft.irfft(out, n=v.shape[-1], norm="forward")

class FNO(torch.nn.Module):
    def __init__(self, ch=24, modes=12):
        super().__init__()
        self.lift, self.proj = torch.nn.Conv1d(1, ch, 1), torch.nn.Conv1d(ch, 1, 1)
        self.spec = torch.nn.ModuleList([SpectralConv(ch, modes) for _ in range(3)])
        self.pw = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 1) for _ in range(3)])
    def forward(self, u):                               # u: (B, N) → 다음 step u: (B, N)
        v = self.lift(u[:, None])
        for s, p in zip(self.spec, self.pw):
            v = torch.nn.functional.gelu(s(v) + p(v))
        return u + self.proj(v)[:, 0]                   # 변화량을 예측

train = trajectories(200, 64, 10)                       # 학습: 해상도 64, (200, 11, 64)
X, Y = train[:, :-1].reshape(-1, 64), train[:, 1:].reshape(-1, 64)
model = FNO(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for it in range(1500):
    idx = torch.randint(0, X.shape[0], (100,))
    opt.zero_grad(); loss = ((model(X[idx]) - Y[idx]) ** 2).mean(); loss.backward(); opt.step()

rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
with torch.no_grad():
    for N in [64, 128]:
        test = trajectories(50, N, 40)                  # 학습하지 않은 초기 조건, 40 step = 시간 4
        u, errs = test[:, 0], []
        for n in range(1, 41):
            u = model(u)
            if n in (10, 20, 40): errs.append(f"{n} step {rel(u, test[:, n]):.1e}")
        print(f"해상도 {N:3d} | 한 step 오차 {rel(model(test[:, 0]), test[:, 1]):.1e} | rollout: " + ", ".join(errs))
```

```output
해상도  64 | 한 step 오차 1.4e-03 | rollout: 10 step 5.1e-03, 20 step 1.2e-02, 40 step 3.9e-02
해상도 128 | 한 step 오차 2.0e-03 | rollout: 10 step 4.4e-03, 20 step 1.1e-02, 40 step 3.6e-02
```

해상도 64에서 학습한 모델이 128에서도 비슷한 오차를 낸다. 한 step 오차는 64에서 $1.4 \times 10^{-3}$, 128에서 $2.0 \times 10^{-3}$으로 128에서 조금 크다. 비선형 FNO에서 해상도 불변성은 근사적으로만 성립한다는 뜻이다. 이 차이가 모드 수나 학습량에 따라 어떻게 변하는지는 실습에서 확인한다. rollout 오차는 두 해상도 모두 step에 따라 커져 40 step에서 약 $4 \times 10^{-2}$이다. 한 step 오차의 40배보다는 작지만, 학습한 시간 범위(10 step)를 넘어서도 계속 자란다. 학습 데이터는 시간 0–1의 상태만 담고 있으므로, 상분리가 더 진행된 뒤의 상태는 학습 분포 밖이다.

**장기 안정성.** autoregressive rollout의 오차는 두 경로로 자란다. 첫째, 한 step의 오차가 다음 입력에 섞여 모델이 학습 분포 밖의 상태를 보게 되고(분포 이동), 그 상태에서 오차가 더 커진다. 둘째, 학습된 연산자가 어떤 모드를 실제보다 조금이라도 증폭하면 그 모드가 지수적으로 자란다. 한 step 평가에서는 둘 다 거의 보이지 않는다. 그래서 동역학 모델의 평가는 **rollout 길이에 따른 오차 곡선**으로 하고, 학습 구간보다 긴 rollout과 보존량·스펙트럼 같은 물리 검사(24.1장)를 함께 본다. 학습 중에 몇 step을 rollout해 그 오차까지 손실에 넣는 방법(multi-step 손실)과 입력에 작은 잡음을 섞어 분포 이동에 둔감하게 만드는 방법이 흔히 쓰인다.

## 제약

**미분 상태와 대수 제약의 처리.** 물리 상태에는 시간에 따라 진화하는 **미분 상태**와 매 순간 대수 관계를 만족해야 하는 **대수 제약**이 섞여 있다(9.7장의 DAE). 비압축성 유체의 속도장은 $\nabla \cdot \mathbf{u} = 0$을, 화학 평형은 질량 작용 법칙을, 전기 회로는 Kirchhoff 법칙을 매 순간 만족해야 한다. 미분 상태만 연산자로 예측하고 대수 변수는 제약식으로 다시 계산하거나, 예측한 상태를 제약 다양체로 사영(projection)하는 방법이 쓰인다. 비압축성 조건이라면 예측한 속도장을 Fourier 공간에서 발산이 0인 성분으로 사영하는 것이 구조 경로(24.1장)다.

같은 생각이 rollout의 안정성에도 쓰인다. 이류는 모든 Fourier 모드의 크기를 보존하므로 $L^2$ 노름이 보존된다. 잡음 섞인 데이터(표준편차 0.02)로 이류의 한 step($\Delta x = 0.01$) 스펙트럼 가중치를 학습하고, 그대로 쓰는 경우와 $|w_k| = 1$로 정규화한 경우의 rollout을 비교한다.

```python {exec="browser" session="operator"}
dt_shift = 0.01                                         # 한 step = 이류 0.01 (모든 모드의 크기 보존)
u_in = on_grid(a_train, 64)
u_out = on_grid(advect(a_train, dt_shift), 64) + rng.normal(0, 2e-2, u_in.shape)   # 잡음 섞인 학습 쌍
w = fit_spectral(u_in, u_out, K=K_IN)
w_unit = w / np.abs(w)                                  # 제약: |w_k| = 1 (L2 노름 보존)
print(f"학습된 |w_k|의 범위: {np.abs(w).min():.4f} ~ {np.abs(w).max():.4f}  (참값 1)")
u0 = on_grid(a_test[:10], 64)
for steps in [10, 100, 1000]:
    out = []
    for weights in [w, w_unit]:
        u = u0.copy()
        for _ in range(steps):
            u = spectral_conv(u, weights)
        true = on_grid(advect(a_test[:10], dt_shift * steps), 64)
        out.append(f"오차 {rel_err(u, true):.1e}, 노름 비 {np.linalg.norm(u) / np.linalg.norm(u0):.2f}")
    print(f"{steps:4d} step | 제약 없음: {out[0]} | |w|=1 제약: {out[1]}")
```

```output
학습된 |w_k|의 범위: 0.9948 ~ 1.0037  (참값 1)
  10 step | 제약 없음: 오차 3.6e-03, 노름 비 1.00 | |w|=1 제약: 오차 2.7e-03, 노름 비 1.00
 100 step | 제약 없음: 오차 3.8e-02, 노름 비 1.01 | |w|=1 제약: 오차 2.7e-02, 노름 비 1.00
1000 step | 제약 없음: 오차 1.4e+00, 노름 비 1.76 | |w|=1 제약: 오차 2.0e-01, 노름 비 1.00
```

학습된 증폭 인자 $|w_k|$는 0.995에서 1.004 사이로, 한 step 정확도로는 거의 완벽하다. 그러나 1을 넘는 모드는 1000 step 동안 지수적으로 자라 노름을 1.76배로 키우고, 오차는 1.4에 이른다. 크기를 1로 고정하면 노름이 정확히 보존되고 오차는 0.20으로, 위상 오차가 쌓인 만큼만 남는다. 제약은 위상 오차를 고치지 못하지만 **폭주를 막는다**. 확산처럼 에너지가 줄어야 하는 문제라면 $|w_k| \le 1$로 제한하는 것이 같은 역할을 한다.

::quiz{id="rollout-growth"}

## 실습

같은 PDE 데이터셋에 DeepONet과 FNO를 학습해 해상도 변화와 장기 rollout에서 성능을 비교한다. 셀 5를 출발점으로 다음을 수행한다.

1. **문제 정의.** 입력 함수(초기 조건), 출력 함수(시간 $\Delta t$ 뒤의 상태), 학습 이산화(64점 균일 격자), 평가 이산화(64, 128, 256점)를 한 표로 쓴다. 각 해상도의 정답을 어떻게 만들었는지(같은 solver, 같은 tolerance)도 적는다.
2. **DeepONet.** branch(센서 64개 → 은닉 → $p$)와 trunk($y$ → 은닉 → $p$)를 MLP로 만들어 같은 데이터로 학습한다. 128점 입력은 센서 위치로 보간해 넣는다.
3. **해상도 평가.** 두 모델의 한 step 오차를 세 해상도에서 비교한다. FNO의 모드 수 $K$를 8, 12, 20으로 바꾸어 고해상도 오차가 어떻게 변하는지 본다.
4. **rollout 평가.** 두 모델을 100 step rollout하고 step에 따른 오차 곡선, 평균값 $\int u\,dx$의 변화, 에너지 스펙트럼의 변화를 그린다. 학습 구간(10 step)을 넘는 지점을 표시한다.
5. **보고.** 어떤 모델이 어떤 조건에서 나은지, 그 차이가 구조(입력 이산화의 고정 여부, 모드 절단)로 설명되는지 쓴다.

이어서 연습문제를 푼다. 채널을 섞는 스펙트럼 합성곱 층을 구현하고, 해상도 불변성과 미분 연산자 표현을 테스트로 확인한다.

::exercise{id="spectral-conv-1d"}

## 실패 사례

- **학습 해상도에서만 평가하고 해상도 불변성을 주장하는 것.** 해상도 불변성은 다른 해상도에서 그 해상도의 정답과 비교해야 확인된다. 그리고 해상도 불변이라는 것은 같은 연산자를 정의한다는 뜻일 뿐, 정확하다는 뜻이 아니다(셀 4의 이류). 저해상도 정답을 보간해 고해상도 정답으로 쓰면 고주파 오차가 비교에서 사라진다.
- **한 step 오차로 동역학 모델을 평가하는 것.** 증폭 인자가 0.4% 큰 모드 하나가 1000 step 뒤 노름을 1.76배로 키웠다(셀 6). 학습 구간보다 긴 rollout의 오차 곡선과 물리 검사를 함께 보고한다.
- **DeepONet에 학습과 다른 센서 배치의 값을 그대로 넣는 것.** 값의 개수가 같아도 위치가 다르면 다른 입력이다(셀 2). 입력 이산화가 바뀌는 응용이라면 보간 단계를 명시하고 그 오차를 따로 잰다.
- **학습 데이터에 없는 주파수·형상·파라미터 범위에서 성능을 기대하는 것.** 데이터에 없는 모드의 가중치는 데이터가 정하지 못한다. 시험 세트가 학습 분포와 얼마나 떨어져 있는지 함께 보고한다.

## 통과 기준

- 학습할 연산자의 입력 함수, 출력 함수, 학습·평가 이산화, 각 해상도의 정답을 만든 방법을 명시한 문제 정의를 쓴다.
- 스펙트럼 합성곱 층을 구현해 서로 다른 해상도에서 같은 함수에 같은 결과를 내는지 확인하고, 연습문제 `spectral-conv-1d`의 테스트를 통과한다.
- DeepONet과 FNO를 같은 데이터로 학습해 학습 해상도와 다른 해상도에서 각 해상도의 정답과 비교한 오차 표를 만든다.
- 한 step 모델을 학습 구간보다 긴 rollout으로 평가해 step에 따른 오차 곡선과 보존량·스펙트럼 검사를 보고한다.

## 참고 문헌

- Chen, T., Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Transactions on Neural Networks*, 6(4), 911–917. doi:10.1109/72.392253 — 연산자의 보편 근사 정리
- Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3, 218–229. doi:10.1038/s42256-021-00302-5 — branch·trunk 구조의 DeepONet
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *International Conference on Learning Representations (ICLR)*. — Fourier 공간에서 적분 커널을 매개화한 FNO
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research*, 24(89), 1–97. — neural operator의 일반 틀, 이산화 불변성, 보편 근사
