---
id: architectures/recurrent-and-ssm
title: "순환 모델과 상태공간 모델"
title_en: "Recurrent and State-Space Models"
tier: core
status: draft
prerequisites: [architectures/inductive-bias, linear-algebra/eigen-and-symmetric]
---

# 순환 모델과 상태공간 모델

물리계는 상태를 가진다. 진자의 다음 위치는 지금의 위치와 속도로 정해지고, 배터리의 다음 전압은 지금의 충전 상태와 온도로 정해진다. 과거 전체를 기억할 필요 없이 **현재 상태** 하나가 미래를 결정한다는 이 가정이 순환 신경망(recurrent neural network, RNN)과 상태공간 모델(state-space model, SSM)의 귀납적 편향이다. 같은 갱신 규칙을 모든 시점에 공유하므로, 시간 축의 가중치 공유이기도 하다.

이 장은 RNN이 왜 먼 과거를 배우기 어려운지를 gradient의 곱으로 보이고, gate를 단 LSTM·GRU, 합성곱으로 시간을 다루는 TCN, 그리고 재귀와 합성곱이라는 두 얼굴을 함께 가진 선형 상태공간 모델을 차례로 다룬다. 마지막으로 시계열 모델을 평가할 때 가장 흔한 함정인 one-step 평가와 free rollout 평가의 차이를 실험으로 확인한다.

## 학습 목표

- RNN·LSTM·GRU의 동작을 이해한다
- 상태공간 모델 계열을 안다
- rollout 평가를 수행한다

## RNN

**정의.** RNN은 시점마다 입력 $\mathbf{x}_t$와 이전 hidden state $\mathbf{h}_{t-1}$로 새 hidden state를 만든다.

$$
\mathbf{h}_t = \tanh(\mathbf{W}_h\mathbf{h}_{t-1} + \mathbf{W}_x\mathbf{x}_t + \mathbf{b}), \qquad \hat{\mathbf{y}}_t = \mathbf{W}_y\mathbf{h}_t.
$$

$\mathbf{h}_t$는 지금까지의 입력을 고정된 크기 $H$의 벡터로 요약한 것이다. 같은 $\mathbf{W}_h, \mathbf{W}_x$를 모든 시점에 쓰므로 파라미터 수는 시퀀스 길이와 무관하고, 학습 때보다 긴 시퀀스에도 적용할 수 있다.

**BPTT.** RNN을 시간 축으로 펼치면 같은 층을 $T$번 쌓은 깊은 신경망이 된다. 이 펼친 그래프에서 역전파하는 것을 BPTT(backpropagation through time)라 한다. 시점 $t$의 손실이 $k$ step 전의 hidden state에 주는 gradient는 Jacobian의 곱이다.

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}} = \prod_{s=t-k+1}^{t} \operatorname{diag}\!\big(1 - \mathbf{h}_s^2\big)\,\mathbf{W}_h.
$$

**gradient 소실과 폭발.** 7.5장에서 활성화 없는 선형 RNN의 gradient가 $(\mathbf{W}^\top)^k$로 전달되어, 스펙트럼 반지름이 1보다 작으면 지수적으로 사라지고 크면 지수적으로 폭발한다는 것을 보았다. tanh가 있으면 매 step 도함수 $1 - h^2 \le 1$이 곱해지므로 소실 쪽으로 더 기운다. 아래 셀은 같은 무작위 행렬을 0.8배와 1.2배로 키워 $\lVert\partial\mathbf{h}_t/\partial\mathbf{h}_0\rVert_2$를 잰다.

```python {exec="browser" session="rnn"}
import numpy as np

rng = np.random.default_rng(0)
H, T = 32, 60
W0 = rng.normal(size=(H, H)) / np.sqrt(H)            # 스펙트럼 반지름 약 1인 무작위 행렬 (H, H)
X = rng.normal(size=(T, H)) * 0.5                    # 입력이 만드는 구동 항 (T, H)

def jacobian_norms(scale, act):                      # ‖∂h_t/∂h_0‖₂, t = 1..T
    W, h, J, norms = scale * W0, np.zeros(H), np.eye(H), []
    for t in range(T):
        h = act(W @ h + X[t])
        D = 1 - h ** 2 if act is np.tanh else np.ones(H)   # 활성화의 도함수 (H,)
        J = (D[:, None] * W) @ J                     # ∂h_t/∂h_{t-1} = diag(D) W
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)

for name, act in [("선형", lambda z: z), ("tanh", np.tanh)]:
    for scale in [0.8, 1.2]:
        n = jacobian_norms(scale, act)
        print(f"{name:4s} W 배율 {scale}: t=10 {n[9]:8.1e}   t=30 {n[29]:8.1e}   t=60 {n[59]:8.1e}")
print(f"LSTM cell 경로(forget gate 0.97): t=60 {0.97 ** 60:.2f}")
```

```output
선형   W 배율 0.8: t=10  3.9e-01   t=30  1.5e-02   t=60  1.3e-04
선형   W 배율 1.2: t=10  2.2e+01   t=30  2.8e+03   t=60  5.0e+06
tanh W 배율 0.8: t=10  3.4e-02   t=30  8.5e-06   t=60  1.1e-10
tanh W 배율 1.2: t=10  9.3e-01   t=30  2.0e-02   t=60  5.3e-04
LSTM cell 경로(forget gate 0.97): t=60 0.16
```

선형 RNN은 배율에 따라 60 step 뒤 $10^{-4}$로 사라지거나 $10^{6}$으로 폭발한다. tanh RNN은 배율 1.2에서도 포화된 뉴런의 도함수가 작아 결국 사라진다. 60 step 전의 입력이 gradient에 남기는 몫이 $10^{-10}$이면, 그 입력과 지금 출력의 관계는 사실상 학습되지 않는다. Bengio 외(1994)는 정보를 오래 안정적으로 저장하는 것과 gradient로 학습하기 쉬운 것 사이에 이런 긴장이 있음을 분석했다.

폭발은 gradient의 노름을 일정 값 이하로 자르는 **gradient clipping**으로 비교적 쉽게 다룬다. 소실은 그렇게 고칠 수 없고, 구조를 바꿔야 한다.

::quiz{id="vanishing-factor"}

## LSTM과 GRU

**LSTM.** Hochreiter와 Schmidhuber(1997)의 long short-term memory(LSTM)는 hidden state와 별도로 **cell state** $\mathbf{c}_t$를 두고, 정보를 넣고 빼는 것을 gate로 조절한다.

$$
\begin{aligned}
\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t &= \sigma(\mathbf{W}_{\{f,i,o\}}\mathbf{x}_t + \mathbf{U}_{\{f,i,o\}}\mathbf{h}_{t-1} + \mathbf{b}_{\{f,i,o\}}), \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c\mathbf{x}_t + \mathbf{U}_c\mathbf{h}_{t-1} + \mathbf{b}_c), \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, \qquad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
\end{aligned}
$$

$\sigma$는 sigmoid로 0과 1 사이의 값을 내고, $\odot$은 원소별 곱이다. forget gate $\mathbf{f}_t$는 이전 기억을 얼마나 남길지, input gate $\mathbf{i}_t$는 새 후보 $\tilde{\mathbf{c}}_t$를 얼마나 넣을지, output gate $\mathbf{o}_t$는 기억을 얼마나 내보낼지 정한다.

핵심은 cell state의 갱신이 **덧셈**이라는 점이다. cell 경로를 따라 $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}$은 대략 $\operatorname{diag}(\mathbf{f}_t)$이고, 같은 가중치 행렬과 활성화 도함수의 곱이 반복되지 않는다. forget gate가 1에 가까우면 gradient가 거의 줄지 않는다. 다만 소실이 사라지는 것은 아니다. 위 셀의 마지막 줄처럼 forget gate가 0.97이어도 60 step이면 0.16이 된다. 장기 의존성은 "가능해지는" 것이지 "보장되는" 것이 아니다.

**GRU.** Cho 외(2014)의 gated recurrent unit(GRU)은 cell state 없이 gate 두 개(update, reset)로 같은 아이디어를 구현한다. update gate $\mathbf{z}_t$가 이전 상태와 새 후보 사이를 보간한다: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}$. $\mathbf{z}_t$와 $1 - \mathbf{z}_t$ 중 어느 쪽이 이전 상태에 곱해지는지는 문헌과 구현마다 다르다(위 식은 PyTorch 문서의 형태다). gate가 하나 적어 파라미터가 LSTM의 4분의 3이다.

PyTorch에서 세 모듈의 shape과 파라미터 수, 그리고 마지막 출력이 첫 시점과 마지막 시점의 입력에서 받는 gradient의 크기 비를 확인한다.

```python {exec="python"}
import torch
import torch.nn as nn

torch.manual_seed(0)
B_, T_, D_in, H_ = 8, 30, 10, 20
x = torch.randn(B_, T_, D_in, requires_grad=True)    # (B, T, D_in)
for cls in [nn.RNN, nn.LSTM, nn.GRU]:
    rnn = cls(D_in, H_, batch_first=True)
    out, _ = rnn(x)                                  # out (B, T, H): 모든 시점의 hidden state
    n_param = sum(p.numel() for p in rnn.parameters())
    g = torch.autograd.grad(out[:, -1].sum(), x)[0]  # (B, T, D_in)
    ratio = (g[:, 0].norm() / g[:, -1].norm()).item()   # 첫 시점 입력 / 마지막 시점 입력의 gradient 크기
    print(f"{cls.__name__:4s} out {tuple(out.shape)}  파라미터 {n_param:5d}  ‖∂y_T/∂x_1‖ / ‖∂y_T/∂x_T‖ = {ratio:.1e}")
```

```output
RNN  out (8, 30, 20)  파라미터   640  ‖∂y_T/∂x_1‖ / ‖∂y_T/∂x_T‖ = 1.8e-10
LSTM out (8, 30, 20)  파라미터  2560  ‖∂y_T/∂x_1‖ / ‖∂y_T/∂x_T‖ = 1.0e-06
GRU  out (8, 30, 20)  파라미터  1920  ‖∂y_T/∂x_1‖ / ‖∂y_T/∂x_T‖ = 7.7e-07
```

파라미터 수는 gate 수에 비례한다. gate마다 입력 가중치 $H \times D_{\text{in}}$, hidden 가중치 $H \times H$, 그리고 PyTorch는 bias를 입력 쪽과 hidden 쪽에 한 벌씩 두므로 $2H$를 가진다. RNN은 $200 + 400 + 40 = 640$, LSTM은 그 4배, GRU는 3배다. gradient 비를 보면 **학습 전의** LSTM·GRU도 30 step 전 입력의 gradient가 $10^{-6}$ 수준으로 작다. 기본 초기화에서 forget gate는 0.5 근처라서 cell 경로도 매 step 절반씩 줄기 때문이다. gate는 장기 기억을 학습할 **수 있는** 경로를 열어 줄 뿐이고, forget gate bias를 크게 초기화하는 관행도 이 때문이다.

::quiz{id="lstm-additive-path"}

::quiz{id="lstm-param-count"}

## 시간 convolution

**TCN.** 시계열을 순환 없이 합성곱으로 다룰 수도 있다. 18.2장의 causal convolution(왼쪽에만 padding해 미래를 보지 않는 합성곱)을 dilation 1, 2, 4, …로 쌓으면 층 수에 지수적인 receptive field를 얻는다. Bai 외(2018)는 이런 구조를 temporal convolutional network(TCN)라 부르고, 여러 시퀀스 과제에서 일반적인 RNN 계열과 비교해 경쟁력 있는 결과를 보고했다.

**RNN과의 비교.** 두 계열의 귀납적 편향은 다르다.

| | RNN·LSTM·GRU | TCN |
|---|---|---|
| 과거를 보는 범위 | 원리상 무한(상태에 요약), 실제로는 gradient가 정한다 | receptive field로 정확히 유한 |
| 학습 | 시점마다 순차적 | 모든 시점을 병렬로 |
| 추론(한 step씩) | 상태 하나만 갱신, 비용 일정 | receptive field만큼의 과거 입력을 보관 |
| 메모리 가정 | 과거가 고정 크기 상태로 요약된다 | 최근 $r$ step 안에 필요한 정보가 있다 |

TCN은 receptive field 밖의 정보를 볼 수 없다는 점이 분명해서 설계와 진단이 쉽다. RNN은 무한한 과거를 볼 수 있다고 하지만, 실제로 먼 과거를 쓰는지는 학습이 정하고 확인하기 어렵다. 다음 절의 상태공간 모델은 두 장점을 함께 노린다.

## 상태공간 모델

**선형 SSM.** 제어 이론과 신호처리에서 오래 쓰여 온 연속시간 선형 상태공간 모델은

$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}u(t), \qquad y(t) = \mathbf{C}\mathbf{x}(t)
$$

다. $\mathbf{x} \in \mathbb{R}^N$은 상태, $u$는 입력, $y$는 출력이다(시계열의 채널마다 이런 SSM을 하나씩 두는 것이 보통이다). 관측 간격 $\Delta$로 입력이 그 간격 동안 일정하다고 보고 정확히 적분하면(zero-order hold) 이산 시스템

$$
\mathbf{x}_k = \bar{\mathbf{A}}\mathbf{x}_{k-1} + \bar{\mathbf{B}}u_k, \qquad y_k = \mathbf{C}\mathbf{x}_k, \qquad \bar{\mathbf{A}} = e^{\Delta\mathbf{A}}, \quad \bar{\mathbf{B}} = \mathbf{A}^{-1}(e^{\Delta\mathbf{A}} - \mathbf{I})\mathbf{B}
$$

를 얻는다. 이것은 활성화가 없는 선형 RNN이다. 7.5장의 대응대로 $\operatorname{Re}\lambda(\mathbf{A}) < 0$이면 $|\lambda(\bar{\mathbf{A}})| < 1$이라 안정하다.

**두 얼굴: 재귀와 합성곱.** 처음 상태가 0이면 재귀를 펼쳐

$$
y_k = \sum_{j=0}^{k} \mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}\, u_{k-j} = \sum_{j=0}^{k} K_j\, u_{k-j}, \qquad K_j = \mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}
$$

로 쓸 수 있다. 같은 모델이 **재귀**(한 step에 상태 하나만 갱신, 추론에 알맞음)이면서 kernel $K$를 쓰는 **합성곱**(모든 시점을 한 번에, FFT로 $O(L\log L)$, 학습에 알맞음)이다. kernel은 단위 충격 입력에 대한 응답(impulse response)이고, $\bar{\mathbf{A}}$의 고윳값이 감쇠 속도와 진동 주기를 정한다.

```python {exec="browser" session="rnn"}
from scipy.linalg import expm

N, L, dt_s = 4, 200, 0.1
A = np.array([[-0.3, 2.0, 0.0, 0.0],                 # 연속시간 상태 행렬 (N, N): 감쇠 진동 한 쌍
              [-2.0, -0.3, 0.0, 0.0],
              [0.0, 0.0, -0.05, 0.0],                # 느리게 감쇠하는 모드
              [0.0, 0.0, 0.0, -1.0]])                # 빠르게 감쇠하는 모드
B, C = np.ones((N, 1)), rng.normal(size=(1, N))      # (N, 1), (1, N)
M = expm(np.block([[A, B], [np.zeros((1, N + 1))]]) * dt_s)   # zero-order hold 이산화
A_bar, B_bar = M[:N, :N], M[:N, N:]                  # (N, N), (N, 1)
u = rng.normal(size=L)                               # 입력 시계열 (L,)

x, y_rec = np.zeros((N, 1)), []
for k in range(L):                                   # 재귀: 한 step씩, 추론에 알맞음
    x = A_bar @ x + B_bar * u[k]
    y_rec.append((C @ x).item())

K = np.array([(C @ np.linalg.matrix_power(A_bar, j) @ B_bar).item() for j in range(L)])   # kernel (L,)
y_conv = np.fft.irfft(np.fft.rfft(u, 2 * L) * np.fft.rfft(K, 2 * L), 2 * L)[:L]           # FFT 합성곱
print("|λ(Ā)|      :", np.round(np.sort(np.abs(np.linalg.eigvals(A_bar))), 3))
print("재귀 = 합성곱:", np.max(np.abs(np.array(y_rec) - y_conv)) < 1e-8)
print("kernel |K_j| j=0, 50, 199:", np.round(np.abs(K[[0, 50, 199]]), 4))
```

```output
|λ(Ā)|      : [0.905 0.97  0.97  0.995]
재귀 = 합성곱: True
kernel |K_j| j=0, 50, 199: [0.1681 0.0358 0.0026]
```

블록 행렬 $\begin{bmatrix}\mathbf{A} & \mathbf{B} \\ \mathbf{0} & 0\end{bmatrix}$의 행렬 지수 한 번으로 $\bar{\mathbf{A}}$와 $\bar{\mathbf{B}}$를 함께 얻는다($\mathbf{A}$의 역행렬을 피하는 방법이다). 연속시간 고윳값 $-0.05$인 느린 모드는 이산화 후 $e^{-0.005} \approx 0.995$가 되어, kernel이 200 step 뒤에도 0이 아니다. 고윳값을 어디에 두느냐가 모델이 얼마나 먼 과거를 기억하는지를 정한다.

::quiz{id="ssm-kernel-formula"}

**S4.** 이 아이디어를 깊은 신경망의 층으로 쓰려면 두 문제를 풀어야 한다. 먼 과거를 기억하는 $\mathbf{A}$를 어떻게 정하는가, 그리고 긴 kernel $K_0, \dots, K_{L-1}$을 행렬 거듭제곱 없이 어떻게 빠르게 계산하는가. Gu 외(2022)의 S4(structured state space sequence model)는 장기 기억에 유리하도록 설계된 상태 행렬(HiPPO 이론에서 온 초기화)을 쓰고, 그 행렬에 저랭크 보정을 둔 구조를 이용해 kernel을 효율적으로 계산했다. 이렇게 만든 SSM 층을 비선형 층과 번갈아 쌓아, 길이 수천–수만인 시퀀스 과제(Long Range Arena)에서 좋은 결과를 보고했다.

**Mamba와 selective SSM.** S4의 $\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C}$는 시간에 따라 변하지 않는다(linear time-invariant, LTI). 그래서 합성곱으로 쓸 수 있지만, 입력의 내용에 따라 "이것은 기억하고 저것은 잊는" 선택을 할 수 없다. Gu와 Dao(2023)의 Mamba는 이산화 간격 $\Delta_k$와 $\mathbf{B}_k, \mathbf{C}_k$를 각 시점의 입력으로 계산하는 **selective SSM**을 제안했다. $\Delta_k$가 크면 상태가 빨리 잊고 새 입력을 크게 받아들이며, 작으면 기억을 유지한다. 대신 kernel이 시점마다 달라져 합성곱으로 쓸 수 없다.

```python {exec="browser" session="rnn"}
def scalar_ssm(u, delta):                            # x_k = exp(-Δ_k) x_{k-1} + Δ_k u_k
    x, ys = 0.0, []
    for u_k, d_k in zip(u, delta):
        x = np.exp(-d_k) * x + d_k * u_k
        ys.append(x)
    return np.array(ys)

content = rng.normal(size=100)                       # 입력의 내용 (L,): Δ_k를 정한다
def impulse_response(j, selective):
    delta = np.log1p(np.exp(2 * content)) if selective else np.full(100, 0.5)   # Δ_k = softplus(2 c_k) 또는 상수
    u = np.zeros(100)
    u[j] = 1.0                                       # 시점 j에 충격 하나
    return scalar_ssm(u, delta)[j:j + 5]             # 충격 직후 5 step의 응답

for selective in [False, True]:
    r20, r60 = impulse_response(20, selective), impulse_response(60, selective)
    print(f"selective={selective!s:5}  시점 20의 응답 {np.round(r20, 3)}  시점 60의 응답 {np.round(r60, 3)}")
```

```output
selective=False  시점 20의 응답 [0.5   0.303 0.184 0.112 0.068]  시점 60의 응답 [0.5   0.303 0.184 0.112 0.068]
selective=True   시점 20의 응답 [1.762 1.654 1.338 1.224 0.553]  시점 60의 응답 [0.151 0.001 0.    0.    0.   ]
```

$\Delta$가 상수이면 충격이 언제 들어오든 응답이 같다. 이것이 하나의 kernel로 쓸 수 있다는 뜻이다. $\Delta_k$가 입력 내용에 따라 바뀌면 시점 20의 충격은 크게 받아들여져 오래 남고, 시점 60의 충격은 작게 들어와 곧 잊힌다. 응답이 시점에 따라 다르므로 고정 kernel이 없다. 그래도 상태에 대해서는 여전히 **선형** 재귀 $x_k = a_k x_{k-1} + b_k$이므로, 결합법칙을 이용한 병렬 scan으로 모든 시점을 $O(\log L)$ 단계에 계산할 수 있다. Mamba는 이를 GPU 메모리 계층에 맞춘 구현으로 학습하고, 추론은 RNN처럼 상태 하나를 갱신한다.

::quiz{id="selective-ssm-scan"}

**병렬 학습과 순차 추론.** 시퀀스 모델의 비용을 정리하면 다음과 같다($L$은 시퀀스 길이).

| 모델 | 학습(한 시퀀스 전체) | 추론(새 step 하나) | 추론 시 보관하는 것 |
|---|---|---|---|
| RNN·LSTM·GRU | $O(L)$ 순차 | $O(1)$ | 고정 크기 상태 |
| TCN | $O(L)$ 병렬 | $O(r)$ | 최근 $r$ step의 입력 |
| LTI SSM(S4 계열) | $O(L\log L)$ 병렬(FFT 합성곱) | $O(1)$ | 고정 크기 상태 |
| selective SSM(Mamba 계열) | $O(L)$ 병렬 scan | $O(1)$ | 고정 크기 상태 |
| Transformer | $O(L^2)$ 병렬 | $O(L)$ | 지난 모든 step의 key·value(18.4장) |

:::note[분야 예시]
선형 SSM은 과학자에게 낯선 모델이 아니다. 칼만 필터의 상태 방정식, 전기회로의 RC 응답, 배터리의 등가회로 모델(전압 응답을 여러 RC 쌍으로 쓰는 모델)이 모두 같은 꼴이다. 배터리 등가회로의 RC 쌍 하나는 시정수 $\tau$인 감쇠 모드 하나, 즉 대각 $\mathbf{A}$의 원소 $-1/\tau$에 해당한다. 깊은 SSM 층은 이런 모드 여러 개를 데이터에서 배우고 비선형 층으로 엮는 모델이라고 볼 수 있다. 학습된 $\bar{\mathbf{A}}$의 고윳값을 시정수로 바꿔 물리 모델과 비교하는 것은 해석의 출발점이 된다.
:::

## 학습과 추론 방식

**teacher forcing.** 다음 값을 예측하는 시계열 모델은 학습할 때 보통 매 시점 **참값** 과거를 입력으로 준다. 이를 teacher forcing이라 한다. 모든 시점의 손실을 병렬로 계산할 수 있고 학습이 안정적이다.

**autoregressive rollout.** 실제로 미래를 예측할 때는 참값 과거가 없다. 처음 몇 개의 관측만 주고, 모델이 낸 예측을 다시 다음 입력으로 넣어 이어 간다(free rollout). 이때 두 가지가 달라진다.

- **오차 누적.** 한 step의 작은 오차가 다음 입력의 오차가 되고, 동역학이 이를 증폭하거나 위상 오차로 쌓는다.
- **exposure bias.** 학습 중에는 참값 과거만 보았으므로, 자기 예측이 만든 약간 어긋난 과거는 학습 분포 밖의 입력이다. 모델이 그런 입력에서 어떻게 행동할지는 학습되지 않았다. Bengio 외(2015)의 scheduled sampling은 학습 중 참값 대신 모델 예측을 넣는 비율을 점점 늘려 이 차이를 줄이는 방법이다.

다음 셀은 감쇠 진자 $\ddot\theta = -(g/\ell)\sin\theta - c\dot\theta$의 궤적(관측 잡음 0.001 rad)으로 AR(2) 모델 $\theta_{t+1} = \phi(\theta_t, \theta_{t-1})^\top\mathbf{w}$를 최소제곱으로 학습하고, 학습에 쓰지 않은 초기 각의 궤적에서 one-step 오차와 free rollout 오차를 비교한다.

```python {exec="browser" session="rnn"}
def pendulum_traj(th0, n_steps, dt_s=0.05, sub=10):  # 감쇠 진자, RK4로 적분해 dt_s마다 기록 [rad]
    f = lambda s: np.stack([s[1], -9.81 * np.sin(s[0]) - 0.05 * s[1]])
    s, h, out = np.stack([np.asarray(th0, float), np.zeros(len(th0))]), dt_s / sub, []
    for _ in range(n_steps):
        out.append(s[0].copy())
        for _ in range(sub):
            k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(out).T                           # (궤적 수, n_steps)

train = pendulum_traj([0.5, 1.0, 1.5], 1200) + 0.001 * rng.normal(size=(3, 1200))   # 관측 잡음 0.001 rad
test = pendulum_traj([1.2], 1200)[0]                                                  # (1200,)
feats = {"선형 AR(2)": lambda a, b: np.c_[a, b],
         "3차 AR(2) ": lambda a, b: np.c_[a, b, a ** 3, b ** 3, a ** 2 * b, a * b ** 2]}
for name, phi in feats.items():                      # θ_{t+1} ≈ φ(θ_t, θ_{t-1}) · w
    w = np.linalg.lstsq(np.vstack([phi(s[1:-1], s[:-2]) for s in train]),
                        np.concatenate([s[2:] for s in train]), rcond=None)[0]
    one = np.sqrt(np.mean((phi(test[1:-1], test[:-2]) @ w - test[2:]) ** 2))   # teacher forcing
    roll = list(test[:2])
    for _ in range(len(test) - 2):                   # 자기 예측을 다시 입력으로
        roll.append((phi(np.array([roll[-1]]), np.array([roll[-2]])) @ w).item())
    err = np.abs(np.array(roll) - test)              # (1200,)
    print(f"{name}  one-step RMSE {one:.1e}   rollout 최대 오차  1 s {err[:20].max():.3f}"
          f"  5 s {err[:100].max():.3f}  20 s {err[:400].max():.3f}")
```

```output
선형 AR(2)  one-step RMSE 6.8e-04   rollout 최대 오차  1 s 0.056  5 s 0.376  20 s 0.428
3차 AR(2)   one-step RMSE 6.0e-05   rollout 최대 오차  1 s 0.006  5 s 0.059  20 s 0.295
```

3차 모델의 one-step RMSE는 $6 \times 10^{-5}$ rad로 관측 잡음보다도 작다. 그러나 rollout 오차는 20 s 안에 0.3 rad, one-step 오차의 약 5000배까지 커진다. 두 모델의 one-step 오차는 10배 차이지만, 이 숫자만으로는 장기 예측이 쓸 만한지 알 수 없다. 학습 데이터의 관측 잡음이 회귀 입력에도 들어가 감쇠율 같은 계수가 조금 치우치고, 그 작은 치우침이 한 step에서는 보이지 않다가 수백 step에 걸쳐 쌓인다.

::quiz{id="one-step-vs-rollout"}

## 불규칙 샘플링

**왜 필요한가.** 임상 기록, 현장 환경 센서, 천문 관측, 배터리 운용 기록은 관측 간격이 일정하지 않고 결측이 흔하다. 앞의 모든 모델은 "한 step = 일정한 시간"을 가정한다. 이 가정을 깨는 데이터를 등간격 모델에 그대로 넣으면 1시간 뒤의 관측과 3일 뒤의 관측이 같은 한 step으로 다뤄진다.

**방법.**

- **시간 간격을 입력으로.** 관측값과 함께 직전 관측과의 간격 $\Delta t_k$를 특징으로 넣는다. 가장 단순하고, 어떤 순환 모델에도 붙일 수 있다.
- **결측 mask.** 등간격 격자에 맞추고 비는 칸을 채울 때는, 그 칸이 관측인지 채운 값인지를 나타내는 mask를 함께 넣는다. 0으로 채운 값과 실제로 0인 관측을 모델이 구별해야 하기 때문이다. 측정 빈도 자체가 정보인 경우도 많다(상태가 나쁜 환자일수록 자주 측정한다).
- **연속시간 모델.** 상태공간 모델은 연속시간에서 정의되므로, 실제 간격으로 $\bar{\mathbf{A}}_k = e^{\Delta t_k\mathbf{A}}$를 계산하면 간격이 다른 관측을 자연스럽게 다룬다. $e^{(s+t)\mathbf{A}} = e^{s\mathbf{A}}e^{t\mathbf{A}}$이므로, 관측이 없는 구간은 상태를 그 시간만큼 흘려보내기만 하면 된다. 상태의 변화율 자체를 신경망으로 두는 neural ODE 계열과 그 확장은 24장(과학 머신러닝)에서 다룬다.

::quiz{id="irregular-sampling"}

## 실습

같은 모델을 one-step과 free rollout으로 평가하고 오차 누적을 분석한다. 코드는 `labs/architectures/recurrent-and-ssm/rollout_eval.py`다. 본문 셀의 감쇠 진자 설정에서 운동 방정식의 형태를 아는 모델(특징 $[\theta_t, \theta_{t-1}, \sin\theta_t]$)을 더하고, 학습 범위 안(초기 각 1.2 rad)과 밖(2.0 rad)의 궤적에서 시간 구간별 rollout RMSE와 오차가 처음 0.1 rad를 넘는 시각을 잰다. 집필 시점의 실행 결과는 다음과 같다.

```output
[범위 안 1.2 rad]
모델          one-step RMSE   rollout RMSE ~ 1 s  ~ 5 s  ~20 s  ~60 s   0.1 rad 넘는 시각
선형 AR(2)            6.8e-04               0.040  0.183  0.221  0.306    1.35 s
3차 AR(2)            6.0e-05               0.003  0.026  0.131  0.197    6.85 s
sin AR(2)           5.5e-05               0.004  0.025  0.125  0.196    7.80 s

[범위 밖 2.0 rad]
모델          one-step RMSE   rollout RMSE ~ 1 s  ~ 5 s  ~20 s  ~60 s   0.1 rad 넘는 시각
선형 AR(2)            3.4e-03               0.492  1.541  1.580  0.979    0.20 s
3차 AR(2)            3.3e-04               0.086  0.195  0.174  0.384    0.55 s
sin AR(2)           8.4e-05               0.008  0.076  0.480  0.648    3.00 s
```

결과를 읽는 법은 네 가지다.

- **one-step 오차는 rollout 성능의 순위를 일부만 알려 준다.** 범위 안에서 3차 모델과 sin 모델의 one-step 오차는 거의 같고(6.0e-05, 5.5e-05), 0.1 rad를 넘는 시각도 7–8 s로 비슷하다. 그 뒤로는 둘 다 0.2 rad 수준까지 어긋난다.
- **학습 범위 밖에서 모델의 차이가 드러난다.** 진폭이 2.0 rad로 커지면 3차 다항식은 $\sin\theta$를 잘 흉내 내지 못해 0.55 s 만에 0.1 rad를 넘는다. 운동 방정식의 형태($\sin\theta$)를 아는 모델은 3.00 s까지 버틴다. 18.1장의 말로 하면 맞는 귀납적 편향이 외삽에서 차이를 만든다.
- **긴 구간의 RMSE는 위상이 어긋나면 포화한다.** 범위 밖 sin 모델의 20 s RMSE(0.480)가 3차 모델(0.174)보다 큰 것은 진동이 반 주기 가까이 어긋난 구간이 길기 때문이다. 오차가 진폭 수준에 이른 뒤의 RMSE는 모델의 우열을 말해 주지 않는다. 그래서 "오차가 기준을 처음 넘는 시각"(valid time)을 함께 보고한다.
- **관측 잡음이 rollout을 망친다.** 스크립트의 관측 잡음을 0으로 바꾸면 범위 안의 sin 모델은 60 s 동안 0.1 rad를 넘지 않는다. 잡음 0.001 rad가 회귀의 입력에도 들어가 계수(특히 감쇠)를 조금 치우치게 하고, 그 치우침이 rollout에서 쌓인 것이다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="ssm-scan-vs-conv"}

::exercise{id="ar-rollout"}

## 실패 사례

- **teacher forcing 성능만 보고 장기 예측 성능을 주장하는 것.** one-step 오차는 rollout 오차를 거의 말해 주지 않는다(실습에서 5000배 차이). 장기 예측을 주장하려면 rollout을 실제로 돌리고, 예측 구간마다의 오차와 오차가 기준을 넘는 시각을 보고한다.
- **rollout 오차를 한 숫자로 요약하는 것.** 긴 구간 전체의 RMSE는 위상이 어긋난 뒤 포화해 모델 순서를 뒤집기도 한다. 구간별 오차 곡선과 valid time을 함께 본다.
- **gradient 폭발을 학습률 탓으로 돌리는 것.** RNN의 손실이 갑자기 튀거나 NaN이 되면 gradient 노름을 기록하고 clipping을 먼저 확인한다. 반대로 손실은 안정한데 먼 과거에 의존하는 패턴을 못 배우면 소실을 의심하고, 위 셀처럼 먼 시점 입력에 대한 gradient 크기를 직접 잰다.
- **불규칙 간격 데이터를 등간격처럼 넣는 것.** 시간 간격과 결측 mask를 버리면 모델은 "다음 관측"과 "다음 시각"을 구별하지 못한다.
- **selective SSM이나 큰 RNN이 긴 맥락을 자동으로 쓴다고 가정하는 것.** 구조가 긴 맥락을 볼 수 있는 것과 학습된 모델이 실제로 쓰는 것은 다르다. 입력의 먼 과거를 바꿔 보는 섭동 실험으로 확인한다.

## 통과 기준

- 시계열 모델의 평가를 one-step과 rollout으로 나누어 보고하고, rollout은 예측 구간별 오차와 오차가 기준을 넘는 시각으로 보고한다.
- 선형 상태공간 모델의 재귀 계산과 합성곱 계산이 같은 출력을 낸다는 것을 코드로 보이고, selective SSM에서 이 동치가 깨지는 이유를 설명한다.
- 연습문제 두 개(`ssm-scan-vs-conv`, `ar-rollout`)의 테스트를 통과한다.

## 참고 문헌

- Bai, Kolter, Koltun (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. — TCN
- Bengio, Simard, Frasconi (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166. — RNN의 장기 의존성 학습의 어려움
- Bengio, Vinyals, Jaitly, Shazeer (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NIPS 28*, 1171–1179. — exposure bias와 scheduled sampling
- Cho et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014*, 1724–1734. — GRU
- Gu, Dao (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752. — selective SSM
- Gu, Goel, Ré (2022). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*. — S4
- Hochreiter, Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. — LSTM
