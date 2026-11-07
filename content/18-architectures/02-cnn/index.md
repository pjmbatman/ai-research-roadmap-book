---
id: architectures/cnn
title: "CNN과 공간 구조"
title_en: "Convolutional Networks"
tier: core
status: draft
prerequisites: [architectures/inductive-bias]
---

# CNN과 공간 구조

영상의 픽셀, 기상장의 격자점, 센서 시계열의 시점처럼 규칙적인 격자 위에 놓인 데이터에서는 가까운 값끼리 강하게 얽혀 있고, 같은 패턴(경계, 전선, 스파이크)이 어느 위치에나 나타날 수 있다. 합성곱 신경망(convolutional neural network, CNN)은 이 두 성질을 국소성과 가중치 공유라는 가정으로 구조에 넣은 모델이다. 18.1장에서 본 것처럼 이 가정이 맞으면 적은 데이터로도 잘 일반화한다.

CNN을 설계할 때 가장 자주 틀리는 것은 크기 계산이다. 출력 shape이 맞지 않아 코드가 멈추는 것은 차라리 낫다. 더 위험한 것은 코드는 돌아가는데 모델이 볼 수 있는 입력 범위(receptive field)가 문제에 필요한 맥락보다 짧은 경우다. 이 장은 합성곱의 출력 크기와 receptive field를 손으로 계산하고 코드로 확인하는 것을 목표로 한다.

## 학습 목표

- convolution의 동작과 receptive field를 계산한다
- 1D·2D·3D CNN을 적용한다

## convolution

**정의.** 2차원 합성곱 층은 입력 $\mathbf{x} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$에 kernel $\mathbf{w} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$를 미끄러뜨리며 적용한다.

$$
y_{o,i,j} = b_o + \sum_{c=1}^{C_{\text{in}}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{o,c,u,v}\, x_{c,\; i + u,\; j + v}.
$$

출력 채널 $o$마다 kernel이 하나씩 있고, 각 kernel은 모든 입력 채널을 한꺼번에 본다. 딥러닝 라이브러리가 "convolution"이라 부르는 연산은 신호처리의 합성곱과 달리 kernel을 뒤집지 않는 **상호상관**(cross-correlation)이다. kernel은 학습되므로 뒤집느냐는 표현력에 차이가 없지만, 신호처리 코드(`np.convolve`, `scipy.signal.convolve2d`)와 결과를 비교할 때는 이 차이를 기억해야 한다.

**네 가지 조절 인자.** 합성곱의 동작은 네 값으로 정해진다.

| 인자 | 뜻 | 효과 |
|---|---|---|
| kernel 크기 $k$ | 한 번에 보는 창의 폭 | 클수록 넓게 보지만 파라미터가 $k^2$으로 는다 |
| stride $s$ | 창을 옮기는 간격 | 출력 해상도를 $1/s$로 줄인다 |
| padding $p$ | 입력 가장자리에 덧붙이는 칸 수 | 출력 크기와 경계 처리를 정한다 |
| dilation $d$ | kernel 원소 사이의 간격 | 파라미터 수는 그대로 두고 창의 실제 폭을 $d(k-1)+1$로 넓힌다 |

**출력 크기.** 한 축의 입력 길이가 $n$이면 출력 길이는

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1
$$

이다. 분자의 $d(k-1)+1$은 dilation을 반영한 창의 실제 폭이고, 내림은 마지막 창이 입력을 벗어나는 위치를 계산하지 않는다는 뜻이다. stride 1에서 입력과 출력의 길이를 같게 하는 "same" padding은 $p = d(k-1)/2$다($k$가 홀수일 때 정수).

아래 셀은 이 정의를 그대로 옮긴 `conv2d`를 만들고, 네 가지 설정에서 출력 shape이 공식과 같은지, 그리고 `scipy.signal.correlate2d`와 같은 값을 내는지 확인한다.

```python {exec="browser" session="cnn"}
import numpy as np
from scipy.signal import correlate2d

def out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

def conv2d(x, w, s=1, p=0, d=1):                     # x (C_in, H, W), w (C_out, C_in, k, k)
    C_out, C_in, k, _ = w.shape
    x = np.pad(x, ((0, 0), (p, p), (p, p)))          # zero padding
    H_out, W_out = out_size(x.shape[1], k, s, 0, d), out_size(x.shape[2], k, s, 0, d)
    y = np.zeros((C_out, H_out, W_out))              # (C_out, H_out, W_out)
    span = d * (k - 1) + 1                           # dilation을 반영한 kernel의 실제 폭
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, i * s:i * s + span:d, j * s:j * s + span:d]   # (C_in, k, k)
            y[:, i, j] = np.einsum("chw,ochw->o", patch, w)
    return y

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 32, 32))                     # (C_in, H, W)
w = rng.normal(size=(8, 3, 5, 5))                    # (C_out, C_in, k, k)
for s, p, d in [(1, 0, 1), (1, 2, 1), (2, 2, 1), (1, 4, 2)]:
    y = conv2d(x, w, s, p, d)
    print(f"stride {s}, padding {p}, dilation {d} → {y.shape}  공식 H_out = {out_size(32, 5, s, p, d)}")
ref = sum(correlate2d(x[c], w[0, c], mode="valid") for c in range(3))   # 채널마다 상호상관 후 합
print("scipy correlate2d와 일치:", np.allclose(conv2d(x, w)[0], ref))
```

```output
stride 1, padding 0, dilation 1 → (8, 28, 28)  공식 H_out = 28
stride 1, padding 2, dilation 1 → (8, 32, 32)  공식 H_out = 32
stride 2, padding 2, dilation 1 → (8, 16, 16)  공식 H_out = 16
stride 1, padding 4, dilation 2 → (8, 32, 32)  공식 H_out = 32
scipy correlate2d와 일치: True
```

$5 \times 5$ kernel에 padding 2는 "same"이고, dilation 2이면 kernel의 실제 폭이 9가 되어 같은 크기를 유지하려면 padding 4가 필요하다. 출력 채널 8개의 kernel은 각각 $3 \times 5 \times 5 = 75$개의 가중치를 가지므로 층 전체의 가중치는 $8 \times 75 = 600$개(bias 8개 별도)다. 이 수는 입력의 $32 \times 32$ 크기와 무관하다.

::quiz{id="conv-output-size"}

**padding의 종류.** zero padding은 경계 밖을 0으로 본다. 실제 신호가 0이 아니면 경계 근처 출력에 인공적인 가장자리가 생긴다. 주기 경계를 가진 물리 시뮬레이션(주기 상자 안의 유체, 경도 방향으로 감기는 전 지구 격자)에서는 반대쪽 끝의 값을 덧붙이는 circular padding이, 경계 값을 반복하는 replicate padding이나 거울상으로 덧붙이는 reflect padding이 더 자연스러울 수 있다. padding은 경계조건에 대한 가정이다.

## 귀납적 편향

**국소성과 가중치 공유.** 18.1장에서 본 대로 합성곱은 dense 층에 두 가지 제약을 건 것이다. 출력 한 칸은 입력의 $k \times k$ 창만 보고(국소성), 모든 위치가 같은 kernel을 쓴다(가중치 공유). 그래서 파라미터 수가 입력 크기와 무관하고, 학습할 때와 다른 크기의 입력에도 같은 층을 적용할 수 있다.

**이동 등변성과 그 한계.** stride 1 합성곱과 원소별 활성화는 이동 등변이다. 입력의 패턴을 옮기면 특징 지도도 같은 만큼 옮겨진다. 그러나 실제 CNN은 등변성을 근사적으로만 가진다.

- **경계.** zero padding을 쓰면 경계 근처에서는 이동이 원형이 아니므로 등변성이 깨진다.
- **stride와 pooling.** stride $s$의 출력은 stride 1 출력에서 $s$칸마다 하나를 고른 것이다. 입력을 $s$의 배수가 아닌 칸만큼 밀면, 고르는 칸이 바뀌어 출력이 단순 이동으로 설명되지 않는다.

```python {exec="browser" session="cnn"}
def conv1d_circ(x, w, s=1):                          # x (T,), w (k,), 원형 경계
    y = np.array([w @ np.roll(x, -t)[:len(w)] for t in range(len(x))])   # stride 1 출력 (T,)
    return y[::s]                                    # stride s = stride 1 출력을 s칸마다 하나씩

x1, w1 = rng.normal(size=16), rng.normal(size=3)
for s in [1, 2]:
    for shift in [1, 2]:
        y_shifted_in = conv1d_circ(np.roll(x1, shift), w1, s)
        y = conv1d_circ(x1, w1, s)
        is_shift = any(np.allclose(y_shifted_in, np.roll(y, m)) for m in range(len(y)))
        print(f"stride {s}, 입력 {shift}칸 이동 → 출력이 이동만 된 것인가: {is_shift}")
```

```output
stride 1, 입력 1칸 이동 → 출력이 이동만 된 것인가: True
stride 1, 입력 2칸 이동 → 출력이 이동만 된 것인가: True
stride 2, 입력 1칸 이동 → 출력이 이동만 된 것인가: False
stride 2, 입력 2칸 이동 → 출력이 이동만 된 것인가: True
```

stride 2 층은 짝수 칸 이동에만 등변이다. stride·pooling 층을 여러 번 거친 깊은 CNN은 전체 stride의 배수 이동에만 정확히 등변이고, 그 사이의 이동에는 출력이 조금씩 흔들린다. 분류 모델이 물체를 한 픽셀 옮겼을 때 예측이 바뀌는 현상의 한 원인이다.

::quiz{id="stride-breaks-equivariance"}

**회전과 크기.** 합성곱은 이동에만 등변이다. 회전이나 크기 변화에는 등변이 아니므로, 물체의 방향이 임의인 데이터(현미경 영상의 세포, 위에서 본 은하)에서는 회전 증강을 쓰거나 회전에 등변인 합성곱(group equivariant CNN)을 쓴다. 이 확장은 7.10장의 일반적인 틀을 그대로 따른다.

## receptive field

**정의.** 출력 한 칸에 영향을 줄 수 있는 입력 칸의 범위를 그 칸의 **receptive field**(RF)라 한다. $k \times k$ 합성곱 한 층의 RF는 $k$이고, 층을 쌓으면 RF가 넓어진다. 층 $l$의 kernel 크기 $k_l$, stride $s_l$, dilation $d_l$에 대해 마지막 층의 RF는

$$
r = 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l \prod_{i<l} s_i
$$

다. 층 $l$은 자기 입력에서 $(k_l - 1)d_l$칸만큼 창을 넓히는데, 그 입력의 한 칸은 원래 입력에서 앞 층들의 stride 곱만큼의 칸에 해당한다. 그래서 stride와 dilation이 RF를 지수적으로 키울 수 있다.

```python {exec="browser" session="cnn"}
def receptive_field(layers):                         # layers: [(k, s, d), ...] 입력 쪽부터
    r, jump = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * jump                      # 이 층이 넓히는 폭(입력 칸 단위)
        jump *= s                                    # 이 층 출력의 한 칸 = 입력의 jump칸
    return r

print("3x3 다섯 층              :", receptive_field([(3, 1, 1)] * 5))
print("3x3 stride 2 다섯 층     :", receptive_field([(3, 2, 1)] * 5))
print("k=3 dilation 1,2,4,8,16 :", receptive_field([(3, 1, 2 ** i) for i in range(5)]))

print("층 수 L   이론 RF   중앙 95% 폭")                # 같은 평균 kernel을 L번 쌓은 선형 CNN
for L in [4, 16, 64]:
    h = np.array([1.0])
    for _ in range(L):
        h = np.convolve(h, np.ones(3) / 3)           # impulse response (2L+1,)
    m = next(m for m in range(L + 1) if h[L - m:L + m + 1].sum() >= 0.95)
    print(f"{L:7d}   {len(h):7d}   {2 * m + 1:11d}")
```

```output
3x3 다섯 층              : 11
3x3 stride 2 다섯 층     : 63
k=3 dilation 1,2,4,8,16 : 63
층 수 L   이론 RF   중앙 95% 폭
      4         9             7
     16        33            13
     64       129            27
```

stride 1인 $3 \times 3$ 층 다섯 개의 RF는 11칸에 그치지만, 매 층 stride 2를 쓰거나 dilation을 2배씩 늘리면 같은 다섯 층으로 63칸을 덮는다. 두 방법의 차이는 해상도다. stride는 출력 해상도를 $2^5 = 32$분의 1로 줄이고, dilation은 해상도를 유지한다. 픽셀마다 또는 시점마다 출력이 필요한 문제(분할, 시계열 예측)에서 dilation이 자주 쓰이는 이유다.

::quiz{id="receptive-field-with-stride"}

**유효 receptive field.** 이론 RF는 영향을 줄 **수 있는** 범위일 뿐, 그 안의 모든 입력이 같은 크기로 기여하지는 않는다. 가장자리 입력에서 출력까지 가는 경로는 하나뿐이지만, 가운데 입력은 경로가 매우 많다. 셀의 아래쪽 표는 모든 원소가 $1/3$인 kernel을 $L$번 쌓은 선형 CNN의 impulse response에서, 영향의 95%가 담긴 가운데 폭을 센 것이다. 이론 RF는 $2L + 1$로 선형으로 늘지만, 95% 폭은 7, 13, 27로 대략 $\sqrt{L}$에 비례해 는다. 같은 kernel을 반복 합성하면 중심극한정리에 따라 Gaussian 모양이 되기 때문이다. Luo 외(2016)는 학습된 CNN에서도 영향이 Gaussian에 가까운 모양으로 가운데에 몰리고, 이론 RF의 일부만 차지한다는 것을 보이고 이를 **유효 receptive field**(effective receptive field)라 불렀다.

::quiz{id="effective-receptive-field"}

**연구에서의 의미.** 문제에 필요한 맥락의 크기를 먼저 물리적 단위로 적는다. "전선의 폭은 약 200 km이고 격자 간격은 25 km이므로 최소 8칸", "응답이 과거 3일의 강수에 의존하고 샘플 간격이 1시간이므로 최소 72 step"처럼. 그다음 RF가 이 크기를 넉넉히 덮도록, 유효 RF가 더 좁다는 것을 감안해 설계한다.

## 대표 구조

**ResNet.** 층을 깊게 쌓으면 RF와 표현력이 늘지만, 단순히 쌓은 깊은 CNN은 학습 자체가 어려워져 학습 오차까지 나빠진다. He 외(2016)는 층이 입력 자체 대신 입력에 더할 변화량을 배우게 하는 **residual connection** $\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)$로 이 문제를 풀었다. 덧셈 경로를 따라 gradient가 층을 건너뛰어 흐르므로(17.1장) 100층이 넘는 CNN도 학습된다. 오늘날 대부분의 깊은 구조(Transformer 포함)가 이 연결을 쓴다.

**encoder–decoder.** 입력과 같은 해상도의 출력이 필요한 문제(분할, 장 예측, 노이즈 제거)에서는 encoder가 stride·pooling으로 해상도를 줄이며 넓은 맥락을 모으고, decoder가 upsampling(전치 합성곱이나 보간 후 합성곱)으로 해상도를 되돌린다. 해상도를 줄이는 동안 RF는 빠르게 넓어지지만, 세밀한 위치 정보는 사라진다.

**U-Net.** Ronneberger 외(2015)는 의료 영상 분할을 위해 encoder의 각 해상도 단계 특징 지도를 같은 해상도의 decoder 단계에 이어 붙이는(skip connection) 구조를 제안했다. 넓은 맥락(깊은 층)과 세밀한 위치(얕은 층)를 함께 쓰므로 픽셀 단위 출력에 강하다. 과학 데이터에서는 격자장에서 격자장으로 가는 문제(기상장 예측, 시뮬레이션 대리 모델, 현미경 영상 분할)의 표준 baseline이다.

::quiz{id="unet-skip-role"}

## 적용 범위

합성곱의 차원은 데이터가 격자를 이루는 축의 수로 정한다. 채널 축은 격자 축이 아니라 "같은 위치의 여러 측정값"이다.

| 차원 | PyTorch 입력 shape | 격자 축 | 데이터 예 |
|---|---|---|---|
| 1D | `(B, C, T)` | 시간 또는 1차원 공간 | 센서 시계열, 스펙트럼, 심전도, 충방전 곡선 |
| 2D | `(B, C, H, W)` | 평면 | 영상, 위경도 격자의 기상장, 2차원 시뮬레이션 단면 |
| 3D | `(B, C, D, H, W)` | 부피 또는 평면 + 시간 | CT·MRI, 3차원 유체장, 영상 시퀀스 |

```python {exec="python"}
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
x1 = torch.randn(4, 2, 1000)                         # (B, C, T): 2채널 센서 시계열
x2 = torch.randn(4, 3, 64, 64)                       # (B, C, H, W): 영상
x3 = torch.randn(4, 1, 16, 32, 32)                   # (B, C, D, H, W): 3차원 장
conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
conv2 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
conv3 = nn.Conv3d(1, 8, kernel_size=3, stride=(1, 2, 2), padding=1)
for name, conv, x in [("1D", conv1, x1), ("2D", conv2, x2), ("3D", conv3, x3)]:
    n_param = sum(p.numel() for p in conv.parameters())
    print(f"{name}: {tuple(x.shape)} → {tuple(conv(x).shape)}  파라미터 {n_param}")

patches = F.pad(x1, (3, 3)).unfold(-1, 7, 2)          # (B, C_in, T_out, k): stride 2로 자른 창
manual = torch.einsum("bctk,ock->bot", patches, conv1.weight) + conv1.bias[:, None]
print("Conv1d = 창마다 가중합(상호상관):", torch.allclose(conv1(x1), manual, atol=1e-5))
```

```output
1D: (4, 2, 1000) → (4, 16, 500)  파라미터 240
2D: (4, 3, 64, 64) → (4, 16, 64, 64)  파라미터 448
3D: (4, 1, 16, 32, 32) → (4, 8, 16, 16, 16)  파라미터 224
Conv1d = 창마다 가중합(상호상관): True
```

파라미터 수는 $C_{\text{out}}(C_{\text{in}} \prod k + 1)$이다. 1D 층은 $16 \times (2 \times 7 + 1) = 240$, 3D 층은 $8 \times (1 \times 27 + 1) = 224$다. 3D 층에 축마다 다른 stride `(1, 2, 2)`를 주면 깊이(또는 시간) 축의 해상도는 유지하고 평면만 줄인다. 마지막 줄은 `Conv1d`가 입력을 stride 간격의 창으로 잘라(`unfold`) 가중치와 축약한 것과 같다는 것, 즉 kernel을 뒤집지 않는 상호상관이라는 것을 확인한다.

**시간 축에 쓸 때.** 시계열 예측에서는 출력 시점 $t$가 미래 입력 $x_{t+1}, x_{t+2}, \dots$를 보면 안 된다. 왼쪽에만 $(k-1)d$칸을 padding하는 **causal convolution**이 이 조건을 보장한다. causal convolution을 dilation과 함께 쌓은 시계열 모델은 18.3장에서 순환 모델과 비교한다.

**채널로 넣을 것과 축으로 넣을 것.** 기상장의 기압면 13개를 채널로 넣으면 2D CNN이고, 깊이 축으로 넣으면 3D CNN이다. 채널로 넣으면 기압면마다 다른 가중치를 쓰고(기압면이 서로 다른 물리량처럼 다뤄짐), 축으로 넣으면 기압면 사이에 같은 kernel을 공유한다(연직 방향의 이동 등변을 가정). 어느 쪽이 맞는지는 연직 방향에 같은 규칙이 성립하는지에 달렸다.

:::note[분야 예시]
기상 예보에서 위경도 격자를 2D CNN에 그대로 넣으면 두 가지 가정이 깨진다. 고위도로 갈수록 격자 한 칸의 실제 면적이 줄어 같은 $3 \times 3$ kernel이 위도마다 다른 물리적 거리를 보고, 경도 방향은 주기 경계인데 zero padding은 날짜변경선에서 장을 끊는다. 경도 방향에 circular padding을 쓰는 것은 첫 번째 대응이고, 구면 위에서 정의된 합성곱이나 그래프 기반 구조(18.6장)는 더 근본적인 대응이다.
:::

:::note[분야 예시]
광물·재료의 X선 회절 패턴이나 분광 스펙트럼은 1차원 격자(각도·파장) 위의 신호다. 봉우리의 위치가 시료에 따라 조금씩 이동하므로, 봉우리 모양을 감지하는 1D 합성곱과 이동에 둔감하게 만드는 pooling의 조합이 잘 맞는다. 다만 봉우리의 **절대 위치** 자체가 물성 정보(격자 상수)를 담는 경우에는 전체 pooling이 그 정보를 지운다. 이동 불변이 필요한 부분과 위치가 정보인 부분을 구분해야 한다.
:::

## 실습

시계열 CNN의 receptive field가 실제 입력 구간을 얼마나 덮는지 계산하고 성능과 연결한다. 코드는 `labs/architectures/cnn/receptive_field_lab.py`다(PyTorch, 집필 환경의 CPU에서 약 5분).

- 입력 $x_t$는 독립인 표준정규 외력이고, 출력은 감쇠 진동자의 impulse response $h_j = e^{-j/\tau}\cos(2\pi j / P)$($\tau = 12$ step, 주기 $P = 20$ step)로 걸러 낸 응답 $y_t = \sum_{j=0}^{95} h_j x_{t-j}$다. 응답은 과거 수십 step의 외력에 의존한다.
- 모델은 kernel 2, dilation 1, 2, 4, …인 causal dilated CNN(채널 32, ReLU, residual)이다. 층 수가 $L$이면 RF는 $2^L$ step이다.
- 입력이 독립이므로 RF 밖의 외력이 만든 응답은 어떤 모델도 예측할 수 없다. 그래서 **RF 밖의 impulse response 에너지 비율** $\sum_{j \ge r} h_j^2 / \sum_j h_j^2$이 정규화 MSE(MSE ÷ 응답의 분산)의 하한이다.

집필 시점의 실행 결과는 다음과 같다.

```output
impulse response: τ = 12 step, 주기 20 step, 길이 96
층 수   RF   RF 밖 에너지 비율   test 정규화 MSE
    2    4               0.341             0.344
    3    8               0.290             0.292
    4   16               0.062             0.063
    5   32               0.004             0.004
    6   64               0.000             0.004
    7  128               0.000             0.000
```

결과를 읽는 법은 세 가지다.

- **RF 밖 에너지 비율이 오차의 하한이고, 학습된 모델은 그 하한에 거의 붙어 있다.** RF 4에서 하한 0.341에 오차 0.344, RF 16에서 0.062에 0.063이다. 이 오차는 채널을 늘리거나 더 오래 학습해도 줄지 않는다. 모델이 볼 수 없는 외력이 만든 응답이기 때문이다.
- **필요한 맥락은 물리적 척도가 정한다.** 응답의 감쇠 시정수가 12 step이므로 에너지($h_j^2$)는 시정수 6 step으로 줄고, RF 32(감쇠 시정수의 약 2.7배)에서 하한이 0.004로 떨어진다. RF를 4에서 8로 늘려도 하한이 조금만 주는 것은 4–7 step 구간의 impulse response가 작기 때문이다(주기 20 step의 코사인이 $j = 5$에서 0을 지난다). RF와 성능의 관계는 단조롭게 매끄럽지 않고, 데이터의 시간 구조를 따라간다.
- **하한이 0이 된 뒤의 오차는 RF가 아니라 학습의 몫이다.** RF 64의 0.004는 같은 학습 step에서 층이 하나 더 많은 RF 128 모델(0.000)보다 크다. 하한과 실제 오차의 차이를 보면 "정보가 없어서 못 맞히는 것"과 "정보는 있는데 못 배운 것"을 구분할 수 있다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="conv2d-forward"}

::exercise{id="receptive-field-calc"}

## 실패 사례

- **receptive field가 필요한 맥락보다 짧은 모델.** 실습처럼 RF 밖의 정보는 모델 크기·학습 시간과 무관하게 쓸 수 없다. 모델이 일정 수준에서 더 좋아지지 않으면 층을 넓히기 전에 RF를 계산해 필요한 맥락의 길이와 비교한다. 유효 RF가 이론 RF보다 좁다는 것도 감안한다.
- **출력 크기를 맞추려고 padding·crop을 임의로 조정하는 것.** encoder–decoder에서 홀수 크기 입력은 downsampling과 upsampling을 거치며 한두 칸이 어긋난다. shape만 맞추려고 끝을 잘라 내면 skip connection으로 이어 붙인 두 특징 지도가 서로 다른 위치를 가리킬 수 있다. 입력 크기를 전체 stride의 배수로 맞추고, 어긋남이 있으면 어느 쪽 가장자리를 잘랐는지 확인한다.
- **경계조건을 padding으로 망가뜨리는 것.** 주기 경계 문제에 zero padding을 쓰면 경계에 인공적인 오차 띠가 생긴다. 오차 지도를 그려 경계 근처에 오차가 몰려 있는지 본다.
- **신호처리의 convolution과 섞어 쓰는 것.** 딥러닝 라이브러리의 합성곱은 kernel을 뒤집지 않는다. 미리 설계한 필터(미분 연산자, 평활 kernel)를 가중치로 넣을 때 `np.convolve`의 결과와 비교하면 방향이 뒤집혀 있다.

## 통과 기준

- 자신이 쓰는 CNN의 층별 출력 크기와 마지막 층의 receptive field를 손으로 계산하고, 문제에 필요한 맥락의 시간·공간 척도(물리 단위)와 비교해 설계를 정당화한다.
- 1D·2D·3D 중 무엇을 쓸지, 무엇을 채널로 넣고 무엇을 격자 축으로 넣을지 데이터의 구조로 설명한다.
- 연습문제 두 개(`conv2d-forward`, `receptive-field-calc`)의 테스트를 통과한다.

## 참고 문헌

- He, Zhang, Ren, Sun (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778. — residual connection
- Luo, Li, Urtasun, Zemel (2016). Understanding the effective receptive field in deep convolutional neural networks. *NIPS 29*, 4898–4906. — 유효 receptive field
- Ronneberger, Fischer, Brox (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, LNCS 9351, 234–241. — U-Net
