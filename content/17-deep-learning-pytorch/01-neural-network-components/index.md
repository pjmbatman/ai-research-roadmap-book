---
id: deep-learning-pytorch/neural-network-components
title: "신경망의 구성 요소"
title_en: "Building Blocks of Neural Networks"
tier: core
status: draft
prerequisites: [linear-algebra/affine-and-nonlinear, calculus-and-autodiff/automatic-differentiation]
---

# 신경망의 구성 요소

7.2장에서 아핀 층과 비선형 활성화를 번갈아 쌓아야 표현력이 생긴다는 것을 보았고, 8.4장에서 깊은 합성에서는 gradient가 층마다 곱해지며 사라지거나 폭발한다는 것을 보았다. 실제로 쓰는 신경망은 이 두 사실 사이에서 균형을 잡도록 설계된 부품들의 조합이다. 활성화는 표현력을, 초기화와 정규화 층은 신호의 크기를, residual connection은 gradient의 통로를, output head는 과제와 손실의 연결을 맡는다.

이 장은 각 부품이 **무엇을 하려고** 들어갔는지를 다룬다. 부품의 역할을 모르면 학습이 실패했을 때 "모델이 문제에 맞지 않는다"는 막연한 결론으로 가기 쉽다. 실제로는 초기화 scale 하나, `eval()` 호출 하나가 원인인 경우가 많다. PyTorch 코드는 개념이 필요한 곳에만 쓰고, 텐서와 Module의 문법은 17.2장과 17.3장에서 다룬다.

## 학습 목표

- 신경망의 각 구성 요소의 역할을 설명한다
- 초기화와 정규화가 학습에 주는 영향을 이해한다

## 층과 활성화

**왜 필요한가.** 신경망의 기본 단위는 아핀 변환 $\mathbf{W}\mathbf{x} + \mathbf{b}$ 뒤에 원소별 비선형 함수 $\sigma$를 붙인 층이다. 7.2장에서 본 것처럼 $\sigma$가 없으면 아무리 쌓아도 하나의 아핀 사상이다. 어떤 $\sigma$를 고르는가는 표현력보다 **학습 동역학**의 문제다. 활성화의 도함수가 역전파 때 곱해지는 국소 미분이기 때문이다.

**정의.** $L$층 다층 퍼셉트론(MLP)은

$$
\mathbf{h}_0 = \mathbf{x}, \qquad \mathbf{a}_\ell = \mathbf{W}_\ell\mathbf{h}_{\ell-1} + \mathbf{b}_\ell, \qquad \mathbf{h}_\ell = \sigma(\mathbf{a}_\ell), \qquad \hat{\mathbf{y}} = \mathbf{W}_{L+1}\mathbf{h}_L + \mathbf{b}_{L+1}
$$

이다. $\mathbf{a}_\ell$을 활성화 이전 값(pre-activation), $\mathbf{h}_\ell$을 활성값(activation)이라 한다. 마지막 층에는 보통 활성화를 붙이지 않고, 과제에 맞는 output head가 붙는다(이 장 마지막 절). 자주 쓰는 활성화는 다음 네 가지다.

| 이름 | 식 | 특징 |
|---|---|---|
| ReLU | $\max(0, z)$ | 양수 쪽 도함수 1, 음수 쪽 0. 계산이 싸고 포화하지 않는다 |
| GELU | $z\,\Phi(z)$ | $\Phi$는 표준정규 CDF. 0 근처에서 매끄럽고 음수 쪽에 작은 값이 남는다 |
| SiLU (Swish) | $z\,\mathrm{sigmoid}(z)$ | GELU와 모양이 비슷한 매끄러운 함수 |
| tanh | $\tanh z$ | 출력이 $(-1, 1)$로 유계, 원점에서 도함수 1, $\lvert z\rvert$가 크면 포화 |

```python {exec="browser" session="components"}
import math
import numpy as np

z = np.array([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])            # (6,) 활성화 이전 값
erf = np.vectorize(math.erf)
acts = {
    "ReLU": lambda z: np.maximum(0.0, z),
    "GELU": lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2))),   # z * Phi(z)
    "SiLU": lambda z: z / (1 + np.exp(-z)),                  # z * sigmoid(z)
    "tanh": np.tanh,
}
h = 1e-5
print("z       " + "".join(f"{v:8.1f}" for v in z))
for name, f in acts.items():
    d = (f(z + h) - f(z - h)) / (2 * h)                      # 중앙차분 도함수
    print(f"{name:5s}   " + "".join(f"{v:8.3f}" for v in f(z)))
    print("  d/dz  " + "".join(f"{v:8.3f}" for v in d))
```

```output
z           -3.0    -1.0    -0.1     0.1     1.0     3.0
ReLU       0.000   0.000   0.000   0.100   1.000   3.000
  d/dz     0.000   0.000   0.000   1.000   1.000   1.000
GELU      -0.004  -0.159  -0.046   0.054   0.841   2.996
  d/dz    -0.012  -0.083   0.420   0.580   1.083   1.012
SiLU      -0.142  -0.269  -0.048   0.052   0.731   2.858
  d/dz    -0.088   0.072   0.450   0.550   0.928   1.088
tanh      -0.995  -0.762  -0.100   0.100   0.762   0.995
  d/dz     0.010   0.420   0.990   0.990   0.420   0.010
```

**직관.** 표의 도함수 행이 역전파에서 곱해지는 값이다. tanh는 $\lvert z\rvert = 3$에서 도함수가 0.010으로, gradient를 100분의 1로 줄인다. 이것이 **포화**(saturation)다. ReLU는 양수 쪽에서 도함수가 정확히 1이라 gradient를 줄이지 않지만, 음수 쪽에서는 0이라 그 뉴런을 지나는 gradient를 완전히 막는다. 모든 입력에서 $a < 0$인 뉴런은 다시 살아나지 못하는데, 이를 죽은 ReLU(dead ReLU)라 한다. GELU와 SiLU는 음수 쪽에 작은 기울기를 남기고 0 근처에서 매끄럽다. 다만 $z = -3$에서 도함수가 음수이므로 단조 증가 함수가 아니고, $z = 1$처럼 도함수가 1을 넘는 구간도 있다.

**연구에서의 의미.** 활성화의 선택은 대개 아키텍처 관례를 따른다(예: 많은 Transformer 구현은 GELU 계열을 쓴다). 과학 문제에서 따로 고려할 점이 하나 있다. 모델 출력을 **입력으로 미분**해야 하는 경우(힘 = 에너지의 위치 gradient, PINN의 PDE 잔차)에는 ReLU의 2계 도함수가 거의 모든 곳에서 0이므로 쓸 수 없고, tanh·SiLU처럼 매끄러운 활성화가 필요하다. 활성화가 입력 공간을 어떻게 조각내는지는 아래 위젯에서 다시 확인한다.

::widget{name="relu-regions"}

:::note[분야 예시]
분자의 퍼텐셜 에너지 $E_\theta(\mathbf{r})$를 학습하는 원자간 퍼텐셜 모델은 원자에 작용하는 힘을 $\mathbf{F} = -\nabla_{\mathbf{r}} E_\theta$로 계산하고, 힘 데이터도 손실에 넣는 경우가 많다. ReLU 신경망의 에너지는 조각별 아핀이라 힘이 조각마다 상수가 되고 경계에서 불연속이다. 분자 동역학 시뮬레이션에서 에너지가 보존되지 않는 원인이 되므로, 이런 모델은 매끄러운 활성화를 쓴다.
:::

## 깊이와 폭

**정의.** 깊이(depth)는 은닉층의 수, 폭(width)은 층당 뉴런 수다. 입력 $n_{\text{in}}$, 출력 $n_{\text{out}}$인 Linear 층의 파라미터 수는 가중치 $n_{\text{in}}n_{\text{out}}$개와 bias $n_{\text{out}}$개를 더한 $n_{\text{out}}(n_{\text{in}} + 1)$이다. 폭 $w$인 은닉층이 이어지면 층마다 $w^2$에 비례하므로, **같은 파라미터 예산에서 깊이를 늘리면 폭은 그 제곱근에 반비례해 줄어든다.**

```python {exec="browser" session="components"}
def mlp_params(sizes):
    """Linear 층(n_in -> n_out)마다 가중치 n_in*n_out개 + bias n_out개"""
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))

for sizes in [[10, 256, 1], [10, 64, 64, 64, 1], [10, 40] + [40] * 7 + [1], [10, 256, 256, 1]]:
    print(f"은닉층 {len(sizes) - 2}개, 폭 {sizes[1]:3d}: 파라미터 {mlp_params(sizes):6d}개")
```

```output
은닉층 1개, 폭 256: 파라미터   3073개
은닉층 3개, 폭  64: 파라미터   9089개
은닉층 8개, 폭  40: 파라미터  11961개
은닉층 2개, 폭 256: 파라미터  68865개
```

폭 256인 층을 하나 더 쌓으면 파라미터가 3073개에서 68865개로 20배 넘게 늘지만, 폭 40인 층 8개는 12000개 남짓이다.

::quiz{id="param-count"}

**표현력과 학습 용이성.** 7.2장에서 본 것처럼 깊이는 선형 영역의 수를 폭보다 빠르게 늘릴 **수 있다**. 그러나 깊을수록 gradient가 지나야 하는 Jacobian의 곱이 길어지고(8.4장), 초기화와 정규화가 맞지 않으면 학습 자체가 되지 않는다. 즉 깊이는 표현력을 주는 대신 학습 용이성을 대가로 요구한다. 아래 절의 초기화·정규화 층·residual connection은 모두 이 대가를 줄이려는 장치다.

**연구에서의 의미.** "더 깊은 모델이 더 좋았다"는 결과는 파라미터 수, 계산량, 학습 설정이 함께 바뀐 비교인 경우가 많다. 깊이의 효과를 주장하려면 파라미터 예산을 맞추고, 각 설정의 학습률을 따로 튜닝해야 한다(10.4장의 공정한 비교와 같은 원칙). 과학 데이터처럼 표본이 적은 문제에서는 파라미터 수가 표본 수보다 훨씬 많아지기 쉬우므로, 파라미터 수를 먼저 세어 두는 습관이 필요하다.

## 초기화

**왜 필요한가.** 학습은 초기 파라미터에서 시작한다. 초기 가중치가 너무 작으면 층을 지날 때마다 신호가 줄어 깊은 층의 활성값과 gradient가 0에 가까워지고, 너무 크면 폭발하거나 tanh가 포화된다. 8.4장의 실험에서 gain 0.5와 3.0이 바로 그 두 경우였다. 초기화(initialization)의 목표는 **층을 지나도 신호의 분산이 유지되도록** 가중치의 scale을 고르는 것이다.

**분산 보존의 원리.** 층 하나 $a_i = \sum_{j=1}^{n_{\text{in}}} W_{ij}h_j$를 생각한다. $W_{ij}$가 평균 0, 분산 $s^2$로 독립이고 $h_j$와도 독립이면

$$
\operatorname{Var}(a_i) = n_{\text{in}}\, s^2\, \mathbb{E}[h_j^2]
$$

이다. 활성화가 선형에 가깝고 $h$의 평균이 0이면 $\mathbb{E}[h^2] = \operatorname{Var}(h)$이므로, $n_{\text{in}}s^2 = 1$이면 분산이 보존된다. ReLU는 입력이 0 중심 대칭일 때 음수 절반을 0으로 만들므로 $\mathbb{E}[h^2] = \tfrac{1}{2}\operatorname{Var}(a)$가 되어, 같은 분산을 유지하려면 $s^2$를 두 배로 해야 한다. 역전파에서는 같은 논리가 $n_{\text{out}}$에 대해 성립한다. 그래서 다음 두 규칙이 나온다.

- **Xavier(Glorot) 초기화**: $s^2 = \dfrac{2}{n_{\text{in}} + n_{\text{out}}}$. forward와 backward의 조건을 절충한 값으로, tanh처럼 원점 근처에서 선형인 활성화를 가정한다(Glorot, Bengio, 2010).
- **He(Kaiming) 초기화**: $s^2 = \dfrac{2}{n_{\text{in}}}$. ReLU가 분산을 절반으로 줄이는 것을 보정한다(He 외, 2015).

폭 128인 층 20개를 쌓고, 층마다 활성값의 표준편차를 잰다.

```python {exec="browser" session="components"}
rng = np.random.default_rng(0)
n, depth = 128, 20
x0 = rng.normal(size=(256, n))                              # (B, D) 표준화된 입력
inits = {"std 0.01": 0.01, "Xavier": np.sqrt(2 / (n + n)), "He": np.sqrt(2 / n)}
for act_name, act in [("tanh", np.tanh), ("ReLU", lambda z: np.maximum(0.0, z))]:
    for init_name, std in inits.items():
        h, stds = x0, []
        for _ in range(depth):
            W = rng.normal(size=(n, n)) * std               # (D_in, D_out)
            h = act(h @ W)                                  # (B, D)
            stds.append(h.std())
        print(f"{act_name:4s} {init_name:8s} 층 1·5·10·20의 활성값 std:",
              " ".join(f"{stds[k]:8.1e}" for k in [0, 4, 9, 19]))
```

```output
tanh std 0.01 층 1·5·10·20의 활성값 std:  1.1e-01  1.9e-05  3.5e-10  1.2e-19
tanh Xavier   층 1·5·10·20의 활성값 std:  6.3e-01  3.3e-01  2.2e-01  1.5e-01
tanh He       층 1·5·10·20의 활성값 std:  7.2e-01  5.7e-01  5.6e-01  5.5e-01
ReLU std 0.01 층 1·5·10·20의 활성값 std:  6.6e-02  2.8e-06  9.7e-12  8.1e-23
ReLU Xavier   층 1·5·10·20의 활성값 std:  5.8e-01  1.6e-01  3.4e-02  9.1e-04
ReLU He       층 1·5·10·20의 활성값 std:  8.2e-01  8.8e-01  1.2e+00  9.7e-01
```

고정된 작은 표준편차 0.01은 층마다 $0.01\sqrt{128} \approx 0.11$배씩 신호를 줄여, 20층 뒤에는 $10^{-19}$ 수준이다. ReLU에 Xavier를 쓰면 층마다 분산이 절반이 되어 20층 뒤 $10^{-3}$ 수준으로 줄고, He 초기화는 1 근처를 유지한다. tanh는 포화 때문에 Xavier에서도 천천히 줄어든다. 이 계산은 폭이 충분히 크고 가중치가 독립이라는 가정 위의 근사이므로, 실제 모델에서는 활성값 통계를 직접 재서 확인한다.

::quiz{id="he-init-std"}

**프레임워크의 기본값.** 초기화 규칙은 라이브러리 기본값에 숨어 있다. 집필 시점 예시로 PyTorch 2.15의 `nn.Linear`는 가중치를 $U(-1/\sqrt{n_{\text{in}}}, 1/\sqrt{n_{\text{in}}})$에서 뽑는다. 표준편차가 $1/\sqrt{3n_{\text{in}}}$로, He 초기화보다 $\sqrt{6}$배 작다. ReLU MLP를 깊게 쌓으면 기본값으로는 활성값이 층마다 줄어든다(실습에서 확인한다). He 초기화는 `nn.init.kaiming_normal_(w, nonlinearity="relu")`처럼 명시적으로 적용한다. 현재 버전의 기본 초기화는 각 모듈의 `reset_parameters` 구현과 문서에서 확인한다.

## 정규화 층

**왜 필요한가.** 초기화는 학습 **시작** 시점의 분산만 맞춘다. 학습이 진행되면 가중치가 바뀌고 활성값의 분포도 바뀐다. 정규화 층(normalization layer)은 매 forward에서 활성값을 평균 0·분산 1 근처로 다시 맞춘 뒤, 학습되는 scale $\gamma$와 shift $\beta$로 필요한 만큼 되돌린다. 깊은 모델을 큰 학습률로 안정적으로 학습시키는 데 널리 쓰인다.

**정의.** 세 층은 **어느 축으로 통계를 내는가**가 다르다. 입력이 $(B, D)$일 때

$$
\text{BatchNorm: } \hat{x}_{bd} = \frac{x_{bd} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}}, \quad \mu_d = \frac{1}{B}\sum_b x_{bd}, \qquad
\text{LayerNorm: } \hat{x}_{bd} = \frac{x_{bd} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}, \quad \mu_b = \frac{1}{D}\sum_d x_{bd}
$$

이다. BatchNorm(Ioffe, Szegedy, 2015)은 **특성마다 batch 축**으로, LayerNorm(Ba 외, 2016)은 **표본마다 특성 축**으로 평균과 분산을 낸다. RMSNorm(Zhang, Sennrich, 2019)은 LayerNorm에서 평균 빼기를 생략하고 제곱평균제곱근 $\sqrt{\frac{1}{D}\sum_d x_{bd}^2}$으로만 나눈다. 세 층 모두 출력은 $y = \gamma\hat{x} + \beta$(RMSNorm은 보통 $\gamma$만)이다.

**train/eval 동작 차이.** LayerNorm과 RMSNorm은 표본 하나 안에서 통계를 내므로 train과 eval에서 동작이 같다. BatchNorm은 다르다. train 모드에서는 현재 batch의 통계로 정규화하면서 이동 평균(running mean·var)을 갱신하고, eval 모드에서는 그 이동 평균으로 정규화한다. 그래서 BatchNorm 모델은 train 모드에서 **한 표본의 출력이 같은 batch의 다른 표본에 따라 달라진다.**

```python {exec="python"}
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(8, 4) * 3 + 5                              # (B, D) 평균 약 5, 표준편차 약 3
bn, ln, rms = nn.BatchNorm1d(4), nn.LayerNorm(4), nn.RMSNorm(4)
y_bn, y_ln, y_rms = bn(x), ln(x), rms(x)                   # 새로 만든 모듈은 train 모드
print("BN  특성별 평균 0, 분산 1:", y_bn.mean(0).abs().max().item() < 1e-6,
      torch.allclose(y_bn.var(0, unbiased=False), torch.ones(4), atol=1e-3))
print("LN  표본별 평균 0, 분산 1:", y_ln.mean(1).abs().max().item() < 1e-6,
      torch.allclose(y_ln.var(1, unbiased=False), torch.ones(8), atol=1e-3))
print("RMS 표본별 제곱평균 1    :", torch.allclose(y_rms.pow(2).mean(1), torch.ones(8), atol=1e-3))
print("BN running_mean       :", bn.running_mean.numpy().round(3))
print("0.1 * batch 평균        :", (0.1 * x.mean(0)).numpy().round(3))

bn.eval(); ln.eval()
with torch.no_grad():
    print("eval에서 출력이 바뀌는가  BN:", not torch.allclose(bn(x), y_bn),
          " LN:", not torch.allclose(ln(x), y_ln))
    print("BN(eval)은 표본 하나도 처리한다:", bn(x[:1]).numpy().round(2))
```

```output
BN  특성별 평균 0, 분산 1: True True
LN  표본별 평균 0, 분산 1: True True
RMS 표본별 제곱평균 1    : True
BN running_mean       : [0.558 0.397 0.571 0.481]
0.1 * batch 평균        : [0.558 0.397 0.571 0.481]
eval에서 출력이 바뀌는가  BN: True  LN: False
BN(eval)은 표본 하나도 처리한다: [[0.81 0.73 3.34 2.2 ]]
```

running mean은 0에서 시작해 batch 하나를 본 뒤 $0.9 \times 0 + 0.1 \times \mu_B$가 되었다(집필 시점 PyTorch의 기본 momentum 0.1). 아직 실제 평균 5와 거리가 멀기 때문에, 이 상태에서 eval로 바꾸면 출력이 train 때와 크게 다르다. 마지막 줄의 값이 0 근처가 아닌 것도 그 때문이다. train 모드의 BatchNorm은 표본이 하나인 batch를 받으면 분산을 낼 수 없어 오류를 낸다.

::quiz{id="norm-batch-dependence"}

**연구에서의 의미.** BatchNorm은 batch 통계에 의존하므로 batch가 작거나(메모리가 큰 3차원 과학 데이터), batch 안 표본들이 상관되어 있거나(같은 시계열에서 자른 창), 학습과 추론의 batch 구성이 다를 때 문제를 일으킨다. 이런 경우 LayerNorm·GroupNorm 같은 표본 단위 정규화를 쓰는 것이 안전하다. 어느 쪽이든 **평가 전에 `model.eval()`을 호출했는지**는 모든 평가 코드에서 확인할 첫 번째 항목이다.

::quiz{id="eval-mode-forgotten"}

## residual connection과 dropout

**residual connection.** 층 $F$를 그대로 쌓는 대신 입력을 더해 $\mathbf{h}_{\ell} = \mathbf{h}_{\ell-1} + F(\mathbf{h}_{\ell-1})$로 쓰는 구조다(He 외, 2016). 층의 Jacobian은

$$
\frac{\partial \mathbf{h}_\ell}{\partial \mathbf{h}_{\ell-1}} = \mathbf{I} + \frac{\partial F}{\partial \mathbf{h}_{\ell-1}}
$$

이 된다. 역전파의 곱 $\prod_\ell(\mathbf{I} + \mathbf{J}_\ell)^\top$을 전개하면 항등 행렬만 곱해지는 항이 하나 있으므로, $\mathbf{J}_\ell$이 아무리 작아도 gradient가 0으로 무너지지 않는다. 출력에서 입력까지 **덧셈만으로 이어진 지름길**이 생기는 것이다. 8.4장과 같은 실험을, 일부러 작게 초기화한 tanh 블록 50개로 반복한다.

```python {exec="browser" session="components"}
rng = np.random.default_rng(1)
D, L = 64, 50
Ws = [rng.normal(size=(D, D)) * 0.5 / np.sqrt(D) for _ in range(L)]   # 작은 초기화
for residual in [False, True]:
    h = rng.normal(size=D)                                  # (D,)
    Js = []
    for W in Ws:
        a = np.tanh(W @ h)                                  # 블록 F(h)
        J = (1 - a**2)[:, None] * W                         # dF/dh (D, D)
        h = h + a if residual else a
        Js.append(np.eye(D) + J if residual else J)         # 층 전체의 Jacobian
    g = np.ones(D) / np.sqrt(D)                             # 출력 쪽 gradient, 노름 1
    for J in reversed(Js):
        g = J.T @ g                                         # 역전파
    print(f"residual={residual!s:5s}  50층 아래 입력의 |grad| = {np.linalg.norm(g):.2e}")
```

```output
residual=False  50층 아래 입력의 |grad| = 4.34e-16
residual=True   50층 아래 입력의 |grad| = 4.69e+00
```

같은 가중치인데 residual이 없으면 gradient가 $10^{-16}$으로 사라지고, residual이 있으면 1 근처의 크기로 입력까지 내려온다. 반대로 블록이 많으면 $\mathbf{I} + \mathbf{J}$의 곱이 조금씩 커질 수 있다(여기서 4.69). 그래서 실제 아키텍처는 residual을 정규화 층과 함께 쓰거나, 블록 출력에 작은 scale을 곱해 시작한다.

::quiz{id="residual-jacobian"}

**dropout.** 학습 중에 각 활성값을 확률 $p$로 0으로 만드는 정규화 기법이다(Srivastava 외, 2014). 특정 뉴런 조합에 의존하는 공적응(co-adaptation)을 줄여 과적합을 막는 효과가 있다. 현재 프레임워크는 대부분 **inverted dropout**을 쓴다. 학습 때 살아남은 값을 $1/(1-p)$배 해서 기댓값을 유지하고, 평가 때는 아무것도 하지 않는다.

```python {exec="browser" session="components"}
p = 0.2
a = np.ones((10000, 8))                                     # (B, D) 모두 1인 활성값
mask = rng.random(a.shape) >= p                             # 확률 1-p로 유지
a_train = a * mask / (1 - p)                                # train: 끄고 1/(1-p)배
print(f"유지된 비율 {mask.mean():.3f}, 남은 값 {a_train.max():.2f}, train 평균 {a_train.mean():.3f}")
print("eval: 그대로 통과, 평균", a.mean())
```

```output
유지된 비율 0.800, 남은 값 1.25, train 평균 0.999
eval: 그대로 통과, 평균 1.0
```

::quiz{id="dropout-scale"}

dropout도 BatchNorm처럼 train/eval에서 동작이 다르다. 평가 때 `model.eval()`을 빠뜨리면 예측이 호출마다 무작위로 바뀐다. 반대로 이 무작위성을 일부러 켜 두고 예측을 여러 번 뽑아 불확실성을 어림하는 방법(MC dropout)도 있다. 불확실성 추정 방법의 평가는 21.1장에서 다룬다.

## output head

**왜 필요한가.** 몸통(backbone)이 만든 표현 $\mathbf{h}_L$을 과제의 출력으로 바꾸는 마지막 층을 output head라 한다. head의 출력 형태와 손실은 짝을 이뤄야 하고, 그 짝은 14.3장에서 다루듯 출력의 확률 모형에서 나온다.

| 과제 | head 출력 | 손실 | 주의 |
|---|---|---|---|
| 회귀 | 실수 $D_{\text{out}}$개 | MSE (가우시안 가정) | 타깃의 단위·스케일을 정규화해 둔다 |
| 이진 분류 | logit 1개 | BCE with logits | sigmoid는 손실 함수 안에서 |
| 다중 분류 | logit $K$개 | cross-entropy | softmax는 손실 함수 안에서 |
| 불확실성 있는 회귀 | 평균과 분산(양수 변환) | 가우시안 NLL | 분산은 softplus 등으로 양수화 |

집필 시점 PyTorch의 `nn.CrossEntropyLoss`와 `nn.BCEWithLogitsLoss`는 **logit**을 입력으로 받아 내부에서 log-softmax·log-sigmoid를 수치적으로 안정하게 계산한다. 모델 끝에 softmax를 붙인 뒤 이 손실에 넣으면 softmax가 두 번 적용되어 gradient가 약해지고 학습이 느려진다. 오류가 나지 않으므로 발견하기 어렵다.

::quiz{id="classification-head"}

**multi-head와 multi-task.** 하나의 몸통에 head를 여러 개 붙이면 여러 과제를 함께 학습할 수 있다. 예를 들어 분자 표현 하나에서 에너지(회귀)와 독성 여부(분류)를 동시에 예측한다. 이때 전체 손실은 $L = \sum_k \lambda_k L_k$이고, 각 손실의 단위와 크기가 다르므로 가중치 $\lambda_k$가 결과를 크게 좌우한다. 한 과제의 손실이 다른 과제보다 몇 자릿수 크면 몸통은 사실상 그 과제만 학습한다. 손실 결합의 원리는 14.3장, 구현은 17.5장에서 다룬다.

:::note[분야 예시]
기후 모델의 출력을 보정하는 신경망은 기온(K)과 강수량(mm/day)을 함께 예측하는 경우가 많다. 두 변수의 단위와 분포가 다르므로 head를 나누고, 기온은 가우시안 NLL, 강수량은 0이 많은 분포에 맞는 손실을 쓰는 식으로 변수마다 head와 손실을 따로 설계한다. 하나의 head에서 두 변수를 MSE로 함께 예측하면 수치가 큰 변수가 손실을 지배한다.
:::

## 실습

같은 파라미터 예산(약 12000개)에서 깊이·폭·활성화·초기화를 바꾸어 층별 활성값과 gradient의 분포를 관찰한다. 코드는 `labs/deep-learning-pytorch/neural-network-components/layer_stats.py`다. 입력 16차원의 회귀 문제(목표 $\sin(2\mathbf{w}^\top\mathbf{x})$ + 잡음)에 대해 은닉층 2개·폭 100과 은닉층 8개·폭 40의 MLP를 만들고, 활성화 2종(ReLU, tanh) × 초기화 3종(PyTorch 기본값, He, 표준편차 0.01)마다 (1) 초기 상태의 층별 활성값 표준편차를 forward hook으로 기록하고, (2) 첫 층 가중치의 gradient 노름을 잰 뒤, (3) Adam(학습률 $10^{-3}$)으로 300 step 학습한 검증 MSE를 보고한다. 집필 시점의 실행 결과는 다음과 같다(PyTorch 2.15, CPU).

```output
데이터: 학습 2048개 + 검증 2048개, 입력 16차원, 검증 목표 분산 0.494, 잡음 분산 0.010

은닉층 2개, 폭 100 (파라미터 11901개)
  활성화 초기화   층별 활성값 std(첫 층 → 마지막 층)       첫 층 |grad|  검증 MSE(300 step 후)
  relu   default  4e-01 2e-01                                       1.2e-01     0.026
  relu   he       8e-01 8e-01                                       4.1e-01     0.089
  relu   small    2e-02 2e-03                                       1.9e-02     0.027
  tanh   default  5e-01 3e-01                                       2.9e-01     0.025
  tanh   he       7e-01 6e-01                                       3.8e-01     0.049
  tanh   small    4e-02 4e-03                                       6.9e-02     0.022

은닉층 8개, 폭 40 (파라미터 12201개)
  활성화 초기화   층별 활성값 std(첫 층 → 마지막 층)       첫 층 |grad|  검증 MSE(300 step 후)
  relu   default  4e-01 1e-01 7e-02 6e-02 5e-02 5e-02 5e-02 5e-02   3.2e-04     0.021
  relu   he       8e-01 7e-01 7e-01 7e-01 6e-01 6e-01 5e-01 5e-01   2.0e-01     0.090
  relu   small    2e-02 9e-04 4e-05 2e-06 6e-08 3e-09 1e-10 5e-12   5.0e-11     0.494
  tanh   default  5e-01 3e-01 2e-01 1e-01 1e-01 1e-01 1e-01 1e-01   1.0e-02     0.020
  tanh   he       7e-01 6e-01 6e-01 5e-01 6e-01 6e-01 6e-01 6e-01   2.8e-01     0.101
  tanh   small    4e-02 2e-03 1e-04 8e-06 6e-07 4e-08 2e-09 1e-10   2.8e-09     0.074
```

결과를 읽는 법은 네 가지다.

- **얕은 모델은 초기화에 둔감하다.** 은닉층 2개에서는 표준편차 0.01로 시작해도 검증 MSE가 0.027로 다른 설정과 비슷하다. 신호가 줄어드는 층이 둘뿐이기 때문이다.
- **깊은 모델에서 작은 초기화는 학습을 막는다.** ReLU·표준편차 0.01·8층은 활성값이 층마다 약 20분의 1씩 줄어 마지막 층에서 $5 \times 10^{-12}$이고, 첫 층 gradient는 $5 \times 10^{-11}$이다. 검증 MSE 0.494는 목표의 분산과 같다. 평균만 예측하는, 즉 **아무것도 배우지 못한** 상태다. 같은 구조를 He로 초기화하면 학습된다. 실패의 원인은 깊이가 아니라 초기화다.
- **Adam은 작은 gradient를 어느 정도 보정한다.** PyTorch 기본 초기화의 ReLU 8층은 첫 층 gradient가 $3 \times 10^{-4}$로 He의 600분의 1이지만 잘 학습되었다(0.021). Adam은 좌표마다 gradient를 그 크기의 이동 평균으로 나누므로(10.4장) gradient의 절대 크기에 덜 민감하다. 같은 실험을 SGD로 하면 결과가 달라질 수 있다.
- **활성값 통계가 좋다고 결과가 가장 좋은 것은 아니다.** He 초기화는 층별 통계를 가장 잘 보존하지만, 이 학습률과 step 수에서는 검증 MSE가 0.09–0.10으로 기본값보다 나쁘다. 초기 함수가 더 크고 복잡해서 생기는 차이로 보이며, 학습률과 함께 튜닝해야 결론을 낼 수 있다. 활성값·gradient 통계는 **학습이 가능한지**를 진단하는 도구이지, 최종 성능의 순위를 정하는 도구가 아니다.

이어서 연습문제를 푼다. BatchNorm의 train/eval 동작을 NumPy로 직접 구현하며 정규화 층의 상태가 어떻게 바뀌는지 확인한다.

::exercise{id="batchnorm-train-eval"}

## 실패 사례

- **초기화를 바꾸고 학습 실패를 모델 구조 탓으로 돌리는 것.** 깊은 모델이 학습되지 않으면 "이 문제에는 깊은 모델이 맞지 않는다"고 결론 내리기 쉽다. 실습에서 같은 8층 모델이 표준편차 0.01에서는 아무것도 배우지 못했고, He 초기화에서는 학습되었다. 구조를 바꾸기 전에 층별 활성값·gradient 통계를 먼저 본다. 다른 코드에서 초기화 함수를 복사해 오거나, 사용자 정의 층에서 기본 초기화가 덮어쓰이는 경우가 특히 위험하다.
- **평가 때 `model.eval()`을 빠뜨리는 것.** BatchNorm은 평가 batch의 통계로 정규화하고 running 통계까지 갱신하며, dropout은 예측을 무작위로 만든다. 검증 점수가 batch 크기나 순서에 따라 바뀌면 이것부터 의심한다.
- **softmax를 두 번 적용하는 것.** 모델 끝의 softmax와 logit을 기대하는 손실 함수가 겹치면 오류 없이 학습만 느려진다.
- **파라미터 예산을 맞추지 않고 깊이·폭의 효과를 주장하는 것.** 깊이를 늘리며 파라미터 수도 몇 배 늘린 비교는 깊이의 효과를 분리하지 못한다.

## 통과 기준

- 층별 활성값 표준편차와 gradient 노름을 기록하는 코드를 작성하고, 그 통계로 학습이 되는 설정과 되지 않는 설정(예: 실습의 ReLU·표준편차 0.01·8층)을 구분해 원인을 설명한다.
- Xavier·He 초기화의 분산 식을 분산 보존 조건에서 유도하고, BatchNorm·LayerNorm·RMSNorm이 통계를 내는 축과 train/eval 동작 차이를 설명한다.
- 연습문제 `batchnorm-train-eval`의 테스트를 통과한다.

## 참고 문헌

- Glorot, Bengio (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*, PMLR 9. — 분산 보존 조건과 Xavier 초기화
- He, Zhang, Ren, Sun (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV 2015*. — ReLU를 위한 He 초기화
- Ioffe, Szegedy (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*. — BatchNorm
- Ba, Kiros, Hinton (2016). Layer normalization. arXiv:1607.06450. — LayerNorm
- Zhang, Sennrich (2019). Root mean square layer normalization. *NeurIPS 32*. — RMSNorm
- He, Zhang, Ren, Sun (2016). Deep residual learning for image recognition. *CVPR 2016*. — residual connection
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958. — dropout
