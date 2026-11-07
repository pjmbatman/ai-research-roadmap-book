---
id: calculus-and-autodiff/automatic-differentiation
title: "자동미분과 역전파"
title_en: "Automatic Differentiation and Backpropagation"
tier: core
status: draft
prerequisites: [calculus-and-autodiff/matrix-calculus]
---

# 자동미분과 역전파

8.2장에서 2층 MLP의 gradient를 손으로 유도했다. 층이 수십 개이고 attention·정규화·잔차 연결이 섞인 모델에서 같은 일을 손으로 하는 사람은 없다. `loss.backward()` 한 줄이 그 일을 하고, 그 비용은 순전파의 몇 배에 불과하다. 이 장은 그 한 줄 안에서 무슨 일이 일어나는지 다룬다.

자동미분(automatic differentiation)은 기호 미분도 수치미분도 아닌 세 번째 방법이다. 프로그램을 기본 연산의 계산 그래프로 보고, 각 연산의 국소 미분을 chain rule로 이어 붙인다. 이어 붙이는 순서에 따라 forward mode와 reverse mode가 있고, 신경망의 역전파(backpropagation)는 reverse mode다. 이 장에서는 30줄짜리 reverse mode 엔진을 직접 만들고 PyTorch autograd와 비교한 뒤, gradient를 검증하고 디버깅하는 방법과 깊은 합성에서 gradient가 사라지거나 폭발하는 이유를 본다.

## 학습 목표

- 계산 그래프와 forward·reverse mode를 설명한다
- 작은 자동미분 엔진을 구현한다
- gradient를 검증하고 디버깅한다

## 미분 계산의 세 방법

컴퓨터로 도함수를 얻는 방법은 세 가지다. 각각의 장단점을 알아야 자동미분이 왜 기본 도구가 되었는지, 그리고 나머지 둘을 언제 쓰는지 판단할 수 있다.

**기호 미분**(symbolic differentiation)은 수식을 받아 도함수의 수식을 돌려준다. 컴퓨터 대수 시스템이 하는 일이다. 결과가 정확하고 사람이 읽을 수 있지만, 합성이 깊어지면 같은 부분식이 반복되며 식이 기하급수적으로 커질 수 있고(expression swell), `if`·반복문이 섞인 프로그램에는 바로 적용하기 어렵다.

**수치미분**(numerical differentiation)은 함수값만으로 도함수를 근사한다. 중앙차분은

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

이다. 어떤 프로그램에도 쓸 수 있지만 두 가지 문제가 있다. 첫째, 입력이 $n$차원이면 gradient를 얻는 데 함수 평가가 $2n$번 필요하다. 파라미터가 백만 개면 순전파를 2백만 번 해야 한다. 둘째, **근사 오차**가 있다. Taylor 전개로 보면 중앙차분의 절단 오차(truncation error)는 $h^2$에 비례하고, $f(x + h) - f(x - h)$에서 가까운 두 수를 빼는 반올림 오차(round-off error)는 대략 $\epsilon/h$에 비례한다. $\epsilon$은 machine epsilon(float64는 약 $2.2 \times 10^{-16}$, float32는 약 $1.2 \times 10^{-7}$)이다. $h$를 줄이면 앞 항은 줄지만 뒤 항은 커진다.

```python {exec="browser" session="autodiff"}
import numpy as np

def f(x):
    return np.exp(np.sin(x))

x0 = 1.0
exact = np.cos(x0) * np.exp(np.sin(x0))               # 해석적 도함수
print("     h      float64 오차   float32 오차")
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]:
    d64 = (f(x0 + h) - f(x0 - h)) / (2 * h)            # 중앙차분
    x32, h32 = np.float32(x0), np.float32(h)
    d32 = (f(x32 + h32) - f(x32 - h32)) / (2 * h32)
    print(f"{h:8.0e}   {abs(d64 - exact):12.2e}   {abs(float(d32) - exact):12.2e}")
```

```output
     h      float64 오차   float32 오차
   1e-01       6.73e-03       6.73e-03
   1e-02       6.75e-05       6.20e-05
   1e-03       6.75e-07       1.05e-04
   1e-04       6.75e-09       1.68e-03
   1e-05       7.48e-11       1.02e-02
   1e-06       1.25e-10       6.13e-02
   1e-08       9.45e-09       1.25e+00
   1e-10       1.08e-06       1.25e+00
```

float64에서는 $h$를 10배 줄일 때마다 오차가 100배씩 줄다가($h^2$), $h = 10^{-5}$ 근처에서 바닥을 찍고 다시 커진다($\epsilon/h$). 두 항이 같아지는 $h \sim \epsilon^{1/3} \approx 6 \times 10^{-6}$이 최적이다. float32에서는 최적점이 $h = 10^{-2}$ 근처로 올라가고 최소 오차도 $6 \times 10^{-5}$에 머문다. $h \le 10^{-8}$에서는 $x + h$와 $x - h$가 float32로 같은 수가 되어 차분이 0이 되고, 오차가 도함수 자체의 크기(1.25)가 된다. 부동소수점 오차는 9.1장에서 자세히 다룬다.

**자동미분**은 프로그램이 실행하는 기본 연산(덧셈, 곱셈, $\exp$, $\tanh$, 행렬 곱 등) 각각의 도함수를 알고 있고, 실행 중에 chain rule로 그 값들을 결합한다. 결과는 근사가 아니라 **부동소수점 정밀도 안에서 정확한** 도함수값이고, 반복문과 분기가 있는 프로그램에도 그대로 동작한다. 다만 도함수의 식이 아니라 **특정 입력에서의 값**을 준다. 기호 미분과 수치미분의 장점을 함께 가지는 셈이다.

## 계산 그래프

자동미분을 하려면 프로그램을 "어떤 값이 어떤 값들로부터 어떤 연산으로 만들어졌는가"의 구조로 바꿔야 한다. 그 구조가 계산 그래프다.

**정의.** 계산 그래프(computational graph)는 노드가 값(입력, 파라미터, 중간값, 출력)이고 엣지가 "이 값이 저 값을 계산하는 데 쓰였다"를 나타내는 방향 비순환 그래프(DAG)다. 각 중간 노드 $v$는 기본 연산 $v = \phi(u_1, \dots, u_k)$의 결과이고, 그 연산의 **국소 미분** $\partial v/\partial u_j$는 입력값만 알면 바로 계산할 수 있다. 예를 들어 곱셈 $v = u_1 u_2$의 국소 미분은 $(u_2, u_1)$, $v = \tanh u$의 국소 미분은 $1 - v^2$이다.

**직관.** 뉴런 하나 $y = \tanh(w_1x_1 + w_2x_2 + b)$를 그래프로 그리면 곱셈 노드 두 개, 덧셈 노드 두 개, $\tanh$ 노드 하나다. 순전파는 그래프를 입력에서 출력으로 따라가며 값을 채운다. 출력 $y$를 입력 $w_1$로 미분하려면 $w_1$에서 $y$까지 가는 경로를 따라 국소 미분을 곱하면 된다.

**chain rule의 그래프 표현.** 8.1장에서 본 것처럼 한 변수가 여러 경로로 출력에 영향을 주면 경로마다의 곱을 더한다. 그래프 언어로 쓰면, 출력 $y$에 대한 노드 $u$의 미분을 $\bar{u} = \partial y/\partial u$(adjoint라 부른다)로 두고

$$
\bar{u} = \sum_{v:\ u \to v} \bar{v}\,\frac{\partial v}{\partial u}
$$

이다. 합은 $u$를 입력으로 쓰는 모든 노드 $v$에 걸친다. $\bar{y} = 1$에서 시작해 이 식을 출력 쪽에서 입력 쪽으로 적용하면 모든 노드의 adjoint가 나온다. 그 순서가 중요하다. 노드 $u$의 adjoint를 계산하려면 $u$를 쓰는 모든 $v$의 adjoint가 먼저 끝나 있어야 한다. 그래프를 **위상 정렬**(topological sort)한 순서의 역순으로 방문하면 이 조건이 보장된다.

::widget{name="autodiff-graph"}

## forward mode와 reverse mode

chain rule은 Jacobian의 곱 $\mathbf{J} = \mathbf{J}_L \cdots \mathbf{J}_2\mathbf{J}_1$이다. 곱셈은 결합법칙을 만족하므로 오른쪽부터 곱하든 왼쪽부터 곱하든 결과는 같다. 그러나 계산량은 크게 다르고, 그 차이가 두 모드를 가른다.

**forward mode: JVP.** 입력 쪽 벡터 $\mathbf{v} \in \mathbb{R}^n$을 하나 정하고 오른쪽부터 곱한다.

$$
\mathbf{J}\mathbf{v} = \mathbf{J}_L\big(\cdots\big(\mathbf{J}_2(\mathbf{J}_1\mathbf{v})\big)\big).
$$

이것을 Jacobian-vector product(JVP)라 한다. 각 단계는 "행렬 × 벡터"이고, 순전파와 같은 방향으로 진행하므로 중간값과 그 방향미분(tangent)을 함께 계산하며 나아가면 된다. 한 번 실행하면 방향 $\mathbf{v}$로의 방향미분 하나를 얻는다. Jacobian 전체($m \times n$)를 얻으려면 $\mathbf{v} = \mathbf{e}_1, \dots, \mathbf{e}_n$으로 **입력 차원 $n$번** 실행한다.

**reverse mode: VJP.** 출력 쪽 벡터 $\mathbf{u} \in \mathbb{R}^m$을 정하고 왼쪽부터 곱한다.

$$
\mathbf{u}^\top\mathbf{J} = \big(\big((\mathbf{u}^\top\mathbf{J}_L)\mathbf{J}_{L-1}\big)\cdots\big)\mathbf{J}_1.
$$

이것을 vector-Jacobian product(VJP)라 한다. 각 단계는 "행벡터 × 행렬"이고 출력에서 입력 방향으로 진행한다. 열벡터로 쓰면 $\mathbf{J}_k^\top\bar{\mathbf{z}}_k$, 즉 8.2장에서 본 "Jacobian의 전치를 곱해 gradient를 아래로 보낸다"는 계산이다. 한 번 실행하면 Jacobian의 한 행, 즉 출력 하나의 모든 입력에 대한 gradient를 얻는다. Jacobian 전체를 얻으려면 **출력 차원 $m$번** 실행한다.

**비용.** 두 모드 모두 한 번 실행하는 비용은 순전파의 작은 상수배다. 차이는 몇 번 실행해야 하는가다.

| | 한 번에 얻는 것 | Jacobian 전체 | 유리한 경우 |
|---|---|---|---|
| forward mode (JVP) | $\mathbf{J}\mathbf{v}$, 열 하나 | $n$번 | 입력이 적고 출력이 많을 때 |
| reverse mode (VJP) | $\mathbf{u}^\top\mathbf{J}$, 행 하나 | $m$번 | 입력이 많고 출력이 적을 때 |

신경망 학습은 파라미터 $n$이 수백만 이상이고 출력(손실)이 스칼라 하나($m = 1$)다. reverse mode 한 번이면 모든 파라미터의 gradient가 나온다. 이것이 딥러닝이 reverse mode를 쓰는 이유다. 반대로 입력이 몇 개이고 출력이 많은 경우, 예를 들어 시뮬레이션의 파라미터 3개가 출력 시계열 1000개에 주는 민감도를 구할 때는 forward mode가 유리하다.

::quiz{id="reverse-mode-cost"}

:::note[분야 예시]
수치예보의 4차원 변분 자료동화(4D-Var)는 수백만 차원의 대기 초기 상태가 관측과의 불일치(스칼라 비용 함수 하나)에 미치는 영향을 계산해 초기 상태를 고친다. 이때 쓰는 수반 모델(adjoint model)은 예보 모델 전체에 reverse mode를 적용한 것과 같은 계산이다. 신경망 분야에서 역전파가 쓰이기 전부터, 기상·해양·공학 설계 최적화에서는 같은 원리가 "수반 방법"(adjoint method)이라는 이름으로 쓰였다.
:::

## 역전파

역전파는 reverse mode 자동미분을 신경망 손실에 적용한 것이다. 이름이 특별해 보이지만 알고리즘은 앞 절의 adjoint 계산 그대로다. 직접 구현해 보면 그 단순함이 보인다.

**알고리즘.** (1) 순전파: 입력에서 출력으로 값을 계산하며, 각 노드에 입력 노드와 국소 미분을 기록한다. (2) 위상 정렬: 출력 노드에서 깊이 우선 탐색으로 그래프를 정렬한다. (3) 역전파: 출력의 adjoint를 1로 두고, 역순으로 방문하며 각 노드의 adjoint에 국소 미분을 곱해 입력 노드의 adjoint에 **더한다**.

아래 엔진은 이 세 단계를 스칼라 단위로 구현한 것이다. 각 연산이 순전파 때 국소 미분을 함께 기록해 두므로, 역전파는 연산의 종류를 몰라도 곱하고 더하기만 하면 된다.

```python {exec="browser" session="autodiff"}
import math

class Value:
    """스칼라 값 하나와 그 gradient를 담는 계산 그래프의 노드"""
    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.parents, self.local_grads = parents, local_grads   # 입력 노드와 국소 미분

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t**2,))

    def backward(self):
        order = []
        def visit(v):                      # 위상 정렬: 입력이 출력보다 먼저 온다
            if v not in order:
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):          # 출력에서 입력 쪽으로
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad   # chain rule, 여러 경로의 기여는 더한다
```

뉴런 하나의 gradient를 엔진, 손 유도, 유한차분으로 비교한다. $a = w_1x_1 + w_2x_2 + b$, $y = \tanh a$이면 손 유도는 $\partial y/\partial w_1 = (1 - y^2)x_1$ 등이다. 마지막 줄은 같은 노드 $u$를 세 번 쓰는 식으로, `+=`가 왜 필요한지 보여 준다.

```python {exec="browser" session="autodiff"}
x1, x2 = Value(2.0), Value(-1.0)
w1, w2, b = Value(-0.5), Value(1.5), Value(0.8)
y = (x1 * w1 + x2 * w2 + b).tanh()                  # 뉴런 하나
y.backward()

s = 1 - y.data**2                                   # tanh'(a)
def y_of(w1_):                                      # 유한차분용: w1만 바꾼 함수
    return math.tanh(2.0 * w1_ + (-1.0) * 1.5 + 0.8)
fd = (y_of(-0.5 + 1e-6) - y_of(-0.5 - 1e-6)) / 2e-6
print(f"y = {y.data:.6f}")
print(f"dy/dw1  엔진 {w1.grad:+.6f}  손 유도 {s * x1.data:+.6f}  유한차분 {fd:+.6f}")
print(f"dy/dw2  엔진 {w2.grad:+.6f}  손 유도 {s * x2.data:+.6f}")
print(f"dy/db   엔진 {b.grad:+.6f}  손 유도 {s:+.6f}")

u = Value(3.0)
z = u * u + u                                       # 같은 노드를 세 번 쓴다
z.backward()
print(f"z = u*u + u, dz/du 엔진 {u.grad}  손 유도 2u + 1 = {2 * 3.0 + 1}")
```

```output
y = -0.935409
dy/dw1  엔진 +0.250020  손 유도 +0.250020  유한차분 +0.250020
dy/dw2  엔진 -0.125010  손 유도 -0.125010
dy/db   엔진 +0.125010  손 유도 +0.125010
z = u*u + u, dz/du 엔진 7.0  손 유도 2u + 1 = 7.0
```

::quiz{id="grad-accumulate"}

같은 계산을 PyTorch autograd로 한다. PyTorch는 텐서 연산 단위로 같은 일을 한다. `requires_grad=True`인 텐서에서 시작한 연산은 그래프에 기록되고, `backward()`가 reverse mode로 `.grad`를 채운다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다.

```python {exec="python"}
import torch

x = torch.tensor([2.0, -1.0], dtype=torch.float64)
w = torch.tensor([-0.5, 1.5], dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
y = torch.tanh(w @ x + b)                           # 같은 뉴런
y.backward()
print(f"y = {y.item():.6f}")
print("dy/dw =", w.grad.numpy().round(6), "  dy/db =", round(b.grad.item(), 6))

w.grad = None
L = torch.tanh(w @ x + 0.8) ** 2                    # 스칼라 손실
(g,) = torch.autograd.grad(L, w, create_graph=True) # gradient도 그래프로 남긴다
v = torch.tensor([1.0, 0.0], dtype=torch.float64)
(Hv,) = torch.autograd.grad(g @ v, w)               # d(g·v)/dw = H v
t = torch.tanh(w @ x + 0.8).item()
H_exact = 2 * (1 - t**2) * (1 - 3 * t**2) * torch.outer(x, x)   # 손 유도 Hessian (2, 2)
print("H v (autograd) =", Hv.numpy().round(6))
print("H v (손 유도)  =", (H_exact @ v).numpy().round(6))
```

```output
y = -0.935409
dy/dw = [ 0.25002 -0.12501]   dy/db = 0.12501
H v (autograd) = [-1.625099  0.812549]
H v (손 유도)  = [-1.625099  0.812549]
```

직접 만든 엔진과 PyTorch가 같은 값을 낸다. 셀의 뒷부분(Hessian-vector product)은 다음 절에서 설명한다.

**gradient 계산과 parameter update의 구분.** 역전파가 하는 일은 여기까지다. 손실의 각 파라미터에 대한 gradient를 계산해 `.grad`에 더할 뿐, 파라미터를 바꾸지 않는다. 파라미터를 바꾸는 것은 optimizer(`optimizer.step()`)이고, 같은 gradient에 SGD·모멘텀·Adam 중 어떤 규칙을 적용할지는 10장의 문제다. 학습 한 step은 순전파 → 손실 → 역전파 → 업데이트의 네 단계이고(17.5장), 역전파는 그중 하나다. 이 구분을 알면 "gradient는 맞는데 학습이 안 된다"(업데이트 규칙·학습률 문제)와 "gradient 자체가 틀렸다"(모델·손실 구현 문제)를 나눠 진단할 수 있다. PyTorch가 `.grad`를 덮어쓰지 않고 더하기 때문에 매 step `zero_grad()`가 필요하다는 점도 같은 맥락이다.

## 메모리와 고차미분

reverse mode의 계산량은 작지만 공짜는 아니다. 역전파는 순전파가 끝난 뒤에 시작하므로, 그동안 국소 미분에 필요한 중간값을 모두 들고 있어야 한다.

**중간값 저장.** 엔진의 `local_grads`가 바로 그 저장소다. 신경망에서는 각 층의 활성값(activation)이 이에 해당하고, 저장 공간은 그래프의 크기, 즉 층 수 × batch 크기 × 폭에 비례한다. 큰 모델에서 GPU 메모리를 가장 많이 차지하는 것은 파라미터가 아니라 이 활성값인 경우가 많다. 대표적인 대응은 두 가지다.

- **gradient checkpointing**(activation recomputation): 일부 층의 활성값만 저장하고, 역전파 때 필요한 나머지는 순전파를 다시 계산해 얻는다. 계산을 더 하는 대신 메모리를 줄인다.
- **추론 시 그래프를 만들지 않기**: 평가·추론에서는 gradient가 필요 없으므로 `torch.no_grad()` 안에서 실행해 그래프와 중간값을 저장하지 않는다.

forward mode는 tangent를 값과 함께 앞으로 밀고 나가므로 중간값을 오래 저장할 필요가 없다. 메모리가 병목이고 입력 차원이 작은 문제에서는 forward mode가 대안이 된다.

**계산 그래프 유지.** PyTorch는 기본적으로 `backward()`가 끝나면 그래프를 해제한다. 같은 그래프로 역전파를 두 번 하려면 `retain_graph=True`가 필요하다. 반대로 학습 루프에서 손실 텐서를 리스트에 그대로 모아 두면(예: `losses.append(loss)`) 각 step의 그래프가 해제되지 않아 메모리가 계속 늘어난다. 기록용으로는 `loss.item()`처럼 숫자만 꺼내 둔다.

**고차미분과 Hessian-vector product.** gradient를 계산하는 과정 자체도 기본 연산들의 합성이므로, 그 과정을 그래프로 기록하면 gradient를 다시 미분할 수 있다. PyTorch에서는 `create_graph=True`가 그 기록을 켠다. 앞 셀은 이 방법으로 Hessian 전체를 만들지 않고 Hessian과 벡터의 곱을 계산했다.

$$
\mathbf{H}\mathbf{v} = \nabla_{\mathbf{w}}\big(\nabla_{\mathbf{w}} L^\top\mathbf{v}\big).
$$

gradient와 고정 벡터 $\mathbf{v}$의 내적은 스칼라이므로, 그것을 다시 reverse mode로 미분하면 $\mathbf{H}\mathbf{v}$가 나온다(Hessian이 대칭이므로 $\mathbf{H}^\top\mathbf{v} = \mathbf{H}\mathbf{v}$다). 비용은 gradient 계산의 몇 배 수준이고, $n \times n$ Hessian을 저장하지 않는다. 파라미터가 백만 개면 Hessian은 $10^{12}$개 성분이라 저장할 수 없지만, Hessian-vector product는 계산할 수 있다. 손실 지형의 곡률 분석(10.6장), 2차 최적화(10.5장), 영향 함수 같은 방법이 이것에 의존한다. 셀에서 손 유도 Hessian은 $L = \tanh^2(a)$, $a = \mathbf{w}^\top\mathbf{x} + b$에서 $\nabla^2 L = 2(1 - t^2)(1 - 3t^2)\,\mathbf{x}\mathbf{x}^\top$($t = \tanh a$)이다.

:::note[분야 예시]
물리 정보 신경망(PINN, 24.2장)은 신경망 $u_\theta(x, t)$가 편미분방정식을 만족하도록 학습한다. 예를 들어 열방정식 $\partial_t u = \kappa\,\partial_{xx}u$의 잔차를 손실에 넣으려면 신경망 출력을 **입력** $x, t$로 두 번 미분해야 하고, 그 잔차를 다시 **파라미터** $\theta$로 미분해야 학습할 수 있다. 자동미분의 고차미분이 이 계산을 가능하게 한다. 같은 도구로 분자 에너지 모델의 힘(에너지의 위치 gradient)을 학습 손실에 넣는 것도 gradient의 gradient 계산이다.
:::

## gradient 검증

자동미분은 기본 연산의 미분이 맞다면 정확하다. 그러나 직접 만든 연산(custom autograd function), 손으로 쓴 역전파, 수치적으로 위험한 손실은 틀릴 수 있고, 틀린 gradient로도 학습은 "어느 정도" 진행되기 때문에 오류가 잘 드러나지 않는다. 그래서 새 연산이나 손실을 만들면 유한차분으로 검증한다.

**상대오차.** 해석적 gradient $g$와 중앙차분 $\hat{g}$를 비교할 때 절대오차 $\lvert g - \hat{g}\rvert$는 값의 크기에 따라 의미가 달라진다. gradient가 $10^3$이면 $10^{-4}$의 차이는 무시할 만하지만, $10^{-5}$이면 전부 틀린 것이다. 그래서 상대오차

$$
r = \frac{\lvert g - \hat{g}\rvert}{\max\big(\lvert g\rvert + \lvert\hat{g}\rvert,\ \delta\big)}
$$

를 쓴다. $\delta$는 둘 다 0일 때 나눗셈을 막는 작은 값이다. float64와 $h \approx 10^{-6}$에서 올바른 구현의 상대오차는 보통 $10^{-7}$ 이하이고, $10^{-2}$ 이상이면 거의 확실히 버그다. 그 사이는 함수의 매끄러움에 따라 해석이 달라진다.

**수치 정밀도.** 첫 번째 셀이 보여 주듯 float32에서는 같은 $h$로도 오차가 몇 자릿수 크다. 검증은 **float64로** 한다. 모델이 float32·bf16으로 학습하더라도, 검증할 때는 같은 연산을 float64로 실행해 비교한다.

**검증 절차.** 실무에서는 다음 순서를 따른다.

1. 작은 입력(batch 2–4개, 작은 폭)과 무작위 파라미터를 float64로 준비한다.
2. 파라미터 텐서마다 따로 상대오차를 잰다. 버그는 그 버그가 지나는 파라미터에서만 드러나므로(8.2장 실습), 위치를 좁힐 수 있다.
3. ReLU·$\lvert x\rvert$·max처럼 꺾이는 점이 있으면, 차분 구간 $[x - h, x + h]$가 꺾이는 점을 지나는 성분에서 큰 오차가 나올 수 있다. 그런 성분은 입력을 조금 옮겨 다시 확인한다.
4. 확률적 요소(dropout, 무작위 증강)는 끄거나 시드를 고정한다. 함수가 호출마다 다르면 차분은 의미가 없다.

PyTorch에는 이 절차를 자동화한 함수(집필 시점 예시: `torch.autograd.gradcheck`)가 있다. 사용법과 기본 허용오차는 버전마다 바뀔 수 있으므로 공식 문서에서 확인한다.

::quiz{id="fd-float32"}

## gradient 소실·폭발

역전파는 층마다 Jacobian의 전치를 곱해 내려간다. 층이 $L$개이면 입력 쪽 gradient는 $L$개 행렬의 곱을 거친다. 곱이 길어지면 크기가 지수적으로 줄거나 커지고, 이것이 깊은 신경망과 긴 시계열 모델의 오래된 문제다.

**원인.** 각 층이 gradient의 노름을 대략 $\gamma$배 한다면 $L$층 아래에서는 $\gamma^L$배가 된다. $\gamma = 0.9$이고 $L = 50$이면 $0.005$, $\gamma = 1.1$이면 $117$이다. 층의 Jacobian은 $\operatorname{diag}(\sigma'(\mathbf{a}))\mathbf{W}$ 꼴이므로 $\gamma$는 가중치의 크기(초기화 scale)와 활성화 함수 도함수의 크기가 함께 정한다. sigmoid의 도함수는 최대 $1/4$이라 층마다 gradient를 줄이고, $\tanh$는 원점 근처에서 1이지만 포화되면 0에 가깝다.

폭 64인 $\tanh$ 층 30개를 쌓고, 가중치 scale(gain)을 바꿔 가며 역전파 중 gradient 노름을 잰다. 출력 쪽 gradient의 노름은 1로 두고, 5층마다 출력한다.

```python {exec="browser" session="autodiff"}
rng = np.random.default_rng(0)
D, L = 64, 30
for gain in [0.5, 1.0, 3.0]:
    h = rng.normal(size=D)                          # (D,) 입력
    Js = []
    for _ in range(L):
        W = rng.normal(size=(D, D)) * gain / np.sqrt(D)   # (D, D)
        h = np.tanh(W @ h)
        Js.append((1 - h**2)[:, None] * W)          # 층의 Jacobian (D, D)
    g = np.ones(D) / np.sqrt(D)                     # 출력 쪽 gradient, 노름 1
    norms = []
    for J in reversed(Js):                          # 역전파: VJP를 차례로
        g = J.T @ g
        norms.append(np.linalg.norm(g))
    print(f"gain {gain}: 층 30→1의 |grad|", " ".join(f"{n:.1e}" for n in norms[4::5]))
```

```output
gain 0.5: 층 30→1의 |grad| 2.6e-02 1.0e-03 4.0e-05 1.2e-06 4.6e-08 1.0e-09
gain 1.0: 층 30→1의 |grad| 1.2e+00 9.3e-01 7.6e-01 4.6e-01 2.3e-01 9.5e-02
gain 3.0: 층 30→1의 |grad| 3.2e+00 6.9e+00 3.1e+01 1.7e+02 8.0e+02 8.6e+02
```

gain 0.5에서는 30층을 내려가는 동안 gradient가 $10^{-9}$로 사라진다. 입력 쪽 층은 사실상 학습되지 않는다. gain 3.0에서는 수백 배로 커진다. gain 1.0(입력 차원의 제곱근으로 나눈 초기화)에서도 $\tanh$의 포화 때문에 천천히 줄어든다. gradient의 크기가 깊이에 대해 안정적이려면 층마다의 배율이 1 근처여야 한다.

**진단 방법.** 학습 중에 다음을 기록하면 소실·폭발을 일찍 발견할 수 있다.

- **층별 gradient 노름.** 입력 쪽 층의 gradient 노름이 출력 쪽보다 몇 자릿수 작으면 소실, 손실이 갑자기 튀거나 `nan`이 되기 직전에 노름이 급증하면 폭발이다.
- **gradient 노름과 파라미터 노름의 비.** 한 step에 파라미터가 얼마나 상대적으로 변하는지 보여 준다. 층마다 이 비가 크게 다르면 일부 층만 학습되고 있다.
- **활성값의 분포.** $\tanh$·sigmoid의 출력이 $\pm 1$이나 0·1에 몰려 있으면 포화, ReLU의 출력이 대부분 0이면 죽은 뉴런이 gradient를 막고 있다.

대응은 이 곱의 배율을 1 근처로 유지하는 설계들이다. 입력 폭에 맞춘 초기화 scale, ReLU 계열 활성화, 정규화 층, 잔차 연결(Jacobian이 $\mathbf{I} + \cdots$ 꼴이 되어 곱이 0으로 무너지지 않는다), 그리고 폭발에 대한 gradient clipping이다. 각각은 17.9장과 18장에서 다룬다.

## 실습

스칼라 자동미분 엔진(micrograd 수준)을 직접 구현하고, 수치미분·손 유도·PyTorch autograd 결과를 비교한다. 코드는 두 파일이다.

- `labs/calculus-and-autodiff/automatic-differentiation/engine.py`: 본문 엔진에 거듭제곱·$\exp$·$\log$·ReLU·뺄셈·나눗셈과 상수 연산을 더하고, 재귀 대신 반복문으로 위상 정렬하며, $\tanh$ 은닉층 MLP 클래스를 둔다. NumPy도 쓰지 않는다.
- `labs/calculus-and-autodiff/automatic-differentiation/micrograd_vs_torch.py`: 2차원 two-moons 데이터 40개에 MLP(2-8-8-1)를 logistic loss로 학습한다. (1) 같은 가중치에서 엔진·PyTorch·유한차분 gradient를 비교하고, (2) 엔진과 PyTorch로 같은 SGD를 200 step 돌린다. PyTorch가 필요하다(`--group torch`).

집필 시점의 실행 결과는 다음과 같다.

```output
[1] 같은 가중치에서 gradient 비교 (MLP 2-8-8-1, 표본 40개)
  손실: 엔진 0.499860334595   PyTorch 0.499860334595
  W1 (8, 2)    최대 |엔진 - PyTorch| = 1.4e-17
  b1 (8,)      최대 |엔진 - PyTorch| = 2.1e-17
  W2 (8, 8)    최대 |엔진 - PyTorch| = 1.4e-17
  b2 (8,)      최대 |엔진 - PyTorch| = 1.4e-17
  W3 (1, 8)    최대 |엔진 - PyTorch| = 1.4e-17
  b3 (1,)      최대 |엔진 - PyTorch| = 2.8e-17
  W1[0,0]  엔진 +0.01670315  유한차분 +0.01670315
  W1[3,1]  엔진 -0.02508607  유한차분 -0.02508607
  W1[7,0]  엔진 -0.00014220  유한차분 -0.00014220
[2] SGD 200 step (학습률 1.0): 엔진 vs PyTorch
  step   0  손실 엔진 0.499860  PyTorch 0.499860  정확도 0.725  파라미터 차이 0.0e+00
  step  50  손실 엔진 0.186170  PyTorch 0.186170  정확도 0.950  파라미터 차이 2.6e-12
  step 100  손실 엔진 0.092693  PyTorch 0.075429  정확도 0.975  파라미터 차이 6.9e-01
  step 150  손실 엔진 0.025266  PyTorch 0.021282  정확도 1.000  파라미터 차이 6.7e-01
  step 200  손실 엔진 0.009754  PyTorch 0.009409  정확도 1.000  파라미터 차이 7.3e-01
  학습 후 엔진의 가중치에서 gradient 최대 차이 3.5e-17
```

결과를 읽는 법은 세 가지다.

- **같은 가중치에서 gradient는 float64의 반올림 수준($10^{-17}$)까지 일치한다.** 엔진은 스칼라를 하나씩 더하고 PyTorch는 행렬 곱으로 더하므로 덧셈 순서가 달라 마지막 자리가 다를 뿐이다.
- **그런데 학습 궤적은 갈라진다.** 50 step까지 $10^{-12}$였던 파라미터 차이가 100 step에서는 0.7이 되었다. 학습률 1.0의 SGD가 이 구간에서 작은 차이를 크게 증폭시키는 동역학을 가졌기 때문이다. 두 구현 모두 문제를 풀었고(정확도 1.000), 어느 쪽도 틀리지 않았다.
- **그래서 gradient는 같은 점에서 비교한다.** 학습 후 엔진의 가중치를 PyTorch에 복사해 다시 비교하면 차이는 $3.5 \times 10^{-17}$이다. 학습 결과(손실 곡선, 최종 파라미터)가 같은지로 gradient 구현을 검증하면, 이번처럼 올바른 구현을 틀렸다고 판단하거나 반대로 작은 버그를 놓칠 수 있다.

이어서 두 연습문제를 푼다. 첫째는 본문의 엔진을 직접 다시 만드는 것이고, 둘째는 이 절의 검증 절차를 함수로 만드는 것이다.

::exercise{id="value-engine"}

::exercise{id="gradient-check"}

## 실패 사례

- **역전파를 학습 전체와 동일시하는 것.** 역전파는 gradient를 계산할 뿐이다. 학습이 안 될 때 gradient 계산(모델·손실 구현)과 업데이트(optimizer·학습률·스케줄)를 나눠 확인하지 않으면 원인을 찾지 못한다. `zero_grad()`를 빠뜨려 gradient가 step마다 누적되는 버그도 이 구분을 모를 때 생긴다.
- **float32로 유한차분 검증을 해 잘못된 결론을 내리는 것.** float32에서 $h = 10^{-6}$의 중앙차분은 올바른 gradient에도 $10^{-2}$ 수준의 상대오차를 낸다. 이것을 보고 구현을 고치기 시작하면 맞는 코드를 망가뜨린다. 검증은 float64로, $h$를 몇 자리 바꿔 가며 결과가 안정적인지 확인한다.
- **학습 결과가 같은지로 gradient 구현을 검증하는 것.** 실습처럼 올바른 두 구현의 학습 궤적도 갈라질 수 있고, 반대로 틀린 gradient로도 손실은 줄어들 수 있다. gradient는 같은 입력·같은 파라미터에서 직접 비교한다.
- **추론이나 기록 중에 그래프를 붙잡아 두는 것.** 손실 텐서를 그대로 리스트에 모으거나 평가를 `no_grad` 없이 실행하면 그래프와 중간값이 해제되지 않아 메모리가 계속 늘어난다.

## 통과 기준

- 자신이 만든 엔진으로 작은 MLP를 학습하고, 같은 가중치에서 gradient가 PyTorch autograd와 $10^{-10}$ 이하의 차이로 일치함을 확인한다.
- 입력·출력 차원이 주어진 함수에 대해 forward mode와 reverse mode 중 어느 쪽이 유리한지 Jacobian 곱의 순서로 설명한다.
- 연습문제 두 개(`value-engine`, `gradient-check`)의 테스트를 통과한다.
