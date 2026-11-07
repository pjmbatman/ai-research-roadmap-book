---
id: deep-learning-pytorch/tensors-and-autograd
title: "PyTorch 텐서와 autograd"
title_en: "PyTorch Tensors and Autograd"
tier: core
status: draft
prerequisites: [python-and-computing/scientific-python, calculus-and-autodiff/automatic-differentiation]
---

# PyTorch 텐서와 autograd

PyTorch 코드의 버그는 대부분 두 곳에서 나온다. 하나는 **텐서의 모양과 종류**다. shape이 broadcasting 규칙에 우연히 맞아 조용히 틀린 값을 내거나, float64 입력이 float32 모델에 들어가거나, GPU에 있어야 할 텐서가 CPU에 남는다. 다른 하나는 **autograd의 동작**이다. gradient가 누적되거나, 기록용 텐서가 그래프를 붙잡아 메모리가 새거나, in-place 수정이 backward에 필요한 값을 덮어쓴다.

5.3장에서 NumPy 배열·broadcasting을, 8.4장에서 reverse mode 자동미분의 원리와 PyTorch의 `backward()`·`.grad`를 보았다. 이 장은 그 두 지식을 PyTorch의 구체적인 동작으로 옮긴다. 목표는 임의의 코드를 보고 **각 텐서의 shape·dtype·device와, 그 텐서가 계산 그래프에 연결되는지**를 예측할 수 있게 되는 것이다.

## 학습 목표

- 텐서의 shape·dtype·device를 다룬다
- autograd의 동작을 정확히 이해한다

## 텐서

**정의.** PyTorch의 텐서(`torch.Tensor`)는 NumPy 배열처럼 같은 dtype의 원소를 담은 다차원 배열이다. NumPy 배열과 다른 점은 두 가지다. **device**(CPU, CUDA GPU 등)를 속성으로 가지고, 연산이 **autograd에 기록될 수 있다.** 텐서마다 다음 세 속성을 항상 의식한다.

- `shape`: 축의 길이. 이 책은 코드에 `# (B, T, D)` 같은 shape 주석을 단다.
- `dtype`: 원소 종류. 학습에는 보통 `float32`, 메모리와 속도를 위해 `bfloat16`·`float16`을 섞어 쓰고(17.5장의 mixed precision), 정수 레이블과 인덱스는 `int64`다.
- `device`: 텐서가 있는 장치. 한 연산의 피연산자는 모두 같은 device에 있어야 한다.

**생성과 기본 dtype.** dtype을 지정하지 않으면 PyTorch가 정한다. Python 실수는 기본 dtype(`float32`)이 되지만, NumPy에서 온 배열은 **원래 dtype을 유지한다.** NumPy의 기본 실수는 `float64`이므로, NumPy로 읽은 데이터를 그대로 넘기면 float64 텐서가 된다.

**NumPy와의 메모리 공유.** `torch.from_numpy(a)`와 `t.numpy()`(CPU 텐서)는 복사하지 않고 같은 메모리를 공유한다. `torch.tensor(a)`는 복사한다. 공유는 큰 데이터를 빠르게 넘기는 데 유용하지만, 한쪽의 제자리 수정이 다른 쪽에 보인다.

```python {exec="python"}
import numpy as np
import torch

print(torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2.0]).dtype,
      torch.from_numpy(np.zeros(2)).dtype)                 # 정수, Python 실수, NumPy float64
temp_c = np.array([20.0, 21.5, 19.0])                      # (3,) NumPy 배열
shared = torch.from_numpy(temp_c)                          # 메모리를 공유한다
copied = torch.tensor(temp_c)                              # 복사한다
shared[0] = -999.0
print("from_numpy 수정 후 원본:", temp_c)
copied[1] = -999.0
print("tensor() 수정 후 원본  :", temp_c)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.ones(2, 3, device=device)                        # (2, 3)
print(x.dtype, x.device.type, x.shape, x.to(torch.float16).dtype)
```

```output
torch.int64 torch.float32 torch.float64
from_numpy 수정 후 원본: [-999.    21.5   19. ]
tensor() 수정 후 원본  : [-999.    21.5   19. ]
torch.float32 cpu torch.Size([2, 3]) torch.float16
```

`shared`를 고치자 원본 `temp_c`가 바뀌었고, `copied`를 고친 것은 원본에 영향이 없다. 마지막 줄은 CPU에서 실행한 결과다. device를 코드에 하드코딩하지 않고 위처럼 한 번 정해 두면, GPU가 있는 환경에서는 같은 코드가 `cuda`를 쓴다. `.to(device)`나 `.to(dtype)`은 필요할 때만 복사한 새 텐서를 돌려준다.

::quiz{id="float64-input"}

:::note[분야 예시]
기상 재분석 자료나 분광 데이터를 NumPy·xarray로 읽으면 대개 float64다. float32 모델에 그대로 넣으면 dtype 오류가 나고, 반대로 모델 전체를 float64로 바꾸면 GPU에서 수 배 이상 느려질 수 있다. 데이터를 읽는 단계(17.4장의 Dataset)에서 dtype을 한 번 정해 두는 것이 안전하다. 반대로 정밀도가 결과를 좌우하는 계산(예: 작은 차이를 빼는 물리량, 8.4장의 gradient 검증)은 float64로 따로 확인한다.
:::

## shape 연산

**indexing과 broadcasting.** PyTorch의 indexing과 broadcasting 규칙은 NumPy와 같다(5.3장). 기본 slicing(`x[2:5]`, `x[:, 0]`)은 view를, 정수 배열·불리언 indexing(`x[idx]`, `x[mask]`)은 복사본을 돌려주는 것도 같다.

**view와 stride.** 텐서는 1차원 메모리(storage) 위에 shape과 **stride**를 얹은 것이다. stride는 각 축으로 한 칸 이동할 때 메모리에서 건너뛰는 원소 수다. `(2, 3)` 행 우선 텐서의 stride는 `(3, 1)`이다. `transpose`·`permute`·`t()`는 메모리를 옮기지 않고 stride만 바꾼 view를 만든다. 이렇게 만들어진 텐서는 메모리 순서가 행 우선 배치와 달라 **contiguous하지 않다.**

- `view(shape)`: 메모리를 그대로 두고 shape만 다시 해석한다. 현재 stride로 표현할 수 없으면 오류를 낸다. **결과가 원본과 메모리를 공유한다는 것이 보장된다.**
- `reshape(shape)`: 가능하면 view를, 불가능하면 복사본을 돌려준다. 결과가 원본과 공유하는지는 상황에 따라 다르다.
- `contiguous()`: contiguous하지 않으면 행 우선으로 복사한다.
- `permute(dims)`: 축 순서를 바꾼다. 이미지 텐서의 `(B, H, W, C)` ↔ `(B, C, H, W)` 변환이 대표적이다.

```python {exec="python"}
x = torch.arange(6.0).reshape(2, 3)                        # (2, 3)
xt = x.t()                                                 # (3, 2) 전치: 같은 메모리, stride만 바뀜
print("stride:", x.stride(), xt.stride(), " contiguous:", x.is_contiguous(), xt.is_contiguous())
print("전치가 같은 메모리인가:", xt.data_ptr() == x.data_ptr())
try:
    xt.view(6)
except RuntimeError as e:
    print("xt.view(6) -> RuntimeError:", "reshape" in str(e))
r = xt.reshape(6)                                          # 필요하면 복사한다
print("xt.reshape(6):", r.tolist(), " 복사했는가:", r.data_ptr() != x.data_ptr())
v = x.view(3, 2)                                           # contiguous면 view가 된다
v[0, 0] = 100.0
print("view를 고치면 원본도 바뀐다:", x[0, 0].item())
img = torch.zeros(8, 32, 32, 3)                            # (B, H, W, C)
print("permute:", tuple(img.permute(0, 3, 1, 2).shape))    # (B, C, H, W)
```

```output
stride: (3, 1) (1, 3)  contiguous: True False
전치가 같은 메모리인가: True
xt.view(6) -> RuntimeError: True
xt.reshape(6): [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]  복사했는가: True
view를 고치면 원본도 바뀐다: 100.0
permute: (8, 3, 32, 32)
```

전치한 `xt`를 펼치려면 원소를 `0, 3, 1, 4, 2, 5` 순서로 읽어야 하는데, 이 순서는 하나의 stride로 표현할 수 없다. 그래서 `view`는 실패하고 `reshape`는 복사한다. 오류 메시지가 `reshape`를 쓰라고 안내하는 것도 이 때문이다.

::quiz{id="view-after-transpose"}

**연구에서의 의미.** shape 버그는 오류보다 **조용한 틀림**이 더 위험하다. 예를 들어 `(B, T, D)`를 `(B, D, T)`로 바꿔야 할 때 `x.view(B, D, T)`를 쓰면 오류 없이 원소가 뒤섞인 텐서가 나온다. shape은 맞지만 의미가 틀린 것이다. 축의 **의미**를 바꾸는 것은 `permute`·`transpose`이고, `view`·`reshape`는 축을 나누거나 합칠 뿐이다. 이 구분을 지키고, 의심스러우면 작은 텐서(`torch.arange`)로 결과의 원소 순서를 직접 확인한다.

## autograd

**정의.** 8.4장에서 본 것처럼 `requires_grad=True`인 텐서에서 시작한 연산은 계산 그래프에 기록된다. PyTorch에서는 다음 용어를 쓴다.

- **잎(leaf) 텐서**: 사용자가 직접 만든 텐서. 모델 파라미터가 대표적이다. `backward()`는 잎 텐서 중 `requires_grad=True`인 것의 `.grad`만 채운다.
- **`grad_fn`**: 연산 결과 텐서가 가리키는, 자신을 만든 연산의 backward 노드. 그래프는 이 노드들의 연결이다.
- **`backward()`**: 스칼라에서 시작해 그래프를 거꾸로 따라가며 잎 텐서의 `.grad`에 gradient를 **더한다.**

**grad 누적과 broadcasting.** 두 가지 동작이 자주 혼동된다. 첫째, `.grad`는 덮어쓰지 않고 누적된다. 둘째, broadcasting으로 여러 번 쓰인 텐서의 gradient는 그 사용처들의 gradient의 **합**이다. 둘 다 8.4장의 adjoint 식에서 "여러 경로의 기여는 더한다"는 같은 원리다.

```python {exec="python"}
w = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)     # (3,) 잎(leaf) 텐서
X = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])      # (2, 3)
b = torch.zeros(3, requires_grad=True)                     # (3,) 두 행에 broadcast된다
loss = ((X * w + b) ** 2).sum()
print("grad_fn:", type(loss.grad_fn).__name__, " 잎인가:", w.is_leaf, (X * w).is_leaf)
loss.backward()
print("w.grad =", w.grad.tolist())
print("b.grad =", b.grad.tolist(), "= 2(Xw+b)를 batch 축으로 합한 값",
      (2 * (X * w + b)).sum(0).tolist())
loss2 = ((X * w + b) ** 2).sum()
loss2.backward()                                           # zero_grad 없이 한 번 더
print("두 번째 backward 후 w.grad =", w.grad.tolist(), "(두 배로 누적)")
```

```output
grad_fn: SumBackward0  잎인가: True False
w.grad = [2.0, -20.0, 10.0]
b.grad = [2.0, -12.0, 2.0] = 2(Xw+b)를 batch 축으로 합한 값 [2.0, -12.0, 2.0]
두 번째 backward 후 w.grad = [4.0, -40.0, 20.0] (두 배로 누적)
```

손으로 확인하면 $\partial L/\partial w_d = \sum_b 2(X_{bd}w_d + b_d)X_{bd}$이므로 $w_1$ 성분은 $2(-4)(2) + 2(-2)(1) = -20$이다. `b.grad`는 shape `(3,)`으로, 두 행의 gradient를 합한 값이다. 같은 손실로 한 번 더 `backward()`를 하자 `w.grad`가 정확히 두 배가 되었다.

::quiz{id="grad-accumulates"}

broadcasting의 gradient가 "broadcast된 축의 합"이라는 것은 NumPy만으로도 확인할 수 있다. 출력 쪽 gradient를 구한 뒤 broadcast된 축으로 합하고, 수치미분과 비교한다.

```python {exec="browser" session="tensors"}
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))                                # (B, D)
b = rng.normal(size=(1, 3))                                # (1, D) -> (B, D)로 broadcast
L = lambda b: np.sum(np.tanh(X + b) ** 2)                  # 스칼라 손실
dLdY = 2 * np.tanh(X + b) * (1 - np.tanh(X + b) ** 2)      # (B, D) 출력 쪽 gradient
grad_b = dLdY.sum(axis=0, keepdims=True)                   # (1, D) broadcast 축으로 합
num = np.zeros_like(b)
for j in range(3):                                         # 중앙차분으로 확인
    e = np.zeros_like(b); e[0, j] = 1e-6
    num[0, j] = (L(b + e) - L(b - e)) / 2e-6
print("합으로 줄인 gradient:", grad_b.round(6))
print("수치미분            :", num.round(6))
print("shape:", dLdY.shape, "->", grad_b.shape)
```

```output
합으로 줄인 gradient: [[-0.816216 -1.317264 -2.125962]]
수치미분            : [[-0.816216 -1.317264 -2.125962]]
shape: (4, 3) -> (1, 3)
```

::quiz{id="broadcast-grad-value"}

**연구에서의 의미.** 이 규칙은 gradient의 **크기**에 영향을 준다. bias가 batch 전체에 broadcast되므로, 손실을 batch에 대해 평균(`mean`)하느냐 합(`sum`)하느냐에 따라 gradient가 batch 크기 배만큼 달라진다. batch 크기를 바꿨는데 학습률을 그대로 두어 결과가 달라졌다면 손실의 집계 방식부터 확인한다(10.4장의 gradient accumulation과 같은 문제).

## 그래프 제어

그래프를 언제 만들고 언제 끊을지 정하는 도구는 네 가지다.

| 도구 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `t.detach()` | 같은 메모리를 공유하되 그래프에서 떨어진 텐서를 돌려준다 | 기록·시각화, target 네트워크처럼 gradient를 막아야 하는 값 |
| `torch.no_grad()` | 블록 안의 연산을 그래프에 기록하지 않는다 | 평가, 파라미터를 직접 고칠 때 |
| `torch.inference_mode()` | `no_grad`보다 강하다. 만든 텐서를 이후 autograd 연산에 쓸 수 없는 대신 버전 추적 등의 비용을 줄인다 | 학습과 섞이지 않는 순수 추론 |
| `t.clone()` | 새 메모리로 복사한다. 미분 가능한 연산이라 그래프에 남는다 | 원본을 보존한 채 수정할 때 |

`detach()`는 메모리를 공유하므로, detach한 텐서를 제자리에서 고치면 원본도 바뀐다. 독립된 사본이 필요하면 `t.detach().clone()`을 쓴다.

**in-place 연산의 위험.** 이름이 `_`로 끝나는 연산(`add_`, `mul_`, `zero_`)과 `x[...] = ...` 대입은 텐서를 제자리에서 바꾼다. 어떤 연산은 backward에 입력이나 출력 값을 저장해 둔다. sigmoid의 국소 미분 $s(1 - s)$에는 출력 $s$가 필요하다. 저장된 텐서를 제자리에서 바꾸면 backward가 틀린 값을 쓰게 되므로, PyTorch는 텐서마다 **버전 번호**(`_version`)를 두고 저장할 때의 버전과 backward 때의 버전이 다르면 오류를 낸다.

```python {exec="python"}
w = torch.randn(3, requires_grad=True)
y = (w * 2).sum()
print("detach:", y.detach().requires_grad, " no_grad 안:", end=" ")
with torch.no_grad():
    print((w * 2).sum().requires_grad)
with torch.inference_mode():
    z = w * 2
try:
    (z * w).sum().backward()
except RuntimeError as e:
    print("inference_mode 텐서를 학습에 쓰면 RuntimeError:", "Inference tensors" in str(e))

a = torch.randn(3, requires_grad=True)
s = torch.sigmoid(a)                                       # backward에 s 자신이 필요하다
print("s 버전:", s._version, end=" -> ")
s.mul_(2)                                                  # in-place 수정
print(s._version)
try:
    s.sum().backward()
except RuntimeError as e:
    print("backward -> RuntimeError:", "modified by an inplace operation" in str(e))
```

```output
detach: False  no_grad 안: False
inference_mode 텐서를 학습에 쓰면 RuntimeError: True
s 버전: 0 -> 1
backward -> RuntimeError: True
```

::quiz{id="inplace-saved-tensor"}

버전 검사는 autograd가 **아는** 수정만 잡는다. `t.data`를 통한 수정이나 `no_grad` 안의 수정은 버전 검사를 피해 가므로, 오류 없이 틀린 gradient가 나올 수 있다(실습에서 재현한다). 또한 view에 대한 in-place 수정은 원본을 바꾼다. 데이터셋 텐서를 slicing한 batch를 제자리에서 정규화하면, 오류 없이 **데이터셋 자체가** 바뀐다. 메모리를 아끼려는 in-place 연산은 프로파일링으로 이득이 확인된 곳에만 쓴다.

::quiz{id="graph-connected"}

## 함수형 도구

**왜 필요한가.** `backward()`는 "스칼라 손실 하나 → 모든 파라미터의 gradient"라는 학습의 기본 경우에 맞춰져 있다. 연구에서는 다른 형태의 미분이 자주 필요하다. 표본마다 따로 구한 gradient(per-sample gradient, 차등 프라이버시·영향 함수·gradient 분산 분석), 입력에 대한 Jacobian(민감도 분석), Hessian-vector product(10.6장) 등이다. 이를 `backward()`와 반복문으로 구현하면 느리고 복잡하다.

**정의.** `torch.func`(집필 시점 예시, PyTorch 2.x)는 JAX와 비슷한 함수 변환을 제공한다. 함수를 받아 새 함수를 돌려주는 방식이다.

- `grad(f)`: 스칼라 함수 `f`의 gradient를 계산하는 함수. 첫 번째 인자에 대해 미분한다.
- `vmap(f, in_dims)`: 표본 하나에 대해 쓴 함수를 batch 축에 대해 벡터화한다. `in_dims`로 어느 인자의 어느 축이 batch인지 지정한다(`None`은 batch가 없는 인자).
- `jacrev(f)`, `jacfwd(f)`: reverse mode·forward mode로 Jacobian을 계산하는 함수(8.4장의 VJP·JVP).

```python {exec="python"}
from torch.func import grad, jacrev, vmap

torch.manual_seed(0)
w = torch.randn(3)                                         # (D,)
X, y = torch.randn(5, 3), torch.randn(5)                   # (N, D), (N,)
def loss_one(w, x, y):                                     # 표본 하나의 손실
    return (x @ w - y) ** 2

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(w, X, y)   # (N, D)
print("표본별 gradient shape:", tuple(per_sample.shape))
manual = 2 * (X @ w - y)[:, None] * X                      # (N, D) 손 유도
print("손 유도와 같은가:", torch.allclose(per_sample, manual))
f = lambda x: torch.tanh(x) * x.sum()                      # R^3 -> R^3
x0 = torch.tensor([0.1, -0.2, 0.3])
J = jacrev(f)(x0)                                          # (출력 3, 입력 3)
J_manual = torch.diag(1 - torch.tanh(x0) ** 2) * x0.sum() + torch.tanh(x0)[:, None]
print("Jacobian shape:", tuple(J.shape), " 손 유도와 같은가:", torch.allclose(J, J_manual))
```

```output
표본별 gradient shape: (5, 3)
손 유도와 같은가: True
Jacobian shape: (3, 3)  손 유도와 같은가: True
```

`grad(loss_one)`은 표본 하나의 gradient를 내는 함수이고, `vmap`이 이를 5개 표본에 대해 한 번에 실행해 `(N, D)` 텐서를 만든다. `backward()`로는 표본별 gradient의 합만 얻을 수 있다는 점과 대조된다. Jacobian은 이 책의 규약대로 (출력 차원 × 입력 차원)이다.

**연구에서의 의미.** `torch.func`의 함수 변환은 순수 함수(입력이 같으면 출력이 같고, 바깥 상태를 바꾸지 않는 함수)를 가정한다. `nn.Module`은 파라미터를 내부 상태로 가지므로, 함수 변환을 쓰려면 파라미터를 인자로 넘기는 형태로 바꿔야 한다(집필 시점 예시: `torch.func.functional_call`). 지원되는 연산과 제약은 버전마다 달라지므로 공식 문서에서 확인한다.

:::note[분야 예시]
신소재 탐색에서 물성 예측 모델의 입력(조성·공정 조건)에 대한 Jacobian은 "어느 조건을 조금 바꾸면 물성이 얼마나 바뀌는가"를 알려 주는 민감도 지도다. 입력이 수십 차원이고 출력이 몇 개이면 `jacrev`, 반대로 입력이 적고 출력(예: 스펙트럼 수백 점)이 많으면 `jacfwd`가 유리하다. 8.4장의 forward·reverse mode 선택 기준이 그대로 적용된다.
:::

## 실습

in-place 연산·detach 누락·grad 누적으로 생기는 버그를 재현하고 수정한다. 코드는 `labs/deep-learning-pytorch/tensors-and-autograd/autograd_bugs.py`다. 버그마다 버그 버전과 수정 버전을 나란히 실행한다.

1. **detach 누락.** 평가 루프에서 손실을 `total = total + loss`로 합산한다. 살아 있는 그래프의 크기를 재기 위해 `torch.autograd.graph.saved_tensors_hooks`로 backward용으로 저장된 텐서를 추적한다.
2. **in-place 수정.** (a) sigmoid 출력을 `mul_`로 수정, (b) 데이터셋 텐서를 slicing한 batch를 `-=`·`/=`로 정규화, (c) `s.data.mul_(2)`로 autograd 몰래 수정.
3. **grad 누적.** 선형회귀를 SGD로 학습하며 `zero_grad()`를 빠뜨린다.

집필 시점의 실행 결과는 다음과 같다(PyTorch 2.15, CPU).

```output
[1] 평가 손실 합산과 그래프 메모리 (batch 50개 뒤 살아 있는 backward용 저장 텐서)
  total = total + loss            :   53.3 MiB
  total = total + loss.item()     :    1.1 MiB
  torch.no_grad() + loss.item()   :    0.0 MiB
[2] in-place 수정
  sigmoid 출력을 mul_로 수정 후 backward : RuntimeError (one of the variables needed for gradient computation has been modified by an inplace operation)
  view를 제자리 정규화: 데이터셋 평균 10.060 -> 0.000  (오류 없이 원본이 바뀜)
  새 텐서로 정규화    : 데이터셋 평균 10.060 -> 10.060
  s.data.mul_(2) 후 a.grad : [-0.3049  0.2486 -1.3416] (틀림, 오류 없음)
  s = s * 2      후 a.grad : [0.47   0.3932 0.21  ] (맞음)
[3] zero_grad 누락 (선형회귀, SGD lr=0.1)
  step  0  zero_grad 있음     7.5033   없음     7.5033
  step  5  zero_grad 있음     1.4047   없음     2.6190
  step 10  zero_grad 있음     0.2731   없음     2.4388
  step 20  zero_grad 있음     0.0109   없음     1.3170
  step 29  zero_grad 있음     0.0006   없음     1.6455
```

결과를 읽는 법은 다음과 같다.

- **[1]** 손실을 텐서째 더하면 `total`이 매 batch의 그래프를 붙잡아, 50 batch 동안 53 MiB가 쌓였다. 모델이 크고 평가 세트가 길면 GPU 메모리 부족으로 이어진다. `.item()`으로 숫자만 꺼내면 마지막 batch의 그래프(1.1 MiB, 변수 `loss`가 아직 가리키고 있다)만 남고, `no_grad` 안에서 평가하면 그래프 자체가 만들어지지 않는다.
- **[2]** 세 경우 중 오류가 나는 것은 (a)뿐이다. (b)는 view를 통해 데이터셋의 평균이 10.06에서 0으로 바뀌었고, 두 번째 epoch부터는 이미 정규화된 데이터를 다시 정규화한다. (c)는 sigmoid의 backward가 $s(1-s)$ 대신 $2s(1-2s)$를 계산해 부호까지 틀린 gradient를 냈다. 올바른 값은 $2s(1-s)$, 예를 들어 $a = 0.5$에서 0.47이다.
- **[3]** `zero_grad()`가 없으면 step $k$의 갱신에 지난 모든 gradient의 합이 쓰인다. 감쇠 없이 과거 gradient를 전부 더하는, momentum 계수 $\beta = 1$인 momentum(10.3장)과 같은 갱신이다. 갱신의 크기가 계속 커지고 지난 방향이 남아, 손실이 줄었다 늘었다 하며 30 step 뒤에도 1.6에 머문다. 오류는 나지 않는다. 17.5장의 training-loop-stepper 위젯에서 같은 현상을 단계별로 볼 수 있다.

이어서 두 연습문제를 푼다. 첫째는 autograd가 broadcasting에 대해 자동으로 하는 일을 직접 구현하는 것이고, 둘째는 stride만으로 view와 contiguous를 판정하는 것이다.

::exercise{id="unbroadcast-grad"}

::exercise{id="strides-and-views"}

## 실패 사례

- **detach 없이 손실 기록용 텐서를 쌓아 메모리가 새는 것.** `losses.append(loss)`, `total += loss`, 평가를 `no_grad` 없이 실행하는 것이 모두 같은 문제다. 기록에는 `loss.item()`(또는 `loss.detach()`)을 쓰고, 평가는 `torch.no_grad()`나 `torch.inference_mode()` 안에서 한다. 메모리가 epoch마다 조금씩 늘면 이것부터 의심한다.
- **view 이후 in-place 수정으로 gradient나 데이터가 깨지는 것.** 버전 검사에 걸리면 오류라도 나지만, `.data` 수정·`no_grad` 안의 수정·view를 통한 원본 수정은 조용히 틀린 결과를 낸다. 데이터 전처리에서는 `x = x - mean`처럼 새 텐서를 만든다.
- **`view`로 축의 의미를 바꾸는 것.** `(B, T, D)`를 `(B, D, T)`로 만들 때 `view`를 쓰면 shape만 맞고 원소가 뒤섞인다. 축 순서는 `permute`·`transpose`로 바꾼다.
- **dtype·device를 암묵적으로 맡기는 것.** NumPy에서 온 float64 텐서, CPU에 남은 텐서는 오류를 내거나(다행인 경우) 계산 비용을 몇 배로 늘린다(조용한 경우). 데이터 로딩 단계에서 dtype을, 학습 루프의 한 곳에서 device를 정한다.

## 통과 기준

- 주어진 코드에서 각 텐서의 shape·dtype·device와 `requires_grad` 여부(그래프에 연결되는지)를 실행 전에 예측하고, 실행해서 맞는지 확인한다.
- 실습의 세 버그(detach 누락, in-place 수정, grad 누적)를 각각 재현하고, 수정 전후의 메모리·gradient·손실 차이로 수정이 맞음을 보인다.
- 연습문제 두 개(`unbroadcast-grad`, `strides-and-views`)의 테스트를 통과한다.

## 참고 문헌

- Paszke 외 (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 32*. — PyTorch의 설계(eager 실행, 동적 계산 그래프, 텐서 메모리 관리)
