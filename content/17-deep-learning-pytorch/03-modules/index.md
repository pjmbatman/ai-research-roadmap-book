---
id: deep-learning-pytorch/modules
title: "nn.Module과 모델 구성"
title_en: "nn.Module and Model Composition"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/tensors-and-autograd]
---

# nn.Module과 모델 구성

17.2장의 텐서와 autograd만으로도 신경망을 학습할 수 있다. 파라미터 텐서를 직접 만들고, 리스트에 모아 optimizer에 넘기고, 저장할 때 하나씩 꺼내면 된다. 그러나 층이 수십 개가 되면 "어떤 텐서가 파라미터인가", "무엇을 저장해야 모델을 복원할 수 있는가", "GPU로 옮길 때 빠진 것은 없는가"를 사람이 관리하기 어렵다. `nn.Module`은 이 장부를 대신 관리하는 컨테이너다.

이 장은 `nn.Module`이 **무엇을 자동으로 추적하고, 무엇은 추적하지 않는가**를 다룬다. 모듈 등록의 규칙을 모르면 층이 optimizer에서 빠지거나, 정규화 통계가 체크포인트에서 사라지거나, 다른 코드의 체크포인트가 오류 없이 절반만 로드되는 일이 생긴다. 모두 오류 메시지 없이 성능만 나빠지는 버그다.

## 학습 목표

- Module로 모델을 구성한다
- parameter와 buffer를 구분한다
- state_dict를 다룬다

## nn.Module

**정의.** `nn.Module`을 상속한 클래스는 두 메서드를 쓴다.

- `__init__`: 하위 모듈과 파라미터를 만들어 **속성으로 대입한다.** 맨 앞에서 `super().__init__()`을 호출해야 등록 장치가 준비된다.
- `forward(x)`: 입력에서 출력을 계산한다. 텐서 연산과 하위 모듈 호출로 이루어지며, autograd가 이 계산을 기록한다.

모델은 `model(x)`로 호출한다. `nn.Module.__call__`이 등록된 hook을 실행하고 그 사이에 `forward`를 호출하므로, `model.forward(x)`를 직접 부르면 hook이 건너뛰어진다(이 장의 hook 절).

```python {exec="python"}
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                                 # 등록 장치를 먼저 만든다
        self.fc1 = nn.Linear(d_in, d_hidden)               # 속성 대입 = 하위 모듈 등록
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                                  # x: (B, d_in)
        return self.fc2(self.act(self.fc1(x)))             # (B, d_out)

torch.manual_seed(0)
model = MLP(4, 16, 2)
for name, p in model.named_parameters():
    print(f"{name:11s} {tuple(p.shape)}")
print("파라미터 수:", sum(p.numel() for p in model.parameters()))
print("출력 shape:", tuple(model(torch.randn(8, 4)).shape))
```

```output
fc1.weight  (16, 4)
fc1.bias    (16,)
fc2.weight  (2, 16)
fc2.bias    (2,)
파라미터 수: 114
출력 shape: (8, 2)
```

`named_parameters()`는 하위 모듈을 재귀적으로 따라가며 `속성이름.파라미터이름` 꼴의 점 경로를 만든다. 이 이름이 state_dict의 키가 되고, 체크포인트 호환성의 기준이 된다. `nn.Linear(4, 16)`의 가중치 shape이 `(16, 4)`, 즉 (출력 × 입력)이라는 점도 확인해 둔다. 층은 $\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$를 계산한다.

`nn.Module`이 이렇게 등록된 하위 모듈과 파라미터를 바탕으로 해 주는 일은 다음과 같다.

| 메서드 | 하는 일 |
|---|---|
| `parameters()`, `named_parameters()` | 학습할 파라미터 목록. optimizer에 넘긴다 |
| `to(device)`, `to(dtype)`, `double()` | 등록된 파라미터·buffer를 모두 옮기거나 변환한다 |
| `train()`, `eval()` | 하위 모듈 전체의 `training` 플래그를 바꾼다(BatchNorm·dropout의 동작이 바뀐다, 17.1장) |
| `state_dict()`, `load_state_dict()` | 저장·복원할 상태의 dict |
| `zero_grad()` | 모든 파라미터의 `.grad`를 비운다 |

::quiz{id="call-not-forward"}

## parameter와 buffer

**왜 필요한가.** 모델의 상태에는 두 종류가 있다. 하나는 gradient로 학습되는 값(가중치)이고, 다른 하나는 **학습되지 않지만 모델과 함께 저장되고 옮겨져야 하는 값**이다. BatchNorm의 running mean·var(17.1장), 입력 표준화에 쓰는 학습 데이터의 평균·표준편차, 위치 인코딩 표가 뒤의 예다. 이런 값을 일반 텐서 속성으로 두면 `.to(device)`도 `state_dict()`도 그 값을 모른다.

**정의.**

- **parameter**: `nn.Parameter`로 감싼 텐서. `requires_grad=True`가 기본이고, `parameters()`에 나타나 optimizer가 갱신한다.
- **buffer**: `self.register_buffer(name, tensor)`로 등록한 텐서. `parameters()`에는 나타나지 않지만 `.to()`가 함께 옮기고 `state_dict()`에 저장된다. `persistent=False`로 등록하면 옮기기는 하되 저장하지 않는다(다시 계산할 수 있는 캐시용).

```python {exec="python"}
class Standardize(nn.Module):
    """학습 데이터의 평균·표준편차로 입력을 표준화한다. 통계는 학습하지 않는다."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("cache", torch.zeros(1), persistent=False)   # 저장하지 않는 buffer
        self.scale = nn.Parameter(torch.ones(len(mean)))                  # 학습되는 값

    def forward(self, x):
        return (x - self.mean) / self.std * self.scale

norm = Standardize(mean=[300.0, 1.0e5], std=[10.0, 2.0e3])   # 예: 온도(K), 압력(Pa)
print("parameters:", [n for n, _ in norm.named_parameters()])
print("buffers   :", [n for n, _ in norm.named_buffers()])
print("state_dict:", list(norm.state_dict().keys()))
print("requires_grad  mean:", norm.mean.requires_grad, " scale:", norm.scale.requires_grad)
print(".double() 뒤 buffer dtype:", norm.double().mean.dtype)
```

```output
parameters: ['scale']
buffers   : ['mean', 'std', 'cache']
state_dict: ['scale', 'mean', 'std']
requires_grad  mean: False  scale: True
.double() 뒤 buffer dtype: torch.float64
```

`cache`는 buffer 목록에는 있지만 `persistent=False`라 state_dict에서 빠졌다. `.double()`은 parameter와 buffer를 함께 변환했다.

::quiz{id="batchnorm-buffers"}

:::note[분야 예시]
기후·대기 모델의 입력은 온도(K), 기압(Pa), 습도(kg/kg)처럼 단위와 크기가 몇 자릿수씩 다르다. 학습 데이터에서 계산한 변수별 평균·표준편차를 buffer로 모델 안에 넣어 두면, 체크포인트 하나로 "어떤 표준화로 학습된 모델인가"까지 복원된다. 표준화 통계를 별도 파일로 관리하다 다른 버전의 통계를 쓰는 사고를 막는다. 이 통계는 **학습 데이터에서만** 계산해야 한다는 점은 15.4장(데이터 분할과 누수)의 원칙 그대로다.
:::

::quiz{id="plain-tensor-attribute"}

## 컨테이너

**왜 필요한가.** 층의 수가 설정에 따라 바뀌면 `self.fc1`, `self.fc2`처럼 하나씩 쓸 수 없고 반복문으로 만들어야 한다. 이때 Python의 `list`·`dict`에 담으면 등록이 일어나지 않는다. 그 이유는 등록이 일어나는 방식에 있다. `nn.Module`은 `__setattr__`를 가로채, 대입되는 값이 Module이나 Parameter일 때만 내부 표에 기록한다. 그 장치를 순수 Python으로 흉내 내면 다음과 같다.

```python {exec="browser" session="modules"}
class Param:
    def __init__(self, n): self.n = n                      # 파라미터 n개라고 하자

class Module:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
    def __setattr__(self, name, value):                    # 속성 대입을 가로챈다
        if isinstance(value, Param): self._params[name] = value
        elif isinstance(value, Module): self._modules[name] = value
        object.__setattr__(self, name, value)
    def named_parameters(self, prefix=""):
        for name, p in self._params.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

class Linear(Module):
    def __init__(self, i, o):
        super().__init__(); self.weight, self.bias = Param(i * o), Param(o)

class Net(Module):
    def __init__(self):
        super().__init__()
        self.head = Linear(8, 1)
        self.blocks = [Linear(8, 8), Linear(8, 8)]         # 일반 list: Module도 Param도 아니다

print([(name, p.n) for name, p in Net().named_parameters()])
```

```output
[('head.weight', 8), ('head.bias', 1)]
```

`blocks`의 두 층은 list이므로 `__setattr__`의 두 조건에 걸리지 않아 아무 표에도 기록되지 않는다. 실제 PyTorch에서도 같은 일이 일어난다. 그래서 PyTorch는 등록을 대신해 주는 컨테이너를 제공한다.

- `nn.Sequential(*modules)`: 하위 모듈을 순서대로 호출하는 Module. 이름은 `0`, `1`, `2`, ...
- `nn.ModuleList(modules)`: 인덱스로 접근하는 list. `forward`는 직접 쓴다(반복문, 건너뛰기 연결 등).
- `nn.ModuleDict(dict)`: 이름으로 접근하는 dict. 과제별 head, 분기 구조에 쓴다.
- `nn.ParameterList`, `nn.ParameterDict`: Module이 아니라 Parameter를 담는 같은 용도의 컨테이너.

```python {exec="python"}
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(8, 8) for _ in range(3)]  # 등록되지 않는다

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(3))

for cls in (Bad, Good):
    m = cls()
    opt = torch.optim.SGD(m.parameters(), lr=0.1) if list(m.parameters()) else None
    print(f"{cls.__name__:4s} parameters {len(list(m.parameters()))}개, "
          f"state_dict 키 {len(m.state_dict())}개, optimizer 생성 {'가능' if opt else '불가'}")
blocks = nn.ModuleDict({"enc": nn.Linear(4, 8), "dec": nn.Linear(8, 4)})
print("ModuleDict 키:", list(blocks.state_dict().keys()))
```

```output
Bad  parameters 0개, state_dict 키 0개, optimizer 생성 불가
Good parameters 6개, state_dict 키 6개, optimizer 생성 가능
ModuleDict 키: ['enc.weight', 'enc.bias', 'dec.weight', 'dec.bias']
```

여기서는 모든 층이 list에 있어 파라미터가 0개이므로 optimizer를 만드는 단계에서 바로 드러난다(빈 파라미터 목록은 오류다). 실제 버그는 대개 일부 층만 list에 있어 optimizer가 오류 없이 만들어지고, 그 층들만 초기값에 머무는 형태다.

::quiz{id="list-not-registered"}

## state_dict

**정의.** `state_dict()`는 등록된 parameter와 persistent buffer를 `{점 경로 이름: 텐서}`로 모은 순서 있는 dict다. 모델을 저장한다는 것은 보통 이 dict를 저장하는 것이다. 모델 객체 전체를 pickle로 저장하는 방법도 있지만, 클래스 정의의 위치와 코드 버전에 묶여 나중에 불러오지 못할 수 있다. **설정(어떤 모델을 만들지) + state_dict(그 모델의 값)**로 저장하는 것이 안전하다.

**로드와 `strict`.** `load_state_dict(sd, strict=True)`(기본값)는 키 집합이 정확히 같아야 하고, shape이 다르면 오류를 낸다. `strict=False`는 이름이 맞는 것만 불러오고 나머지는 건너뛰며, 건너뛴 목록을 `missing_keys`(모델에 있는데 dict에 없음)·`unexpected_keys`(dict에 있는데 모델에 없음)로 돌려준다.

```python {exec="python"}
import io

torch.manual_seed(0)
src = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
buf = io.BytesIO()
torch.save(src.state_dict(), buf)                          # 파일 대신 메모리 버퍼
buf.seek(0)
sd = torch.load(buf, weights_only=True)                    # 텐서만 읽는다
print("키:", list(sd.keys()))

dst = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
print("strict 로드:", dst.load_state_dict(sd))
x = torch.randn(5, 4)
print("eval 출력이 같은가:", torch.equal(src.eval()(x), dst.eval()(x)))

wrapped = {"module." + k: v for k, v in sd.items()}       # 다른 코드가 붙인 접두사
try:
    dst.load_state_dict(wrapped)
except RuntimeError as e:
    print("접두사가 다르면 strict 로드 RuntimeError:", "Missing key" in str(e))
res = dst.load_state_dict(wrapped, strict=False)          # 오류 없이 아무것도 로드하지 않는다
print("strict=False  missing", len(res.missing_keys), " unexpected", len(res.unexpected_keys))
```

```output
키: ['0.weight', '0.bias', '1.weight', '1.bias', '1.running_mean', '1.running_var', '1.num_batches_tracked', '2.weight', '2.bias']
strict 로드: <All keys matched successfully>
eval 출력이 같은가: True
접두사가 다르면 strict 로드 RuntimeError: True
strict=False  missing 8  unexpected 9
```

state_dict에는 BatchNorm의 buffer 세 개가 들어 있고, 같은 구조의 새 모델에 불러오면 eval 출력이 비트 단위로 같다. 키에 `module.` 접두사가 붙은 dict는 `strict=True`에서 오류를 내지만, `strict=False`에서는 **오류 없이 아무것도 로드하지 않는다.** unexpected가 9개인데 missing이 8개인 것은 집필 시점 PyTorch의 BatchNorm이 예전 체크포인트와의 호환을 위해 `num_batches_tracked`가 없으면 기본값으로 채우기 때문이다. 이런 세부는 버전마다 달라질 수 있으므로, 로드 결과는 항상 반환값으로 확인한다.

`torch.load`의 `weights_only=True`는 텐서와 기본 자료형만 읽고 임의의 Python 객체는 거부한다. pickle은 불러오는 순간 코드를 실행할 수 있으므로, 출처를 확인하지 않은 체크포인트는 이 옵션으로 읽는다(집필 시점 예시로 PyTorch 2.15에서는 기본값이 이미 `True`다. 사용 중인 버전의 문서에서 확인한다).

**이름 불일치 처리.** 다른 코드의 체크포인트를 쓸 때는 다음 순서를 따른다.

1. 키 목록을 출력해 모델의 키와 비교한다(missing, unexpected, shape이 다른 키).
2. 접두사(`module.`, `model.`)나 바뀐 층 이름은 **키를 바꾼 새 dict**를 만들어 맞춘다. 모델 코드를 체크포인트에 맞춰 고치지 않는다.
3. 출력 층처럼 의도적으로 shape이 다른 부분은 dict에서 빼고, `strict=False`의 반환값이 정확히 그 키들만 missing으로 보고하는지 확인한다.
4. 로드 후 알려진 입력에 대한 출력이나 평가 지표로 복원이 맞는지 확인한다.

::quiz{id="strict-false-silent"}

::quiz{id="state-dict-key-count"}

## hook

**왜 필요한가.** 학습이 잘 되지 않을 때 중간 층의 활성값과 gradient를 봐야 한다(17.1장의 층별 통계). `forward`를 고쳐 중간값을 반환하게 하면 모델 코드가 진단 코드로 오염되고, 남이 만든 모델에는 쓸 수도 없다. hook은 모델 코드를 건드리지 않고 특정 모듈의 입출력을 관찰하는 장치다.

**정의.** 집필 시점 PyTorch의 주요 hook은 다음과 같다.

- `module.register_forward_hook(fn)`: forward 직후 `fn(module, inputs, output)`을 호출한다.
- `module.register_forward_pre_hook(fn)`: forward 직전 `fn(module, inputs)`를 호출한다.
- `module.register_full_backward_hook(fn)`: backward 중 이 모듈의 gradient가 계산되면 `fn(module, grad_input, grad_output)`을 호출한다. `grad_output`은 모듈 출력에 대한 손실의 gradient다.
- `tensor.register_hook(fn)`: 텐서 하나의 gradient가 계산될 때 호출한다.

모든 등록 함수는 handle을 돌려주고, `handle.remove()`로 제거한다.

```python {exec="python"}
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
records = {}
def save_stats(name):
    def hook(module, inputs, output):                      # forward 직후 호출된다
        records[name] = f"shape {tuple(output.shape)}, std {output.std().item():.4f}"
    return hook
def save_grad(name):
    def hook(module, grad_input, grad_output):             # backward 중 호출된다
        records[name] = f"|dL/d출력| {grad_output[0].norm().item():.4f}"
    return hook
handles = [net[i].register_forward_hook(save_stats(f"tanh{i}")) for i in (1, 3)]
handles += [net[i].register_full_backward_hook(save_grad(f"linear{i}")) for i in (2, 4)]
net(torch.randn(64, 4)).pow(2).mean().backward()
for k, v in records.items():                               # 호출된 순서대로
    print(f"{k:8s}", v)
for h in handles:
    h.remove()                                             # 다 쓰면 반드시 제거한다
```

```output
tanh1    shape (64, 32), std 0.5127
tanh3    shape (64, 32), std 0.2914
linear4  |dL/d출력| 0.0185
linear2  |dL/d출력| 0.0098
```

forward hook은 입력에서 출력 쪽으로, backward hook은 출력에서 입력 쪽으로 호출되었다. 두 번째 tanh의 출력이 첫 번째보다 작고, gradient도 아래층으로 갈수록 줄어드는 것이 보인다. 17.1장 실습의 층별 통계가 이 방법으로 기록되었다.

**연구에서의 의미.** hook은 강력한 만큼 부작용이 있다. 제거하지 않은 hook은 모든 forward마다 실행되어 학습을 느리게 하고, hook 안에서 출력 텐서를 저장하면 17.2장의 메모리 누수와 같은 문제가 생긴다. 기록할 때는 `.detach()`하거나 통계값(`.item()`)만 남기고, 필요한 구간이 끝나면 handle을 제거한다. hook 안에서 출력을 **수정**하는 것(반환값으로 출력을 바꾸는 기능)은 디버깅용 개입 실험 외에는 피한다. 모델의 동작이 코드에서 보이지 않게 되기 때문이다.

## 설정에서 모델 만들기

**왜 필요한가.** 실험마다 모델 코드를 고치면, 어떤 결과가 어떤 모델에서 나왔는지 코드의 이력을 뒤져야 알 수 있다. 모델 구조를 **설정 파일**(TOML·YAML)로 기술하고 코드가 설정에서 모델을 만들게 하면, 실험 간 차이가 설정 파일의 diff로 남고, 체크포인트에 설정을 함께 저장해 모델을 정확히 다시 만들 수 있다.

**registry 패턴.** 설정의 `type = "mlp"` 같은 문자열을 실제 클래스로 바꾸는 표가 registry다. 클래스를 정의할 때 decorator로 이름을 등록해 두고, `build(cfg)`가 `type`으로 클래스를 찾아 나머지 키를 생성자 인자로 넘긴다.

```toml
# labs/deep-learning-pytorch/modules/configs/mlp.toml (일부)
[model]
type = "mlp"
d_in = 3
d_out = 1
hidden = [32, 32]
activation = "silu"
norm = "layernorm"
```

```python
REGISTRY = {}

def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

@register("mlp")
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden, activation="relu", norm="none", dropout=0.0): ...

def build(cfg):
    cfg = dict(cfg)
    cls = REGISTRY[cfg.pop("type")]
    return cls(**cfg)
```

설계할 때 지킬 점은 세 가지다. 첫째, 등록되지 않은 `type`에는 등록된 이름 목록을 담은 오류를 내 오타를 바로 찾게 한다. 둘째, 설정에 없는 인자의 기본값은 코드에 있으므로, 기본값을 바꾸면 예전 설정의 의미가 바뀐다. 체크포인트에는 기본값까지 채운 **완전한 설정**을 저장하는 것이 안전하다. 셋째, registry는 편리한 만큼 코드 추적을 어렵게 한다. 모델 종류가 몇 개뿐이면 `if`문 몇 줄이 더 읽기 쉽다. 설정 파일의 형식·조합·override·기록의 일반 원칙은 6.3장에서 다뤘다.

## 실습

설정 파일로 구성되는 모델을 만들고, hook으로 중간 활성값을 기록하고, state_dict로 정확히 복원한다. 코드는 `labs/deep-learning-pytorch/modules/config_model.py`, 설정은 `configs/mlp.toml`이다. 모델은 입력 표준화(buffer) + MLP(ModuleList, LayerNorm, SiLU, dropout)이고, registry로 설정에서 만든다. 스크립트는 (1) 모든 파라미터와 buffer를 나열하고, (2) 활성화 층마다 forward hook을 걸어 출력 통계를 기록한 뒤 hook을 제거하고, (3) 설정과 state_dict를 함께 저장했다가 **다른 시드로 초기화한** 새 모델에 불러와 출력이 같은지 확인하고, (4) 등록되지 않은 type의 오류 메시지를 보인다. 집필 시점의 실행 결과는 다음과 같다(PyTorch 2.15, CPU).

```output
[1] 파라미터
  net.blocks.0.0.weight        (32, 3)    requires_grad=True
  net.blocks.0.0.bias          (32,)      requires_grad=True
  net.blocks.0.1.weight        (32,)      requires_grad=True
  net.blocks.0.1.bias          (32,)      requires_grad=True
  net.blocks.1.0.weight        (32, 32)   requires_grad=True
  net.blocks.1.0.bias          (32,)      requires_grad=True
  net.blocks.1.1.weight        (32,)      requires_grad=True
  net.blocks.1.1.bias          (32,)      requires_grad=True
  net.head.weight              (1, 32)    requires_grad=True
  net.head.bias                (1,)       requires_grad=True
[1] buffer
  input_norm.mean              (3,)       [300.0, 100000.0, 0.5]
  input_norm.std               (3,)       [10.0, 2000.0, 0.2]
  파라미터 합계 1345개, state_dict 키 12개
[2] 활성화 층별 출력 (forward hook)
  net.blocks.0.2         평균 +0.213  표준편차 0.564
  net.blocks.1.2         평균 +0.206  표준편차 0.610
  남은 hook 수: 0
[3] state_dict 복원
  load_state_dict: <All keys matched successfully>
  설정 + state_dict로 다시 만든 모델의 출력이 비트 단위로 같은가: True
[4] 등록되지 않은 type: "알 수 없는 type 'transformer'. 등록된 이름: ['mlp', 'standardize']"
```

파라미터 10개 텐서와 buffer 2개를 합친 12개가 state_dict의 키다. dropout과 활성화 층은 상태가 없으므로 이름에는 번호(`blocks.0.2`)만 차지하고 state_dict에는 나타나지 않는다. 이름의 `blocks.0.1`이 LayerNorm이라는 것은 이름만으로는 알 수 없다. 체크포인트를 다른 사람이 읽어야 한다면, `nn.Sequential`의 번호 대신 의미 있는 이름(`ModuleDict` 또는 속성)을 쓰는 편이 낫다.

이어서 두 연습문제를 푼다. 첫째는 실습의 registry를 중첩 설정까지 다루도록 일반화하는 것이고, 둘째는 체크포인트를 불러오기 전에 이름과 shape의 불일치를 진단하는 것이다.

::exercise{id="config-registry"}

::exercise{id="state-dict-diff"}

## 실패 사례

- **일반 list에 넣은 층이 optimizer에 등록되지 않는 것.** 오류 없이 그 층만 초기값에 머문다. 모델을 만든 직후 `sum(p.numel() for p in model.parameters())`를 손으로 계산한 값과 비교하는 습관이 이 버그를 잡는다. 같은 문제가 dict, tuple, 그리고 `nn.Module`이 아닌 사용자 클래스 안에 넣은 층에도 생긴다.
- **학습하지 않는 상태를 일반 텐서 속성으로 두는 것.** `.to(device)` 후 device 불일치 오류가 나거나(다행인 경우), 체크포인트에서 빠져 복원한 모델이 다른 표준화를 쓴다(조용한 경우). `register_buffer`를 쓴다.
- **`strict=False`를 오류를 없애는 스위치로 쓰는 것.** 이름이 하나도 맞지 않아도 오류가 나지 않는다. 반환값의 `missing_keys`·`unexpected_keys`를 반드시 확인하고, 의도한 키만 빠졌는지 검증한다.
- **모델 객체 전체를 pickle로 저장하는 것.** 클래스 이름이나 모듈 위치를 바꾸면 불러오지 못한다. 설정 + state_dict로 저장한다.

## 통과 기준

- 실습 모델의 모든 파라미터·buffer를 이름·shape과 함께 나열하고, 파라미터 수를 손으로 계산한 값과 일치시킨다.
- 설정과 state_dict만으로 모델을 다시 만들어, 다른 초기화에서 시작해도 eval 출력이 원래 모델과 비트 단위로 같음을 확인한다.
- 연습문제 두 개(`config-registry`, `state-dict-diff`)의 테스트를 통과한다.
