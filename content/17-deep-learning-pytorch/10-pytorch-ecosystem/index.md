---
id: deep-learning-pytorch/pytorch-ecosystem
title: "PyTorch 생태계와 대안"
title_en: "The PyTorch Ecosystem and Alternatives"
tier: reference
status: draft
prerequisites: [deep-learning-pytorch/trainer-design]
---

# PyTorch 생태계와 대안

PyTorch로 연구한다는 것은 PyTorch 하나만 쓴다는 뜻이 아니다. 사전학습 모델은 Hugging Face에서, 그래프 신경망은 PyG에서, ODE solver는 torchdiffeq에서 가져오고, 학습 속도는 `torch.compile`에 맡긴다. 이 라이브러리들은 수년짜리 연구의 기반이 되므로, 각각이 무엇을 대신해 주는지와 언제 기대를 저버리는지를 알아야 한다.

이 장은 참조용이다. 앞 장들의 학습 루프·trainer·평가를 그대로 두고, 그 주변의 도구를 **역할** 중심으로 정리한다. 도구의 API와 버전은 빠르게 바뀌므로, 이 장의 코드와 명령은 모두 집필 시점(PyTorch 2.15, JAX 0.12, uv 0.11)에 실행해 확인한 예시이고, 실습 환경에 설치되어 있지 않은 도구는 이름과 역할만 적는다. 실제로 쓸 때는 각 도구의 공식 문서에서 현재 상태를 확인한다.

## 학습 목표

- 주요 라이브러리의 역할을 안다
- JAX 등 대안 프레임워크의 특징을 이해한다

## 컴파일과 성능

PyTorch의 기본 실행 방식은 **eager mode**다. Python이 코드를 한 줄씩 실행하면서 연산 하나마다 커널 하나를 호출한다. 디버깅이 쉽고 `print`와 `if`를 자유롭게 쓸 수 있지만, 작은 연산이 많으면 Python과 커널 호출의 오버헤드가 쌓이고, 여러 연산을 하나의 커널로 합치는(fusion) 최적화를 할 수 없다.

`torch.compile`은 이 문제를 함수 단위로 푼다. 함수를 처음 호출할 때 Python 실행을 추적해 텐서 연산을 **그래프**로 잡고, 그래프를 backend에 넘겨 최적화된 코드를 만든다. 기본 backend는 `"inductor"`다(집필 시점 docstring). 컴파일 결과는 입력의 shape·dtype 같은 조건(guard)과 함께 저장되고, 조건이 맞는 다음 호출에서 재사용된다.

무엇이 그래프로 잡히는지는 직접 세어 보는 것이 가장 정확하다. 잡힌 그래프를 세기만 하고 그대로 실행하는 backend를 만들어, 분기가 없는 함수와 텐서 값에 의존하는 분기가 있는 함수를 비교한다.

```python {exec="python"}
import torch

torch.manual_seed(0)
graphs = []
def counting_backend(gm, example_inputs):               # 잡힌 그래프를 세기만 하고 그대로 실행
    graphs.append(len(gm.graph.nodes))
    return gm.forward

def with_branch(x):
    y = torch.sin(x) ** 2 + torch.cos(x) ** 2
    if y.sum() > 0:                                     # 텐서 값에 의존하는 Python 분기
        return y * 2
    return y

def without_branch(x):
    return (torch.sin(x) ** 2 + torch.cos(x) ** 2) * 2

for fn in (without_branch, with_branch):
    graphs.clear()
    cfn = torch.compile(fn, backend=counting_backend)
    x = torch.randn(8)
    same = torch.allclose(cfn(x), fn(x))
    print(f"{fn.__name__:15s} 결과 일치 {same}, 그래프 {len(graphs)}개 (노드 수 {graphs})")

for n in (8, 16, 32, 64):                               # 입력 shape을 바꿔 가며 호출
    cfn(torch.randn(n))
    print(f"shape ({n},) 호출 뒤 누적 그래프 {len(graphs)}개")
```

```output
without_branch  결과 일치 True, 그래프 1개 (노드 수 [8])
with_branch     결과 일치 True, 그래프 2개 (노드 수 [9, 3])
shape (8,) 호출 뒤 누적 그래프 2개
shape (16,) 호출 뒤 누적 그래프 4개
shape (32,) 호출 뒤 누적 그래프 4개
shape (64,) 호출 뒤 누적 그래프 4개
```

- **graph break.** 분기 조건 `y.sum() > 0`은 텐서의 **값**이라 컴파일 시점에 알 수 없다. 그래서 그래프가 조건 계산까지(노드 9개)와 선택된 분기(노드 3개)로 나뉘었다. 결과는 eager와 같지만 그래프 사이를 Python이 잇는다. `print`, 텐서를 Python 숫자로 바꾸는 `.item()`, 지원되지 않는 라이브러리 호출도 graph break를 만든다. `fullgraph=True`를 주면 graph break를 오류로 알려 준다.
- **재컴파일.** 같은 shape (8,)로 다시 부르면 저장된 결과를 쓴다. shape (16,)에서 그래프가 두 개 늘었다. 이때 PyTorch는 크기가 바뀌는 차원을 기호로 다루는 동적 shape 그래프를 만들어, 그 뒤의 (32,), (64,)에서는 다시 컴파일하지 않았다(집필 시점 기본 동작). 재컴파일이 너무 많으면 정해진 한도(docstring 기준 기본 8회) 뒤에 eager로 돌아간다.

::quiz{id="compile-graph-break"}

**주의점.** `torch.compile`은 코드 한 줄로 켤 수 있지만, 공짜는 아니다.

- **첫 호출이 느리다.** 그래프 캡처와 코드 생성에 수 초에서 수십 초가 걸린다. 짧은 실험이나 shape이 자주 바뀌는 입력에서는 이득보다 비용이 클 수 있다.
- **결과가 비트 단위로 같지 않을 수 있다.** 연산을 합치면 반올림 순서가 바뀐다. 허용 오차 안의 일치를 확인한다.
- **디버깅이 어렵다.** 오류가 생성된 코드 안에서 나면 원래 코드의 어디인지 찾기 어렵다. 먼저 eager로 맞게 동작하는지 확인하고 컴파일은 마지막에 켠다. 문제가 생기면 끄고(`disable=True`) 재현한다.
- **측정은 warm-up 뒤에.** 처음 몇 step을 버리고, 실제 학습과 같은 shape에서, GPU라면 동기화한 뒤 시간을 잰다. 성능 측정 일반은 22.4장에서 다룬다.

::quiz{id="compile-benchmark"}

**함수 변환 `torch.func`.** PyTorch에는 JAX의 영향을 받은 함수 변환 모듈 `torch.func`도 있다. `grad`는 함수를 받아 그 gradient를 계산하는 함수를 돌려주고, `vmap`은 표본 하나에 대한 함수를 batch 축을 따라 벡터화하며, `jacrev`는 Jacobian을 계산한다. `functional_call`은 모듈의 파라미터를 인자로 넘겨 모듈을 순수 함수처럼 부른다. 조합하면 반복문 없이 **표본별 gradient**를 얻는다.

```python {exec="python"}
from torch import nn
from torch.func import functional_call, grad, jacrev, vmap

model = nn.Linear(3, 1)
params = {k: v.detach() for k, v in model.named_parameters()}
x, y = torch.randn(5, 3), torch.randn(5, 1)             # (B, 3), (B, 1)

def loss_one(params, xi, yi):                           # 표본 하나의 손실. 파라미터를 인자로 받는 순수 함수
    return ((functional_call(model, params, (xi[None],)) - yi) ** 2).mean()

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
loop = torch.stack([torch.autograd.grad(loss_one(dict(model.named_parameters()), x[i], y[i]),
                                        model.weight)[0] for i in range(5)])
print("표본별 weight gradient", tuple(per_sample["weight"].shape), " 반복문과 일치", torch.allclose(per_sample["weight"], loop))
print("표본별 gradient norm", per_sample["weight"].flatten(1).norm(dim=1).numpy().round(3))

def field(r):                                           # 2차원 위치 → 2차원 벡터장
    return torch.stack([r[0] * r[1], torch.sin(r[0])])
print("Jacobian at (1, 2)\n", jacrev(field)(torch.tensor([1.0, 2.0])).numpy().round(4))
```

```output
표본별 weight gradient (5, 1, 3)  반복문과 일치 True
표본별 gradient norm [1.31  1.259 6.542 3.385 3.991]
Jacobian at (1, 2)
 [[2.     1.    ]
 [0.5403 0.    ]]
```

표본별 gradient는 batch 평균 gradient로는 볼 수 없는 정보를 준다. 세 번째 표본(index 2)의 gradient norm(6.542)이 다른 표본보다 크다. label 오류나 이상치 표본을 찾을 때(17.9장), 표본마다 gradient를 잘라 붙이는 차등 프라이버시 학습에서 이 값을 쓴다. Jacobian은 물리 법칙을 손실에 넣을 때 필요한 미분(24.2장)과 같은 계산이다.

## 도메인 라이브러리

분야별로 데이터 형식, 표준 모델, 사전학습 가중치를 제공하는 라이브러리가 있다. 아래는 역할의 요약이다(집필 시점 예시. 이 책의 실습 환경에는 설치되어 있지 않다).

| 라이브러리 | 역할 | 연구에서 쓰는 곳 |
|---|---|---|
| Hugging Face `transformers` | Transformer 계열 사전학습 모델과 tokenizer, 모델 저장소에서 내려받기 | 언어 모델, 단백질·분자 언어 모델의 fine-tuning(19장) |
| Hugging Face `datasets` | 공개 데이터셋 내려받기, 열 단위 저장 형식, 전처리 파이프라인 | 큰 공개 데이터의 재현 가능한 로딩 |
| `timm` | 이미지 모델 구현과 사전학습 가중치 모음 | 영상·현미경·위성 이미지의 backbone |
| PyTorch Geometric(PyG) | 그래프 자료 구조, message passing 층, 그래프 데이터셋 | 분자·결정·메시 위의 그래프 신경망(18.6장) |
| `torchvision` | 이미지 변환·증강, 표준 이미지 모델과 데이터셋 | 이미지 전처리, baseline 모델 |
| `torchaudio` | 오디오 입출력, 스펙트로그램 같은 신호 변환 | 음향·진동 신호 |

이런 라이브러리는 수백 줄의 코드와 수천 GPU 시간의 사전학습을 대신해 준다. 그만큼 **내 결과의 일부가 그 라이브러리의 버전에 달려** 있다. 같은 모델 이름이라도 저장소의 가중치가 갱신될 수 있고, 전처리 기본값(이미지 크기 조정 방식, tokenizer 설정)이 버전에 따라 달라질 수 있다. 사전학습 가중치를 쓸 때는 라이브러리 버전과 함께 가중치의 정확한 판(revision)을 기록하고, 전처리 설정을 inference contract(17.6장)에 넣는다.

## 과학 계산 연계

과학 문제에서 PyTorch는 신경망만 학습하는 도구가 아니라 **미분 가능한 수치 계산 엔진**이기도 하다. autograd는 신경망 층이든 수치 적분기든 텐서 연산으로 쓰인 계산이면 모두 미분한다(8.4장).

**미분가능 시뮬레이터.** 시뮬레이터를 텐서 연산으로 쓰면, 시뮬레이션 결과와 관측의 차이를 시뮬레이터의 물리 파라미터로 미분할 수 있다. 감쇠 진동자 $\ddot{x} = -kx - c\dot{x}$를 300 step 적분하는 코드를 그대로 미분해, 관측한 궤적에서 감쇠 계수 $c$를 추정한다.

```python {exec="python"}
def simulate(c, k=4.0, dt=0.01, steps=300):
    """감쇠 진동자 x'' = -k x - c x'를 semi-implicit Euler로 적분한다. 반환: (steps,) 위치"""
    x, v, xs = torch.tensor(1.0), torch.tensor(0.0), []
    for _ in range(steps):
        v = v + dt * (-k * x - c * v)
        x = x + dt * v
        xs.append(x)
    return torch.stack(xs)

observed = simulate(torch.tensor(0.8))                  # '측정값': 참값 c = 0.8
c = torch.tensor(0.1, requires_grad=True)               # 초기 추정
opt = torch.optim.Adam([c], lr=0.05)
for it in range(201):
    loss = ((simulate(c) - observed) ** 2).mean()       # 300 step 전체를 통과하는 역전파
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0:
        print(f"반복 {it:3d}  c = {c.item():.4f}  손실 {loss.item():.2e}")
```

```output
반복   0  c = 0.1500  손실 8.02e-02
반복  50  c = 0.8397  손실 1.66e-04
반복 100  c = 0.8043  손실 1.50e-06
반복 150  c = 0.8003  손실 4.72e-09
반복 200  c = 0.8000  손실 4.19e-11
```

추정값이 참값 0.8에 수렴했다. 역전파는 적분기의 300 step을 거슬러 올라가며 각 step의 미분을 곱해 나간다. 같은 원리로 신경망과 물리 모델을 하나의 계산 그래프로 묶어 함께 학습할 수 있고(24.5장), 이것이 "end-to-end"의 과학 계산판이다. 비용도 같은 원리에서 나온다. 역전파를 위해 모든 step의 중간값을 저장하므로 step이 길어지면 메모리가 선형으로 늘고, 혼돈계처럼 민감한 시스템에서는 긴 구간의 gradient가 폭발한다.

**torchdiffeq.** ODE를 푸는 solver(고정 step과 적응 step)를 PyTorch 텐서 연산으로 구현하고, 그 해를 autograd로 미분하게 해 주는 라이브러리다. Neural ODE(24.3장)를 구현할 때 널리 쓰인다. 집필 시점(0.2.5)에 `odeint`와, 메모리를 아끼기 위해 역방향 ODE를 풀어 gradient를 계산하는 adjoint 방법의 `odeint_adjoint` 함수를 확인했다. adjoint 방법은 중간값을 저장하지 않는 대신 역방향 적분의 수치 오차가 gradient에 들어간다는 점에서 위의 직접 역전파와 다르다.

:::note[분야 예시]
기후 모델의 구름 물리처럼 해상도 아래의 과정을 신경망으로 대신하는 연구에서는, 신경망 출력이 수치 모델에 들어가 다음 시간 step의 상태를 만든다. 신경망을 한 step 오차로만 학습하면 수백 step을 돌렸을 때 불안정해지는 경우가 있어, 여러 step을 펼친(rollout) 미분가능 시뮬레이션으로 학습하는 방법이 연구된다. 단백질이나 결정 구조를 다루는 분야에서는 에너지를 신경망으로 예측하고 그 위치 미분으로 힘을 얻어, 분자동역학 적분기에 넣는다. 두 경우 모두 "시뮬레이터 안의 신경망"을 미분해야 하므로, 시뮬레이터가 어떤 프레임워크로 쓰였는지가 도구 선택을 정한다.
:::

## 대안 프레임워크

**JAX**는 NumPy와 비슷한 배열 API 위에, 함수를 다른 함수로 바꾸는 **조합 가능한 변환**을 제공하는 프레임워크다.

- `jax.grad`: 함수 → 그 gradient를 계산하는 함수
- `jax.jit`: 함수 → XLA 컴파일러로 컴파일된 함수
- `jax.vmap`: 표본 하나에 대한 함수 → batch에 대한 함수

이 변환들은 **순수 함수**(같은 입력에 같은 출력, 부작용 없음)를 전제한다. 그래서 JAX 배열은 제자리 수정이 안 되고, 모델 파라미터는 객체 안에 숨지 않고 함수의 인자와 반환값으로 오간다. 난수도 전역 상태가 아니라 명시적인 key(`jax.random.key`)를 나누어(`jax.random.split`) 쓴다. 실습의 비교 코드에서 JAX 학습 step은 다음과 같다(집필 시점 JAX 0.12에서 실행 확인).

```python
import jax
import jax.numpy as jnp

def loss_fn(p, x, y):                                     # 순수 함수: 파라미터를 인자로 받는다
    pred = jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    return jnp.mean((pred - y) ** 2)

@jax.jit                                                  # 한 step 전체를 컴파일
def step(p, x, y):
    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
    return jax.tree_util.tree_map(lambda v, gv: v - LR * gv, p, g), loss   # 새 파라미터를 반환

p, loss = step(p, X, Y)                                   # 호출하는 쪽이 파라미터를 교체한다
```

::quiz{id="jax-pure-update"}

JAX 자체에는 신경망 층이나 optimizer가 들어 있지 않다. 그 위에 라이브러리가 쌓인다(역할만 적는다).

- **Flax**: JAX 위의 신경망 라이브러리. 층 정의와 파라미터 관리 방식을 제공한다.
- **Equinox**: 모델을 파라미터를 담은 Python 객체(PyTree)로 표현해, JAX 변환에 그대로 넘길 수 있게 하는 라이브러리.

PyTorch와 JAX의 차이를 정리하면 다음과 같다.

| | PyTorch | JAX |
|---|---|---|
| 기본 실행 | eager. 컴파일은 선택(`torch.compile`) | eager도 되지만 보통 `jit`으로 컴파일 |
| 상태 | 모듈이 파라미터를 갖고, optimizer가 제자리에서 갱신 | 파라미터를 인자로 받고 새 값을 반환 |
| 미분 | `backward()`가 `.grad`에 누적 | `grad`가 gradient를 반환값으로 |
| 벡터화 | 대부분 batch 축을 직접 다룸, `torch.func.vmap` | `vmap`이 핵심 도구 |
| 난수 | 전역 RNG 상태 | 명시적 key |
| 생태계 | 사전학습 모델·도메인 라이브러리가 가장 넓음 | 과학 계산·미분가능 시뮬레이션·대규모 TPU 학습에서 강함 |

**선택 기준.** 프레임워크의 우열보다 **연구가 기대는 것**이 무엇인지가 선택을 정한다.

- 써야 할 사전학습 모델이나 도메인 라이브러리가 한쪽에만 있다면 그쪽이다. 다시 구현하는 비용이 프레임워크의 장점보다 크다.
- 연구의 핵심이 고차 미분, 표본별 미분, 미분가능 시뮬레이터처럼 **함수 변환의 조합**이라면 JAX가 자연스럽다. PyTorch에서도 `torch.func`로 같은 일을 할 수 있다.
- 연구실과 협업자가 쓰는 프레임워크, 연구실의 trainer와 평가 코드(17.7장)를 공유할 수 있는지도 중요하다.

## 라이브러리 선택 원칙

외부 라이브러리 하나를 연구에 넣는 것은 그 라이브러리의 미래에 연구의 일부를 맡기는 것이다. 몇 년짜리 연구에서 실제로 문제가 되는 기준은 다음과 같다.

- **유지보수 상태.** 최근 릴리스 시점, 이슈와 PR에 대한 응답, 관리자 수, 최신 PyTorch·Python 버전 지원. 저장소의 별 수는 과거의 관심일 뿐, 지금 유지되는지를 말해 주지 않는다. 논문과 함께 공개된 코드는 논문이 나온 뒤 갱신이 멈추는 경우가 많다.
- **라이선스.** 코드의 라이선스(MIT·BSD·Apache 같은 허용형인지, GPL 같은 copyleft인지)와 **사전학습 가중치·데이터의 라이선스**는 따로 확인한다. 비상업 조건이 붙은 가중치는 산업 협력 과제에 쓸 수 없을 수 있다(15.5장).
- **의존성의 무게.** 특정 PyTorch·CUDA 버전을 고정하는 라이브러리는 다른 라이브러리와 충돌하고, 환경을 갱신할 때마다 걸림돌이 된다. 무엇이 무엇을 끌고 오는지 확인한다.
- **필요한 부분의 크기.** 라이브러리에서 함수 하나만 쓴다면, 그 함수를 라이선스를 지켜 옮겨 오고 테스트로 감싸는 편이 의존성 하나를 늘리는 것보다 나을 수 있다.

확인 절차의 예는 다음과 같다(집필 시점 예시. uv 0.11).

```bash
uv tree --project labs --depth 1                   # 프로젝트가 직접 의존하는 패키지와 버전
uv pip show --python labs/.venv/bin/python torch   # 설치된 패키지의 버전·의존성 정보
```

라이브러리를 고른 뒤에는 lockfile로 버전을 고정하고(3.3장), 연구 기록에 고른 이유를 한 줄 남긴다. 나중에 라이브러리를 바꿔야 할 때 그 이유가 아직 유효한지 판단할 수 있다.

::quiz{id="library-selection"}

## 실습

같은 작은 모델을 PyTorch와 JAX로 구현해 코드 구조와 성능을 비교한다. 코드는 `labs/deep-learning-pytorch/pytorch-ecosystem/compare_torch_jax.py`다. 두 구현은 NumPy로 만든 같은 초기 가중치와 데이터에서 시작해, full-batch gradient descent로 500 step 학습한다. PyTorch 쪽은 eager로 파라미터를 제자리에서 갱신하고, JAX 쪽은 `jit`으로 컴파일한 순수 함수가 새 파라미터를 반환한다. JAX는 실습 공통 환경에 없으므로 `--with`로 이 실행에만 더한다.

```bash
uv run --project labs --group torch --with jax python labs/deep-learning-pytorch/pytorch-ecosystem/compare_torch_jax.py
```

집필 시점의 한 실행 결과(CPU, PyTorch 2.15, JAX 0.12)는 다음과 같다. **시간은 기계와 부하에 따라 실행마다 달라진다.**

```output
PyTorch (eager)  최종 손실 0.010195  첫 step      3.8 ms  이후 step 중앙값  0.348 ms
JAX (jit)        최종 손실 0.010195  첫 step    277.3 ms  이후 step 중앙값  0.100 ms
```

- **결과는 같다.** 같은 초기값과 같은 알고리즘이면 프레임워크가 달라도 손실이 소수 여섯째 자리까지 같다. 다르다면 초기화·데이터·업데이트 규칙 중 하나가 다르다는 뜻이다. 두 구현을 비교하기 전에 이것부터 맞춘다.
- **첫 step과 이후 step이 다르다.** JAX의 첫 step은 컴파일 시간 때문에 수백 ms가 걸렸고, 이후 step은 PyTorch eager보다 빨랐다. 이 정도로 작은 모델에서는 계산보다 호출 오버헤드가 시간을 차지하므로, 한 step 전체를 컴파일하는 쪽이 유리하다. 모델이 커지면 차이는 줄거나 바뀔 수 있다. 결론을 일반화하지 말고 자신의 모델 크기에서 잰다.

이어서 다음을 한다.

1. PyTorch 쪽에 `torch.compile`을 적용하고(eager 함수를 감싸거나 `torch.func`로 순수 함수 형태로 바꾸어), 첫 step과 이후 step의 시간을 같은 방식으로 잰다.
2. 모델 폭을 32에서 1024로 늘려 같은 비교를 반복하고, 프레임워크 간 차이가 어떻게 바뀌는지 기록한다.
3. 두 코드에서 "파라미터가 어디에 있고 누가 바꾸는가"를 표시하고, 연구실 trainer(17.7장)를 JAX로 옮긴다면 무엇이 바뀌어야 하는지 한 문단으로 쓴다.

## 실패 사례

- **유지보수가 끝난 라이브러리에 연구 전체를 의존하는 것.** 논문 코드나 개인 프로젝트에 데이터 로딩부터 모델까지 맡겼다가, 새 PyTorch·CUDA·Python 버전에서 설치가 안 되어 환경을 몇 년 전 버전에 묶어 둔다. 보안 패치도, 새 GPU 지원도 받을 수 없다. 핵심 의존성은 유지보수 상태를 확인하고, 대체할 방법(다른 라이브러리, 필요한 부분의 직접 구현)을 미리 알아 둔다.
- **컴파일·새 프레임워크의 속도를 첫 실행으로 판단하는 것.** 첫 호출의 컴파일 시간을 포함해 "느리다"고 하거나, 작은 예에서 본 속도 차이를 실제 모델에 일반화한다.
- **라이브러리의 기본 전처리를 기록하지 않는 것.** 사전학습 모델의 tokenizer·이미지 변환 설정이 버전에 따라 바뀌어, 같은 코드가 다른 입력을 만든다.
- **도구를 바꾸면서 결과 일치를 확인하지 않는 것.** 프레임워크를 옮기거나 컴파일을 켠 뒤, 작은 문제에서 이전 구현과 같은 결과가 나오는지 확인하지 않고 큰 실험을 돌린다.

## 통과 기준

- 자신의 연구에 필요한 라이브러리 세 개 이상을 유지보수 상태·라이선스·의존성 기준으로 평가하고, 선택 근거를 한 문단씩 기록한다.
- 같은 작은 모델을 두 가지 방식(PyTorch와 JAX, 또는 eager와 `torch.compile`)으로 실행해 결과가 일치함을 확인하고, warm-up을 뺀 시간을 비교한다.
