"""17.10 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

모든 셀이 PyTorch가 필요하다(exec="python"). 그래프 수는 PyTorch 버전에 따라 달라질 수 있다(집필 시점 2.15).
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/pytorch-ecosystem/cells.py
"""
# %% cell 1 — torch.compile이 실제로 무엇을 잡아 가는지 세어 본다
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

# %% cell 2 — torch.func: 함수 변환으로 표본별 gradient와 Jacobian
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

# %% cell 3 — 미분가능 시뮬레이션: 적분기를 통과해 감쇠 계수를 추정한다
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
