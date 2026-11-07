"""17.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 3만 순수 Python 브라우저 셀이고(exec="browser"), 나머지는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/modules/cells.py
"""
# %% cell 1 — nn.Module 정의와 하위 모듈 등록
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

# %% cell 2 — parameter와 buffer
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

# %% cell 3 — 등록은 __setattr__가 한다: Module의 등록 장치 흉내 내기
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

# %% cell 4 — 일반 list와 ModuleList
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

# %% cell 5 — state_dict 저장·로드와 strict
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

# %% cell 6 — forward·backward hook으로 내부 관찰
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
