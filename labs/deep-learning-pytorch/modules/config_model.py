"""17.3 실습: 설정 파일로 모델을 만들고, hook으로 중간 활성값을 기록하고, state_dict로 정확히 복원한다.

실행: uv run --project labs --group torch python labs/deep-learning-pytorch/modules/config_model.py
설정: labs/deep-learning-pytorch/modules/configs/mlp.toml
"""
import io
import tomllib
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
REGISTRY = {}


def register(name):
    """클래스를 이름으로 등록하는 decorator. 설정의 type 문자열이 이 이름을 가리킨다."""
    def deco(cls):
        if name in REGISTRY:
            raise KeyError(f"이미 등록된 이름: {name}")
        REGISTRY[name] = cls
        return cls
    return deco


def build(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in REGISTRY:
        raise KeyError(f"알 수 없는 type '{kind}'. 등록된 이름: {sorted(REGISTRY)}")
    return REGISTRY[kind](**cfg)


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
NORMS = {"none": None, "layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d}


@register("standardize")
class Standardize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):                                   # x: (B, D)
        return (x - self.mean) / self.std


@register("mlp")
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden, activation="relu", norm="none", dropout=0.0):
        super().__init__()
        blocks, prev = [], d_in
        for width in hidden:
            layers = [nn.Linear(prev, width)]
            if NORMS[norm] is not None:
                layers.append(NORMS[norm](width))
            layers += [ACTIVATIONS[activation](), nn.Dropout(dropout)]
            blocks.append(nn.Sequential(*layers))
            prev = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev, d_out)

    def forward(self, x):                                   # x: (B, d_in)
        for block in self.blocks:
            x = block(x)
        return self.head(x)                                 # (B, d_out)


class Model(nn.Module):
    """입력 표준화 + 몸통. 설정 전체에서 만든다."""
    def __init__(self, cfg):
        super().__init__()
        self.input_norm = build(cfg["input_norm"])
        self.net = build(cfg["model"])

    def forward(self, x):
        return self.net(self.input_norm(x))


def record_activations(model, x):
    """활성화 층마다 forward hook을 걸어 (평균, 표준편차)를 기록하고 hook을 제거한다."""
    stats, handles = {}, []
    for name, m in model.named_modules():
        if isinstance(m, tuple(ACTIVATIONS.values())):
            def hook(mod, inp, out, name=name):
                stats[name] = (out.mean().item(), out.std().item())
            handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return stats


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = tomllib.loads((HERE / "configs/mlp.toml").read_text(encoding="utf-8"))
    model = Model(cfg)

    print("[1] 파라미터")
    for name, p in model.named_parameters():
        print(f"  {name:28s} {str(tuple(p.shape)):10s} requires_grad={p.requires_grad}")
    print("[1] buffer")
    for name, b in model.named_buffers():
        print(f"  {name:28s} {str(tuple(b.shape)):10s} {[round(v, 4) for v in b.tolist()]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 합계 {n_params}개, state_dict 키 {len(model.state_dict())}개")

    x = torch.stack([300 + 10 * torch.randn(256), 1e5 + 2e3 * torch.randn(256), torch.rand(256)], dim=1)  # (B, 3)
    model.eval()
    print("[2] 활성화 층별 출력 (forward hook)")
    for name, (m, s) in record_activations(model, x).items():
        print(f"  {name:22s} 평균 {m:+.3f}  표준편차 {s:.3f}")
    print("  남은 hook 수:", sum(len(m._forward_hooks) for m in model.modules()))

    buf = io.BytesIO()
    torch.save({"config": cfg, "state_dict": model.state_dict()}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=True)
    torch.manual_seed(123)                                  # 다른 무작위 초기화에서 시작
    restored = Model(ckpt["config"])
    result = restored.load_state_dict(ckpt["state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        same = torch.equal(model(x), restored(x))
    print("[3] state_dict 복원")
    print("  load_state_dict:", result)
    print("  설정 + state_dict로 다시 만든 모델의 출력이 비트 단위로 같은가:", same)
    try:
        build({"type": "transformer"})
    except KeyError as e:
        print("[4] 등록되지 않은 type:", e)


if __name__ == "__main__":
    main()
