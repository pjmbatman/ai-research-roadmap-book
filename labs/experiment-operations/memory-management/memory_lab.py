"""22.5 실습: 학습 메모리를 손으로 계산하고 측정값과 비교한 뒤, 절약 기법별 효과를 잰다.

CPU에서는 backward를 위해 저장되는 텐서(saved tensors)를 hook으로 세어 활성값 메모리를 측정한다.
CUDA가 있으면 같은 설정마다 torch.cuda.max_memory_allocated()로 최대 할당량도 함께 잰다.

실행: uv run --project labs --group torch python labs/experiment-operations/memory-management/memory_lab.py
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDTHS = [512, 2048, 2048, 2048, 2048, 512]          # ReLU MLP의 층 폭
BATCH = 128


def build():
    layers = [m for i, o in zip(WIDTHS[:-1], WIDTHS[1:]) for m in (nn.Linear(i, o), nn.ReLU())]
    return nn.Sequential(*layers[:-1]).to(DEVICE)


def predicted(batch, act_bytes=4, n_params=None):
    """손 계산: 파라미터·gradient·Adam 상태(16 byte/파라미터)와 활성값(Linear 입력 폭의 합)."""
    static = 16 * n_params
    acts = batch * sum(WIDTHS[:-1]) * act_bytes
    return static, acts


def measure(model, batch, mode):
    """mode: plain | checkpoint | autocast.

    backward용 저장 텐서를 활성값과 가중치 사본(autocast가 만든 bf16 가중치)으로 나눠 byte로 돌려주고,
    CUDA면 최대 할당량도 돌려준다.
    """
    params = {p.data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        if t.data_ptr() not in params:
            kind = "act" if t.dim() > 0 and t.shape[0] == batch else "weight"   # 첫 축이 batch면 활성값
            seen[t.data_ptr()] = (kind, t.numel() * t.element_size())
        return t

    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(batch, WIDTHS[0], device=DEVICE)        # (B, D)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if mode == "checkpoint":
            half = len(model) // 2 + 1                       # 두 구간으로 나눈다
            h = checkpoint(model[:half], x, use_reentrant=False)
            out = checkpoint(model[half:], h, use_reentrant=False)
        elif mode == "autocast":
            dtype = torch.bfloat16
            with torch.autocast(device_type=DEVICE, dtype=dtype):
                out = model(x)
        else:
            out = model(x)
        loss = out.float().pow(2).mean()
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
    opt.zero_grad(set_to_none=True)
    act = sum(n for k, n in seen.values() if k == "act")
    weight = sum(n for k, n in seen.values() if k == "weight")
    return act, weight, peak


def main():
    model = build()
    n_params = sum(p.numel() for p in model.parameters())
    static, acts = predicted(BATCH, n_params=n_params)
    print(f"장치 {DEVICE}, 층 폭 {WIDTHS}, batch {BATCH}, 파라미터 {n_params:,}개")
    print(f"손 계산: 파라미터+gradient+Adam {static / 2**20:7.1f} MiB, 활성값 {acts / 2**20:5.2f} MiB")
    rows = [("그대로", BATCH, "plain"), ("batch 절반", BATCH // 2, "plain"),
            ("checkpointing(2구간)", BATCH, "checkpoint"), ("autocast bf16", BATCH, "autocast")]
    for name, b, mode in rows:
        act, weight, peak = measure(model, b, mode)
        line = f"{name:22s} 저장된 활성값 {act / 2**20:5.2f} MiB  가중치 사본 {weight / 2**20:5.1f} MiB"
        if peak is not None:
            line += f"  최대 할당 {peak / 2**20:7.1f} MiB"
        print(line)


if __name__ == "__main__":
    main()
