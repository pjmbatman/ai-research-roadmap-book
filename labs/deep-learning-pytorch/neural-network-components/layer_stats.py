"""17.1 실습: 같은 파라미터 예산에서 깊이·폭·활성화·초기화를 바꾸며 층별 활성값과 gradient를 관찰한다.

실행: uv run --project labs --group torch python labs/deep-learning-pytorch/neural-network-components/layer_stats.py
CPU에서 수 초 안에 끝난다. 시드를 고정하므로 같은 환경에서는 같은 결과가 나온다.
"""
import torch
from torch import nn

D_IN, N = 16, 2048


def make_data(seed=0):
    """학습용 N개와 검증용 N개. 목표는 sin(2 w·x) + 잡음(표준편차 0.1)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2 * N, D_IN, generator=g)                          # (2N, D_in)
    w = torch.randn(D_IN, generator=g) / D_IN**0.5
    y = (torch.sin(2 * x @ w) + 0.1 * torch.randn(2 * N, generator=g))[:, None]  # (2N, 1)
    return x[:N], y[:N], x[N:], y[N:]


def make_mlp(depth, width, act, init, seed=0):
    torch.manual_seed(seed)
    sizes = [D_IN] + [width] * depth
    layers = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        lin = nn.Linear(i, o)
        if init == "he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
        elif init == "small":
            nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
        # init == "default": PyTorch 기본 초기화를 그대로 둔다
        layers += [lin, {"relu": nn.ReLU(), "tanh": nn.Tanh()}[act]]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def layer_stats(model, x, y):
    """은닉층마다 (활성값 std, 가중치 gradient 노름)을 돌려준다."""
    acts = []
    hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
             for m in model if isinstance(m, (nn.ReLU, nn.Tanh))]
    model.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    for h in hooks:
        h.remove()
    lins = [m for m in model if isinstance(m, nn.Linear)][:-1]
    return [(a.std().item(), lin.weight.grad.norm().item()) for a, lin in zip(acts, lins)]


def train(model, x, y, steps=300, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


@torch.no_grad()
def val_mse(model, x, y):
    return nn.functional.mse_loss(model(x), y).item()


def main():
    torch.set_num_threads(1)              # 작은 모델은 스레드 하나가 빠르고, 결과도 스레드 수에 따라 바뀌지 않는다
    x, y, xv, yv = make_data()
    print(f"데이터: 학습 {N}개 + 검증 {N}개, 입력 {D_IN}차원, 검증 목표 분산 {yv.var().item():.3f}, 잡음 분산 0.010")
    for depth, width in [(2, 100), (8, 40)]:
        n_params = sum(p.numel() for p in make_mlp(depth, width, "relu", "default").parameters())
        print(f"\n은닉층 {depth}개, 폭 {width} (파라미터 {n_params}개)")
        print("  활성화 초기화   층별 활성값 std(첫 층 → 마지막 층)       첫 층 |grad|  검증 MSE(300 step 후)")
        for act in ["relu", "tanh"]:
            for init in ["default", "he", "small"]:
                model = make_mlp(depth, width, act, init)
                st = layer_stats(model, x, y)
                stds = " ".join(f"{s:.0e}" for s, _ in st)
                train(model, x, y)
                print(f"  {act:5s}  {init:8s} {stds:<48s} {st[0][1]:8.1e}     {val_mse(model, xv, yv):.3f}")


if __name__ == "__main__":
    main()
