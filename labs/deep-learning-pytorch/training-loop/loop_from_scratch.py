"""17.5 실습: 프레임워크 없이(NumPy만으로) 학습 루프를 처음부터 작성하고, 같은 루프를 PyTorch로 쓴 것과 비교한다.

실행: uv run --project labs --group torch python labs/deep-learning-pytorch/training-loop/loop_from_scratch.py
NumPy 부분은 PyTorch 없이도 돈다. 각 줄의 주석은 대응하는 수식이다.

모델: y_hat = W2 tanh(W1 x + b1) + b2   (2층 MLP, 회귀)
손실: L = (1/B) sum_i (y_hat_i - y_i)^2
갱신: Adam (Kingma, Ba, 2015; 10.4장)
"""
import numpy as np

D_IN, H, LR, STEPS, BATCH = 1, 32, 1e-2, 3000, 16


def make_data(n=32, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, size=(n, D_IN)), axis=0)          # (N, 1)
    y = np.sin(2 * x) + 0.3 * x + 0.1 * rng.normal(size=(n, 1))      # (N, 1)
    return x, y


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(H, D_IN)) * np.sqrt(1 / D_IN),        # (H, D)  Xavier 근처
        "b1": np.zeros(H),                                           # (H,)
        "W2": rng.normal(size=(1, H)) * np.sqrt(1 / H),              # (1, H)
        "b2": np.zeros(1),                                           # (1,)
    }


def forward(p, x):
    a = x @ p["W1"].T + p["b1"]                                      # a = W1 x + b1           (B, H)
    h = np.tanh(a)                                                   # h = tanh(a)             (B, H)
    y_hat = h @ p["W2"].T + p["b2"]                                  # y_hat = W2 h + b2       (B, 1)
    return y_hat, h


def loss_and_grads(p, x, y):
    B = len(x)
    y_hat, h = forward(p, x)                                         # forward
    loss = np.mean((y_hat - y) ** 2)                                 # L = (1/B) sum (y_hat - y)^2
    d_yhat = 2 * (y_hat - y) / B                                     # dL/dy_hat               (B, 1)
    g = {"W2": d_yhat.T @ h,                                         # dL/dW2 = d_yhat^T h     (1, H)
         "b2": d_yhat.sum(0)}                                        # dL/db2 = sum_i d_yhat   (1,)
    d_h = d_yhat @ p["W2"]                                           # dL/dh = d_yhat W2       (B, H)
    d_a = d_h * (1 - h**2)                                           # dL/da = dL/dh * tanh'(a) (B, H)
    g["W1"] = d_a.T @ x                                              # dL/dW1 = d_a^T x        (H, D)
    g["b1"] = d_a.sum(0)                                             # dL/db1                  (H,)
    return loss, g


def adam_init(p):
    return {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}, 0


def adam_step(p, g, state, t, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    for k in p:
        m, v = state[k]
        m = b1 * m + (1 - b1) * g[k]                                 # m_t = b1 m + (1-b1) g
        v = b2 * v + (1 - b2) * g[k] ** 2                            # v_t = b2 v + (1-b2) g^2
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)              # bias correction
        p[k] = p[k] - lr * m_hat / (np.sqrt(v_hat) + eps)            # theta <- theta - lr m_hat/(sqrt(v_hat)+eps)
        state[k] = (m, v)
    return state, t


def batch_order(n, steps, batch, seed=0):
    """step마다 쓸 표본 인덱스. epoch마다 섞는다(17.4장)."""
    rng, out = np.random.default_rng(seed), []
    while len(out) < steps:
        perm = rng.permutation(n)
        out += [perm[i:i + batch] for i in range(0, n - batch + 1, batch)]   # drop_last
    return out[:steps]


def train_numpy(x, y, order):
    p = init_params()
    state, t = adam_init(p)
    losses = []
    for idx in order:                                                # 한 step
        xb, yb = x[idx], y[idx]                                      # batch 구성   (B, 1), (B, 1)
        loss, g = loss_and_grads(p, xb, yb)                          # forward → loss → backward
        state, t = adam_step(p, g, state, t)                         # update (g는 매 step 새로 계산 = zero_grad)
        losses.append(loss)
    return p, losses


def train_torch(x, y, order):
    import torch
    from torch import nn

    p0 = init_params()
    model = nn.Sequential(nn.Linear(D_IN, H), nn.Tanh(), nn.Linear(H, 1)).double()
    with torch.no_grad():                                            # NumPy와 같은 초기값
        model[0].weight.copy_(torch.from_numpy(p0["W1"])); model[0].bias.copy_(torch.from_numpy(p0["b1"]))
        model[2].weight.copy_(torch.from_numpy(p0["W2"])); model[2].bias.copy_(torch.from_numpy(p0["b2"]))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    losses = []
    for idx in order:
        idx = torch.from_numpy(idx)
        pred = model(xt[idx])                                        # forward
        loss = nn.functional.mse_loss(pred, yt[idx])                 # loss
        opt.zero_grad()                                              # .grad 비우기
        loss.backward()                                              # backward
        opt.step()                                                   # update
        losses.append(loss.item())
    return model, losses


def main():
    x, y = make_data()
    order = batch_order(len(x), STEPS, BATCH)
    p, losses = train_numpy(x, y, order)
    full = np.mean((forward(p, x)[0] - y) ** 2)
    print(f"[NumPy] 표본 {len(x)}개, batch {BATCH}, Adam lr={LR}, {STEPS} step")
    for k in [0, 100, 500, 1000, 2000, STEPS - 1]:
        print(f"  step {k:4d}  batch 손실 {losses[k]:.6f}")
    print(f"  학습 데이터 전체 MSE {full:.2e}  (잡음 분산 1.0e-02보다 작으면 잡음까지 외운 것)")
    one = [np.arange(0, 32, 4)] * 2000                               # 같은 batch 하나(8개)만 반복
    _, one_losses = train_numpy(x, y, one)
    print(f"[과적합 점검] batch 하나(표본 8개)를 2000 step: 손실 {one_losses[0]:.3f} -> {one_losses[-1]:.1e}")
    try:
        import torch
    except ImportError:
        print("[PyTorch] torch가 없어 비교를 건너뛴다(--group torch로 실행)")
        return
    torch.set_num_threads(1)
    model, tlosses = train_torch(x, y, order)
    diffs = np.abs(np.array(losses) - np.array(tlosses))
    print("[PyTorch] 같은 초기값·같은 batch 순서·float64")
    for k in [0, 100, 1000, STEPS - 1]:
        print(f"  step {k:4d}  |손실 차이| {diffs[k]:.1e}")
    w_diff = np.abs(model[0].weight.detach().numpy() - p["W1"]).max()
    print(f"  최종 W1 최대 차이 {w_diff:.1e}")


if __name__ == "__main__":
    main()
