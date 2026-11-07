"""10.6 실습: 학습된 두 모델 사이의 선형 보간 경로와 Hessian 최대 고윳값.

모델: 입력 2차원, tanh 은닉층 하나(뉴런 H개), 출력 1개인 MLP. NumPy로 직접 역전파한다.
데이터: 2차원 회귀 y = sin(2 x1) cos(x2) + 잡음.

1) 선형 보간 경로의 손실 장벽
   - 서로 다른 시드에서 시작해 학습한 두 해
   - 같은 초기값에서 시작해 mini-batch 순서만 다르게 학습한 두 해
   - 한 해와, 그 해의 은닉 뉴런 순서만 바꾼 해(같은 함수)
2) full-batch 경사하강에서 학습률별 Hessian 최대 고윳값(sharpness)의 변화와 2/lr의 비교
3) 무작위 방향 2차원 단면을 outputs/slice2d.png로 저장

실행: uv run --project labs python labs/optimization/nonconvex-landscape/interpolation_hessian.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"
H = 32
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(256, 2))                          # (N, 2)
Y = np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.05 * rng.normal(size=256)   # (N,)
P = 4 * H + 1                                                  # W1 (2H), b1 (H), w2 (H), b2 (1)


def unpack(th):
    return th[:2 * H].reshape(2, H), th[2 * H:3 * H], th[3 * H:4 * H], th[4 * H]


def loss_grad(th, idx=slice(None)):
    W1, b1, w2, b2 = unpack(th)
    x, y = X[idx], Y[idx]
    a = np.tanh(x @ W1 + b1)                                   # (B, H)
    r = a @ w2 + b2 - y                                        # (B,)
    n = len(y)
    da = np.outer(r, w2) * (1 - a**2) / n                      # (B, H)
    g = np.concatenate([(x.T @ da).ravel(), da.sum(0), a.T @ r / n, [r.mean()]])
    return 0.5 * np.mean(r**2), g


def init(seed):
    r = np.random.default_rng(seed)
    return np.concatenate([r.normal(size=2 * H) / np.sqrt(2), np.zeros(H), r.normal(size=H) / np.sqrt(H), [0.0]])


def sgd(th, data_seed, lr=0.05, epochs=300, batch=32, beta=0.9):
    r, v = np.random.default_rng(data_seed), np.zeros_like(th)
    for _ in range(epochs):
        order = r.permutation(len(Y))
        for i in range(0, len(Y), batch):
            g = loss_grad(th, order[i:i + batch])[1]
            v = beta * v + g
            th = th - lr * v
    return th


def path(a, b, ts):
    return np.array([loss_grad((1 - t) * a + t * b)[0] for t in ts])


def permute(th, perm):
    W1, b1, w2, b2 = unpack(th)
    return np.concatenate([W1[:, perm].ravel(), b1[perm], w2[perm], [b2]])


def top_eig(th, iters=200, h=1e-5, seed=0):
    """Hessian-vector product(gradient 중앙 차분)를 쓴 power iteration. 절댓값이 가장 큰 고윳값."""
    v = np.random.default_rng(seed).normal(size=th.size)
    lam = 0.0
    for _ in range(iters):
        v = v / np.linalg.norm(v)
        Hv = (loss_grad(th + h * v)[1] - loss_grad(th - h * v)[1]) / (2 * h)
        lam = v @ Hv
        v = Hv
    return lam


def part1():
    print(f"1) 선형 보간 경로 (MLP 은닉 {H}, momentum SGD lr 0.05, batch 32, 300 epoch)")
    ts = np.linspace(0, 1, 21)
    A = sgd(init(0), data_seed=100)
    B = sgd(init(1), data_seed=101)
    C = sgd(init(0), data_seed=102)                            # A와 같은 초기값, batch 순서만 다르다
    Ap = permute(A, np.random.default_rng(7).permutation(H))
    print("   경로                                  L(0)     L(1)     경로 최대   장벽(최대 - 양 끝 큰 값)")
    for name, a, b in [("다른 시드 (A -> B)", A, B), ("같은 초기값, 다른 batch 순서 (A -> C)", A, C),
                       ("뉴런 순서만 바꾼 같은 함수 (A -> A')", A, Ap)]:
        L = path(a, b, ts)
        print(f"   {name:34s} {L[0]:.4f}  {L[-1]:.4f}  {L.max():9.4f}  {L.max() - max(L[0], L[-1]):9.4f}")
    print(f"   |A - B| = {np.linalg.norm(A - B):.2f},  |A - C| = {np.linalg.norm(A - C):.2f},  |A - A'| = {np.linalg.norm(A - Ap):.2f}")
    print("   A 경로 손실(t = 0, 0.1, ..., 1, 다른 시드):", np.round(path(A, B, np.linspace(0, 1, 11)), 3))
    return A


def part2():
    print("\n2) full-batch 경사하강의 sharpness(Hessian 최대 고윳값)와 2/lr, 같은 초기값")
    checkpoints = [0, 100, 300, 1000, 3000, 10000]
    print("   lr     2/lr   " + "".join(f"{k:>8d}" for k in checkpoints) + "   최종 손실")
    for lr in [0.02, 0.05, 0.1, 0.2, 0.4]:
        th, row = init(3), []
        with np.errstate(all="ignore"):
            for k in range(checkpoints[-1] + 1):
                if k in checkpoints:
                    row.append(top_eig(th, iters=100))
                if k < checkpoints[-1]:
                    th = th - lr * loss_grad(th)[1]
        if not np.isfinite(th).all():
            print(f"   {lr:<5}  {2 / lr:5.1f}   발산")
            continue
        print(f"   {lr:<5}  {2 / lr:5.1f}   " + "".join(f"{x:8.2f}" for x in row) + f"   {loss_grad(th)[0]:.4f}")


def part3(A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    r = np.random.default_rng(5)
    d1, d2 = r.normal(size=P), r.normal(size=P)
    d1 *= np.linalg.norm(A) / np.linalg.norm(d1) * 0.1
    d2 *= np.linalg.norm(A) / np.linalg.norm(d2) * 0.1
    s = np.linspace(-1, 1, 41)
    Z = np.array([[loss_grad(A + a * d1 + b * d2)[0] for a in s] for b in s])   # (41, 41)
    fig, ax = plt.subplots(figsize=(5, 4))
    cs = ax.contourf(s, s, np.log10(Z), levels=30)
    fig.colorbar(cs, label="log10 loss")
    ax.plot(0, 0, "w*")
    ax.set_title("random 2D slice around a trained solution")
    fig.tight_layout()
    fig.savefig(OUT / "slice2d.png", dpi=120)
    print(f"\n3) 무작위 2차원 단면: 중심 손실 {Z[20, 20]:.4f}, 단면 안 최소 {Z.min():.4f}, 최대 {Z.max():.4f}"
          f" -> {OUT.name}/slice2d.png")


if __name__ == "__main__":
    A = part1()
    part2()
    part3(A)
