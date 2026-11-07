"""10.3 실습: 조건수가 다른 이차함수에서 GD·momentum·SGD의 학습률 범위를 실험하고 궤적을 시각화한다.

1) 최대 안정 학습률: 무작위 회전을 준 이차함수 f(x) = x^T H x / 2에서, 2000 step 뒤 |x|가 처음보다 커지지 않는
   가장 큰 학습률을 이분법으로 찾아 이론값 2/L (GD), 2(1+beta)/L (heavy ball)과 비교한다.
2) 같은 상대 학습률 lr * L에서 조건수에 따라 반복 수가 어떻게 달라지는지 본다.
3) 2차원 궤적 그림: outputs/trajectories.png

실행: uv run --project labs python labs/optimization/first-order-methods/lr_sweep.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"


def random_quadratic(kappa, d, rng):
    """고윳값이 1..kappa(로그 간격)이고 고유벡터가 무작위인 (d, d) Hessian."""
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q @ np.diag(np.geomspace(1, kappa, d)) @ Q.T


def stable(H, lr, beta=0.0, steps=2000, seed=0):
    """heavy ball(beta=0이면 GD)이 steps 뒤 |x| <= |x0|이면 안정으로 본다."""
    x = np.random.default_rng(seed).normal(size=H.shape[0])
    n0, v = np.linalg.norm(x), np.zeros_like(x)
    for _ in range(steps):
        v = beta * v - lr * H @ x
        x = x + v
        if not np.isfinite(x).all() or np.linalg.norm(x) > 1e6 * n0:
            return False
    return np.linalg.norm(x) <= n0


def max_stable_lr(is_stable, lo, hi, iters=40):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if is_stable(mid) else (lo, mid)
    return lo


def steps_to_tol(H, lr, tol=1e-6, max_steps=200_000):
    x = np.ones(H.shape[0])
    n0 = np.linalg.norm(x)
    for k in range(1, max_steps + 1):
        x = x - lr * H @ x
        if np.linalg.norm(x) < tol * n0:
            return k
        if np.linalg.norm(x) > 1e6:
            return None
    return None


def main():
    rng = np.random.default_rng(0)
    print("1) 최대 안정 학습률 (무작위 회전 이차함수, d = 10)")
    print("   (실험값 x L로 표시. 이론: GD 2, heavy ball 2(1 + beta) = 3.8)")
    print("   kappa       L    GD 실험 x L   HB(0.9) 실험 x L")
    for kappa in [10, 100, 1000]:
        H = random_quadratic(kappa, 10, rng)
        L = np.linalg.eigvalsh(H)[-1]
        gd = max_stable_lr(lambda lr: stable(H, lr), 0.0, 10.0)
        hb = max_stable_lr(lambda lr: stable(H, lr, beta=0.9), 0.0, 10.0)
        print(f"  {kappa:5d}  {L:7.1f}   {gd * L:11.3f}   {hb * L:16.3f}")

    print("\n2) 상대 학습률 lr * L에 따른 반복 수 (|x| 1e-6배까지, '-'는 발산 또는 20만 step 초과)")
    ratios = [0.5, 1.0, 1.5, 1.9, 1.99, 2.01]
    print("   kappa  " + "  ".join(f"lr*L={r:<5}" for r in ratios))
    for kappa in [10, 100, 1000]:
        H = random_quadratic(kappa, 10, rng)
        L = np.linalg.eigvalsh(H)[-1]
        row = [steps_to_tol(H, r / L) for r in ratios]
        print(f"  {kappa:5d}  " + "  ".join(f"{(s if s else '-')!s:>10}" for s in row))

    OUT.mkdir(exist_ok=True)
    Hq = np.diag([1.0, 20.0])
    fig, ax = plt.subplots(figsize=(7, 4))
    g = np.linspace(-11, 11, 200)
    A, Bm = np.meshgrid(g, np.linspace(-3, 3, 120))
    ax.contour(A, Bm, 0.5 * (A**2 + 20 * Bm**2), levels=15, linewidths=0.5, colors="gray")
    for name, lr, beta, nes in [("GD lr=1.9/L", 1.9 / 20, 0.0, False), ("heavy ball 0.9", 1 / 20, 0.9, False),
                                ("Nesterov 0.9", 1 / 20, 0.9, True)]:
        x, v, path = np.array([10.0, 2.0]), np.zeros(2), [np.array([10.0, 2.0])]
        for _ in range(60):
            look = x + beta * v if nes else x
            v = beta * v - lr * np.diag(Hq) * look
            x = x + v
            path.append(x.copy())
        P = np.array(path)
        ax.plot(P[:, 0], P[:, 1], ".-", ms=3, lw=0.8, label=name)
    rng_s = np.random.default_rng(0)
    x, path = np.array([10.0, 2.0]), [np.array([10.0, 2.0])]
    for _ in range(60):                                   # 잡음 섞인 gradient(SGD 흉내)
        x = x - 0.04 * (np.diag(Hq) * x + rng_s.normal(size=2) * 2.0)
        path.append(x.copy())
    P = np.array(path)
    ax.plot(P[:, 0], P[:, 1], ".-", ms=3, lw=0.8, label="noisy GD (SGD-like)")
    ax.set_title("f = (x^2 + 20 y^2) / 2, 60 steps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=120)


if __name__ == "__main__":
    main()
