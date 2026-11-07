"""9.4 실습: 비선형 연립방정식을 Newton 방법으로 풀고, 초기값에 따른 수렴 영역(basin of attraction)을 시각화한다.

방정식: 복소수 z^3 = 1을 실수 2변수 연립방정식으로 쓴 것
    F(x, y) = [x^3 - 3 x y^2 - 1,  3 x^2 y - y^3] = 0
근은 세 개: (1, 0), (-1/2, ±sqrt(3)/2).
[-2, 2]^2 격자의 각 점에서 Newton(순수)과 damped Newton(|F|가 줄 때까지 step을 반으로)을 돌려
어느 근으로 갔는지, 몇 번 만에 갔는지를 센다. 그림은 outputs/newton_basins.png에 저장한다.

실행: uv run --project labs python labs/numerical-methods/nonlinear-equations/newton_basins.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOTS = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])   # (3, 2)
OUT = Path(__file__).parent / "outputs"


def F(x, y):
    return np.stack([x**3 - 3 * x * y**2 - 1, 3 * x**2 * y - y**3])               # (2, ...)


def newton_grid(x, y, max_iter=50, tol=1e-10, damped=False):
    """격자 전체에 대해 벡터화한 Newton. 2x2 Jacobian은 Cramer 공식으로 푼다."""
    x, y = x.copy(), y.copy()
    iters = np.full(x.shape, max_iter)
    done = np.zeros(x.shape, bool)
    for k in range(max_iter):
        f1, f2 = F(x, y)
        res = np.hypot(f1, f2)
        newly = (res < tol) & ~done
        iters[newly], done = k, done | newly
        a, b = 3 * x**2 - 3 * y**2, -6 * x * y                                       # J = [[a, b], [c, d]]
        c, d = 6 * x * y, 3 * x**2 - 3 * y**2
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)                            # 특이 Jacobian 보호
        dx, dy = (d * f1 - b * f2) / det, (a * f2 - c * f1) / det
        t = np.ones_like(x)
        if damped:
            for _ in range(30):                                                      # backtracking
                g1, g2 = F(x - t * dx, y - t * dy)
                worse = (np.hypot(g1, g2) >= res) & ~done
                if not worse.any():
                    break
                t = np.where(worse, t / 2, t)
        x = np.where(done, x, x - t * dx)
        y = np.where(done, y, y - t * dy)
    dist = np.linalg.norm(np.stack([x, y])[None] - ROOTS[:, :, None, None], axis=1)  # (3, H, W)
    which = np.where(dist.min(0) < 1e-6, dist.argmin(0), -1)                         # -1 = 수렴 실패
    return which, iters


def summarize(name, which, iters):
    total = which.size
    shares = [np.mean(which == r) for r in range(3)]
    ok = which >= 0
    print(f"{name:13s} 근1 {shares[0]:6.1%}  근2 {shares[1]:6.1%}  근3 {shares[2]:6.1%}  "
          f"실패 {np.mean(~ok):6.2%}  평균 반복 {iters[ok].mean():5.2f}  최대 반복 {iters[ok].max():3d}  "
          f"(격자 {int(np.sqrt(total))}^2)")


def main():
    n = 400
    g = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(g, g)                                                         # (n, n)
    results = {}
    for name, damped in [("Newton", False), ("damped Newton", True)]:
        which, iters = newton_grid(X, Y, damped=damped)
        results[name] = (which, iters)
        summarize(name, which, iters)

    # basin 경계 근처: 0.01 떨어진 초기값들이 서로 다른 근으로 간다. 원점은 Jacobian이 특이하다.
    for x0, y0 in [(0.9, 0.1), (-1.14, 0.2), (-1.13, 0.2), (-1.12, 0.2), (0.0, 0.0)]:
        w, it = newton_grid(np.array([[x0]]), np.array([[y0]]))
        dest = f"근{w[0, 0] + 1} ({it[0, 0]}번)" if w[0, 0] >= 0 else "수렴 실패"
        print(f"초기값 ({x0:5.2f}, {y0:4.1f}) → {dest}")

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    cmap = matplotlib.colors.ListedColormap(["black", "#4c72b0", "#dd8452", "#55a868"])
    for ax, (name, (which, iters)) in zip(axes, results.items()):
        shade = 1 - 0.6 * np.clip(iters / 20, 0, 1)
        ax.imshow(which + 1, extent=[-2, 2, -2, 2], origin="lower", cmap=cmap, vmin=0, vmax=3, alpha=1.0)
        ax.imshow(shade, extent=[-2, 2, -2, 2], origin="lower", cmap="gray", alpha=0.35)
        ax.plot(ROOTS[:, 0], ROOTS[:, 1], "w*", ms=12)
        ax.set_title(f"{name}: basins of z^3 = 1")
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
    fig.tight_layout()
    fig.savefig(OUT / "newton_basins.png", dpi=120)
    print(f"그림: {OUT / 'newton_basins.png'}")


if __name__ == "__main__":
    main()
