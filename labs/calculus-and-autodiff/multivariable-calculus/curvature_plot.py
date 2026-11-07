"""8.1 실습: 2변수 손실함수의 등고선 위에 gradient와 Hessian 고유벡터를 그린다.

손실은 Rosenbrock 함수 f(x, y) = (1 - x)^2 + b (y - x^2)^2 (b = 10)이다.
각 점에서 음의 gradient(내려가는 방향)를 화살표로, Hessian 고유벡터를 고윳값 부호에 따라
실선(양의 곡률)·점선(음의 곡률)으로 그리고, 수치를 표로 출력한다.

실행: uv run --project labs python labs/calculus-and-autodiff/multivariable-calculus/curvature_plot.py
결과 그림: labs/calculus-and-autodiff/multivariable-calculus/outputs/curvature.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = 10.0
OUT = Path(__file__).parent / "outputs"


def f(x, y):
    return (1 - x) ** 2 + B * (y - x**2) ** 2


def grad(x, y):
    return np.array([-2 * (1 - x) - 4 * B * x * (y - x**2), 2 * B * (y - x**2)])  # (2,)


def hess(x, y):
    return np.array([[2 - 4 * B * y + 12 * B * x**2, -4 * B * x],
                     [-4 * B * x, 2 * B]])  # (2, 2)


def numeric_grad(x, y, eps=1e-6):
    return np.array([(f(x + eps, y) - f(x - eps, y)) / (2 * eps),
                     (f(x, y + eps) - f(x, y - eps)) / (2 * eps)])


def numeric_hess(x, y, eps=1e-5):
    cols = [(grad(x + eps, y) - grad(x - eps, y)) / (2 * eps),
            (grad(x, y + eps) - grad(x, y - eps)) / (2 * eps)]
    return np.stack(cols, axis=1)


POINTS = [(1.0, 1.0), (0.0, 0.5), (-1.0, 1.5), (0.5, -0.3), (-0.6, 0.0)]


def main():
    print(f"{'점':>14} {'f':>8} {'|grad|':>8} {'λ1':>9} {'λ2':>9}  곡률")
    for x, y in POINTS:
        g, H = grad(x, y), hess(x, y)
        assert np.allclose(g, numeric_grad(x, y), atol=1e-5)
        assert np.allclose(H, numeric_hess(x, y), atol=1e-4)
        assert np.allclose(H, H.T)
        lam = np.linalg.eigvalsh(H)
        kind = "볼록(양의 정부호)" if lam[0] > 0 else ("부정부호" if lam[1] > 0 else "오목")
        print(f"{str((x, y)):>14} {f(x, y):8.3f} {np.linalg.norm(g):8.3f} {lam[0]:9.3f} {lam[1]:9.3f}  {kind}")

    xs, ys = np.linspace(-1.5, 1.5, 400), np.linspace(-0.8, 2.0, 400)
    X, Y = np.meshgrid(xs, ys)  # (400, 400)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contour(X, Y, f(X, Y), levels=np.logspace(-1.5, 2, 18), cmap="viridis", linewidths=0.8)
    for x, y in POINTS:
        g = grad(x, y)
        d = -g / (np.linalg.norm(g) + 1e-12) * 0.25
        if np.linalg.norm(g) > 1e-9:
            ax.annotate("", xy=(x + d[0], y + d[1]), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
        lam, V = np.linalg.eigh(hess(x, y))
        for k in range(2):
            v = V[:, k] * 0.2
            style = "-" if lam[k] > 0 else ":"
            ax.plot([x - v[0], x + v[0]], [y - v[1], y + v[1]], style, color="black", lw=1.5)
        ax.plot(x, y, "o", color="black", ms=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Rosenbrock (b=10): -grad (red), Hessian eigenvectors (solid +, dotted -)")
    ax.set_aspect("equal")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "curvature.png", dpi=120, bbox_inches="tight")
    print("그림 저장:", (OUT / "curvature.png").relative_to(Path(__file__).parents[3]))


if __name__ == "__main__":
    main()
