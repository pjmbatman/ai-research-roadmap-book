"""8.3 실습: 2차원 분포에 가역 변환을 적용해 밀도 변환 공식을 Monte Carlo 표본으로 검증한다.

z ~ N(0, I_2)에 가역 변환
    y1 = z1
    y2 = z2 * exp(a z1) + b z1^2
을 적용한다(바나나 모양 분포). Jacobian은 하삼각이고 det J = exp(a z1)이다.
밀도 변환 공식 p_Y(y) = p_Z(g^{-1}(y)) / |det J_g(g^{-1}(y))|를
(1) 상자 확률, (2) 전체 질량, (3) 평균 로그우도로 표본과 비교한다.

실행: uv run --project labs python labs/calculus-and-autodiff/integration/density_transform.py
결과 그림: labs/calculus-and-autodiff/integration/outputs/density_transform.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A_, B_ = 0.5, 0.6
OUT = Path(__file__).parent / "outputs"


def forward(z):
    """z: (N, 2) -> y: (N, 2), log|det J|: (N,)"""
    y = np.stack([z[:, 0], z[:, 1] * np.exp(A_ * z[:, 0]) + B_ * z[:, 0] ** 2], axis=1)
    return y, A_ * z[:, 0]


def inverse(y):
    """y: (N, 2) -> z: (N, 2)"""
    z1 = y[..., 0]
    z2 = (y[..., 1] - B_ * z1**2) * np.exp(-A_ * z1)
    return np.stack([z1, z2], axis=-1)


def log_pz(z):
    return -0.5 * np.sum(z**2, axis=-1) - np.log(2 * np.pi)


def log_py(y):
    z = inverse(y)
    return log_pz(z) - A_ * z[..., 0]          # log p_Z(z) - log|det J_g(z)|


def log_py_wrong(y):
    return log_pz(inverse(y))                   # Jacobian 항 누락


def grid_integral(fn, x_lo, x_hi, y_lo, y_hi, n=600):
    xs = np.linspace(x_lo, x_hi, n + 1)
    ys = np.linspace(y_lo, y_hi, n + 1)
    xc, yc = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    G = np.stack(np.meshgrid(xc, yc, indexing="ij"), axis=-1)   # (n, n, 2)
    return np.exp(fn(G)).sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])


def main():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2_000_000, 2))
    y, logdet = forward(z)
    assert np.allclose(inverse(y), z)

    print("[1] 상자 확률: 표본 비율 vs 변환 공식의 적분")
    boxes = [(-1, 0, -1, 1), (0, 1, 0, 2), (1, 2, 0.5, 3), (-2, -1, 0, 2), (-0.5, 0.5, -0.5, 0.5)]
    for x_lo, x_hi, y_lo, y_hi in boxes:
        inside = (y[:, 0] > x_lo) & (y[:, 0] < x_hi) & (y[:, 1] > y_lo) & (y[:, 1] < y_hi)
        mc = inside.mean()
        se = np.sqrt(mc * (1 - mc) / len(y))
        exact = grid_integral(log_py, x_lo, x_hi, y_lo, y_hi)
        wrong = grid_integral(log_py_wrong, x_lo, x_hi, y_lo, y_hi)
        print(f"  [{x_lo:4.1f},{x_hi:4.1f}]x[{y_lo:4.1f},{y_hi:4.1f}]  표본 {mc:.4f} (±{2 * se:.4f})"
              f"  공식 {exact:.4f}  Jacobian 누락 {wrong:.4f}")

    print("[2] 전체 질량 ([-6, 6] x [-15, 40] 격자)")
    print(f"  올바른 밀도 {grid_integral(log_py, -6, 6, -15, 40, n=1500):.4f}"
          f"   Jacobian 누락 {grid_integral(log_py_wrong, -6, 6, -15, 40, n=1500):.4f}")

    print("[3] 평균 로그우도 (normalizing flow의 학습 목표와 같은 양)")
    ll = log_py(y).mean()
    ll_parts = (log_pz(z) - logdet).mean()
    print(f"  공식으로 계산한 mean log p_Y(y)   {ll:.4f}")
    print(f"  log p_Z(z) - log|det J| 의 평균   {ll_parts:.4f}")
    print(f"  이론값 -(1 + log 2 pi)            {-(1 + np.log(2 * np.pi)):.4f}")

    xs, ys = np.linspace(-3, 3, 300), np.linspace(-3, 8, 300)
    G = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].scatter(y[:5000, 0], y[:5000, 1], s=1, alpha=0.3)
    ax[0].set_title("samples y = g(z)")
    ax[1].contourf(xs, ys, np.exp(log_py(G)), levels=20, cmap="viridis")
    ax[1].set_title("p_Y(y) from change of variables")
    for a in ax:
        a.set_xlim(-3, 3)
        a.set_ylim(-3, 8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "density_transform.png", dpi=110, bbox_inches="tight")
    print("그림 저장:", (OUT / "density_transform.png").relative_to(Path(__file__).parents[3]))


if __name__ == "__main__":
    main()
