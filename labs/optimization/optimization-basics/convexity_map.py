"""10.1 실습: 여러 손실함수의 볼록성을 Hessian으로 판별하고 등고선과 함께 시각화한다.

각 손실은 파라미터 두 개 (a, b)의 함수다. [-3, 3]^2 위 격자의 각 점에서 중앙 차분으로 Hessian을 구하고,
최소 고윳값이 음수인 점(국소적으로 볼록이 아닌 점)의 비율을 센다.

- 격자 위 모든 점에서 Hessian이 PSD이면 '이 영역에서 볼록일 가능성이 높다'는 증거일 뿐 증명은 아니다.
- 한 점에서라도 Hessian에 음의 고윳값이 있으면 그 함수는 (그 영역을 포함하는 볼록 집합 위에서) 볼록이 아니다.

실행: uv run --project labs python labs/optimization/optimization-basics/convexity_map.py
결과 그림: labs/optimization/optimization-basics/outputs/convexity_map.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"
H = 1e-3      # 중앙 차분 간격
TOL = -1e-6   # 수치 오차를 감안한 PSD 판정 기준

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, size=200)                                   # (N,) 입력
y_lin = 1.5 * x - 0.5 + 0.3 * rng.normal(size=200)                 # 선형회귀용 목표
y_cls = (x + 0.3 * rng.normal(size=200) > 0).astype(float)         # 이진 분류 목표
y_nn = np.tanh(1.5 * x) + 0.1 * rng.normal(size=200)               # 은닉 뉴런 하나짜리 신경망용 목표


def mse_linear(a, b):
    """선형회귀 MSE: mean((a x + b - y)^2)"""
    return np.mean((a[..., None] * x + b[..., None] - y_lin) ** 2, axis=-1)


def logistic(a, b):
    """로지스틱 회귀 음의 로그우도: mean(log(1 + e^z) - y z), z = a x + b"""
    z = a[..., None] * x + b[..., None]
    return np.mean(np.logaddexp(0.0, z) - y_cls * z, axis=-1)


def tiny_nn(a, b):
    """은닉 뉴런 하나: mean((b tanh(a x) - y)^2). (a, b)와 (-a, -b)가 같은 함수를 만든다."""
    return np.mean((b[..., None] * np.tanh(a[..., None] * x) - y_nn) ** 2, axis=-1)


def rosenbrock(a, b):
    return (1 - a) ** 2 + 100 * (b - a**2) ** 2


def double_well(a, b):
    return a**4 / 4 - a**2 / 2 + 0.2 * a + b**2 / 2


LOSSES = {   # 표 이름: (그림 제목, 함수). 그림 제목은 한글 글꼴이 없는 환경을 위해 영어로 쓴다.
    "선형회귀 MSE": ("linear regression MSE", mse_linear),
    "로지스틱 회귀": ("logistic regression NLL", logistic),
    "뉴런 1개 신경망": ("1-neuron network MSE", tiny_nn),
    "Rosenbrock": ("Rosenbrock", rosenbrock),
    "이중 우물": ("tilted double well", double_well),
}


def hessian_min_eig(f, A, B):
    """격자 (A, B)의 각 점에서 2x2 Hessian의 최소 고윳값."""
    faa = (f(A + H, B) - 2 * f(A, B) + f(A - H, B)) / H**2
    fbb = (f(A, B + H) - 2 * f(A, B) + f(A, B - H)) / H**2
    fab = (f(A + H, B + H) - f(A + H, B - H) - f(A - H, B + H) + f(A - H, B - H)) / (4 * H**2)
    return (faa + fbb) / 2 - np.sqrt(((faa - fbb) / 2) ** 2 + fab**2)


def main():
    OUT.mkdir(exist_ok=True)
    g = np.linspace(-3, 3, 201)
    A, B = np.meshgrid(g, g)                                       # (201, 201)
    fig, axes = plt.subplots(1, len(LOSSES), figsize=(4 * len(LOSSES), 4))
    print(f"격자 201x201, 범위 [-3, 3]^2, PSD 판정 기준 최소 고윳값 > {TOL}")
    print("최소 고윳값  PSD 비율   손실: 판정")
    for ax, (name, (title, f)) in zip(axes, LOSSES.items()):
        lam = hessian_min_eig(f, A, B)
        psd = lam > TOL
        verdict = "볼록일 가능성 높음" if psd.all() else "볼록 아님"
        print(f"{lam.min():10.3f}  {psd.mean():8.3f}   {name}: {verdict}")
        Z = f(A, B)
        ax.contour(A, B, np.log1p(Z - Z.min()), levels=20, linewidths=0.6)
        ax.contourf(A, B, ~psd, levels=[0.5, 1.5], colors=["tab:red"], alpha=0.25)
        ax.set_title(f"{title}\nred: Hessian not PSD", fontsize=9)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "convexity_map.png", dpi=120)


if __name__ == "__main__":
    main()
