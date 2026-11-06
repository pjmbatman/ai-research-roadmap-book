"""7.2 실습: 활성화 없는 MLP와 ReLU MLP가 2차원 평면을 어떻게 나누는지 본다.

ReLU MLP는 입력마다 '어떤 뉴런이 켜졌는가'(활성 패턴)가 정해지고, 같은 패턴을 가진 입력들에는
같은 아핀 사상이 적용된다. 그래서 격자 위의 서로 다른 활성 패턴 수가 선형 영역 수의 추정치가 된다.

실행: uv run --project labs python labs/linear-algebra/affine-and-nonlinear/plane_warp.py
결과 그림: labs/linear-algebra/affine-and-nonlinear/outputs/plane_warp.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"


def init_mlp(widths, rng):
    """widths = [입력, 은닉1, ..., 출력]. 층마다 (W, b)."""
    return [(rng.normal(size=(m, n)) / np.sqrt(n), rng.normal(size=m) * 0.5) for n, m in zip(widths[:-1], widths[1:])]


def forward(params, X, activation):
    """X: (N, 2). 마지막 층 출력과 은닉층 활성 패턴(켜짐 여부)을 반환한다."""
    h, pattern = X, []
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b                      # (N, m)
        if i < len(params) - 1 and activation:
            pattern.append(h > 0)
            h = np.maximum(0.0, h)
    return h, (np.concatenate(pattern, axis=1) if pattern else np.zeros((len(X), 0), bool))


def count_regions(pattern):
    return len(np.unique(pattern, axis=0)) if pattern.shape[1] else 1


def main():
    rng = np.random.default_rng(0)
    g = np.linspace(-3, 3, 400)
    XX, YY = np.meshgrid(g, g)
    X = np.column_stack([XX.ravel(), YY.ravel()])            # (160000, 2)

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 그림 제목은 한글 글꼴이 없는 환경에서도 보이도록 영어로 쓴다.
    configs = [("활성화 없음, 은닉 [8, 8]", "no activation, hidden [8, 8]", [2, 8, 8, 1], False),
               ("ReLU, 은닉 [8]", "ReLU, hidden [8]", [2, 8, 1], True),
               ("ReLU, 은닉 [8, 8]", "ReLU, hidden [8, 8]", [2, 8, 8, 1], True)]
    for ax, (name, label, widths, act) in zip(axes, configs):
        params = init_mlp(widths, rng)
        _, pattern = forward(params, X, act)
        n = count_regions(pattern)
        ids = np.unique(pattern, axis=0, return_inverse=True)[1].reshape(XX.shape) if pattern.shape[1] else np.zeros(XX.shape)
        ax.imshow(ids % 20, extent=(-3, 3, -3, 3), origin="lower", cmap="tab20", interpolation="nearest")
        ax.set_title(f"{label}\n{n} linear regions")
        ax.set_xticks([]), ax.set_yticks([])
        print(f"{name:20s} 선형 영역 {n:4d}개")
    fig.tight_layout()
    fig.savefig(OUT / "plane_warp.png", dpi=120)


if __name__ == "__main__":
    main()
