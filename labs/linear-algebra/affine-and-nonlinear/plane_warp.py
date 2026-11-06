"""7.2 실습: 활성화 없는 MLP와 ReLU MLP가 2차원 평면을 어떻게 나누는지 본다.

ReLU MLP에서는 입력마다 '어떤 뉴런이 켜졌는가'(활성 패턴)가 정해지고, 같은 패턴을 가진 입력들에는
같은 아핀 사상이 적용된다. 격자 위에서 서로 다른 활성 패턴의 수를 세어 '활성 영역' 수를 추정한다.

- 격자 추정은 아주 작은 영역을 놓칠 수 있으므로 실제 활성 영역 수의 하한이다.
- 이웃한 두 활성 영역의 아핀 사상이 우연히 같을 수 있으므로, 활성 영역 수는 '서로 다른 아핀 조각' 수의 상한이다.
- 활성화가 없으면 정의상 영역은 1개다(세지 않고 1로 둔다).

실행: uv run --project labs python labs/linear-algebra/affine-and-nonlinear/plane_warp.py
결과 그림: labs/linear-algebra/affine-and-nonlinear/outputs/plane_warp.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"
GRID = 1000   # 격자 해상도 (한 변의 점 수)


def init_mlp(widths, rng):
    """widths = [입력, 은닉1, ..., 출력]. 층마다 (W, b)."""
    return [(rng.normal(size=(m, n)) / np.sqrt(n), rng.normal(size=m) * 0.5) for n, m in zip(widths[:-1], widths[1:])]


def activation_pattern(params, X):
    """X: (N, 2). 은닉층들의 켜짐 여부를 이어 붙인 (N, 은닉 뉴런 수) 불리언 배열."""
    h, pattern = X, []
    for W, b in params[:-1]:
        h = h @ W.T + b                      # (N, m)
        pattern.append(h > 0)
        h = np.maximum(0.0, h)
    return np.concatenate(pattern, axis=1)


def count_regions(pattern):
    return len(np.unique(pattern, axis=0))


def grid_points():
    g = np.linspace(-3, 3, GRID)
    XX, YY = np.meshgrid(g, g)
    return XX, np.column_stack([XX.ravel(), YY.ravel()])     # (GRID*GRID, 2)


CONFIGS = [("은닉 [8]", "hidden [8]", [2, 8, 1]),
           ("은닉 [16]", "hidden [16]", [2, 16, 1]),
           ("은닉 [8, 8]", "hidden [8, 8]", [2, 8, 8, 1])]


def main():
    XX, X = grid_points()
    OUT.mkdir(exist_ok=True)

    # 1) 시드 0의 그림과 영역 수
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(12, 4))
    print("시드 0, 격자 1000x1000, 범위 [-3, 3]^2")
    print("활성화 없음 (정의상)  영역    1개")
    for ax, (name, label, widths) in zip(axes, CONFIGS):
        pattern = activation_pattern(init_mlp(widths, np.random.default_rng(0)), X)
        ids = np.unique(pattern, axis=0, return_inverse=True)[1].reshape(XX.shape)
        n = ids.max() + 1
        ax.imshow(ids % 20, extent=(-3, 3, -3, 3), origin="lower", cmap="tab20", interpolation="nearest")
        ax.set_title(f"ReLU, {label}\n{n} activation regions")    # 그림 제목은 한글 글꼴이 없는 환경을 위해 영어
        ax.set_xticks([]), ax.set_yticks([])
        print(f"ReLU {name:10s}  영역 {n:4d}개")
    fig.tight_layout()
    fig.savefig(OUT / "plane_warp.png", dpi=120)

    # 2) 시드 0–9의 평균: 같은 총 뉴런 수(16개)에서 폭과 깊이 비교
    print("\n시드 0–9 평균")
    for name, _, widths in CONFIGS:
        counts = [count_regions(activation_pattern(init_mlp(widths, np.random.default_rng(s)), X)) for s in range(10)]
        print(f"ReLU {name:10s}  평균 {np.mean(counts):6.1f}개  (최소 {min(counts)}, 최대 {max(counts)})")


if __name__ == "__main__":
    main()
