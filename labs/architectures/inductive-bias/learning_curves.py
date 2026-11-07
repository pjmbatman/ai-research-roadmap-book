"""18.1 실습: 구조적 가정이 다른 세 모델의 데이터 크기별 성능 곡선.

세 모델은 모두 '고정된 무작위 특징 + 마지막 층 ridge 회귀'로 학습한다. 특징을 만드는 방식만 다르다.
  - 선형     : 입력 그대로 (가정: 출력이 입력의 선형 함수)
  - MLP      : 입력 전체를 섞는 무작위 ReLU 특징 2000개 (가정 없음)
  - 합성곱   : 길이 3 창마다 같은 무작위 ReLU 특징 100개를 적용하고 위치에 대해 합 (국소성·가중치 공유·이동 불변)
목표 두 가지:
  - 위치 무관  y = Σ_t ReLU(x_t x_{t+1}) / √T          (합성곱 모델의 가정과 맞음)
  - 위치 의존  y = Σ_t w_t ReLU(x_t x_{t+1}) / √T, w_t = 0 → 2  (이동 불변 가정이 틀림)

실행: uv run --project labs python labs/architectures/inductive-bias/learning_curves.py
"""
import numpy as np

T, K = 32, 3
SIZES = [20, 100, 1000, 10000, 40000]
SEEDS = [0, 1, 2]


def target(X, w_pos):
    return np.maximum(0, X * np.roll(X, -1, axis=1)) @ w_pos / np.sqrt(T)


def make_models(rng, m_mlp=2000, m_cnn=100):
    A_mlp, b_mlp = rng.normal(size=(T, m_mlp)) / np.sqrt(T), rng.normal(size=m_mlp)
    A_cnn, b_cnn = rng.normal(size=(K, m_cnn)), rng.normal(size=m_cnn)
    relu = lambda z: np.maximum(0.0, z)
    return {
        "선형": lambda X: X,
        "MLP": lambda X: relu(X @ A_mlp + b_mlp),
        "합성곱": lambda X: relu(np.stack([np.roll(X, -j, axis=1) for j in range(K)], -1) @ A_cnn + b_cnn).sum(1),
    }


def r2(feat, Xtr, ytr, Xte, yte, lam=1.0):
    F, G = np.c_[feat(Xtr), np.ones(len(Xtr))], np.c_[feat(Xte), np.ones(len(Xte))]
    w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ ytr)
    return 1 - np.mean((G @ w - yte) ** 2) / np.var(yte)


def main():
    targets = {"위치 무관 목표": np.ones(T), "위치 의존 목표": np.linspace(0, 2, T)}
    for tname, w_pos in targets.items():
        res = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            models = make_models(rng)
            Xte = rng.normal(size=(5000, T))
            yte = target(Xte, w_pos)
            for N in SIZES:
                Xtr = rng.normal(size=(N, T))
                ytr = target(Xtr, w_pos)
                for mname, f in models.items():
                    res.setdefault((mname, N), []).append(r2(f, Xtr, ytr, Xte, yte))
        print(f"{tname}: test R² (시드 {len(SEEDS)}개 평균)")
        print("      N" + "".join(f"{m:>10}" for m in ["선형", "MLP", "합성곱"]))
        for N in SIZES:
            print(f"{N:7d}" + "".join(f"{np.mean(res[(m, N)]):10.3f}" for m in ["선형", "MLP", "합성곱"]))
        print()


if __name__ == "__main__":
    main()
