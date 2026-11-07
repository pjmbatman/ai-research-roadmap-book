"""24.8 실습: 같은 입자 데이터에서 대칭을 넣은 모델과 넣지 않은 모델의 데이터 효율·회전 일반화를 비교한다.

입자 4개의 Morse 쌍 퍼텐셜 에너지를 예측한다. 학습 구조는 x축으로 길게 정렬된 분포에서 뽑고,
시험은 같은 분포(같은 방향)와 무작위로 회전한 분포에서 한다. 표현 세 가지를 RBF 커널 ridge로 비교한다.

- 좌표: 좌표를 펼쳐 그대로 넣는다(대칭 없음).
- 좌표+증강: 학습 구조마다 무작위 회전 2개를 더한다(대칭을 데이터로 권장).
- 불변 특징: 정렬한 쌍별 거리(회전·반사·평행이동·순열에 정확히 불변).

데이터 시드 5개의 평균 RMSE를 보고한다.

실행: uv run --project labs python labs/scientific-ml/geometric-models/equivariance_experiment.py
"""
import numpy as np

IU = np.triu_indices(4, 1)
SCALE = np.array([1.5, 0.8, 0.4])


def rotation(q):
    w, a, b, c = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                     [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                     [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)]])


def pair_dist(X):
    return np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)[:, IU[0], IU[1]]


def energy(X):
    return np.sum((1 - np.exp(-(pair_dist(X) - 1.0))) ** 2 - 1, axis=1)


def rotate_all(X, r):
    return np.stack([xb @ rotation(r.normal(size=4)).T for xb in X])


REPS = {
    "좌표": lambda X: X.reshape(len(X), -1),
    "불변 특징": lambda X: np.sort(pair_dist(X), axis=1),
}


def krr(F_tr, y_tr, F_te, lam=1e-3):
    sq = lambda a, b: np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T
    K = lambda a, b: np.exp(-0.5 * sq(a, b) / np.median(sq(F_tr, F_tr)))
    alpha = np.linalg.solve(K(F_tr, F_tr) + lam * np.eye(len(F_tr)), y_tr - y_tr.mean())
    return K(F_te, F_tr) @ alpha + y_tr.mean()


def run(seed, sizes):
    r = np.random.default_rng(seed)
    X_pool, X_te = r.normal(size=(max(sizes), 4, 3)) * SCALE, r.normal(size=(500, 4, 3)) * SCALE
    X_rot = rotate_all(X_te, r)
    out = {}
    for n in sizes:
        X_tr = X_pool[:n]
        X_aug = np.concatenate([X_tr, rotate_all(X_tr, r), rotate_all(X_tr, r)])
        for name, f, Xa in [("좌표", REPS["좌표"], X_tr), ("좌표+증강", REPS["좌표"], X_aug), ("불변 특징", REPS["불변 특징"], X_tr)]:
            for test_name, Xt in [("같은 방향", X_te), ("회전", X_rot)]:
                pred = krr(f(Xa), energy(Xa), f(Xt))
                out[(name, n, test_name)] = np.sqrt(np.mean((pred - energy(Xt)) ** 2))
    return out


if __name__ == "__main__":
    sizes = [25, 50, 100, 200, 400]
    runs = [run(seed, sizes) for seed in range(5)]
    print(f"시드 5개 평균 RMSE (같은 방향 / 회전), 시험 구조 500개, 에너지 표준편차 {np.std(energy(np.random.default_rng(0).normal(size=(5000, 4, 3)) * SCALE)):.2f}")
    print("N    " + "".join(f"{name:>18}" for name in ["좌표", "좌표+증강", "불변 특징"]))
    for n in sizes:
        cells = []
        for name in ["좌표", "좌표+증강", "불변 특징"]:
            a = np.mean([o[(name, n, "같은 방향")] for o in runs])
            b = np.mean([o[(name, n, "회전")] for o in runs])
            cells.append(f"{a:8.2f} / {b:5.2f}")
        print(f"{n:<5}" + "".join(f"{c:>18}" for c in cells))
