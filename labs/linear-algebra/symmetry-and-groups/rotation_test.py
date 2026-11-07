"""7.10 실습: 2차원 점구름을 회전시키며 일반 MLP와 회전 불변 특징 기반 모델의 출력을 비교한다.

실행: uv run --project labs python labs/linear-algebra/symmetry-and-groups/rotation_test.py

설정
- 점구름: 점 6개, x 방향으로 길쭉한 분포(표준편차 x 2.0, y 0.5). 학습 데이터는 모두 이 방향으로 정렬되어 있다.
- 스칼라 목표(회전 불변): 회전 반경 R_g = sqrt(mean |x_i - c|^2)
- 벡터 목표(회전 등변): v = mean_i |x_i - c|^2 (x_i - c)

모델
A. 좌표를 펼친 MLP (12 입력)
B. A + 무작위 회전 증강
C. 불변 특징(정렬한 쌍별 거리 15개) MLP
벡터 목표에 대해서는
C'. 불변 특징 MLP로 벡터 두 성분을 직접 예측 (출력이 회전하지 않는다)
D. 등변 구성: v_hat = mean_i g(r_i) (x_i - c), g는 r_i에 대한 RBF 기저의 선형결합(최소제곱)

평가
- test 오차: 학습과 같은 방향 / 무작위로 회전한 test 점구름
- 회전 민감도: test 점구름 하나를 36개 각도로 돌렸을 때 스칼라 예측의 표준편차(평균)
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

N_PTS = 6
rng = np.random.default_rng(0)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def clouds(n):
    return rng.normal(size=(n, N_PTS, 2)) * np.array([2.0, 0.5])          # (n, N, 2)


def rotate_all(X, thetas):
    return np.stack([x @ rot(t).T for x, t in zip(X, thetas)])


def rg(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(Xc**2, axis=-1), axis=1))               # (n,)


def vec_target(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    r2 = np.sum(Xc**2, axis=-1, keepdims=True)
    return np.mean(r2 * Xc, axis=1)                                       # (n, 2)


def raw(X):
    return X.reshape(len(X), -1)


def inv(X):
    d = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)
    iu = np.triu_indices(N_PTS, 1)
    return np.sort(d[:, iu[0], iu[1]], axis=1)                            # (n, 15)


def mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, random_state=0)


def radial_design(X, centers, width=0.6):
    """등변 모델의 설계 행렬: v_hat = Σ_m θ_m mean_i φ_m(r_i) (x_i - c) → (n, 2, M)"""
    Xc = X - X.mean(axis=1, keepdims=True)
    r = np.linalg.norm(Xc, axis=-1)                                        # (n, N)
    phi = np.exp(-((r[..., None] - centers) ** 2) / (2 * width**2))       # (n, N, M)
    return np.einsum("nim,nid->ndm", phi, Xc) / N_PTS                     # (n, 2, M)


def tiny(x):
    return f"{'< 1e-12':>16}" if x < 1e-12 else f"{x:16.2e}"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X_tr, X_te = clouds(3000), clouds(500)
    X_te_rot = rotate_all(X_te, rng.uniform(0, 2 * np.pi, len(X_te)))
    X_aug = rotate_all(X_tr, rng.uniform(0, 2 * np.pi, len(X_tr)))
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    models = {
        "A 좌표 MLP            ": (mlp().fit(raw(X_tr), rg(X_tr)), raw),
        "B 좌표 MLP + 회전 증강": (mlp().fit(raw(X_aug), rg(X_aug)), raw),
        "C 불변 특징 MLP        ": (mlp().fit(inv(X_tr), rg(X_tr)), inv),
    }
    print("스칼라 목표 R_g (회전 불변)")
    print("  모델                       test RMSE   회전한 test RMSE   회전 민감도")
    for name, (m, feat) in models.items():
        e_same = rmse(m.predict(feat(X_te)), rg(X_te))
        e_rot = rmse(m.predict(feat(X_te_rot)), rg(X_te_rot))
        sens = np.mean([np.std(m.predict(feat(np.stack([x @ rot(a).T for a in angles])))) for x in X_te[:50]])
        print(f"  {name}   {e_same:9.4f}   {e_rot:16.4f}   {tiny(sens)}")

    print("\n벡터 목표 v (회전 등변)")
    v_tr, v_te, v_te_rot = vec_target(X_tr), vec_target(X_te), vec_target(X_te_rot)
    m_inv = mlp().fit(inv(X_tr), v_tr)
    centers = np.linspace(0, 6, 12)
    A = radial_design(X_tr, centers)                                       # (n, 2, M)
    theta, *_ = np.linalg.lstsq(A.reshape(-1, len(centers)), v_tr.reshape(-1), rcond=None)
    eq = lambda X: radial_design(X, centers) @ theta                       # (n, 2)
    print("  모델                       test RMSE   회전한 test RMSE   등변성 오차")
    for name, f in [("C' 불변 특징 MLP       ", lambda X: m_inv.predict(inv(X))), ("D 등변 구성            ", eq)]:
        x0 = X_te[:50]
        R = rot(0.9)
        eq_err = rmse(f(x0 @ R.T), f(x0) @ R.T)
        print(f"  {name}   {rmse(f(X_te), v_te):9.4f}   {rmse(f(X_te_rot), v_te_rot):16.4f}   {tiny(eq_err)}")
    print(f"  (참고: 항상 0을 예측하면 test RMSE = {rmse(np.zeros_like(v_te), v_te):.4f})")


if __name__ == "__main__":
    main()
