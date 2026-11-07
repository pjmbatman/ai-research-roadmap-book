"""24.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

모든 셀은 NumPy만 쓰는 브라우저 셀이다(exec="browser", session="geometric").
실행: uv run --project labs python labs/scientific-ml/geometric-models/cells.py
"""
# %% cell 1 — E(n) 등변 message passing 층과 등변성 검증
import numpy as np

rng = np.random.default_rng(0)
N, D_H, D_M = 6, 4, 8                                    # 노드 수, 노드 특징 차원, 메시지 차원
def mlp(d_in, d_out):                                    # 무작위 가중치의 작은 MLP (학습은 생략)
    W1, W2 = rng.normal(size=(d_in, 16)) / np.sqrt(d_in), rng.normal(size=(16, d_out)) / 4
    return lambda z: np.tanh(z @ W1) @ W2
phi_e, phi_x, phi_h = mlp(2 * D_H + 1, D_M), mlp(D_M, 1), mlp(D_H + D_M, D_H)

def egnn_layer(h, x, r_cut=2.5):                         # h: (N, D_H) 불변 특징, x: (N, 3) 좌표
    diff = x[:, None, :] - x[None, :, :]                 # (N, N, 3) x_i - x_j: 회전에 등변
    d2 = np.sum(diff**2, axis=-1, keepdims=True)         # (N, N, 1) 거리 제곱: 불변
    adj = (d2 < r_cut**2) & ~np.eye(N, dtype=bool)[..., None]          # (N, N, 1) 반경 그래프
    pair = np.concatenate([np.repeat(h[:, None], N, 1), np.repeat(h[None], N, 0), d2], -1)
    m = phi_e(pair) * adj                                # (N, N, D_M) 메시지: 불변량으로만 만든다
    x_new = x + np.sum(diff * phi_x(m) * adj, axis=1) / (N - 1)        # 상대 벡터 × 불변 스칼라
    h_new = h + phi_h(np.concatenate([h, m.sum(axis=1)], -1))
    return h_new, x_new

def rotation(q):                                         # quaternion (4,) → 회전 행렬 (3, 3)
    w, a, b, c = q / np.linalg.norm(q)
    return np.array([[1 - 2*(b*b + c*c), 2*(a*b - c*w), 2*(a*c + b*w)],
                     [2*(a*b + c*w), 1 - 2*(a*a + c*c), 2*(b*c - a*w)],
                     [2*(a*c - b*w), 2*(b*c + a*w), 1 - 2*(a*a + b*b)]])
h, x = rng.normal(size=(N, D_H)), rng.normal(size=(N, 3))
Q = rotation(rng.normal(size=4)) @ np.diag([-1.0, 1, 1])  # 회전 + 반사 (3, 3)
t, P = rng.normal(size=3), np.eye(N)[rng.permutation(N)]  # 평행이동 (3,), 치환 행렬 (N, N)
h1, x1 = egnn_layer(h, x)
h2, x2 = egnn_layer(P @ h, P @ x @ Q.T + t)               # 순서를 바꾸고 회전·반사·이동한 입력
print("h' 불변 (순서만 함께 바뀜) :", np.allclose(h2, P @ h1))
print("x' 등변 (x' -> Q x' + t)  :", np.allclose(x2, P @ x1 @ Q.T + t))

# %% cell 2 — 데이터 효율과 회전된 시험셋
iu = np.triu_indices(4, 1)                               # 입자 4개의 쌍 6개
def energy(X):                                           # X: (B, 4, 3) → (B,) Morse 쌍 퍼텐셜의 합
    d = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)[:, iu[0], iu[1]]   # (B, 6)
    return np.sum((1 - np.exp(-(d - 1.0)))**2 - 1, axis=1)
def rotate_all(X, r):                                    # 표본마다 무작위 회전
    return np.stack([xb @ rotation(r.normal(size=4)).T for xb in X])
raw = lambda X: X.reshape(len(X), -1)                    # (B, 12) 좌표를 펼친 표현
inv = lambda X: np.sort(np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)[:, iu[0], iu[1]], axis=1)  # (B, 6)

def krr(F_tr, y_tr, F_te, lam=1e-3):                     # RBF 커널 ridge, 길이척도는 중앙값 휴리스틱
    sq = lambda a, b: np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T
    K = lambda a, b: np.exp(-0.5 * sq(a, b) / np.median(sq(F_tr, F_tr)))
    alpha = np.linalg.solve(K(F_tr, F_tr) + lam * np.eye(len(F_tr)), y_tr - y_tr.mean())
    return K(F_te, F_tr) @ alpha + y_tr.mean()

r = np.random.default_rng(1)
scale = np.array([1.5, 0.8, 0.4])                        # 학습 구조는 x축으로 길게 정렬되어 있다
X_pool, X_te = r.normal(size=(300, 4, 3)) * scale, r.normal(size=(300, 4, 3)) * scale
X_te_rot = rotate_all(X_te, r)
print("RMSE (같은 방향 시험셋 / 회전된 시험셋)")
for n_tr in [25, 100, 300]:
    X_tr = X_pool[:n_tr]
    X_aug = np.concatenate([X_tr, rotate_all(X_tr, r), rotate_all(X_tr, r)])   # 회전 증강 3배
    row = []
    for name, f, Xa in [("좌표", raw, X_tr), ("좌표+증강", raw, X_aug), ("불변 특징", inv, X_tr)]:
        rmse = [np.sqrt(np.mean((krr(f(Xa), energy(Xa), f(Xt)) - energy(Xt))**2)) for Xt in (X_te, X_te_rot)]
        row.append(f"{name} {rmse[0]:.2f}/{rmse[1]:.2f}")
    print(f"N = {n_tr:3d}  " + "   ".join(row))

# %% cell 3 — 외부장이 대칭을 깰 때
def energy_field(X, kappa=1.0):                          # z축 방향의 장: 결합이 z축과 평행할수록 에너지가 높다
    d = (X[:, :, None] - X[:, None])[:, iu[0], iu[1]]    # (B, 6, 3) 쌍별 상대 벡터
    cos2 = d[..., 2]**2 / np.sum(d**2, axis=-1)          # (B, 6) z축과 이루는 각의 cos^2
    return energy(X) + kappa * np.sum(cos2, axis=1)

def inv_z(X):                                            # z축 회전·평행이동·순열에만 불변인 특징 (B, 12)
    d = (X[:, :, None] - X[:, None])[:, iu[0], iu[1]]
    dist, dz = np.linalg.norm(d, axis=-1), np.abs(d[..., 2])
    order = np.argsort(dist, axis=1)                     # 쌍을 거리 순으로 정렬하되 (거리, |dz|) 짝은 유지
    return np.concatenate([np.take_along_axis(dist, order, 1), np.take_along_axis(dz, order, 1)], axis=1)

X_tr, X_te = rotate_all(r.normal(size=(300, 4, 3)), r), rotate_all(r.normal(size=(300, 4, 3)), r)
for name, f in [("E(3) 불변 특징      ", inv), ("z축 회전 불변 특징  ", inv_z)]:
    pred = krr(f(X_tr), energy_field(X_tr), f(X_te))
    print(f"{name} RMSE {np.sqrt(np.mean((pred - energy_field(X_te))**2)):.3f}")
pred = krr(inv(X_tr), energy(X_tr), inv(X_te))
print(f"(참고) 장이 없을 때 E(3) 불변 특징 RMSE {np.sqrt(np.mean((pred - energy(X_te))**2)):.3f}")
