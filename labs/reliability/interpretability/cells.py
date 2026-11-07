"""21.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/interpretability/cells.py
"""
# %% cell 1 — 상관된 특징에서의 순열 중요도
import numpy as np

rng = np.random.default_rng(0)
def make(n):
    x1, x2, x4 = rng.normal(size=(3, n))
    x3 = x1 + 0.1 * rng.normal(size=n)                  # x1의 거의 복사본 (상관 0.995)
    y = 2 * x1 + x2 ** 2 + 0.3 * rng.normal(size=n)     # 참 관계에 x3, x4는 없다
    return np.stack([x1, x2, x3, x4], axis=1), y        # (N, 4), (N,)

phi = lambda X: np.column_stack([X, X[:, 1] ** 2])      # 특징 + x2²
X_tr, y_tr = make(2000)
P = phi(X_tr)
coef = np.linalg.solve(P.T @ P + 10 * np.eye(5), P.T @ y_tr)   # ridge (λ = 10)
predict = lambda X: phi(X) @ coef
print("ridge 가중치 (x1, x2, x3, x4, x2²):", np.round(coef, 2) + 0.0)

X_te, y_te = make(5000)
base = np.mean((predict(X_te) - y_te) ** 2)
def perm_importance(cols, reps=10):
    out = []
    for _ in range(reps):
        Xp = X_te.copy()
        Xp[:, cols] = X_te[rng.permutation(len(X_te))][:, cols]   # 열(들)을 함께 섞는다
        out.append(np.mean((predict(Xp) - y_te) ** 2) - base)
    return np.mean(out)

for name, cols in (("x1", [0]), ("x2", [1]), ("x3", [2]), ("x4", [3]), ("x1+x3 함께", [0, 2])):
    print(f"{name:10s} MSE 증가 {perm_importance(cols):6.2f}")

keep = [1, 2, 3, 4]                                     # x1을 빼고 다시 학습
c2 = np.linalg.solve(P[:, keep].T @ P[:, keep] + 10 * np.eye(4), P[:, keep].T @ y_tr)
refit = np.mean((phi(X_te)[:, keep] @ c2 - y_te) ** 2) - base
print(f"x1을 빼고 다시 학습했을 때 MSE 증가 {refit:.2f}")

# %% cell 2 — 부분 의존(PDP)과 개별 조건부 기대(ICE)
Z = rng.normal(size=(2000, 2))                          # (N, 2) 두 입력 z1, z2 (독립)
t = Z[:, 0] * Z[:, 1] + 0.1 * rng.normal(size=2000)     # z1의 효과 부호가 z2에 따라 바뀐다
Q = np.column_stack([Z, Z[:, 0] * Z[:, 1]])             # (N, 3) 특징
b = np.linalg.lstsq(Q, t, rcond=None)[0]
g = lambda Z: np.column_stack([Z, Z[:, 0] * Z[:, 1]]) @ b

grid = np.linspace(-2, 2, 5)
ice = np.array([g(np.column_stack([np.full(len(Z), v), Z[:, 1]])) for v in grid])   # (5, N)
print("z1 격자       ", grid)
print("PDP (ICE 평균)", np.round(ice.mean(axis=1), 2) + 0.0)
slopes = (ice[-1] - ice[0]) / 4                          # 사례별 ICE 기울기
print(f"ICE 기울기: 10% 분위수 {np.quantile(slopes, 0.1):.2f}, 90% 분위수 {np.quantile(slopes, 0.9):.2f}")

# %% cell 3 — gradient×input과 integrated gradients, 기준값의 영향
def f(x):                                               # x: (D,) 입력 3개
    return x[0] ** 2 + 3 * x[1] + 2 * np.minimum(x[2], 1.0)   # 셋째 항은 x3 > 1에서 포화

def grad(x, h=1e-5):                                    # 중심 차분 gradient, (D,)
    E = np.eye(len(x)) * h
    return np.array([(f(x + e) - f(x - e)) / (2 * h) for e in E])

def integrated_gradients(x, base, steps=50):
    alphas = (np.arange(steps) + 0.5) / steps           # 경로 위 중점들
    g_mean = np.mean([grad(base + a * (x - base)) for a in alphas], axis=0)
    return (x - base) * g_mean

x = np.array([2.0, 1.0, 2.0])
for name, attr, ref in (("gradient×input", x * grad(x), np.zeros(3)),
                        ("IG, 기준 0", integrated_gradients(x, np.zeros(3)), np.zeros(3)),
                        ("IG, 기준 (1,0,0)", integrated_gradients(x, np.array([1.0, 0, 0])), np.array([1.0, 0, 0]))):
    print(f"{name:16s} 기여 {np.round(attr, 2) + 0.0}  합 {attr.sum():.2f}  f(x) - f(기준) {f(x) - f(ref):.2f}")

# %% cell 4 — 기여도는 인과가 아니다: 개입 실험으로 확인
def simulate(n, do_x1=None, do_x3_shift=0.0):
    x1 = rng.normal(size=n) if do_x1 is None else np.full(n, do_x1)   # 원인 (예: 반응 온도)
    x2 = rng.normal(size=n)                                             # 무관한 변수
    y = 2 * x1 + 0.5 * rng.normal(size=n)                               # 결과 (예: 수율)
    x3 = y + 0.1 * rng.normal(size=n) + do_x3_shift                     # y의 하류 측정 (예: 생성물 센서)
    return np.column_stack([x1, x2, x3, np.ones(n)]), y

Xs, ys = simulate(5000)
w = np.linalg.lstsq(Xs, ys, rcond=None)[0]
print("회귀 계수 (x1, x2, x3):", np.round(w[:3], 2) + 0.0)

for name, kw in (("do(x1 = 1)", dict(do_x1=1.0)), ("do(x1 = 0)", dict(do_x1=0.0)),
                 ("x3에 +1 개입", dict(do_x3_shift=1.0)), ("개입 없음", dict())):
    Xi, yi = simulate(20000, **kw)
    print(f"{name:10s}  실제 y 평균 {yi.mean():6.2f}   모델 예측 평균 {(Xi @ w).mean():6.2f}")
