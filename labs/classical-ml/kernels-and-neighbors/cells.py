"""16.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/kernels-and-neighbors/cells.py
"""
# %% cell 1 — kNN: 거리는 단위를 탄다
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n = 600
conc_mol_l = rng.uniform(0.01, 0.2, n)                     # 농도 [mol/L]
temp_k = rng.uniform(280, 360, n)                          # 온도 [K]
rate = 50 * conc_mol_l**2 * np.exp(-2000 * (1 / temp_k - 1 / 320))   # 반응 속도 (2차 반응 + Arrhenius)
rate += rng.normal(0, 0.02, n)                             # 측정 잡음
X = np.column_stack([conc_mol_l, temp_k])                  # (600, 2)
tr, te = slice(0, 400), slice(400, None)
for label, model in [("표준화 없이", KNeighborsRegressor(10)),
                     ("표준화 후", make_pipeline(StandardScaler(), KNeighborsRegressor(10)))]:
    model.fit(X[tr], rate[tr])
    print(f"{label}: 시험 R² {model.score(X[te], rate[te]):.3f}")

# %% cell 2 — 차원의 저주
from sklearn.linear_model import RidgeCV

def task(n, d_noise, rng):                                 # 출력은 앞의 두 특징에만 의존
    X = rng.uniform(-1, 1, (n, 2 + d_noise))               # (n, 2 + d_noise)
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, n)
    return X, y

for d_noise in (0, 4, 16, 64):
    r = np.random.default_rng(1)
    Xa, ya = task(500, d_noise, r)
    Xb, yb = task(500, d_noise, r)
    knn = KNeighborsRegressor(10, algorithm="kd_tree").fit(Xa, ya).score(Xb, yb)
    lin = RidgeCV().fit(Xa, ya).score(Xb, yb)
    d = np.linalg.norm(Xb[:1] - Xa, axis=1)                # 시험점 하나에서 학습점까지 거리, (500,)
    print(f"차원 {2 + d_noise:2d}: 최근접/최원 거리 비 {d.min() / d.max():.2f}  kNN R² {knn:.2f}  선형 R² {lin:.2f}")

# %% cell 3 — SVM: 마진과 서포트 벡터
from sklearn.svm import SVC

rng = np.random.default_rng(2)
Xs = np.vstack([rng.normal([-1.5, 0], 0.7, (100, 2)), rng.normal([1.5, 0], 0.7, (100, 2))])   # (200, 2)
ys = np.repeat([0, 1], 100)
for C in (0.01, 1.0, 100.0):
    svm = SVC(kernel="linear", C=C).fit(Xs, ys)
    w = svm.coef_[0]
    print(f"C = {C:6.2f}: 서포트 벡터 {svm.n_support_.sum():3d}개, 마진 폭 2/||w|| = {2 / np.linalg.norm(w):.2f}, 학습 정확도 {svm.score(Xs, ys):.3f}")
svm = SVC(kernel="linear", C=1.0, tol=1e-6).fit(Xs, ys)    # 해를 정밀하게 구한다
keep = np.ones(200, bool)
keep[np.setdiff1d(np.arange(200), svm.support_)[:50]] = False   # 서포트 벡터가 아닌 점 50개 제거
svm2 = SVC(kernel="linear", C=1.0, tol=1e-6).fit(Xs[keep], ys[keep])
same = np.allclose(svm.coef_, svm2.coef_, atol=1e-4) and np.allclose(svm.intercept_, svm2.intercept_, atol=1e-4)
print("서포트 벡터가 아닌 점 50개를 지워도 경계가 같은가:", same)

# %% cell 4 — 커널 릿지 회귀와 GP 사후평균
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

rng = np.random.default_rng(3)
x_tr = np.sort(rng.uniform(0, 10, 30))[:, None]            # 측정 위치, (30, 1)
y_tr = np.sin(x_tr[:, 0]) + rng.normal(0, 0.1, 30)
x_te = np.linspace(-2, 12, 141)[:, None]                   # 학습 범위 밖 포함, (141, 1)
ell, noise_var = 1.0, 0.1**2
krr = KernelRidge(kernel="rbf", gamma=1 / (2 * ell**2), alpha=noise_var).fit(x_tr, y_tr)
gp = GaussianProcessRegressor(RBF(ell), alpha=noise_var, optimizer=None).fit(x_tr, y_tr)
mean, std = gp.predict(x_te, return_std=True)              # (141,), (141,)
print("KRR 예측 = GP 사후평균:", np.allclose(krr.predict(x_te), mean, atol=1e-8))
for x0 in (5.0, 11.0, 12.0):
    i = np.argmin(np.abs(x_te[:, 0] - x0))
    print(f"x = {x0:4.1f}: 사후평균 {mean[i]:+.2f}, 사후표준편차 {std[i]:.2f}, 참값 {np.sin(x0):+.2f}")
for ell_try in (0.1, 1.0, 10.0):
    m = KernelRidge(kernel="rbf", gamma=1 / (2 * ell_try**2), alpha=noise_var).fit(x_tr, y_tr)
    rmse = np.sqrt(np.mean((m.predict(x_te[20:121]) - np.sin(x_te[20:121, 0])) ** 2))
    print(f"길이 척도 {ell_try:4.1f}: 학습 범위 안 RMSE {rmse:.3f}")

# %% cell 5 — 확장성: Nyström과 random features
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(4)
Xl = rng.uniform(-3, 3, (2500, 2))                         # (2500, 2)
yl = np.sin(Xl[:, 0]) * np.cos(Xl[:, 1]) + rng.normal(0, 0.1, 2500)
Xa, ya, Xb, yb = Xl[:1500], yl[:1500], Xl[1500:], yl[1500:]
exact = KernelRidge(kernel="rbf", gamma=0.5, alpha=0.01).fit(Xa, ya)
print(f"정확한 KRR (1500×1500 Gram 행렬): 시험 R² {exact.score(Xb, yb):.3f}")
for m in (10, 50, 200):
    for name, feat in [("Nyström", Nystroem(gamma=0.5, n_components=m, random_state=0)),
                       ("random features", RBFSampler(gamma=0.5, n_components=m, random_state=0))]:
        model = make_pipeline(feat, Ridge(alpha=0.01)).fit(Xa, ya)
        print(f"m = {m:3d} {name:15s}: 시험 R² {model.score(Xb, yb):.3f}")
