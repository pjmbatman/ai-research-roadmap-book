"""14.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/information-and-learning-theory/generalization-basics/cells.py
"""
# %% cell 1 — 경험 위험은 고른 모델에 대해 낙관적이다
import numpy as np

def f_true(x):
    return np.sin(2 * np.pi * x)

def sample(rng, n, noise=0.3):
    x = rng.uniform(0, 1, n)
    return x, f_true(x) + noise * rng.normal(size=n)

rng = np.random.default_rng(0)
x_te, y_te = sample(rng, 100_000)                  # 모집단 위험을 근사할 큰 시험 집합
fixed = np.array([0.0, 6.0, -18.0, 12.0])          # 데이터를 보기 전에 정한 3차식 계수
emp_fixed, emp_erm, pop_erm = [], [], []
for _ in range(500):
    x, y = sample(rng, 15)
    emp_fixed.append(np.mean((np.polyval(fixed[::-1], x) - y) ** 2))
    c = np.polyfit(x, y, 7)                        # ERM: 7차 다항식 최소제곱
    emp_erm.append(np.mean((np.polyval(c, x) - y) ** 2))
    pop_erm.append(np.mean((np.polyval(c, x_te) - y_te) ** 2))
print(f"고정한 모델  경험 위험 평균 {np.mean(emp_fixed):.3f}   모집단 위험 {np.mean((np.polyval(fixed[::-1], x_te) - y_te) ** 2):.3f}")
print(f"ERM 모델     경험 위험 평균 {np.mean(emp_erm):.3f}   모집단 위험 중앙값 {np.median(pop_erm):.3f}")

# %% cell 2 — bias–variance 분해
x_grid = np.linspace(0.05, 0.95, 50)               # 평가 지점, (M,)
noise = 0.3
print("차수   bias²    variance  noise    합      시험 MSE")
for deg in (1, 3, 5, 6):
    preds, test = [], []
    for seed in range(400):
        r = np.random.default_rng(seed)
        x, y = sample(r, 20, noise)
        c = np.polyfit(x, y, deg)
        preds.append(np.polyval(c, x_grid))        # (M,)
        y_new = f_true(x_grid) + noise * r.normal(size=50)
        test.append(np.mean((preds[-1] - y_new) ** 2))
    P = np.array(preds)                            # (R, M)
    bias2 = np.mean((P.mean(0) - f_true(x_grid)) ** 2)
    var = np.mean(P.var(0))
    print(f"{deg:3d}   {bias2:.4f}   {var:.4f}    {noise**2:.4f}   {bias2 + var + noise**2:.4f}   {np.mean(test):.4f}")

# %% cell 3 — 학습 곡선
print("   N    3차: 학습   검증     7차: 학습   검증")
for n in (15, 30, 60, 120, 480):
    row = []
    for deg in (3, 7):
        tr, va = [], []
        for seed in range(200):
            r = np.random.default_rng(seed)
            x, y = sample(r, n)
            c = np.polyfit(x, y, deg)
            tr.append(np.mean((np.polyval(c, x) - y) ** 2))
            va.append(np.mean((np.polyval(c, x_te[:2000]) - y_te[:2000]) ** 2))
        row += [np.mean(tr), np.median(va)]
    print(f"{n:4d}      {row[0]:.3f}   {row[1]:.3f}        {row[2]:.3f}   {row[3]:.3f}")

# %% cell 4 — 무작위 레이블 맞추기: 용량의 직관
rng = np.random.default_rng(1)
n = 100
labels = rng.choice([-1.0, 1.0], size=n)           # 입력과 무관한 레이블, (N,)
X_all = rng.normal(size=(n + 1000, 200))           # 특징 200개까지, (N + 1000, 200)
print("특징 수 p   학습 정확도   시험 정확도")
for p in (5, 20, 50, 90, 100, 150):
    X, X_te = X_all[:n, :p], X_all[n:, :p]
    w = np.linalg.pinv(X) @ labels                 # 최소 노름 최소제곱, (p,)
    acc_tr = np.mean(np.sign(X @ w) == labels)
    y_te_rand = rng.choice([-1.0, 1.0], size=1000)
    acc_te = np.mean(np.sign(X_te @ w) == y_te_rand)
    print(f"{p:6d}        {acc_tr:.2f}          {acc_te:.2f}")

# %% cell 5 — 시험 집합에서 모델 100개 중 최고를 고르면
rng = np.random.default_rng(2)
true_err, n_test, k = 0.30, 500, 100               # 모든 모델의 참 오류율이 0.30
best = []
for _ in range(2000):
    errs = rng.binomial(n_test, true_err, size=k) / n_test   # 모델 k개의 시험 오류율, (k,)
    best.append(errs.min())
delta = 0.05
eps_one = np.sqrt(np.log(2 / delta) / (2 * n_test))          # Hoeffding, 모델 1개
eps_k = np.sqrt(np.log(2 * k / delta) / (2 * n_test))        # union bound, 모델 k개
print(f"고른 모델의 시험 오류율 평균 {np.mean(best):.3f}  (참값 {true_err})")
print(f"Hoeffding 폭: 모델 1개 ±{eps_one:.3f},  모델 {k}개 ±{eps_k:.3f}")
print(f"참값 - 고른 값 > 모델 1개 폭인 비율 {np.mean(true_err - np.array(best) > eps_one):.2f}")
