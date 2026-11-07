"""15.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/synthetic-data/cells.py
"""
# %% cell 1 — 설정 하나가 데이터셋 하나를 정한다
import hashlib
import json
import numpy as np
from scipy.stats import qmc

def simulate(zeta, omega_rad_s, alpha=0.0, t_end_s=10.0, dt_s=0.01, every=50):
    """x'' + 2ζω x' + ω² x + α x³ = 0, x(0) = 1, v(0) = 0을 RK4로 푼다. 반환 (N, T)."""
    z, w = np.asarray(zeta, float), np.asarray(omega_rad_s, float)       # (N,), (N,)
    x, v, out = np.ones_like(z), np.zeros_like(z), []
    f = lambda x, v: (v, -2 * z * w * v - w**2 * x - alpha * x**3)
    for k in range(int(round(t_end_s / dt_s))):
        if k % every == 0:
            out.append(x)                                                # 0.5 s마다 기록
        a1, b1 = f(x, v)
        a2, b2 = f(x + 0.5 * dt_s * a1, v + 0.5 * dt_s * b1)
        a3, b3 = f(x + 0.5 * dt_s * a2, v + 0.5 * dt_s * b2)
        a4, b4 = f(x + dt_s * a3, v + dt_s * b3)
        x, v = x + dt_s / 6 * (a1 + 2 * a2 + 2 * a3 + a4), v + dt_s / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return np.stack(out, axis=1)

config = {"simulator": "duffing-rk4", "version": 1, "design": "lhs", "n": 400, "seed": 0,
          "bounds": {"zeta": [0.05, 0.5], "omega_rad_s": [1.0, 4.0]}, "alpha": 0.0, "dt_s": 0.01}

def make_dataset(cfg):
    unit = qmc.LatinHypercube(d=2, seed=cfg["seed"]).random(cfg["n"])     # (n, 2) in [0, 1)
    lo, hi = zip(*cfg["bounds"].values())
    P = qmc.scale(unit, lo, hi)                                           # (n, 2): ζ, ω
    return P, simulate(P[:, 0], P[:, 1], cfg["alpha"], dt_s=cfg["dt_s"])

P, X_sim = make_dataset(config)                                           # (400, 2), (400, 20)
cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
print("설정 해시:", cfg_hash[:16], " 데이터 shape:", X_sim.shape)
print("같은 설정으로 다시 만든 데이터가 같은가:", np.array_equal(make_dataset(config)[1], X_sim))

# %% cell 2 — 같은 16점, 다른 커버리지
n = 16
g = (np.arange(4) + 0.5) / 4                                      # (4,) 축마다 4단계
designs = {
    "grid":   np.array([(a, b) for a in g for b in g]),           # (16, 2)
    "random": np.random.default_rng(0).random((n, 2)),
    "LHS":    qmc.LatinHypercube(d=2, seed=0).random(n),
    "Sobol":  qmc.Sobol(d=2, seed=0).random(n),
}
for name, S in designs.items():
    cover = len(np.unique(np.floor(S[:, 0] * n)))                 # ζ축 투영이 덮는 칸 수(16칸 중)
    print(f"{name:6s}  ζ축 16칸 중 {cover:2d}칸  discrepancy {qmc.discrepancy(S):.4f}")

# %% cell 3 — sim-to-real 간극: 시뮬레이터가 모르는 것
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(1)
P_real = np.column_stack([rng.uniform(0.05, 0.5, 200), rng.uniform(1.0, 4.0, 200)])  # (200, 2)
X_real = (simulate(P_real[:, 0], P_real[:, 1], alpha=1.0)        # 실제 계: 비선형 강성 α = 1
          + 0.02 + rng.normal(0, 0.02, (200, 20)))               # 센서 offset과 잡음
y_sim, y_real = P[:, 0], P_real[:, 0]                             # 목표: 감쇠비 ζ

def rmse(model, X, y):
    return np.sqrt(np.mean((model.predict(X) - y) ** 2))

model = Ridge(alpha=1e-3).fit(X_sim[:300], y_sim[:300])          # 시뮬레이션만으로 학습
print(f"ζ RMSE: sim test {rmse(model, X_sim[300:], y_sim[300:]):.3f}, real {rmse(model, X_real[50:], y_real[50:]):.3f}")

X_dom = np.vstack([X_sim[:200], X_real])                          # (400, 20) sim과 real을 섞는다
d_dom = np.r_[np.zeros(200), np.ones(200)]                        # 0 = sim, 1 = real
auc = cross_val_score(LogisticRegression(max_iter=1000), X_dom, d_dom, cv=5, scoring="roc_auc").mean()
print(f"sim과 real을 구별하는 분류기의 AUC: {auc:.2f}")

# %% cell 4 — 간극을 줄이는 세 가지 방법
X_r20, y_r20 = X_real[:20], y_real[:20]                           # 라벨이 있는 실측 20개
mixed = Ridge(alpha=1e-3).fit(np.vstack([X_sim[:300], X_r20]), np.r_[y_sim[:300], y_r20])
real_only = Ridge(alpha=1e-3).fit(X_r20, y_r20)
alpha_rand = rng.uniform(0.0, 4.0, 300)                           # (300,) 모르는 물리를 무작위로
X_dr = (simulate(P[:300, 0], P[:300, 1], alpha=alpha_rand)
        + rng.uniform(-0.1, 0.1, (300, 1)) + rng.normal(0, 0.02, (300, 20)))
randomized = Ridge(alpha=1e-3).fit(X_dr, y_sim[:300])
for name, m in [("sim + 실측 20", mixed), ("실측 20만", real_only), ("domain randomization", randomized)]:
    print(f"{name:20s} real ζ RMSE {rmse(m, X_real[50:], y_real[50:]):.3f}")
