"""16.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/baselines/cells.py
"""
# %% cell 1 — 다섯 가지 단순 baseline
import numpy as np

rng = np.random.default_rng(0)
cyc = np.arange(1, 401)                                     # 사이클 1–400, (T,)
def make_cells(n, rng, fade=1.0):                           # 셀 n개의 용량 곡선, (n, T)
    a = fade * rng.uniform(0.003, 0.007, (n, 1))            # √n 항 (SEI 성장)
    b = fade * rng.uniform(0.0, 3e-4, (n, 1))               # 선형 항
    c0 = 2.0 + rng.normal(0, 0.01, (n, 1))                  # 초기 용량 [Ah]
    return c0 - a * np.sqrt(cyc) - b * cyc + rng.normal(0, 0.002, (n, len(cyc)))

cap_tr, cap_te = make_cells(40, rng), make_cells(20, rng)  # 셀 단위 분할: 학습 40, 시험 20
obs, fut = cyc <= 200, cyc > 200                            # 200사이클까지 보고 이후를 예측

def per_cell_fit(cap, basis, rows):                         # 셀마다 최소제곱 → 미래 구간 예측
    B = np.column_stack(basis(cyc))                         # (T, P)
    coef = np.linalg.lstsq(B[rows], cap[:, rows].T, rcond=None)[0]   # (P, n)
    return (B[fut] @ coef).T                                # (n, 200)

pred = {
    "평균":          np.full((20, 200), cap_tr[:, fut].mean()),
    "최근값":        np.repeat(cap_te[:, obs][:, -10:].mean(1, keepdims=True), 200, 1),
    "선형 추세":     per_cell_fit(cap_te, lambda n: (np.ones_like(n), n), (cyc > 100) & obs),
    "물리식 c0-a√n": per_cell_fit(cap_te, lambda n: (np.ones_like(n), np.sqrt(n)), obs),
    "룩업 테이블":   np.tile(cap_tr[:, fut].mean(0), (20, 1)),
}
mae_mah = {k: 1000 * np.abs(p - cap_te[:, fut]).mean(1) for k, p in pred.items()}   # 셀별 MAE, (20,)
for k, v in mae_mah.items():
    print(f"MAE {v.mean():5.1f} mAh  {k}")

# %% cell 2 — 강한 baseline: 같은 예산으로 튜닝
def trend(cap, w):                                          # 마지막 w 사이클로 선형 추세
    return per_cell_fit(cap, lambda n: (np.ones_like(n), n), (cyc > 200 - w) & obs)
def physics2(cap):                                          # c0 - a√n - b n
    return per_cell_fit(cap, lambda n: (np.ones_like(n), np.sqrt(n), n), obs)
def lookup_anchored(cap):                                   # 학습 셀의 평균 열화량 + 현재값
    drop = cap_tr[:, fut].mean(0) - cap_tr[:, obs][:, -10:].mean()
    return cap[:, obs][:, -10:].mean(1, keepdims=True) + drop

mae = lambda p, cap: 1000 * np.abs(p - cap[:, fut]).mean(1)  # 셀별 MAE [mAh]
tr_mae = {w: mae(trend(cap_tr, w), cap_tr).mean() for w in (25, 50, 100, 200)}
print("창 길이별 학습 셀 MAE:", "  ".join(f"w={w} {v:.1f}" for w, v in tr_mae.items()), "mAh")
w_best = min(tr_mae, key=tr_mae.get)                        # 창 길이는 학습 셀로 고른다
strong = {f"선형 추세 w={w_best}": trend(cap_te, w_best), "물리식 + 선형항": physics2(cap_te),
          "룩업 + 현재값": lookup_anchored(cap_te)}
for k, p in strong.items():
    mae_mah[k] = mae(p, cap_te)
    print(f"시험 MAE {mae_mah[k].mean():5.1f} mAh  {k}")

# %% cell 3 — 새 모델은 가장 강한 baseline을 이기는가
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=np.logspace(-6, 0, 13))              # 앞 200사이클 → 뒤 200사이클
model.fit(cap_tr[:, obs][:, ::10], cap_tr[:, fut])          # (40, 20) → (40, 200)
mae_mah["Ridge"] = mae(model.predict(cap_te[:, obs][:, ::10]), cap_te)
best = min((k for k in mae_mah if k != "Ridge"), key=lambda k: mae_mah[k].mean())
d = mae_mah["Ridge"] - mae_mah[best]                        # 셀별 짝지은 차이, (20,)
boot = np.random.default_rng(1).choice(d, (5000, d.size)).mean(1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Ridge MAE {mae_mah['Ridge'].mean():.1f} mAh, 최강 baseline({best}) {mae_mah[best].mean():.1f} mAh")
print(f"차이 {d.mean():+.1f} mAh, 95% CI [{lo:+.1f}, {hi:+.1f}], Ridge가 이긴 셀 {np.sum(d < 0)}/20")
print(f"평균 baseline 대비 skill {1 - mae_mah['Ridge'].mean() / mae_mah['평균'].mean():.2f}")
print(f"최강 baseline 대비 skill {1 - mae_mah['Ridge'].mean() / mae_mah[best].mean():.2f}")
