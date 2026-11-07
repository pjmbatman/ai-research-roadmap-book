"""23.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/analyzing-results/cells.py
"""
# %% cell 1 — 평균 개선을 조건별로 나누기
import numpy as np

rng = np.random.default_rng(8)
n_by_cond = {"0 °C": 60, "25 °C": 300, "45 °C": 80}       # 조건별 시험 셀 수
true_gap = {"0 °C": +0.30, "25 °C": -0.25, "45 °C": -0.12}  # 참 차이 B − A (%p)
cond, err_a, err_b = [], [], []
for c, n in n_by_cond.items():
    hard = rng.gamma(2.0, 0.6, size=n)                       # 셀마다 공통 난이도, (n,)
    err_a.append(np.abs(hard + rng.normal(0, 0.4, n)))      # A의 절대오차 (%p)
    err_b.append(np.abs(hard + true_gap[c] + rng.normal(0, 0.4, n)))
    cond += [c] * n
cond, err_a, err_b = np.array(cond), np.concatenate(err_a), np.concatenate(err_b)   # (440,)
d = err_b - err_a                                            # 짝지은 차이, 음수면 B가 낫다

print(f"전체   n={len(d):3d}  A {err_a.mean():.3f}  B {err_b.mean():.3f}  B−A {d.mean():+.3f} %p")
for c in n_by_cond:
    m = cond == c
    print(f"{c:6s} n={m.sum():3d}  A {err_a[m].mean():.3f}  B {err_b[m].mean():.3f}"
          f"  B−A {d[m].mean():+.3f} %p  B가 나은 셀 {np.mean(d[m] < 0):.0%}")

# %% cell 2 — 조건별 짝지은 bootstrap 구간과 판정
def paired_boot(d, B=4000, seed=1):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(B, len(d)))            # 셀을 복원추출, (B, n)
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])

def verdict(lo, hi):
    return "B 우세" if hi < 0 else "A 우세" if lo > 0 else "판정 보류"

for name, m in [("전체", np.ones_like(d, bool))] + [(c, cond == c) for c in n_by_cond]:
    lo, hi = paired_boot(d[m])
    print(f"{name:6s} B−A {d[m].mean():+.3f}  95% 구간 [{lo:+.3f}, {hi:+.3f}]  → {verdict(lo, hi)}")

# %% cell 3 — 조건 세 개를 동시에 주장할 때: 보정 없음·Bonferroni·Holm
from scipy import stats

names = list(n_by_cond)
p = np.array([stats.ttest_rel(err_b[cond == c], err_a[cond == c]).pvalue for c in names])
alpha, k = 0.05, len(p)
order = np.argsort(p)                                        # 작은 p부터
holm = np.zeros(k, bool)
for rank, i in enumerate(order):                             # rank = 0, 1, 2
    if p[i] > alpha / (k - rank):
        break                                                # 처음 실패하면 멈춘다
    holm[i] = True
for i, c in enumerate(names):
    print(f"{c:6s} p = {p[i]:.1e}  보정 없음 {p[i] < alpha!s:5s}  "
          f"Bonferroni {p[i] < alpha / k!s:5s}  Holm {holm[i]!s:5s}")

# %% cell 4 — ablation 표 읽기: seed 5개, 짝지은 차이
t4 = stats.t.ppf(0.975, df=4)                                # 자유도 4의 97.5% 분위수
seed_fx = rng.normal(0, 0.04, size=5)                        # seed마다 공통 효과 (분할·초기화), (5,)
true_rmse = {"B 전체": 1.00, "− 물리 손실": 1.12, "− attention": 1.01,
             "A (baseline)": 1.20, "A 확대": 1.09}       # A 확대: B와 같은 파라미터 수
runs = {m: v + seed_fx + rng.normal(0, 0.02, 5) for m, v in true_rmse.items()}   # 각 (5,)
full = runs["B 전체"]
for name, r in runs.items():
    diff = r - full                                          # 같은 seed끼리 짝지은 차이, (5,)
    half = t4 * diff.std(ddof=1) / np.sqrt(5)                # 95% t 구간의 반폭
    gap = "" if name == "B 전체" else f"B 전체 대비 {diff.mean():+.3f} ± {half:.3f}"
    print(f"{name:13s} RMSE {r.mean():.3f} ± {r.std(ddof=1):.3f}   {gap}".rstrip())
