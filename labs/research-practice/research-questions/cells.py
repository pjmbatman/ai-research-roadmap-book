"""23.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/research-questions/cells.py
"""
# %% cell 1 — 결과를 본 뒤 고른 가설
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
S, K, n = 4000, 10, 20                  # 가상 연구 4000번, 하위 그룹 10개, 그룹당 실행 20개
a = rng.normal(size=(S, K, n))          # baseline 점수 (효과 없음), (S, K, n)
b = rng.normal(size=(S, K, n))          # 제안 방법 점수 (효과 없음), (S, K, n)
diff = b.mean(-1) - a.mean(-1)          # (S, K)
se = np.sqrt(a.var(-1, ddof=1) / n + b.var(-1, ddof=1) / n)
p = stats.t.sf(diff / se, df=2 * n - 2)    # 단측 p-value: "제안 방법이 낫다", (S, K)

pre = p[:, 0] < 0.05                    # 사전에 정한 하위 그룹 하나만 검정
post = p.min(axis=1) < 0.05             # 결과를 보고 가장 좋은 하위 그룹을 가설로
best = diff[np.arange(S), p.argmin(axis=1)]
print(f"참 효과 0에서 '유의한 개선'을 보고할 확률: 사전 가설 {pre.mean():.3f}, 사후 가설 {post.mean():.3f}")
print(f"사후 가설로 보고된 효과의 평균 크기 |차이| = {np.abs(best[post]).mean():.2f} (참값 0, 단위: 실행 간 표준편차)")

# %% cell 2 — 사전에 정한 판정 규칙
delta, sd = 0.02, 0.015                 # 의미 있는 최소 개선, seed 간 짝지은 차이의 표준편차
def decide(true_effect, n_seed, trials=4000, seed=1):
    r = np.random.default_rng(seed)
    d = r.normal(true_effect, sd, size=(trials, n_seed))    # 짝지은 차이, (trials, n_seed)
    half = stats.t.ppf(0.95, n_seed - 1) * d.std(1, ddof=1) / np.sqrt(n_seed)
    lo, hi = d.mean(1) - half, d.mean(1) + half             # 단측 95% 하한·상한, (trials,)
    return [np.mean(c) for c in ((lo > 0) & (hi >= delta),   # 성공: 개선이 있고 의미 있는 크기일 수 있다
                                 (lo > 0) & (hi < delta),    # 작은 개선: 있지만 delta보다 작다
                                 (lo <= 0) & (hi < delta),   # 중단: 의미 있는 개선을 배제한다
                                 (lo <= 0) & (hi >= delta))] # 보류: 어느 쪽도 말할 수 없다

print("참 효과  seed 수 |  성공  작은개선  중단   보류")
for eff in (0.0, 0.01, 0.03):
    for n_seed in (3, 10, 30):
        print(f"  {eff:.2f}    {n_seed:>3}   | " + "   ".join(f"{v:.2f}" for v in decide(eff, n_seed)))
