"""23.11 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/peer-review/cells.py
"""
# %% cell 1 — 관측 일치율과 Cohen kappa
import numpy as np

def cohen_kappa(a, b):
    """두 평가자의 범주 판정 a, b (N,)의 Cohen kappa."""
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)                                            # 관측 일치율
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))  # 우연 일치율
    return (p_o - p_e) / (1 - p_e)

# 논문 20편에 대한 권고 (1 = 채택, 0 = 거절)
r1 = np.array([1, 1, 1, 1] + [0] * 16)             # (20,)
r2 = np.array([1, 1, 0, 0, 1, 1] + [0] * 14)       # (20,)
r3 = np.zeros(20, dtype=int)                       # (20,) 무조건 거절하는 심사자
for name, other in [("r2", r2), ("r3", r3)]:
    print(f"r1 vs {name}: 일치율 {np.mean(r1 == other):.2f}, kappa {cohen_kappa(r1, other):.3f}")

# %% cell 2 — 심사자별 점수 보정
reviews = [("P1", "A", 4), ("P1", "C", 6), ("P2", "A", 3), ("P2", "B", 9),
           ("P3", "B", 8), ("P3", "C", 7), ("P4", "A", 5), ("P4", "C", 9),
           ("P5", "B", 7), ("P5", "C", 4), ("P6", "A", 2), ("P6", "B", 6)]
papers = sorted({p for p, _, _ in reviews})
by_rev = {r: np.array([s for _, rr, s in reviews if rr == r], float) for r in "ABC"}
for r, s in by_rev.items():
    print(f"심사자 {r}: 평균 {s.mean():.2f}, 표준편차 {s.std():.2f}")

def z(r, s):                                        # 심사자 r 안에서 표준화한 점수
    return (s - by_rev[r].mean()) / by_rev[r].std()

raw = {p: np.mean([s for pp, _, s in reviews if pp == p]) for p in papers}
cal = {p: np.mean([z(r, s) for pp, r, s in reviews if pp == p]) for p in papers}
print("원점수 순위:", sorted(papers, key=lambda p: -raw[p]))
print("보정 순위:  ", sorted(papers, key=lambda p: -cal[p]))
for p in papers:
    print(f"{p}: 원점수 평균 {raw[p]:.1f}, 보정 점수 {cal[p]:+.2f}")
