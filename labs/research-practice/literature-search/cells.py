"""23.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/literature-search/cells.py
"""
# %% cell 1 — 인용 그래프에서 핵심 논문 찾기
import numpy as np

papers = ["원전15", "후속17", "후속18", "반박19", "벤치20", "후속21", "서베이23", "최신24"]
cites = {  # 인용하는 논문 → 인용된 논문들
    "후속17": ["원전15"], "후속18": ["원전15", "후속17"], "반박19": ["원전15", "후속18"],
    "벤치20": ["후속17", "후속18"], "후속21": ["후속18", "벤치20", "반박19"],
    "서베이23": ["후속17", "후속18", "반박19", "벤치20", "후속21"],
    "최신24": ["후속21", "벤치20", "원전15"],
}
n, idx = len(papers), {p: i for i, p in enumerate(papers)}
A = np.zeros((n, n))                         # A[j, i] = 1: i가 j를 인용, (n, n)
for src, dsts in cites.items():
    for dst in dsts:
        A[idx[dst], idx[src]] = 1.0
out = A.sum(axis=0)                          # 각 논문의 참고문헌 수, (n,)
P = np.where(out > 0, A / np.maximum(out, 1), 1.0 / n)   # 열 확률행렬, 참고문헌 없는 논문은 균등
d, r = 0.85, np.full(n, 1.0 / n)
for _ in range(100):
    r = (1 - d) / n + d * P @ r              # power iteration, (n,)

for i in np.argsort(-r, kind="stable"):
    print(f"{papers[i]:<6} 피인용 {int(A[i].sum())}  PageRank {r[i]:.3f}")
seed = "후속18"
print("backward(참고문헌):", cites[seed])
print("forward(피인용):", [p for p, ds in cites.items() if seed in ds])

# %% cell 2 — 검색 결과 중복 제거
import re

records = [  # 세 도구에서 모은 가상의 검색 결과: (출처, 제목, 연도, DOI, arXiv)
    ("arxiv",    "Neural Solvers Learn Operators of PDEs", 2020, None,          "2010.00001"),
    ("semantic", "Learning solution operators of PDEs.",   2021, "10.0000/j.1", "2010.00001"),
    ("journal",  "Learning Solution Operators of PDEs",    2021, "10.0000/j.1", None),
    ("scholar",  "Physics-Informed Surrogates: A Review",  2022, "10.0000/r.7", None),
    ("semantic", "Physics-informed surrogates - a review", 2022, None,          None),
    ("arxiv",    "Graph Networks for Battery Aging",       2023, None,          "2301.00042"),
    ("scholar",  "Graph networks for battery ageing",      2023, None,          None),
]
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

parent, seen = list(range(len(records))), {}  # union-find: 같은 논문이면 한 묶음
def find(i):
    while parent[i] != i:
        i = parent[i]
    return i
for i, (_, title, _, doi, arx) in enumerate(records):
    for key in (("doi", doi), ("arxiv", arx), ("title", norm(title))):
        if key[1] is not None:                 # 처음 본 키면 자기 자신과 묶는다(변화 없음)
            parent[find(i)] = find(seen.setdefault(key, i))
groups = {}
for i in range(len(records)):
    groups.setdefault(find(i), []).append(records[i][0])
print("제목 문자열 그대로:", len({r[1] for r in records}), "편")
print("정규화 제목:", len({norm(r[1]) for r in records}), "편")
print("식별자 + 정규화 제목:", len(groups), "편")
for g in groups.values():
    print("  ", g)

# %% cell 3 — 철자 변이는 확인 후보로
from difflib import SequenceMatcher

reps = [norm(records[k][1]) for k in groups]
for a in range(len(reps)):
    for b in range(a + 1, len(reps)):
        s = SequenceMatcher(None, reps[a], reps[b]).ratio()
        if s > 0.9:
            print(f"확인 후보: '{reps[a]}' ~ '{reps[b]}' (유사도 {s:.2f})")
