"""23.1 퀴즈(content/23-research-practice/01-literature-search/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/01-literature-search/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

PAPERS = ["원전15", "후속17", "후속18", "반박19", "벤치20", "후속21", "서베이23", "최신24"]
CITES = {
    "후속17": ["원전15"], "후속18": ["원전15", "후속17"], "반박19": ["원전15", "후속18"],
    "벤치20": ["후속17", "후속18"], "후속21": ["후속18", "벤치20", "반박19"],
    "서베이23": ["후속17", "후속18", "반박19", "벤치20", "후속21"],
    "최신24": ["후속21", "벤치20", "원전15"],
}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def pagerank(A, d=0.85):
    n = len(A)
    out = A.sum(axis=0)
    P = np.where(out > 0, A / np.maximum(out, 1), 1.0 / n)
    r = np.full(n, 1.0 / n)
    for _ in range(200):
        r = (1 - d) / n + d * P @ r
    return r


def graph():
    idx = {p: i for i, p in enumerate(PAPERS)}
    A = np.zeros((len(PAPERS), len(PAPERS)))
    for s, ds in CITES.items():
        for t in ds:
            A[idx[t], idx[s]] = 1.0
    return A, idx


def test_same_citations_different_pagerank():
    A, idx = graph()
    r = pagerank(A)
    a, b = idx["원전15"], idx["후속18"]
    assert A[a].sum() == A[b].sum() == 4
    assert round(r[a], 3) == 0.313 and round(r[b], 3) == 0.158
    assert correct("citation-count-meaning") == [0]


def test_reference_count_does_not_lower_own_score():
    # 후속18의 참고문헌을 하나 줄여도 후속18 자신의 점수는 그대로다(받는 인용이 같으므로 정상상태에서 자기 몫은 같음).
    A, idx = graph()
    r1 = pagerank(A)
    A2 = A.copy()
    A2[idx["후속17"], idx["후속18"]] = 0.0
    r2 = pagerank(A2)
    assert abs(r1[idx["후속18"]] - r2[idx["후속18"]]) < 0.01


def test_forward_is_citing_papers():
    seed = "원전15"
    forward = [p for p, ds in CITES.items() if seed in ds]
    assert set(forward) == {"후속17", "후속18", "반박19", "최신24"}
    assert CITES.get(seed) is None                     # backward로는 이후 연구를 찾을 수 없다
    assert correct("forward-snowball") == [0]


def test_other_answers():
    assert correct("dedup-identifier") == [0]
    assert correct("map-by-assumption") == [0, 1, 2]
    assert correct("cite-from-abstract") == [0]
