"""0.2 퀴즈(content/00-orientation/02-how-to-read/quiz.toml) 정답을 계산과 book.toml로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[3]
QUIZ = ROOT / "content/00-orientation/02-how-to-read/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
BOOK = tomllib.loads((ROOT / "book.toml").read_text(encoding="utf-8"))
LAYER = {t["code"]: t["layer"] for t in BOOK["tracks"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_shape_reading():
    X = np.zeros((32, 3))     # (N, D)
    W = np.zeros((3, 16))     # (D, H)
    assert (X @ W).size == Q["shape-reading"]["answer"]


def test_layer_rule():
    # 선택지 순서대로 (장의 트랙, 선수 장의 트랙)
    links = [("SML", "MLA"), ("DOM", "SML"), ("MTH", "MLC"), ("RSE", "INF")]
    violates = [i for i, (a, b) in enumerate(links) if LAYER[b] > LAYER[a]]
    assert correct("layer-rule") == violates

