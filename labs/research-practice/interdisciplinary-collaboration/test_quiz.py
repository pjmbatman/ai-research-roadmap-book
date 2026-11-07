"""23.13 퀴즈(content/23-research-practice/13-interdisciplinary-collaboration/quiz.toml)에서 계산으로 확인할 수 있는 정답을 검증한다."""
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/13-interdisciplinary-collaboration/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def test_hit_rate_enrichment():
    hits, picked, base_rate = 3, 10, 0.05
    enrichment = (hits / picked) / base_rate
    q = Q["hit-rate-enrichment"]
    assert abs(enrichment - q["answer"]) <= q["tolerance"]
    lo, hi, step = q["slider"]["min"], q["slider"]["max"], q["slider"]["step"]
    assert lo <= q["answer"] <= hi and abs((q["answer"] - lo) / step - round((q["answer"] - lo) / step)) < 1e-9
