"""3.5 퀴즈(content/03-environments/05-docker/quiz.toml) 정답을 계산으로 검증한다."""
import hashlib
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/03-environments/05-docker/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_shm_batches():
    num_workers, prefetch_factor, mib_per_batch = 4, 2, 16
    q = Q["shm-batches"]
    assert num_workers * prefetch_factor * mib_per_batch == q["answer"] == 128
    s = q["slider"]
    assert s["min"] <= q["answer"] <= s["max"] and (q["answer"] - s["min"]) % s["step"] == 0
    assert q["answer"] > 64                                   # Docker 기본 /dev/shm 64 MiB를 넘는다


def test_digest_is_content_hash():
    a = hashlib.sha256(b'{"layers": ["base", "numpy-2.5.4"]}').hexdigest()
    b = hashlib.sha256(b'{"layers": ["base", "numpy-2.5.5"]}').hexdigest()
    assert a != b and a == hashlib.sha256(b'{"layers": ["base", "numpy-2.5.4"]}').hexdigest()
    assert correct("tag-vs-digest") == [0]


def test_layer_order():
    def rebuilt(df, changed):
        for i, (_, inputs) in enumerate(df):
            if inputs & changed:
                return len(df) - i
        return 0
    code_first = [("FROM", set()), ("COPY .", {"lock", "code"}), ("RUN sync", set())]
    lock_first = [("FROM", set()), ("COPY lock", {"lock"}), ("RUN sync", set()), ("COPY .", {"lock", "code"})]
    assert rebuilt(code_first, {"code"}) == 2 and rebuilt(lock_first, {"code"}) == 1
    assert correct("layer-order") == [0]


def test_other_answers():
    assert correct("image-vs-container") == [0]
    assert correct("host-driver") == [0]
    assert correct("results-outside") == [0]
    assert correct("data-in-image") == [0, 1, 2]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
