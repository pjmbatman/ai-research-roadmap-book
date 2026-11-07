"""15.5 퀴즈(content/15-data/05-dataset-versioning/quiz.toml) 정답을 계산으로 검증한다."""
import hashlib
import hmac
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/15-data/05-dataset-versioning/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def test_content_hash_vs_name():
    a, b = b"cell,cap\nC00,2.000\n", b"cell,cap\nC00,1.999\n"
    assert len(a) == len(b) and sha(a) != sha(b)              # 크기는 같아도 해시는 다르다
    assert correct("content-hash-vs-name") == [0]


def test_crlf_hash():
    lf, crlf = b"a,b\n1,2\n", b"a,b\r\n1,2\r\n"
    assert sha(lf) != sha(crlf)
    assert sha(crlf.replace(b"\r\n", b"\n")) == sha(lf)
    assert correct("crlf-hash") == [0]


def test_hash_split_stability():
    split = lambda c: int(sha(c.encode()), 16) % 1000 < 200
    v1 = [f"C{i:03d}" for i in range(100)]
    before = {c: split(c) for c in v1}
    after = {c: split(c) for c in v1 + [f"C{i:03d}" for i in range(100, 150)]}
    changed = sum(before[c] != after[c] for c in v1)
    assert changed == Q["hash-split-stability"]["answer"] == 0


def test_unsalted_hash_is_reversible():
    table = {sha(f"P{i:04d}".encode()): f"P{i:04d}" for i in range(10_000)}
    assert table[sha(b"P0427")] == "P0427"
    pseudo = hmac.new(b"secret", b"P0427", hashlib.sha256).hexdigest()
    assert pseudo not in table
    assert correct("unsalted-hash-pseudonym") == [0]


def test_other_keys():
    assert correct("datasheet-contents") == [0, 1, 2]
    assert correct("no-license") == [0]
    assert correct("retrain-scope") == [0]
