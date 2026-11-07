"""22.1 퀴즈(content/22-experiment-operations/01-experiment-tracking/quiz.toml) 정답을 계산으로 검증한다."""
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/01-experiment-tracking/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def config_hash(cfg, ignored=("out_dir", "log_level")):
    core = {k: v for k, v in cfg.items() if k not in ignored}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]


def test_config_hash_ignored_keys():
    a = {"lr": 0.1, "seed": 0, "out_dir": "/scratch/a", "log_level": "INFO"}
    b = {"seed": 0, "lr": 0.1, "out_dir": "/home/b", "log_level": "DEBUG"}
    assert config_hash(a) == config_hash(b)                     # 결과와 무관한 키·키 순서는 hash를 바꾸지 않는다
    assert config_hash(a) != config_hash({**a, "seed": 1})       # seed는 결과를 바꾸므로 hash에 남는다
    assert correct("config-hash-ignored-keys") == [0]


def test_jsonl_survives_truncation():
    records = [{"epoch": e, "val_loss": 1.0 / e} for e in range(1, 6)]
    jsonl = "".join(json.dumps(r) + "\n" for r in records)
    whole = json.dumps(records)
    cut_l, cut_w = jsonl[: len(jsonl) - 10], whole[: len(whole) - 10]   # 쓰는 도중 죽었다
    good = []
    for line in cut_l.splitlines():
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert len(good) == 4                                        # 마지막 한 줄만 잃는다
    try:
        json.loads(cut_w)
        whole_ok = True
    except json.JSONDecodeError:
        whole_ok = False
    assert not whole_ok                                          # 통째로 쓴 파일은 전부 읽을 수 없다
    assert correct("jsonl-append") == [0]


def test_naive_mean_with_failed():
    naive = np.mean([0.30, 0.32, 0.31, 7.9])
    assert np.isclose(naive, Q["naive-mean-with-failed"]["answer"], atol=Q["naive-mean-with-failed"]["tolerance"])
    assert np.isclose(np.mean([0.30, 0.32, 0.31]), 0.31)


def test_choice_keys():
    assert correct("what-to-record") == [0, 1, 2]
    assert correct("dirty-commit") == [1]
    assert correct("failed-run-in-table") == [0]
    assert correct("dashboard-vs-raw") == [0]


def test_ema_smoothing_changes_last_value():
    acc = np.array([0.60, 0.70, 0.78, 0.83, 0.86, 0.88])         # (6,) 원본 검증 정확도
    s, smoothed = acc[0], []
    for v in acc:                                                # 지수이동평균(weight 0.9)
        s = 0.9 * s + 0.1 * v
        smoothed.append(s)
    assert not np.isclose(smoothed[-1], acc[-1])
