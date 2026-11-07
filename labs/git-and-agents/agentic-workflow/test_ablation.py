"""4.6 실습 ablation 코드의 무결성 장치(고정 분할, 분할 지문, test 접근 기록)를 확인한다."""
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "ablation"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "ablation" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp, fte = load("experiment"), load("final_test_eval")
QUIZ = HERE.parents[2] / "content/04-git-and-agents/06-agentic-workflow/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_split_is_disjoint_and_fixed():
    tr, va, te = exp.split(1200)
    assert len(set(tr) | set(va) | set(te)) == 1200
    assert not (set(tr) & set(te)) and not (set(va) & set(te))
    assert exp.split_fingerprint(te) == exp.split_fingerprint(exp.split(1200)[2])
    assert exp.split_fingerprint(te) != exp.split_fingerprint(exp.split(1200, seed=1)[2])


def test_drop_removes_group():
    assert exp.columns("noise") == [0, 1, 2, 3, 4]
    assert len(exp.columns(None)) == 15


def test_run_reports_mean_std_and_fingerprint():
    res = exp.run("noise", seeds=(0, 1))
    assert len(res["val_rmse"]) == 2 and res["val_rmse_std"] >= 0
    assert res["test_fingerprint"] == exp.split_fingerprint(exp.split(1200)[2])


def test_final_eval_logs_every_access(tmp_path):
    log = tmp_path / "access.log"
    for _ in range(2):
        fte.evaluate("noise", 0, log_path=log)
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert all(e["test_fingerprint"] == entries[0]["test_fingerprint"] for e in entries)


def test_selection_bias_quiz():
    rng = np.random.default_rng(1)
    acc = rng.binomial(200, 0.7, size=(4000, 20)) / 200
    assert acc.max(axis=1).mean() > 0.74                   # 20개 중 최고는 참값 0.70보다 크다
    assert correct("test-set-selection") == [0]
