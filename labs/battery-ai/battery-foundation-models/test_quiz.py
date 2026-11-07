"""30.7 퀴즈(content/30-battery-ai/07-battery-foundation-models/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/30-battery-ai/07-battery-foundation-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run_cells():
    ns, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec((HERE / "cells.py").read_text(encoding="utf-8"), ns)
    return ns, buf.getvalue()


NS, OUT = run_cells()


def test_cycle_split_leak():
    assert "행 단위 무작위 분할 (A·B)  수명 오차   3.3 %" in OUT
    assert "셀 단위 분할 (A·B)         수명 오차  27.9 %" in OUT
    assert correct("cycle-split-leak") == [0]


def test_sign_convention_merge():
    # 부호가 반대인 두 데이터셋: 전류 적분과 용량 변화의 상관이 데이터셋마다 반대
    rng = np.random.default_rng(0)
    i_a = rng.uniform(0.5, 3.0, 100)
    dq_ah = i_a * 0.1                                   # 충전으로 용량 증가
    assert np.corrcoef(i_a, dq_ah)[0, 1] > 0.99 and np.corrcoef(-i_a, dq_ah)[0, 1] < -0.99
    assert correct("sign-convention-merge") == [0]


def test_missing_modality():
    assert correct("missing-modality") == [0]


def test_negative_transfer_chemistry():
    assert "미세조정이 나은 분할 16%, 이득 5–95% 범위 [-0.334, +0.134]" in OUT
    assert "D (다른 화학): RMSE 중앙값 처음부터 0.468, 미세조정 0.574, zero-shot 0.870" in OUT
    # 사전학습 쪽으로 강하게 정규화하면 zero-shot에 가까워진다
    ridge, w_pre = NS["ridge"], NS["w_pre"]
    rng = np.random.default_rng(1)
    Xt = rng.normal(size=(6, 8))
    yt = rng.normal(size=6)
    assert np.allclose(ridge(Xt, yt, w_pre, lam=1e8), w_pre, atol=1e-6)
    assert correct("negative-transfer-chemistry") == [0]


def test_controlled_transfer_baseline():
    assert correct("controlled-transfer-baseline") == [0, 1, 2]


def test_claim_checklist():
    assert correct("claim-checklist") == [0, 1, 2]
