import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


next_targets = load().next_targets


def row(item, claimed, evidenced, needed=True):
    return {"item": item, "claimed": claimed, "evidenced": evidenced, "needed": needed}


def test_lowest_levels_first():
    rows = [row("개념 설명", 4, 4), row("유도→코드", 2, 2), row("재현 가능성", 3, 3), row("한계 명시", 1, 1)]
    assert next_targets(rows) == ["한계 명시", "유도→코드"], "수준이 가장 낮은 두 항목을 낮은 순서로 반환해야 합니다"


def test_evidence_rule_lowers_level():
    # 스스로는 4로 매겼지만 근거는 1까지만 있다 → 수준 1로 본다
    rows = [row("실패 원인 분리", 4, 1), row("주장 판단", 2, 2), row("논문 연결", 3, 3)]
    assert next_targets(rows) == ["실패 원인 분리", "주장 판단"], (
        "수준은 min(claimed, evidenced)로 계산합니다. 근거 없는 수준은 인정하지 않습니다"
    )


def test_evidence_above_claim_does_not_raise_level():
    # 근거가 더 높아도 스스로 매긴 수준을 넘지 않는다
    rows = [row("A", 1, 5), row("B", 2, 2)]
    assert next_targets(rows, k=1) == ["A"]


def test_only_needed_items():
    rows = [row("운영: 분산 학습", 0, 0, needed=False), row("주장 판단", 2, 2), row("한계 명시", 3, 3)]
    assert next_targets(rows) == ["주장 판단", "한계 명시"], "자기 연구에 필요하지 않은 항목(needed=False)은 고르지 않습니다"


def test_ties_keep_table_order():
    rows = [row("C", 2, 2), row("A", 2, 3), row("B", 3, 2)]
    assert next_targets(rows, k=3) == ["C", "A", "B"], "수준이 같으면 rows의 순서를 따릅니다(이름순이 아닙니다)"


def test_k_larger_than_needed_items():
    rows = [row("A", 3, 3), row("B", 1, 1, needed=False)]
    assert next_targets(rows, k=5) == ["A"]


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        next_targets([row("A", 6, 2)])
    with pytest.raises(ValueError):
        next_targets([row("A", 2, -1)])
    with pytest.raises(ValueError):
        next_targets([row("A", 2.5, 2)])
