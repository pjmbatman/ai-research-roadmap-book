"""17.9 실습: 체크리스트가 버그 다섯 개를 모두 잡고, 버그가 없으면 모두 통과하는지 확인한다."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from checklist import CHECKS  # noqa: E402
from pipeline import BUGS  # noqa: E402


def test_clean_pipeline_passes_all_checks():
    for name, fn in CHECKS:
        assert fn(set())[0], name


@pytest.mark.parametrize("bug", BUGS)
def test_each_bug_is_caught(bug):
    failed = [name for name, fn in CHECKS if not fn({bug})[0]]
    assert failed, f"{bug}를 잡는 검사가 없습니다"
