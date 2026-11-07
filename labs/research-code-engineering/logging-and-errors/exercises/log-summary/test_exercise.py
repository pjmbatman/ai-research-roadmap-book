import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summarize = load().summarize

LOG = [
    '{"level": "INFO", "event": "config", "lr": 0.3, "seed": 0}',
    '{"level": "INFO", "event": "train", "step": 0, "loss": 2.31}',
    '{"level": "WARNING", "event": "grad_clip", "step": 1, "norm": 54.2}',
    '{"level": "INFO", "event": "train", "step": 1, "loss": 1.72}',
    '{"level": "INFO", "event": "train", "step": 2, "loss": NaN}',
    '{"level": "ERROR", "event": "abort", "step": 2}',
    '{"level": "INFO", "event": "tr',
]


def test_example_log():
    s = summarize(LOG)
    assert s["levels"] == {"INFO": 4, "WARNING": 1, "ERROR": 1}, f"레벨별 개수가 다릅니다: {s['levels']}"
    assert s["first_bad_step"] == 2, f"처음으로 loss가 NaN인 step은 2입니다. 현재 {s['first_bad_step']}"
    assert s["config"] == {"lr": 0.3, "seed": 0}, f"config에서 level·event를 뺀 dict입니다. 현재 {s['config']}"
    assert s["malformed"] == 1, "잘린 마지막 줄 하나가 malformed입니다"


def test_infinity_and_null_are_bad():
    for token in ["Infinity", "-Infinity", "null"]:
        s = summarize([f'{{"step": 7, "loss": {token}}}'])
        assert s["first_bad_step"] == 7, f"loss가 {token}이면 유한하지 않은 값입니다"


def test_first_in_log_order():
    lines = ['{"step": 5, "loss": NaN}', '{"step": 3, "loss": NaN}']   # 재시작으로 step이 되돌아간 로그
    assert summarize(lines)["first_bad_step"] == 5, "가장 작은 step이 아니라 로그에서 처음 나온 레코드입니다"


def test_no_bad_and_no_config():
    s = summarize(['{"level": "INFO", "step": 0, "loss": 1.0}', '{"level": "INFO", "event": "eval"}'])
    assert s["first_bad_step"] is None and s["config"] is None
    assert s["malformed"] == 0


def test_blank_lines_print_lines_and_non_objects():
    lines = ["", "   ", "epoch 3 done", "[1, 2, 3]", '"just a string"', '{"level": "DEBUG"}']
    s = summarize(lines)
    assert s["malformed"] == 3, f"print 줄 하나와 객체가 아닌 JSON 두 줄이 malformed입니다. 현재 {s['malformed']}"
    assert s["levels"] == {"DEBUG": 1}


def test_empty_log():
    assert summarize([]) == {"levels": {}, "first_bad_step": None, "config": None, "malformed": 0}


def test_first_config_wins():
    lines = ['{"event": "config", "lr": 0.1}', '{"event": "config", "lr": 0.9}']
    assert summarize(lines)["config"] == {"lr": 0.1}, "첫 config 레코드를 씁니다"
