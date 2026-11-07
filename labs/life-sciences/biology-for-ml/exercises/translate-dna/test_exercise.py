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


translate = load().translate


def test_basic_and_stop():
    assert translate("ATGGCTTAA") == "MA"
    assert translate("ATGGCTAAAGGTTGGTAA") == "MAKGW"
    assert translate("ATGTAAGCT") == "M", "정지 코돈에서 멈춰야 합니다(뒤의 코돈은 번역하지 않음)"


def test_trailing_bases_dropped():
    assert translate("ATGGCTGC") == "MA", "끝에 남는 1–2개 염기는 버립니다"


def test_lowercase():
    assert translate("atggcttaa") == "MA", "소문자도 대문자로 바꿔 처리합니다"


def test_frames():
    seq = "AATGGCTTGG"
    assert translate(seq, frame=0) == "NGL"
    assert translate(seq, frame=1) == "MAW", "frame=1은 두 번째 염기부터 읽습니다"
    assert translate(seq, frame=2) == "WL"


def test_minus_strand():
    assert translate("TTAAGCCAT", strand="-") == "MA", "'-' 가닥은 역상보(뒤집고 A↔T, C↔G) 서열입니다"
    assert translate("CCATTTCAT", strand="-") == "MKW"


def test_minus_strand_frame():
    # 역상보: ATGGCTTGGA → frame 1: TGG CTT GGA = W L G
    assert translate("TCCAAGCCAT", frame=1, strand="-") == "WLG"
