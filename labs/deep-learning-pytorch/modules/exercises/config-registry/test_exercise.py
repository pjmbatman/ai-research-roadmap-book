import copy
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


Registry = load().Registry


def make():
    reg = Registry()

    @reg.register("linear")
    class Linear:
        def __init__(self, d_in, d_out, bias=True):
            self.d_in, self.d_out, self.bias = d_in, d_out, bias

    @reg.register("seq")
    class Seq:
        def __init__(self, layers):
            self.layers = layers

    @reg.register("residual")
    class Residual:
        def __init__(self, inner):
            self.inner = inner

    return reg, Linear, Seq, Residual


def test_register_returns_class():
    reg = Registry()

    class A:
        pass

    assert reg.register("a")(A) is A, "decorator는 받은 클래스를 그대로 돌려줘야 합니다"


def test_build_simple():
    reg, Linear, _, _ = make()
    m = reg.build({"type": "linear", "d_in": 3, "d_out": 8})
    assert isinstance(m, Linear) and (m.d_in, m.d_out, m.bias) == (3, 8, True)


def test_build_nested_list():
    reg, Linear, Seq, _ = make()
    cfg = {"type": "seq", "layers": [{"type": "linear", "d_in": 3, "d_out": 8},
                                     {"type": "linear", "d_in": 8, "d_out": 1, "bias": False}]}
    m = reg.build(cfg)
    assert isinstance(m, Seq) and len(m.layers) == 2
    assert all(isinstance(l, Linear) for l in m.layers), "list 안의 설정도 build해야 합니다"
    assert m.layers[1].bias is False


def test_build_nested_dict():
    reg, Linear, _, Residual = make()
    m = reg.build({"type": "residual", "inner": {"type": "linear", "d_in": 4, "d_out": 4}})
    assert isinstance(m.inner, Linear), "type을 가진 dict 인자도 build해야 합니다"


def test_plain_values_untouched():
    reg = Registry()

    @reg.register("holder")
    class Holder:
        def __init__(self, sizes, meta):
            self.sizes, self.meta = sizes, meta

    m = reg.build({"type": "holder", "sizes": [32, 32], "meta": {"unit": "K"}})
    assert m.sizes == [32, 32] and m.meta == {"unit": "K"}, "type이 없는 dict와 숫자 list는 그대로 넘깁니다"


def test_cfg_not_modified_and_repeatable():
    reg, _, Seq, _ = make()
    cfg = {"type": "seq", "layers": [{"type": "linear", "d_in": 3, "d_out": 8}]}
    before = copy.deepcopy(cfg)
    a, b = reg.build(cfg), reg.build(cfg)
    assert cfg == before, "build가 설정 dict를 바꾸면 안 됩니다"
    assert a is not b and a.layers[0] is not b.layers[0], "build할 때마다 새 객체를 만들어야 합니다"


def test_unknown_type_lists_available():
    reg, *_ = make()
    with pytest.raises(KeyError) as e:
        reg.build({"type": "lineer", "d_in": 1, "d_out": 1})
    msg = str(e.value)
    assert "lineer" in msg and "['linear', 'residual', 'seq']" in msg, \
        "오류 메시지에 잘못된 type과 정렬된 등록 이름 목록이 있어야 합니다"


def test_duplicate_name_rejected():
    reg, *_ = make()
    with pytest.raises(KeyError):
        reg.register("linear")(type("Other", (), {}))
