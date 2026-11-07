import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_streams = load().make_streams


def draws(gen, n=5):
    return gen.random(n)                                    # (n,)


def test_returns_generators_for_all_names():
    s = make_streams(0, ["init", "data", "aug"])
    assert set(s) == {"init", "data", "aug"}, "이름마다 생성기 하나를 반환해야 합니다"
    assert all(isinstance(g, np.random.Generator) for g in s.values()), "값은 np.random.Generator여야 합니다"


def test_same_seed_and_name_reproduce():
    a = draws(make_streams(7, ["init"])["init"])
    b = draws(make_streams(7, ["init"])["init"])
    assert np.array_equal(a, b), "같은 seed·같은 이름이면 같은 난수가 나와야 합니다"


def test_adding_a_stream_does_not_change_others():
    before = make_streams(0, ["init", "data"])
    after = make_streams(0, ["init", "aug", "data", "dropout"])
    for name in ["init", "data"]:
        assert np.array_equal(draws(before[name]), draws(after[name])), \
            f"새 용도를 추가했더니 '{name}' 스트림이 바뀌었습니다. 스트림은 (seed, 이름)만으로 정해져야 합니다"


def test_order_does_not_matter():
    a = make_streams(3, ["init", "data", "aug"])
    b = make_streams(3, ["aug", "init", "data"])
    for name in a:
        assert np.array_equal(draws(a[name]), draws(b[name])), "목록의 순서가 스트림을 바꾸면 안 됩니다"


def test_different_names_and_seeds_differ():
    s = make_streams(0, ["init", "data"])
    assert not np.array_equal(draws(s["init"]), draws(s["data"])), "이름이 다르면 다른 스트림이어야 합니다"
    a = draws(make_streams(0, ["init"])["init"])
    b = draws(make_streams(1, ["init"])["init"])
    assert not np.array_equal(a, b), "seed가 다르면 다른 스트림이어야 합니다"


def test_streams_are_independent_objects():
    s = make_streams(0, ["init", "data"])
    first = draws(make_streams(0, ["data"])["data"])
    s["init"].random(1000)                                  # init 스트림을 많이 소비해도
    assert np.array_equal(draws(s["data"]), first), "한 스트림의 소비가 다른 스트림에 영향을 주면 안 됩니다"
