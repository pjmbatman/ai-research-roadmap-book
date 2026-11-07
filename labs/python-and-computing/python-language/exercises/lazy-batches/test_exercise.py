import importlib.util
import itertools
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


batched = load().batched


class CountingStream:
    """next()가 몇 번 불렸는지 세는 iterator. len()과 인덱싱을 지원하지 않는다."""

    def __init__(self, n=None):
        self.n, self.reads = n, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n is not None and self.reads >= self.n:
            raise StopIteration
        self.reads += 1
        return self.reads - 1


def test_basic():
    assert list(batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batched(range(6), 3)) == [[0, 1, 2], [3, 4, 5]], "빈 batch를 내보내면 안 됩니다"
    assert list(batched([], 4)) == []
    assert list(batched("abcde", 2)) == [["a", "b"], ["c", "d"], ["e"]]


def test_invalid_batch_size_raises_on_call():
    with pytest.raises(ValueError):
        batched(range(5), 0)            # next()를 부르기 전에 오류가 나야 합니다


def test_infinite_stream():
    first_three = list(itertools.islice(batched(itertools.count(), 4), 3))
    assert first_three == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_reads_only_what_is_needed():
    stream = CountingStream()
    gen = batched(stream, 5)
    assert stream.reads == 0, "generator를 만들기만 했는데 입력을 읽었습니다"
    next(gen)
    assert stream.reads == 5, f"첫 batch에 필요한 5개만 읽어야 합니다. 현재 {stream.reads}개"
    next(gen)
    assert stream.reads == 10


def test_works_with_plain_iterator():
    assert list(batched(CountingStream(5), 2)) == [[0, 1], [2, 3], [4]]


def test_returns_iterator():
    gen = batched(range(3), 2)
    assert iter(gen) is gen, "list가 아니라 iterator(generator)를 반환해야 합니다"
