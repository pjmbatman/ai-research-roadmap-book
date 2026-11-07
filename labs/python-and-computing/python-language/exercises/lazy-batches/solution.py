from collections.abc import Iterable, Iterator
from itertools import islice


def batched(samples: Iterable, batch_size: int) -> Iterator[list]:
    """samples를 크기 batch_size의 리스트로 묶어 하나씩 내보낸다.

    마지막 batch는 더 작을 수 있다. batch_size < 1이면 호출하는 즉시 ValueError.
    입력은 필요한 만큼만 읽는다(끝이 없는 스트림에서도 동작).
    """
    if batch_size < 1:                 # generator 바깥에서 검사해야 호출 즉시 오류가 난다
        raise ValueError("batch_size는 1 이상이어야 합니다")
    return _batches(iter(samples), batch_size)


def _batches(it: Iterator, batch_size: int) -> Iterator[list]:
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch
