import numpy as np


def ring_all_reduce(arrays):
    """worker W개가 고리로 이어져 있다고 보고 ring all-reduce(합)를 흉내 낸다.

    arrays: 길이 W의 리스트. 각 원소는 worker 하나의 1차원 배열, 모두 같은 shape (n,)
    반환: (results, sent_bytes)
        results: 길이 W의 리스트. 모든 원소가 sum(arrays)와 같은 (n,) 배열
        sent_bytes: 길이 W의 리스트. worker마다 보낸 byte 수의 합
    규칙: 각 단계에서 worker i는 이웃 (i + 1) % W에게만 청크 하나를 보낸다. 입력 배열은 바꾸지 않는다.
    """
    raise NotImplementedError("배열을 W개 청크로 나누고 reduce-scatter와 all-gather를 차례로 구현하세요")
