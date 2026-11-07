import numpy as np


def ring_all_reduce(arrays):
    """worker W개가 고리로 이어져 있다고 보고 ring all-reduce(합)를 흉내 낸다.

    arrays: 길이 W의 리스트. 각 원소는 worker 하나의 1차원 배열, 모두 같은 shape (n,)
    반환: (results, sent_bytes)
        results: 길이 W의 리스트. 모든 원소가 sum(arrays)와 같은 (n,) 배열
        sent_bytes: 길이 W의 리스트. worker마다 보낸 byte 수의 합
    규칙: 각 단계에서 worker i는 이웃 (i + 1) % W에게만 청크 하나를 보낸다. 입력 배열은 바꾸지 않는다.
    """
    W = len(arrays)
    chunks = [[c.copy() for c in np.array_split(np.asarray(a, dtype=float), W)] for a in arrays]  # [worker][청크]
    sent = [0] * W
    # 1단계 reduce-scatter: W-1번. 단계 s에서 worker i는 청크 (i - s) % W를 다음 worker에게 보내고, 받은 쪽은 더한다.
    for s in range(W - 1):
        msgs = [(i, (i - s) % W, chunks[i][(i - s) % W].copy()) for i in range(W)]   # 동시에 보낸다
        for i, c, data in msgs:
            chunks[(i + 1) % W][c] += data
            sent[i] += data.nbytes
    # 이제 worker i는 청크 (i + 1) % W의 전체 합을 가지고 있다.
    # 2단계 all-gather: W-1번. 단계 s에서 worker i는 청크 (i + 1 - s) % W를 다음 worker에게 보내고, 받은 쪽은 덮어쓴다.
    for s in range(W - 1):
        msgs = [(i, (i + 1 - s) % W, chunks[i][(i + 1 - s) % W].copy()) for i in range(W)]
        for i, c, data in msgs:
            chunks[(i + 1) % W][c] = data
            sent[i] += data.nbytes
    results = [np.concatenate(ch) for ch in chunks]
    return results, sent
