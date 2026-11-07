import numpy as np


def identity_clusters(seqs, threshold):
    """길이가 같은 서열들의 단일 연결 군집 번호.

    동일성(같은 위치에 같은 글자의 비율) >= threshold인 쌍을 잇는 그래프의 연결 성분.
    번호는 처음 등장하는 순서대로 0, 1, 2, ...
    반환: (len(seqs),) 정수 배열
    """
    S = np.array([[ord(c) for c in s] for s in seqs])      # (N, L)
    ident = (S[:, None, :] == S[None, :, :]).mean(-1)       # (N, N)
    adj = ident >= threshold
    labels = np.full(len(seqs), -1)
    k = 0
    for start in range(len(seqs)):                          # 처음 등장 순서대로 번호를 붙인다
        if labels[start] >= 0:
            continue
        labels[start] = k
        stack = [start]
        while stack:                                        # 깊이 우선 탐색으로 연결 성분 채우기
            i = stack.pop()
            for j in np.where(adj[i] & (labels < 0))[0]:
                labels[j] = k
                stack.append(j)
        k += 1
    return labels
