import numpy as np


def kmer_counts(seq, k, alphabet="ACGT"):
    """seq의 k-mer 개수 벡터. 길이 len(alphabet)**k, alphabet 순서의 사전식 정렬.

    소문자는 대문자로 바꾸고, 알파벳 밖 글자(N 등)가 들어간 k-mer는 세지 않는다.
    반환: (len(alphabet)**k,) 배열
    """
    n = len(alphabet)
    lookup = {c: i for i, c in enumerate(alphabet)}
    idx = [lookup.get(c, -1) for c in seq.upper()]       # 알파벳 밖 글자는 -1
    v = np.zeros(n ** k)                                 # (|A|^k,)
    for s in range(len(idx) - k + 1):
        window = idx[s:s + k]
        if min(window) < 0:
            continue
        code = 0
        for j in window:
            code = code * n + j
        v[code] += 1
    return v
