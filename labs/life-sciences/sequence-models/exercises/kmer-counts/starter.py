import numpy as np


def kmer_counts(seq, k, alphabet="ACGT"):
    """seq의 k-mer 개수 벡터. 길이 len(alphabet)**k, alphabet 순서의 사전식 정렬.

    소문자는 대문자로 바꾸고, 알파벳 밖 글자(N 등)가 들어간 k-mer는 세지 않는다.
    반환: (len(alphabet)**k,) 배열
    """
    raise NotImplementedError("k-mer를 세는 코드를 작성하세요")
