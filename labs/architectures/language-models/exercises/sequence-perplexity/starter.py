import numpy as np


def perplexity(logits, targets):
    """위치별 logit과 정답 토큰으로 perplexity를 계산한다.

    logits: (T, V) 위치마다의 logit
    targets: (T,) 정답 토큰 번호
    반환: float, exp(평균 음의 로그우도)
    """
    raise NotImplementedError("log-sum-exp로 로그 확률을 구하고 perplexity를 계산하세요")
