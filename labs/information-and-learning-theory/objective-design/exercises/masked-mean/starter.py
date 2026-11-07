import numpy as np


def masked_token_sum_count(loss, mask):
    """실제 token 손실의 합과 token 수를 (float, int)로 반환한다. loss, mask: (B, T)"""
    raise NotImplementedError("mask로 합과 개수를 구하세요")


def masked_token_mean(loss, mask):
    """실제 token에 대한 평균 손실."""
    raise NotImplementedError("token 평균을 계산하세요")


def masked_sequence_mean(loss, mask):
    """시퀀스별 평균의 평균. 실제 token이 없는 시퀀스는 뺀다."""
    raise NotImplementedError("시퀀스 평균의 평균을 계산하세요")
