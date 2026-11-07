import numpy as np


def pad_batch(seqs, pad_value=0.0):
    """(T_i, C) 배열 list -> (x (B, T_max, C), mask (B, T_max) bool, lengths (B,))"""
    raise NotImplementedError("가장 긴 길이에 맞춰 채우고 mask를 만드세요")


def masked_mse(pred, target, mask):
    """mask가 True인 위치에서만 제곱오차의 원소 평균. mask가 모두 False면 ValueError."""
    raise NotImplementedError("mask로 고른 원소에서만 평균하세요")
