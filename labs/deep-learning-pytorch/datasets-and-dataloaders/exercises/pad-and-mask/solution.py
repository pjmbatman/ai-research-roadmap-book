import numpy as np


def pad_batch(seqs, pad_value=0.0):
    """(T_i, C) 배열 list -> (x (B, T_max, C), mask (B, T_max) bool, lengths (B,))"""
    seqs = [np.asarray(s, dtype=float) for s in seqs]
    lengths = np.array([len(s) for s in seqs], dtype=int)              # (B,)
    B, T, C = len(seqs), int(lengths.max()), seqs[0].shape[1]
    x = np.full((B, T, C), pad_value, dtype=float)                     # (B, T_max, C)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    mask = np.arange(T)[None, :] < lengths[:, None]                    # (B, T_max)
    return x, mask, lengths


def masked_mse(pred, target, mask):
    """mask가 True인 위치에서만 제곱오차의 원소 평균. mask가 모두 False면 ValueError."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("mask에 실제 값이 하나도 없다")
    err = (np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)) ** 2   # (B, T, C)
    return float(err[mask].mean())                                      # err[mask]: (N_valid, C)
