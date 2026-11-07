import numpy as np


def make_mask(lengths, T, causal):
    """padding mask(와 causal mask)를 합친 self-attention용 bool mask.

    lengths: (B,) 각 시퀀스의 실제 길이(1 이상 T 이하). 그 뒤는 PAD
    T: batch의 시퀀스 길이
    causal: True이면 query t는 key s <= t만 본다
    반환: (B, T, T) bool, mask[b, t, s] = query t가 key s를 볼 수 있는가
    """
    raise NotImplementedError("key 쪽 padding 조건과 causal 조건을 broadcasting으로 합치세요")
