import numpy as np


def sample_probs(logits, temperature=1.0, top_p=1.0):
    """온도를 적용한 softmax에 nucleus(top-p) 필터를 걸어, 샘플링에 쓸 확률 벡터 (V,)를 반환한다."""
    raise NotImplementedError("온도 softmax와 top-p 필터를 구현하세요")
