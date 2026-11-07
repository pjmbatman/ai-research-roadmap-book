def row_major_strides(shape):
    """행 우선 텐서의 stride(원소 단위). 예: (2, 3, 4) -> (12, 4, 1)"""
    raise NotImplementedError("뒤 축들의 길이를 곱하세요")


def permute_strides(shape, strides, dims):
    """permute(dims) 결과의 (shape, strides)."""
    raise NotImplementedError("축 순서를 dims대로 바꾸세요")


def is_contiguous(shape, strides):
    """행 우선으로 빈틈없이 놓였는지. 길이 1인 축의 stride는 무시한다."""
    raise NotImplementedError("행 우선 stride와 비교하세요")
