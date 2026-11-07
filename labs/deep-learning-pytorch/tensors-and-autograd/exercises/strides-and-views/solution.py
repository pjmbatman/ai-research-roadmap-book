def row_major_strides(shape):
    """행 우선 텐서의 stride(원소 단위). 예: (2, 3, 4) -> (12, 4, 1)"""
    strides, step = [], 1
    for n in reversed(tuple(shape)):
        strides.append(step)
        step *= n
    return tuple(reversed(strides))


def permute_strides(shape, strides, dims):
    """permute(dims) 결과의 (shape, strides)."""
    return tuple(shape[d] for d in dims), tuple(strides[d] for d in dims)


def is_contiguous(shape, strides):
    """행 우선으로 빈틈없이 놓였는지. 길이 1인 축의 stride는 무시한다."""
    expected = row_major_strides(shape)
    return all(n == 1 or s == e for n, s, e in zip(shape, strides, expected))
