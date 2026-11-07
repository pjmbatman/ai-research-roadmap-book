import numpy as np


def random_fourier_features(X, m, length_scale, seed=0):
    """RBF 커널(길이 척도 length_scale)을 근사하는 random Fourier features.

    X: (N, d). 반환: Z (N, m), Z @ Z.T ≈ RBF Gram 행렬
    """
    raise NotImplementedError("ω ~ N(0, ℓ⁻² I), b ~ U(0, 2π)로 sqrt(2/m) cos(X ω + b)를 만드세요")
