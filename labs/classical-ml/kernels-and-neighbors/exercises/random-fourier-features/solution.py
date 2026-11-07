import numpy as np


def random_fourier_features(X, m, length_scale, seed=0):
    """RBF 커널(길이 척도 length_scale)을 근사하는 random Fourier features.

    X: (N, d). 반환: Z (N, m), Z @ Z.T ≈ RBF Gram 행렬
    """
    X = np.asarray(X, dtype=float)                                  # (N, d)
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 1.0 / length_scale, size=(X.shape[1], m))   # (d, m)
    b = rng.uniform(0.0, 2 * np.pi, size=m)                         # (m,)
    return np.sqrt(2.0 / m) * np.cos(X @ omega + b)                 # (N, m)
