"""7.7 실습: attention · bilinear layer · batch 공분산을 einsum과 loop로 각각 구현하고 비교한다.

실행: uv run --project labs python labs/linear-algebra/tensors-and-einsum/einsum_vs_loop.py

세 연산 모두 (1) einsum 한 줄, (2) Python loop로 구현해 결과가 같은지 확인하고 실행 시간을 잰다.
시간은 기계마다 다르다. 비교할 것은 비율과 결과의 일치다.
"""
import time

import numpy as np


def timed(fn, *args, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return out, best


# ---------------------------------------------------------------- attention
def attention_einsum(Q, K, V):
    """Q (B, T, D), K (B, S, D), V (B, S, D) → (B, T, D)"""
    scores = np.einsum("btd,bsd->bts", Q, K) / np.sqrt(Q.shape[-1])  # (B, T, S)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return np.einsum("bts,bsd->btd", w, V)                            # (B, T, D)


def attention_loop(Q, K, V):
    B, T, D = Q.shape
    S = K.shape[1]
    out = np.zeros((B, T, V.shape[-1]))
    for b in range(B):
        for t in range(T):
            s = np.array([Q[b, t] @ K[b, j] for j in range(S)]) / np.sqrt(D)  # (S,)
            w = np.exp(s - s.max())
            w /= w.sum()
            for j in range(S):
                out[b, t] += w[j] * V[b, j]
    return out


# ---------------------------------------------------------------- bilinear
def bilinear_einsum(x, W, z):
    """x (B, I), W (K, I, J), z (B, J) → y (B, K), y[b,k] = Σ_ij x[b,i] W[k,i,j] z[b,j]"""
    return np.einsum("bi,kij,bj->bk", x, W, z)


def bilinear_loop(x, W, z):
    B, K = x.shape[0], W.shape[0]
    y = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            y[b, k] = x[b] @ W[k] @ z[b]
    return y


# ---------------------------------------------------------------- batch covariance
def batch_cov_einsum(X):
    """X (B, N, D) → (B, D, D), 표본 공분산(N-1로 나눔)"""
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.einsum("bnd,bne->bde", Xc, Xc) / (X.shape[1] - 1)


def batch_cov_loop(X):
    B, N, D = X.shape
    out = np.zeros((B, D, D))
    for b in range(B):
        mu = X[b].mean(axis=0)
        for n in range(N):
            d = X[b, n] - mu
            out[b] += np.outer(d, d)
        out[b] /= N - 1
    return out


def main():
    rng = np.random.default_rng(0)
    cases = [
        ("attention  B=8,T=S=64,D=32", attention_einsum, attention_loop,
         (rng.normal(size=(8, 64, 32)), rng.normal(size=(8, 64, 32)), rng.normal(size=(8, 64, 32)))),
        ("bilinear   B=256,I=J=16,K=8", bilinear_einsum, bilinear_loop,
         (rng.normal(size=(256, 16)), rng.normal(size=(8, 16, 16)), rng.normal(size=(256, 16)))),
        ("batch cov  B=32,N=200,D=6  ", batch_cov_einsum, batch_cov_loop,
         (rng.normal(size=(32, 200, 6)),)),
    ]
    for name, fe, fl, args in cases:
        ye, te = timed(fe, *args)
        yl, tl = timed(fl, *args)
        print(f"{name}  shape {ye.shape}  일치 {np.allclose(ye, yl)}  "
              f"einsum {te * 1e3:7.2f} ms  loop {tl * 1e3:8.1f} ms  (loop/einsum ≈ {tl / te:,.0f}배)")


if __name__ == "__main__":
    main()
