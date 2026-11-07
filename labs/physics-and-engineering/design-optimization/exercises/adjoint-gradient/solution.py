import numpy as np


def adjoint_gradient(k, f, c, h):
    """1차원 정상 열전도 K(k) T = f에서 J = c^T T와 dJ/dk를 adjoint 방법으로 계산한다.

    k: (n,) 요소 열전도도, f: (n,) 절점 열부하, c: (n,) 목적 가중치, h: 요소 길이
    반환: J (float), grad (n,)
    """
    k, f, c = (np.asarray(v, dtype=float) for v in (k, f, c))
    main = np.append(k[:-1] + k[1:], k[-1]) / h                     # (n,) 절점 1..n의 대각 원소
    K = np.diag(main) - np.diag(k[1:] / h, 1) - np.diag(k[1:] / h, -1)
    T = np.linalg.solve(K, f)                                        # 해석
    lam = np.linalg.solve(K.T, c)                                    # adjoint 해석
    Tt, lt = np.concatenate([[0.0], T]), np.concatenate([[0.0], lam])   # 고정 절점 0을 붙인다
    grad = -np.diff(lt) * np.diff(Tt) / h                            # (n,)
    return float(c @ T), grad
