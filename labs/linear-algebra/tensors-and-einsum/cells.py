"""7.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/tensors-and-einsum/cells.py
"""
# %% cell 1 — einsum과 익숙한 연산의 대응
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 4))          # (I, J)
B = rng.normal(size=(4, 5))          # (J, K)
C = rng.normal(size=(3, 4))          # (I, J)
S = rng.normal(size=(4, 4))          # (J, J)

print("행렬 곱 ij,jk->ik :", np.allclose(np.einsum("ij,jk->ik", A, B), A @ B))
print("전치   ij->ji     :", np.allclose(np.einsum("ij->ji", A), A.T))
print("trace  ii->       :", np.isclose(np.einsum("ii->", S), np.trace(S)))
print("대각   ii->i      :", np.allclose(np.einsum("ii->i", S), np.diag(S)))
print("행 합  ij->i      :", np.allclose(np.einsum("ij->i", A), A.sum(axis=1)))
print("Frobenius ij,ij-> :", np.isclose(np.einsum("ij,ij->", A, C), np.trace(A.T @ C)))

# %% cell 2 — attention 점수: einsum 한 줄과 loop
Bn, T, Sn, D = 2, 3, 4, 8
Q = rng.normal(size=(Bn, T, D))                           # (B, T, D)
K = rng.normal(size=(Bn, Sn, D))                          # (B, S, D)
V = rng.normal(size=(Bn, Sn, D))                          # (B, S, D)

scores = np.einsum("btd,bsd->bts", Q, K) / np.sqrt(D)    # (B, T, S)
w = np.exp(scores - scores.max(axis=-1, keepdims=True))
w = w / w.sum(axis=-1, keepdims=True)                     # (B, T, S), 마지막 축 합 1
out = np.einsum("bts,bsd->btd", w, V)                     # (B, T, D)

loop = np.zeros((Bn, T, Sn))
for b in range(Bn):
    for t in range(T):
        for s in range(Sn):
            loop[b, t, s] = Q[b, t] @ K[b, s] / np.sqrt(D)
print("scores", scores.shape, " out", out.shape)
print("loop과 일치:", np.allclose(scores, loop))
print("가중치 행 합:", w.sum(axis=-1)[0])

# %% cell 3 — reshape와 transpose는 다르다
X = np.arange(6).reshape(2, 3)       # (2, 3)
print("X.T =\n", X.T)
print("X.reshape(3, 2) =\n", X.reshape(3, 2))

P = np.array([[1, 10], [100, 1000]])
R = np.array([[1, 2], [3, 4]])
kron = np.einsum("ij,kl->ikjl", P, R).reshape(4, 4)   # (2, 2, 2, 2) → (4, 4)
bad = np.einsum("ij,kl->ijkl", P, R).reshape(4, 4)    # 축 순서를 잘못 둔 경우
print("ikjl → Kronecker 일치:", np.array_equal(kron, np.kron(P, R)))
print("ijkl → Kronecker 일치:", np.array_equal(bad, np.kron(P, R)))

# %% cell 4 — broadcasting과 batch 공분산
pts = rng.normal(size=(5, 3))                          # (N, D)
ctr = rng.normal(size=(4, 3))                          # (M, D)
diff = pts[:, None, :] - ctr[None, :, :]               # (N, 1, D) - (1, M, D) → (N, M, D)
dist = np.sqrt(np.einsum("nmd,nmd->nm", diff, diff))   # (N, M)
print("dist", dist.shape)

Xb = rng.normal(size=(6, 50, 3))                       # (B, N, D)
Xc = Xb - Xb.mean(axis=1, keepdims=True)               # 샘플 축(N)으로 중심화
cov = np.einsum("bnd,bne->bde", Xc, Xc) / (50 - 1)     # (B, D, D)
ref = np.stack([np.cov(x, rowvar=False) for x in Xb])  # (B, D, D)
print("cov", cov.shape, " np.cov와 일치:", np.allclose(cov, ref))

# %% cell 5 — CP 분해: 랭크-R 텐서 만들기
I, J, Kd, Rk = 10, 20, 30, 2
a = rng.normal(size=(I, Rk))                     # (I, R)
b = rng.normal(size=(J, Rk))                     # (J, R)
c = rng.normal(size=(Kd, Rk))                    # (K, R)
Tn = np.einsum("ir,jr,kr->ijk", a, b, c)         # (I, J, K)

unfold = Tn.reshape(I, J * Kd)                   # mode-1 펼침 (I, J*K)
print("텐서 원소 수    :", Tn.size)
print("CP 파라미터 수  :", Rk * (I + J + Kd))
print("mode-1 펼침 rank:", np.linalg.matrix_rank(unfold))
