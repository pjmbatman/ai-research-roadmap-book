# 최적 저랭크 근사

Eckart–Young 정리에 의해, 행렬 $\mathbf{A}$의 최적 rank-$k$ 근사는 SVD의 앞 $k$개 항만 남긴

$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top
$$

이고, 그 Frobenius 오차는 $\lVert\mathbf{A} - \mathbf{A}_k\rVert_F = \sqrt{\sum_{i > k}\sigma_i^2}$이다.

함수 `low_rank_approx(A, k)`를 완성해 `(A_k, err)`를 반환한다.

| 반환 | shape | 내용 |
|---|---|---|
| `A_k` | `A`와 같음 | 최적 rank-$k$ 근사 |
| `err` | 스칼라 | 버린 특이값으로 계산한 Frobenius 오차 $\sqrt{\sum_{i>k}\sigma_i^2}$ |

`k`가 0이면 영행렬과 $\lVert\mathbf{A}\rVert_F$를, `k`가 $\min(m, n)$ 이상이면 `A` 자신과 0을 반환한다.

테스트는 다음을 확인한다.

- `A_k`의 rank가 `k`이고, `err`가 실제 오차 `np.linalg.norm(A - A_k)`와 같다.
- 같은 rank의 다른 근사(무작위 부분공간으로의 사영)보다 오차가 작다.
- 경계 경우(`k = 0`, `k ≥ min(m, n)`)

힌트: `np.linalg.svd(A, full_matrices=False)`가 돌려주는 `Vt`의 **행**이 $\mathbf{v}_i^\top$이다. `U[:, :k] * s[:k]`처럼 브로드캐스팅을 쓰면 대각행렬을 만들지 않아도 된다.
