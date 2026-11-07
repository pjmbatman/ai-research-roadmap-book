# SVD로 PCA 구현하기

데이터 행렬 $\mathbf{X} \in \mathbb{R}^{N \times D}$(행 = 샘플)의 주성분을 SVD로 구한다. 열 평균 $\boldsymbol{\mu}$를 뺀 $\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}$의 thin SVD가 $\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$이면

- 주성분 방향: $\mathbf{V}$의 앞 $k$개 열(`Vt`의 앞 $k$개 **행**)
- 각 방향의 분산: $\sigma_i^2 / (N - 1)$
- 점수(score): $\mathbf{X}_c\mathbf{V}_k$

이다. 함수 `pca(X, k)`를 완성해 딕셔너리를 반환한다.

| 키 | shape | 내용 |
|---|---|---|
| `"mean"` | `(D,)` | 열 평균 |
| `"components"` | `(k, D)` | 행 = 주성분 방향(단위 벡터), 분산이 큰 순서 |
| `"explained_variance"` | `(k,)` | 각 주성분의 분산, 내림차순 |
| `"scores"` | `(N, k)` | 중심화한 데이터의 주성분 좌표 |

`np.cov`나 `np.linalg.eig`·`eigh`는 쓰지 않는다(테스트가 비교 기준으로 쓴다).

테스트는 다음을 확인한다.

- 분산과 방향이 공분산 행렬의 고유분해와 같다(방향은 부호를 무시하고 비교).
- 평균이 원점에서 먼 데이터에서도 올바른 방향을 찾는다(중심화).
- 점수의 표본 분산이 `explained_variance`와 같고, 점수끼리 상관이 없다.
- `scores @ components + mean`이 rank-$k$ 재구성이며, $k = D$이면 원래 데이터로 돌아간다.
