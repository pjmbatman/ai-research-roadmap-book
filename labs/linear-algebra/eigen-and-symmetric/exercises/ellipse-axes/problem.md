# 이차형식의 타원 축

대칭 양의 정부호 행렬 $\mathbf{S}$에 대해 $\mathbf{x}^\top\mathbf{S}\mathbf{x} = 1$인 점들의 집합은 타원(고차원에서는 타원체)이다. 스펙트럴 정리 $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$로 좌표를 바꾸면 $\sum_i \lambda_i y_i^2 = 1$이 되므로, 타원의 축은 고유벡터 $\mathbf{q}_i$ 방향이고 반축의 길이는 $1/\sqrt{\lambda_i}$이다.

함수 `ellipse_axes(S)`를 완성해 `(lengths, directions)`를 반환한다.

| 반환 | shape | 내용 |
|---|---|---|
| `lengths` | `(n,)` | 반축의 길이, **긴 것부터** 내림차순 |
| `directions` | `(n, n)` | $i$번째 **열**이 `lengths[i]`에 대응하는 단위 방향 벡터 |

$\mathbf{S}$가 대칭이 아니거나(허용오차 `1e-10`) 양의 정부호가 아니면 `ValueError`를 발생시킨다.

테스트는 다음을 확인한다.

- 대각행렬 $\operatorname{diag}(4, 0.25)$에서 반축 $(2, 0.5)$와 방향 $(0, \pm 1)$, $(\pm 1, 0)$
- 반축 끝점 $\ell_i\mathbf{d}_i$가 타원 위에 있다: $(\ell_i\mathbf{d}_i)^\top\mathbf{S}(\ell_i\mathbf{d}_i) = 1$
- 방향 벡터들이 정규직교다
- 고윳값에 0이나 음수가 있으면, 또는 비대칭이면 `ValueError`

힌트: `np.linalg.eigh`는 고윳값을 **오름차순**으로 준다. 긴 반축은 작은 고윳값에 대응한다는 점을 떠올리면 정렬을 따로 하지 않아도 된다.
