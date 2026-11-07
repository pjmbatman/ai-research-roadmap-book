# 사후분산으로 다음 실행점 고르기

비싼 시뮬레이터를 이미 `x_tr`에서 실행했다. 다음 실행점은 GP 사후분산이 가장 큰 후보로 고른다. 커널은 RBF

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

이고, 관측 잡음 분산은 $\sigma_n^2$이다. 후보 $x_*$의 잠재함수 사후분산은

$$
\sigma^2_{\text{post}}(x_*) = k(x_*, x_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

이다. $\mathbf{K}$는 학습 입력끼리의 커널 행렬, $(\mathbf{k}_*)_i = k(x_i, x_*)$다.

함수 `next_point(x_tr, x_cand, ell, sf=1.0, sn=1e-3)`를 완성해 `(x_next, var)`를 반환한다.

| 인자·반환 | shape | 뜻 |
|---|---|---|
| `x_tr` | `(N,)` | 이미 실행한 입력 |
| `x_cand` | `(M,)` | 후보 입력 |
| `var` | `(M,)` | 후보마다의 사후분산 |
| `x_next` | 스칼라 | `var`가 가장 큰 후보 |

관측값 $\mathbf{y}$는 인자에 없다. 하이퍼파라미터를 고정하면 사후분산은 $\mathbf{y}$와 무관하기 때문이다(13.7장).

힌트: 역행렬 대신 `np.linalg.cholesky`와 `np.linalg.solve`를 쓴다. $\mathbf{L}\mathbf{v} = \mathbf{k}_*$를 풀면 $\mathbf{k}_*^\top(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_* = \lVert \mathbf{v} \rVert^2$이다.
