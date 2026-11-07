# RBF kernel ridge 회귀

1차원 입력 $x_1, \dots, x_N$과 관측 $y_1, \dots, y_N$이 있다. RBF 커널

$$
k(x, x') = \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

이 정하는 RKHS $\mathcal{H}$에서 정규화된 최소제곱 문제

$$
\min_{f \in \mathcal{H}} \sum_{i=1}^{N} \big(y_i - f(x_i)\big)^2 + \lambda \lVert f \rVert_{\mathcal{H}}^2
$$

를 푼다. 표현자 정리에 따라 해는 $f(x) = \sum_{i=1}^N \alpha_i k(x, x_i)$ 꼴이고, 계수는

$$
(\mathbf{K} + \lambda \mathbf{I})\boldsymbol{\alpha} = \mathbf{y}, \qquad K_{ij} = k(x_i, x_j)
$$

를 풀어 얻는다. 함수 `kernel_ridge_predict(x_train, y_train, x_test, ell, lam)`를 완성해 `x_test`의 각 점에서 $f$의 값을 반환한다.

| 인자 | shape | 의미 |
|---|---|---|
| `x_train` | `(N,)` | 학습 입력 |
| `y_train` | `(N,)` | 학습 관측 |
| `x_test` | `(M,)` | 예측할 입력 |
| `ell` | 스칼라 | 길이 척도 $\ell > 0$ |
| `lam` | 스칼라 | 정규화 강도 $\lambda \ge 0$ |

반환값의 shape은 `(M,)`이다.

테스트는 다음을 확인한다.

- 위 닫힌 형태의 해와 같다.
- $\lambda$가 아주 작으면 학습 점을 (거의) 지나간다.
- 학습 점에서 멀리 떨어진 곳의 예측은 0에 가깝다. 해가 학습 점을 중심으로 한 커널 함수의 합이기 때문이다.
- 예측은 $\mathbf{y}$에 대해 선형이다.

힌트: 역행렬 `np.linalg.inv`보다 `np.linalg.solve`가 더 정확하고 빠르다. 쌍별 차이 $(x_i - x'_j)$는 broadcasting으로 한 번에 만든다(7.7장).
