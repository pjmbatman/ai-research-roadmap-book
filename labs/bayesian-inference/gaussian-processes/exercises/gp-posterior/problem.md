# GP 회귀의 사후분포와 주변우도

1차원 입력에 RBF 커널 $k(x, x') = \sigma_f^2 \exp\!\big(-(x - x')^2 / (2\ell^2)\big)$을 쓰는 영평균 GP를 구현한다. 관측은 $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$이다. $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$로 두면

$$
\mu(x_*) = \mathbf{k}_*^\top \mathbf{K}_y^{-1}\mathbf{y}, \qquad
\sigma^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^\top \mathbf{K}_y^{-1}\mathbf{k}_*,
$$

$$
\log p(\mathbf{y} \mid \mathbf{X}) = -\tfrac12\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y} - \tfrac12\log\det\mathbf{K}_y - \tfrac{N}{2}\log 2\pi.
$$

1. `gp_posterior(x_train, y_train, x_test, ell, sf, sn)`: 시험 입력마다 잠재함수 $f$의 사후 평균과 분산 `(mean, var)`. 둘 다 shape `(M,)`. 역행렬을 직접 계산하지 말고 Cholesky 분해(`np.linalg.cholesky`)와 선형계 풀이를 쓴다.
2. `log_marginal_likelihood(x_train, y_train, ell, sf, sn)`: 위의 로그 주변우도. $\log\det\mathbf{K}_y = 2\sum_i \log L_{ii}$를 쓴다.

입력은 리스트로 올 수도 있으니 `np.asarray`로 바꾼다. 테스트는 다음을 확인한다. 잡음이 거의 없으면 학습 점을 지나고(보간), 데이터에서 멀어지면 평균은 0, 분산은 $\sigma_f^2$로 돌아간다. 주변우도는 다변량 정규분포 $\mathcal{N}(\mathbf{0}, \mathbf{K}_y)$의 로그 밀도와 같다.
