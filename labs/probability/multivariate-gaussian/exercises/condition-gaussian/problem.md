# 다변량 가우시안의 조건부분포

$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$의 성분을 관측하지 않은 것 $a$와 관측한 것 $b$로 나누면

$$
\mathbf{x}_a \mid \mathbf{x}_b \sim \mathcal{N}\big(\boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b),\ \ \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}\big)
$$

이다. 함수 `condition(mu, Sigma, obs_idx, x_obs)`를 완성한다.

| 인자 | 뜻 | shape |
|---|---|---|
| `mu` | 평균 | `(D,)` |
| `Sigma` | 공분산 (대칭, 양의 정부호) | `(D, D)` |
| `obs_idx` | 관측한 성분의 인덱스 목록 | 길이 `k` |
| `x_obs` | 관측값 (`obs_idx` 순서) | `(k,)` |

반환값은 `(cond_mean, cond_cov)`이고, 관측하지 않은 성분들을 **인덱스 오름차순**으로 놓은 조건부 평균 `(D - k,)`와 조건부 공분산 `(D - k, D - k)`다.

테스트는 다음을 확인한다.

- 2차원에서 $\mu_2 + \rho\frac{\sigma_2}{\sigma_1}(x_1 - \mu_1)$, $\sigma_2^2(1 - \rho^2)$와 같다.
- 정밀도 행렬 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$로 쓴 공식 $\boldsymbol{\Sigma}_{a \mid b} = \boldsymbol{\Lambda}_{aa}^{-1}$과 같다.
- 조건부 공분산은 관측값 `x_obs`에 의존하지 않는다.
- 관측 인덱스가 정렬되어 있지 않아도 맞는다.

힌트: $\boldsymbol{\Sigma}_{bb}^{-1}$를 직접 만들지 말고 `np.linalg.solve(Sigma_bb, ...)`를 쓴다. 인덱스 배열로 블록을 뽑을 때는 `Sigma[np.ix_(a, b)]`가 편하다.
