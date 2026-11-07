# 가우시안 근사의 Fréchet 거리(FID의 핵심 계산)

FID는 실제 특징과 생성 특징에 각각 가우시안 $\mathcal{N}(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$, $\mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$를 맞춘 뒤, 두 가우시안의 Fréchet 거리를 계산한다.

$$
d^2 = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert^2 + \operatorname{tr}\big(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r\boldsymbol{\Sigma}_g)^{1/2}\big).
$$

함수 두 개를 완성한다.

1. `frechet_from_stats(mu1, S1, mu2, S2)`: 평균 벡터(shape `(d,)`)와 공분산 행렬(shape `(d, d)`)로 $d^2$를 반환한다. 행렬 제곱근은 `scipy.linalg.sqrtm`을 쓰고, 수치 오차로 생기는 아주 작은 허수부는 버린다(`.real`).
2. `frechet_from_features(a, b)`: 특징 행렬 `a`(shape `(N, d)`), `b`(shape `(M, d)`)에서 표본 평균과 표본 공분산(`np.cov(..., rowvar=False)`)을 구해 1을 호출한다.

테스트는 다음을 확인한다.

- 같은 가우시안이면 0이다.
- 공분산이 같고 평균만 다르면 평균 차이의 제곱 노름이다.
- 1차원에서는 $(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$이다.
- 두 인자의 순서를 바꿔도 값이 같다.
- 평균과 공분산이 같고 모양만 다른 두 표본 집합(가우시안과 두 봉우리 분포)에는 거의 0을 준다. FID가 가우시안 밖의 구조를 보지 못한다는 뜻이다.
