# 최소제곱 계수의 표준오차

선형모형 $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$, $\boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$에서

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}, \qquad
\hat\sigma^2 = \frac{\lVert \mathbf{y} - \mathbf{X}\hat{\boldsymbol\beta} \rVert^2}{n - p}, \qquad
\widehat{\operatorname{Cov}}(\hat{\boldsymbol\beta}) = \hat\sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}
$$

이다. $n$은 관측 수, $p$는 $\mathbf{X}$의 열 수(절편 포함)다.

`ols_fit(X, y)`를 완성한다. 사전 `{"beta", "se", "sigma2"}`를 반환한다.

- `beta`: shape `(p,)` 계수
- `se`: shape `(p,)` 표준오차, 즉 공분산 행렬 대각 성분의 제곱근
- `sigma2`: 잔차 분산 추정값 $\hat\sigma^2$ ($n - p$로 나눈다)

테스트는 손으로 풀 수 있는 예, `np.linalg.lstsq`와의 일치, 그리고 반복 실험에서 보고된 표준오차가 계수의 실제 퍼짐과 맞는지를 확인한다.
