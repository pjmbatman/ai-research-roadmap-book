# 표준화와 벌점 없는 절편을 포함한 Ridge 회귀

함수 `fit_ridge(X, y, lam)`를 완성해, 원래 단위의 계수 `w` (shape `(p,)`)와 절편 `b` (스칼라)를 반환한다. 예측은 `X @ w + b`다.

모델은 다음 순서로 만든다.

1. 각 특징을 학습 데이터의 평균 $\mu_j$와 표준편차 $s_j$(모집단 표준편차, `ddof=0`)로 표준화한다: $z_{ij} = (x_{ij} - \mu_j)/s_j$.
2. 출력을 중심화한다: $\tilde y_i = y_i - \bar y$.
3. 표준화된 특징으로 Ridge 해를 구한다.

$$
\mathbf{v} = (\mathbf{Z}^\top\mathbf{Z} + \lambda\mathbf{I})^{-1}\mathbf{Z}^\top\tilde{\mathbf{y}}
$$

4. 원래 단위로 되돌린다: $w_j = v_j / s_j$, $b = \bar y - \sum_j \mu_j w_j$.

중심화한 덕분에 절편은 벌점을 받지 않는다. 벌점은 표준화된 계수 $\mathbf{v}$에만 걸리므로, 특징의 단위를 바꿔도 예측이 바뀌지 않는다.

테스트는 다음을 확인한다.

- `lam = 0`이면 절편이 있는 보통 최소제곱과 같다.
- 한 특징의 단위를 1000배 바꿔도 예측이 같다.
- `lam`이 매우 크면 `w`는 0에 가까워지고 `b`는 `y`의 평균에 가까워진다(절편이 벌점을 받지 않으므로).
- 표준화된 공간에서 Ridge의 최적 조건 $\mathbf{Z}^\top(\tilde{\mathbf{y}} - \mathbf{Z}\mathbf{v}) = \lambda\mathbf{v}$를 만족한다.

힌트: `np.linalg.solve`를 쓴다. 역행렬을 직접 만들 필요는 없다.
