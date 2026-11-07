# Levenberg–Marquardt step 구현하기

비선형 최소제곱 $\min_\theta \frac{1}{2}\lVert \mathbf{r}(\theta) \rVert^2$에서 Levenberg–Marquardt(LM) 방법의 한 step $\boldsymbol{\delta}$는

$$
(\mathbf{J}^\top\mathbf{J} + \lambda\mathbf{I})\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r}
$$

의 해다. $\mathbf{J}$는 잔차 $\mathbf{r}$의 Jacobian(shape `(N, p)`), $\lambda \ge 0$은 감쇠 계수다. 함수 `lm_step(J, r, lam)`을 완성해 shape `(p,)`의 step을 반환한다.

테스트는 다음을 확인한다.

- $\lambda = 0$이면 Gauss–Newton step이다. 잔차가 파라미터에 선형이면($\mathbf{r} = \mathbf{A}\theta - \mathbf{b}$) 한 step에 최소제곱해에 도달한다.
- $\lambda$가 아주 크면 step은 $-\mathbf{J}^\top\mathbf{r}/\lambda$, 즉 짧은 gradient descent step에 가깝다.
- $\lambda$를 키울수록 step이 짧아지고, 어떤 $\lambda > 0$에서도 선형 근사 $\mathbf{r} + \mathbf{J}\boldsymbol{\delta}$의 크기는 $\lVert \mathbf{r} \rVert$보다 작다.

힌트: $\mathbf{J}^\top\mathbf{J}$의 역행렬을 직접 만들지 말고 `np.linalg.solve`로 연립방정식을 푼다.
