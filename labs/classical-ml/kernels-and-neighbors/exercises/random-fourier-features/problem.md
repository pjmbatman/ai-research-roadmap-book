# RBF 커널의 random Fourier features

RBF 커널

$$
k(\mathbf{x}, \mathbf{x}') = \exp\!\Big(-\frac{\lVert\mathbf{x} - \mathbf{x}'\rVert^2}{2\ell^2}\Big)
$$

은 $m$개의 무작위 특징으로 근사할 수 있다(Rahimi & Recht, 2007). $\boldsymbol\omega_j \sim \mathcal{N}(\mathbf{0}, \ell^{-2}\mathbf{I})$와 $b_j \sim \mathrm{U}(0, 2\pi)$를 $j = 1, \dots, m$에 대해 뽑고

$$
z_j(\mathbf{x}) = \sqrt{\tfrac{2}{m}}\cos(\boldsymbol\omega_j^\top\mathbf{x} + b_j)
$$

로 두면, $\mathbf{z}(\mathbf{x})^\top\mathbf{z}(\mathbf{x}')$의 기댓값이 정확히 $k(\mathbf{x}, \mathbf{x}')$이고, $m$이 커질수록 오차가 $O(1/\sqrt{m})$로 줄어든다.

함수 `random_fourier_features(X, m, length_scale, seed=0)`를 완성해 shape `(N, m)`의 특징 행렬 `Z`를 반환한다. `X`의 shape은 `(N, d)`다.

- 난수는 `rng = np.random.default_rng(seed)` 하나로 만든다. 같은 `seed`면 같은 특징을 내야 한다.
- 같은 `seed`로 다른 입력 집합을 변환하면 **같은** $\boldsymbol\omega, b$를 써야 한다. 학습 데이터와 시험 데이터를 같은 특징 공간에 놓기 위해서다. `ω`는 `(d, m)`, `b`는 `(m,)` 순서로 뽑는다.

테스트는 다음을 확인한다.

- shape과 재현성.
- $\mathbf{Z}\mathbf{Z}^\top$이 RBF Gram 행렬에 가깝고, 특징 수를 늘리면 오차가 줄어든다.
- 길이 척도를 바꾸면 근사하는 커널도 바뀐다.
- 대각 성분 $\mathbf{z}(\mathbf{x})^\top\mathbf{z}(\mathbf{x})$는 평균적으로 $k(\mathbf{x}, \mathbf{x}) = 1$에 가깝다.
