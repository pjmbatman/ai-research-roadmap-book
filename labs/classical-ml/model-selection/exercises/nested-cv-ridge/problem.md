# ridge 회귀의 nested CV

ridge 회귀의 정규화 강도 $\lambda$를 교차검증으로 고르면, 고른 $\lambda$의 CV 오차는 "후보 중 가장 운 좋은 값"이라 실제 성능보다 낙관적이다. nested CV는 $\lambda$ 선택을 바깥 fold의 학습 부분 안에서만 하고, 바깥 시험 fold는 평가에만 쓴다. 세 함수를 완성한다.

1. `ridge_fit(X, y, lam)`: 절편 없는 ridge 회귀의 계수 $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$, shape `(D,)`.
2. `cv_mse(X, y, lam, k)`: $k$-fold CV의 평균 제곱오차. **fold 배정은 `np.arange(N) % k`로 고정**한다(샘플 $i$는 fold $i \bmod k$). fold마다 나머지로 학습하고 그 fold에서 MSE를 구한 뒤, fold별 MSE를 단순 평균한다.
3. `nested_cv_mse(X, y, lambdas, k_outer=5, k_inner=5)`: 반환값은 `(estimate, chosen)`.
   - 바깥 fold 배정은 `np.arange(N) % k_outer`.
   - 바깥 fold $f$마다 학습 부분 `X[outer != f]`에서 `cv_mse(..., k_inner)`로 `lambdas`를 비교해 최솟값의 $\lambda$를 고른다(동점이면 앞의 것). 학습 부분 전체로 다시 학습해 바깥 시험 fold에서 MSE를 잰다.
   - `estimate`는 바깥 fold MSE들의 평균, `chosen`은 바깥 fold마다 고른 $\lambda$의 리스트다.

데이터는 이미 중심화되어 있다고 가정하므로 절편은 쓰지 않는다. 테스트는 계산이 정의와 맞는지, 바깥 시험 fold가 $\lambda$ 선택에 쓰이지 않았는지, 그리고 신호가 없는 데이터에서 단순 CV의 최솟값이 nested 추정보다 평균적으로 낙관적인지를 확인한다.
