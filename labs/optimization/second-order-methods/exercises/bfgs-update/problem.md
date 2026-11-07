# BFGS 역Hessian 갱신 구현하기

BFGS는 역Hessian의 근사 $\mathbf{H}_k$를 step마다 갱신한다. 파라미터 변화 $\mathbf{s} = \mathbf{x}_{k+1} - \mathbf{x}_k$와 gradient 변화 $\mathbf{y} = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$로

$$
\mathbf{H}_{k+1} = (\mathbf{I} - \rho\,\mathbf{s}\mathbf{y}^\top)\,\mathbf{H}_k\,(\mathbf{I} - \rho\,\mathbf{y}\mathbf{s}^\top) + \rho\,\mathbf{s}\mathbf{s}^\top, \qquad \rho = \frac{1}{\mathbf{y}^\top\mathbf{s}}
$$

이다. 함수 `bfgs_update(Hinv, s, y)`를 완성해 새 근사 `(n, n)`을 반환한다. 입력 `Hinv`를 제자리에서 바꾸지 않는다.

테스트는 다음을 확인한다.

- **secant 조건** $\mathbf{H}_{k+1}\mathbf{y} = \mathbf{s}$를 만족한다.
- $\mathbf{y}^\top\mathbf{s} > 0$이면 대칭·양의 정부호가 유지된다.
- 이차함수 $\frac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x} - \mathbf{b}^\top\mathbf{x}$에서 exact line search와 함께 $n$번 갱신하면 최솟점에 도달하고 $\mathbf{H}_n = \mathbf{A}^{-1}$이 된다.

힌트: `np.outer(s, y)`는 $\mathbf{s}\mathbf{y}^\top$이다.
