# 미분 연산자의 행렬

차수 $n$ 이하인 다항식 전체 $\mathcal{P}_n$에서 미분 $D(p) = p'$는 선형사상이다. 단항식 basis $(1, x, x^2, \dots, x^n)$에서 다항식 $p(x) = c_0 + c_1 x + \cdots + c_n x^n$의 좌표는 계수 벡터 $(c_0, c_1, \dots, c_n)$이다(상수항부터).

함수 `derivative_matrix(n)`을 완성해, 이 basis에서 $D: \mathcal{P}_n \to \mathcal{P}_n$를 나타내는 `(n + 1, n + 1)` 행렬을 반환한다. 행렬 $\mathbf{D}$는 모든 계수 벡터 `c`에 대해 `D @ c`가 $p'$의 계수 벡터(길이 `n + 1`, 마지막 성분은 0)가 되어야 한다.

테스트는 다음을 확인한다.

- shape이 `(n + 1, n + 1)`이다.
- 무작위 다항식에 대해 `numpy.polynomial.polynomial.polyder`의 결과와 같다.
- 상수 다항식은 0으로 간다(핵).
- $\mathbf{D}^{n+1} = \mathbf{0}$이다($n + 1$번 미분하면 0).

힌트: 행렬의 $j$번째 열(0부터 센다)은 $D(x^j) = j\,x^{j-1}$의 좌표다.
