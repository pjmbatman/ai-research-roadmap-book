# 중앙차분으로 Jacobian 만들기

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 Jacobian $\mathbf{J}_f(\mathbf{x})$는 (출력 차원 × 입력 차원) 행렬이고, $j$번째 열은 입력의 $j$번째 성분만 움직였을 때 출력이 변하는 비율이다. 중앙차분으로 이 열을 근사하면

$$
\mathbf{J}_f(\mathbf{x})_{:,j} \approx \frac{f(\mathbf{x} + \varepsilon \mathbf{e}_j) - f(\mathbf{x} - \varepsilon \mathbf{e}_j)}{2\varepsilon}
$$

이다. 여기서 $\mathbf{e}_j$는 $j$번째 성분만 1인 단위벡터다.

함수 `jacobian(f, x, eps=1e-6)`을 완성한다.

| 인자 | 설명 |
|---|---|
| `f` | 1차원 배열 `(n,)`을 받아 1차원 배열 `(m,)` 또는 스칼라를 돌려주는 함수 |
| `x` | 미분할 점, shape `(n,)` |
| `eps` | 차분 간격 |

반환값은 shape `(m, n)`인 NumPy 배열이다. `f`가 스칼라를 돌려주면 `m = 1`로 보고 `(1, n)`을 반환한다. 이 경우 반환값의 전치가 gradient(열벡터)다.

테스트는 선형사상(Jacobian이 행렬 자체), 극좌표 변환(해석적 Jacobian과 비교), 입력·출력 차원이 다른 함수의 shape, 스칼라 함수를 확인한다.

힌트: 열을 하나씩 계산해 `np.stack(..., axis=1)`로 붙인다. `x`를 float 배열로 바꿔 두면 정수 입력에서도 안전하다.
