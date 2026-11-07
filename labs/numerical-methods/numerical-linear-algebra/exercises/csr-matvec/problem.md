# CSR 형식의 행렬–벡터 곱

CSR(compressed sparse row) 형식은 $m \times n$ 희소 행렬을 세 배열로 저장한다.

- `data`: 0이 아닌 값들을 행 순서대로 이어 붙인 배열 (길이 nnz)
- `indices`: 각 값의 열 번호 (길이 nnz)
- `indptr`: 행 $i$의 값이 `data[indptr[i]:indptr[i+1]]`에 있음을 알려 주는 포인터 (길이 $m+1$)

예를 들어

$$
\mathbf{A} = \begin{bmatrix} 4 & 0 & 1 \\ 0 & 0 & 0 \\ 2 & 3 & 0 \end{bmatrix}
$$

은 `data = [4, 1, 2, 3]`, `indices = [0, 2, 0, 1]`, `indptr = [0, 2, 2, 4]`다. 둘째 행은 비어 있어 `indptr[1] == indptr[2]`이다.

함수 `csr_matvec(data, indices, indptr, x)`를 완성해 $\mathbf{y} = \mathbf{A}\mathbf{x}$를 반환한다.

- 행의 수는 `len(indptr) - 1`이다. 반환 shape은 `(m,)`이다.
- `scipy.sparse`나 dense 행렬을 만들지 않고 세 배열만으로 계산한다.
- 빈 행의 결과는 0이다.

힌트: 행 $i$에 대해 `data[indptr[i]:indptr[i+1]]`와 `x[indices[indptr[i]:indptr[i+1]]]`의 내적을 구한다. 계산량은 nnz에 비례하고 $m \times n$과는 무관하다.
