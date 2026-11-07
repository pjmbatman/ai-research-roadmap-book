# 기저 변환

선형사상 $T: V \to V$를 옛 basis에서 표현한 행렬이 $\mathbf{A}$이다. 새 basis 벡터들을 옛 basis의 좌표로 적어 열로 쌓은 가역행렬을 $\mathbf{S}$라 하면, 한 벡터의 좌표는 $[\mathbf{v}]_{\text{old}} = \mathbf{S}\,[\mathbf{v}]_{\text{new}}$로 이어진다.

함수 `change_basis(A, S)`를 완성해, 같은 사상 $T$를 새 basis에서 표현한 행렬을 반환한다.

| 인자 | shape | 의미 |
|---|---|---|
| `A` | `(n, n)` | 옛 basis에서의 행렬 |
| `S` | `(n, n)` | 열 = 새 basis 벡터의 옛 좌표 |

테스트는 다음을 확인한다.

- 새 좌표 $\mathbf{c}$로 준 벡터에 대해, 새 행렬로 계산한 결과를 옛 좌표로 옮긴 값이 옛 행렬로 계산한 값과 같다: $\mathbf{S}(\mathbf{A}_{\text{new}}\mathbf{c}) = \mathbf{A}(\mathbf{S}\mathbf{c})$.
- 본문의 예($x\,d/dx$를 Legendre basis로 옮기기)가 맞다.
- trace와 행렬식이 바뀌지 않는다.

힌트: 새 좌표 → 옛 좌표 → $\mathbf{A}$ 적용 → 새 좌표 순서로 생각한다. $\mathbf{S}^{-1}$을 직접 만들기보다 `np.linalg.solve`로 연립방정식을 푸는 편이 정확하다.
