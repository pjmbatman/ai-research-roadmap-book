# 해의 구조 판정

선형시스템 $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해는 없거나, 정확히 하나이거나, 무수히 많다. 함수 `solution_structure(A, b, tol=1e-10)`를 완성해 셋 중 하나를 문자열 `"none"`, `"unique"`, `"infinite"`로 반환한다.

| 인자 | shape | 의미 |
|---|---|---|
| `A` | `(m, n)` | 계수 행렬 |
| `b` | `(m,)` | 우변 |
| `tol` | 스칼라 | 수치적 rank의 허용오차. `np.linalg.matrix_rank(..., tol=tol)`에 넘긴다 |

판정 규칙은 본문의 두 사실이다.

- **존재:** $\mathbf{b} \in C(\mathbf{A})$이어야 한다. $\operatorname{rank}[\mathbf{A} \mid \mathbf{b}] = \operatorname{rank}\mathbf{A}$와 같다.
- **유일성:** 해가 있을 때, null space가 $\{\mathbf{0}\}$이면(rank–nullity로 $r = n$) 유일하다.

테스트는 정사각·과결정·미결정 시스템, rank가 부족한 시스템, 잡음 수준에서만 독립인 열이 있는 시스템을 확인한다.

힌트: 행렬 `[A | b]`는 `np.column_stack([A, b])`로 만든다.
