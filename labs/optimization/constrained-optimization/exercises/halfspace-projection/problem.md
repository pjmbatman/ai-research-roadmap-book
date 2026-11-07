# 반공간으로의 투영과 KKT 승수

점 $\mathbf{p} \in \mathbb{R}^n$에서 가장 가까운 반공간 $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le c\}$의 점을 찾는 문제

$$
\min_{\mathbf{x}} \ \lVert \mathbf{x} - \mathbf{p} \rVert^2 \quad \text{subject to} \quad \mathbf{a}^\top\mathbf{x} - c \le 0
$$

를 KKT 조건으로 푼다. 함수 `project_halfspace(p, a, c)`를 완성해 최적해 `x`(shape `(n,)`)와 승수 `mu`(스칼라)를 반환한다.

KKT 조건은 다음 네 가지다.

1. stationarity: $2(\mathbf{x} - \mathbf{p}) + \mu\,\mathbf{a} = \mathbf{0}$
2. primal feasibility: $\mathbf{a}^\top\mathbf{x} \le c$
3. dual feasibility: $\mu \ge 0$
4. 상보적 여유: $\mu\,(\mathbf{a}^\top\mathbf{x} - c) = 0$

테스트는 반환값이 네 조건을 모두 만족하는지, 그리고 반공간 안의 무작위 점들보다 $\mathbf{p}$에 더 가까운지 확인한다.

힌트: 두 경우로 나눈다. $\mathbf{p}$가 이미 반공간 안에 있으면 제약이 비활성이다. 그렇지 않으면 제약이 활성($\mathbf{a}^\top\mathbf{x} = c$)이고, stationarity 식에서 $\mathbf{x}$를 $\mu$로 나타낸 뒤 활성 조건에 넣어 $\mu$를 구한다.
