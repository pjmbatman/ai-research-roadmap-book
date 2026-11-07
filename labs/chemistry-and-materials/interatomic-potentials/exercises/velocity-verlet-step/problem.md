# velocity Verlet 한 스텝

분자동역학의 velocity Verlet 적분기는 위치 $\mathbf{R}$, 속도 $\mathbf{V}$, 힘 $\mathbf{F}$를 시간 간격 $\Delta t$만큼 전진시킨다(질량 1). 한 스텝은 세 단계다.

$$
\mathbf{V}_{1/2} = \mathbf{V} + \tfrac{1}{2}\Delta t\, \mathbf{F}(\mathbf{R}), \qquad
\mathbf{R}' = \mathbf{R} + \Delta t\, \mathbf{V}_{1/2}, \qquad
\mathbf{V}' = \mathbf{V}_{1/2} + \tfrac{1}{2}\Delta t\, \mathbf{F}(\mathbf{R}').
$$

함수 `verlet_step(R, V, force_fn, dt)`를 완성해 갱신된 `(R_new, V_new)`를 반환한다.

| 인자 | 의미 |
|---|---|
| `R` | `(n, 3)` 현재 위치 |
| `V` | `(n, 3)` 현재 속도 |
| `force_fn` | 위치 배열을 받아 힘 `(n, 3)`을 반환하는 함수 |
| `dt` | 시간 간격 |

핵심은 **힘을 두 번** 쓰는 것이다. 반 스텝 속도 갱신에 현재 위치의 힘을, 나머지 반 스텝에 **갱신된 위치**의 힘을 쓴다.

테스트는 다음을 확인한다.

- 조화 퍼텐셜에서 한 스텝이 위 식과 정확히 일치한다.
- Lennard-Jones 계에서 여러 스텝을 돌렸을 때 총에너지가 보존된다(보존력에서 velocity Verlet의 핵심 성질).
- 힘이 0이면 등속 운동($\mathbf{R}' = \mathbf{R} + \Delta t\,\mathbf{V}$)이 된다.
