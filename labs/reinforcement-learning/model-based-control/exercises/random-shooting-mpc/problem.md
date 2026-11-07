# 랜덤 슈팅 MPC

랜덤 슈팅은 가장 단순한 샘플링 기반 MPC다. 매 step 다음을 한다.

1. 입력 한계 $[\mathbf{u}_{\text{low}}, \mathbf{u}_{\text{high}}]$ 안에서 균등분포로 길이 $H$의 입력열 $N$개를 뽑는다.
2. 각 입력열로 현재 상태에서 모형을 $H$ step 굴리며, 각 step의 비용 `stage_cost(다음 상태, 입력)`을 더한다.
3. 총비용이 가장 작은 입력열의 **첫 입력**을 반환한다.

함수 `random_shooting(x, model, stage_cost, H, N, u_low, u_high, rng)`를 완성한다.

| 인자 | 뜻 |
|---|---|
| `x` | shape `(n,)`의 현재 상태 |
| `model` | `model(X, U)`: shape `(N, n)`, `(N, m)` → `(N, n)`의 다음 상태 (후보 N개를 한꺼번에) |
| `stage_cost` | `stage_cost(X_next, U)`: shape `(N, n)`, `(N, m)` → `(N,)` |
| `H`, `N` | 지평, 후보 수 |
| `u_low`, `u_high` | shape `(m,)`의 입력 한계 |
| `rng` | `np.random.Generator` (표집에 이것만 쓴다) |

반환값은 shape `(m,)`의 첫 입력이다. 테스트는 한계 준수, 1 step 문제의 최적 입력, 한계가 활성화되는 경우, 지연이 있어 지평이 필요한 시스템, 첫 입력만 적용하고 다시 계획하는 폐루프를 확인한다.

힌트: 후보 입력열을 한꺼번에 `rng.uniform(u_low, u_high, size=(N, H, m))`으로 뽑고, 상태는 `np.tile(x, (N, 1))`로 시작한다.
