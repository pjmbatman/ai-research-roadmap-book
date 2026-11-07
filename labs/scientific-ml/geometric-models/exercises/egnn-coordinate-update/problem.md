# EGNN의 좌표 갱신 구현하기

EGNN 층의 좌표 갱신은 상대 벡터에 불변 스칼라 가중치를 곱해 더한다.

$$
\mathbf{x}_i' = \mathbf{x}_i + \frac{1}{N - 1}\sum_{j \neq i} (\mathbf{x}_i - \mathbf{x}_j)\,\phi\big(\lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2\big).
$$

여기서 $\phi$는 거리 제곱을 받아 스칼라 가중치를 내는 함수다(실제 EGNN에서는 노드 특징도 함께 받는 MLP지만, 이 문제에서는 거리 제곱만 받는다).

함수 `egnn_coords(x, phi)`를 완성해 갱신된 좌표 `x_new`(shape `(N, D)`)를 반환한다.

| 인자 | 뜻 |
|---|---|
| `x` | `(N, D)` 좌표, $N \ge 2$ |
| `phi` | 거리 제곱 배열 `(N, N)`을 받아 같은 shape의 가중치를 돌려주는 함수(원소별로 동작) |

$j = i$ 항은 상대 벡터가 0이라 결과에 영향이 없지만, $\phi$를 대각 성분에 적용해도 괜찮도록 구현한다.

테스트는 다음을 확인한다.

- 입력을 회전·반사 $\mathbf{Q}$하고 평행이동 $\mathbf{t}$하면 출력이 $\mathbf{Q}\mathbf{x}' + \mathbf{t}$가 된다.
- 노드 순서를 바꾸면 출력 순서도 똑같이 바뀐다.
- 두 점의 간단한 경우에 손으로 계산한 값과 같다.

힌트: `diff = x[:, None, :] - x[None, :, :]`는 `(N, N, D)` shape의 모든 $\mathbf{x}_i - \mathbf{x}_j$다.
