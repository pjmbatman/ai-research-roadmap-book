# 두 아핀 층을 하나로 접기

활성화 함수 없이 쌓은 두 층

$$
g(\mathbf{x}) = \mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

는 하나의 아핀 사상 $\mathbf{W}\mathbf{x} + \mathbf{b}$와 같다. 함수 `fold(W1, b1, W2, b2)`를 완성해 `(W, b)`를 반환한다.

| 인자 | shape |
|---|---|
| `W1` | `(h, d)` |
| `b1` | `(h,)` |
| `W2` | `(m, h)` |
| `b2` | `(m,)` |

반환하는 `W`의 shape은 `(m, d)`, `b`의 shape은 `(m,)`이다. 테스트는 무작위 입력에서 `g(x)`와 `W @ x + b`가 같은지 확인한다.

힌트: 먼저 $g(\mathbf{x})$를 전개해 $\mathbf{x}$가 곱해진 항과 나머지 항을 모은다.
