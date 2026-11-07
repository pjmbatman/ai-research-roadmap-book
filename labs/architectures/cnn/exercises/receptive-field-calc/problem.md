# receptive field 계산하기

1차원 합성곱 층을 입력 쪽부터 차례로 쌓는다. 층 $l$은 kernel 크기 $k_l$, stride $s_l$, dilation $d_l$을 가진다. 마지막 층 출력 한 칸이 영향을 받을 수 있는 입력의 범위(이론 receptive field)의 길이 $r$과, 마지막 층 출력에서 한 칸 옆으로 갈 때 입력에서 이동하는 칸 수(jump) $J$를 구한다.

$$
r = 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l \prod_{i<l} s_i, \qquad J = \prod_{l=1}^{L} s_l.
$$

함수 `receptive_field(layers)`를 완성해 `(r, J)`를 반환한다. `layers`는 `(k, s, d)` 튜플의 리스트이고, 비어 있으면 `(1, 1)`이다.

테스트는 다음을 확인한다.

- 3×1 합성곱 다섯 층(stride 1)은 $r = 11$, $J = 1$이다.
- dilation 1, 2, 4, 8인 kernel 2 causal 층 네 개는 $r = 16$이다(시계열 TCN의 전형적인 구성).
- 무작위로 만든 층 구성에서, 입력 칸 하나를 바꿔 보며 출력 한 칸이 영향을 받는 입력 범위를 직접 센 값과 같다.

힌트: 앞 층들의 stride 곱을 누적하는 변수를 두고, 층마다 그 값에 $(k-1)d$를 곱해 $r$에 더한다. 누적 stride는 이 층의 $(k-1)d$를 더한 **뒤에** 갱신한다.
