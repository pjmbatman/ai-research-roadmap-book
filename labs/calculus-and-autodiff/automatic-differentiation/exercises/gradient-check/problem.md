# 상대오차로 gradient 검증하기

해석적으로 구한(또는 직접 만든 자동미분이 계산한) gradient가 맞는지 중앙차분으로 확인하는 함수 `grad_check(f, grad, x, eps=1e-6)`를 완성한다.

| 인자 | 설명 |
|---|---|
| `f` | 1차원 배열 `(n,)`을 받아 스칼라를 돌려주는 함수 |
| `grad` | 검사할 gradient, shape `(n,)` |
| `x` | 검사할 점, shape `(n,)` |
| `eps` | 중앙차분 간격 |

반환값은 성분별 상대오차의 최댓값(스칼라)이다. 성분 $i$의 상대오차는

$$
r_i = \frac{\lvert g_i - \hat{g}_i \rvert}{\max\big(\lvert g_i \rvert + \lvert \hat{g}_i \rvert,\ 10^{-12}\big)}
$$

이고, $\hat{g}_i = \big(f(\mathbf{x} + \varepsilon\mathbf{e}_i) - f(\mathbf{x} - \varepsilon\mathbf{e}_i)\big)/(2\varepsilon)$는 중앙차분이다. 분모의 $10^{-12}$는 두 값이 모두 0인 성분에서 0으로 나누지 않기 위한 것이다.

요구 사항:

- `x`가 float32 배열이나 정수 배열로 주어져도 **float64로 바꿔** 계산한다. float32로 차분하면 반올림 오차 때문에 올바른 gradient도 큰 상대오차를 보인다(본문 참고).
- 입력 `x`를 제자리에서 바꾸지 않는다.

테스트는 올바른 gradient에서 상대오차가 $10^{-7}$ 미만인지, 상수배가 틀린 gradient와 한 성분의 부호가 틀린 gradient에서 $10^{-2}$보다 큰지, gradient가 0인 성분에서 `nan`이 나오지 않는지, float32 입력을 float64로 처리하는지 확인한다.

힌트: `x = np.array(x, dtype=np.float64)`는 복사본을 만든다.
