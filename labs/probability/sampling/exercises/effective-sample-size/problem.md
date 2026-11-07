# 자기 정규화 importance sampling과 유효 표본 크기

제안분포 $q$에서 뽑은 표본 $x_1, \dots, x_N$과 로그 가중치 $\log w_i = \log \tilde{p}(x_i) - \log q(x_i)$가 있다. 목표 밀도 $\tilde{p}$는 정규화 상수를 모르는 상태다. 자기 정규화(self-normalized) importance sampling 추정량과 유효 표본 크기(ESS)는

$$
\hat{I} = \frac{\sum_i w_i f(x_i)}{\sum_i w_i}, \qquad
\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}
$$

이다. 함수 `self_normalized_is(logw, fx)`를 완성해 `(estimate, ess)`를 반환한다. `logw`와 `fx`는 shape `(N,)`인 배열이다.

**주의.** 로그 가중치는 수백에서 수천이 될 수 있다. `np.exp(logw)`를 그대로 계산하면 overflow가 나서 `inf`·`nan`이 된다. 두 식 모두 가중치에 같은 상수를 곱해도 값이 변하지 않는다는 점을 이용한다.

테스트는 다음을 확인한다.

- 가중치가 모두 같으면 ESS = N이고 추정값은 `fx`의 평균이다.
- 가중치 하나가 압도적이면 ESS는 1에 가깝다.
- 로그 가중치가 1000 근처여도 결과가 유한하다.
- 목표 $\mathcal{N}(0, 1)$, 제안 $\mathcal{N}(0, 2^2)$에서 $\mathbb{E}[X^2] = 1$을 추정한다.
