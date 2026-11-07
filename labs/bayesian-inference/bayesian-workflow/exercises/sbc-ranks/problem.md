# 시뮬레이션 기반 보정(SBC)의 순위 통계량

사후분포를 계산하는 코드가 맞는지 확인하는 일반적인 방법이 시뮬레이션 기반 보정(simulation-based calibration, SBC)이다. 한 번의 시뮬레이션은 다음과 같다.

1. 사전분포에서 참값 $\tilde\theta \sim p(\theta)$를 뽑는다.
2. 그 참값으로 데이터 $\tilde y \sim p(y \mid \tilde\theta)$를 시뮬레이션한다.
3. 검사할 추론 코드로 사후 표본 $\theta^{(1)}, \dots, \theta^{(L)}$을 얻는다.
4. 참값의 순위 $r = \#\{\ell : \theta^{(\ell)} < \tilde\theta\}$를 기록한다. $r$은 $0, 1, \dots, L$ 중 하나다.

추론이 정확하면 $r$은 $\{0, \dots, L\}$ 위에서 균일분포를 따른다.

두 함수를 완성한다.

1. `sbc_ranks(prior_sample, simulate, posterior_sample, n_sims, L, rng)`: 위 과정을 `n_sims`번 반복해 순위 배열(정수, shape `(n_sims,)`)을 반환한다. 세 함수 인자의 호출 방식은 다음과 같다.
   - `prior_sample(rng)` → 참값 하나
   - `simulate(theta, rng)` → 데이터
   - `posterior_sample(data, L, rng)` → 사후 표본 배열 shape `(L,)`
2. `rank_counts(ranks, L, n_bins)`: 순위를 $0, \dots, L$ 범위의 같은 너비 구간 `n_bins`개로 나눈 개수. $(L + 1)$이 `n_bins`로 나누어떨어진다고 가정한다(예: $L = 99$, 10칸). 합은 `len(ranks)`다.

테스트는 Gamma–Poisson 켤레 모델로 확인한다. 올바른 사후분포에서는 순위가 균일하고, 분산을 줄인(과신한) 사후분포에서는 양 끝 칸에 순위가 몰리며, 평균을 옮긴(편향된) 사후분포에서는 한쪽으로 몰린다.
