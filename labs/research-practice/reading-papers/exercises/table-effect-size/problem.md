# 논문 표의 수치로 차이의 구간과 효과 크기 계산하기

논문의 결과 표에 두 방법의 평균, 퍼짐, 실행 횟수가 있다. 세 함수를 완성한다.

1. `sd_from_se(se, n)`: "±"가 표준오차일 때 실행 간 표준편차 $\mathrm{SD} = \mathrm{SE}\sqrt{n}$을 반환한다.
2. `welch_ci(m1, s1, n1, m2, s2, n2, level=0.95)`: 표준편차 `s1`, `s2`를 받아 차이 $m_2 - m_1$과 그 Welch 신뢰구간 `(diff, lo, hi)`를 반환한다.
   - 표준오차 $\mathrm{se} = \sqrt{s_1^2/n_1 + s_2^2/n_2}$
   - 자유도(Welch–Satterthwaite) $\nu = \dfrac{\mathrm{se}^4}{(s_1^2/n_1)^2/(n_1-1) + (s_2^2/n_2)^2/(n_2-1)}$
   - 구간 $\text{diff} \pm t_{(1+\text{level})/2,\,\nu}\,\mathrm{se}$. $t$ 분위수는 `scipy.stats.t.ppf`로 구한다.
3. `cohens_d(m1, s1, n1, m2, s2, n2)`: 합동 표준편차 $s_p = \sqrt{\dfrac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$로 나눈 표준화 차이 $(m_2 - m_1)/s_p$를 반환한다.

테스트는 구간이 0을 포함하지 않는 것과 Welch 검정의 p-value가 $1 - \text{level}$보다 작은 것이 일치하는지, ±를 표준오차로 해석하면 구간이 넓어지는지 확인한다.
