# 쿨롱 카운팅과 오차 누적

쿨롱 카운팅은 전류를 적분해 SOC를 추적한다(방전 전류 $I > 0$).

$$
\mathrm{SOC}_{k+1} = \mathrm{SOC}_k - \frac{\eta(I_k)\, I_k\, (t_{k+1} - t_k)}{3600\, Q}, \qquad
\eta(I) = \begin{cases} 1 & I \ge 0 \text{ (방전)} \\ \eta_\text{chg} & I < 0 \text{ (충전)} \end{cases}
$$

충전 효율 $\eta_\text{chg} \le 1$은 충전 전하 중 실제로 저장되는 비율이다.

**1단계.** 함수 `soc_coulomb(time_s, current_a, q_ah, soc0, eta_chg=1.0)`를 완성해 `(T,)` SOC 배열을 반환한다. `soc[0] = soc0`이고, 측정 $k$의 전류가 다음 측정 시각까지 유지된다고 본다(영차 유지). 기록 간격은 일정하지 않을 수 있다. SOC를 0–1로 자르지 않는다(오차가 어떻게 쌓이는지 보이게 하기 위해서다).

**2단계.** 함수 `offset_drift_pct(offset_a, hours, q_ah)`를 완성한다. 전류 센서에 일정한 offset `offset_a`가 있을 때, `hours`시간 뒤 쿨롱 카운팅 SOC에 쌓이는 오차의 크기를 %p 단위로 반환한다(부호 없이).

테스트는 정전류 방전, 불균일한 기록 간격, 충전 효율, 센서 offset에 의한 오차가 시간에 비례해 쌓이는 것을 확인한다.
