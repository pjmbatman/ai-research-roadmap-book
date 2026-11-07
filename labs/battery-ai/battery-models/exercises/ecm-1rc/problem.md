# 1-RC 등가회로 모델 적분하기

1-RC 등가회로 모델(방전 전류 $I > 0$)은 다음과 같다.

$$
V_k = \mathrm{OCV}(\mathrm{SOC}_k) - I_k R_0 - V_{1,k}, \qquad
\frac{dV_1}{dt} = -\frac{V_1}{R_1 C_1} + \frac{I}{C_1}, \qquad
\frac{d\,\mathrm{SOC}}{dt} = -\frac{I}{3600\,Q}.
$$

함수 `simulate_1rc(current_a, dt_s, r0_ohm, r1_ohm, c1_f, q_ah, soc0, ocv)`를 완성해 `(voltage_v, soc)`를 반환한다.

| 인자 | 설명 |
|---|---|
| `current_a` | `(T,)` 전류 [A], 방전 + |
| `dt_s` | 샘플 간격 [s]. 전류는 한 간격 동안 일정하다(영차 유지) |
| `r0_ohm`, `r1_ohm`, `c1_f` | 직렬 저항 [Ω], RC 쌍의 저항 [Ω]과 커패시턴스 [F] |
| `q_ah` | 용량 [Ah] |
| `soc0` | 초기 SOC (0–1). 초기 $V_1 = 0$ |
| `ocv` | SOC 배열을 받아 OCV [V]를 돌려주는 함수 |

반환하는 `voltage_v[k]`와 `soc[k]`는 $k$번째 샘플 시각의 값이다(`soc[0] = soc0`). 그다음 상태는 $k$번째 전류로 $\Delta t$만큼 진행해 얻는다.

**요구 사항.** $V_1$은 전진 Euler가 아니라, 전류가 일정한 구간에서의 정확한 해

$$
V_{1,k+1} = e^{-\Delta t/\tau} V_{1,k} + R_1\big(1 - e^{-\Delta t/\tau}\big) I_k, \qquad \tau = R_1 C_1
$$

로 갱신한다. 테스트는 $\Delta t$가 $\tau$보다 훨씬 큰 경우에도 해가 안정하고 정확한지 확인한다.

힌트: 반복문 한 번에 (1) 현재 상태로 전압을 기록하고 (2) 상태를 다음 시각으로 진행한다.
