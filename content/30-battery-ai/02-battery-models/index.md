---
id: battery-ai/battery-models
title: "배터리 모델의 계층"
title_en: "The Hierarchy of Battery Models"
tier: core
status: draft
prerequisites: [battery-ai/battery-fundamentals, numerical-methods/pde-and-dae]
---

# 배터리 모델의 계층

배터리 연구에서 "모델"은 하나가 아니다. 저항 몇 개와 커패시터로 단자 전압을 흉내 내는 등가회로 모델부터, 전극 두께 방향과 입자 반경 방향의 리튬 농도를 편미분방정식으로 푸는 전기화학 모델까지 계층이 있다. 상태 수는 몇 개에서 수백 개 이상으로 늘고, 파라미터 수와 계산 비용도 함께 는다. 어느 수준이 "옳은" 모델인지는 정해져 있지 않다. 연구 질문이 요구하는 내부 변수와 운전 범위가 수준을 정한다.

ML 연구자에게 이 계층이 중요한 이유는 세 가지다. 첫째, 물리 모델은 학습 데이터를 만드는 시뮬레이터이자(15.6장) 하이브리드 모델의 뼈대다(24.5장). 둘째, 상태 추정(30.4장)과 파라미터 추정(30.5장)은 모두 이 모델들 위에서 정의된다. 셋째, 모델이 맞는 범위를 벗어난 시뮬레이션 데이터로 학습한 ML 모델은 그 오류를 그대로 물려받는다. 이 장의 OCV 곡선과 파라미터는 모두 가상의 예시 셀의 값이다.

## 학습 목표

- ECM에서 DFN까지 각 모델 수준의 상태·가정·계산 비용을 비교한다
- 1-RC 등가회로 모델을 적분하고 펄스 응답에서 파라미터를 읽는다
- 연구 목적과 운전 범위에 맞는 모델 수준을 고르고 시뮬레이션 설정을 점검한다

## 등가회로 모델(ECM)

**왜 필요한가.** 배터리 관리 시스템(BMS)은 작은 마이크로컨트롤러에서 1초에 여러 번 SOC와 출력 한계를 계산해야 한다. 전극 내부를 푸는 모델은 너무 무겁다. 단자에서 보이는 전압 응답만 재현하는 가벼운 모델이 필요하다.

**정의.** 등가회로 모델(ECM, equivalent circuit model)은 셀을 전압원 하나와 수동 소자의 회로로 표현한다. 가장 흔한 1-RC 모델은 SOC에 의존하는 전압원 $\mathrm{OCV}(\mathrm{SOC})$, 직렬 저항 $R_0$, 그리고 저항 $R_1$과 커패시터 $C_1$을 병렬로 묶은 RC 쌍 하나로 이루어진다. 방전 전류를 양수로 쓰면

$$
V = \mathrm{OCV}(\mathrm{SOC}) - I R_0 - V_1, \qquad
\frac{dV_1}{dt} = -\frac{V_1}{R_1 C_1} + \frac{I}{C_1}, \qquad
\frac{d\,\mathrm{SOC}}{dt} = -\frac{I}{3600\,Q}.
$$

$V$는 단자 전압, $V_1$은 RC 쌍에 걸린 전압(분극), $Q$는 용량[Ah]이다. 상태는 $(\mathrm{SOC}, V_1)$ 두 개다. RC 쌍을 더 붙이면(2-RC, 3-RC) 서로 다른 시간 규모의 분극을 따로 표현한다.

**직관.** 전류를 계단처럼 바꾸면 $R_0$ 때문에 전압이 즉시 $\Delta I \cdot R_0$만큼 뛰고, 그 뒤 $V_1$이 시상수 $\tau = R_1 C_1$로 $I R_1$을 향해 지수적으로 따라간다. 전류를 끊으면 반대로 즉시 $I R_0$만큼 회복하고, $V_1$이 같은 시상수로 0을 향해 풀린다. 회로 소자가 셀 안의 특정 부품에 일대일로 대응하지는 않는다. $R_0$은 전해질·집전체·접촉 저항을, RC 쌍은 전하 전달과 이중층, 짧은 시간의 확산을 뭉뚱그려 나타낸다.

**수식에서 코드로.** 전류가 샘플 간격 $\Delta t$ 동안 일정하다고 보면(영차 유지, zero-order hold), $V_1$의 선형 ODE는 정확히 풀린다.

$$
V_{1,k+1} = e^{-\Delta t/\tau}\, V_{1,k} + R_1\big(1 - e^{-\Delta t/\tau}\big)\, I_k.
$$

아래 코드는 이 식을 RC 쌍 여러 개에 대해 한꺼번에 쓰도록 짰다(cell 5에서 2-RC를 쓴다). $R_0 = 20$ mΩ, $R_1 = 15$ mΩ, $C_1 = 2$ kF($\tau = 30$ s), 용량 3 Ah, 초기 SOC 80%인 가상의 셀에 1C(3 A) 방전 펄스를 360 s 건다.

```python {exec="browser" session="ecm"}
import numpy as np

Q_AH, SOC0 = 3.0, 0.8                              # 가상의 예시 셀 (위젯과 같은 값)
R0_OHM, R1_OHM, C1_F = 0.020, 0.015, 2000.0        # tau = R1 C1 = 30 s

def ocv_v(soc):                                    # 가상의 예시 셀의 OCV-SOC 곡선 [V]
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

def simulate(i_a, dt_s, r0, rc, soc0=SOC0, euler=False):
    """i_a: (T,) 방전 + 전류. rc: [(R, C), ...] RC 쌍 목록. 반환: (T,) 단자 전압 [V]"""
    R, C = np.array(rc).T                          # (P,), (P,)
    a = np.exp(-dt_s / (R * C))                    # 영차 유지에서의 정확한 감쇠율
    v_rc, soc, v = np.zeros(len(R)), soc0, np.empty(len(i_a))
    for k, i in enumerate(i_a):
        v[k] = ocv_v(soc) - i * r0 - v_rc.sum()
        if euler:                                  # 전진 Euler: dV/dt = -V/(RC) + I/C
            v_rc = v_rc + dt_s * (-v_rc / (R * C) + i / C)
        else:
            v_rc = a * v_rc + R * (1 - a) * i
        soc -= i * dt_s / 3600 / Q_AH
    return v

t_s = np.arange(0, 901.0)                          # (T,) 1 s 간격
T_ON, T_OFF = 90, 450                              # 1C 방전 펄스 360 s
i_a = np.where((t_s >= T_ON) & (t_s < T_OFF), 3.0, 0.0)  # (T,)
v = simulate(i_a, 1.0, R0_OHM, [(R1_OHM, C1_F)])
for k in [89, 90, 120, 449, 450, 480, 900]:
    print(f"t = {k:3d} s  I = {i_a[k]:.1f} A  V = {v[k]:.4f} V")
```

```output
t =  89 s  I = 0.0 A  V = 3.9926 V
t =  90 s  I = 3.0 A  V = 3.9326 V
t = 120 s  I = 3.0 A  V = 3.8977 V
t = 449 s  I = 3.0 A  V = 3.8126 V
t = 450 s  I = 0.0 A  V = 3.8724 V
t = 480 s  I = 0.0 A  V = 3.9009 V
t = 900 s  I = 0.0 A  V = 3.9174 V
```

펄스가 시작되는 순간($t$ = 89 → 90 s) 전압은 정확히 60 mV($= 3\ \mathrm{A} \times 20\ \mathrm{m\Omega}$) 떨어지고, 30 s 뒤에는 RC 몫이 더해져 더 내려간다. 펄스가 끝나는 순간에는 60 mV 회복하고, 이후 휴지 동안 천천히 더 오른다. 휴지 끝(900 s)의 전압 3.9174 V가 처음(3.9926 V)보다 낮은 것은 SOC가 10%p 줄어 OCV가 내려갔기 때문이다. 같은 모델을 아래 위젯에서 직접 움직여 본다. $R_0$를 0으로 두면 즉시 강하가 사라지고, $C_1$을 키우면 같은 $R_1$에서 느린 응답이 더 느려진다.

::widget{name="battery-ecm"}

**이산화가 중요하다.** 9.6장에서 본 대로, 전진 Euler로 $dV_1/dt$를 적분하면 증폭률이 $1 - \Delta t/\tau$이므로 $\Delta t > 2\tau$에서 발산한다. 정확한 이산화의 증폭률 $e^{-\Delta t/\tau}$는 어떤 $\Delta t$에서도 0과 1 사이다. 1 s 간격 해를 기준으로 두 방법을 비교한다.

```python {exec="browser" session="ecm"}
ref = simulate(i_a, 1.0, R0_OHM, [(R1_OHM, C1_F)])                 # (901,) 기준해
for dt in [5.0, 30.0, 45.0, 90.0]:
    idx = np.arange(0, 901, int(dt))                              # dt 간격의 시각
    i_dt = i_a[idx]                                               # 펄스 경계가 dt의 배수
    for name, eu in [("정확", False), ("Euler", True)]:
        err = np.max(np.abs(simulate(i_dt, dt, R0_OHM, [(R1_OHM, C1_F)], euler=eu) - ref[idx]))
        print(f"dt = {dt:2.0f} s  dt/tau = {dt / 30:4.2f}  {name:5s} 최대 오차 {err * 1e3:9.3f} mV")
```

```output
dt =  5 s  dt/tau = 0.17  정확    최대 오차     0.000 mV
dt =  5 s  dt/tau = 0.17  Euler 최대 오차     1.484 mV
dt = 30 s  dt/tau = 1.00  정확    최대 오차     0.000 mV
dt = 30 s  dt/tau = 1.00  Euler 최대 오차    16.555 mV
dt = 45 s  dt/tau = 1.50  정확    최대 오차     0.000 mV
dt = 45 s  dt/tau = 1.50  Euler 최대 오차    32.541 mV
dt = 90 s  dt/tau = 3.00  정확    최대 오차     0.000 mV
dt = 90 s  dt/tau = 3.00  Euler 최대 오차 21600.000 mV
```

영차 유지의 정확한 해는 전류가 실제로 간격 안에서 일정하면 step 크기와 무관하게 오차가 없다. 전진 Euler는 $\Delta t = \tau$에서 이미 17 mV 가까이 틀리고, $\Delta t = 3\tau$에서는 발산한다. BMS의 RC 시상수는 짧게는 수 초이고, 저온에서 작아진 $C$나 여러 RC 쌍 중 가장 빠른 쌍이 stiffness를 만든다. 전류가 간격 안에서 변한다면(실제 주행 데이터) 영차 유지 자체가 근사이므로, 샘플 간격을 전류 변화의 시간 규모보다 짧게 잡는다.

::quiz{id="euler-step-instability"}

**파라미터 읽기.** HPPC 같은 펄스 시험(30.1장)에서 1-RC 파라미터는 응답의 모양에서 바로 읽힌다. 펄스 시작의 즉시 강하가 $R_0$, 휴지 구간의 지수 회복이 $\tau$와 $R_1$을 준다. 휴지가 시작될 때 $V_1 = I R_1 (1 - e^{-T_\text{pulse}/\tau})$이고 그 뒤 $V(t) = V_\infty - V_1 e^{-t/\tau}$로 회복하므로, 휴지 구간에 지수 함수를 맞추면 된다. 1 mV 측정 잡음을 넣고 해 본다.

```python {exec="browser" session="ecm"}
from scipy.optimize import curve_fit

rng = np.random.default_rng(0)
v_meas = v + rng.normal(0, 0.001, v.shape)                       # 전압 측정 잡음 1 mV
dv_on = v_meas[T_ON - 1] - v_meas[T_ON]                          # 펄스 시작의 즉시 강하
r0_hat = dv_on / 3.0
tr, vr = t_s[T_OFF + 1:] - T_OFF, v_meas[T_OFF + 1:]             # 휴지 구간 (펄스 직후 제외)
def relax(t, v_inf, a, tau):                                     # V(t) = V_inf - a e^{-t/tau}
    return v_inf - a * np.exp(-t / tau)
(v_inf, a_hat, tau_hat), _ = curve_fit(relax, tr, vr, p0=[vr[-1], 0.05, 10.0])
r1_hat = a_hat / (3.0 * (1 - np.exp(-(T_OFF - T_ON) / tau_hat)))  # 휴지 시작 시 V1 = I R1 (1 - e^{-360/tau})
print(f"R0: 참 {R0_OHM * 1e3:.1f} mOhm, 추정 {r0_hat * 1e3:.1f} mOhm")
print(f"tau: 참 {R1_OHM * C1_F:.1f} s, 추정 {tau_hat:.1f} s")
print(f"R1: 참 {R1_OHM * 1e3:.1f} mOhm, 추정 {r1_hat * 1e3:.1f} mOhm, C1 추정 {tau_hat / r1_hat:.0f} F")
```

```output
R0: 참 20.0 mOhm, 추정 20.7 mOhm
tau: 참 30.0 s, 추정 30.0 s
R1: 참 15.0 mOhm, 추정 15.1 mOhm, C1 추정 1988 F
```

세 파라미터가 모두 잘 돌아온다. $R_0$은 두 샘플의 차이 하나로 정하므로 잡음에 가장 민감하다(잡음 1 mV가 차이에서 약 1.4 mV가 되고, 3 A로 나누면 약 0.5 mΩ). 실제 시험에서는 즉시 강하를 몇 ms 만에 재느냐(샘플링 주기)에 따라 $R_0$에 빠른 RC 몫이 섞이므로, $R_0$의 값에는 측정 시간 창을 함께 적는다. 여러 SOC·온도·전류에서 이 추정을 반복해 파라미터를 표로 만드는 것이 ECM 파라미터화의 기본이고, 전체 프로파일에 최적화로 맞추는 방법과 식별 가능성 문제는 30.5장과 24.6장에서 다룬다.

**OCV-SOC 곡선.** ECM의 정확도는 대부분 OCV 곡선이 정한다. OCV는 충분히 쉰 셀의 전압으로, 낮은 C-rate(예: C/20) 충방전 곡선의 평균이나 GITT 휴지 끝의 전압으로 얻는다. 충전 방향과 방전 방향의 OCV가 다른 **히스테리시스**가 있는 화학(특히 LFP)에서는 곡선 하나로 부족하다. OCV 곡선의 기울기는 전압이 SOC에 대해 얼마나 정보를 주는지를 정한다. 같은 5 mV 전압 오차가 SOC로 얼마의 오차가 되는지, 이 장의 예시 곡선과 평탄 구간이 넓은 가상의 곡선(LFP와 비슷한 모양)에서 비교한다.

```python {exec="browser" session="ecm"}
def ocv_flat_v(soc):                     # 평탄 구간이 넓은 가상의 곡선 (LFP와 비슷한 모양)
    return 3.3 + 0.03 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

for soc in [0.1, 0.3, 0.5, 0.7, 0.9]:
    h = 1e-4
    s1 = (ocv_v(soc + h) - ocv_v(soc - h)) / (2 * h) / 100       # [V / %SOC]
    s2 = (ocv_flat_v(soc + h) - ocv_flat_v(soc - h)) / (2 * h) / 100
    print(f"SOC {soc:.0%}: 기울기 {s1 * 1e3:5.2f} / {s2 * 1e3:5.2f} mV/%, "
          f"전압 오차 5 mV → SOC 오차 {0.005 / s1:4.1f} / {0.005 / s2:4.1f} %p")
```

```output
SOC 10%: 기울기  9.27 /  4.07 mV/%, 전압 오차 5 mV → SOC 오차  0.5 /  1.2 %p
SOC 30%: 기울기  7.31 /  2.11 mV/%, 전압 오차 5 mV → SOC 오차  0.7 /  2.4 %p
SOC 50%: 기울기  7.04 /  1.84 mV/%, 전압 오차 5 mV → SOC 오차  0.7 /  2.7 %p
SOC 70%: 기울기  7.31 /  2.11 mV/%, 전압 오차 5 mV → SOC 오차  0.7 /  2.4 %p
SOC 90%: 기울기  9.27 /  4.07 mV/%, 전압 오차 5 mV → SOC 오차  0.5 /  1.2 %p
```

기울기가 7 mV/% 근처인 곡선에서 5 mV는 0.7 %p지만, 중간 SOC에서 기울기가 2 mV/% 아래로 떨어지는 평탄한 곡선에서는 2.4–2.7 %p가 된다. 24.6장의 관측 가능성 관점에서, 평탄 구간에서는 전압이 SOC에 대해 거의 정보를 주지 않는다. 이 사실은 모델을 아무리 정교하게 해도 바뀌지 않는다.

::quiz{id="ocv-slope-soc-error"}

## 전기화학 모델

**왜 필요한가.** ECM은 단자 전압을 재현할 뿐, 전극 안에서 무슨 일이 일어나는지는 말하지 않는다. 고속 충전 중 음극 표면이 리튬 석출 조건에 닿는지, 두꺼운 전극의 어느 깊이가 덜 쓰이는지 같은 질문에는 내부 농도와 전위가 필요하다. ECM은 또 맞춘 시간 규모 밖에서 틀린다. 참 셀이 시상수 30 s와 600 s인 두 분극 과정을 가진다고 하고(2-RC), 60 s 펄스로 맞춘 1-RC를 30분 연속 방전에 써 본다.

```python {exec="browser" session="ecm"}
rc_true = [(0.015, 2000.0), (0.010, 60000.0)]                    # 참 셀: 2-RC (tau 30 s, 600 s)
t_fit = np.arange(0, 181.0)
i_fit = np.where((t_fit >= 10) & (t_fit < 70), 3.0, 0.0)         # 60 s 펄스 + 휴지
v_fit = simulate(i_fit, 1.0, R0_OHM, rc_true)
def model(_, r1, c1_kf):                                         # 1-RC, R0는 즉시 강하로 고정
    return simulate(i_fit, 1.0, R0_OHM, [(r1, c1_kf * 1e3)])
(r1_hat, c1_hat), _ = curve_fit(model, t_fit, v_fit, p0=[0.01, 1.0])
t_long = np.arange(0, 1801.0)
i_long = np.full(t_long.shape, 3.0)                              # 1C 연속 방전 30분
err = simulate(i_long, 1.0, R0_OHM, [(r1_hat, c1_hat * 1e3)]) - simulate(i_long, 1.0, R0_OHM, rc_true)
rmse_fit = np.sqrt(np.mean((model(None, r1_hat, c1_hat) - v_fit) ** 2))
print(f"맞춘 1-RC: R1 {r1_hat * 1e3:.1f} mOhm, tau {r1_hat * c1_hat * 1e3:.0f} s, 맞춤 구간 RMSE {rmse_fit * 1e3:.1f} mV")
for k in [60, 300, 900, 1800]:
    print(f"연속 방전 {k:4d} s: 1-RC 예측 - 참 = {err[k] * 1e3:+6.1f} mV")
```

```output
맞춘 1-RC: R1 17.0 mOhm, tau 35 s, 맞춤 구간 RMSE 0.7 mV
연속 방전   60 s: 1-RC 예측 - 참 =   -0.1 mV
연속 방전  300 s: 1-RC 예측 - 참 =   +5.8 mV
연속 방전  900 s: 1-RC 예측 - 참 =  +17.3 mV
연속 방전 1800 s: 1-RC 예측 - 참 =  +22.5 mV
```

맞춤 구간에서는 RMSE 0.7 mV로 거의 완벽하지만, 연속 방전이 길어질수록 1-RC는 전압을 높게 예측하고 30분 뒤에는 22 mV 넘게 틀린다. 60 s 펄스에는 600 s짜리 느린 과정이 거의 드러나지 않으므로, 1-RC는 그 몫을 조금 큰 $R_1$과 $\tau$로 흡수했을 뿐이다. 실제 셀에서 이런 느린 과정의 대표가 고체 입자 안의 리튬 확산이다. 맞춤 오차가 작다는 것은 그 데이터의 시간 규모 안에서만 의미가 있다.

**DFN.** Doyle–Fuller–Newman 모델(DFN)은 다공성 전극 이론과 농축 용액 이론에 기반한 모델로, 흔히 의사 2차원(P2D, pseudo-two-dimensional) 모델이라고도 부른다. 출발점이 된 논문(Doyle 외, 1993)은 리튬 금속 음극·고분자 전해질·삽입형 양극 셀을 다뤘고, 같은 틀이 이후 양쪽 전극이 모두 삽입형인 리튬이온 셀에 쓰이게 되었다. 두 개의 공간 좌표를 쓴다. 하나는 음극–분리막–양극을 가로지르는 두께 방향 $x$이고, 다른 하나는 각 $x$ 위치에 놓인 구형 활물질 입자의 반경 방향 $r$이다. 풀어야 하는 양은 다음과 같다.

- 입자 안 고체 리튬 농도 $c_s(x, r, t)$: 구 좌표의 확산 방정식 $\partial c_s/\partial t = \frac{1}{r^2}\partial_r\!\left(D_s r^2 \partial_r c_s\right)$
- 전해질 리튬이온 농도 $c_e(x, t)$: 확산과 반응에 의한 생성·소멸
- 고체 전위 $\phi_s(x, t)$와 전해질 전위 $\phi_e(x, t)$: 전하 보존(시간 도함수가 없다)
- 입자 표면의 반응 속도: Butler–Volmer 식으로 과전압과 표면 농도에 연결

단자 전압은 양쪽 집전체의 고체 전위 차이로 얻는다. 파라미터는 수십 개(전극 두께, 공극률, 입자 반경, 확산 계수, 반응 속도 상수, 각 전극의 개방 전위 곡선 등)이고, 많은 파라미터가 서로 상쇄되어 단자 전압만으로는 따로 정해지지 않는다(30.5장).

**SPM과 SPMe.** 단일 입자 모델(SPM, single particle model)은 전해질의 농도와 전위가 균일하다고 가정한다. 그러면 반응이 전극 두께 방향으로 고르게 일어나므로, 각 전극을 대표 입자 하나로 줄일 수 있다. 상태는 두 입자의 반경 방향 농도뿐이다. 낮은 C-rate에서는 DFN과 잘 맞지만, 높은 C-rate에서는 전해질 농도 기울기와 그에 따른 과전압을 놓친다. 전해질 효과를 보정항으로 더한 모델을 SPMe(SPM with electrolyte)라 한다.

::quiz{id="spm-assumption"}

| 모델 | 상태(대략) | 주요 가정 | 계산 비용 | 쓰임 |
|---|---|---|---|---|
| ECM (1-RC, 2-RC) | 2–3개 | 회로로 단자 응답을 흉내 냄. 파라미터가 SOC·온도의 표 | 매우 작음 | BMS, 상태 추정, 대규모 팩 시뮬레이션 |
| SPM | 입자 2개 × 반경 격자 | 전해질 균일, 전극마다 입자 하나 | 작음 | 낮은 C-rate의 내부 농도, 열화 부모델의 뼈대 |
| SPMe | SPM + 전해질 몫 | 전해질 효과를 근사적으로 보정 | 작음–중간 | 중간 C-rate까지의 빠른 물리 모델 |
| DFN (P2D) | $x$ 격자 × 입자 반경 격자 | 다공성 전극·농축 용액 이론 | 큼 | 고율·두꺼운 전극, 석출 판단, 셀 설계 |

이 표의 "상태 수"는 이산화 격자에 따라 달라지므로 비교의 감만 준다. 같은 모델이라도 격자를 촘촘히 하면 비용이 빠르게 는다.

## 열 모델과 열화 모델

**열 모델.** 셀의 저항·확산·반응 속도는 모두 온도에 크게 의존하고, 셀은 운전 중 스스로 발열한다. 가장 단순한 열 모델은 셀을 온도 하나로 보는 집중 열 모델(lumped thermal model)이다.

$$
m c_p \frac{dT}{dt} = I\big(\mathrm{OCV} - V\big) - I\,T\,\frac{\partial\,\mathrm{OCV}}{\partial T} - hA\,(T - T_\text{amb}).
$$

$m c_p$는 열용량, 첫째 항은 과전압에 의한 비가역 발열(방전 + 규약에서 항상 0 이상), 둘째 항은 반응의 엔트로피 변화에 의한 가역 열(전류 방향과 $\partial\,\mathrm{OCV}/\partial T$의 부호에 따라 발열도 흡열도 된다), 마지막 항은 주변으로의 냉각이다. 저항이 온도의 함수이고 발열이 저항의 함수이므로 전기 모델과 열 모델은 **양방향으로 결합**된다. 파라미터의 온도 의존성은 흔히 Arrhenius 꼴 $k(T) = k_\text{ref}\exp\!\big[-\tfrac{E_a}{R}\big(\tfrac{1}{T} - \tfrac{1}{T_\text{ref}}\big)\big]$로 쓴다. 큰 셀이나 팩에서는 셀 안의 온도 분포가 중요해져 공간 열 모델이 필요하다.

**열화 모델.** 두 갈래가 있다.

- **경험적 모델**: 용량 손실을 시간·누적 전하량·온도·SOC의 함수로 맞춘다. 달력 열화를 시간의 제곱근에, 사이클 열화를 누적 전하량의 거듭제곱에 비례시키고 온도 의존성을 Arrhenius 꼴로 넣는 식이 흔하다. 맞춘 범위 안에서는 쓸 만하지만, knee처럼 기작이 바뀌는 현상은 맞춘 데이터에 없으면 예측하지 못한다.
- **물리 기반 모델**: SPM이나 DFN에 부반응을 더한다. 예를 들어 SEI 성장을 음극 표면의 부반응 전류로 넣으면 리튬 재고가 줄고 SEI 저항이 늘며, 석출 부모델은 음극 표면 전위가 리튬 금속 기준 0 V 아래로 내려갈 때 석출 전류를 흘린다. 기작을 해석할 수 있지만 부반응 파라미터를 정하기가 어렵다.

:::note[분야 예시]
- **셀 설계**: 전극 두께를 늘리면 에너지 밀도는 오르지만 고율에서 전해질 수송이 병목이 된다. 이런 설계 탐색은 DFN 수준 모델 없이는 할 수 없고, 수천 번의 DFN 시뮬레이션을 대리모델로 바꾸는 것이 24.5장의 전형적인 응용이다.
- **팩 열 관리**: 수백 개 셀로 된 팩의 냉각 설계에서는 셀마다 DFN을 돌리기 어렵다. 셀은 ECM과 집중 열 모델로, 팩은 열 유동 모델로 표현하는 다중 규모 접근이 흔하다.
:::

## DAE 구조

**왜 필요한가.** 배터리 모델을 직접 짜거나 시뮬레이터가 실패하는 이유를 이해하려면, 이 모델들이 ODE가 아니라 DAE라는 사실을 알아야 한다(9.7장).

**ECM에서도 나타난다.** 정전류 운전에서 1-RC 모델은 ODE와 출력 식이다. 그런데 CC-CV 충전의 CV 구간에서는 단자 전압 $V = V_\max$가 주어지고 전류 $I$가 미지수가 된다.

$$
\frac{d\,\mathrm{SOC}}{dt} = -\frac{I}{3600Q}, \quad \frac{dV_1}{dt} = -\frac{V_1}{R_1C_1} + \frac{I}{C_1}, \quad 0 = \mathrm{OCV}(\mathrm{SOC}) - I R_0 - V_1 - V_\max.
$$

$I$는 도함수가 나오지 않는 대수 변수이고, $\partial g/\partial I = -R_0 \neq 0$이므로 index 1이다. 매 step에서 마지막 식을 $I$에 대해 풀면 된다. $R_0 = 0$인 모델이라면 $g$에 $I$가 직접 나오지 않아 한 번 더 미분해야 하므로 index가 올라간다. 정전력(CP) 운전 $V I = P$도 같은 구조다.

::quiz{id="cv-charging-dae"}

**전기화학 모델에서.** DFN에서 농도 $c_s$, $c_e$는 미분 상태이고, 전위 $\phi_s$, $\phi_e$는 시간 도함수가 없는 대수 변수다. 공간을 이산화하면 큰 index-1 DAE가 된다. 그래서 다음이 필요하다.

- **일관된 초기조건**: 시작 시각의 전위들은 주어진 농도와 전류에서 대수 방정식을 만족해야 한다. 휴지에서 시작하면 쉽지만, 전류가 흐르는 도중의 상태에서 재시작하거나 운전 모드를 바꾸는 순간(CC → CV)에는 대수 변수를 다시 풀어야 한다.
- **stiff solver**: 이산화된 확산 방정식은 격자가 촘촘할수록 stiff해진다(9.6장). 암시적 BDF 계열과 대수 방정식을 함께 푸는 solver를 쓴다.
- **이벤트 처리**: 컷오프 전압 도달, CC → CV 전환은 solver의 이벤트로 처리한다. 고정 step으로 넘어가면 컷오프를 지나친다.

## 시뮬레이션 도구

**왜 필요한가.** SPM이나 DFN을 직접 이산화해 짜는 것은 좋은 연습이지만, 연구에서는 검증된 구현을 쓰고 그 설정을 정확히 기록하는 쪽이 재현성에 낫다.

집필 시점 예시로, 오픈소스 Python 패키지 **PyBaMM**(Python Battery Mathematical Modelling)이 SPM·SPMe·DFN과 열·열화 부모델, 문헌에서 가져온 파라미터 세트, DAE solver를 함께 제공한다(Sulzer 외, 2021). API와 지원 모델은 버전에 따라 바뀌므로 사용 전에 공식 문서에서 확인한다. 어떤 도구를 쓰든, 결과를 보고하거나 시뮬레이션 데이터를 학습에 쓰기 전에 다음을 점검하고 기록한다.

1. **모델과 옵션**: 어느 모델(SPM/SPMe/DFN)에 어떤 부모델(열, SEI, 석출, 입자 크기 분포)을 켰는가.
2. **파라미터 세트의 출처**: 어느 셀(화학·형식·제조사)에서, 어떤 온도와 C-rate 범위의 시험으로 정했는가. 이 범위가 곧 모델의 적용 범위다.
3. **solver 설정**: 상대·절대 tolerance, 공간 격자 수. tolerance와 격자를 한 단계씩 바꿔도 결과가 변하지 않는지(수렴) 확인한다(9.7장의 V&V).
4. **실험 정의**: 전류 프로파일, 컷오프 전압, 휴지, 초기 SOC와 온도. 초기 상태가 일관된지 확인한다.
5. **실패의 기록**: solver가 수렴하지 않은 조건을 지우지 말고 기록한다. 그 조건이 곧 모델이나 파라미터가 무너지는 영역이다.

:::tip[연구 팁]
시뮬레이션으로 ML 학습 데이터를 만들 때는 시뮬레이터의 적용 범위 밖 조건을 샘플링하지 않도록 범위를 파라미터 세트의 시험 범위에 묶어 두고, 실패한 시뮬레이션의 비율을 조건별로 보고한다. 실패한 조건을 조용히 빼면 학습 데이터의 분포가 쉬운 조건 쪽으로 치우친다.
:::

## 실습

1-RC 모델의 적분기를 직접 만든다. 영차 유지의 정확한 이산화를 쓰고, 테스트는 즉시 강하·정상 상태·시상수·SOC 적산과 큰 step에서의 안정성을 확인한다.

::exercise{id="ecm-1rc"}

이어서 다음을 해 본다(시뮬레이션 도구가 설치된 로컬 환경에서).

1. 같은 운전 프로파일(예: 1C 방전 후 휴지, 3C 방전 후 휴지)을 ECM과, 쓸 수 있다면 SPM·DFN으로 시뮬레이션한다. ECM 파라미터는 DFN의 저율 펄스 응답에서 이 장의 방법으로 뽑는다.
2. 세 모델의 단자 전압 차이와 계산 시간을 C-rate별로 표로 만든다.
3. ECM과 SPM이 DFN에서 몇 mV 이상 벗어나기 시작하는 C-rate를 찾고, 그 이유를 이 장의 가정으로 설명한다.

## 실패 사례

- **모델의 적용 범위를 넘는 C-rate·온도에서 결과를 믿는 것.** 짧은 펄스로 맞춘 ECM은 긴 고율 운전에서 느린 확산을 놓치고(cell 5), 25 °C 파라미터 세트는 저온에서 저항과 확산을 크게 틀린다. 시뮬레이션 결과에는 파라미터를 정한 시험의 범위를 함께 적는다.
- **맞춤 오차가 작으니 모델이 맞다고 결론 내리는 것.** 맞춤에 쓴 데이터의 시간 규모와 조건 밖에서 따로 검증해야 한다. 파라미터가 많은 모델일수록 맞춤 오차는 쉽게 작아진다.
- **명시적 고정 step으로 RC 쌍이나 확산 방정식을 적분하는 것.** 시상수보다 큰 step에서 발산하거나, 발산하지 않더라도 조용히 틀린다. 정확한 이산화나 암시적 solver를 쓴다.
- **CC → CV 전환이나 재시작에서 대수 변수를 다시 풀지 않는 것.** 일관되지 않은 초기조건에서 출발하면 solver가 실패하거나 첫 step에 비물리적 과도 응답이 생긴다.

## 통과 기준

- ECM·SPM·SPMe·DFN의 상태, 핵심 가정, 계산 비용을 표로 비교하고, 주어진 연구 질문에 맞는 수준을 근거와 함께 고른다.
- 1-RC 모델을 정확한 이산화로 적분하고(연습문제 `ecm-1rc` 통과), 펄스 응답에서 $R_0$, $R_1$, $\tau$를 추정한다.
- 시뮬레이션 결과를 보고할 때 모델 옵션, 파라미터 세트의 출처와 적용 범위, solver tolerance와 격자 수렴 확인을 함께 적는다.

## 참고 문헌

- Doyle, M., Fuller, T. F., Newman, J. (1993). Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell. *Journal of The Electrochemical Society*, 140(6), 1526–1533. — DFN 모델의 출발점이 된 논문
- Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., Chapman, S. J. (2021). Python Battery Mathematical Modelling (PyBaMM). *Journal of Open Research Software*, 9(1), 14. — SPM·SPMe·DFN 등을 구현한 오픈소스 시뮬레이션 패키지
