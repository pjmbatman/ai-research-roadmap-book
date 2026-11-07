---
id: physics-and-engineering/turbulence-and-closure
title: "난류와 closure 모델 학습"
title_en: "Learning Turbulence and Closure Models"
tier: advanced
status: draft
prerequisites: [physics-and-engineering/surrogate-modeling, scientific-ml/hybrid-and-surrogate-models]
---

# 난류와 closure 모델 학습

난류는 큰 소용돌이에서 작은 소용돌이로 에너지가 옮겨 가며, 가장 작은 규모에서 점성으로 소산되는 흐름이다. 모든 규모를 격자로 해상하는 직접 수치 모사(direct numerical simulation, DNS)는 Reynolds 수가 커질수록 감당할 수 없이 비싸진다. 그래서 공학 시뮬레이션은 거친 격자에서 큰 규모만 풀고, 해상하지 못한 작은 규모의 효과를 모델로 대신한다. 그 모델이 **closure**다.

closure는 머신러닝이 물리 시뮬레이션에 들어가는 가장 자연스러운 자리 중 하나다. 고해상도 데이터에서 "거친 격자가 놓친 것"을 계산할 수 있으니, 그것을 회귀하면 된다. 그런데 이 장의 계산이 보이듯, 데이터에서 잘 맞는 closure가 solver 안에서도 잘 동작한다는 보장은 없다. 24.5장의 하이브리드 모델 일반론을 난류 closure에 적용해, **학습 문제를 어떻게 정의하고 결합 후 무엇을 평가해야 하는가**를 다룬다.

## 학습 목표

- closure 문제를 학습 문제로 정의한다
- closure를 solver에 결합했을 때의 안정성과 정확도를 평가한다
- 새 Reynolds 수에서 closure의 일반화를 점검한다

## closure 문제

**왜 필요한가.** 난류의 규모는 대표 길이 $L$에서 점성이 지배하는 Kolmogorov 규모 $\eta$까지 퍼져 있고, 그 비는 대략 $L/\eta \sim \mathrm{Re}^{3/4}$이다. 3차원에서 모든 규모를 해상하려면 격자점이 $\mathrm{Re}^{9/4}$에 비례해 필요하고, 시간 step 수까지 고려한 비용은 대략 $\mathrm{Re}^3$에 비례한다(Pope, 2000). Re가 10배 커지면 비용은 1000배가 된다. 비행기 날개나 대기 경계층처럼 Re가 $10^6$–$10^8$인 문제를 DNS로 푸는 것은 현실적이지 않다.

**정의.** 비선형 방정식을 거친 격자에서 풀기 위해 해를 필터(공간 평균) $\bar{\cdot}$로 거른다고 하자. Burgers 방정식 $u_t + (u^2/2)_x = \nu u_{xx}$에 필터를 적용하면

$$
\bar{u}_t + \left(\frac{\bar{u}^2}{2}\right)_x = \nu \bar{u}_{xx} - \tau_x, \qquad \tau = \frac{1}{2}\left(\overline{u^2} - \bar{u}^2\right)
$$

가 된다. 필터와 곱셈이 교환되지 않기 때문에($\overline{u^2} \ne \bar{u}^2$) 걸러 낸 변수 $\bar{u}$만으로는 방정식이 닫히지 않는다. 남는 항 $\tau$를 **subgrid 응력**(subgrid-scale stress)이라 하고, $\tau$를 $\bar{u}$의 함수로 근사하는 것을 closure라 한다. 3차원 Navier–Stokes에서는 같은 이유로 $\tau_{ij} = \overline{u_i u_j} - \bar{u}_i\bar{u}_j$가 생긴다.

필터의 종류에 따라 두 계열로 나뉜다.

- **LES**(large eddy simulation): 공간 필터로 큰 소용돌이는 풀고 격자보다 작은 규모만 모델링한다. 해가 시간에 따라 변하는 큰 구조를 담는다.
- **RANS**(Reynolds-averaged Navier–Stokes): 시간 평균(앙상블 평균)으로 모든 요동을 평균해 버리고 평균 흐름만 푼다. 요동 전체의 효과인 Reynolds 응력을 모델링해야 해서 모델 오차가 크지만 훨씬 싸다. 공학 설계의 대부분이 여기에 기댄다.

**직관.** 고해상도 해가 있으면 $\tau$는 정확히 계산된다. 고해상도 격자의 값을 $K$칸씩 묶어 평균하는 블록 필터를 쓰면, $\overline{u^2} - \bar{u}^2$은 블록 안의 **분산**이다. 그래서 $\tau \ge 0$이 항상 성립한다. 이런 물리적 일관성 조건을 **실현성**(realizability)이라 한다. 3차원 LES의 $\tau_{ij}$는 양의 준정부호 행렬이어야 한다는 같은 성질을 가진다.

파수 1–5의 무작위 초기 장에서 점성 Burgers 방정식($\nu = 0.04$)을 256칸 격자로 풀고, 8칸씩 묶어 32칸 거친 격자의 $\bar{u}$와 $\tau$를 만든다. 시드 0–3은 학습, 4–6은 시험에 쓴다. 거친 격자의 이류항은 에너지를 보존하는 중심 유속으로 이산화해, 수치 확산이 closure의 역할을 대신하지 않게 한다.

```python {exec="browser" session="les"}
import numpy as np

L, Nf, Nc = 2 * np.pi, 256, 32                       # 고해상도 256칸, 거친 격자 32칸
K, D = Nf // Nc, L / Nc                               # 필터 폭: 고해상도 8칸 = 거친 셀 하나

def ic(seed, N):                                      # 파수 1–5의 무작위 초기 장
    rng, x = np.random.default_rng(seed), np.arange(N) * L / N
    return sum(rng.normal() / k * np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))

def rhs(u, dx, nu, tau=None):                         # u_t + (u²/2)_x = nu u_xx - tau_x, 에너지 보존형 중심 유속
    ur = np.roll(u, -1)
    F = (u**2 + u * ur + ur**2) / 6 - nu * (ur - u) / dx   # (N,) 면 유속
    if tau is not None:
        F = F + 0.5 * (tau + np.roll(tau, -1))
    return -(F - np.roll(F, 1)) / dx

def run(u, dx, nu, model=None, T=2.0, nsave=20):      # Heun 적분, 저장한 상태 (nsave, N)
    n = int(np.ceil(T / (0.3 * min(dx / 3, dx**2 / (2 * nu + 1e-12)))))
    out = []
    for i in range(n):
        u1 = u + T / n * rhs(u, dx, nu, model(u) if model else None)
        u = 0.5 * (u + u1 + T / n * rhs(u1, dx, nu, model(u1) if model else None))
        if (i + 1) % (n // nsave) == 0:
            out.append(u)
    return np.array(out)[:nsave]

filt = lambda u: u.reshape(*u.shape[:-1], -1, K).mean(-1)    # 블록 평균 필터 (..., Nf) → (..., Nc)
tau_of = lambda U: 0.5 * (filt(U**2) - filt(U) ** 2)         # subgrid 응력 τ = (평균(u²) − ū²)/2
fine = {s: run(ic(s, Nf), L / Nf, 0.04) for s in range(7)}   # 시드 0–3 학습, 4–6 시험; 각 (20, 256)
Ub, Tau = filt(fine[0]), tau_of(fine[0])                      # (20, 32), (20, 32)
print("τ의 최솟값 ≥ 0:", bool(Tau.min() >= 0), f" τ 최댓값 {Tau.max():.3f}")
print(f"거친 격자에서 τ_x 크기 / 이류항 (ū²/2)_x 크기: {np.linalg.norm(np.gradient(Tau, D, axis=1)) / np.linalg.norm(np.gradient(Ub**2 / 2, D, axis=1)):.2f}")
```

```output
τ의 최솟값 ≥ 0: True  τ 최댓값 0.012
거친 격자에서 τ_x 크기 / 이류항 (ū²/2)_x 크기: 0.05
```

$\tau$는 모든 점에서 0 이상이다. 크기는 이류항의 약 5%로 작아 보이지만, 시간 적분에서 누적되면 거친 격자 해의 에너지와 충격파 모양을 바꾼다(셀 3).

::quiz{id="filtered-product-stress"}

**학습 문제로 쓰면.** 입력은 거친 격자의 상태 $\bar{u}$(와 그 기울기 등 국소 특징), 출력은 $\tau$, 데이터는 고해상도 해를 걸러 만든 쌍 $\{(\bar{u}, \tau)\}$다. 국소 특징만 쓰면 모델은 평행이동 등변성을 갖고, 한 스냅샷의 모든 격자점이 표본이 된다. 이 정의에는 숨은 가정이 둘 있다. 하나는 학습에 쓴 필터가 실제 거친 격자 solver가 암묵적으로 적용하는 필터와 같다는 가정이고, 다른 하나는 $\tau$가 현재 시각의 국소 상태만으로 결정된다는 가정이다. 둘 다 정확히 성립하지 않으며, 다음 절들의 결과는 그 차이에서 나온다.

## 학습 기반 closure

**왜 필요한가.** 전통적인 closure는 물리적 논증으로 만든 식에 상수 한두 개를 맞춘 것이다. 데이터 기반 closure는 더 유연한 함수로 고해상도 데이터를 직접 맞춘다. 유연성이 커지는 만큼, 전통 모델이 구조적으로 보장하던 성질을 잃지 않게 하는 것이 설계의 핵심이다.

**세 모델.**

- **Smagorinsky 모델**(eddy viscosity): $\tau = -\nu_t\,\bar{u}_x$, $\nu_t = C\Delta^2|\bar{u}_x| \ge 0$. 작은 규모가 큰 규모의 에너지를 빼앗아 가는 효과를 추가 점성으로 표현한다(Smagorinsky, 1963). $\Delta$는 필터 폭이다. $C \ge 0$이면 항상 에너지를 소산시킨다. 상수 $C$를 흐름에 따라 계산 중에 정하는 동적 모델도 있다(Germano 외, 1991).
- **기울기 모델**(gradient model): $\tau = C_g\Delta^2\bar{u}_x^2$. 필터를 Taylor 전개하면 나오는 형태로, 블록 안 분산이 기울기 제곱에 비례한다는 직관과 같다.
- **선형 회귀**: 국소 특징 5개($\bar{u}_x^2$, $-|\bar{u}_x|\bar{u}_x$, $\bar{u}\bar{u}_x$, $\bar{u}^2$, $\bar{u}_x$)의 선형 결합을 최소제곱으로 맞춘다. 물리 구조 없이 데이터에 맞춘 모델의 대표다.

**a priori 평가**는 학습에 쓰지 않은 고해상도 해를 걸러, 모델의 $\tau$ 예측을 참 $\tau$와 비교하는 것이다. solver를 돌리지 않는 일반적인 회귀 평가다. 여기에 물리적 진단 하나를 더한다. $\int \bar{u}(-\tau_x)\,dx = \int \tau\,\bar{u}_x\,dx$는 closure가 해상된 운동에너지를 바꾸는 속도다(주기 경계에서 부분적분). 음수이면 에너지를 빼앗고(소산), 양수이면 작은 규모의 에너지를 큰 규모로 되돌린다(**역산란**, backscatter).

```python {exec="browser" session="les"}
ux = lambda ub: (np.roll(ub, -1, -1) - np.roll(ub, 1, -1)) / (2 * D)
feats = lambda ub: np.stack([ux(ub)**2, -np.abs(ux(ub)) * ux(ub), ub * ux(ub), ub**2, ux(ub)], -1)
Ub_tr = np.concatenate([filt(fine[s]) for s in range(4)])     # (80, 32)
y = np.concatenate([tau_of(fine[s]) for s in range(4)]).ravel()
fit = lambda f: f.ravel() @ y / (f.ravel() @ f.ravel())        # 계수 하나의 최소제곱
smag_f = -D**2 * np.abs(ux(Ub_tr)) * ux(Ub_tr)                 # Smagorinsky: τ = −(C Δ²|ū_x|) ū_x
C_s, C_g = max(fit(smag_f), 0.0), fit(D**2 * ux(Ub_tr) ** 2)   # C_s ≥ 0으로 제한
w = np.linalg.lstsq(feats(Ub_tr).reshape(-1, 5), y, rcond=None)[0]
models = {"Smagorinsky": lambda ub: -C_s * D**2 * np.abs(ux(ub)) * ux(ub),
          "기울기 모델": lambda ub: C_g * D**2 * ux(ub) ** 2,
          "선형 회귀(5특징)": lambda ub: feats(ub) @ w}
Ub_te = np.concatenate([filt(fine[s]) for s in (4, 5, 6)])     # (60, 32)
T_te = np.concatenate([tau_of(fine[s]) for s in (4, 5, 6)])
Pi = lambda t: (t * ux(Ub_te)).sum(1) * D                    # (60,) 스냅샷별 에너지 전달 ∫ τ ū_x dx (< 0이면 소산)
print(f"참 τ         에너지 전달 평균 {Pi(T_te).mean():+.4f}  역산란(전달 > 0) 스냅샷 {np.mean(Pi(T_te) > 0):.0%}")
for name, m in models.items():
    p = m(Ub_te)
    print(f"{name:12s} 상관 {np.corrcoef(p.ravel(), T_te.ravel())[0, 1]:.3f}  "
          f"상대 오차 {np.linalg.norm(p - T_te) / np.linalg.norm(T_te):.3f}  에너지 전달 평균 {Pi(p).mean():+.4f}  역산란 스냅샷 {np.mean(Pi(p) > 0):.0%}")
```

```output
참 τ         에너지 전달 평균 -0.1165  역산란(전달 > 0) 스냅샷 2%
Smagorinsky  상관 0.802  상대 오차 0.681  에너지 전달 평균 -0.2037  역산란 스냅샷 0%
기울기 모델       상관 0.833  상대 오차 0.638  에너지 전달 평균 -0.1659  역산란 스냅샷 2%
선형 회귀(5특징)   상관 0.827  상대 오차 0.583  에너지 전달 평균 -0.1260  역산란 스냅샷 23%
```

a priori 지표만 보면 Smagorinsky가 가장 나쁘다. 상관은 가장 낮고 상대 오차는 가장 크며, 에너지 전달의 평균(−0.20)이 참값(−0.12)의 두 배 가까이 되어 에너지를 과하게 빼앗는다. 선형 회귀는 상대 오차가 가장 작고 에너지 전달 평균도 참값에 가깝다. 그러나 선형 회귀는 시험 스냅샷의 23%에서 전체 에너지를 거꾸로 주입한다. 참 $\tau$에서 그런 스냅샷은 2%뿐이다. 평균 오차는 작아도 물리적으로 다른 일을 하는 모델이다.

:::note[분야 예시]
- **항공·자동차**: 날개와 차체 주위 유동은 대부분 RANS로 설계한다. RANS 모델이 박리 영역에서 틀리는 것이 잘 알려져 있어, 고해상도 데이터나 실험으로 모델 항을 보정하는 데이터 기반 연구가 활발하다(Duraisamy 외, 2019).
- **대기·해양**: 기후 모델의 격자는 수십 km라서 구름 대류, 해양 중규모 소용돌이를 해상하지 못한다. 이들의 효과를 나타내는 모수화(parameterization)가 곧 closure이고, 고해상도 모의로 모수화를 학습하는 연구가 활발하다. 기상·기후 응용 전반은 27장에서 다룬다.
- **핵융합 플라즈마**: 장치 전체의 수송 계산은 미세 난류를 해상하지 않고, 국소 기울기로부터 난류 열유속을 주는 모델에 기댄다. 26.2장의 대리모델이 이 자리에서 closure 역할을 한다.
:::

**물리 제약.** 데이터 기반 closure에 넣을 수 있는 제약은 다음과 같다.

- **대칭**: 좌표 회전·반사·Galilean 변환(일정 속도로 움직이는 좌표계)에 대해 올바르게 변환해야 한다. 특징을 불변량으로 만들면 구조로 보장된다. 위 선형 회귀의 $\bar{u}\bar{u}_x$, $\bar{u}^2$ 항은 Galilean 변환에 불변이 아니다. 배경 속도가 바뀌면 예측이 바뀐다.
- **실현성**: $\tau \ge 0$(3차원에서는 양의 준정부호).
- **소산 부호**: 에너지를 주입하는 정도를 제한한다. Smagorinsky처럼 $\nu_t \ge 0$의 형태로 출력하게 하면 구조로 보장된다.

## solver와의 결합

**왜 필요한가.** closure의 목적은 $\tau$를 맞추는 것이 아니라, **closure를 넣은 거친 격자 시뮬레이션이 고해상도 해를 거른 결과를 맞추는 것**이다. 둘은 다르다. 시간 적분 중에 closure의 오차가 거친 격자 해를 바꾸고, 바뀐 해가 다시 closure의 입력이 된다. 학습 데이터의 분포 밖으로 나간 입력에서 closure가 어떻게 행동하는지가 결과를 정한다.

**a posteriori 평가**는 closure를 solver에 넣고 초기조건부터 시간 적분한 결과를 걸러 낸 고해상도 궤적과 비교하는 것이다. 같은 세 모델과 closure가 없는 경우를 비교한다.

```python {exec="browser" session="les"}
def a_posteriori(nu, refs, T=2.0):
    for name, m in {"모델 없음": None, **models}.items():
        errs, e_err = [], []
        for s, U in refs.items():
            ref = filt(U)                                         # (20, 32) 걸러 낸 고해상도 궤적
            les = run(filt(ic(s, Nf)), D, nu, m, T=T)             # (20, 32) 거친 격자 + closure
            errs.append(np.linalg.norm(les[-1] - ref[-1]) / np.linalg.norm(ref[-1]))
            e_err.append(np.mean(les[-1] ** 2) / np.mean(ref[-1] ** 2) - 1)
        print(f"  {name:12s} 최종 시각 상대 오차 {np.mean(errs):.3f}  에너지 오차 {np.mean(e_err):+6.1%}")

print("nu = 0.04 (학습과 같은 조건)")
a_posteriori(0.04, {s: fine[s] for s in (4, 5, 6)})
```

```output
nu = 0.04 (학습과 같은 조건)
  모델 없음        최종 시각 상대 오차 0.116  에너지 오차  +7.3%
  Smagorinsky  최종 시각 상대 오차 0.091  에너지 오차  +2.0%
  기울기 모델       최종 시각 상대 오차 0.093  에너지 오차  +5.0%
  선형 회귀(5특징)   최종 시각 상대 오차 0.114  에너지 오차  +6.7%
```

순위가 뒤집혔다. a priori에서 가장 나빴던 Smagorinsky가 결합 후에는 가장 정확하고, a priori에서 가장 좋았던 선형 회귀는 closure가 없는 경우와 거의 같다. 에너지 오차가 이유를 보여 준다. closure가 없으면 거친 격자가 에너지를 7% 과대 보유한다. 에너지를 보존하는 중심 유속이 격자 규모에 쌓인 에너지를 없애지 못하기 때문이다. Smagorinsky의 "과한" 소산은 subgrid 효과에 더해 거친 격자의 **이산화 오차**까지 메운다. a priori 목표인 $\tau$는 필터로 정의되어 이산화 오차를 모르므로, $\tau$를 정확히 맞춘 모델은 이 부분을 놓친다. 학습 데이터의 목표 정의가 solver와 맞지 않으면, a priori 정확도는 결합 후 성능을 예측하지 못한다.

::widget{name="closure-apriori"}

::quiz{id="a-priori-not-enough"}

**결합 후 불안정성.** 이 예에서는 발산까지 가지 않았지만, 역산란을 허용하는 모델이나 학습 범위 밖 입력에서 큰 값을 내는 모델을 solver에 넣으면 시간 적분이 발산할 수 있다. 결합 후 평가에서는 정확도와 함께 다음을 확인한다.

- 학습 구간보다 긴 시간까지 적분했을 때 발산하거나 비물리적 값(최대 원리 위반, 음수 에너지)이 생기는가
- 에너지·보존량이 물리적으로 맞는 방향으로 변하는가
- 시간 간격·격자를 바꿔도 결과가 유지되는가. closure는 특정 격자 간격에 맞춰 학습되므로 격자를 바꾸면 다시 검증한다

a posteriori 손실로 직접 학습하는 방법도 있다. 학습 중에 solver를 함께 돌리고, 그 결과와 고해상도 궤적의 차이를 손실로 쓴다. 미분 가능한 solver가 필요하고(24.3장) 학습이 비싸지만, a priori 목표와 실제 목적의 불일치를 없앤다.

## 일반화

**왜 필요한가.** closure를 학습하는 이유는 고해상도 계산을 할 수 없는 조건, 즉 더 큰 Re나 새 형상에서 쓰기 위해서다. 학습 조건에서만 잘 맞는 closure는 그 목적을 이루지 못한다.

학습($\nu = 0.04$)보다 Re가 두 배 큰 $\nu = 0.02$에서 같은 closure를 시험한다. 이 조건은 256칸 격자로 아직 해상할 수 있으므로 기준 해가 있다. 이어서 점성이 없는 극한 $\nu = 0$에서는 기준 해 없이 물리 검사만 한다. 점성 없는 Burgers 방정식의 올바른(엔트로피) 해는 충격파에서 에너지를 잃고, 최댓값이 초기 최댓값을 넘지 않는다.

```python {exec="browser" session="les"}
print("nu = 0.02 (학습하지 않은 조건, 고해상도 기준 있음)")
a_posteriori(0.02, {s: run(ic(s, Nf), L / Nf, 0.02) for s in (4, 5, 6)})
print("nu = 0 (점성 없음, 기준 해 없이 물리 검사만)")
for name, m in {"모델 없음": None, **models}.items():
    u0 = filt(ic(4, Nf))
    u = run(u0, D, 0.0, m, T=4.0)[-1]
    print(f"  {name:12s} 에너지 E(4)/E(0) {np.mean(u**2) / np.mean(u0**2):.3f}  "
          f"최댓값 {np.abs(u).max():.2f} (초기 {np.abs(u0).max():.2f})")
```

```output
nu = 0.02 (학습하지 않은 조건, 고해상도 기준 있음)
  모델 없음        최종 시각 상대 오차 0.274  에너지 오차 +14.2%
  Smagorinsky  최종 시각 상대 오차 0.174  에너지 오차  +4.4%
  기울기 모델       최종 시각 상대 오차 0.205  에너지 오차  +8.4%
  선형 회귀(5특징)   최종 시각 상대 오차 0.214  에너지 오차  +9.9%
nu = 0 (점성 없음, 기준 해 없이 물리 검사만)
  모델 없음        에너지 E(4)/E(0) 1.001  최댓값 2.99 (초기 1.69)
  Smagorinsky  에너지 E(4)/E(0) 0.338  최댓값 1.46 (초기 1.69)
  기울기 모델       에너지 E(4)/E(0) 0.720  최댓값 2.07 (초기 1.69)
  선형 회귀(5특징)   에너지 E(4)/E(0) 0.422  최댓값 1.43 (초기 1.69)
```

$\nu = 0.02$에서 closure가 없는 경우의 오차는 27%로 커졌고, 모든 closure가 이를 줄였다. Smagorinsky가 여전히 가장 좋다. $\nu = 0$에서는 물리 검사가 모델들을 가른다. closure가 없으면 에너지가 정확히 보존되고(E(4)/E(0) = 1.001) 최댓값이 초기의 1.69에서 2.99로 커진다. 둘 다 틀린 물리다. 거친 격자의 에너지 보존형 이산화는 충격파에서 에너지를 잃어야 한다는 것을 모르기 때문이다. 기울기 모델도 최댓값이 2.07로 커져 최대 원리를 어긴다. Smagorinsky와 선형 회귀는 두 검사를 통과한다. Smagorinsky의 소산은 $\nu$와 무관하게 $C\Delta^2|\bar{u}_x|$로 정해지므로, 점성이 사라져도 격자 규모에서 에너지를 빼는 구조가 남는다. 높은 Re에서 소산율이 점성과 무관해지는 난류의 성질과 맞는 구조다.

::quiz{id="inviscid-limit-check"}

**연구에서의 의미.** 이 작은 실험의 결론을 일반화할 수는 없다. 실제 연구에서 데이터 기반 closure가 전통 모델을 넘어서는 경우도 많다. 그러나 평가의 원칙은 일반적이다.

- a priori 지표는 모델 선택의 근거가 될 수 없다. 반드시 결합해 평가한다.
- 학습 Re 하나에서의 결과는 일반화의 증거가 아니다. 학습 범위 밖의 Re, 다른 형상(벽이 있는 흐름, 박리가 있는 흐름), 다른 격자 간격에서 결합 평가를 한다.
- 기준 해가 없는 조건에서는 물리 검사(에너지 수지, 실현성, 최대 원리)가 유일한 근거다. 물리 검사가 26.2장에서처럼 필요조건일 뿐이라는 점도 함께 기억한다.

## 실습

간단한 난류 모델의 보정 항을 학습하고, a priori 오차와 solver에 결합한 a posteriori 결과를 비교한다. 코드는 `labs/physics-and-engineering/turbulence-and-closure/closure_study.py`다. 본문과 같은 설정에서 학습 시드를 8개로 늘리고, 세 closure를 비교한다. (1) Smagorinsky($C \ge 0$), (2) 5점 스텐실의 $\bar{u}$를 입력으로 $\tau$를 출력하는 MLP, (3) 같은 MLP가 $\nu_t \ge 0$을 출력하게 해 $\tau = -\nu_t\bar{u}_x$로 쓰는 소산 제약 MLP. 학습 조건($\nu = 0.04$)과 새 조건($\nu = 0.02$)에서, 학습 궤적 길이($T = 2$)와 그 세 배($T = 6$)까지 결합 적분한다. 집필 시점의 실행 결과는 다음과 같다.

```output
[a priori] 시험 스냅샷 120개에서 tau 예측
  Smagorinsky    상대 오차 0.675  상관 0.843
  MLP            상대 오차 0.201  상관 0.985
  MLP + 소산 제약    상대 오차 0.194  상관 0.985
[a posteriori] nu = 0.04, T = 2.0
  모델 없음          평균 상대 오차 0.107
  Smagorinsky    평균 상대 오차 0.058
  MLP            평균 상대 오차 0.078
  MLP + 소산 제약    평균 상대 오차 0.074
[a posteriori] nu = 0.04, T = 6.0
  모델 없음          평균 상대 오차 0.068
  Smagorinsky    평균 상대 오차 0.046
  MLP            평균 상대 오차 0.061
  MLP + 소산 제약    평균 상대 오차 0.055
[a posteriori] nu = 0.02, T = 2.0
  모델 없음          평균 상대 오차 0.251
  Smagorinsky    평균 상대 오차 0.141
  MLP            평균 상대 오차 0.168
  MLP + 소산 제약    평균 상대 오차 0.162
[a posteriori] nu = 0.02, T = 6.0
  모델 없음          평균 상대 오차 0.106
  Smagorinsky    평균 상대 오차 0.072
  MLP            평균 상대 오차 0.082
  MLP + 소산 제약    평균 상대 오차 0.080
```

결과를 읽는 법은 다음과 같다.

- **a priori 오차가 세 배 이상 작아도 결합 후에는 진다.** 두 MLP의 a priori 상대 오차(0.20, 0.19)는 Smagorinsky(0.675)보다 훨씬 작고 상관은 0.985에 이른다. 그런데 네 가지 결합 조건 모두에서 Smagorinsky의 오차가 가장 작다. 본문 셀 3과 같은 이유로, $\tau$를 정확히 맞추는 것과 거친 격자 solver를 정확하게 만드는 것은 다른 목표다.
- **소산 제약은 결합 후 결과를 조금씩 개선한다.** 같은 구조의 MLP에서 $\nu_t \ge 0$ 제약을 넣은 쪽이 네 조건 모두에서 오차가 약간 작다. a priori 지표로는 두 MLP의 차이가 보이지 않는다.
- **긴 시간 적분과 새 Re는 평가에 꼭 필요한 축이다.** $T = 6$에서는 해가 감쇠해 모든 모델의 오차가 줄지만, 모델 사이의 순위는 학습 조건과 같은지 따로 확인해야 한다. 학습 구간만 보고 결론 내리지 않는다.

스크립트를 고쳐 다음을 확인한다. (1) MLP의 목표를 $\tau$ 대신 "고해상도 해를 거른 시간 미분 − 거친 격자 solver의 우변"(이산화 오차까지 포함한 보정)으로 바꾸면 a priori·a posteriori 결과가 어떻게 바뀌는가. (2) 거친 격자의 이류 유속을 26.1장의 Rusanov 유속으로 바꾸면 closure가 여전히 필요한가. (3) 학습에 $\nu = 0.02$ 데이터를 섞으면 $\nu = 0.04$ 결과는 나빠지는가.

이어서 블록 필터와 subgrid 응력을 직접 계산하는 연습문제를 푼다.

::exercise{id="subgrid-stress"}

## 실패 사례

- **a priori 오차가 작으면 결합 후에도 잘 동작한다고 가정하는 것.** 본문과 실습에서 a priori 오차가 가장 작은 모델이 결합 후에는 가장 나쁘거나 closure 없음과 비슷했다. a priori 목표는 solver의 이산화 오차를 모르고, 결합 후에는 모델이 자기 오차가 만든 입력을 다시 받는다. closure 논문의 핵심 결과는 결합 평가여야 한다.
- **학습 필터와 solver의 실제 필터가 다른 것을 무시하는 것.** 블록 평균으로 만든 $\tau$를 다른 이산화의 solver에 넣으면 목표 자체가 어긋난다. 학습 데이터를 만든 필터와 거친 격자 solver의 이산화를 함께 기록하고, 가능하면 solver와 일관된 목표를 쓴다.
- **에너지 수지를 보지 않는 것.** 평균 오차가 작아도 역산란을 과하게 내는 모델은 장기 적분에서 에너지를 쌓는다. 스냅샷별 에너지 전달의 부호를 진단한다.
- **학습 Re에서만 평가하는 것.** closure는 고해상도 계산이 불가능한 조건에서 쓰려고 만든다. 학습 범위 밖의 Re에서 결합 평가를 하고, 기준 해가 없으면 물리 검사로 대신한다.

## 통과 기준

- closure 학습 문제의 입력·출력·필터·데이터 생성 조건을 명시하고, 학습 목표가 solver의 이산화와 일관된지 설명한다.
- closure 모델을 solver에 결합한 상태에서 정확도(걸러 낸 고해상도 해와의 오차)와 안정성(긴 시간 적분, 에너지 수지, 최대 원리)을 보고하고, a priori 지표와의 차이를 설명한다.
- 학습하지 않은 Reynolds 수 하나 이상에서 결합 평가를 하고, 기준 해가 없는 조건에서는 물리 검사 결과를 보고한다.
- 연습문제 `subgrid-stress`의 테스트를 통과한다.

## 참고 문헌

- Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press. ISBN 978-0-521-59886-6. — 난류의 규모, DNS 비용의 Re 의존성, LES와 subgrid 모델
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations: I. The basic experiment. *Monthly Weather Review*, 91(3), 99–164. — eddy viscosity형 subgrid 모델의 원 논문
- Germano, M., Piomelli, U., Moin, P., Cabot, W. H. (1991). A dynamic subgrid-scale eddy viscosity model. *Physics of Fluids A*, 3(7), 1760–1765. doi:10.1063/1.857955 — 계수를 계산 중에 정하는 동적 모델
- Duraisamy, K., Iaccarino, G., Xiao, H. (2019). Turbulence modeling in the age of data. *Annual Review of Fluid Mechanics*, 51, 357–377. doi:10.1146/annurev-fluid-010518-040547 — 데이터 기반 난류 모델링의 개관
