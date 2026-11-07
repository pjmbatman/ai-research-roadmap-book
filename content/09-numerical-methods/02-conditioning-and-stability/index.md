---
id: numerical-methods/conditioning-and-stability
title: "조건수·안정성·수렴"
title_en: "Conditioning, Stability, and Convergence"
tier: core
status: draft
prerequisites: [numerical-methods/floating-point]
---

# 조건수·안정성·수렴

수치 계산의 답이 틀렸을 때 원인은 둘 중 하나다. **문제**가 입력의 작은 오차에 지나치게 민감하거나, **알고리즘**이 계산 도중 쓸데없이 오차를 키웠거나. 앞의 것은 어떤 알고리즘으로도 고칠 수 없고 문제를 다시 세워야 한다. 뒤의 것은 식이나 알고리즘을 바꾸면 고쳐진다. 두 경우를 구분하지 못하면, 멀쩡한 solver를 바꾸느라 시간을 쓰거나 고칠 수 있는 오차를 데이터 탓으로 돌리게 된다.

이 장은 문제의 민감도를 재는 **조건수**, 알고리즘의 품질을 재는 **안정성**, 반복 알고리즘이 얼마나 빨리 답에 다가가는지를 재는 **수렴 차수**를 다룬다. 마지막으로 조건수를 문제 쪽에서 줄이는 **스케일링과 전처리**를 본다. 9.1장의 machine epsilon $\varepsilon$과 unit roundoff $u = \varepsilon/2$를 그대로 쓴다.

## 학습 목표

- 문제의 조건수와 알고리즘의 안정성을 구분한다
- 수렴 속도를 분석한다

## 조건수

**왜 필요한가.** 입력에는 언제나 오차가 있다. 측정 잡음, 이전 계산의 반올림, float64로 저장하는 순간의 $u$ 크기 오차. 이 작은 입력 오차가 출력에서 얼마나 커지는지는 알고리즘과 무관한 **문제 자체의 성질**이다. 이 크기를 모르면 계산 결과의 유효숫자가 몇 자리인지 말할 수 없다.

**정의.** 함수 $f$가 입력 $x$를 출력 $f(x)$로 보내는 문제를 생각한다. 입력의 작은 상대 섭동이 출력의 상대 변화로 몇 배 증폭되는지를 **상대 조건수**(relative condition number)라 한다.

$$
\kappa_f(x) = \lim_{\delta \to 0} \sup_{\|\Delta x\| \le \delta} \frac{\|f(x + \Delta x) - f(x)\| / \|f(x)\|}{\|\Delta x\| / \|x\|}.
$$

스칼라 미분가능 함수라면 $\kappa_f(x) = |x f'(x) / f(x)|$다. 몇 가지 예를 보면 감이 온다.

- $f(x) = \sqrt{x}$: $\kappa = 1/2$. 입력의 상대 오차가 절반으로 줄어든다. 조건이 좋다.
- $f(x) = e^{x}$: $\kappa = |x|$. $x = 100$이면 입력의 상대 오차가 100배가 된다.
- $f(x) = x - 1$: $\kappa = |x/(x-1)|$. $x$가 1에 가까우면 무한히 커진다. 9.1장의 **파국적 상쇄는 뺄셈이라는 문제가 $x \approx 1$에서 조건이 나쁘다는 사실**의 다른 이름이다.

**선형시스템의 조건수.** $\mathbf{A}\mathbf{x} = \mathbf{b}$에서 $\mathbf{b}$가 $\Delta\mathbf{b}$만큼 바뀌면 해는 $\Delta\mathbf{x} = \mathbf{A}^{-1}\Delta\mathbf{b}$만큼 바뀐다. 노름의 성질로 정리하면

$$
\frac{\|\Delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A})\,\frac{\|\Delta\mathbf{b}\|}{\|\mathbf{b}\|}, \qquad \kappa(\mathbf{A}) = \|\mathbf{A}\|\,\|\mathbf{A}^{-1}\| = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}
$$

이다. 마지막 등호는 2-노름일 때이고, $\sigma$는 7.6장의 특잇값이다. $\mathbf{A}$ 자체가 섭동될 때도 같은 $\kappa(\mathbf{A})$가 증폭률을 정한다. $\kappa(\mathbf{A}) \ge 1$이고, 직교행렬은 $\kappa = 1$로 가장 조건이 좋다. 특이행렬은 $\sigma_{\min} = 0$이라 $\kappa = \infty$다.

**직관.** $\mathbf{A}$는 단위 구를 축 길이가 $\sigma_1, \dots, \sigma_n$인 타원체로 보낸다(7.6장). 해를 구하는 것은 그 역을 따라가는 일이다. 가장 짧은 축 방향으로 $\mathbf{b}$가 조금 흔들리면, 그 흔들림은 $1/\sigma_{\min}$배로 부풀려진 채 $\mathbf{x}$에 나타난다. 타원체가 납작할수록 조건이 나쁘다. 경험칙으로 **조건수가 $10^{k}$이면 유효숫자 $k$자리를 잃는다.** float64의 약 16자리에서 $\kappa = 10^{10}$이면 6자리 정도가 남는다.

**코드.** Hilbert 행렬 $H_{ij} = 1/(i + j - 1)$은 조건이 나쁜 행렬의 대표 예다. 이 행렬은 구간 $[0, 1]$에서 단항식 $1, t, t^2, \dots$의 내적 $\int_0^1 t^{i-1} t^{j-1}\,dt$를 모은 Gram 행렬, 즉 다항식 회귀의 정규방정식 행렬 $\mathbf{X}^\top\mathbf{X}/N$의 연속 극한이다. $\mathbf{b}$에 $10^{-10}$ 크기의 섭동을 주고 해의 상대 변화를 잰다.

```python {exec="browser" session="cond"}
import numpy as np
from scipy.linalg import hilbert

rng = np.random.default_rng(0)
for n in [5, 8, 11]:
    A = hilbert(n)                                   # (n, n) 조건이 나쁜 행렬
    x = np.ones(n)
    b = A @ x
    db = 1e-10 * rng.normal(size=n)                  # b에 작은 섭동
    dx = np.linalg.solve(A, b + db) - np.linalg.solve(A, b)
    amp = (np.linalg.norm(dx) / np.linalg.norm(x)) / (np.linalg.norm(db) / np.linalg.norm(b))
    print(f"n = {n:2d}  cond(A) = {np.linalg.cond(A):.1e}  상대 오차 증폭 = {amp:.1e}")
```

```output
n =  5  cond(A) = 4.8e+05  상대 오차 증폭 = 2.9e+04
n =  8  cond(A) = 1.5e+10  상대 오차 증폭 = 1.3e+09
n = 11  cond(A) = 5.2e+14  상대 오차 증폭 = 2.4e+14
```

$11 \times 11$ 행렬에서 $\mathbf{b}$의 상대 섭동이 해에서 $10^{14}$배로 커졌다. 증폭률은 항상 $\kappa(\mathbf{A})$ 이하이고, 무작위 섭동은 최악의 방향(가장 작은 특잇값의 방향)과 일부만 겹치므로 상한보다 조금 작게 나온다. 여기서 쓴 `np.linalg.solve`는 아무 잘못이 없다. 같은 섭동을 주면 어떤 알고리즘으로 풀어도 해는 이만큼 바뀐다.

::widget{name="condition-ellipse"}

::quiz{id="digits-lost"}

**연구에서의 의미.** 조건수는 "이 데이터와 이 모델로 이 정밀도의 답을 얻을 수 있는가"를 알려 준다. 회귀 계수의 부호가 데이터 몇 개를 바꿀 때마다 뒤집히거나, 같은 실험을 반복할 때 추정한 물리 상수가 크게 흔들린다면 먼저 설계 행렬의 조건수를 본다. 조건이 나쁜 문제를 고치는 방법은 알고리즘이 아니라 문제 쪽에 있다. 측정을 더 다양한 조건에서 하거나, 파라미터를 다시 정의하거나(예: 서로 강하게 상관된 두 파라미터를 합과 차로 바꾸기), 정규화(regularization)로 문제 자체를 바꾸는 것이다. 정규화는 답을 바꾼다는 점을 기록해 둔다(7.3장의 ridge).

:::note[분야 예시]
**역문제**(inverse problem)는 대부분 조건이 나쁘다. 열전도에서 나중 시각의 온도 분포로 처음 분포를 복원하거나, 흐린 현미경 영상에서 선명한 영상을 복원하거나, CT에서 투영 데이터로 단면을 재구성하는 문제가 그렇다. 앞방향 과정(확산, 흐림, 적분)은 세부 구조를 부드럽게 지우고, 그 결과 연산자의 특잇값이 0을 향해 빠르게 줄어든다. 역방향으로 풀면 작은 특잇값 방향의 측정 잡음이 $1/\sigma$배로 증폭된다. Tikhonov 정규화나 절단 SVD는 이 방향을 버리거나 누르는 방법이고, 그 대가로 해에 편향이 생긴다.
:::

## 알고리즘의 안정성

**왜 필요한가.** 조건이 좋은 문제도 알고리즘을 잘못 고르면 엉터리 답이 나온다. 알고리즘이 "할 만큼 했는지"를 판단할 기준이 필요하다. 그런데 참값을 모르는 상황에서 결과의 오차를 직접 잴 수는 없다.

**정의.** 문제 $f$와 알고리즘 $\hat{f}$에 대해 두 가지 오차를 정의한다.

- **forward error**: 출력의 오차 $\|\hat{f}(x) - f(x)\| / \|f(x)\|$. 우리가 궁극적으로 관심 있는 양이다.
- **backward error**: 계산 결과 $\hat{f}(x)$가 **조금 바뀐 입력에 대한 정확한 답**이라고 볼 때, 그 입력 변화의 최소 크기. 즉 $\hat{f}(x) = f(x + \Delta x)$를 만족하는 가장 작은 $\|\Delta x\| / \|x\|$.

backward error가 항상 $u$의 작은 배수인 알고리즘을 **backward stable**하다고 한다. "입력을 저장하는 순간 생기는 오차만큼만 틀린 답"을 내는 알고리즘이므로, 알고리즘에 기대할 수 있는 최선이다. 두 오차는 조건수로 연결된다.

$$
\text{forward error} \;\lesssim\; \kappa \times \text{backward error}.
$$

이 부등식이 이 장 전체의 요약이다. 오른쪽의 $\kappa$는 문제가, backward error는 알고리즘이 책임진다. backward stable한 알고리즘의 forward error는 대략 $\kappa u$이고, 이보다 나은 답은 일반적으로 기대할 수 없다.

선형시스템에서 backward error는 계산할 수 있다는 점이 중요하다. 정규화한 잔차 $\|\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}\| / (\|\mathbf{A}\|\,\|\hat{\mathbf{x}}\|)$가 바로 그것이다(정확히는 $\hat{\mathbf{x}}$를 정확한 해로 만드는 $\mathbf{A}$의 최소 섭동 크기와 같다). 참값을 몰라도 알고리즘이 제 몫을 했는지 확인할 수 있다.

**코드.** Hilbert 시스템을 두 방법으로 푼다. `np.linalg.solve`는 부분 피벗팅 LU 분해(9.3장)를 쓰고, 다른 하나는 역행렬을 만든 뒤 곱한다. 수학적으로 두 방법은 같은 답을 낸다.

```python {exec="browser" session="cond"}
eps = np.finfo(float).eps
for n in [5, 8, 11]:
    A = hilbert(n)
    x = np.ones(n)
    b = A @ x
    kappa = np.linalg.cond(A)
    print(f"n = {n:2d}  cond*eps = {kappa * eps:.1e}")
    for name, x_hat in [("solve", np.linalg.solve(A, b)),    # LU + 부분 피벗팅
                        ("inv @ b", np.linalg.inv(A) @ b)]:   # 역행렬을 만든 뒤 곱
        fwd = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
        bwd = np.linalg.norm(b - A @ x_hat) / (np.linalg.norm(A, 2) * np.linalg.norm(x_hat))
        # 오차의 정확한 값은 LAPACK 구현마다 다르므로, 크기 관계만 판정한다
        print(f"   {name:8s} backward <= 10 eps: {bwd <= 10 * eps!s:5}  "
              f"forward <= cond*backward: {fwd <= kappa * bwd!s:5}  forward < 1e-5: {fwd < 1e-5}")
```

```output
n =  5  cond*eps = 1.1e-10
   solve    backward <= 10 eps: True   forward <= cond*backward: True   forward < 1e-5: True
   inv @ b  backward <= 10 eps: False  forward <= cond*backward: True   forward < 1e-5: True
n =  8  cond*eps = 3.4e-06
   solve    backward <= 10 eps: True   forward <= cond*backward: True   forward < 1e-5: True
   inv @ b  backward <= 10 eps: False  forward <= cond*backward: True   forward < 1e-5: True
n = 11  cond*eps = 1.2e-01
   solve    backward <= 10 eps: True   forward <= cond*backward: True   forward < 1e-5: False
   inv @ b  backward <= 10 eps: False  forward <= cond*backward: True   forward < 1e-5: False
```

세 가지를 읽는다.

- **`solve`의 backward error는 조건수와 무관하게 $10\varepsilon$ 이하다.** backward stable하게 동작했다. forward error는 $\kappa \times$ backward error 아래에 있고, 따라서 $\kappa\varepsilon$ 수준을 넘지 않는다. $n = 11$에서 해가 다섯 자리도 맞지 않는 것(집필 환경에서 상대오차 $10^{-3}$ 안팎)은 알고리즘이 아니라 문제 탓이다.
- **역행렬을 거치면 backward error가 조건수와 함께 커진다.** 역행렬을 곱하는 방법은 backward stable하지 않다. 집필 환경에서 그 backward error는 $n = 5$에서 $10^{-13}$–$10^{-12}$, $n = 11$에서 $4 \times 10^{-4}$ 안팎이었고, forward error도 `solve`보다 수 배에서 100배 가까이 컸다. 이런 반올림 오차의 정확한 값은 LAPACK 구현과 플랫폼마다 달라서 셀은 크기 관계만 판정해 찍는다. 역행렬을 명시적으로 만들지 말라는 조언(9.3장)의 근거다.
- **작은 잔차가 정확한 해를 뜻하지는 않는다.** $n = 11$에서 `solve`의 잔차는 $10\varepsilon$ 이하인데 해의 상대오차는 $10^{-5}$를 넘는다. 잔차는 알고리즘을 평가하고, 해의 정확도는 잔차에 조건수를 곱해야 알 수 있다.

::quiz{id="cond-or-algorithm"}

**조건이 좋은 문제, 불안정한 알고리즘.** 반대 경우도 있다. $x^2 - 10^8 x + 1 = 0$의 작은 근은 약 $10^{-8}$이고, 계수를 상대적으로 조금 바꾸면 근도 상대적으로 그만큼만 변한다. 조건수가 약 1인 좋은 문제다. 그런데 근의 공식 $(-b - \sqrt{b^2 - 4c})/2$는 거의 같은 두 수 $10^8$과 $\sqrt{10^{16} - 4}$를 뺀다. 이 상쇄는 **알고리즘이 스스로 만든** 조건 나쁜 중간 단계다. 큰 근은 부호가 같은 두 수의 합이라 상쇄가 없으므로 먼저 구하고, 두 근의 곱이 $c$라는 Vieta 관계로 작은 근을 구한다.

```python {exec="browser" session="cond"}
b, c = -1e8, 1.0                                     # x^2 + b x + c = 0
d = np.sqrt(b * b - 4 * c)
naive = (-b - d) / 2                                 # 근의 공식 그대로: 상쇄
q = -(b - d) / 2                                     # 부호가 같은 두 수의 합
stable = c / q                                       # Vieta: x1 * x2 = c
exact = 1e-8                                         # 참값 1e-8 + 1e-24 + ...의 float64 값
for name, r in [("근의 공식", naive), ("Vieta 이용", stable)]:
    print(f"{name:8s} 작은 근 = {r:.16e}  상대 오차 = {abs(r - exact) / exact:.1e}")
```

```output
근의 공식    작은 근 = 7.4505805969238281e-09  상대 오차 = 2.5e-01
Vieta 이용 작은 근 = 1.0000000000000000e-08  상대 오차 = 0.0e+00
```

같은 문제에서 한 알고리즘은 25% 틀리고 다른 알고리즘은 모든 자릿수가 맞는다. 문제의 조건수가 1이므로 25%의 오차는 전부 알고리즘의 책임이다.

::quiz{id="quadratic-small-root"}

**연구에서의 의미.** 수치 결과를 의심할 때의 순서는 이렇다. (1) 문제의 조건수를 추정한다(`np.linalg.cond`, 또는 입력을 조금씩 흔들어 출력의 변화를 본다). (2) backward error나 잔차를 계산한다. (3) 잔차가 $u$ 수준인데 결과가 나쁘면 문제를 다시 세우고, 잔차가 크면 알고리즘을 바꾼다. 표준 라이브러리의 핵심 알고리즘인 Householder QR, SVD, Cholesky(대칭 양의 정부호 행렬), 부분 피벗팅 LU(실제로 쓰이는 거의 모든 행렬에서)는 backward stable하다. 정규방정식, 명시적 역행렬, classical Gram–Schmidt, 부호가 바뀌는 급수의 직접 합산은 조건이 나쁜 중간 단계를 만들거나 backward stable하지 않아 주의해야 한다.

## 수렴

**왜 필요한가.** 비선형 방정식(9.4장), 큰 선형시스템(9.3장), 최적화(10장)는 반복법으로 푼다. 반복법은 답에 점점 다가가는 수열 $x_0, x_1, \dots$을 만든다. 몇 번 반복해야 하는지, 언제 멈춰야 하는지, 왜 어떤 문제에서 유난히 느린지를 판단하려면 수렴 속도를 정량적으로 말할 수 있어야 한다.

**정의.** 오차를 $e_k = \|x_k - x^*\|$라 하자.

- **선형 수렴**(linear convergence): $e_{k+1} \le \rho\, e_k$, $0 < \rho < 1$. 매 반복 오차가 일정한 비율 $\rho$로 줄어 맞는 자릿수가 **일정하게** 늘어난다. 오차를 $10^{-d}$배로 줄이는 데 약 $d / \log_{10}(1/\rho)$번이 든다.
- **수렴 차수 $p$**(order of convergence): $e_{k+1} \approx C\, e_k^{\,p}$. $p = 2$이면 **2차 수렴**으로, 맞는 자릿수가 매 반복 약 **두 배**가 된다. $1 < p < 2$는 초선형(superlinear) 수렴이다.

| 방법 | 수렴 | 다루는 곳 |
|---|---|---|
| 이분법 | 선형, $\rho = 1/2$ | 9.4장 |
| 고정 step gradient descent | 선형, $\rho = (\kappa - 1)/(\kappa + 1)$ (이차 함수) | 이 절, 10장 |
| secant | $p = (1 + \sqrt{5})/2 \approx 1.618$ | 9.4장 |
| Newton | $p = 2$ (단순근 근처) | 9.4장 |

**직관.** 선형 수렴과 2차 수렴의 차이는 극적이다. $\rho = 0.5$인 선형 수렴은 오차를 $10^{-16}$까지 줄이는 데 약 53번이 걸리지만, 2차 수렴은 오차 $10^{-1}$에서 시작해 $10^{-2}, 10^{-4}, 10^{-8}, 10^{-16}$으로 네 번이면 된다. 반대로 선형 수렴의 $\rho$가 1에 가까우면 반복이 끝없이 길어진다. 조건수가 바로 그 $\rho$를 정한다.

**수식과 코드.** 이차 함수 $f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x}$($\mathbf{A}$는 대칭 양의 정부호, 고유값 $\mu = \lambda_{\min}$, $L = \lambda_{\max}$)를 step $\alpha$의 gradient descent $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha\mathbf{A}\mathbf{x}_k$로 최소화하면, 고유벡터 방향마다 오차가 $|1 - \alpha\lambda|$배로 줄어든다. 가장 느린 방향의 축소율을 최소화하는 step은 $\alpha = 2/(\mu + L)$이고, 그때

$$
\rho = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = L/\mu
$$

이다. $\kappa$가 크면 $\rho \approx 1 - 2/\kappa$이므로 필요한 반복 수는 대략 $\tfrac{\kappa}{2}\ln(1/\text{tol})$, **조건수에 비례**한다.

```python {exec="browser" session="cond"}
def gd_iters(A, tol=1e-8, max_iter=100_000):
    """f(x) = x^T A x / 2를 최적 고정 step의 gradient descent로 최소화한 반복 수."""
    lam = np.linalg.eigvalsh(A)
    step = 2 / (lam[0] + lam[-1])
    x = np.ones(A.shape[0])                          # 최솟값은 x* = 0
    for k in range(max_iter):
        if np.linalg.norm(x) < tol:
            return k
        x = x - step * (A @ x)
    return max_iter

for kappa in [10, 100, 1000]:
    A = np.diag([1.0, kappa])                        # (2, 2), 조건수 kappa
    rate = (kappa - 1) / (kappa + 1)                 # 한 반복당 오차 축소율
    print(f"kappa = {kappa:5d}  반복 수 = {gd_iters(A):5d}  "
          f"예측 = {np.log(1e-8 / np.sqrt(2)) / np.log(rate):7.1f}")
```

```output
kappa =    10  반복 수 =    94  예측 =    93.5
kappa =   100  반복 수 =   939  예측 =   938.3
kappa =  1000  반복 수 =  9384  예측 =  9383.6
```

조건수가 10배 커질 때마다 반복 수도 정확히 10배가 된다. 예측값은 시작점의 노름 $\sqrt{2}$를 $10^{-8}$까지 줄이는 데 필요한 $k$를 $\rho^k\sqrt{2} = 10^{-8}$에서 푼 것이다. 여기서는 두 고유 방향의 축소율이 모두 정확히 $\rho$라 예측이 거의 정확하다. 이 장 앞부분에서 조건수가 정확도를 정했다면, 여기서는 같은 조건수가 **속도**를 정한다.

::quiz{id="order-from-errors"}

**정지 조건.** 반복법은 참값을 모르므로 다음 중 하나 이상으로 멈춘다.

- **step 크기**: $\|x_{k+1} - x_k\| \le \text{atol} + \text{rtol}\,\|x_k\|$. 절대 허용 오차와 상대 허용 오차를 함께 쓰는 것은 9.1장의 `np.allclose`와 같은 이유다.
- **잔차**: 방정식 $F(x) = 0$이면 $\|F(x_k)\| \le \text{tol}$, 선형시스템이면 $\|\mathbf{b} - \mathbf{A}\mathbf{x}_k\| \le \text{tol}\,\|\mathbf{b}\|$.
- **최대 반복 수**: 수렴하지 않는 경우를 대비한 안전장치. 최대 반복 수에 도달해 멈췄다면 그 사실을 반드시 기록하고 결과를 쓰지 않는다.

두 가지 함정이 있다. 첫째, **선형 수렴에서 step이 작다고 답에 가까운 것은 아니다.** $e_{k+1} \approx \rho e_k$이면 남은 오차는 대략 $\|x_{k+1} - x_k\| \cdot \rho/(1 - \rho)$로, $\rho = 0.999$이면 step의 약 1000배다. 느리게 수렴하는 반복이 "더 이상 변하지 않는다"고 멈춘 지점은 답에서 멀 수 있다. 둘째, **허용 오차를 $\kappa u$보다 작게 잡으면 영원히 멈추지 않을 수 있다.** 반올림 오차 때문에 그보다 정확한 답은 존재하지 않는다.

::quiz{id="gd-iterations-kappa"}

## 스케일링과 전처리

**왜 필요한가.** 조건이 나쁜 문제 중 상당수는 본질적으로 어려운 것이 아니라 **변수의 단위를 잘못 고른** 것이다. 길이를 m와 μm로, 농도를 mol/L와 nmol/L로 섞어 쓰면, 물리적으로는 아무 문제가 없어도 행렬의 열이나 손실 함수의 곡률이 수백만 배씩 차이 나서 조건수가 커진다. 이런 인위적인 나쁜 조건은 변수를 다시 재는 것만으로 없앨 수 있다.

**정의.** **스케일링**은 대각행렬 $\mathbf{D}$로 변수를 $\mathbf{x} = \mathbf{D}\mathbf{y}$처럼 바꾸는 것이다. 더 일반적으로 **전처리**(preconditioning)는 $\mathbf{A}$를 근사하면서 역을 적용하기 쉬운 행렬 $\mathbf{M}$을 골라, 원래 시스템 대신

$$
\mathbf{M}^{-1}\mathbf{A}\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}
$$

를 푸는 것이다. 해는 같고, $\kappa(\mathbf{M}^{-1}\mathbf{A})$가 $\kappa(\mathbf{A})$보다 훨씬 작으면 반복법이 빨라진다. $\mathbf{M} = \mathbf{A}$이면 조건수가 1이지만 $\mathbf{M}^{-1}$을 적용하는 것이 원래 문제만큼 어렵다. 좋은 전처리는 그 사이의 타협이다. 가장 단순한 것이 대각 원소만 쓰는 **Jacobi 전처리** $\mathbf{M} = \operatorname{diag}(\mathbf{A})$이고, 대칭성을 지키려면 $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$($\mathbf{D} = \operatorname{diag}(\mathbf{A})$)로 양쪽에서 스케일링한다. 불완전 Cholesky·LU, multigrid 같은 강력한 전처리는 9.3장과 9.7장에서 이름이 다시 나온다.

**코드.** 조건이 좋은 $2 \times 2$ 행렬 $\mathbf{S}$를 두 번째 변수의 단위만 100배 바꿔 $\mathbf{A} = \mathbf{D}\mathbf{S}\mathbf{D}$로 만든다. 물리적으로는 같은 문제다. 앞 절의 `gd_iters`를 다시 쓴다.

```python {exec="browser" session="cond"}
S = np.array([[2.0, 0.5],
              [0.5, 1.0]])                           # 조건이 좋은 SPD 행렬
D = np.diag([1.0, 100.0])                            # 두 변수의 단위가 100배 다름
A = D @ S @ D                                        # 같은 문제, 나쁜 스케일
d = 1 / np.sqrt(np.diag(A))
A_scaled = A * np.outer(d, d)                        # D^-1/2 A D^-1/2 (Jacobi 스케일링)
for name, M in [("원래 A", A), ("스케일 후", A_scaled)]:
    print(f"{name:6s} cond = {np.linalg.cond(M):.1e}  GD 반복 수 = {gd_iters(M)}")
```

```output
원래 A   cond = 5.7e+03  GD 반복 수 = 53624
스케일 후  cond = 2.1e+00  GD 반복 수 = 19
```

단위를 맞추기만 했는데 조건수가 5700에서 2로, 반복 수가 5만 번에서 19번으로 줄었다. 대각 스케일링은 계산 비용이 거의 없으므로, 반복법을 쓰기 전에 항상 먼저 해 볼 일이다.

::quiz{id="scaling-changes-kappa"}

**연구에서의 의미.** 머신러닝의 여러 관행이 같은 생각에서 나온다. 입력 특징을 표준화하는 것은 손실 함수의 곡률을 방향마다 비슷하게 맞추는 스케일링이다. Adam 같은 적응적 optimizer(10.4장)가 파라미터마다 다른 step을 쓰는 것은 gradient의 크기 통계로 추정한 대각 전처리로 해석할 수 있다. 물리 문제에서는 **비차원화**(nondimensionalization)가 같은 역할을 한다. 변수를 특성 길이·시간·농도로 나누어 크기가 1 근처인 무차원량으로 바꾸면, 방정식의 조건이 좋아지고 어떤 항이 지배적인지도 드러난다. 물리 기반 신경망에서 손실 항의 크기가 수천 배 차이 나 학습이 한쪽 항만 줄이는 문제도 대개 스케일링부터 점검한다(24장).

:::note[분야 예시]
화학 반응 속도론에서는 농도가 $1$ mol/L인 반응물과 $10^{-9}$ mol/L인 라디칼 중간체가 한 상태 벡터에 함께 들어간다. 배터리 모델의 파라미터에는 전극 두께(약 $10^{-4}$ m), 확산 계수(약 $10^{-14}$ m²/s), 전류(수 A)가 섞인다. 이런 문제를 원래 단위 그대로 최적화하거나 Jacobian으로 선형화하면 조건수가 수십 자릿수에 이르기 쉽다. 각 변수를 대표값으로 나누어 1 근처로 맞춘 뒤 풀고, 결과를 다시 원래 단위로 되돌리는 것이 표준 절차다. 많은 ODE solver와 최적화 라이브러리가 변수별 스케일(예: 성분별 허용 오차)을 받는 이유이기도 하다(9.6장).
:::

## 실습

수학적으로 동등한 계산식이 다른 결과를 내는 세 사례를 만들고, 원인이 문제의 조건인지 알고리즘인지 판정한다. 코드는 `labs/numerical-methods/conditioning-and-stability/equivalent_formulas.py`다. 각 사례에서 문제의 조건수를 따로 계산해 $\kappa\varepsilon$를 기준선으로 두고, 두 계산식의 상대 오차를 비교한다.

1. $e^{-x}$를 Taylor 급수 $\sum_k (-x)^k/k!$로 그대로 더하는 방법과, 양수 항만 있는 $\sum_k x^k/k!$을 더한 뒤 역수를 취하는 방법. 이 문제의 조건수는 $|x|$다.
2. $x^2 - Bx + 1 = 0$의 작은 근을 근의 공식으로 구하는 방법과 Vieta 관계로 구하는 방법. 참값은 50자리 십진 연산으로 계산한다.
3. 잡음 없는 다항식 최소제곱을 정규방정식·QR·SVD로 푸는 방법(7.4장 실습의 연장). 잔차가 0인 최소제곱 문제의 조건수는 $\kappa(\mathbf{X})$다.

집필 시점의 실행 결과는 다음과 같다.

```output
[1] e^{-x}를 Taylor 급수로: 문제의 조건수 = |x|
     x     cond  cond*eps      교대 급수      1/급수
     1        1   2.2e-16    3.0e-16   1.5e-16
     5        5   1.1e-15    1.2e-13   0.0e+00
    10       10   2.2e-15    2.0e-09   0.0e+00
    20       20   4.4e-15    2.0e+00   4.0e-16
    30       30   6.7e-15    6.5e+07   2.7e-16

[2] x^2 - B x + 1 = 0의 작은 근: 계수에 대한 조건수 ≈ 1
     B     cond  cond*eps      근의 공식     Vieta
 1e+02        1   2.2e-16    1.2e-13   0.0e+00
 1e+04        1   2.2e-16    1.1e-09   0.0e+00
 1e+06        1   2.2e-16    7.6e-06   0.0e+00
 1e+08        1   2.2e-16    2.5e-01   0.0e+00

[3] 다항식 최소제곱(잡음 없음): 문제의 조건수 ≈ cond(X)
  차수   cond(X)  cond*eps cond^2*eps      정규방정식        QR       SVD
   4   6.4e+02   1.4e-13    9.2e-11    1.0e-11   2.6e-14   1.4e-14
   8   6.3e+05   1.4e-10    8.7e-05    3.7e-06   1.8e-11   6.9e-12
  12   6.8e+08   1.5e-07    1.0e+02    1.1e+01   8.4e-09   1.6e-08
```

판정은 각 오차를 $\kappa\varepsilon$와 비교해서 한다.

- **사례 1과 2는 알고리즘 탓이다.** 문제의 조건수는 30 이하인데 교대 급수와 근의 공식의 오차는 $\kappa\varepsilon$보다 수십 자릿수 크다. 교대 급수는 $x = 30$에서 크기가 $10^{12}$에 이르는 항들을 더하고 빼서 $10^{-13}$의 답을 만들려 하므로, 가장 큰 항의 반올림 오차만으로도 답이 묻힌다. 다른 식은 같은 조건에서 $\varepsilon$ 수준이다.
- **사례 3의 정규방정식도 알고리즘 탓이다.** 정규방정식의 오차는 $\kappa^2\varepsilon$를 따라가고, QR과 SVD는 $\kappa\varepsilon$를 따라간다. $\mathbf{X}^\top\mathbf{X}$를 만드는 순간 조건수가 제곱된 **새 문제**를 풀기 때문이다. 원래 문제의 조건수만큼의 오차는 피할 수 없지만 그 제곱은 피할 수 있다.
- **사례 3의 QR·SVD에 남은 오차는 문제 탓이다.** 차수 12에서 QR도 $10^{-8}$의 오차를 낸다. 이것은 $\kappa\varepsilon$와 같은 수준이라 어떤 backward stable 알고리즘으로도 더 줄일 수 없다. 더 정확한 계수가 필요하면 기저를 바꾸는(예: 직교다항식, 9.5장) 등 문제를 다시 세워야 한다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="stable-quadratic-roots"}

::exercise{id="convergence-order"}

## 실패 사례

- **나쁜 조건의 문제를 알고리즘 탓으로 돌리는 것.** 해가 불안정하다고 solver를 LU에서 QR로, QR에서 SVD로 바꾸며 시간을 쓰지만 결과는 그대로다. 잔차(backward error)를 계산해 $u$ 수준이면 알고리즘은 제 몫을 한 것이다. 조건수를 보고, 측정 설계·파라미터화·정규화 같은 문제 쪽의 해결책을 찾는다.
- **반대로, 불안정한 알고리즘의 오차를 문제나 데이터 탓으로 돌리는 것.** 정규방정식으로 푼 회귀 계수가 이상하면 "데이터가 나쁘다"고 결론 내리기 쉽다. 같은 문제를 QR이나 `lstsq`로 다시 풀어 결과가 크게 바뀌면 알고리즘 문제다.
- **잔차가 작으니 해가 정확하다고 결론 내리는 것.** 잔차는 backward error이고, 해의 오차는 여기에 조건수를 곱한 만큼까지 클 수 있다. 조건수를 함께 보고한다.
- **정지 조건을 조건수와 무관하게 정하는 것.** 허용 오차를 $\kappa u$보다 작게 잡으면 반복이 끝나지 않고, 선형 수렴이 느린 문제에서 step 크기만으로 멈추면 답에서 먼 곳에서 멈춘다. 최대 반복 수에 도달했는지와 최종 잔차를 함께 기록한다.

## 통과 기준

- 주어진 수치 문제에서 조건수와 backward error(잔차)를 계산하고, forward error $\lesssim \kappa \times$ backward error 관계로 오차가 문제의 조건 때문인지 알고리즘 때문인지 판별해 근거를 적는다.
- 반복법의 오차 수열에서 수렴 차수와 선형 수렴의 축소율을 추정하고, 조건수와 반복 수의 관계를 설명한다.
- 연습문제 두 개(`stable-quadratic-roots`, `convergence-order`)의 테스트를 통과한다.

## 참고 문헌

- Trefethen, Bau (1997). *Numerical Linear Algebra*. SIAM. — 조건수, forward·backward stability의 정의와 선형대수 알고리즘의 안정성
- Higham (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. — backward error 분석, 선형시스템의 backward error 공식
- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — gradient descent의 수렴률과 조건수, 수렴 차수의 정의
