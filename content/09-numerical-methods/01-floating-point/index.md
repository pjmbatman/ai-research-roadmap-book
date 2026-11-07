---
id: numerical-methods/floating-point
title: "부동소수점과 수치 오차"
title_en: "Floating Point and Numerical Error"
tier: core
status: draft
prerequisites: [python-and-computing/computer-architecture]
---

# 부동소수점과 수치 오차

컴퓨터는 실수를 다루지 못한다. 다루는 것은 유한한 비트로 표현할 수 있는 수의 유한 집합이고, 모든 연산 결과는 그 집합의 가장 가까운 원소로 반올림된다. 대부분의 계산에서 이 차이는 보이지 않지만, 학습 도중 loss가 갑자기 `nan`이 되거나, 같은 코드가 GPU에서 매번 조금씩 다른 값을 내거나, float16으로 바꾸자마자 학습이 멈추는 순간에는 이 차이가 전부다.

이 장은 부동소수점 수가 어떻게 생겼는지, 오차가 어디서 생기고 어떻게 커지는지, 그리고 수학적으로 같은 식 중에서 수치적으로 안전한 식을 고르는 법을 다룬다. 다음 장(9.2)의 조건수와 안정성, 그 뒤의 모든 수치 알고리즘이 이 장의 언어 위에서 설명된다.

## 학습 목표

- 부동소수점 표현과 정밀도를 이해한다
- 수치적으로 안정한 계산식을 작성한다

## 부동소수점 표현

**왜 필요한가.** "float32는 소수점 아래 일곱 자리까지 정확하다"는 말은 반만 맞다. 정확한 것은 **유효숫자**의 개수이고, 소수점의 위치는 수의 크기에 따라 움직인다. 이 구조를 알아야 어떤 계산이 어디서 정밀도를 잃는지, 어떤 형식을 어디에 써도 되는지 판단할 수 있다.

**정의.** IEEE 754 표준의 이진 부동소수점(binary floating point) 수는 부호 비트 $s$, 지수(exponent) $e$, 가수(significand, mantissa) $f$의 세 칸으로 이루어지고, 정규수(normal number)의 값은

$$
x = (-1)^s \times (1.f)_2 \times 2^{\,e - \text{bias}}
$$

이다. $(1.f)_2$는 이진수 $1.f_1f_2\cdots f_p$이고, 맨 앞의 1은 저장하지 않는다(숨은 비트). 가수가 $p$비트이면 유효숫자는 $p+1$비트다. bias는 지수를 부호 없이 저장하기 위한 상수로, 지수 비트가 $k$개면 $2^{k-1}-1$이다. 지수 칸이 모두 0이면 0과 **subnormal 수**(정규수보다 작은 수를 정밀도를 줄여 가며 표현), 모두 1이면 무한대(`inf`)와 `NaN`(not a number)을 나타낸다.

**직관.** 부동소수점 수는 과학적 표기법 $1.234 \times 10^{5}$의 이진판이다. 각 2의 거듭제곱 구간 $[2^E, 2^{E+1})$ 안에 같은 개수($2^p$개)의 수가 **등간격**으로 놓인다. 그래서 이웃한 두 수의 간격, 즉 **ulp**(unit in the last place)는 수의 크기에 비례한다. 1 근처에서 촘촘하던 수들이 $10^{6}$ 근처에서는 듬성듬성해진다. 반면 **상대** 간격은 어느 구간에서나 거의 같다.

이 상대 간격을 대표하는 수가 **machine epsilon** $\varepsilon = 2^{-p}$, 즉 1과 1 다음 수의 간격이다. 실수 $x$를 가장 가까운 부동소수점 수 $\mathrm{fl}(x)$로 반올림하면 정규수 범위에서

$$
\mathrm{fl}(x) = x(1 + \delta), \qquad |\delta| \le u = \varepsilon / 2
$$

가 성립한다. $u$를 unit roundoff라 한다(책에 따라 $\varepsilon$ 자체를 unit roundoff라 부르기도 하니 정의를 확인한다). 덧셈·곱셈 같은 기본 연산도 "정확한 결과를 계산한 뒤 반올림한 것"과 같도록 표준이 정한다. 즉 $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$, $|\delta| \le u$다. 이 한 줄이 수치해석에서 오차를 분석하는 출발점이다.

아래 위젯에서 수를 고르고 float32·float16·bfloat16의 비트를 비교한다. 0.1이 어느 형식에서도 정확히 표현되지 않는 것, 65504를 넘으면 float16이 `inf`가 되는 것, bfloat16의 비트 배치가 float32의 앞 16비트와 같은 것을 확인한다.

::widget{name="float-precision"}

같은 내용을 NumPy로 확인한다.

```python {exec="browser" session="fp"}
import numpy as np

for dt in [np.float64, np.float32, np.float16]:
    fi = np.finfo(dt)
    print(f"{fi.dtype.name:8s} 가수 {fi.nmant:2d}비트  eps = {fi.eps:.2e}  "
          f"최대 = {fi.max:.2e}  최소 정규수 = {fi.smallest_normal:.2e}")

print(f"float32(0.1)의 저장값 = {float(np.float32(0.1)):.20f}")
print(f"float64(0.1)의 저장값 = {0.1:.20f}")
print("0.1 + 0.2 == 0.3 :", 0.1 + 0.2 == 0.3)
for v in [1.0, 1024.0, 2.0**24]:
    print(f"x = {v:>10.0f}  float32의 ulp = {np.spacing(np.float32(v)):.3e}")
print("float32(2**24 + 1) =", int(np.float32(2**24 + 1)))
```

```output
float64  가수 52비트  eps = 2.22e-16  최대 = 1.80e+308  최소 정규수 = 2.23e-308
float32  가수 23비트  eps = 1.19e-07  최대 = 3.40e+38  최소 정규수 = 1.18e-38
float16  가수 10비트  eps = 9.77e-04  최대 = 6.55e+04  최소 정규수 = 6.10e-05
float32(0.1)의 저장값 = 0.10000000149011611938
float64(0.1)의 저장값 = 0.10000000000000000555
0.1 + 0.2 == 0.3 : False
x =          1  float32의 ulp = 1.192e-07
x =       1024  float32의 ulp = 1.221e-04
x =   16777216  float32의 ulp = 2.000e+00
float32(2**24 + 1) = 16777216
```

0.1은 이진수로 $0.0\overline{0011}$의 무한소수라 어떤 이진 형식으로도 정확히 저장되지 않는다. 그래서 `0.1 + 0.2 == 0.3`이 거짓이다. float32의 ulp는 1에서 $1.2 \times 10^{-7}$, 1024에서 $1.2 \times 10^{-4}$, $2^{24}$에서 2다. $2^{24}$부터는 이웃한 수의 간격이 2라서 홀수 정수 $2^{24}+1$을 표현할 수 없다.

::quiz{id="machine-epsilon-meaning"}

**딥러닝에서 쓰는 형식.** 정밀도를 낮추면 메모리와 대역폭이 줄고 행렬 곱 하드웨어(tensor core 등)의 처리량이 커진다. 그래서 학습·추론에서 여러 형식을 섞어 쓴다. 아래 표의 값은 비트 배치에서 계산한 것이다. fp8은 집필 시점에 널리 쓰이는 두 변형(E4M3, E5M2)을 예로 들었고, 하드웨어와 라이브러리마다 특수값 처리 등 세부가 다를 수 있으니 쓰는 환경의 문서를 확인한다.

| 형식 | 부호·지수·가수 비트 | $\varepsilon$ | 최댓값 | 최소 정규수 | 십진 유효숫자(대략) |
|---|---|---|---|---|---|
| fp64 | 1 · 11 · 52 | $2.2 \times 10^{-16}$ | $1.8 \times 10^{308}$ | $2.2 \times 10^{-308}$ | 15–16 |
| fp32 | 1 · 8 · 23 | $1.2 \times 10^{-7}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | 7 |
| tf32 | 1 · 8 · 10 | $9.8 \times 10^{-4}$ | fp32와 비슷 | fp32와 같음 | 3 |
| fp16 | 1 · 5 · 10 | $9.8 \times 10^{-4}$ | 65504 | $6.1 \times 10^{-5}$ | 3 |
| bf16 | 1 · 8 · 7 | $7.8 \times 10^{-3}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | 2–3 |
| fp8 E4M3 | 1 · 4 · 3 | $0.125$ | 448 | $1.6 \times 10^{-2}$ | 1 |
| fp8 E5M2 | 1 · 5 · 2 | $0.25$ | 57344 | $6.1 \times 10^{-5}$ | 1 미만 |

표에서 두 가지를 읽는다. 첫째, **지수 비트는 범위를, 가수 비트는 정밀도를 정한다.** bf16은 fp32와 지수 비트가 같아 범위가 같고, 정밀도를 내주었다. fp16은 반대로 정밀도가 bf16보다 높지만 범위가 좁다. 둘째, tf32는 저장 형식이라기보다 연산 형식이다. fp32 텐서를 입력으로 받되 행렬 곱 내부에서 가수를 10비트로 줄여 계산하는 모드로, 쓰는 프레임워크에서 켜져 있는지 확인해야 한다(PyTorch의 관련 설정 이름과 기본값은 버전마다 바뀌어 왔으니 집필 시점의 문서를 확인한다).

## 오차의 종류

**왜 필요한가.** 결과가 기대와 다를 때, 오차가 표현의 한계에서 왔는지, 알고리즘이 수학적 근사를 했기 때문인지 구분해야 고칠 방법이 정해진다. 정밀도를 올려 해결되는 문제가 있고, 정밀도를 아무리 올려도 해결되지 않는 문제가 있다.

**정의.** 참값 $x$의 근삿값을 $\hat{x}$라 할 때

$$
\text{절대 오차} = |\hat{x} - x|, \qquad \text{상대 오차} = \frac{|\hat{x} - x|}{|x|}.
$$

상대 오차가 $10^{-k}$이면 대략 유효숫자 $k$자리가 맞는다. 부동소수점의 정밀도는 상대 오차로 정해지므로, 크기가 크게 다른 양을 비교할 때는 상대 오차를 본다. 다만 참값이 0 근처이면 상대 오차는 의미를 잃으므로 절대 오차를 함께 본다(`np.allclose`가 `rtol`과 `atol`을 모두 받는 이유다).

오차의 출처는 세 가지로 나눈다.

- **반올림 오차**(rounding error). 저장과 연산마다 생기는 $u$ 크기의 상대 오차. 정밀도를 높이면 줄어든다.
- **절단 오차**(truncation error). 무한 과정을 유한하게 끊어서 생기는 오차. Taylor 급수를 몇 항에서 자르거나, 도함수를 유한한 $h$의 차분으로 바꾸거나, 반복법을 몇 번 만에 멈추는 것. 정밀도와 무관하고, 알고리즘의 파라미터($h$, 항 수, 반복 횟수)로 줄인다.
- **누적 오차**(accumulated error). 작은 오차가 수많은 연산을 거치며 쌓이거나 증폭된 것. 합, 시간 적분, 긴 반복에서 나타난다.

**코드.** 두 오차가 반대 방향으로 움직이는 대표적인 예가 전진 차분 $f'(x) \approx (f(x+h) - f(x))/h$이다. Taylor 전개에서 절단 오차는 대략 $\tfrac{h}{2}|f''(x)|$로 $h$에 비례해 줄지만, $f(x+h)$와 $f(x)$의 반올림 오차 $u|f(x)|$가 $h$로 나뉘어 대략 $u|f(x)|/h$로 커진다. 두 항의 합은 $h \approx \sqrt{u}$, float64에서 $10^{-8}$ 근처에서 가장 작다.

```python {exec="browser" session="fp"}
x0 = 1.0
exact = np.cos(x0)                      # d/dx sin(x)
for h in [1e-1, 1e-4, 1e-8, 1e-12]:
    approx = (np.sin(x0 + h) - np.sin(x0)) / h
    print(f"h = {h:.0e}  상대 오차 = {abs(approx - exact) / abs(exact):.2e}")
```

```output
h = 1e-01  상대 오차 = 7.95e-02
h = 1e-04  상대 오차 = 7.79e-05
h = 1e-08  상대 오차 = 5.50e-09
h = 1e-12  상대 오차 = 8.00e-05
```

$h$를 $10^{-1}$에서 $10^{-4}$로 줄이면 오차가 약 1000배 줄어 절단 오차가 $h$에 비례함을 보여 준다. 그러나 $h = 10^{-12}$에서는 오차가 오히려 $h = 10^{-4}$ 수준으로 돌아온다. 반올림 오차가 지배하기 때문이다. "$h$가 작을수록 정확하다"는 직관은 절단 오차만 볼 때의 이야기다. 8.1장에서 수치 미분으로 gradient를 검사할 때 $h$를 $10^{-5}$–$10^{-7}$ 근처로 고른 이유가 이것이고, 같은 계산을 float32로 하면 $\sqrt{u}$가 $10^{-4}$ 수준이라 최적 $h$도 커진다.

**연구에서의 의미.** 결과가 이상할 때 정밀도를 float64로 올려 본다. 결과가 바뀌면 반올림 오차가 원인이고, 바뀌지 않으면 절단 오차나 모델 자체의 문제다. 반대로 시뮬레이션의 시간 간격이나 격자를 줄여 결과가 바뀌는지 보는 것은 절단 오차를 점검하는 방법이다(9.6장, 9.7장).

## 위험한 연산

모든 연산이 똑같이 위험한 것은 아니다. 곱셈과 나눗셈은 상대 오차를 거의 그대로 전달한다. 위험한 것은 세 가지다.

**파국적 상쇄**(catastrophic cancellation). 거의 같은 두 수를 빼면 앞쪽의 같은 자릿수가 모두 지워지고, 원래 각 수에 묻어 있던 반올림 오차만 남는다. 뺄셈 자체는 정확하게 계산되지만, 입력에 이미 있던 작은 상대 오차가 결과에서는 거대한 상대 오차가 된다. $a(1+\delta_a) - b(1+\delta_b)$의 상대 오차는 대략

$$
\frac{|a\delta_a - b\delta_b|}{|a - b|} \le \frac{|a| + |b|}{|a - b|}\, u
$$

로, $a \approx b$이면 분모가 작아 $u$의 수많은 배가 된다. 예를 들어 $(1 - \cos x)/x^2$는 $x \to 0$에서 $1/2$로 가지만, 정의대로 계산하면 $\cos x$가 1에 너무 가까워 문제가 된다. 반각 공식 $1 - \cos x = 2\sin^2(x/2)$로 바꾸면 뺄셈이 사라진다.

```python {exec="browser" session="fp"}
for x in [1e-2, 1e-5, 1e-8]:
    naive = (1 - np.cos(x)) / x**2                 # 1에 가까운 두 수의 뺄셈
    stable = 2 * np.sin(x / 2) ** 2 / x**2         # 같은 식, 뺄셈 없음
    print(f"x = {x:.0e}  naive = {naive:.10f}  stable = {stable:.10f}")

grad = 1e-8                              # 작은 gradient 값
print("float16(1e-8)          =", np.float16(grad))           # underflow → 0
print("float16(1e-8 * 2**16)  =", np.float16(grad * 2**16))   # loss scaling
with np.errstate(over="ignore"):
    print("float16(300) ** 2      =", np.float16(300) ** 2)   # overflow → inf
```

```output
x = 1e-02  naive = 0.4999958333  stable = 0.4999958333
x = 1e-05  naive = 0.5000000414  stable = 0.5000000000
x = 1e-08  naive = 0.0000000000  stable = 0.5000000000
float16(1e-8)          = 0.0
float16(1e-8 * 2**16)  = 0.000655
float16(300) ** 2      = inf
```

$x = 10^{-8}$에서 $\cos x = 1 - 5 \times 10^{-17}$은 float64에서 정확히 1로 반올림되어, 정의대로 계산한 값은 0이 된다. 참값 $0.5$와 비교하면 상대 오차 100%다. 같은 수학을 다른 식으로 쓰자 모든 자릿수가 정확해졌다.

::quiz{id="cancellation-stable-form"}

**overflow와 underflow.** 결과의 크기가 최댓값을 넘으면 `inf`(overflow), 최소 정규수보다 작으면 subnormal을 거쳐 0(underflow)이 된다. `inf - inf`, `0 * inf`, `inf / inf`는 `NaN`이고, `NaN`은 이후의 모든 연산으로 번진다. 학습 중 갑자기 loss가 `nan`이 되는 대부분의 경우가 어딘가의 overflow가 번진 것이다. float16은 범위가 좁아 위험이 특히 크다. 최댓값이 65504라 300의 제곱만 해도 `inf`가 되고, 가장 작은 subnormal $2^{-24} \approx 6.0 \times 10^{-8}$의 절반보다 작은 값은 0이 된다. 혼합 정밀도 학습의 **loss scaling**은 loss에 $2^{16}$ 같은 큰 수를 곱해 backward의 gradient를 float16 범위 안으로 끌어올린 뒤, optimizer 단계에서 다시 나누는 기법이다(Micikevicius 외, 2018). 위 출력에서 $10^{-8}$은 그냥 저장하면 0이 되지만 $2^{16}$을 곱하면 살아남는다.

**비결합성.** 실수의 덧셈은 결합법칙 $(a+b)+c = a+(b+c)$를 만족하지만 부동소수점 덧셈은 그렇지 않다. 매 덧셈 뒤의 반올림이 중간 결과에 따라 다르기 때문이다. 결과적으로 **합의 순서가 결과를 바꾼다.** 수많은 수를 앞에서부터 차례로 더하면, 합이 커질수록 ulp가 커져 새로 더하는 작은 항의 아래 자릿수가 잘려 나간다. $n$개를 차례로 더할 때 오차의 상한은 대략 $n u \sum |x_i|$로 $n$에 비례한다.

이를 줄이는 방법은 두 가지다. **쌍별 합**(pairwise summation)은 배열을 반으로 나눠 각각 더한 뒤 합치는 것을 재귀적으로 반복해, 오차 상한을 $O(u \log n)$으로 줄인다. NumPy의 `np.sum`이 부동소수점 배열에 대해 쓰는 방식이다(내부 블록 단위로 적용된다). **Kahan 보정 합**(compensated summation)은 덧셈마다 잃은 아래 자릿수를 보정 변수 $c$에 모아 다음 덧셈에서 되돌려 주어, 오차 상한을 $n$과 거의 무관한 $O(u)$로 만든다(Kahan, 1965).

```python {exec="browser" session="fp"}
rng = np.random.default_rng(0)
vals = rng.uniform(0, 1, 100_000).astype(np.float32)        # (100000,)
exact = np.sum(vals.astype(np.float64))

def naive_sum(a):
    s = np.float32(0)
    for v in a:
        s += v
    return s

def kahan_sum(a):
    s, c = np.float32(0), np.float32(0)
    for v in a:
        y = v - c              # 지난번에 잃은 부분을 보정
        t = s + y
        c = (t - s) - y        # 이번 덧셈에서 잃은 부분
        s = t
    return s

for name, s in [("순서대로", naive_sum(vals)), ("Kahan", kahan_sum(vals)),
                ("np.sum(쌍별)", np.sum(vals))]:
    print(f"{name:12s} 상대 오차 = {abs(float(s) - exact) / exact:.1e}")
```

```output
순서대로         상대 오차 = 4.1e-06
Kahan        상대 오차 = 2.0e-08
np.sum(쌍별)   상대 오차 = 2.0e-08
```

float32 10만 개를 차례로 더하면 상대 오차가 $4 \times 10^{-6}$으로, float32의 $u \approx 6 \times 10^{-8}$보다 약 70배 크다. Kahan 합과 쌍별 합은 $u$ 수준을 유지한다. 보정 식 `c = (t - s) - y`는 대수적으로 정리하면 0이다. 수학적으로 0인 식이 부동소수점에서는 정보를 담는다는 것, 그래서 컴파일러가 "수학적으로 같은" 식으로 바꾸는 최적화(예: 일부 컴파일러의 fast-math 옵션)가 이런 알고리즘을 망가뜨릴 수 있다는 것이 이 예의 교훈이다.

:::caution[함정]
GPU의 병렬 합은 스레드가 끝나는 순서나 블록 분할에 따라 덧셈 순서가 달라질 수 있다. 그래서 같은 입력, 같은 시드로도 실행마다 마지막 몇 자리가 다를 수 있다. 이 차이가 학습의 수천 step을 거치며 커지면 두 실행의 결과가 눈에 띄게 갈라진다. 재현성이 중요한 실험에서는 프레임워크의 결정적(deterministic) 연산 설정을 켜고, 그 대가로 속도가 느려질 수 있음을 기록한다. "재현이 안 된다"를 버그로 단정하기 전에 이 효과의 크기부터 잰다.
:::

:::note[분야 예시]
기후·날씨 모델은 매 시간 단계마다 기온 같은 상태량에 작은 증분(tendency)을 더한다. float32에서 288 근처 수의 간격은 $2^{-15} \approx 3.1 \times 10^{-5}$이므로, 288 K에 $10^{-6}$ K를 더하면 값이 전혀 바뀌지 않는다. 한 단계의 증분이 작아지도록 시간 간격을 줄이면 오히려 이런 손실이 늘어날 수 있다. 그래서 저정밀도로 모델을 돌리는 연구에서는 상태량을 기준값으로부터의 편차로 저장하거나, 누적만 높은 정밀도로 하는 등의 설계를 따로 검토한다.
:::

## 안정한 계산식

같은 함수를 계산하는 식이 여러 개일 때, 중간 결과가 overflow하지 않고 상쇄를 피하는 식을 고른다. 딥러닝과 통계 계산에서 반복해서 쓰이는 식을 정리한다.

**log-sum-exp.** 확률을 로그 공간에서 다룰 때 $\log \sum_j e^{z_j}$가 계속 나온다(정규화 상수, mixture 모델, softmax의 분모). $z_j$가 크면 $e^{z_j}$가 overflow하고, 모두 매우 작으면 underflow로 $\log 0 = -\infty$가 된다. 모든 $z_j$에서 같은 상수 $m$을 빼면

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}
$$

이 성립하고, $m = \max_j z_j$로 고르면 지수의 인자가 모두 0 이하, 그중 하나는 정확히 0이어서 합이 1 이상이 된다. overflow도 $\log 0$도 생기지 않는다. 같은 이유로 softmax도 최댓값을 빼고 계산한다. $\mathrm{softmax}(\mathbf{z})_k = e^{z_k - m} / \sum_j e^{z_j - m}$은 원래 식과 수학적으로 같다.

**log1p, expm1.** $x$가 아주 작을 때 $\log(1 + x)$를 계산하면, $1 + x$를 만드는 순간 $x$의 아래 자릿수가 1의 ulp에 잘려 나간다. `np.log1p(x)`는 $\log(1+x)$를 $1 + x$를 만들지 않고 계산한다. 마찬가지로 `np.expm1(x)`는 $e^x - 1$의 상쇄를 피한다. 확률 $p$가 아주 작을 때 $\log(1-p)$는 `np.log1p(-p)`로 쓴다.

**sigmoid와 log-sigmoid.** $\sigma(x) = 1/(1 + e^{-x})$는 $x$가 큰 음수이면 $e^{-x}$가 overflow한다. $x \ge 0$에서는 $1/(1+e^{-x})$, $x < 0$에서는 $e^{x}/(1+e^{x})$로 나누어 계산하면 지수의 인자가 항상 0 이하다. binary cross-entropy에 필요한 $\log \sigma(x) = -\log(1 + e^{-x})$는 `-np.logaddexp(0, -x)`처럼 log-sum-exp의 특수한 경우로 계산한다. 정의대로 $\sigma(x)$를 먼저 구하고 로그를 취하면, $x = -800$에서 $\sigma(x)$가 0으로 underflow해 $\log 0 = -\infty$가 된다. 딥러닝 프레임워크가 sigmoid와 loss를 합친 함수(PyTorch의 `BCEWithLogitsLoss`, softmax와 합친 `cross_entropy`)를 따로 제공하는 이유가 이것이다.

**분산.** 분산의 공식 $\mathrm{Var}(t) = \mathbb{E}[t^2] - (\mathbb{E}[t])^2$은 데이터를 한 번만 훑으면 되지만, 평균이 표준편차보다 훨씬 크면 거의 같은 두 큰 수의 뺄셈이 된다. 평균을 먼저 구해 빼고 제곱하는 두 번 훑기(two-pass) 방식이나, 한 번 훑으면서도 안정한 Welford 알고리즘을 쓴다.

```python {exec="browser" session="fp"}
z = np.array([1000.0, 1001.0, 1002.0])            # (3,) logits

with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.sum(np.exp(z))
print("naive softmax :", naive)

m = np.max(z)
lse = m + np.log(np.sum(np.exp(z - m)))           # log-sum-exp
print("stable softmax:", np.exp(z - lse))
print("log-sum-exp   :", lse)

x = 1e-12
print(f"log(1 + x) = {np.log(1 + x):.6e}   log1p(x) = {np.log1p(x):.6e}")

t = 1e8 + rng.normal(0, 1, 1000)                  # (1000,) 큰 offset + 분산 1
one_pass = np.mean(t**2) - np.mean(t) ** 2        # E[t²] - E[t]²
two_pass = np.mean((t - np.mean(t)) ** 2)         # 평균을 먼저 뺀다
print(f"분산: 한 번 훑기 = {one_pass:.4f}   두 번 훑기 = {two_pass:.4f}")
```

```output
naive softmax : [nan nan nan]
stable softmax: [0.09003057 0.24472847 0.66524096]
log-sum-exp   : 1002.4076059644444
log(1 + x) = 1.000089e-12   log1p(x) = 1.000000e-12
분산: 한 번 훑기 = 0.0000   두 번 훑기 = 1.0109
```

naive softmax는 $e^{1000}$이 float64의 최댓값(약 $e^{709.8}$)을 넘어 `inf / inf = nan`이 된다. 최댓값을 뺀 식은 logits $(0, 1, 2)$의 softmax와 같은 값을 낸다. `log(1 + x)`는 상대 오차가 $10^{-4}$ 수준으로, float64인데도 유효숫자가 다섯 자리밖에 맞지 않는다. 분산의 한 번 훑기 공식은 $t^2 \approx 10^{16}$ 근처에서 float64의 간격이 2라서, 참값 약 1인 분산이 반올림 오차에 완전히 묻혀 0이 나왔다.

::quiz{id="softmax-shift"}

:::note[분야 예시]
통계역학에서 에너지 준위 $E_i$를 가진 계의 자유 에너지는 $F = -k_B T \log \sum_i e^{-E_i / k_B T}$이다. 에너지를 $k_B T$ 단위로 재면 $E_i / k_B T$가 수백에서 수천에 이르는 일이 흔해서, 분배함수 $\sum_i e^{-E_i/k_BT}$를 직접 계산하면 underflow한다. 실제 계산은 가장 낮은 에너지를 기준으로 빼는 log-sum-exp로 한다. 베이지안 모델의 marginal likelihood, 입자 필터의 가중치 정규화도 같은 구조다.
:::

**연구에서의 의미.** 안정한 식을 쓰는 가장 좋은 방법은 대부분 직접 쓰지 않는 것이다. `scipy.special.logsumexp`, `np.log1p`, `scipy.special.expit`(sigmoid), 프레임워크의 `log_softmax`·`cross_entropy`는 이미 안정한 식으로 구현되어 있다. 직접 쓴 수식 코드에서 `log(sum(exp(...)))`, `log(sigmoid(...))`, `exp(...) - 1`, `mean(x**2) - mean(x)**2`가 보이면 그 자리를 먼저 의심한다.

## 실습

softmax 계산식과 합의 순서가 오차에 주는 영향을 정밀도별로 비교한다. 코드는 `labs/numerical-methods/floating-point/precision_lab.py`다. 앞부분은 10개의 logits의 크기를 키우며 naive softmax와 최댓값을 뺀 softmax를 float32·float16으로 계산해 float64 기준값과 비교하고, 뒷부분은 크기가 $10^{-3}$부터 $10^{3}$까지 걸친 양수 20만 개를 여러 순서·방법으로 float32에서 더한다. 집필 시점의 실행 결과는 다음과 같다.

```output
[1] softmax: float64 기준값과의 최대 절대 오차 (nan = 계산 실패)
  logit 크기  naive f32  stable f32  naive f16  stable f16
         1    2.0e-08     1.3e-08    4.4e-05     9.3e-05
         3    6.6e-08     2.6e-08    1.1e-04     2.3e-04
        10    4.0e-08     4.0e-08        nan     4.2e-04
        30    2.5e-08     3.5e-08        nan     2.2e-05
       100        nan     2.2e-16        nan     3.2e-16

[2] float32 합 20만 개: 크기가 1e-3 ~ 1e3에 걸친 양수 (기준: math.fsum)
  원래 순서                상대 오차 = 2.1e-04
  오름차순(작은 것부터)         상대 오차 = 1.6e-07
  내림차순(큰 것부터)          상대 오차 = 3.6e-04
  np.sum (쌍별 합)        상대 오차 = 4.5e-08
  Kahan (원래 순서)        상대 오차 = 2.5e-08
  float64로 누적          상대 오차 = 8.2e-15
```

결과를 읽는 법은 세 가지다.

- **naive softmax는 정밀도가 아니라 범위 때문에 실패한다.** 가장 큰 logit이 float16의 $\log(65504) \approx 11.1$, float32의 $\log(3.4 \times 10^{38}) \approx 88.7$을 넘는 순간 `nan`이 된다. 그 전까지는 naive와 안정한 식의 오차가 비슷하다. 안정한 식은 모든 크기에서 계산에 성공한다. logit 크기 100에서 오차가 아주 작은 것은 softmax가 거의 one-hot이 되어 비교할 값 자체가 작기 때문이다.
- **같은 수들의 합이 순서에 따라 1000배 넘게 달라진다.** 큰 것부터 더하면 합이 일찍 커져 작은 항들이 잘려 나가고, 작은 것부터 더하면 작은 항들이 먼저 모여 살아남는다. 정렬은 비용이 들고 언제나 가능한 것도 아니므로, 실무에서는 쌍별 합(`np.sum`), Kahan 합, 또는 더 높은 정밀도의 누적기를 쓴다.
- **누적만 float64로 해도 오차가 거의 사라진다.** 저정밀도로 저장하고 곱하되 합은 높은 정밀도로 누적하는 것이 혼합 정밀도 연산의 기본 설계이고, 행렬 곱 하드웨어가 fp16·bf16 입력을 fp32로 누적하는 이유다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="stable-log-softmax"}

::exercise{id="kahan-sum"}

## 실패 사례

- **fp16에서 gradient가 underflow하는 것.** 작은 gradient(예: $10^{-8}$)는 float16에서 0이 되고, 그 파라미터는 학습되지 않는다. loss가 줄지 않는데 `nan`도 없어서 발견이 늦다. 증상이 보이면 gradient의 크기 분포(0인 비율, 최솟값)를 기록해 보고, loss scaling을 쓰거나(프레임워크의 자동 혼합 정밀도 도구는 scale을 동적으로 조절한다) 범위가 넓은 bf16으로 바꾼다. 반대로 scale을 너무 크게 잡으면 gradient가 overflow해 `inf`가 되므로, 동적 scaler는 `inf`가 나온 step을 건너뛰고 scale을 줄인다.
- **수학적으로 같은 식이 수치적으로도 같다고 가정하는 것.** $\mathbb{E}[t^2] - \mathbb{E}[t]^2$와 $\mathbb{E}[(t - \bar{t})^2]$, $(1 - \cos x)/x^2$와 $2\sin^2(x/2)/x^2$, $\log(\mathrm{softmax}(\mathbf{z}))$와 $\mathbf{z} - \mathrm{logsumexp}(\mathbf{z})$는 수학적으로 같지만 부동소수점에서는 결과가 전혀 다를 수 있다. 식을 "정리"해 코드를 짧게 만들 때마다 상쇄와 overflow가 새로 생기지 않는지 확인한다.
- **부동소수점 수를 `==`로 비교하는 것.** `0.1 + 0.2 == 0.3`은 거짓이다. 테스트와 수렴 판정에는 `np.isclose`·`np.allclose`처럼 허용 오차가 있는 비교를 쓰고, `rtol`·`atol`을 문제의 크기에 맞게 정한다.
- **NaN의 위치를 찾지 않고 학습률만 낮추는 것.** NaN은 한 번 생기면 모든 값으로 번지므로, 최종 loss만 봐서는 원인을 알 수 없다. NumPy에서는 `np.seterr(all="raise")`로 overflow·invalid 연산을 예외로 바꾸고, PyTorch에서는 anomaly detection(`torch.autograd.set_detect_anomaly`)이나 forward hook으로 층마다 `isfinite`를 검사해 처음 생긴 지점을 찾는다.

## 통과 기준

- fp64·fp32·fp16·bf16 중 하나를 주면 machine epsilon, 최댓값, 최소 정규수를 비트 배치에서 계산하고, 그 형식에서 주어진 계산(예: 큰 수에 작은 증분 더하기)이 정밀도를 잃는지 판단한다.
- NaN이 발생한 학습이나 계산에서 처음 `inf`·`nan`이 생긴 연산을 찾아, 원인(overflow·underflow·상쇄)을 밝히고 안정한 식으로 고친다.
- 연습문제 두 개(`stable-log-softmax`, `kahan-sum`)의 테스트를 통과한다.

## 참고 문헌

- Goldberg (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys*, 23(1), 5–48. — 부동소수점 표현과 반올림 오차 분석의 고전적 해설
- Higham (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. — 합의 오차 분석, 보정 합, 안정성 분석 전반
- Kahan (1965). Further remarks on reducing truncation errors. *Communications of the ACM*, 8(1), 40. — 보정 합 알고리즘
- Micikevicius 외 (2018). Mixed precision training. *ICLR 2018*. — fp16 학습의 loss scaling과 fp32 누적
