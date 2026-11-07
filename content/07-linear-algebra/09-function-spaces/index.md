---
id: linear-algebra/function-spaces
title: "함수공간과 연산자"
title_en: "Function Spaces and Operators"
tier: advanced
status: draft
prerequisites: [linear-algebra/inner-products-and-projection]
---

# 함수공간과 연산자

과학 데이터의 많은 부분은 숫자 몇 개가 아니라 **함수**다. 시간에 따른 온도, 파장에 따른 흡광도, 공간에 펼쳐진 속도장과 압력장이 그렇다. 컴퓨터에는 이 함수를 격자 위의 값이나 basis의 계수 같은 유한한 숫자 배열로 저장할 수밖에 없다. 그런데 배열은 함수의 한 가지 **표현**일 뿐이다. 같은 함수를 더 촘촘한 격자에 담으면 배열의 길이와 값이 모두 바뀐다. 배열을 함수 자체로 착각하면 해상도가 바뀌는 순간 무너지는 모델을 만들게 된다.

이 장은 7.4장의 내적·노름·직교·사영을 함수에 적용한다. 함수를 벡터로 보면 Fourier 급수는 정규직교 basis 전개가 되고, 미분은 행렬처럼 다룰 수 있는 선형 연산자가 되며, 커널 방법과 Gaussian process는 특정한 함수공간 위의 추정 문제가 된다. 24.4장의 neural operator와 13.7장의 Gaussian process를 읽기 위한 언어를 여기서 준비한다.

## 학습 목표

- 함수를 벡터로 다루는 관점을 익힌다
- 연산자와 이산화의 관계를 이해한다
- 커널과 RKHS의 직관을 얻는다

## 함수를 벡터로

7.1장에서 본 것처럼 구간 $[a, b]$ 위의 실숫값 함수들은 벡터공간을 이룬다. 두 함수의 합 $(f + g)(x) = f(x) + g(x)$와 스칼라 곱 $(\alpha f)(x) = \alpha f(x)$가 벡터공간의 공리를 모두 만족한다. 벡터의 "성분"이 $i = 1, \dots, n$으로 번호 붙은 숫자였다면, 함수의 "성분"은 $x \in [a, b]$로 번호 붙은 값 $f(x)$다.

**$L^2$ 내적.** 유한차원 내적 $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_i u_i v_i$에서 합을 적분으로 바꾸면 함수의 내적이 된다.

$$
\langle f, g \rangle = \int_a^b f(x)\,g(x)\,dx, \qquad \lVert f \rVert = \sqrt{\langle f, f \rangle}.
$$

$\lVert f \rVert^2 < \infty$인 함수들의 공간을 $L^2([a, b])$라 한다. 이 내적으로 7.4장의 개념이 모두 옮겨 온다. 두 함수가 직교한다는 것은 $\langle f, g \rangle = 0$이고, 두 함수의 거리 $\lVert f - g \rVert$는 차이의 제곱을 구간 전체에서 적분한 값의 제곱근이다. 회귀 모델의 평균제곱오차는 이 거리를 표본 점에서 근사한 것이다.

::quiz{id="l2-norm-sin"}

**격자 값은 벡터지만 함수가 아니다.** 함수 $f$를 $[0, 1]$의 $N$칸 중점 $x_i$에서 표본한 벡터 $\mathbf{f} = (f(x_1), \dots, f(x_N))$를 생각한다. 두 벡터의 내적 $\mathbf{f}^\top\mathbf{g}$는 $N$개 항의 합이라 $N$에 비례해 커진다. 칸 너비 $h = 1/N$을 곱해야 적분(중점 규칙)의 근사가 되고, 해상도와 무관한 값으로 수렴한다.

```python {exec="browser" session="funcspace"}
import numpy as np

f = lambda x: np.exp(x)
g = lambda x: np.sin(np.pi * x)
exact = np.pi * (np.e + 1) / (np.pi**2 + 1)        # ∫_0^1 e^x sin(πx) dx

for N in [10, 100, 1000]:
    x = (np.arange(N) + 0.5) / N                    # [0, 1]을 N칸으로 나눈 중점, (N,)
    h = 1.0 / N                                     # 칸 너비
    print(f"N = {N:4d}  f·g = {f(x) @ g(x):9.3f}   h (f·g) = {h * (f(x) @ g(x)):.6f}")
print(f"정확한 <f, g> = {exact:.6f}")
```

```output
N =   10  f·g =    10.796   h (f·g) = 1.079555
N =  100  f·g =   107.473   h (f·g) = 1.074727
N = 1000  f·g =  1074.679   h (f·g) = 1.074679
정확한 <f, g> = 1.074678
```

::quiz{id="grid-inner-product-scaling"}

**무한차원의 의미.** 다항식 $1, x, x^2, \dots$는 모두 서로 일차독립이므로, 함수공간은 유한개의 basis로 span할 수 없다. 이것이 함수공간이 **무한차원**이라는 뜻이다. 실용적인 결과는 두 가지다. 첫째, 어떤 유한한 표현(격자 $N$개, basis $K$개)도 근사이고, 근사 오차를 따로 따져야 한다. 둘째, 유한차원에서 당연했던 성질이 깨질 수 있다. 예를 들어 유한차원에서 모든 선형사상은 연속이지만, 함수공간 위의 미분은 그렇지 않다(아래 연산자 절).

## Hilbert 공간과 Banach 공간

**왜 필요한가.** 함수를 basis로 전개하거나 근사를 점점 정밀하게 만들 때, 근사의 극한이 다시 같은 공간 안에 있어야 이론이 성립한다. 유리수만 쓰면 $1, 1.4, 1.41, 1.414, \dots$처럼 점점 가까워지는 수열의 극한 $\sqrt{2}$가 유리수가 아니어서 "극한이 없는" 일이 생긴다. 공간에 이런 구멍이 없다는 성질이 **완비성**(completeness)이다. 정확히는 서로 점점 가까워지는 모든 수열(Cauchy 수열)이 그 공간 안의 점으로 수렴한다는 뜻이다.

**정의.** 노름이 있고 그 노름에 대해 완비인 벡터공간을 **Banach 공간**이라 한다. 내적이 있고 그 내적이 만드는 노름에 대해 완비인 공간을 **Hilbert 공간**이라 한다. Hilbert 공간은 Banach 공간 중에서 내적, 즉 각도와 직교를 가진 공간이다.

| 공간 | 노름 | 종류 |
|---|---|---|
| $\mathbb{R}^n$ | 유클리드 노름 | Hilbert (유한차원) |
| $L^2([a, b])$ | $\big(\int \lvert f \rvert^2\big)^{1/2}$ | Hilbert |
| $L^1([a, b])$ | $\int \lvert f \rvert$ | Banach (내적 없음) |
| $C([a, b])$ (연속함수) | $\max_x \lvert f(x) \rvert$ | Banach (내적 없음) |

어떤 노름을 쓰느냐는 "두 함수가 가깝다"의 뜻을 정한다. $L^2$ 노름에서는 아주 좁은 구간의 큰 오차가 작게 평가될 수 있고, 최대 노름에서는 한 점의 큰 오차가 곧 큰 거리다. 7.2장의 보편 근사 정리가 말하는 "균등 근사"는 최대 노름의 근사다.

**직교 basis 전개.** Hilbert 공간에서는 정규직교 basis $\{e_k\}$로 모든 원소를 전개할 수 있다.

$$
f = \sum_{k} \langle f, e_k \rangle\, e_k, \qquad \lVert f \rVert^2 = \sum_k \lvert \langle f, e_k \rangle \rvert^2.
$$

둘째 식(Parseval 등식)은 피타고라스 정리의 무한차원판이다. 계수 $\langle f, e_k \rangle$는 7.4장의 사영 계수와 같고, 앞의 $K$개 항에서 자른 합은 $K$차원 부분공간으로의 최적 근사($L^2$ 거리 기준)다. $[0, 2\pi]$ 위에서 $\{1, \cos kx, \sin kx\}$가 서로 직교한다는 사실이 Fourier 급수의 기반이고, $[-1, 1]$ 위에서는 Legendre 다항식 $P_0, P_1, \dots$가 서로 직교한다.

basis가 직교가 아니면 계수를 구할 때 Gram 행렬 $G_{ij} = \langle b_i, b_j \rangle$을 풀어야 한다(정규방정식 $\mathbf{G}\mathbf{c} = (\langle f, b_i \rangle)_i$). 같은 다항식 공간을 단항식 $1, x, \dots, x^{n-1}$과 Legendre 다항식으로 각각 span하고 Gram 행렬의 조건수를 비교한다. 적분은 Gauss–Legendre 구적으로 계산한다(9.5장).

```python {exec="browser" session="funcspace"}
xq, wq = np.polynomial.legendre.leggauss(40)        # [-1, 1] 위 Gauss–Legendre 구적점·가중치, (40,)

def gram(B):                                        # B (n, Q): 구적점에서 basis 함수 값
    return (B * wq) @ B.T                           # G_ij = <b_i, b_j>, (n, n)

for n in [4, 8, 12]:
    mono = np.vander(xq, n, increasing=True).T      # 1, x, ..., x^(n-1)   (n, 40)
    leg = np.polynomial.legendre.legvander(xq, n - 1).T   # P_0, ..., P_(n-1)   (n, 40)
    print(f"n = {n:2d}  cond(단항식 Gram) = {np.linalg.cond(gram(mono)):9.2e}"
          f"   cond(Legendre Gram) = {np.linalg.cond(gram(leg)):5.1f}")

G = gram(np.polynomial.legendre.legvander(xq, 3).T)
print("Legendre Gram 대각:", np.round(np.diag(G), 4), " 비대각 ≈ 0:", np.abs(G - np.diag(np.diag(G))).max() < 1e-12)
```

```output
n =  4  cond(단항식 Gram) =  6.76e+01   cond(Legendre Gram) =   7.0
n =  8  cond(단항식 Gram) =  5.52e+04   cond(Legendre Gram) =  15.0
n = 12  cond(단항식 Gram) =  5.25e+07   cond(Legendre Gram) =  23.0
Legendre Gram 대각: [2.     0.6667 0.4    0.2857]  비대각 ≈ 0: True
```

Legendre Gram 행렬은 대각($\langle P_k, P_k \rangle = 2/(2k+1)$)이라 조건수가 천천히 늘고, 단항식 Gram 행렬은 차수 12에서 조건수가 $5 \times 10^7$을 넘는다. 고차 단항식 $x^{10}$과 $x^{11}$이 구간 위에서 거의 같은 모양이라 거의 일차종속이기 때문이다. 다항식 회귀에서 `np.polyfit`에 높은 차수를 주면 경고가 나오는 이유이기도 하다.

::quiz{id="monomial-gram"}

## 연산자

**정의.** 함수를 받아 함수를 내놓는 사상을 **연산자**(operator)라 한다. 선형사상의 조건 $T(\alpha f + \beta g) = \alpha Tf + \beta Tg$를 만족하면 선형 연산자다. 과학 계산에서 자주 만나는 선형 연산자는 다음과 같다.

| 연산자 | 정의 | 행렬로 보면 |
|---|---|---|
| 미분 | $(Df)(x) = f'(x)$ | 차분 행렬 |
| 적분 | $(If)(x) = \int_a^x f(t)\,dt$ | 하삼각 누적합 행렬 |
| 곱셈 | $(M_w f)(x) = w(x) f(x)$ | 대각행렬 |
| 적분 연산자 | $(Kf)(x) = \int k(x, y) f(y)\,dy$ | 커널 값 $k(x_i, y_j)$에 구적 가중치를 곱한 행렬 |

열방정식, 파동방정식, Schrödinger 방정식처럼 물리 법칙의 상당수는 "어떤 선형 연산자를 상태 함수에 작용한다"는 꼴이다. 선형 연산자의 고유함수는 고유벡터의 무한차원판이다. 주기 구간에서 $\frac{d}{dx}e^{ikx} = ik\,e^{ikx}$이므로 복소 지수함수 $e^{ikx}$는 미분 연산자의 고유함수이고 고윳값은 $ik$다. Fourier basis에서 미분이 **대각행렬**(계수에 $ik$를 곱하는 연산)이 되는 이유다.

**유계성.** 유한차원 행렬 $\mathbf{A}$는 언제나 $\lVert \mathbf{A}\mathbf{x} \rVert \le C \lVert \mathbf{x} \rVert$인 상수 $C$(연산자 노름)를 가진다. 함수공간의 연산자는 그렇지 않을 수 있다. $f_k(x) = \sin kx$는 $k$와 무관하게 $\lVert f_k \rVert^2 = \pi$인데, 미분하면 $\lVert Df_k \rVert^2 = k^2\pi$로 끝없이 커진다. 미분은 **유계가 아닌**(unbounded) 연산자다. 직관적으로는 "작게 빠르게 흔들리는 함수의 미분은 클 수 있다"는 뜻이고, 측정 잡음이 있는 데이터를 미분하면 잡음이 증폭되는 이유가 이것이다. 반대로 적분 연산자는 흔들림을 평균해 줄이므로 유계이고, 고주파 성분을 억누른다.

**두 표현에서의 미분 행렬.** 주기 함수 $u(x) = e^{\sin x}$의 미분을 두 방식으로 근사한다. 격자 값 표현에서는 중심 차분 행렬 $\mathbf{D}_{\text{fd}}$, $(\mathbf{D}_{\text{fd}}\mathbf{u})_i = (u_{i+1} - u_{i-1})/(2h)$를 쓴다. Fourier 표현에서는 격자 값을 이산 Fourier 변환으로 계수로 바꾸고, 계수에 $ik$를 곱한 뒤 다시 격자 값으로 돌아온다.

```python {exec="browser" session="funcspace"}
u = lambda x: np.exp(np.sin(x))                     # 주기 2π인 매끄러운 함수
du = lambda x: np.cos(x) * np.exp(np.sin(x))        # 정확한 도함수

for N in [16, 32, 64]:
    x = 2 * np.pi * np.arange(N) / N                # (N,)
    h = 2 * np.pi / N
    D_fd = (np.roll(np.eye(N), 1, axis=1) - np.roll(np.eye(N), -1, axis=1)) / (2 * h)  # (N, N) 중심 차분
    k = np.fft.fftfreq(N, d=1.0 / N)                # 파수 0, 1, ..., -1
    k[N // 2] = 0.0                                 # 짝수 N의 Nyquist 모드는 0으로
    d_spec = np.real(np.fft.ifft(1j * k * np.fft.fft(u(x))))   # Fourier 계수에 ik를 곱함
    err_fd = np.abs(D_fd @ u(x) - du(x)).max()
    err_sp = np.abs(d_spec - du(x)).max()
    sp = "< 1e-12" if err_sp < 1e-12 else f"{err_sp:.1e}"
    print(f"N = {N:2d}  유한차분 오차 = {err_fd:.2e}   Fourier 오차 = {sp}")
```

```output
N = 16  유한차분 오차 = 9.36e-02   Fourier 오차 = 1.8e-07
N = 32  유한차분 오차 = 2.58e-02   Fourier 오차 = < 1e-12
N = 64  유한차분 오차 = 6.51e-03   Fourier 오차 = < 1e-12
```

유한차분 오차는 $N$을 두 배로 할 때마다 약 4분의 1로 준다. Taylor 전개로 오차가 $h^2 u^{(3)}/6$ 크기이기 때문이다(2차 정확도). 매끄러운 주기 함수의 Fourier 미분은 $N = 32$에서 이미 부동소수점 반올림 수준에 이른다. 대신 Fourier 방법은 주기성과 매끄러움을 가정한다. 경계가 있거나 불연속이 있는 함수에서는 이 장점이 사라지고 진동(Gibbs 현상)이 생긴다. 어떤 표현이 좋은지는 함수의 성질이 정한다.

::quiz{id="fd-error-order"}

## 이산화

**왜 필요한가.** 컴퓨터는 무한차원 공간을 다룰 수 없으므로, 연속 문제를 유한차원 문제로 바꿔야 한다. 이 과정이 **이산화**(discretization)다. 크게 두 가지 길이 있다.

- **격자(점) 표현.** 함수를 점 $x_1, \dots, x_N$의 값으로 나타낸다. 연산자는 유한차분처럼 이웃한 값을 섞는 행렬이 된다. 국소적이고 직관적이며, 복잡한 영역과 경계에 쓰기 쉽다.
- **basis(계수) 표현.** 함수를 $u \approx \sum_{k=1}^K c_k \phi_k$로 놓고 계수 $\mathbf{c}$를 미지수로 삼는다. 연산자 $T$는 $(T\phi_j)$를 다시 basis로 사영한 행렬 $A_{ij} = \langle \phi_i, T\phi_j \rangle$이 된다. Fourier·다항식 basis(스펙트럴 방법), 국소적인 조각 다항식 basis(유한요소법)가 이 계열이다.

**해상도와 이산화 오차.** 이산 연산자가 연속 연산자를 얼마나 잘 따라가는지는 함수의 "주파수"에 따라 다르다. 주기 격자에서 $-d^2/dx^2$를 중심 차분으로 근사하면, 연속 연산자의 고윳값 $k^2$에 대응하는 이산 고윳값은 $\frac{4}{h^2}\sin^2\frac{kh}{2}$다. 낮은 파수에서는 거의 같고, 격자가 표현할 수 있는 가장 높은 파수 근처에서는 크게 어긋난다.

```python {exec="browser" session="funcspace"}
def neg_laplacian(N):                               # 주기 격자 위 -d²/dx²의 중심 차분 행렬, (N, N)
    h = 2 * np.pi / N
    I = np.eye(N)
    return (2 * I - np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1)) / h**2

lam = np.linalg.eigvalsh(neg_laplacian(64))         # 오름차순, (64,)
for kk in [1, 4, 16, 32]:
    print(f"k = {kk:2d}  연속 고윳값 k² = {kk**2:5d}   이산 고윳값 = {lam[2 * kk - 1]:8.2f}")
for N in [32, 64, 128]:
    print(f"N = {N:3d}  최대 고윳값 = {np.linalg.eigvalsh(neg_laplacian(N)).max():8.1f}")
```

```output
k =  1  연속 고윳값 k² =     1   이산 고윳값 =     1.00
k =  4  연속 고윳값 k² =    16   이산 고윳값 =    15.80
k = 16  연속 고윳값 k² =   256   이산 고윳값 =   207.51
k = 32  연속 고윳값 k² =  1024   이산 고윳값 =   415.01
N =  32  최대 고윳값 =    103.8
N =  64  최대 고윳값 =    415.0
N = 128  최대 고윳값 =   1660.0
```

$N = 64$ 격자에서 $k = 1, 4$는 정확하지만 $k = 32$(격자가 표현할 수 있는 최고 파수)의 고윳값은 정확한 값의 40%다. 그리고 $N$을 두 배로 할 때마다 최대 고윳값이 4배가 된다. 이것은 오차가 아니라 미분이 유계가 아니라는 성질이 드러난 것이다. 더 세밀한 격자는 더 높은 파수를 표현하고, 그만큼 큰 고윳값을 가진다. 이산화된 미분 행렬의 조건수가 해상도와 함께 커지고, 명시적 시간적분의 안정한 step 크기가 $h^2$에 비례해 줄어드는 이유다(9.6장).

::quiz{id="derivative-unbounded"}

정리하면 이산화된 연산자는 **어떤 해상도에서 본 연속 연산자의 그림자**다. 연구에서 중요한 질문은 "이 행렬이 무엇인가"가 아니라 "해상도를 바꾸면 이 행렬이 같은 연속 연산자의 다른 그림자로 바뀌는가"다.

:::note[분야 예시]
기후 모델과 수치 기상 예측 모델은 같은 물리 방정식을 서로 다른 격자 해상도로 푼다. 격자가 성기면 격자 한 칸보다 작은 과정(구름, 난류)은 직접 표현되지 않고, 그 효과를 별도의 근사(parameterization)로 넣어야 한다. 데이터 기반 기상 모델을 한 해상도의 재분석 자료로 학습하고 다른 해상도에 적용할 때, 모델이 연속장의 성질을 배웠는지 그 격자의 성질을 배웠는지가 문제가 된다(27장).
:::

:::note[분야 예시]
분광 데이터는 파장의 함수다. 장비마다 파장 격자(간격과 범위)가 다르므로, 한 장비의 스펙트럼 벡터 1024개로 학습한 모델은 2048점 장비의 스펙트럼에 그대로 쓸 수 없다. 공통 격자로 보간하는 것도 이산화의 선택이고, 보간이 좁은 흡수 봉우리를 뭉개면 정보가 사라진다. 스펙트럼을 "길이 1024 벡터"가 아니라 "파장의 함수"로 다루면 이런 선택을 의식하게 된다.
:::

**neural operator의 관점.** 24.4장의 neural operator는 함수를 함수로 보내는 사상(예: 초기 조건 → 나중 시각의 해)을 학습한다. 핵심 설계 원칙은 학습되는 파라미터를 특정 격자가 아니라 연속 대상에 묶는 것이다. 예를 들어 Fourier 계수 공간에서 낮은 파수 모드마다 가중치를 두면, 그 가중치는 격자 해상도와 무관하게 정의된다. 위의 Fourier 미분이 "계수에 $ik$를 곱하는" 연산이라 어떤 $N$에도 그대로 쓰인 것과 같은 원리다.

## 커널과 RKHS

**왜 필요한가.** 입력 $x$에서 출력 $y$로 가는 함수를 데이터로 추정할 때, 선형 모델은 너무 좁고 모든 함수의 공간은 너무 넓다. 매끄러움 같은 성질을 가진 함수공간을 정하고 그 안에서 데이터에 맞는 함수를 찾으면 좋다. 커널은 이런 함수공간을 하나의 함수 $k(x, x')$로 정하는 방법이다.

**양의 정부호 커널.** 대칭 함수 $k(x, x')$가 임의의 점 $x_1, \dots, x_N$에 대해 Gram 행렬 $K_{ij} = k(x_i, x_j)$를 항상 양의 준정부호(7.5장)로 만들면 **양의 정부호 커널**(positive definite kernel)이라 한다. 대표적인 예가 RBF(가우시안) 커널 $k(x, x') = \exp\!\big(-(x - x')^2/(2\ell^2)\big)$이다. $\ell$은 길이 척도로, 얼마나 떨어진 입력까지 비슷하다고 볼지 정한다.

**커널 트릭.** 양의 정부호 커널은 어떤 특징 사상 $\phi$에 대해 $k(x, x') = \langle \phi(x), \phi(x') \rangle$으로 쓸 수 있다. $k(x, x') = (xx' + 1)^2$은 $\phi(x) = (x^2, \sqrt{2}x, 1)$의 내적이고, RBF 커널의 $\phi$는 무한차원이다. 내적만 필요한 알고리즘(ridge 회귀, SVM, PCA 등)은 $\phi$를 계산하지 않고 $k$만으로 실행할 수 있다. 이것이 커널 트릭(kernel trick)이다.

::quiz{id="valid-kernels"}

**RKHS.** 커널 하나가 함수공간 하나를 정한다. 학습 점 위에 커널 함수를 세운 선형결합 $f(x) = \sum_i \alpha_i k(x, x_i)$들을 모으고 완비화한 Hilbert 공간을 **재생 커널 Hilbert 공간**(reproducing kernel Hilbert space, RKHS) $\mathcal{H}_k$라 한다. 이름은 재생 성질(reproducing property)에서 왔다.

$$
f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}}.
$$

한 점에서 함수값을 읽는 일이 커널 함수와의 내적으로 표현된다는 뜻이다. RKHS 노름 $\lVert f \rVert_{\mathcal{H}}$는 함수가 커널이 정한 의미에서 얼마나 "복잡한가"를 잰다. RBF 커널에서는 길이 척도 $\ell$보다 빠르게 흔들리는 함수일수록 노름이 크다.

**표현자 정리.** RKHS 위에서 정규화된 최소제곱 문제

$$
\min_{f \in \mathcal{H}} \sum_{i=1}^N \big(y_i - f(x_i)\big)^2 + \lambda \lVert f \rVert_{\mathcal{H}}^2
$$

의 해는 $f(x) = \sum_{i=1}^N \alpha_i k(x, x_i)$ 꼴이다. 이것이 **표현자 정리**(representer theorem)의 직관적인 내용이다. 이유는 7.4장의 직교 분해다. $f$를 학습 점의 커널 함수들이 span하는 부분공간 성분과 그에 직교하는 성분으로 나누면, 재생 성질 때문에 직교 성분은 학습 점에서의 값 $f(x_i)$를 바꾸지 못하고 노름만 키운다. 그러니 최적해에서는 직교 성분이 0이다. 무한차원 문제가 $N$개 계수에 대한 유한차원 문제로 줄고, 계수는 $(\mathbf{K} + \lambda\mathbf{I})\boldsymbol{\alpha} = \mathbf{y}$로 구한다.

```python {exec="browser" session="funcspace"}
rng = np.random.default_rng(0)
x_tr = np.sort(rng.uniform(0, 1, 20))               # (N,)
y_tr = np.sin(2 * np.pi * x_tr) + 0.1 * rng.normal(size=20)

def k_rbf(a, b, ell=0.1):                           # (n,), (m,) → (n, m)
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * ell**2))

K = k_rbf(x_tr, x_tr)                               # (N, N)
print("Gram 최소 고윳값 ≥ 0 :", np.linalg.eigvalsh(K).min() > -1e-10)
alpha = np.linalg.solve(K + 1e-2 * np.eye(20), y_tr)    # (N,) 표현자 계수
x_te = np.linspace(0, 1, 200)
f_te = k_rbf(x_te, x_tr) @ alpha                    # f(x) = Σ_i α_i k(x, x_i), (200,)
print("계수 수 =", alpha.size, "  test RMSE =", round(np.sqrt(np.mean((f_te - np.sin(2 * np.pi * x_te)) ** 2)), 3))
```

```output
Gram 최소 고윳값 ≥ 0 : True
계수 수 = 20   test RMSE = 0.079
```

::quiz{id="representer-theorem"}

**연구에서의 의미.** 커널 방법은 16.3장에서 분류·회귀 도구로 다시 나온다. 13.7장의 Gaussian process 회귀는 같은 수식을 확률로 읽은 것이다. 커널 $k$를 사전분포의 공분산 함수로, $\lambda$를 관측 잡음의 분산으로 두면 GP 사후분포의 평균은 위의 kernel ridge 예측과 같은 식이 되고, GP는 여기에 예측의 불확실성(사후분산)을 더한다. 대가는 계산량이다. $N \times N$ 선형계를 풀어야 하므로 학습 점이 많아지면 $O(N^3)$의 비용이 든다.

## 실습

함수를 Fourier basis와 격자 값으로 각각 표현하고, 미분 연산자를 두 표현에서 행렬로 근사해 비교한다. 코드는 `labs/linear-algebra/function-spaces/operator_matrices.py`다. 세 부분으로 되어 있다.

1. 격자 표현에서 유한차분 행렬 $\mathbf{D}_{\text{fd}}$와 스펙트럴 미분 행렬 $\mathbf{D}_{\text{sp}} = \mathbf{F}^{-1}\operatorname{diag}(ik)\mathbf{F}$를 만들어 오차를 비교한다.
2. 실수 Fourier basis $\{1, \cos kx, \sin kx\}_{k \le K}$의 계수 공간에서 미분 행렬(블록 대각)을 만들고, $K$에 따른 근사 오차를 본다.
3. $N = 32$ 격자에서 무작위 저주파 함수들로 3-점 stencil(이웃 값의 가중합)을 최소제곱으로 "학습"한 뒤, 같은 가중치를 $N = 64, 128$ 격자에 그대로 쓴다.

집필 시점의 실행 결과는 다음과 같다.

```output
1) 격자 표현: 최대 오차
     N   유한차분 D_fd   스펙트럴 D_sp
     8       3.40e-01       4.32e-03
    16       9.36e-02       1.76e-07
    32       2.58e-02       3.77e-15
    64       6.51e-03       1.07e-14
   128       1.63e-03       2.00e-14

2) Fourier 표현: 계수 K개까지 자른 뒤 계수 공간에서 미분 (오차는 x 200점에서 최대)
     K   행렬 크기   u 근사 오차   u' 근사 오차
     2     5x5        5.04e-02       1.54e-01
     4     9x9        5.91e-04       2.99e-03
     8    17x17       1.16e-08       1.05e-07
    12    25x25       4.17e-14       5.41e-13

   유한차분 행렬의 고윳값 허수부 sin(kh)/h와 정확한 값 k (N = 32)
   k =  1   sin(kh)/h =  0.994   정확한 값 = 1
   k =  4   sin(kh)/h =  3.601   정확한 값 = 4
   k =  8   sin(kh)/h =  5.093   정확한 값 = 8
   k = 15   sin(kh)/h =  0.994   정확한 값 = 15

3) 해상도 바꾸기: N = 32에서 학습한 3-점 stencil을 N = 64에 그대로 적용
   학습한 stencil w = [-2.75 -0.    2.75]   (참고: ±1/(2h) = ±2.546)
   N =  32   stencil 상대 오차 =  0.069   Fourier 승수 ik 상대 오차 = 1.8e-15
   N =  64   stencil 상대 오차 =  0.462   Fourier 승수 ik 상대 오차 = 4.1e-15
   N = 128   stencil 상대 오차 =  0.730   Fourier 승수 ik 상대 오차 = 8.2e-15
```

결과를 읽는 법은 세 가지다.

- **같은 연산자, 다른 행렬.** 격자 표현의 미분 행렬은 $N \times N$이고 이웃끼리만 섞는다. Fourier 표현의 미분 행렬은 블록 대각이다. 둘 다 같은 연속 연산자 $d/dx$의 근사이고, 매끄러운 주기 함수에서는 Fourier 쪽이 같은 크기로 훨씬 정확하다.
- **유한차분은 높은 파수에서 틀린다.** 중심 차분의 고윳값은 $ik$ 대신 $i\sin(kh)/h$라서, $k = 15$(격자가 표현하는 거의 최고 파수)에서는 정확한 값 15 대신 0.994가 된다. 빠르게 흔들리는 성분의 미분을 거의 0으로 본다는 뜻이다.
- **격자 단위로 학습한 가중치는 해상도를 옮기지 못한다.** 학습한 stencil의 가중치 $\pm 2.75$에는 학습 격자의 간격 $h$가 숨어 있다. 격자를 두 배로 세밀하게 하면 이웃 값의 차이가 절반이 되므로 출력도 약 절반이 되고, 상대 오차가 0.069에서 0.462로 커진다. 연속 대상(파수 $k$)에 묶인 Fourier 승수 $ik$는 해상도와 무관하게 정확하다. 가중치가 $\pm 1/(2h) = \pm 2.546$보다 큰 것은 학습 데이터의 파수(1–4)에서 $\sin(kh)/h < k$인 유한차분의 과소 추정을 보정하도록 맞춰졌기 때문이다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="periodic-derivative-matrix"}

::exercise{id="kernel-ridge"}

## 실패 사례

- **격자 위 값 벡터를 함수 자체로 착각해 해상도 변화에 무너지는 모델을 만드는 것.** 64 × 64 격자에서 학습한 CNN의 필터나 MLP의 가중치에는 격자 간격이 숨어 있다. 128 × 128 입력에서는 같은 필터가 다른 물리적 길이를 보고, 미분 같은 연산은 크기가 달라진다(실습 3). 해상도를 바꿔 쓸 모델이라면 학습 때부터 여러 해상도로 평가하고, 가능하면 연속 대상(물리 좌표, 파수)에 묶인 파라미터를 쓴다.
- **격자 값 벡터의 노름을 그대로 손실이나 거리로 쓰는 것.** `np.sum((u_pred - u_true)**2)`는 격자점 수에 비례한다. 해상도가 다른 실험의 손실을 비교하려면 칸 너비를 곱하거나 평균을 쓴다.
- **잡음 있는 데이터를 그대로 미분하는 것.** 미분은 유계가 아닌 연산자라 고주파 잡음을 증폭한다. 측정된 위치에서 속도를, 전압에서 미분 용량을 계산할 때는 평활화나 정규화를 먼저 하고, 그 선택이 결론을 바꾸는지 확인한다.
- **양의 정부호가 아닌 "커널"을 쓰는 것.** 직접 설계한 유사도 함수가 양의 준정부호 Gram 행렬을 만들지 않으면, kernel ridge나 GP의 선형계가 불안정하고 GP의 사후분산이 음수가 될 수 있다. 데이터에서 Gram 행렬의 최소 고윳값을 확인한다.

## 통과 기준

- neural operator와 Gaussian process가 함수공간 관점에서 각각 무엇을 학습하는지 설명한다. neural operator는 함수에서 함수로 가는 연산자를, GP는 커널이 정한 함수공간 위의 함수 하나(와 그 불확실성)를 추정한다는 점, 그리고 두 경우 모두 격자나 학습 점이 아닌 연속 대상에 묶인 표현이 필요한 이유를 포함한다.
- 같은 함수의 미분을 격자 표현과 Fourier 표현에서 각각 행렬로 근사하고, 해상도를 바꿀 때 오차가 어떻게 변하는지 수치로 보인다.
- 연습문제 두 개(`periodic-derivative-matrix`, `kernel-ridge`)의 테스트를 통과한다.
