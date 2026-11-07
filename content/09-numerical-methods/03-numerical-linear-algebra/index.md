---
id: numerical-methods/numerical-linear-algebra
title: "수치 선형대수"
title_en: "Numerical Linear Algebra"
tier: core
status: draft
prerequisites: [numerical-methods/conditioning-and-stability, linear-algebra/svd-and-pca]
---

# 수치 선형대수

과학 계산의 시간 대부분은 결국 선형시스템 $\mathbf{A}\mathbf{x} = \mathbf{b}$를 푸는 데 쓰인다. 편미분방정식의 이산화(9.7장), 암시적 ODE solver와 Newton 방법의 매 step(9.4장, 9.6장), 최소제곱과 Gaussian process, 2차 최적화가 모두 그렇다. 7장이 선형대수의 **무엇**을 다뤘다면, 이 장은 그것을 **어떻게** 계산하는지를 다룬다. 같은 $\mathbf{x}$라도 행렬의 크기·구조·조건에 따라 알맞은 계산법이 다르고, 잘못 고르면 결과가 틀리거나 메모리에 들어가지 않거나 몇 시간이 걸린다.

이 장의 질문은 하나다. **주어진 행렬에 어떤 solver를 쓸 것인가.** 이를 위해 분해에 기반한 직접법, 행렬–벡터 곱만 쓰는 반복법, 0이 대부분인 희소 행렬의 저장과 풀이, 행렬을 아예 만들지 않는 대규모 계산을 차례로 본다.

## 학습 목표

- 선형시스템의 직접법과 반복법을 선택한다
- 희소 행렬을 다룬다

## 직접법

**왜 필요한가.** 크기가 수천 이하인 dense 행렬이나, 같은 행렬로 여러 오른쪽 변을 풀어야 하는 경우에는 행렬을 한 번 **분해**(factorization)해 두는 것이 가장 정확하고 빠르다. 직접법은 정해진 횟수의 연산으로 끝나고, 결과가 backward stable해서 9.2장의 기준으로 "할 수 있는 만큼 정확한" 답을 낸다.

**정의.** 행렬을 풀기 쉬운 행렬(삼각, 직교, 대각)의 곱으로 쓰는 것이 분해다. 삼각 시스템은 전진·후진 대입으로 $O(n^2)$에 풀리므로, 분해에 드는 $O(n^3)$ 연산을 한 번만 하면 이후 오른쪽 변마다 $O(n^2)$이면 된다.

| 분해 | 형태 | 조건 | 연산량(dense, 대략) | 주된 쓰임 |
|---|---|---|---|---|
| LU | $\mathbf{P}\mathbf{A} = \mathbf{L}\mathbf{U}$ | 정사각 가역 | $\tfrac{2}{3}n^3$ | 일반 선형시스템 |
| Cholesky | $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ | 대칭 양의 정부호(SPD) | $\tfrac{1}{3}n^3$ | 공분산·커널 행렬, Hessian, SPD 판정 |
| QR | $\mathbf{A} = \mathbf{Q}\mathbf{R}$ | $m \times n$, $m \ge n$ | $2mn^2 - \tfrac{2}{3}n^3$ (Householder) | 최소제곱, 직교 기저 |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ | 제한 없음 | QR보다 큰 상수 | rank가 부족한 최소제곱, pseudoinverse, 저랭크 근사 |

$\mathbf{L}$은 하삼각, $\mathbf{U}$와 $\mathbf{R}$은 상삼각, $\mathbf{Q}$는 열이 정규직교인 행렬, $\mathbf{P}$는 행의 순서를 바꾸는 치환행렬이다. QR과 SVD의 기하는 7.4장과 7.6장에서 보았다.

**LU와 피벗팅.** LU 분해는 Gauss 소거를 행렬로 기록한 것이다. $k$번째 단계에서 대각 원소 $a_{kk}$(pivot)로 아래 행들을 소거하는데, pivot이 0이면 진행할 수 없고, 0은 아니어도 아주 작으면 곱하는 수 $a_{ik}/a_{kk}$가 거대해져 다른 원소의 정보를 반올림으로 지운다. **부분 피벗팅**(partial pivoting)은 각 단계에서 그 열의 남은 원소 중 절댓값이 가장 큰 것을 pivot으로 고르도록 행을 바꾼다. 행렬 $\mathbf{P}$가 그 교환을 기록한다.

```python {exec="browser" session="nla"}
import numpy as np
import scipy.linalg as sla

A = np.array([[1e-20, 1.0],
              [1.0,   1.0]])
b = np.array([1.0, 2.0])                       # 참해 ≈ (1, 1)

# 피벗팅 없는 Gauss 소거: 1e-20을 pivot으로 쓴다
l21 = A[1, 0] / A[0, 0]                        # 1e20
u22 = A[1, 1] - l21 * A[0, 1]                  # 1 - 1e20 → -1e20 (1이 사라진다)
x2 = (b[1] - l21 * b[0]) / u22
x1 = (b[0] - A[0, 1] * x2) / A[0, 0]
print("피벗팅 없음 :", np.array([x1, x2]))

lu, piv = sla.lu_factor(A)                     # 부분 피벗팅: 큰 원소를 pivot으로
print("부분 피벗팅 :", sla.lu_solve((lu, piv), b))
print("cond(A)     :", f"{np.linalg.cond(A):.2f}")
```

```output
피벗팅 없음 : [0. 1.]
부분 피벗팅 : [1. 1.]
cond(A)     : 2.62
```

조건수가 2.62로 아주 좋은 문제인데, 피벗팅 없는 소거는 $x_1$을 완전히 틀린다. $1 - 10^{20}$을 계산하는 순간 1이 반올림으로 사라졌기 때문이다. 9.2장의 용어로 이것은 **불안정한 알고리즘**의 전형이다. 부분 피벗팅을 쓴 LU는 이론적으로 최악의 경우가 있지만 실제로 쓰이는 행렬에서는 사실상 backward stable하고, `np.linalg.solve`와 `scipy.linalg.solve`가 기본으로 쓰는 방법이다. `lu_factor`로 분해를 한 번 해 두고 `lu_solve`를 여러 번 부르면, 같은 행렬로 여러 오른쪽 변을 풀 때 분해를 반복하지 않는다.

::quiz{id="factor-once"}

**Cholesky.** SPD 행렬은 $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$로 분해할 수 있고, 피벗팅 없이도 안정하며 LU의 절반 연산으로 끝난다. 공분산 행렬, Gaussian process의 커널 행렬, 최소화 문제의 Hessian, 정규방정식 행렬 $\mathbf{X}^\top\mathbf{X}$가 모두 SPD(또는 양의 준정부호)다. Cholesky는 대각 원소의 제곱근을 계산하다가 음수를 만나면 실패하므로, **행렬이 수치적으로 양의 정부호인지 검사하는 가장 싼 방법**이기도 하다. 다음은 Gaussian process에서 흔한 상황이다.

```python {exec="browser" session="nla"}
x = np.linspace(0, 1, 50)                                  # (50,) 입력 위치
K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.3**2)  # (50, 50) RBF 커널
eps = np.finfo(float).eps                                 # 반올림 수준의 값은 환경마다 달라 판정만 찍는다
print("cond(K)*eps > 1:", np.linalg.cond(K) * eps > 1, "  최소 고유값 < 1e-12:", np.linalg.eigvalsh(K)[0] < 1e-12)
try:
    sla.cholesky(K, lower=True)
except np.linalg.LinAlgError:
    print("jitter 없음 : LinAlgError (수치적으로 양의 정부호가 아니다)")

L = sla.cholesky(K + 1e-6 * np.eye(50), lower=True)        # K + jitter*I
y = np.sin(2 * np.pi * x)                                  # (50,) 관측값
alpha = sla.cho_solve((L, True), y)                        # (K + jitter I)^-1 y
print("jitter 1e-6 : 성공, 잔차 < 1e-10:", np.linalg.norm((K + 1e-6 * np.eye(50)) @ alpha - y) < 1e-10)
```

```output
cond(K)*eps > 1: True   최소 고유값 < 1e-12: True
jitter 없음 : LinAlgError (수치적으로 양의 정부호가 아니다)
jitter 1e-6 : 성공, 잔차 < 1e-10: True
```

RBF 커널 행렬은 이론적으로 양의 정부호지만, 점이 촘촘하고 length scale이 길면 행들이 거의 같아져 가장 작은 고유값이 반올림 오차 수준이 되고, 계산값은 음수로 나오기도 한다(집필 환경에서 $10^{-15}$ 크기의 음수). 조건수도 $1/\varepsilon$을 넘어 9.2장의 기준으로는 믿을 자릿수가 없다. 대각에 작은 양수(jitter)를 더하면 모든 고유값이 그만큼 커져 분해가 성공한다. jitter는 관측 잡음 분산을 조금 더한 것과 같아서 모델을 바꾸므로, 값을 기록하고 결과가 그 크기에 민감하지 않은지 확인한다.

::quiz{id="cholesky-jitter"}

**최소제곱.** 직사각 행렬의 최소제곱은 QR이 기본이다. 정규방정식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$를 Cholesky로 풀면 가장 빠르지만 조건수가 제곱되므로(7.4장 실습, 9.2장 실습), $\kappa(\mathbf{X})$가 작다고 확인된 경우에만 쓴다. 열이 거의 종속이거나 rank가 부족하면 SVD 기반의 `np.linalg.lstsq`를 써서 작은 특잇값을 잘라 낸다.

**연구에서의 의미.** "역행렬을 계산하지 말라"는 조언은 이 절의 요약이다. $\mathbf{A}^{-1}\mathbf{b}$가 수식에 보이면 코드에서는 `solve`(또는 분해 + 대입)로, $\mathbf{B}\mathbf{A}^{-1}$이면 $\mathbf{A}^\top\mathbf{Z}^\top = \mathbf{B}^\top$을 푸는 것으로 바꾼다. 9.2장에서 본 것처럼 역행렬을 거치면 정확도가 떨어지고, 희소 행렬이면 역행렬이 꽉 차서 메모리 문제가 된다. 역행렬 자체가 답인 드문 경우(예: 공분산 행렬의 역에서 편상관을 읽는 것)에만 계산한다. Gaussian process의 log marginal likelihood에 필요한 $\log\det\mathbf{K}$도 Cholesky 인수의 대각에서 $2\sum_i \log L_{ii}$로 얻는다. `np.log(np.linalg.det(K))`는 행렬식이 쉽게 underflow·overflow하므로 쓰지 않는다.

::quiz{id="explicit-inverse"}

## 반복법

**왜 필요한가.** 3차원 격자 $100^3$ 위의 PDE는 미지수가 $10^6$개다. dense 행렬로는 저장조차 불가능하고($8 \times 10^{12}$ 바이트), 희소 직접법도 분해 과정에서 0이던 자리가 채워져(fill-in, 다음 절) 메모리와 시간이 폭발할 수 있다. 반복법은 행렬–벡터 곱 $\mathbf{v} \mapsto \mathbf{A}\mathbf{v}$만 반복해서 해에 다가간다. 희소 행렬의 곱은 0이 아닌 원소 수에 비례하는 비용이고, 필요한 정확도에서 멈출 수도 있다.

**정적 반복법.** $\mathbf{A}$를 $\mathbf{M}$(역을 적용하기 쉬운 부분)과 나머지로 나누어

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{M}^{-1}(\mathbf{b} - \mathbf{A}\mathbf{x}_k)
$$

를 반복한다. 오차는 매 반복 $\mathbf{I} - \mathbf{M}^{-1}\mathbf{A}$를 곱한 만큼 변하므로, 이 행렬의 spectral radius(고유값 절댓값의 최댓값) $\rho$가 1보다 작으면 수렴하고, 9.2장의 선형 수렴 축소율이 바로 $\rho$다. $\mathbf{M}$을 $\mathbf{A}$의 대각으로 두면 **Jacobi**, 하삼각 부분으로 두면 **Gauss–Seidel**이다. Jacobi는 엄격한 대각 지배(strictly diagonally dominant) 행렬에서, Gauss–Seidel은 엄격한 대각 지배 행렬과 SPD 행렬에서 수렴이 보장된다.

**Krylov 부분공간 방법.** 더 강력한 방법은 지금까지 만든 벡터 $\mathbf{b}, \mathbf{A}\mathbf{b}, \mathbf{A}^2\mathbf{b}, \dots, \mathbf{A}^{k-1}\mathbf{b}$가 펼치는 Krylov 부분공간 $\mathcal{K}_k$ 안에서 가장 좋은 근사해를 매번 고른다.

- **conjugate gradient**(CG, Hestenes와 Stiefel, 1952): $\mathbf{A}$가 SPD일 때, $\mathcal{K}_k$ 안에서 $\mathbf{A}$-노름 오차 $\|\mathbf{x}_k - \mathbf{x}^*\|_{\mathbf{A}} = \sqrt{(\mathbf{x}_k - \mathbf{x}^*)^\top\mathbf{A}(\mathbf{x}_k - \mathbf{x}^*)}$를 최소화한다. 반복마다 행렬–벡터 곱 한 번과 벡터 몇 개만 필요하다. 정확한 산술에서는 최대 $n$번 안에 정확한 해에 도달하고, 실제로는 다음 오차 상한이 반복 수를 정한다.

$$
\frac{\|\mathbf{x}_k - \mathbf{x}^*\|_{\mathbf{A}}}{\|\mathbf{x}_0 - \mathbf{x}^*\|_{\mathbf{A}}} \le 2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{k}.
$$

  9.2장의 gradient descent 축소율 $(\kappa-1)/(\kappa+1)$에서 $\kappa$가 $\sqrt{\kappa}$로 바뀌었다. 반복 수가 $\kappa$가 아니라 $\sqrt{\kappa}$에 비례한다는 뜻이다.
- **GMRES**: 대칭이 아닌 일반 행렬에서 $\mathcal{K}_k$ 안의 잔차 노름을 최소화한다. 지금까지의 기저 벡터를 모두 저장해야 해서 메모리가 반복 수에 비례해 늘어나므로, 보통 일정 횟수마다 다시 시작(restart)한다. BiCGSTAB 같은 방법은 짧은 점화식으로 메모리를 일정하게 유지하는 대신 수렴이 덜 매끄럽다.

1차원 Poisson 행렬(대각 2, 이웃 $-1$인 삼중대각, SPD)에 세 방법을 적용해 상대 잔차 $10^{-8}$까지의 반복 수를 센다.

```python {exec="browser" session="nla"}
import scipy.sparse.linalg as spla

n = 50
A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)       # (n, n) 1차원 Poisson
b = np.ones(n)

def stationary(A, b, M, tol=1e-8, max_iter=50_000):
    """x_{k+1} = x_k + M^{-1}(b - A x_k). M = 대각(Jacobi) 또는 하삼각(Gauss–Seidel)."""
    x = np.zeros_like(b)
    for k in range(max_iter):
        r = b - A @ x
        if np.linalg.norm(r) <= tol * np.linalg.norm(b):
            return k
        x = x + sla.solve_triangular(M, r, lower=True)
    return max_iter

print("Jacobi       반복 수:", stationary(A, b, np.diag(np.diag(A))))
print("Gauss–Seidel 반복 수:", stationary(A, b, np.tril(A)))
it = []
x_cg, info = spla.cg(A, b, rtol=1e-8, callback=lambda xk: it.append(1))
print("CG           반복 수:", len(it))
```

```output
Jacobi       반복 수: 9653
Gauss–Seidel 반복 수: 4828
CG           반복 수: 25
```

Jacobi의 축소율은 $\cos(\pi/51) \approx 0.998$이라 오차를 $10^{-8}$배로 줄이는 데 약 $\ln 10^{-8} / \ln 0.998 \approx 9700$번이 든다. Gauss–Seidel은 이 문제에서 축소율이 Jacobi의 제곱이라 절반이다. 둘 다 격자가 촘촘해질수록 반복 수가 $n^2$에 비례해 늘어 그대로는 실용적이지 않다. CG는 25번에 끝났다. $n = 50$보다도 적은 것은, 오른쪽 변 $\mathbf{b} = \mathbf{1}$이 좌우 대칭이라 대칭인 고유벡터 25개 방향의 성분만 가지고 있고, CG가 그 방향들만 처리하면 되기 때문이다. 정적 반복법은 그 자체보다 **multigrid 방법의 smoother**(고주파 오차를 빠르게 지우는 단계)로 더 많이 쓰인다.

::quiz{id="cg-requirements"}

**전처리.** CG의 반복 수가 $\sqrt{\kappa}$에 비례하므로, 9.2장의 전처리가 여기서 실용적인 의미를 갖는다. $\mathbf{M} \approx \mathbf{A}$이면서 $\mathbf{M}^{-1}\mathbf{v}$가 싼 행렬을 골라 $\kappa(\mathbf{M}^{-1}\mathbf{A})$를 줄인다. 대각(Jacobi) 전처리, 분해 과정에서 fill-in을 버리는 불완전 Cholesky·LU, PDE 행렬에서 격자 크기와 거의 무관한 반복 수를 내는 multigrid가 대표적이다. 대규모 PDE solver의 성능은 대부분 전처리가 정한다. 전처리 없는 CG를 큰 문제에 그대로 쓰고 "반복법은 느리다"고 결론 내리지 않는다.

:::note[분야 예시]
비압축성 유체 시뮬레이션은 매 시간 단계마다 압력에 대한 Poisson 방정식을 푼다. 격자가 수백만 개이고 행렬은 SPD이므로 CG(또는 그 변형)에 multigrid나 불완전 Cholesky 전처리를 붙이는 것이 표준적인 선택이다. 시간 단계 사이에 압력장이 크게 변하지 않으므로, 이전 단계의 해를 초기값으로 쓰면 반복 수가 더 줄어든다. 직접법은 분해를 재사용할 수 있는 작은 2차원 문제에서나 경쟁력이 있다.
:::

## 희소 행렬

**왜 필요한가.** PDE 이산화, 그래프(분자 결합, 소셜 네트워크, 메시), 유한요소 강성 행렬은 행마다 0이 아닌 원소가 몇 개뿐이다. 이런 **희소 행렬**(sparse matrix)을 dense로 저장하면 메모리 대부분이 0을 담는 데 쓰이고, 곱셈도 0과의 곱에 시간을 쓴다.

**정의와 저장 형식.** 0이 아닌 원소의 수를 nnz라 한다. 희소 형식은 nnz에 비례하는 메모리만 쓴다.

| 형식 | 저장 내용 | 강점 | 약점 |
|---|---|---|---|
| COO | (행, 열, 값) 세 배열 | 원소를 모아 만들기 쉽다. 중복 좌표는 합쳐진다 | 연산에는 변환이 필요 |
| CSR | 값, 열 번호, 행 시작 위치(`indptr`) | 행 단위 접근, 행렬–벡터 곱 | 원소 삽입이 비싸다 |
| CSC | 값, 행 번호, 열 시작 위치 | 열 단위 접근, 희소 직접법의 입력 | 행 접근이 비싸다 |

실무의 흐름은 "COO(또는 `sp.diags`, `sp.kron` 같은 생성 함수)로 만들고, CSR이나 CSC로 바꿔서 계산한다"이다. CSR 행렬에 원소를 하나씩 대입하면 매번 배열 전체를 다시 만들어야 해서 느리다(SciPy는 경고를 낸다). 연습문제 `csr-matvec`에서 CSR의 세 배열로 곱셈을 직접 구현한다.

**희소 직접법과 fill-in.** 희소 행렬을 LU나 Cholesky로 분해하면, 소거 중에 한 행의 배수를 다른 행에서 빼면서 원래 0이던 자리에 값이 생긴다. 이를 **fill-in**이라 한다. fill-in의 양은 행·열의 **순서**에 크게 의존하므로, 희소 직접법은 분해 전에 COLAMD·AMD·nested dissection 같은 재정렬 알고리즘으로 순서를 바꾼다. 다음 코드는 $100 \times 100$ 격자의 2차원 Poisson 행렬(5점 stencil)을 만들어 저장량과 fill-in을 재고, 같은 시스템을 희소 LU와 CG로 푼다.

```python {exec="browser" session="nla"}
import scipy.sparse as sp

m = 100                                                     # m x m 격자 → N = 10^4 미지수
T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
I = sp.identity(m)
A = (sp.kron(I, T) + sp.kron(T, I)).tocsr()                 # (N, N) 5점 Laplacian
N = A.shape[0]
csr_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
print(f"N = {N}, nnz = {A.nnz}, 밀도 = {A.nnz / N**2:.1e}")
print(f"저장: CSR {csr_mb:.2f} MB, dense {N * N * 8 / 1e9:.1f} GB")

b = np.ones(N)
for order in ["NATURAL", "COLAMD"]:
    lu = spla.splu(A.tocsc(), permc_spec=order)
    print(f"LU({order:7s}) nnz(L+U) = {lu.L.nnz + lu.U.nnz:7d} (A의 {(lu.L.nnz + lu.U.nnz) / A.nnz:.0f}배)")
x_direct = lu.solve(b)

it = []
x_cg, info = spla.cg(A, b, rtol=1e-10, callback=lambda xk: it.append(1))
print(f"CG 반복 수 = {len(it)}, 직접법과의 상대 차이 = "
      f"{np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct):.1e}")
```

```output
N = 10000, nnz = 49600, 밀도 = 5.0e-04
저장: CSR 0.64 MB, dense 0.8 GB
LU(NATURAL) nnz(L+U) = 2000198 (A의 40배)
LU(COLAMD ) nnz(L+U) =  645750 (A의 13배)
CG 반복 수 = 208, 직접법과의 상대 차이 = 3.4e-13
```

읽을 것은 세 가지다.

- **CSR은 dense보다 1000배 이상 작다.** 행렬의 0.05%만 0이 아니다.
- **희소 LU의 인수는 원래 행렬보다 훨씬 조밀하다.** 원래 순서로는 40배, COLAMD 재정렬로는 13배다. 순서 하나로 메모리가 3배 차이 난다. 3차원 문제에서는 fill-in이 훨씬 빠르게 커진다(실습).
- **CG는 A 외에 벡터 몇 개만 저장하고 같은 답을 낸다.** 반복 수 208은 $\sqrt{\kappa}$에 비례하는데, 이 행렬의 조건수는 격자 크기 $m$의 제곱에 비례하므로 반복 수는 $m$에 비례해 는다.

::quiz{id="dense-memory"}

::quiz{id="sparse-fill-in"}

**연구에서의 의미.** SciPy의 `scipy.sparse.linalg.spsolve`·`splu`는 SuperLU를 쓰며, 대칭 양의 정부호 행렬에 특화된 CHOLMOD, 대규모 병렬 직접법인 MUMPS·PARDISO 같은 전문 라이브러리도 있다. GPU에서는 cuSPARSE 같은 희소 연산 라이브러리와 PyTorch의 `torch.sparse`가 같은 형식(COO, CSR)을 쓴다. 그래프 신경망의 메시지 전달은 인접 행렬과 특징 행렬의 희소–dense 곱이다. 어떤 도구든 한 가지는 공통이다. 큰 희소 행렬에 `.toarray()`나 `.todense()`를 무심코 부르는 순간 희소성의 이점이 모두 사라진다.

## 대규모 계산

**왜 필요한가.** 어떤 행렬은 희소 형식으로도 만들 수 없거나 만들 필요가 없다. 신경망의 Jacobian과 Hessian은 원소가 파라미터 수의 제곱만큼 있지만, 자동미분은 Jacobian–벡터 곱과 Hessian–벡터 곱을 원소를 만들지 않고 계산한다(8.4장). 영상 처리의 합성곱, FFT, 시뮬레이터 한 step도 "벡터를 넣으면 벡터가 나오는" 선형 연산이다.

**행렬을 만들지 않는 계산.** 반복법이 행렬–벡터 곱만 쓴다는 점을 그대로 이용한다. 행렬 대신 곱셈 함수를 넘기는 것을 **matrix-free** 계산이라 한다. SciPy의 `LinearOperator`는 `matvec` 함수만으로 행렬처럼 동작하는 객체를 만든다. 비선형 방정식의 Newton–Krylov 방법(9.4장)은 Jacobian을 만들지 않고 Jacobian–벡터 곱만으로 Newton step의 선형시스템을 GMRES로 푼다.

**무작위 선형대수.** 큰 행렬 $\mathbf{M}$($m \times n$)의 상위 $k$개 특잇값만 필요하다면 전체 SVD($O(mn\min(m,n))$)는 낭비다. **무작위 range finder**(Halko 외, 2011)는 무작위 행렬 $\boldsymbol{\Omega}$($n \times (k+p)$)를 곱해 $\mathbf{Y} = \mathbf{M}\boldsymbol{\Omega}$를 만들고, $\mathbf{Y}$의 QR로 $\mathbf{M}$의 열공간을 근사하는 정규직교 기저 $\mathbf{Q}$를 얻는다. 그다음 작은 행렬 $\mathbf{Q}^\top\mathbf{M}$($(k+p) \times n$)의 SVD를 구한다. $p$는 5–10 정도의 여유 표본이다. 특잇값이 빠르게 감소할수록 정확하고, 그렇지 않으면 $(\mathbf{M}\mathbf{M}^\top)^q\mathbf{M}\boldsymbol{\Omega}$처럼 거듭제곱 반복을 몇 번 더해 정확도를 높인다.

```python {exec="browser" session="nla"}
def apply_laplacian(v):                                     # A @ v를 행렬 없이: 5점 stencil
    u = v.reshape(m, m)                                     # (m, m) 격자 값
    out = 4.0 * u
    out[1:, :] -= u[:-1, :]
    out[:-1, :] -= u[1:, :]
    out[:, 1:] -= u[:, :-1]
    out[:, :-1] -= u[:, 1:]
    return out.ravel()                                      # (N,)

op = spla.LinearOperator((N, N), matvec=apply_laplacian)
x_mf, info = spla.cg(op, b, rtol=1e-10)
print(f"matrix-free CG와 직접법의 상대 차이 = {np.linalg.norm(x_mf - x_direct) / np.linalg.norm(x_direct):.1e}")

rng = np.random.default_rng(0)
U0, _ = np.linalg.qr(rng.normal(size=(2000, 30)))           # (2000, 30)
V0, _ = np.linalg.qr(rng.normal(size=(500, 30)))            # (500, 30)
M = (U0 * 0.7 ** np.arange(30)) @ V0.T                      # (2000, 500) 특잇값이 빠르게 감소
k, p = 10, 10                                               # 원하는 rank, 여유 표본
Q, _ = np.linalg.qr(M @ rng.normal(size=(500, k + p)))      # (2000, k+p) 열공간의 근사 기저
s_rand = np.linalg.svd(Q.T @ M, compute_uv=False)[:k]       # (k+p, 500) 작은 행렬의 SVD
s_full = np.linalg.svd(M, compute_uv=False)[:k]
print(f"무작위 SVD 상위 {k}개 특잇값의 최대 상대 오차 = {np.max(np.abs(s_rand - s_full) / s_full):.1e}")
```

```output
matrix-free CG와 직접법의 상대 차이 = 3.4e-13
무작위 SVD 상위 10개 특잇값의 최대 상대 오차 = 2.4e-04
```

stencil 함수 하나로 앞 절의 CSR 행렬과 같은 답을 얻었다. 행렬은 어디에도 저장되지 않았다. 무작위 SVD는 $2000 \times 500$ 행렬에 무작위 벡터 20개를 곱하고 $20 \times 500$ 행렬의 SVD만 구해, 상위 10개 특잇값을 상대 오차 $10^{-4}$ 수준으로 얻었다. scikit-learn의 `PCA(svd_solver="randomized")`와 `sklearn.utils.extmath.randomized_svd`가 같은 아이디어를 쓴다.

:::note[분야 예시]
continuous normalizing flow 같은 생성모델과 확산 모델의 likelihood 계산에는 Jacobian의 trace(divergence)가 필요하다. 차원이 크면 Jacobian을 만들 수 없으므로, 무작위 벡터 $\mathbf{z}$에 대해 $\mathbb{E}[\mathbf{z}^\top\mathbf{J}\mathbf{z}] = \operatorname{tr}\mathbf{J}$인 성질(Hutchinson 추정량)을 써서 Jacobian–벡터 곱 몇 번으로 trace를 추정한다(19장). 행렬을 만들지 않는 계산과 무작위 선형대수가 함께 쓰이는 예다.
:::

## 실습

2차원·3차원 Poisson 시스템을 크기를 키워 가며 희소 LU(`splu`, COLAMD 순서)와 전처리 없는 CG(상대 잔차 $10^{-10}$)로 풀고 시간·메모리·정확도를 비교한다. 코드는 `labs/numerical-methods/numerical-linear-algebra/sparse_vs_cg.py`다. 알고 있는 해 $\mathbf{x}_{\text{true}}$에서 $\mathbf{b} = \mathbf{A}\mathbf{x}_{\text{true}}$를 만들어 오차를 잰다. LU 메모리는 인수의 nnz에 원소당 약 12바이트(값 8, 인덱스 4)를 곱한 추정값이다. 집필 시점, 한 CPU 스레드(`OMP_NUM_THREADS=1`)에서의 결과다. 시간은 컴퓨터마다 다르므로 크기에 따라 늘어나는 비율을 본다.

```output
 문제      격자        N    A[MB]    LU[MB]    LU[s]     LU 오차  CG 반복   CG[s]     CG 오차
2D   50^2     2500      0.2       1.4     0.01   1.8e-15    165    0.01   1.1e-09
2D  100^2    10000      0.6       7.7     0.03   2.8e-15    311    0.04   2.8e-09
2D  200^2    40000      2.6      41.7     0.24   5.5e-15    583    0.17   1.4e-08
2D  400^2   160000     10.2     210.5     1.17   1.8e-14   1067    1.66   3.6e-08
3D   20^3     8000      0.7      44.6     0.29   1.0e-15     86    0.01   4.5e-10
3D   30^3    27000      2.3     310.1     4.13   1.8e-15    123    0.03   1.2e-09
3D   40^3    64000      5.5    1148.8    24.37   2.1e-15    156    0.12   1.8e-09
```

결과를 읽는 법은 네 가지다.

- **2차원에서는 희소 LU가 충분히 경쟁력 있다.** $N = 1.6 \times 10^5$에서도 1초 남짓이고, 분해를 재사용하면 오른쪽 변이 여러 개일 때 더 유리하다. 메모리는 A의 약 20배다.
- **3차원에서는 fill-in이 폭발한다.** $N = 64000$은 2차원의 $N = 160000$보다 작은데, LU 인수가 1.1 GB, 시간이 24초다. 3차원 격자에서는 재정렬해도 인수의 크기가 미지수 수보다 훨씬 빠르게 커진다. 같은 문제에서 CG는 0.12초, 메모리는 A 정도다.
- **CG의 반복 수는 격자 한 변의 길이 $m$에 비례한다.** 2차원에서 $m$이 두 배가 될 때마다 반복 수가 약 두 배다($\sqrt{\kappa} \propto m$). 전처리(불완전 Cholesky, multigrid)가 이 증가를 줄이는 수단이다.
- **CG의 오차는 허용 오차보다 크다.** 상대 잔차를 $10^{-10}$까지 줄였지만 해의 오차는 $10^{-9}$–$10^{-8}$이다. 잔차와 오차의 비는 조건수까지 커질 수 있고(9.2장), 격자가 촘촘할수록 조건수가 크다. 반복법의 `rtol`은 잔차에 대한 기준이지 해의 오차에 대한 보장이 아니다.

이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="conjugate-gradient"}

::exercise{id="csr-matvec"}

## 실패 사례

- **역행렬을 명시적으로 계산하는 것.** `inv(A) @ b`는 `solve(A, b)`보다 느리고 덜 정확하며, 희소 행렬이면 꽉 찬 역행렬 때문에 메모리가 터진다. 같은 행렬로 여러 번 풀어야 하면 역행렬이 아니라 분해(`lu_factor`, `cho_factor`, `splu`)를 저장해 재사용한다. $\log\det$가 필요하면 분해의 대각에서 구한다.
- **큰 희소 행렬을 dense로 바꾸는 것.** 디버깅 중에 `.toarray()`로 바꿔 보는 코드가 그대로 남아 큰 문제에서 메모리를 다 쓰는 일이 흔하다. 희소 행렬의 크기와 nnz를 로그로 남기고, dense 변환은 작은 크기에서만 허용하도록 검사를 둔다.
- **CG를 대칭 양의 정부호가 아닌 행렬에 쓰는 것.** 대칭이 아니거나 부정부호인 행렬에서 CG는 발산하거나 엉뚱한 값에서 멈춘다. 비대칭 이산화(예: 이류 항이 있는 PDE)에는 GMRES·BiCGSTAB를 쓴다. 반복법이 반환하는 수렴 플래그(`info`)를 확인하지 않고 결과를 쓰는 것도 같은 부류의 실수다.
- **반복법의 허용 오차를 해의 오차로 읽는 것.** `rtol`은 잔차 기준이다. 조건이 나쁜 문제에서는 잔차가 작아도 해의 오차가 클 수 있으므로, 중요한 결과는 허용 오차를 한 자릿수 줄여 답이 바뀌는지 확인한다.

## 통과 기준

- 행렬의 크기·구조(dense/희소, 대칭·양의 정부호 여부, 조건수)와 오른쪽 변의 개수가 주어지면 LU·Cholesky·QR·SVD·CG·GMRES 중 알맞은 solver를 고르고, 연산량·메모리·안정성 근거로 이유를 설명한다.
- 희소 행렬을 CSR로 만들고, 같은 시스템을 희소 직접법과 CG로 풀어 시간·메모리·정확도를 측정해 비교한다.
- 연습문제 두 개(`conjugate-gradient`, `csr-matvec`)의 테스트를 통과한다.

## 참고 문헌

- Trefethen, Bau (1997). *Numerical Linear Algebra*. SIAM. — LU·Cholesky·QR·SVD의 알고리즘과 안정성, Krylov 방법 입문
- Golub, Van Loan (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. — 분해 알고리즘의 연산량과 구현
- Saad (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. — 정적 반복법, Krylov 부분공간 방법, 전처리, 희소 저장 형식
- Hestenes, Stiefel (1952). Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409–436. — CG의 원 논문
- Halko, Martinsson, Tropp (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288. — 무작위 range finder와 무작위 SVD
