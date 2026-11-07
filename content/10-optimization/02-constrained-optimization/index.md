---
id: optimization/constrained-optimization
title: "제약 최적화와 쌍대성"
title_en: "Constrained Optimization and Duality"
tier: core
status: draft
prerequisites: [optimization/optimization-basics]
---

# 제약 최적화와 쌍대성

과학 문제의 해는 대개 아무 값이나 될 수 없다. 농도는 음수가 아니고, 성분 비율의 합은 1이며, 에너지와 질량은 보존된다. 10.1장에서 본 것처럼 제약이 있으면 최적해가 feasible set의 경계에 놓이고, 그 점에서 gradient는 0이 아니다. 그러면 "gradient가 0인 점을 찾는다"는 무제약 최적화의 처방을 그대로 쓸 수 없다. 이 장은 제약이 있는 문제의 최적성 조건(Lagrange 승수와 KKT 조건)을 세우고, 그 승수가 무엇을 뜻하는지 쌍대성으로 해석한다.

이 장의 두 번째 주제는 머신러닝에서 제약이 숨어 있는 방식이다. 신경망 학습은 거의 언제나 무제약 문제로 쓰지만, 손실에 더하는 정규화 항은 사실상 제약의 다른 표현이다. Ridge 회귀의 $\lambda$와 노름 제약의 반지름 $r$은 KKT 조건으로 연결된다. 이 연결을 알면 정규화 계수를 "얼마나 작은 파라미터를 원하는가"라는 제약으로 읽을 수 있고, penalty·투영·근위 연산 중 무엇으로 제약을 다룰지 근거를 갖고 고를 수 있다.

## 학습 목표

- Lagrangian과 KKT 조건을 사용한다
- 제약과 penalty·정규화의 관계를 이해한다

## Lagrange 승수

등식 제약만 있는 문제부터 본다.

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad h_j(\mathbf{x}) = 0, \quad j = 1, \dots, p.
$$

**기하적 해석.** 제약이 하나뿐인 2차원 문제를 생각한다. 제약 $h(\mathbf{x}) = 0$은 평면 위의 곡선이고, 우리는 이 곡선 위를 따라가며 $f$가 가장 작은 점을 찾는다. 곡선 위의 어떤 점에서 $f$의 등고선이 곡선을 **가로지르면**, 곡선을 따라 한쪽으로 움직여 더 낮은 등고선으로 갈 수 있으므로 그 점은 최적이 아니다. 따라서 최적해에서는 등고선이 제약 곡선에 **접한다**. 접한다는 것은 두 곡선의 법선이 평행하다는 뜻이고, $f$의 등고선의 법선은 $\nabla f$, 제약 곡선의 법선은 $\nabla h$다.

$$
\nabla f(\mathbf{x}^\star) + \nu\,\nabla h(\mathbf{x}^\star) = \mathbf{0}.
$$

이 비례 상수 $\nu$를 **Lagrange 승수**(Lagrange multiplier)라 한다. 제약이 여러 개이면 $\nabla f$가 제약 법선들의 일차결합이 된다. 이 조건을 한 함수로 묶은 것이 **Lagrangian**이다.

$$
L(\mathbf{x}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{j=1}^{p} \nu_j h_j(\mathbf{x}).
$$

$\nabla_{\mathbf{x}} L = \mathbf{0}$은 위의 평행 조건이고, $\nabla_{\boldsymbol{\nu}} L = \mathbf{0}$은 제약 $h_j(\mathbf{x}) = 0$ 자체다. 제약이 있는 문제가 $L$의 stationary point를 찾는 문제로 바뀐다. 이 조건은 최적해에서 제약 gradient들이 일차독립일 때(regularity 조건) 필요조건이다.

예로 $\min x^2 + 2y^2$ s.t. $x + y = b$를 푼다. $\nabla_{\mathbf{x}} L = (2x + \nu,\ 4y + \nu) = \mathbf{0}$과 $x + y = b$는 미지수 세 개의 선형 연립방정식이다.

```python {exec="browser" session="constrained"}
import numpy as np

# min x^2 + 2y^2  s.t.  x + y = b.   L = x^2 + 2y^2 + nu (x + y - b)
def solve(b):
    K = np.array([[2.0, 0.0, 1.0],      # dL/dx = 2x + nu = 0
                  [0.0, 4.0, 1.0],      # dL/dy = 4y + nu = 0
                  [1.0, 1.0, 0.0]])     # x + y = b
    x, y, nu = np.linalg.solve(K, [0.0, 0.0, b])
    return x, y, nu, x**2 + 2 * y**2

x, y, nu, p_star = solve(1.0)
print(f"x = {x:.4f}, y = {y:.4f}, nu = {nu:.4f}, f* = {p_star:.4f}")
print("grad f =", np.array([2 * x, 4 * y]).round(4), "  grad h = [1. 1.]")
d = 1e-3
print(f"df*/db (수치) = {(solve(1 + d)[3] - solve(1 - d)[3]) / (2 * d):.4f}   -nu = {-nu:.4f}")
```

```output
x = 0.6667, y = 0.3333, nu = -1.3333, f* = 0.6667
grad f = [1.3333 1.3333]   grad h = [1. 1.]
df*/db (수치) = 1.3333   -nu = 1.3333
```

$b = 1$일 때 해는 $(2/3, 1/3)$이고, 그 점에서 $\nabla f = (4/3, 4/3)$은 0이 아니지만 $\nabla h = (1, 1)$과 평행하다.

::quiz{id="lagrange-gradient-parallel"}

**승수의 의미.** 마지막 줄이 승수의 해석이다. 제약의 우변 $b$를 조금 바꾸면 최적값 $f^\star(b)$가 바뀌는데, 그 변화율이 $-\nu$와 같다.

$$
\frac{d f^\star}{d b} = -\nu.
$$

여기서는 $f^\star(b) = \frac{2}{3}b^2$이므로 $b = 1$에서 $4/3$이고, 승수 $\nu = -4/3$의 부호를 바꾼 값과 같다. 승수는 "제약을 한 단위 풀어 주면 목적함수를 얼마나 개선할 수 있는가", 즉 제약의 **값**이다. 경제학에서는 이를 shadow price라 부른다. 등식 제약의 승수는 부호에 제한이 없다. $h = 0$을 $-h = 0$으로 써도 같은 제약이므로 승수의 부호는 표기에 달려 있다.

:::note[분야 예시]
통계역학의 Boltzmann 분포는 Lagrange 승수 문제의 해다. 에너지 준위 $E_i$를 갖는 상태들의 확률 $p_i$에 대해, 엔트로피 $-\sum_i p_i \log p_i$를 정규화 $\sum_i p_i = 1$과 평균 에너지 $\sum_i p_i E_i = U$라는 두 등식 제약 아래에서 최대화한다. Lagrangian의 $p_i$에 대한 stationarity 조건에서 $p_i \propto \exp(-\beta E_i)$가 나오고, 평균 에너지 제약의 승수 $\beta$가 역온도 $1/(k_B T)$로 해석된다. 온도는 "평균 에너지 제약의 값"이다.
:::

## KKT 조건

부등식 제약 $g_i(\mathbf{x}) \le 0$이 있으면 상황이 하나 더 늘어난다. 최적해에서 제약은 **활성**(active, $g_i = 0$, 경계 위)이거나 **비활성**(inactive, $g_i < 0$, 경계 안쪽)이다. 비활성 제약은 최적해 근처에서 아무 역할도 하지 않으므로 없는 것처럼 다뤄야 하고, 활성 제약은 등식 제약처럼 다뤄야 한다. 그런데 어느 제약이 활성인지는 풀기 전에는 모른다. 이 두 경우를 한 번에 표현한 것이 **KKT 조건**(Karush–Kuhn–Tucker conditions)이다. Lagrangian을

$$
L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})
$$

로 두면, 최적해 $\mathbf{x}^\star$와 승수 $\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star$는 다음 네 조건을 만족한다.

1. **stationarity**: $\nabla f(\mathbf{x}^\star) + \sum_i \lambda_i^\star \nabla g_i(\mathbf{x}^\star) + \sum_j \nu_j^\star \nabla h_j(\mathbf{x}^\star) = \mathbf{0}$
2. **primal feasibility**: $g_i(\mathbf{x}^\star) \le 0$, $h_j(\mathbf{x}^\star) = 0$
3. **dual feasibility**: $\lambda_i^\star \ge 0$
4. **상보적 여유**(complementary slackness): $\lambda_i^\star\, g_i(\mathbf{x}^\star) = 0$

상보적 여유가 활성·비활성의 구분을 담는다. 각 부등식 제약에 대해 승수가 0이거나($\lambda_i = 0$, 제약이 해를 밀어내지 않음) 제약이 등호로 성립한다($g_i = 0$, 경계 위). dual feasibility $\lambda_i \ge 0$은 방향을 정한다. 부등식 제약은 feasible set 안쪽으로만 밀 수 있으므로, $-\nabla f$가 feasible set **바깥**을 향해야(즉 내려가는 방향이 막혀 있어야) 최적이다.

KKT 조건은 적절한 제약 자격 조건(constraint qualification) 아래에서 최적해의 필요조건이고, 문제가 볼록이면(목적함수와 $g_i$가 볼록, $h_j$가 아핀) 충분조건이기도 하다. 볼록 문제에서는 KKT 조건을 만족하는 점을 찾으면 그것이 전역 최적해다.

10.1장의 첫 문제를 일반화해, 점 $\mathbf{p} = (2, 1)$에서 반공간 $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le c\}$까지 가장 가까운 점을 찾는다. stationarity $2(\mathbf{x} - \mathbf{p}) + \mu\mathbf{a} = \mathbf{0}$에서 $\mathbf{x} = \mathbf{p} - \frac{\mu}{2}\mathbf{a}$이고, 제약이 활성이면 $\mathbf{a}^\top\mathbf{x} = c$에서 $\mu$가 정해진다. $c$를 바꿔 가며 승수와 여유를 본다.

```python {exec="browser" session="constrained"}
# min ||x - p||^2  s.t.  a^T x <= c   (반공간으로의 투영)
p, a = np.array([2.0, 1.0]), np.array([1.0, 1.0])
for c in [0.0, 1.0, 2.0, 3.0, 4.0]:
    mu = max(0.0, 2 * (a @ p - c) / (a @ a))        # 승수: 제약이 활성일 때만 양수
    x = p - mu / 2 * a                              # stationarity: 2(x - p) + mu a = 0
    slack = c - a @ x                               # 여유: 0 이상
    print(f"c = {c:.0f}  x* = {x}  mu = {mu:.1f}  여유 = {slack:.1f}  "
          f"mu*여유 = {mu * slack:.1f}  f* = {(x - p) @ (x - p):.2f}")
```

```output
c = 0  x* = [ 0.5 -0.5]  mu = 3.0  여유 = 0.0  mu*여유 = 0.0  f* = 4.50
c = 1  x* = [1. 0.]  mu = 2.0  여유 = 0.0  mu*여유 = 0.0  f* = 2.00
c = 2  x* = [1.5 0.5]  mu = 1.0  여유 = 0.0  mu*여유 = 0.0  f* = 0.50
c = 3  x* = [2. 1.]  mu = 0.0  여유 = 0.0  mu*여유 = 0.0  f* = 0.00
c = 4  x* = [2. 1.]  mu = 0.0  여유 = 1.0  mu*여유 = 0.0  f* = 0.00
```

표의 각 줄이 KKT 조건의 한 경우다.

- $c = 0, 1, 2$: $\mathbf{p}$가 반공간 밖에 있어 제약이 활성이다(여유 0). 승수 $\mu$는 양수이고, 제약이 느슨해질수록($c$가 클수록) 줄어든다. $c = 1$은 10.1장의 예와 같은 해 $(1, 0)$이다.
- $c = 4$: $\mathbf{p}$가 이미 반공간 안에 있다. 여유가 1이므로 상보적 여유 때문에 $\mu = 0$이다.
- $c = 3$: 경계 사례다. 제약이 등호로 성립하지만 $\mu = 0$이다. 제약이 해에 닿아 있지만 밀지는 않는다.

등식 제약의 경우처럼 승수는 민감도다. 이 문제의 최적값은 $f^\star(c) = (3 - c)^2/2$($c \le 3$)이므로 $df^\star/dc = -(3 - c) = -\mu$다. 표에서 $c$를 1 늘릴 때마다 $f^\star$가 줄어드는 정도가 승수의 크기를 따라간다.

::quiz{id="multiplier-value"}

::quiz{id="inactive-constraint"}

:::note[분야 예시]
화학 평형 계산은 주어진 온도·압력에서 Gibbs 자유에너지를 최소화하는 문제로 쓸 수 있다. 결정변수는 각 화학종의 몰수이고, 원소별 원자 수 보존이 등식 제약, 몰수가 음수가 아니라는 조건이 부등식 제약이다. 등식 제약의 Lagrange 승수는 원소별 화학 퍼텐셜로 해석된다. 존재할 수 있지만 평형에서 거의 생기지 않는 화학종은 몰수 제약이 활성(몰수 0)이 되는 경우이고, KKT의 상보적 여유가 이를 표현한다.
:::

## 쌍대성

Lagrangian을 $\mathbf{x}$에 대해 먼저 최소화하면 승수만의 함수가 남는다. 이를 **쌍대 함수**(dual function)라 한다.

$$
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \min_{\mathbf{x}} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}).
$$

쌍대 함수에는 원 문제가 볼록이든 아니든 성립하는 두 성질이 있다.

- **쌍대 함수는 항상 오목하다.** $\boldsymbol{\lambda}, \boldsymbol{\nu}$에 대한 아핀 함수들의 점별 최솟값이기 때문이다.
- **약쌍대성**(weak duality): $\boldsymbol{\lambda} \ge \mathbf{0}$이면 $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^\star$. feasible point $\mathbf{x}$에서는 $\lambda_i g_i(\mathbf{x}) \le 0$, $\nu_j h_j(\mathbf{x}) = 0$이므로 $L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le f(\mathbf{x})$이고, $g$는 그 최솟값이므로 더 작다.

그래서 쌍대 함수의 값은 언제나 최적값의 **하한**이다. 가장 좋은 하한을 찾는 문제

$$
\max_{\boldsymbol{\lambda} \ge \mathbf{0},\ \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})
$$

를 **쌍대 문제**(dual problem)라 하고, 그 최적값 $d^\star$와 $p^\star$의 차이 $p^\star - d^\star \ge 0$을 **쌍대 간극**(duality gap)이라 한다. 간극이 0이면 **강쌍대성**(strong duality)이 성립한다고 한다. 볼록 문제에서는 Slater 조건(모든 부등식 제약을 엄격하게 만족하는 feasible point가 존재) 같은 가벼운 조건 아래 강쌍대성이 성립한다. 볼록이 아닌 문제에서는 일반적으로 간극이 있다.

앞 절의 등식 제약 문제($\min x^2 + 2y^2$ s.t. $x + y = 1$)에서 쌍대 함수를 직접 구하면, $\nabla_{\mathbf{x}} L = \mathbf{0}$에서 $x = -\nu/2$, $y = -\nu/4$이고 이를 $L$에 넣어 $g(\nu) = -\frac{3}{8}\nu^2 - \nu$를 얻는다.

```python {exec="browser" session="constrained"}
# cell 1의 문제: g(nu) = min_{x,y} L(x, y, nu) = -3 nu^2 / 8 - nu
nus = np.linspace(-4, 2, 601)
g = -3 * nus**2 / 8 - nus
i = np.argmax(g)
print(f"max g(nu) = {g[i]:.4f}  (nu = {nus[i]:.2f})   p* = {p_star:.4f}")
print("모든 nu에서 g(nu) <= p* :", bool(np.all(g <= p_star + 1e-12)))
for nu0 in [-3.0, -1.0, 1.0]:
    print(f"g({nu0:+.0f}) = {-3 * nu0**2 / 8 - nu0:+.4f}  <= p*")
```

```output
max g(nu) = 0.6667  (nu = -1.33)   p* = 0.6667
모든 nu에서 g(nu) <= p* : True
g(-3) = -0.3750  <= p*
g(-1) = +0.6250  <= p*
g(+1) = -1.3750  <= p*
```

$g$는 오목한 이차함수이고, 모든 $\nu$에서 $p^\star = 2/3$ 아래에 있다(약쌍대성). 최댓값은 $\nu = -4/3$에서 정확히 $2/3$이다(강쌍대성). 이 최대점의 $\nu$는 첫 절에서 구한 Lagrange 승수와 같다.

::quiz{id="weak-duality"}

쌍대성이 연구에서 쓰이는 방식은 셋이다. 첫째, **하한 인증서**다. 어떤 $\boldsymbol{\lambda} \ge \mathbf{0}$에서든 $g$를 계산하면 $p^\star$의 하한을 얻으므로, 현재 해 $f(\mathbf{x})$와 $g$의 차이로 "최적값에서 최대 얼마나 떨어져 있는가"를 보증할 수 있다. 볼록 최적화 solver가 멈출 때 보고하는 duality gap이 이것이다. 둘째, **다른 형태의 문제**다. 서포트 벡터 머신(SVM)의 쌍대 문제는 데이터 사이의 내적만으로 쓰이므로 커널 방법(16.3장)으로 이어진다. 셋째, **승수의 해석**이다. 강쌍대성이 성립하면 쌍대 문제의 해가 곧 각 제약의 민감도다.

## 제약과 penalty

제약을 직접 다루는 알고리즘은 복잡하다. 그래서 흔히 제약을 목적함수의 항으로 옮겨 무제약 문제로 바꾼다. 방법은 세 가지가 대표적이다.

- **penalty 방법**: 제약 위반에 벌점을 준다. 등식 제약은 $f(\mathbf{x}) + \frac{\rho}{2}\sum_j h_j(\mathbf{x})^2$, 부등식 제약은 $\frac{\rho}{2}\sum_i \max(0, g_i(\mathbf{x}))^2$를 더한다. feasible set 밖에서 출발해도 되지만, 유한한 $\rho$의 해는 제약을 정확히 만족하지 않는다.
- **barrier 방법**: 부등식 제약의 경계에 다가갈수록 무한대로 커지는 항, 예를 들어 $-\frac{1}{t}\sum_i \log(-g_i(\mathbf{x}))$를 더한다. 해가 항상 feasible set 안쪽에 머물고, $t$를 키우며 경계에 다가간다. 내부점 방법(interior-point method)의 기초다.
- **augmented Lagrangian**: Lagrangian에 이차 penalty를 더한 $L_\rho(\mathbf{x}, \boldsymbol{\nu}) = f + \sum_j \nu_j h_j + \frac{\rho}{2}\sum_j h_j^2$를 $\mathbf{x}$에 대해 최소화하고, 승수를 $\nu_j \leftarrow \nu_j + \rho\,h_j(\mathbf{x})$로 갱신하기를 반복한다. 승수가 제약을 맞추는 역할을 맡으므로 $\rho$를 무한히 키우지 않아도 된다.

첫 절의 문제를 penalty 방법과 augmented Lagrangian으로 풀어 제약 위반과 Hessian 조건수를 비교한다.

```python {exec="browser" session="constrained"}
# cell 1의 문제를 penalty로: min x^2 + 2y^2 + (rho/2)(x + y - 1)^2 + nu (x + y - 1)
def penalized(rho, nu=0.0):
    H = np.array([[2 + rho, rho], [rho, 4 + rho]])          # (2, 2) Hessian
    x, y = np.linalg.solve(H, [rho - nu, rho - nu])
    return x + y - 1, np.linalg.cond(H)                     # 제약 위반, 조건수

for rho in [1, 10, 100, 1e4, 1e6]:
    viol, cond = penalized(rho)
    print(f"penalty  rho = {rho:>9.0f}  위반 = {viol:+.2e}  조건수 = {cond:.1e}")

nu = 0.0
for k in range(1, 6):
    viol, cond = penalized(10.0, nu)
    nu += 10.0 * viol                                       # 승수 갱신
    print(f"AL 반복 {k}  rho = 10  위반 = {viol:+.2e}  조건수 = {cond:.1e}  nu = {nu:.4f}")
```

```output
penalty  rho =         1  위반 = -5.71e-01  조건수 = 2.1e+00
penalty  rho =        10  위반 = -1.18e-01  조건수 = 7.8e+00
penalty  rho =       100  위반 = -1.32e-02  조건수 = 6.8e+01
penalty  rho =     10000  위반 = -1.33e-04  조건수 = 6.7e+03
penalty  rho =   1000000  위반 = -1.33e-06  조건수 = 6.7e+05
AL 반복 1  rho = 10  위반 = -1.18e-01  조건수 = 7.8e+00  nu = -1.1765
AL 반복 2  rho = 10  위반 = -1.38e-02  조건수 = 7.8e+00  nu = -1.3149
AL 반복 3  rho = 10  위반 = -1.63e-03  조건수 = 7.8e+00  nu = -1.3312
AL 반복 4  rho = 10  위반 = -1.92e-04  조건수 = 7.8e+00  nu = -1.3331
AL 반복 5  rho = 10  위반 = -2.25e-05  조건수 = 7.8e+00  nu = -1.3333
```

penalty 방법에서 위반은 정확히 $-4/(4 + 3\rho)$로, $\rho$를 100배 키울 때마다 약 100배 줄어든다. 그러나 $\rho = 10^6$에서도 0이 아니다. 같은 비율로 Hessian의 조건수가 커진다. penalty 항이 제약의 법선 방향에만 $\rho$ 크기의 곡률을 더하기 때문이다. 10.3장에서 보듯 경사하강의 반복 수는 조건수에 비례하므로, 큰 $\rho$는 수렴을 그만큼 느리게 만든다. augmented Lagrangian은 $\rho = 10$에 고정한 채로 반복마다 위반을 약 8분의 1로 줄이고, 승수 $\nu$는 첫 절의 Lagrange 승수 $-4/3$으로 수렴한다.

::quiz{id="penalty-not-exact"}

**정규화 항과 노름 제약의 동치.** 머신러닝의 정규화는 penalty 방법과 모양이 같다. Ridge 회귀

$$
\min_{\mathbf{w}} \ L(\mathbf{w}) + \frac{\lambda}{2}\lVert \mathbf{w} \rVert^2
$$

와 노름 제약 문제

$$
\min_{\mathbf{w}} \ L(\mathbf{w}) \quad \text{subject to} \quad \lVert \mathbf{w} \rVert^2 \le r^2
$$

를 비교한다. 제약 문제의 Lagrangian은 $L(\mathbf{w}) + \frac{\mu}{2}(\lVert \mathbf{w} \rVert^2 - r^2)$이고(승수를 $\mu/2$로 잡았다), stationarity 조건은 $\nabla L(\mathbf{w}) + \mu\mathbf{w} = \mathbf{0}$이다. 이는 Ridge의 1계 조건 $\nabla L(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$과 $\mu = \lambda$일 때 똑같다. 따라서 $L$이 볼록일 때, Ridge 해 $\mathbf{w}_\lambda$는 $r = \lVert \mathbf{w}_\lambda \rVert$인 제약 문제의 해이고, 그 KKT 승수는 $\lambda$다. 제약이 활성이므로 상보적 여유도 만족한다. 거꾸로 제약이 활성인 제약 문제의 해는 승수 $\mu$를 $\lambda$로 둔 Ridge의 해다. 큰 $\lambda$는 작은 반지름 $r$에 대응한다.

::quiz{id="ridge-radius"}

다만 두 형태는 **같은 해 집합**을 주지만 **같은 하이퍼파라미터**는 아니다. $\lambda$와 $r$의 대응 관계는 데이터에 따라 달라진다. 데이터를 바꾸면 같은 $\lambda$가 다른 $r$에 대응한다. 물리적으로 의미 있는 상한(예: 파라미터 크기의 사전 지식)이 있다면 제약 형태로 쓰는 편이 해석하기 쉽고, 하이퍼파라미터를 교차검증으로 고를 거라면 penalty 형태가 계산하기 쉽다.

## 투영과 근위 연산

제약을 penalty로 바꾸지 않고 직접 다루는 1차 방법도 있다. 두 가지가 기본이다.

**Projected gradient.** feasible set $\mathcal{C}$가 볼록이고 그 위로의 투영 $\Pi_{\mathcal{C}}(\mathbf{v}) = \arg\min_{\mathbf{x} \in \mathcal{C}} \lVert \mathbf{x} - \mathbf{v} \rVert$를 쉽게 계산할 수 있으면, gradient step 뒤에 투영한다.

$$
\mathbf{x}_{k+1} = \Pi_{\mathcal{C}}\big(\mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)\big).
$$

상자 $[l, u]$로의 투영은 좌표별로 잘라 내기(clip), 반지름 $r$인 공으로의 투영은 노름이 $r$보다 크면 $r/\lVert \mathbf{v} \rVert$배로 줄이기, 반공간으로의 투영은 앞 절에서 KKT로 구한 식이다. 반복마다 해가 feasible하다는 것이 장점이다. 실습에서 Ridge의 노름 제약 형태를 이 방법으로 푼다.

**근위 연산.** 목적함수가 매끄러운 부분 $f$와 매끄럽지 않은 부분 $R$의 합 $f(\mathbf{x}) + R(\mathbf{x})$일 때, $R$의 **근위 연산**(proximal operator)

$$
\operatorname{prox}_{tR}(\mathbf{v}) = \arg\min_{\mathbf{x}} \Big( R(\mathbf{x}) + \frac{1}{2t}\lVert \mathbf{x} - \mathbf{v} \rVert^2 \Big)
$$

를 쓴다. "$\mathbf{v}$에서 너무 멀리 가지 않으면서 $R$을 줄이는 점"이다. $R$이 집합 $\mathcal{C}$의 지시함수(안에서 0, 밖에서 $+\infty$)이면 근위 연산은 투영과 같다. 즉 근위 연산은 투영의 일반화다. proximal gradient 방법은 $f$에 대해 gradient step, $R$에 대해 근위 연산을 번갈아 적용한다.

$$
\mathbf{x}_{k+1} = \operatorname{prox}_{\eta R}\big(\mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)\big).
$$

$R(\mathbf{x}) = \lambda\lVert \mathbf{x} \rVert_1$이면 근위 연산은 좌표별 **soft thresholding**이다.

$$
\big[\operatorname{prox}_{t\lambda\lVert\cdot\rVert_1}(\mathbf{v})\big]_i = \operatorname{sign}(v_i)\max(\lvert v_i \rvert - t\lambda,\ 0).
$$

절댓값이 $t\lambda$ 이하인 성분은 정확히 0이 되고, 나머지는 $t\lambda$만큼 0 쪽으로 줄어든다. 이 반복을 Lasso에 적용한 것이 ISTA(iterative shrinkage-thresholding algorithm)다. 20개 계수 중 3개만 0이 아닌 참 모델에서 생성한 데이터로 확인한다.

```python {exec="browser" session="constrained"}
rng = np.random.default_rng(0)
A = rng.normal(size=(50, 20))                               # (N, D)
w_true = np.zeros(20)
w_true[[2, 7, 11]] = [1.5, -2.0, 1.0]
y = A @ w_true + 0.1 * rng.normal(size=50)                  # (N,)

def soft(v, t):                                             # prox of t * ||.||_1
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

lam = 5.0
step = 1 / np.linalg.norm(A, 2) ** 2                        # 1 / L, L = sigma_max(A)^2
w = np.zeros(20)
for _ in range(500):                                        # ISTA: 0.5||Aw - y||^2 + lam ||w||_1
    w = soft(w - step * A.T @ (A @ w - y), step * lam)
w_ridge = np.linalg.solve(A.T @ A + lam * np.eye(20), A.T @ y)
print("L1 해의 0이 아닌 좌표 :", np.flatnonzero(w), " 값", w[w != 0].round(3))
print("L1 해의 정확한 0 개수 :", int(np.sum(w == 0)), "/ 20")
print("Ridge 해의 정확한 0 개수:", int(np.sum(w_ridge == 0)), "/ 20")
```

```output
L1 해의 0이 아닌 좌표 : [ 2  7 11]  값 [ 1.355 -1.823  0.873]
L1 해의 정확한 0 개수 : 17 / 20
Ridge 해의 정확한 0 개수: 0 / 20
```

L1 해는 참 모델의 0이 아닌 좌표 2, 7, 11을 정확히 찾고 나머지 17개를 **정확히** 0으로 만든다. 같은 $\lambda$의 Ridge 해에는 정확한 0이 하나도 없다. L2의 근위 연산 $\mathbf{v}/(1 + t\lambda)$은 모든 성분을 같은 비율로 줄일 뿐이기 때문이다. L1 해의 0이 아닌 값(1.355, −1.823, 0.873)은 참값(1.5, −2.0, 1.0)보다 크기가 작다. soft thresholding이 살아남은 계수도 0 쪽으로 줄이기 때문이고, 이를 L1 정규화의 수축 편향(shrinkage bias)이라 한다.

::quiz{id="l1-exact-zeros"}

:::note[분야 예시]
희소성 가정을 L1 항으로 넣는 방법은 신호 복원에서 널리 쓰인다. 압축 센싱(compressed sensing) MRI 재구성은 측정 횟수를 줄인 데이터에서, 영상이 어떤 변환 영역(예: wavelet)에서 희소하다는 가정을 L1 항으로 넣어 영상을 복원한다. 동역학계 식별에서도 후보 함수 사전(dictionary)의 계수에 희소성을 가정해, 데이터를 설명하는 데 필요한 최소한의 항만 남기는 방식이 쓰인다. 두 문제 모두 L1 정규화 문제로 쓸 수 있고, 이 절의 proximal gradient가 그 기본 풀이법 중 하나다.
:::

## 실습

같은 문제를 제약식과 penalty 방식으로 풀어 해와 수렴을 비교한다. 코드는 `labs/optimization/constrained-optimization/penalty_vs_constraint.py`다. 열들이 서로 상관된 선형회귀 데이터($N = 100$, $D = 10$)에서 손실 $L(\mathbf{w}) = \lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2/(2N)$을 세 방식으로 최소화한다.

1. Ridge(penalty) 형태를 닫힌 해와 경사하강으로 푼다.
2. $r = \lVert \mathbf{w}_{\text{ridge}}(\lambda) \rVert$로 둔 노름 제약 형태를 projected gradient로 풀고, 해에서 KKT 승수 $\mu = -\nabla L(\mathbf{w})^\top\mathbf{w} / \lVert \mathbf{w} \rVert^2$를 계산한다.
3. 같은 노름 제약을 이차 penalty $\frac{\rho}{2}\max(0, \lVert \mathbf{w} \rVert^2 - r^2)^2$로 바꾸어 경사하강으로 푼다.

수렴 판정은 반복 사이 이동량/학습률이 $10^{-8}$ 미만일 때다. 집필 시점의 실행 결과는 다음과 같다.

```output
데이터 N=100, D=10, X^T X / N의 조건수 31.7

1) Ridge(penalty) 해와 노름 제약 해의 비교
   lam    r=||w_ridge||   ||w_pen - w_con||   KKT 승수 mu   GD 반복(penalty)  PGD 반복(제약)
   0.01         2.2029            2.10e-08        0.0100               535             417
   0.10         1.8782            1.51e-08        0.1000               424             338
   1.00         0.8358            3.23e-09        1.0000               143             126
  10.00         0.1390            5.06e-10       10.0000                27              22

2) 이차 penalty 방법으로 같은 노름 제약을 풀 때 (lam = 1에 해당하는 r)
   rho        제약 위반 ||w||^2 - r^2   ||w - w*||   GD 반복
        1                3.453e-01     1.90e-01       238
       10                4.724e-02     2.83e-02       807
      100                4.970e-03     3.02e-03      6727
     1000                4.997e-04     3.04e-04     65957
```

결과를 읽는 법은 세 가지다.

- **두 형태의 해는 수렴 판정 오차 수준($10^{-8}$ 이하)에서 같고, 제약 형태의 KKT 승수는 $\lambda$와 소수점 넷째 자리까지 같다.** 앞 절에서 유도한 동치 $\mu = \lambda$를 수치로 확인한 것이다. $\lambda$가 커질수록 대응하는 반지름 $r$은 2.20에서 0.14로 줄어든다.
- **이차 penalty로 노름 제약을 풀면 $\rho$를 10배 키울 때마다 위반은 약 10분의 1로 줄지만, 반복 수는 약 10배 늘어난다.** 안정한 학습률이 penalty 곡률 $\rho$에 반비례해 작아져야 하기 때문이다. $\rho = 1000$에서도 해는 참 제약 해에서 $3 \times 10^{-4}$만큼 떨어져 있다.
- **투영을 쉽게 계산할 수 있는 제약이라면 projected gradient가 penalty보다 낫다.** 이 실험에서 projected gradient는 Ridge의 경사하강보다도 적은 반복으로 같은 해에 도달했다. 공 위로의 투영은 노름을 재서 나누는 계산뿐이다.

이어서 두 연습문제를 푼다.

::exercise{id="halfspace-projection"}

::exercise{id="soft-threshold-ista"}

## 실패 사례

- **penalty 계수를 키우기만 하면 제약이 만족된다고 생각하는 것.** 이차 penalty의 해는 유한한 $\rho$에서 항상 제약을 조금 어긴다. 위반은 약 $1/\rho$로 줄지만, 그 대가로 조건수가 $\rho$에 비례해 커져 최적화가 느려지거나 수치적으로 불안정해진다. 질량 보존처럼 정확히 지켜야 하는 제약은 투영, augmented Lagrangian, 또는 제약을 구조적으로 만족하는 매개화(예: softmax로 확률 단체 위의 값 만들기)로 다룬다.
- **승수의 부호와 의미를 확인하지 않는 것.** 부등식 제약의 승수가 음수로 나오면 KKT 조건이 깨진 것이고, 해가 틀렸거나 제약의 방향($g \le 0$)을 반대로 쓴 것이다. 승수는 민감도이므로, 크기가 비정상적으로 큰 제약은 해를 강하게 밀고 있는 제약이다. 그 제약이 물리적으로 타당한지 다시 검토할 신호다.
- **정규화 계수를 데이터와 무관한 상수로 옮겨 쓰는 것.** 같은 $\lambda$라도 데이터의 크기·단위·샘플 수가 바뀌면 대응하는 노름 제약 $r$이 달라진다. 다른 데이터셋에서 쓴 $\lambda$를 그대로 가져오지 말고 다시 고른다.

## 통과 기준

- Ridge 회귀를 노름 제약 문제로 다시 쓰고, KKT stationarity 조건으로 $\mu = \lambda$가 되는 이유를 설명한다. 실습 결과에서 두 해와 승수가 일치하는지 확인한다.
- 이차 penalty와 augmented Lagrangian에서 제약 위반이 어떻게 줄어드는지 수치로 보이고 차이를 설명한다.
- 연습문제 두 개(`halfspace-projection`, `soft-threshold-ista`)의 테스트를 통과한다.

## 참고 문헌

- Boyd, Vandenberghe (2004). *Convex Optimization*. Cambridge University Press. — Lagrangian, 쌍대 함수, 약·강 쌍대성, Slater 조건, KKT 조건
- Nocedal, Wright (2006). *Numerical Optimization* (2nd ed.). Springer. — 제약 최적화의 최적성 조건, penalty·augmented Lagrangian 방법
- Parikh, Boyd (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1(3), 127–239. — 근위 연산의 정의와 성질, proximal gradient 방법, Lasso 예
