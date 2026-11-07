# 좌표 하강법과 soft-thresholding으로 Lasso 풀기

Lasso 목적함수

$$
L(\mathbf{w}) = \frac{1}{2N}\lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2 + \alpha\lVert\mathbf{w}\rVert_1
$$

는 $\lVert\mathbf{w}\rVert_1$ 때문에 0에서 미분할 수 없어 닫힌 해가 없다. 그러나 **계수 하나만** 움직이고 나머지를 고정하면 문제는 1차원이 되고, 해가 닫힌 형태로 나온다. 계수를 하나씩 돌아가며 이 1차원 해로 바꾸는 것이 좌표 하강법(coordinate descent)이다.

두 함수를 완성한다. `X`는 열마다 평균이 0이라고 가정하고 절편은 다루지 않는다.

**1. `soft_threshold(z, t)`** — $S(z, t) = \operatorname{sign}(z)\max(|z| - t, 0)$. 배열에 원소별로 동작해야 한다. $|z| \le t$이면 정확히 0이다.

**2. `lasso_cd(X, y, alpha, n_sweeps=500)`** — 모든 계수를 0에서 시작해, 매 sweep마다 $j = 1, \dots, p$ 순서로

$$
\rho_j = \frac{1}{N}\mathbf{x}_j^\top\big(\mathbf{y} - \mathbf{X}\mathbf{w} + \mathbf{x}_j w_j\big), \qquad
w_j \leftarrow \frac{S(\rho_j, \alpha)}{\lVert\mathbf{x}_j\rVert^2 / N}
$$

로 바꾼다. $\rho_j$는 "$j$번째 특징을 뺀 나머지 모델의 잔차"와 $\mathbf{x}_j$의 상관이다. 이 상관이 $\alpha$보다 작으면 $w_j$는 정확히 0이 된다. 이것이 Lasso가 특징을 선택하는 방식이다. `n_sweeps`번 반복한 뒤 `w` (shape `(p,)`)를 반환한다.

테스트는 다음을 확인한다.

- `soft_threshold`의 값과 정확한 0.
- 특징이 직교하는 경우($\mathbf{X}^\top\mathbf{X}/N = \mathbf{I}$), 해가 $S(\mathbf{X}^\top\mathbf{y}/N, \alpha)$와 같다.
- $\alpha \ge \max_j |\mathbf{x}_j^\top\mathbf{y}|/N$이면 모든 계수가 0이다.
- 최적 조건(KKT): $w_j \neq 0$이면 $\mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}\mathbf{w})/N = \alpha\operatorname{sign}(w_j)$, $w_j = 0$이면 $|\mathbf{x}_j^\top(\mathbf{y} - \mathbf{X}\mathbf{w})/N| \le \alpha$.
