# 덩어리로 평균과 분산 구하기

메모리보다 큰 데이터는 한 번에 읽지 못한다. 덩어리(chunk)를 하나씩 읽어 통계량을 갱신하고, 읽은 덩어리는 버린다. 평균은 합과 개수만 들고 다니면 되지만, 분산은 조심해야 한다.

$\operatorname{Var}(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$을 그대로 쓰면, 값이 큰 offset 위에서 조금씩 변할 때(예: Unix 시각, 절대온도) 큰 두 수의 차로 작은 수를 구하게 되어 유효숫자가 사라진다(9.1장의 소거 오차).

대신 덩어리마다 개수 $n$, 평균 $\bar{x}$, 편차 제곱합 $M_2 = \sum_i (x_i - \bar{x})^2$을 구하고, 두 묶음 $A$, $B$의 통계량을 다음 식으로 합친다.

$$
n = n_A + n_B, \quad \delta = \bar{x}_B - \bar{x}_A, \quad
\bar{x} = \bar{x}_A + \delta \frac{n_B}{n}, \quad
M_2 = M_{2,A} + M_{2,B} + \delta^2 \frac{n_A n_B}{n}.
$$

모든 덩어리를 합친 뒤 모분산은 $M_2 / n$이다.

함수 `chunked_mean_var(chunks)`를 완성한다. `chunks`는 1차원 배열을 하나씩 내놓는 iterable(예: generator)이고, **한 번만** 순회할 수 있다. `(n, mean, var)`를 반환한다. `var`는 모분산(`ddof=0`)이다. 빈 덩어리는 건너뛴다.

테스트는 offset이 $10^9$인 데이터에서도 `np.var`와 같은 값을 요구한다.
