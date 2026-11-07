# 자기상관을 보정한 추세의 표준오차

연평균 기온 편차처럼 자기상관이 있는 시계열에서 최소제곱 추세의 표준오차를 그대로 쓰면, 추세가 실제보다 확실해 보인다. 함수 `trend_test(y)`를 완성해 `(slope, se_naive, n_eff, se_adj)`를 반환한다.

`y`의 길이를 $n$, 시점을 가운데로 옮긴 $t_i = i - (n-1)/2$라 할 때:

1. **기울기** $b = \sum_i t_i y_i \big/ \sum_i t_i^2$, 잔차 $e_i = y_i - \bar{y} - b\,t_i$.
2. **순진한 표준오차** $\mathrm{SE} = \sqrt{\dfrac{\sum_i e_i^2}{(n-2)\sum_i t_i^2}}$.
3. 잔차의 **lag-1 자기상관** $r_1 = \sum_{i \ge 1} e_i e_{i-1} \big/ \sum_i e_i^2$.
4. **유효 표본 수** $n_\text{eff} = n\,\dfrac{1 - r_1}{1 + r_1}$. 단, 3보다 작으면 3, $n$보다 크면 $n$으로 자른다($r_1 < 0$이어도 표본 수를 늘리지 않는다).
5. **보정한 표준오차** $\mathrm{SE}_\text{adj} = \mathrm{SE}\sqrt{\dfrac{n-2}{n_\text{eff}-2}}$.

모든 반환값은 Python `float`(또는 0차원 NumPy 값)이다.

힌트: 2번의 분모 $(n-2)$가 5번에서 $(n_\text{eff}-2)$로 바뀐다고 보면 된다. 잔차의 자유도를 유효 표본 수로 다시 세는 것이다.
