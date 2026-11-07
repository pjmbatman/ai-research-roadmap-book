# 여러 센서의 경보로 고장 확률 구하기

장비가 고장($F = 1$)일 사전확률이 `prior`다. 센서 $k$개가 장비를 감시하고, 센서 $j$는

- 고장일 때 확률 `sens[j]`로 경보를 낸다: $P(A_j = 1 \mid F = 1)$
- 정상일 때 확률 `fpr[j]`로 경보를 낸다(위경보): $P(A_j = 1 \mid F = 0)$

센서들은 장비 상태가 주어지면 서로 **조건부 독립**이다. 즉

$$
P(a_1, \dots, a_k \mid F) = \prod_{j=1}^k P(a_j \mid F).
$$

함수 `posterior_fault(prior, sens, fpr, alarms)`를 완성해 $P(F = 1 \mid a_1, \dots, a_k)$를 반환한다. `sens`, `fpr`, `alarms`는 shape `(k,)`이고, `alarms[j]`는 0(조용함) 또는 1(경보)이다. 경보가 **울리지 않은** 센서도 정보다. 그 센서의 우도는 $1 - \text{sens}$ 또는 $1 - \text{fpr}$이다.

테스트는 다음을 확인한다.

- 센서 하나일 때 베이즈 정리와 같다.
- 센서를 하나씩 순서대로 반영(오늘의 사후 = 다음 단계의 사전)해도, 한 번에 반영해도 결과가 같다.
- 시뮬레이션으로 센 비율과 맞는다.
- 센서가 2000개여도 `nan` 없이 계산된다.

힌트: 우도의 곱은 센서가 많으면 0으로 underflow한다. 로그 오즈(log-odds) $\log\frac{P(F=1 \mid \cdot)}{P(F=0 \mid \cdot)} = \log\frac{\text{prior}}{1 - \text{prior}} + \sum_j \log\frac{P(a_j \mid F=1)}{P(a_j \mid F=0)}$로 더한 뒤 마지막에 sigmoid로 확률로 바꾼다. sigmoid도 큰 음수·양수 입력에서 overflow하지 않게 쓴다.
