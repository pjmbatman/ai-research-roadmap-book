# gradient norm 로그에서 이상 step 찾기

학습 중 step마다 기록한 global gradient norm 배열 `norms`가 있다. 발산이나 이상 batch는 대개 gradient norm이 **평소보다 갑자기 커지는** 것으로 먼저 나타난다. 함수 `find_spikes(norms, window=20, factor=5.0, min_history=5)`를 완성해 이상 step의 index를 오름차순 리스트로 반환한다.

step $i$는 다음 중 하나이면 이상이다.

1. **비유한 값**: `norms[i]`가 NaN이거나 무한대다. 이력의 길이와 상관없이 항상 이상이다.
2. **spike**: 직전 이력의 중앙값을 $m_i$라 할 때 `norms[i] > factor * m_i`이다.
   - 직전 이력은 step $i$ **이전**의 정상 값 중 최근 `window`개다. 이상으로 판정된 step의 값은 이력에 넣지 않는다(spike가 기준을 끌어올리지 않게).
   - 이력에 정상 값이 `min_history`개보다 적으면 spike 판정을 하지 않는다(학습 초반의 큰 gradient를 오탐하지 않게).

평균 대신 **중앙값**을 쓰는 이유는, 기준 자체가 몇 개의 큰 값에 끌려가지 않게 하려는 것이다.

예: `norms = [1.0, 1.1, 0.9, 1.0, 1.2, 1.0, 9.0, 1.1, nan, 1.0]`, `window=20`, `factor=5`, `min_history=5`이면 step 6(9.0 > 5 × 1.0)과 step 8(NaN)이 이상이므로 `[6, 8]`을 반환한다.
