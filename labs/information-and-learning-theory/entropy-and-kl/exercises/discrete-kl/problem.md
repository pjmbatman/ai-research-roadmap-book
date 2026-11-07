# 이산 KL과 교차엔트로피 구현하기

같은 결과 집합 위의 두 이산분포 `p`, `q`(shape `(K,)`, 각각 합이 1)에 대해 두 함수를 완성한다. 로그는 자연로그(nat)를 쓴다.

- `cross_entropy(p, q)`: $H(p, q) = -\sum_k p_k \log q_k$
- `kl_divergence(p, q)`: $D_{\mathrm{KL}}(p \,\|\, q) = \sum_k p_k \log \dfrac{p_k}{q_k}$

두 규약을 지켜야 한다.

1. $p_k = 0$인 항은 0으로 둔다($0 \log 0 = 0$, $0 \log(0/q) = 0$). 이때 $q_k$가 0이어도 된다.
2. $p_k > 0$인데 $q_k = 0$인 $k$가 있으면 두 함수 모두 `np.inf`를 반환한다.

테스트는 다음을 확인한다.

- $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$ (여기서 $H(p) = H(p, p)$)
- $D_{\mathrm{KL}}(p \,\|\, p) = 0$, $D_{\mathrm{KL}} \ge 0$
- 비대칭: 일반적으로 $D_{\mathrm{KL}}(p \,\|\, q) \neq D_{\mathrm{KL}}(q \,\|\, p)$
- 위의 두 규약, 그리고 결과에 `nan`이 나오지 않는 것

힌트: `np.log(0)`은 `-inf`이고 `0 * -inf`는 `nan`이다. 먼저 `p > 0`인 위치만 골라 계산한다.
