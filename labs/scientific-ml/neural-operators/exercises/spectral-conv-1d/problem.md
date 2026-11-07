# 스펙트럼 합성곱 층

FNO의 핵심 층은 입력을 Fourier 계수로 바꾸고, 낮은 $K$개 모드에서만 채널을 섞는 가중치를 곱한 뒤 격자로 되돌린다. 입력 채널 $C_{\text{in}}$개, 출력 채널 $C_{\text{out}}$개일 때

$$
\hat{v}_{o}(k) = \sum_{i=1}^{C_{\text{in}}} W_{i,o,k}\,\hat{u}_i(k) \quad (0 \le k < K), \qquad \hat{v}_o(k) = 0 \quad (k \ge K)
$$

이다. $\hat{u}_i(k)$는 채널 $i$의 $k$번째 Fourier 계수다.

함수 `spectral_conv1d(u, weights)`를 완성한다.

| 인자 | shape | 뜻 |
|---|---|---|
| `u` | `(C_in, N)` 실수 | 주기 격자 $[0, 1)$의 $N$점 값 |
| `weights` | `(C_in, C_out, K)` 복소수 | 모드별 채널 혼합 가중치, $K \le N/2 + 1$ |

반환값은 shape `(C_out, N)`의 실수 배열이다.

**해상도 불변성의 조건.** 같은 함수를 다른 $N$으로 샘플링해도 같은 결과가 나오려면 Fourier 계수가 $N$에 의존하지 않아야 한다. `np.fft.rfft`는 계수에 $N$을 곱한 값을 주므로, `np.fft.rfft(u, axis=-1) / N`으로 정규화하고, 되돌릴 때 `np.fft.irfft(v_hat * N, n=N, axis=-1)`을 쓴다(또는 두 함수의 `norm="forward"` 옵션).

테스트는 다음을 확인한다.

- 출력 shape과 실수성
- 같은 함수를 32점과 128점에서 샘플링했을 때, 겹치는 점에서 출력이 같다(해상도 불변성)
- 파수 $K$ 이상의 성분은 출력에서 사라진다
- 가중치 $W_{0,0,k} = 2\pi i k$는 미분 연산자다: $\sin 2\pi x$를 넣으면 $2\pi\cos 2\pi x$가 나온다

힌트: 채널 혼합은 `np.einsum("ik,iok->ok", u_hat[:, :K], weights)`로 쓸 수 있다.
