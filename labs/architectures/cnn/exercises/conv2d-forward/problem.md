# conv2d를 NumPy로 구현하기

딥러닝 라이브러리의 2차원 합성곱(정확히는 상호상관)을 직접 구현한다. 입력 `x`의 shape은 `(C_in, H, W)`, kernel `w`의 shape은 `(C_out, C_in, kH, kW)`, bias `b`의 shape은 `(C_out,)`이다. 출력 `y`의 원소는

$$
y_{o,i,j} = b_o + \sum_{c=0}^{C_{\text{in}}-1} \sum_{u=0}^{k_H-1} \sum_{v=0}^{k_W-1} w_{o,c,u,v}\; \tilde{x}_{c,\; i s + u,\; j s + v}
$$

이다. $\tilde{\mathbf{x}}$는 높이·너비 양쪽에 `padding`칸씩 0을 덧붙인 입력, $s$는 `stride`다(높이와 너비에 같은 값을 쓴다). kernel을 뒤집지 않는다는 점에 주의한다.

함수 `conv2d(x, w, b, stride=1, padding=0)`를 완성해 `y`를 반환한다. 출력 shape은 `(C_out, H_out, W_out)`이고

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1
$$

이다($W_{\text{out}}$도 같다).

테스트는 다음을 확인한다.

- 여러 `stride`·`padding`·직사각형 kernel에서 출력 shape이 공식과 같다.
- 출력값이 원소 정의를 그대로 계산한 값과 같다.
- stride 1, padding 0, 채널 1개일 때 `scipy.signal.correlate2d(..., mode="valid")`와 같다.

힌트: 출력 위치 `(i, j)`마다 입력에서 `(C_in, kH, kW)` 조각을 잘라 `w`와 곱해 더한다. `np.einsum("chw,ochw->o", patch, w)`가 채널과 kernel 축을 한 번에 축약한다.
