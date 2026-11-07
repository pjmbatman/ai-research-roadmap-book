def receptive_field(layers):
    """쌓은 1차원 합성곱 층의 이론 receptive field와 jump.

    layers: [(k, s, d), ...] 입력 쪽부터. k = kernel 크기, s = stride, d = dilation
    반환: (r, J)  r = receptive field 길이(입력 칸 수), J = 출력 한 칸이 입력에서 건너뛰는 칸 수
    """
    r, jump = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * jump      # 이 층이 넓히는 폭: 이 층 입력 한 칸 = 원래 입력 jump칸
        jump *= s
    return r, jump
