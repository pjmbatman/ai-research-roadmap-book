import numpy as np


def check_standardize(standardize):
    """standardize 구현이 명세를 어기면 AssertionError를 낸다.

    명세: x (N, D) → 새 배열 (N, D). 각 열의 평균 0, 표준편차(ddof=0) 1.
    값이 모두 같은 열은 0. 입력 x는 바꾸지 않는다.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[5.0, -3.0, 100.0], scale=[2.0, 0.1, 30.0], size=(7, 3))   # (7, 3): 정사각이 아니다
    x[:, 1] = 4.2                                                                 # 값이 모두 같은 열
    x_before = x.copy()

    z = standardize(x)

    assert z.shape == x.shape, f"shape {z.shape} != {x.shape}"
    assert np.array_equal(x, x_before), "입력 배열이 바뀌었다"
    assert np.all(np.isfinite(z)), "NaN 또는 무한대가 있다 (값이 모두 같은 열?)"
    np.testing.assert_allclose(z[:, 1], 0.0, atol=1e-12, err_msg="값이 모두 같은 열은 0이어야 한다")
    varying = [0, 2]
    np.testing.assert_allclose(z[:, varying].mean(axis=0), 0.0, atol=1e-10, err_msg="열 평균이 0이 아니다")
    np.testing.assert_allclose(z[:, varying].std(axis=0), 1.0, rtol=1e-10, err_msg="열 표준편차(ddof=0)가 1이 아니다")
    # 독립적인 기준: 손으로 계산할 수 있는 작은 예. [1, 2, 3]의 평균 2, 표준편차 sqrt(2/3)
    small = standardize(np.array([[1.0], [2.0], [3.0]]))                        # (3, 1)
    np.testing.assert_allclose(small[:, 0], np.array([-1.0, 0.0, 1.0]) / np.sqrt(2 / 3), rtol=1e-12)
