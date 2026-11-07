import math


def quadratic_roots(a, b, c):
    """a x^2 + b x + c = 0의 두 실근을 상쇄 없이 계산해 (작은 근, 큰 근)으로 반환한다.

    가정: a != 0, b^2 - 4ac > 0
    """
    d = math.sqrt(b * b - 4 * a * c)
    sign = 1.0 if b >= 0 else -1.0       # b = 0일 때도 q가 0이 되지 않게
    q = -0.5 * (b + sign * d)            # 부호가 같은 두 수의 합: 상쇄 없음
    r1, r2 = q / a, c / q                # Vieta: r1 * r2 = c / a
    return (min(r1, r2), max(r1, r2))
