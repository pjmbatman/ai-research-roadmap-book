import numpy as np


def simulate_ou(x0, theta, mu, sigma, dt, n_steps, n_paths, rng):
    """dX = theta (mu - X) dt + sigma dW 를 Euler–Maruyama로 시뮬레이션한다.

    반환: (n_steps + 1, n_paths) 배열. 0번째 행은 x0.
    """
    raise NotImplementedError("시간 방향으로 Euler–Maruyama 갱신을 반복하세요")
