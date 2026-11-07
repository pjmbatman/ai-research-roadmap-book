import numpy as np


def simulate_ou(x0, theta, mu, sigma, dt, n_steps, n_paths, rng):
    """dX = theta (mu - X) dt + sigma dW 를 Euler–Maruyama로 시뮬레이션한다.

    반환: (n_steps + 1, n_paths) 배열. 0번째 행은 x0.
    """
    x = np.empty((n_steps + 1, n_paths))                 # (n_steps + 1, n_paths)
    x[0] = x0
    for k in range(n_steps):
        dw = np.sqrt(dt) * rng.normal(size=n_paths)      # (n_paths,) 브라운 운동 증분, 분산 dt
        x[k + 1] = x[k] + theta * (mu - x[k]) * dt + sigma * dw
    return x
