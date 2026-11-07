"""26.1 실습: 점성 Burgers 시뮬레이션의 해상도 연구와 데이터셋 생성 조건 기록.

실행: uv run --project labs python labs/physics-and-engineering/simulation-fundamentals/resolution_study.py
- Re = 100, 400 두 조건에서 격자를 바꾸며 출력량(소산율, 최대 기울기)·step 수·계산 시간을 잰다.
- 가장 고운 세 격자로 Richardson 외삽값을 구해 각 격자의 상대 오차를 추정한다.
- 데이터셋 한 항목에 붙일 생성 조건(메타데이터)을 JSON으로 출력한다.
"""
import json
import platform
import time

import numpy as np


def burgers(Re, N, t_star=0.3):
    """무차원 u_t + u u_x = (1/Re) u_xx, 주기 경계 [0, 1), u0 = sin(2 pi x). explicit Euler + 중심차분."""
    nu = 1.0 / Re
    dx = 1.0 / N
    u = np.sin(2 * np.pi * np.arange(N) * dx)
    n = int(np.ceil(t_star / (0.2 * min(dx, dx**2 / nu))))
    dt = t_star / n
    for _ in range(n):
        up, um = np.roll(u, -1), np.roll(u, 1)
        u = u - dt * u * (up - um) / (2 * dx) + dt * nu * (up - 2 * u + um) / dx**2
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    return {"N": N, "steps": n, "dissipation": nu * np.sum(ux**2) * dx, "max_grad": float(np.max(np.abs(ux)))}


def study(Re, grids):
    rows = []
    for N in grids:
        t0 = time.perf_counter()
        r = burgers(Re, N)
        r["time_s"] = time.perf_counter() - t0
        rows.append(r)
    f1, f2, f3 = (r["dissipation"] for r in rows[-3:])
    p = np.log2(abs(f2 - f1) / abs(f3 - f2))
    f_ext = f3 + (f3 - f2) / (2**p - 1)
    print(f"[Re = {Re}]  소산율 관측 차수(가장 고운 세 격자) {p:.2f}, 외삽값 {f_ext:.5f}")
    for r in rows:
        err = abs(r["dissipation"] - f_ext) / f_ext
        print(f"  N={r['N']:5d}  격자 Re {Re / r['N']:6.2f}  step {r['steps']:7d}  "
              f"소산율 {r['dissipation']:.5f} (상대 오차 {err:6.2%})  최대 기울기 {r['max_grad']:8.2f}  "
              f"시간 {r['time_s']:7.3f} s")
    return rows


if __name__ == "__main__":
    study(100, [32, 64, 128, 256, 512, 1024])
    study(400, [64, 128, 256, 512, 1024, 2048])
    meta = {
        "equation": "u_t + u u_x = (1/Re) u_xx (무차원 점성 Burgers)",
        "domain": "[0, 1), 주기 경계",
        "initial_condition": "u0 = sin(2 pi x)",
        "parameters": {"Re": 100, "t_star": 0.3},
        "discretization": "2차 중심차분, N = 256, explicit Euler, dt = 0.2 min(dx, Re dx^2) 이하",
        "output": "t_star에서의 u(x) 256점, 소산율",
        "convergence": "소산율 관측 차수 약 2, N=256 상대 오차는 resolution_study.py 표 참조",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))
