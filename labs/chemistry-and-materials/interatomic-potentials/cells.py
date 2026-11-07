"""28.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

단위는 Lennard-Jones 환산 단위(eps = sigma = mass = 1).

실행: uv run --project labs python labs/chemistry-and-materials/interatomic-potentials/cells.py
"""
# %% cell 1 — Lennard-Jones 에너지와 해석적 힘, 수치 미분으로 검증
import numpy as np

def lj_energy(R):                                  # R (n, 3), 스칼라 에너지
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)   # (n, n)
    iu = np.triu_indices(len(R), 1)                # 각 쌍을 한 번만
    inv6 = (1 / d[iu]) ** 6
    return np.sum(4 * (inv6 ** 2 - inv6))

def lj_forces(R):                                  # 해석적 힘 F = -dE/dR (n, 3)
    diff = R[:, None] - R[None]                    # (n, n, 3)
    d = np.linalg.norm(diff, axis=-1)              # (n, n)
    np.fill_diagonal(d, np.inf)                    # 자기 자신 제외
    inv6 = (1 / d) ** 6
    coeff = 24 * (2 * inv6 ** 2 - inv6) / d ** 2   # (n, n)
    return np.einsum("ij,ijk->ik", coeff, diff)    # 쌍마다 방향 벡터에 곱해 합

re = 2 ** (1 / 6)                                   # LJ 평형 거리
R = np.array([[0, 0, 0], [re, 0, 0], [-re, 0, 0], [0, re, 0],
              [0, -re, 0], [0, 0, re], [0, 0, -re]], float)   # 중심 + 팔면체 7원자
R = R + np.random.default_rng(0).normal(size=R.shape) * 0.05  # 평형에서 살짝 흔들기
F = lj_forces(R)
h = 1e-5                                            # 중심 차분으로 수치 힘
F_num = np.zeros_like(R)
for i in range(R.shape[0]):
    for k in range(3):
        Rp, Rm = R.copy(), R.copy()
        Rp[i, k] += h; Rm[i, k] -= h
        F_num[i, k] = -(lj_energy(Rp) - lj_energy(Rm)) / (2 * h)
print("에너지 E =", round(float(lj_energy(R)), 4))
print("해석적 힘과 수치 힘의 최대 차이 < 1e-6:", np.abs(F - F_num).max() < 1e-6)
print("전체 힘의 합(뉴턴 제3법칙) ~ 0:", np.abs(F.sum(0)).max() < 1e-10)

# %% cell 2 — 힘이 에너지보다 정보가 많다: 자유도 세기
n = 20
print(f"원자 {n}개 배치 하나에서 한 번의 양자화학 계산이 주는 제약:")
print(f"  에너지만 학습: 스칼라 {1}개")
print(f"  힘도 학습    : 에너지 1개 + 힘 {n} × 3 = {3 * n}개 성분")
print(f"  → 같은 계산 비용으로 힘은 배치마다 {3 * n + 1}배의 제약을 준다")

# %% cell 3 — velocity Verlet와 에너지 보존의 차수
def velocity_verlet(R, V, dt, steps):
    F = lj_forces(R)
    traj_E = []
    for _ in range(steps):
        V = V + 0.5 * dt * F                       # 반 스텝 속도
        R = R + dt * V                             # 위치 갱신
        F = lj_forces(R)                           # 새 힘
        V = V + 0.5 * dt * F                       # 나머지 반 스텝
        traj_E.append(0.5 * np.sum(V ** 2) + lj_energy(R))
    return np.array(traj_E)

R0 = np.array([[0.0, 0, 0], [1.2, 0, 0]])          # 두 원자, 평형 1.122보다 약간 멂
V0 = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
E0 = 0.5 * np.sum(V0 ** 2) + lj_energy(R0)
prev = None
for dt in [0.05, 0.01, 0.002]:
    E = velocity_verlet(R0.copy(), V0.copy(), dt, int(2.0 / dt))
    drift = np.abs(E - E0).max()
    ratio = "" if prev is None else f"  (dt 1/5 → 표류 1/{prev / drift:.0f})"
    print(f"dt = {dt:.3f}  총에너지 표류 {drift:.2e}{ratio}")
    prev = drift

# %% cell 4 — 힘이 에너지의 gradient가 아니면(비보존력) 에너지가 새고 계가 터진다
cluster = np.array([[0, 0, 0], [re, 0, 0], [-re, 0, 0], [0, re, 0],
                    [0, -re, 0], [0, 0, re], [0, 0, -re]], float)

def run(force_fn, steps=4000, dt=0.004):
    R = cluster + np.random.default_rng(1).normal(size=cluster.shape) * 0.03
    V = np.zeros_like(R)
    F = force_fn(R)
    Es, size = [], []
    for _ in range(steps):
        V = V + 0.5 * dt * F; R = R + dt * V; F = force_fn(R); V = V + 0.5 * dt * F
        Es.append(0.5 * np.sum(V ** 2) + lj_energy(R))
        size.append(np.linalg.norm(R - R.mean(0), axis=1).max())
    return np.array(Es), np.array(size)

omega = np.array([0, 0, 1.0])
def rms_force_error(gain):                          # 참 힘 대비 추가된 힘의 RMS (평형 근처 배치에서)
    extra = gain * np.cross(omega, cluster - cluster.mean(0))
    return np.sqrt(np.mean(extra ** 2)) / np.sqrt(np.mean(lj_forces(cluster + 0.03) ** 2))

for gain in [0.0, 0.05, 0.15]:
    fn = (lambda R: lj_forces(R)) if gain == 0 else (lambda R, g=gain: lj_forces(R) + g * np.cross(omega, R - R.mean(0)))
    Es, size = run(fn)
    label = "참 힘(보존력)        " if gain == 0 else f"비보존력 (힘 RMS 오차 {rms_force_error(gain):.0%})"
    gained = Es[-1] - Es[0] > 0.1                   # 에너지가 유의미하게 늘었는가
    blew_up = size.max() > 2.0                      # 계가 평형 크기의 두 배 넘게 퍼졌는가
    print(f"{label}  에너지 증가 {gained!s:5s}  계가 터짐 {blew_up!s:5s}")
