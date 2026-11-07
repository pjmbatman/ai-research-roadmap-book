"""28.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/chemistry-and-materials/molecular-representations/cells.py
"""
# %% cell 1 — SMILES 부분집합을 그래프로 읽고 표기 차이를 지우기
import numpy as np
from itertools import permutations

VALENCE = {"C": 4, "N": 3, "O": 2}

def parse(smiles):                    # 고리·다중결합 없는 SMILES 부분집합만 읽는다
    atoms, bonds, stack, prev = [], [], [], None
    for ch in smiles:
        if ch == "(":
            stack.append(prev)
        elif ch == ")":
            prev = stack.pop()
        else:
            atoms.append(ch)
            if prev is not None:
                bonds.append((prev, len(atoms) - 1))
            prev = len(atoms) - 1
    A = np.zeros((len(atoms), len(atoms)), dtype=int)   # 인접 행렬 (n, n)
    for i, j in bonds:
        A[i, j] = A[j, i] = 1
    return atoms, A

def canonical(atoms, A):              # 모든 번호 매기기 중 사전순 최소 (원자 7개 이하에서만)
    return min(("".join(atoms[i] for i in p), A[np.ix_(p, p)].tobytes())
               for p in permutations(range(len(atoms))))

ref = canonical(*parse("CCO"))
for s in ["CCO", "OCC", "C(O)C", "COC"]:
    atoms, A = parse(s)
    n_h = sum(VALENCE[a] for a in atoms) - A.sum()     # A.sum() = 2 × 결합 수
    print(f"{s:6s} 원자 {''.join(atoms)}  차수 {A.sum(1)}  암묵 H {n_h}  CCO와 같은 그래프: {canonical(atoms, A) == ref}")

# %% cell 2 — 반경 1 원자 환경 지문과 Tanimoto 유사도
import zlib

def fingerprint(atoms, A, n_bits=64):
    ids = [f"{a}{d}" for a, d in zip(atoms, A.sum(1))]             # 반경 0: 원소 + 차수
    env = [ids[i] + "|" + ",".join(sorted(ids[j] for j in np.flatnonzero(A[i])))
           for i in range(len(atoms))]                               # 반경 1: 이웃까지
    fp = np.zeros(n_bits, dtype=bool)                                # (n_bits,)
    for f in set(ids) | set(env):
        fp[zlib.crc32(f.encode()) % n_bits] = True                   # 해시 → 비트 위치
    return fp

def tanimoto(a, b):
    return (a & b).sum() / (a | b).sum()

names = ["CCO", "OCC", "COC", "CCCO", "CCN"]
fps = [fingerprint(*parse(s)) for s in names]
print("켜진 비트 수:", [int(f.sum()) for f in fps])
print("Tanimoto(CCO, ·):", {s: round(float(tanimoto(fps[0], f)), 3) for s, f in zip(names, fps)})

# %% cell 3 — 3차원 좌표와 Coulomb 행렬 고윳값
ANG_TO_BOHR = 1.8897259886

def coulomb_matrix(Z, R_bohr):                     # Z (n,), R_bohr (n, 3)
    D = np.linalg.norm(R_bohr[:, None] - R_bohr[None], axis=-1)   # (n, n)
    np.fill_diagonal(D, 1.0)
    M = np.outer(Z, Z) / D
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return M

def cm_eigs(Z, R_bohr):                            # 크기순 고윳값 (n,)
    return np.sort(np.linalg.eigvalsh(coulomb_matrix(Z, R_bohr)))[::-1]

Z = np.array([8.0, 1.0, 1.0])                                        # 물: O, H, H
R_ang = np.array([[0.0, 0.0, 0.0], [0.7572, 0.5865, 0.0], [-0.7572, 0.5865, 0.0]])
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
Q *= np.sign(np.linalg.det(Q))                                       # 회전 (det = +1)
perm = np.array([2, 0, 1])
R2 = (R_ang @ Q.T + np.array([1.0, -2.0, 0.5]))[perm]                # 회전·이동·번호 바꾸기
e1, e2 = cm_eigs(Z, R_ang * ANG_TO_BOHR), cm_eigs(Z[perm], R2 * ANG_TO_BOHR)
print("고윳값        :", np.round(e1, 3))
print("좌표가 같은가 :", np.allclose(R_ang, R2), "  고윳값 차이 < 1e-8:", np.abs(e1 - e2).max() < 1e-8)

# %% cell 4 — 거울상: 거리 기반 표현은 키랄성을 보지 못한다
Z4 = np.array([6.0, 1.0, 9.0, 17.0, 35.0])                           # C, H, F, Cl, Br
dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
bond_ang = np.array([1.09, 1.35, 1.77, 1.94])                        # C–H, C–F, C–Cl, C–Br
R = np.vstack([np.zeros(3), dirs * bond_ang[:, None]])               # (5, 3) Å
R_mirror = R * np.array([-1.0, 1.0, 1.0])                            # x → −x 반사

def signed_volume(R):                                                # F, Cl, Br가 만드는 부호 있는 부피
    return np.linalg.det(R[2:5] - R[0])

d = np.abs(cm_eigs(Z4, R * ANG_TO_BOHR) - cm_eigs(Z4, R_mirror * ANG_TO_BOHR)).max()
print("고윳값 차이 < 1e-8:", d < 1e-8)
print(f"부호 있는 부피: 원래 {signed_volume(R):+.3f}, 거울상 {signed_volume(R_mirror):+.3f} Å^3")

# %% cell 5 — 주기 결정: 이웃 수는 셀 선택과 원점에 무관하다
def neighbor_counts(frac, cell_ang, cutoff_ang):   # frac (n, 3) 분율 좌표, 직육면체 셀 변 cell_ang (3,)
    n_img = np.ceil(cutoff_ang / cell_ang).astype(int)
    shifts = np.array([(i, j, k) for i in range(-n_img[0], n_img[0] + 1)
                       for j in range(-n_img[1], n_img[1] + 1)
                       for k in range(-n_img[2], n_img[2] + 1)]) * cell_ang   # (S, 3)
    cart = frac * cell_ang                                                     # (n, 3)
    d = np.linalg.norm(cart[:, None, None] - cart[None, :, None] - shifts[None, None], axis=-1)  # (n, n, S)
    return ((d > 1e-6) & (d < cutoff_ang)).sum(axis=(1, 2))

a_ang = 5.64                                                          # 암염(NaCl) 입방 셀의 격자 상수(Å)
fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
conv = np.vstack([fcc, fcc + [0.5, 0, 0]]) % 1.0                      # Na 4개 + Cl 4개 (8, 3)
super_ = np.vstack([(conv + [i, 0, 0]) / [2, 1, 1] for i in range(2)])  # 2×1×1 초격자 (16, 3)
shifted = (conv + 0.25) % 1.0                                         # 원점만 옮김
for name, frac, cell in [("입방 셀", conv, [a_ang] * 3), ("2×1×1 초격자", super_, [2 * a_ang, a_ang, a_ang]),
                         ("원점 이동", shifted, [a_ang] * 3)]:
    c3, c4 = neighbor_counts(frac, np.array(cell), 3.0), neighbor_counts(frac, np.array(cell), 4.1)
    print(f"{name:10s} 원자 {len(frac):2d}개  3.0 Å 안 이웃 {set(c3.tolist())}  4.1 Å 안 이웃 {set(c4.tolist())}")
print("분율 좌표를 그대로 펼친 벡터가 같은가:", np.allclose(conv, shifted))
