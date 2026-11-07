"""28.1 퀴즈(content/28-chemistry-and-materials/01-molecular-representations/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from itertools import permutations
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/28-chemistry-and-materials/01-molecular-representations/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def parse(smiles):
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
    A = np.zeros((len(atoms), len(atoms)), dtype=int)
    for i, j in bonds:
        A[i, j] = A[j, i] = 1
    return atoms, A


def canonical(atoms, A):
    return min(("".join(atoms[i] for i in p), A[np.ix_(p, p)].tobytes()) for p in permutations(range(len(atoms))))


def random_rotation():
    Q_, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q_ * np.sign(np.linalg.det(Q_))


def test_smiles_duplicates():
    assert "CCO" != "OCC"
    assert canonical(*parse("CCO")) == canonical(*parse("OCC"))      # 다른 문자열, 같은 분자
    assert correct("smiles-duplicates") == [0]


def test_tanimoto_one():
    assert correct("tanimoto-one") == [1]


def test_chirality_invariance():
    R = rng.normal(size=(5, 3))
    mirror = R * np.array([-1.0, 1.0, 1.0])
    dist = lambda X: np.linalg.norm(X[:, None] - X[None], axis=-1)
    assert np.allclose(dist(R), dist(mirror))                         # 모든 거리가 같다
    vol = lambda X: np.linalg.det(X[1:4] - X[0])
    assert np.isclose(vol(R), -vol(mirror))                           # 부호 있는 부피는 뒤집힌다
    assert correct("chirality-invariance") == [0]


def test_rotation_invariant_targets():
    R = rng.normal(size=(6, 3)) * 1.5
    q = rng.normal(size=6)

    def lj(X):
        d = np.linalg.norm(X[:, None] - X[None], axis=-1)[np.triu_indices(len(X), 1)]
        return np.sum(4 * (d ** -12 - d ** -6))

    def forces(X, h=1e-6):
        F = np.zeros_like(X)
        for i in range(X.shape[0]):
            for k in range(3):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, k] += h
                Xm[i, k] -= h
                F[i, k] = -(lj(Xp) - lj(Xm)) / (2 * h)
        return F

    Rot = random_rotation()
    R2 = R @ Rot.T
    assert np.isclose(lj(R), lj(R2))                                  # 에너지: 불변
    assert not np.allclose(forces(R), forces(R2), atol=1e-6)          # 힘: 불변이 아니다
    assert np.allclose(forces(R) @ Rot.T, forces(R2), atol=1e-4)      # 힘: 등변
    dip = lambda X: q @ X
    assert not np.allclose(dip(R), dip(R2))                           # 쌍극자 벡터: 불변이 아니다
    assert np.isclose(np.linalg.norm(dip(R)), np.linalg.norm(dip(R2)))
    assert correct("rotation-invariant-targets") == [0, 1]


def test_crystal_neighbors():
    a, cut = 3.0, 4.5
    shifts = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) * a
    d = np.linalg.norm(shifts, axis=1)
    assert ((d > 0) & (d < cut)).sum() == Q["crystal-neighbors"]["answer"] == 18


def test_representation_choice():
    assert correct("representation-choice") == [0]
