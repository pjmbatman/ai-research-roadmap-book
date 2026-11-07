---
id: chemistry-and-materials/molecular-representations
title: "분자와 결정의 표현"
title_en: "Representing Molecules and Crystals"
tier: core
status: draft
prerequisites: [architectures/graph-networks]
---

# 분자와 결정의 표현

모델은 분자를 직접 보지 못한다. 모델이 보는 것은 연구자가 고른 **표현**(representation)이다. 문자열, 비트 벡터, 그래프, 원자 좌표, 주기적인 단위 세포는 같은 물질을 서로 다른 정보와 서로 다른 대칭으로 적는다. 어떤 표현은 같은 분자를 두 개의 다른 샘플로 만들고, 어떤 표현은 거울상 분자를 구별하지 못하며, 어떤 표현은 추론할 때 구할 수 없는 정보(정확한 3차원 구조)를 요구한다. 이런 차이는 모델 구조보다 먼저 결과를 정한다.

이 장은 방법을 다시 가르치지 않는다. 그래프 신경망은 18.6장, 대칭과 불변성은 7.10장, 등변 모델은 24.8장에서 다룬다. 여기서는 그 도구를 화학·재료에 적용할 때 필요한 판단을 다룬다. **이 표현은 무엇을 담고, 무엇을 버리고, 어떤 변환에 대해 같은 값을 내는가.** 예제는 화학 라이브러리 없이 NumPy만으로 계산한다. 실제 연구에서는 RDKit 같은 cheminformatics 도구를 쓰지만, 그 도구가 내부에서 하는 일을 작은 계산으로 직접 확인하는 것이 목적이다.

## 학습 목표

- 분자·결정의 표현 방식을 비교한다
- 표현이 담는 정보와 대칭을 판단한다

## 문자열과 지문

**왜 필요한가.** 화합물 데이터베이스와 논문 부록은 분자를 대부분 한 줄 문자열로 적는다. 문자열은 저장·검색·공유가 편하지만, 같은 분자를 여러 방식으로 적을 수 있다. 이 사실을 모르고 문자열을 샘플의 키로 쓰면 중복 제거와 데이터 분할이 조용히 틀어진다.

**SMILES.** SMILES(Simplified Molecular Input Line Entry System)는 분자 그래프를 깊이 우선으로 따라가며 원자 기호를 적는 표기다(Weininger, 1988). `CCO`는 탄소–탄소–산소가 이어진 에탄올이다. 괄호는 곁가지, 숫자는 고리를 닫는 결합, `=`·`#`는 이중·삼중 결합을 뜻하고, 수소는 원자가(valence)를 채우는 만큼 암묵적으로 붙는다. 그래프를 어느 원자에서 시작해 어느 순서로 따라가느냐에 따라 `CCO`, `OCC`, `C(O)C`가 모두 에탄올이다.

아래 코드는 고리와 다중결합이 없는 SMILES의 작은 부분집합만 읽는 장난감 파서다. 문자열을 인접 행렬로 바꾸고, 원자 번호를 매기는 모든 방법 중 사전순으로 가장 작은 것을 골라 **표준형**(canonical form)을 만든다. 표준형이 같으면 같은 그래프다.

```python {exec="browser" session="rep"}
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
```

```output
CCO    원자 CCO  차수 [1 2 1]  암묵 H 6  CCO와 같은 그래프: True
OCC    원자 OCC  차수 [1 2 1]  암묵 H 6  CCO와 같은 그래프: True
C(O)C  원자 COC  차수 [2 1 1]  암묵 H 6  CCO와 같은 그래프: True
COC    원자 COC  차수 [1 2 1]  암묵 H 6  CCO와 같은 그래프: False
```

세 문자열은 원자 순서와 차수 배열이 달라도 같은 그래프다. 반면 `COC`(다이메틸 에터)는 분자식이 $\mathrm{C_2H_6O}$로 에탄올과 같지만 원자의 연결이 다른 **구조 이성질체**(constitutional isomer)다. 분자식만 보는 표현은 둘을 구별하지 못한다.

모든 순열을 시도하는 방법은 원자가 늘면 $n!$로 폭발하므로 실제 도구는 그래프 불변량으로 원자 순위를 매기는 표준화 알고리즘을 쓴다. 이렇게 만든 **canonical SMILES**는 같은 도구 안에서는 분자마다 하나지만, 도구가 다르거나 버전이 바뀌면 같은 분자에 다른 문자열을 낼 수 있다. 데이터셋을 합칠 때는 한 도구로 모든 문자열을 다시 표준화한 뒤 비교한다. RDKit에서는 대략 다음 모양이다(집필 시점 예시이며 실행 블록이 아니다. 함수 이름과 인자는 설치한 버전의 문서에서 확인한다).

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

mol = Chem.MolFromSmiles("OCC")             # 파싱 실패 시 None
print(Chem.MolToSmiles(mol))                # canonical SMILES
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = gen.GetFingerprint(mol)                # 비트 지문
```

::quiz{id="smiles-duplicates"}

**지문.** 문자열은 길이가 제각각이라 고전적인 회귀 모델에 바로 넣을 수 없다. **분자 지문**(molecular fingerprint)은 분자를 고정 길이 비트 벡터로 바꾼다. 가장 널리 쓰이는 ECFP(extended-connectivity fingerprint) 계열은 원자마다 자기와 이웃의 식별자를 반복해서 합쳐 반경 $r$까지의 원자 환경을 만들고, 각 환경을 해시 함수로 비트 위치에 대응시킨다(Rogers, Hahn, 2010). 두 지문의 유사도는 흔히 **Tanimoto 계수**로 잰다.

$$
T(\mathbf{a}, \mathbf{b}) = \frac{\lvert \mathbf{a} \wedge \mathbf{b} \rvert}{\lvert \mathbf{a} \vee \mathbf{b} \rvert}.
$$

분자는 공통으로 켜진 비트 수, 분모는 어느 한쪽에라도 켜진 비트 수다. 0이면 공유하는 환경이 없고, 1이면 지문이 같다.

```python {exec="browser" session="rep"}
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
```

```output
켜진 비트 수: [5, 5, 4, 6, 4]
Tanimoto(CCO, ·): {'CCO': 1.0, 'OCC': 1.0, 'COC': 0.125, 'CCCO': 0.833, 'CCN': 0.5}
```

지문은 원자 순서에 의존하지 않으므로 `CCO`와 `OCC`의 유사도는 1이다. 1-프로판올(`CCCO`)은 에탄올과 같은 말단 환경을 많이 공유해 0.833, 다이메틸 에터는 0.125다. 출력에는 지문의 한계도 하나 숨어 있다. 에탄올의 원자 환경은 반경 0에서 3개, 반경 1에서 3개로 모두 6가지인데 켜진 비트는 5개다. 서로 다른 두 환경(`O1`과 `C2|C1,O1`)이 64칸 중 같은 칸에 떨어진 **해시 충돌**이다. 실제 지문은 1024–2048비트를 흔히 쓰지만 충돌은 줄어들 뿐 사라지지 않는다. 그래서 Tanimoto 1.0은 "지문이 보는 반경과 비트 안에서 구별되지 않는다"는 뜻이지 "같은 분자"라는 뜻이 아니다.

::quiz{id="tanimoto-one"}

**기술자.** 지문이 구조 조각의 유무를 적는다면, **기술자**(descriptor)는 분자량, 분배계수 추정치(logP), 극성 표면적, 회전 가능한 결합 수처럼 사람이 정의한 물리화학적 수치를 적는다. 기술자는 해석하기 쉽고 데이터가 적을 때 강한 baseline이 되지만, 어떤 기술자를 고르느냐가 곧 사전 지식이다. 계산 방식이 도구마다 조금씩 다르므로(같은 이름의 logP라도 추정 모델이 다르다) 도구와 버전을 기록한다.

## 그래프 표현

**정의.** 분자 그래프는 원자를 노드, 공유 결합을 간선으로 둔다. 18.6장의 인접 행렬 $\mathbf{A}$와 노드 특징 $\mathbf{X}$가 그대로 쓰인다. 노드 특징은 보통 원소, 차수, 형식 전하, 붙은 수소 수, 혼성 상태, 방향족 여부, 고리 소속을 one-hot이나 정수로 적고, 간선 특징은 결합 차수(단일·이중·삼중·방향족), 공액 여부, 고리 소속을 적는다.

**지문과의 관계.** 반경 $r$의 ECFP는 "각 원자의 $r$-홉 이웃 환경을 모아 집합으로 만든 것"이다. $L$층 message passing GNN도 각 원자의 $L$-홉 이웃을 모은다. 차이는 모으는 방식이다. 지문은 환경을 고정된 해시로 이름 붙여 세고, GNN은 환경을 학습한 연속 벡터로 요약한다. 그래서 GNN은 비슷한 환경에 비슷한 벡터를 줄 수 있지만, 데이터가 적으면 그 요약을 배우는 데 필요한 샘플이 부족하다. 18.6장에서 본 것처럼 합 집계의 message passing은 1-WL 검사보다 구별력이 강할 수 없다. 모든 원자의 이웃 구성이 반복해서 같아지는 서로 다른 그래프는 같은 표현을 받는다. 육원자 고리 두 개가 붙은 데칼린과 오원자 고리 두 개가 단일 결합으로 이어진 바이사이클로펜틸(둘 다 탄소 10개)이 그런 예다.

**수소와 방향족.** 수소를 암묵적으로 둘지(노드 특징의 "붙은 수소 수") 명시적 노드로 둘지는 선택이다. 2차원 그래프 모델은 대개 암묵적 수소를 쓰고, 3차원 퍼텐셜(28.3장)은 수소 원자의 위치가 에너지에 직접 기여하므로 명시적으로 둔다. 방향족 고리를 단일·이중 결합의 교대(Kekulé 구조)로 적을지 방향족 결합으로 적을지도 도구에 따라 다르고, 이 차이가 같은 분자에 다른 그래프를 만들 수 있다. 표준화 단계(28.5장)에서 하나로 맞춘다.

**그래프가 버리는 것.** 결합 그래프는 원자의 연결만 적는다. 결합 길이, 각도, 3차원 배치, 그리고 공간 배치로만 구별되는 **입체 이성질체**(stereoisomer)는 기본 그래프에 없다. SMILES는 `@`, `@@`, `/`, `\`로 입체 정보를 적을 수 있지만, 그래프 특징에 이 정보를 넣지 않으면 모델에 전달되지 않는다.

:::note[분야 예시]
고분자는 같은 반복 단위가 수백–수천 번 이어진 사슬이라 분자 하나를 그래프 하나로 적기 어렵다. 그래서 반복 단위를 그래프로 적고 양 끝의 연결 지점을 특수 원자로 표시하는 방식이 흔히 쓰인다. 이 표현은 사슬 길이 분포, 가지 구조, 결정화도처럼 물성을 크게 바꾸는 정보를 담지 않는다. 고분자의 유리 전이 온도를 반복 단위만으로 예측하는 모델은, 같은 반복 단위로 만든 다른 시료 사이의 차이를 원리적으로 설명할 수 없다.
:::

## 3차원 표현

**왜 필요한가.** 분자의 에너지, 쌍극자 모멘트, 진동 스펙트럼은 원자의 3차원 배치로 정해진다. 같은 결합 그래프라도 결합이 회전하면 에너지가 달라진다. 이런 물성을 예측하려면 좌표가 필요하다.

**대칭.** 원자 좌표 $\mathbf{R} \in \mathbb{R}^{n \times 3}$는 그 자체로는 좋은 입력이 아니다. 분자를 돌리거나 옮기거나 원자 번호를 바꿔 매기면 좌표 배열은 완전히 달라지지만 분자는 같다. 7.10장의 언어로 쓰면 에너지는 회전·이동·같은 원소의 순열에 **불변**이고, 원자에 작용하는 힘은 회전에 **등변**이다. 좌표를 그대로 넣는 모델은 이 대칭을 데이터로 배워야 한다.

**Coulomb 행렬.** 초기의 불변 표현 중 하나인 Coulomb 행렬은 원자 쌍의 핵 전하와 거리로 행렬을 만든다(Rupp 외, 2012).

$$
M_{ii} = \tfrac{1}{2} Z_i^{2.4}, \qquad M_{ij} = \frac{Z_i Z_j}{\lVert \mathbf{R}_i - \mathbf{R}_j \rVert} \quad (i \neq j).
$$

$Z_i$는 원자 번호, 거리는 원자 단위(bohr)다. 거리만 쓰므로 회전과 이동에 불변이다. 원자 번호를 바꿔 매기면 $\mathbf{M}$은 $\mathbf{P}\mathbf{M}\mathbf{P}^\top$로 바뀌는데, 고윳값은 이 변환에 불변이다. 그래서 정렬한 고윳값을 분자의 기술자로 쓸 수 있다.

```python {exec="browser" session="rep"}
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
```

```output
고윳값        : [74.05   0.316  0.151]
좌표가 같은가 : False   고윳값 차이 < 1e-8: True
```

**거울상.** 거리는 반사에도 불변이다. 그래서 거리만 쓰는 표현은 거울상 관계인 두 분자, 즉 **거울상 이성질체**(enantiomer)를 구별하지 못한다. 탄소 하나에 H, F, Cl, Br이 붙은 키랄 분자로 확인한다. 반사에 부호가 바뀌는 양(세 결합 벡터가 만드는 부호 있는 부피)은 둘을 구별한다.

```python {exec="browser" session="rep"}
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
```

```output
고윳값 차이 < 1e-8: True
부호 있는 부피: 원래 -3.569, 거울상 +3.569 Å^3
```

거울상 이성질체는 대부분의 스칼라 물리량(에너지, 끓는점)이 같지만, 키랄 환경과의 상호작용(효소·수용체 결합, 편광의 회전)은 다르다. 약물 활성처럼 키랄성이 중요한 물성에 반사까지 불변인 표현을 쓰면, 모델은 두 거울상에 같은 값을 낼 수밖에 없다. 7.10장에서 말한 "대칭군은 예측 대상이 정한다"의 화학판이다. 등변 모델에서 회전 $SO(3)$만 대칭으로 두고 반사는 빼는 선택이 여기서 나온다(24.8장).

::quiz{id="chirality-invariance"}

**형태와 그 출처.** 실온의 분자는 하나의 고정된 구조가 아니라 여러 **형태**(conformer) 사이를 오간다. 3차원 표현을 쓰려면 "어떤 좌표인가"를 정해야 한다. 양자화학 계산으로 최적화한 구조인지, 결정 구조에서 얻은 것인지, 형태 생성 도구로 만든 것 중 하나인지에 따라 같은 분자에도 다른 입력이 들어간다. QM9은 GDB-17 화학 공간에서 무거운 원자(C, N, O, F) 9개 이하인 약 13만 4천 개 분자의 구조와 물성을 B3LYP/6-31G(2df,p) 수준의 밀도범함수 계산으로 구한 데이터셋으로(Ramakrishnan 외, 2014), 입력 좌표가 레이블과 같은 계산에서 나온다. 반면 실험 용해도를 3차원 모델로 예측할 때는 추론 시점에 그 분자의 "진짜" 구조를 모르므로, 학습과 추론에서 같은 형태 생성 절차를 써야 한다. 그렇지 않으면 학습 때는 정확한 구조를, 배포 때는 근사 구조를 보는 불일치가 생긴다.

## 주기적 결정

**정의.** 결정은 **단위 세포**(unit cell) 하나를 세 격자 벡터 $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ 방향으로 무한히 반복한 구조다. 세포 안 원자의 위치는 보통 **분율 좌표**(fractional coordinate) $\mathbf{f} \in [0, 1)^3$로 적고, 실제 위치는 $\mathbf{r} = f_1\mathbf{a}_1 + f_2\mathbf{a}_2 + f_3\mathbf{a}_3$다. 원자 $i$의 이웃을 찾을 때는 세포 안의 원자뿐 아니라 이웃 세포로 평행이동한 상(image) $\mathbf{r}_j + n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$까지 본다. 이것이 **주기 경계 조건**(periodic boundary condition)이다.

**표현의 대칭.** 결정에는 분자에 없는 모호함이 있다. 같은 결정을 더 큰 초격자(supercell)로 적어도, 원점을 옮겨도, 격자 벡터를 다르게 골라도 같은 결정이다. 분율 좌표를 펼친 벡터는 이 선택마다 달라지므로 좋은 입력이 아니다. **결정 그래프**(crystal graph)는 주기 경계로 찾은 이웃까지의 거리를 간선으로 둬서 이 선택에 무관한 표현을 만든다(Xie, Grossman, 2018). 암염(NaCl) 구조의 입방 셀로 확인한다.

```python {exec="browser" session="rep"}
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
```

```output
입방 셀       원자  8개  3.0 Å 안 이웃 {6}  4.1 Å 안 이웃 {18}
2×1×1 초격자  원자 16개  3.0 Å 안 이웃 {6}  4.1 Å 안 이웃 {18}
원점 이동      원자  8개  3.0 Å 안 이웃 {6}  4.1 Å 안 이웃 {18}
분율 좌표를 그대로 펼친 벡터가 같은가: False
```

모든 원자가 $a/2 = 2.82$ Å 거리에 반대 이온 6개, $a/\sqrt{2} \approx 3.99$ Å 거리에 같은 이온 12개를 가진다. 이 숫자는 셀을 어떻게 잡든 같다. 원자 수는 셀 선택에 따라 8개, 16개로 달라지므로, 결정 전체의 물성(형성 에너지)은 **원자당** 값으로 정의하고 모델의 readout도 합이 아니라 평균을 쓰는 것이 자연스럽다. 합 readout은 같은 결정을 큰 셀로 적었을 때 다른 값을 낸다.

이웃 상을 찾을 때는 셀 한 변보다 cutoff가 길면 한 칸 너머의 상까지 봐야 한다. 코드의 `n_img`가 그 일을 한다. 셀이 작은데 가장 가까운 상만 보는 "최소 상 규약"(minimum image convention)을 쓰면 이웃을 빠뜨린다.

**조성만 쓰는 표현.** 구조가 알려지지 않은 새 재료를 탐색할 때는 화학식(조성)만으로 물성을 예측하기도 한다. 원소 비율과 원소 물성(전기음성도, 원자 반경)의 통계를 특징으로 쓴다. 구조 정보가 없으므로 같은 조성의 다른 결정상(다형체, polymorph)을 구별하지 못한다. 이 한계는 오류가 아니라 질문의 범위다. "이 조성이 유망한가"에는 답하고 "이 구조가 안정한가"에는 답하지 못한다.

:::note[분야 예시]
탄소만으로 이루어진 다이아몬드와 흑연은 조성이 같지만 경도, 전기 전도도, 광학 성질이 전혀 다르다. 조성 기반 모델은 둘에 같은 예측을 낸다. 배터리 양극재처럼 같은 조성이 합성 조건에 따라 다른 상으로 결정화하는 재료에서는, 조성 모델로 후보를 거른 뒤 구조 기반 계산으로 상을 확인하는 2단계 선별이 이 한계를 다루는 한 방법이다.
:::

## 표현의 선택

표현을 고르는 질문은 세 가지다. **무엇을 예측하는가**(대칭과 필요한 정보), **추론 시점에 무엇을 알 수 있는가**(입력의 출처), **데이터가 얼마나 있는가**(학습 가능한 표현의 크기).

| 표현 | 담는 정보 | 불변성 | 잘 맞는 경우 | 주의 |
|---|---|---|---|---|
| SMILES 문자열 | 연결, (선택) 입체 | 없음(표기마다 다름) | 저장·교환, 문자열 생성 모델 | 표준화 없이 키로 쓰지 않는다 |
| 지문 | $r$-홉 환경의 유무·개수 | 원자 순서 | 수백–수천 개 데이터, 유사도 검색 | 해시 충돌, 입체 정보 누락 |
| 기술자 | 정의한 물리화학량 | 정의에 따름 | 적은 데이터, 해석 | 기술자 선택이 곧 가정 |
| 분자 그래프 | 연결, 원자·결합 특징 | 원자 순서 | 수천 개 이상, 2차원 구조로 정해지는 물성 | 3차원·입체 정보 없음 |
| 3차원 좌표(등변 모델) | 배치, 거리, 각도 | 회전·이동(선택: 반사) | 에너지·힘, 형태에 민감한 물성 | 좌표의 출처가 학습·추론에서 같아야 |
| 결정 그래프 | 주기 이웃, 거리 | 셀 선택, 원점, 회전 | 무기 결정의 물성 | 원자당 물성으로 정의 |
| 조성 | 원소 비율 | 구조 전체 | 구조 미지의 탐색 | 다형체 구별 불가 |

연구에서는 대개 이 순서로 진행한다. 먼저 지문이나 기술자에 16장의 고전 모델을 붙인 baseline을 만든다. 데이터가 수천 개를 넘거나 물성이 국소 환경을 넘는 상호작용에 달려 있으면 그래프 모델을 시험한다. 물성이 3차원 배치에 직접 달려 있고 좌표를 믿을 수 있으면 등변 모델(24.8장)로 간다. 어느 단계에서든 비교는 같은 분할(28.2장의 scaffold 분할)에서 한다. MoleculeNet 벤치마크 논문은 양자역학·생물물리 데이터에서는 학습 알고리즘보다 물리를 반영한 특징의 선택이 더 중요할 수 있다고 보고했다(Wu 외, 2018). 표현이 모델보다 먼저라는 이 장의 주장과 같은 방향이다.

::quiz{id="representation-choice"}

## 실습

같은 합성 분자 물성 데이터에서 지문 기반 모델과 그래프 신경망의 데이터 효율을 비교한다. 코드는 `labs/chemistry-and-materials/molecular-representations/fp_vs_gnn.py`다(PyTorch가 필요하다). 데이터는 실제 분자가 아니라 원자가 규칙(C 4, N 3, O 2)을 지키는 무작위 분자 그래프 4000개다. 물성은 두 항의 합으로 정했다.

- **국소 항**: 원자마다 (원소, 차수)에 따른 기여를 더한다. 원자단 기여법(group contribution)과 같은 꼴이다.
- **비국소 항**: 그래프 거리 3–4에 있는 헤테로원자(N·O) 쌍마다 0.5를 더한다. 반경 2 지문의 원자 환경 하나에는 들어가지 않는 범위다.

지문은 반경 0–2 환경의 개수를 1024칸에 해시한 count fingerprint이고 모델은 Ridge 회귀다. GNN은 4층 합 집계 message passing에 합 readout이다. 시험 분자 1000개는 고정했다. 집필 시점의 실행 결과(시드 0, 한 번 실행)는 다음과 같다.

```output
분자 4000개, 물성 표준편차 2.273, 시험 1000개
  학습 수    지문+Ridge MAE   GNN MAE
   100           0.580     0.461
   300           0.465     0.405
  1000           0.419     0.229
  3000           0.398     0.151
```

- **지문 모델은 약 0.4에서 멈춘다.** 학습 데이터를 30배 늘려도 오차가 0.580에서 0.398로 줄 뿐이다. 거리 3–4의 원자 쌍은 반경 2 환경 하나 안에 들어가지 않으므로, 데이터가 아무리 많아도 이 표현으로는 비국소 항을 정확히 셀 수 없다. 표현이 정한 오차 바닥이다.
- **GNN은 데이터가 늘수록 계속 좋아진다.** 4층 message passing은 4홉 안의 원자 쌍 정보를 모을 수 있다.
- **이 결과를 일반화하지 않는다.** 이 데이터는 물성을 GNN 입력(원소, 차수)으로 직접 정의했으므로 GNN에 유리하다. 실제 화학 데이터에서는 적은 데이터에서 지문 모델이 그래프 모델과 비슷하거나 나은 경우도 흔하다. 실습의 요점은 어느 모델이 이기느냐가 아니라, **학습 곡선이 평평해지는 곳이 표현의 한계를 알려 준다**는 것이다. 시드를 바꿔 반복하고, 비국소 항을 빼면 두 곡선이 어떻게 바뀌는지 확인한다.

이어서 Coulomb 행렬 고윳값 기술자를 직접 구현한다. 테스트가 회전·이동·순열 불변성을 확인한다.

::exercise{id="coulomb-matrix"}

## 실패 사례

- **같은 분자의 서로 다른 SMILES 표기를 다른 샘플로 취급하는 것.** 데이터베이스 두 곳에서 모은 `OCC`와 `CCO`가 각각 train과 test에 들어가면, 모델은 시험에서 이미 본 분자를 맞힌다. 문자열 증강(같은 분자의 여러 SMILES를 학습에 쓰는 기법)을 쓸 때도 분할은 표준화한 분자 단위로 먼저 한다.
- **Tanimoto 1.0을 같은 분자로 판정하는 것.** 해시 충돌과 제한된 반경 때문에 다른 분자도 지문이 같을 수 있다. 중복 판정은 표준화한 식별자로 하고, 지문은 근접 중복 탐지에 쓴다.
- **추론 때 구할 수 없는 3차원 구조로 학습하는 것.** 양자화학으로 최적화한 구조로 학습한 모델을, 배포할 때는 빠른 형태 생성 도구의 구조로 쓰면 입력 분포가 달라진다.
- **키랄성이 중요한 물성에 반사 불변 표현을 쓰는 것.** 거울상 이성질체에 같은 예측이 나오는 것은 모델의 한계가 아니라 표현의 정의다.
- **결정의 전체 물성을 합 readout으로 예측하는 것.** 같은 결정을 초격자로 적으면 예측이 달라진다. 원자당 물성과 평균 readout을 쓴다.

## 통과 기준

- 맡은 과제 하나에 대해 예측 대상의 대칭(불변·등변, 반사 포함 여부)과 추론 시점에 쓸 수 있는 입력을 적고, 그에 맞는 표현을 표의 근거로 고른다.
- 데이터셋의 분자를 한 도구로 표준화한 뒤 중복 수를 세어 보고한다.
- 연습문제 `coulomb-matrix`의 테스트를 통과한다.

## 참고 문헌

- Weininger (1988). SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. *Journal of Chemical Information and Computer Sciences*, 28(1), 31–36. — SMILES 표기
- Rogers, Hahn (2010). Extended-Connectivity Fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754. — ECFP 지문의 구성
- Rupp, Tkatchenko, Müller, von Lilienfeld (2012). Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning. *Physical Review Letters*, 108, 058301. — Coulomb 행렬
- Ramakrishnan, Dral, Rupp, von Lilienfeld (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1, 140022. — QM9 데이터셋
- Xie, Grossman (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, 120, 145301. — 결정 그래프
- Wu, Ramsundar, Feinberg, Gomes, Geniesse, Pappu, Leswing, Pande (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9, 513–530. — 분자 기계학습 벤치마크와 특징 선택의 영향
