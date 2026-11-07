"""15.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/dataset-versioning/cells.py
"""
# %% cell 1 — 같은 내용, 다른 바이트, 다른 해시
import hashlib
import io
import numpy as np
import pandas as pd

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

df = pd.DataFrame({
    "cell_id": np.repeat(["C00", "C01", "C02"], 4),
    "cycle": np.tile(np.arange(1, 5), 3),
    "capacity_ah": [2.0, 1.99, 1.985, 1.98, 2.01, 2.0, 1.995, 1.99, 1.97, 1.96, 1.955, 1.95],
})                                                           # (12, 3)
variants = {
    "원본       ": df.to_csv(index=False).encode(),
    "줄 끝 CRLF ": df.to_csv(index=False, lineterminator="\r\n").encode(),
    "행 순서 다름": df.sort_values("capacity_ah").to_csv(index=False).encode(),
}
for name, b in variants.items():
    print(name, sha256(b)[:16])

def canonical(b: bytes) -> bytes:                            # 읽어서 정렬한 뒤 정해진 형식으로 다시 쓴다
    d = pd.read_csv(io.BytesIO(b)).sort_values(["cell_id", "cycle"])
    return d.to_csv(index=False, lineterminator="\n").encode()

print("정규화 후 해시가 모두 같은가:", len({sha256(canonical(b)) for b in variants.values()}) == 1)

# %% cell 2 — 파일 목록(manifest)과 데이터셋 해시
files = {f"raw/{c}.csv": df[df["cell_id"] == c].to_csv(index=False).encode()
         for c in ["C00", "C01", "C02"]}

def manifest(files: dict) -> str:
    return "".join(f"{sha256(files[p])}  {len(files[p])}  {p}\n" for p in sorted(files))

m1 = manifest(files)
for line in m1.splitlines():
    h, size, path = line.split("  ")
    print(h[:16], size, path)                                 # 화면에는 해시 앞 16자리만
print("데이터셋 해시 v1:", sha256(m1.encode())[:16])

files_v2 = dict(files)
files_v2["raw/C01.csv"] += b"C01,5,1.985\n"                  # 누군가 조용히 한 줄을 추가했다
m2 = manifest(files_v2)
print("데이터셋 해시 v2:", sha256(m2.encode())[:16])
by_path = lambda m: {line.split("  ")[2]: line for line in m.splitlines()}
old, new = by_path(m1), by_path(m2)
print("바뀐 파일:", [p for p in new if old.get(p) != new[p]])

# %% cell 3 — 해시로 정한 분할은 데이터가 늘어도 흔들리지 않는다
def hash_split(group_id: str, test_frac: float = 0.2) -> str:
    h = int(sha256(group_id.encode()), 16)
    return "test" if h % 1000 < test_frac * 1000 else "train"

def perm_split(ids: list, seed: int = 0, test_frac: float = 0.2) -> dict:
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_test = int(test_frac * len(ids))
    return {ids[i]: ("test" if r < n_test else "train") for r, i in enumerate(perm)}

v1 = [f"C{i:03d}" for i in range(100)]                       # 셀 100개
v2 = v1 + [f"C{i:03d}" for i in range(100, 150)]              # 셀 50개 추가
p1, p2 = perm_split(v1), perm_split(v2)
h1, h2 = {c: hash_split(c) for c in v1}, {c: hash_split(c) for c in v2}
print("무작위 순열: 기존 셀 중 분할이 바뀐 수", sum(p1[c] != p2[c] for c in v1))
print("해시 기반  : 기존 셀 중 분할이 바뀐 수", sum(h1[c] != h2[c] for c in v1))
n1, n2 = list(h1.values()).count("test"), list(h2.values()).count("test")
print(f"해시 기반 test 셀 수: v1 {n1}/100, v2 {n2}/150")

# %% cell 4 — 소금 없는 해시는 가명이 아니다
import hmac

patient = "P0427"
leaked = sha256(patient.encode())                             # 공개된 데이터의 "익명" ID
table = {sha256(f"P{i:04d}".encode()): f"P{i:04d}" for i in range(10_000)}
print("가능한 ID 1만 개를 해시해 되찾은 ID:", table[leaked])

key = b"store-this-key-outside-the-dataset"                   # 실제 키는 데이터와 따로 보관
pseudo = hmac.new(key, patient.encode(), hashlib.sha256).hexdigest()
print("키가 있는 HMAC 가명이 표에 있는가:", pseudo in table)
