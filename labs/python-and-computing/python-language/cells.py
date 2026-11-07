"""5.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/python-and-computing/python-language/cells.py
"""
# %% cell 1 — 이름표와 복사의 깊이
import copy

a = [1, 2, 3]
b = a                      # 새 객체가 아니라 같은 객체에 이름표를 하나 더 붙인다
b.append(4)
print("a =", a, " a is b:", a is b)
x, y = [1, 2], [1, 2]
print("x == y:", x == y, " x is y:", x is y)

grid = [[0, 0], [0, 0]]
shallow = copy.copy(grid)          # list(grid), grid[:]와 같다
deep = copy.deepcopy(grid)
grid[0][0] = 99
print("shallow:", shallow, " deep:", deep)
print("shallow[0] is grid[0]:", shallow[0] is grid[0])

rows = [[0] * 3] * 2               # 같은 내부 리스트가 두 번 들어간다
rows[0][0] = 1
print("rows:", rows)

# %% cell 2 — 가변 기본 인자와 closure의 늦은 바인딩
def add_sample(x, batch=[]):       # 기본값은 def 문이 실행될 때 한 번 만들어진다
    batch.append(x)
    return batch

print(add_sample(1), add_sample(2))
print("기본값 객체:", add_sample.__defaults__)

def add_sample_fixed(x, batch=None):
    if batch is None:
        batch = []
    batch.append(x)
    return batch

print(add_sample_fixed(1), add_sample_fixed(2))

scalers = [lambda x: x * k for k in range(3)]
print("늦은 바인딩:", [f(10) for f in scalers])
scalers = [lambda x, k=k: x * k for k in range(3)]   # 정의 시점의 k를 기본값으로 고정
print("고정한 뒤  :", [f(10) for f in scalers])

# %% cell 3 — 예외가 나도 정리되는 context manager
from contextlib import contextmanager

@contextmanager
def run_scope(name):
    print(f"[{name}] 자원 확보")
    try:
        yield name
    finally:                       # 정상 종료든 예외든 반드시 실행된다
        print(f"[{name}] 자원 해제")

try:
    with run_scope("run-01"):
        raise FloatingPointError("loss가 nan이 됨")
except FloatingPointError as e:
    print("밖에서 잡은 예외:", e)

# %% cell 4 — generator는 지연 계산하고 한 번 소모된다
import sys

n = 100_000
squares_list = [i * i for i in range(n)]     # 원소 n개를 지금 모두 만든다
squares_gen = (i * i for i in range(n))      # 요청할 때 하나씩 만든다
print("list가 generator보다 1000배 이상 큰가:",
      sys.getsizeof(squares_list) > 1000 * sys.getsizeof(squares_gen))
print("첫 번째 sum:", sum(squares_gen))
print("두 번째 sum:", sum(squares_gen))      # 이미 소모되었다

def read_batches(n_samples, batch_size):
    for start in range(0, n_samples, batch_size):
        yield list(range(start, min(start + batch_size, n_samples)))

print([len(b) for b in read_batches(10, 4)])

# %% cell 5 — dataclass와 강제되지 않는 type hint
from dataclasses import dataclass, field

@dataclass
class MLPConfig:
    width: int = 64
    depth: int = 2
    tags: list[str] = field(default_factory=list)   # 인스턴스마다 새 리스트

c1, c2 = MLPConfig(), MLPConfig(depth=4)
c1.tags.append("baseline")
print(c1, c2, sep="\n")
print("값 비교 ==:", c1 == MLPConfig(tags=["baseline"]))

try:
    @dataclass
    class Bad:
        tags: list = []
except ValueError as e:
    print("ValueError:", e)

print(MLPConfig(width="64"))     # type hint는 실행 중에 검사되지 않는다
