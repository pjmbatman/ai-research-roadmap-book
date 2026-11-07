---
id: python-and-computing/python-language
title: "Python 언어의 핵심"
title_en: "Core Python"
tier: core
status: draft
prerequisites: []
---

# Python 언어의 핵심

연구 코드의 버그 가운데 상당수는 알고리즘이 아니라 언어의 동작을 잘못 안 데서 나온다. 실험 설정을 "복사"했는데 원본이 바뀌고, 실행마다 새로 시작해야 할 기록이 이전 실행의 값을 품고, epoch마다 다른 학습률을 쓰려 했는데 마지막 값만 쓰인다. 이런 버그는 오류 메시지 없이 숫자만 조금 틀리게 만들기 때문에, 논문 그림까지 살아남기 쉽다.

이 장은 Python 문법을 처음부터 가르치지 않는다. 반복문과 함수를 써 본 독자를 가정하고, 연구 코드에서 실제로 결과를 바꾸는 언어의 핵심 동작, 즉 **객체와 참조, 함수의 기본값과 closure, 자원 정리, 지연 계산, 클래스, type hint, 데코레이터**를 정확히 다룬다. 이후 NumPy(5.3장)와 PyTorch(17장)에서 만나는 view·복사·in-place 연산의 함정도 모두 이 장의 참조 모델 위에서 설명된다.

## 학습 목표

- 객체·참조·가변성을 정확히 이해한다
- 함수·클래스·제너레이터를 연구 코드에 맞게 사용한다
- type hint를 작성한다

## 객체와 참조

수학에서 $x = 3$은 "$x$의 값은 3이다"라는 선언이다. Python의 `x = [1, 2, 3]`은 다르다. 이 문장은 리스트 **객체**를 하나 만들고, `x`라는 **이름**(name)을 그 객체에 붙인다. 변수는 값을 담는 상자가 아니라 객체에 붙이는 **이름표**다. 그래서 `b = a`는 복사가 아니라 같은 객체에 이름표를 하나 더 붙이는 일이다. 이 모델을 정확히 쥐고 있지 않으면 "복사한 줄 알았던 객체가 공유되는" 버그를 예측할 수 없다.

모든 객체는 세 가지를 가진다.

- **정체**(identity): 객체가 살아 있는 동안 바뀌지 않는 고유한 값. `id(obj)`로 보고, `a is b`로 두 이름이 같은 객체를 가리키는지 묻는다.
- **타입**(type): 가능한 연산을 정한다. `type(obj)`.
- **값**(value): `a == b`는 값이 같은지 묻는다. 무엇을 같다고 볼지는 타입의 `__eq__`가 정한다.

객체의 값을 제자리에서 바꿀 수 있으면 **가변**(mutable), 없으면 **불변**(immutable)이다. `list`·`dict`·`set`과 NumPy 배열은 가변이고, `int`·`float`·`str`·`tuple`·`frozenset`은 불변이다. 불변 객체에 `x += 1`을 하면 객체가 바뀌는 것이 아니라 새 객체가 만들어지고 이름 `x`가 그쪽으로 옮겨 간다. 가변 객체에 `lst += [1]`을 하면 같은 객체가 제자리에서 바뀐다. 같은 `+=`가 타입에 따라 다른 일을 하는 셈이다.

```python {exec="browser" session="pylang"}
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
```

```output
a = [1, 2, 3, 4]  a is b: True
x == y: True  x is y: False
shallow: [[99, 0], [0, 0]]  deep: [[0, 0], [0, 0]]
shallow[0] is grid[0]: True
rows: [[1, 0, 0], [1, 0, 0]]
```

출력의 세 부분이 각각 하나의 규칙이다.

1. **할당은 복사하지 않는다.** `b = a` 뒤에는 이름이 둘, 객체는 하나다.
2. **복사에는 깊이가 있다.** 얕은 복사(shallow copy)는 바깥 컨테이너만 새로 만들고, 안에 담긴 객체는 같은 것을 가리킨다. 깊은 복사(deep copy)는 안쪽 객체까지 재귀적으로 새로 만든다. `list(x)`, `x[:]`, `x.copy()`, `dict(d)`, `copy.copy`는 모두 얕은 복사다.
3. **반복 연산자는 참조를 반복한다.** `[[0] * 3] * 2`는 내부 리스트 하나를 두 번 담은 리스트다. 격자를 만들 때는 `[[0] * 3 for _ in range(2)]`처럼 행마다 새 객체를 만든다.

`is`는 `x is None`처럼 객체의 정체를 물을 때만 쓴다. 값 비교에 `is`를 쓰면 작은 정수나 짧은 문자열에서 우연히 맞다가 다른 값에서 틀린다. CPython이 작은 정수 객체를 미리 만들어 재사용하는 구현 세부 때문이다. 최근 Python은 `x is 1`처럼 리터럴과 `is`로 비교하면 `SyntaxWarning`을 낸다.

::quiz{id="shallow-copy-nested"}

:::note[분야 예시]
2차원 열전도 시뮬레이션에서 온도 격자를 `T = [[20.0] * nx] * ny`로 만들고 왼쪽 경계 한 칸에 `T[0][0] = 100.0`을 넣으면, 모든 행의 첫 칸이 100이 된다. 경계 조건이 한 점이 아니라 한 줄 전체에 걸린 것이다. 시뮬레이션은 오류 없이 돌고 결과는 물리적으로 그럴듯해 보이기 때문에, 이런 버그는 해석해(analytic solution)와 비교하는 검증(9.7장) 전까지 드러나지 않는다. 실제 수치 계산에서는 처음부터 NumPy 배열 `np.full((ny, nx), 20.0)`을 쓰는 것이 맞지만, NumPy에도 같은 종류의 함정(view)이 있다(5.3장).
:::

**연구에서의 의미.** 참조 모델은 함수 인자 전달에도 그대로 적용된다. Python은 인자를 넘길 때 객체를 복사하지 않고 같은 객체에 매개변수 이름을 붙인다(call by sharing이라 부르기도 한다). 그래서 함수 안에서 인자로 받은 리스트·dict·배열을 제자리에서 바꾸면 호출한 쪽의 객체가 바뀐다. 전처리 함수가 입력 데이터를 조용히 바꾸면, 같은 데이터를 두 번째로 쓰는 평가 단계가 다른 입력을 보게 된다. 원칙은 두 가지다. 입력을 바꾸지 않는 함수는 새 객체를 반환하고, 입력을 바꾸는 함수는 이름(`normalize_`처럼 밑줄로 끝내는 PyTorch 관례)과 docstring으로 그 사실을 밝힌다.

## 함수

함수에 관한 대부분의 함정은 **언제 평가되는가**라는 질문 하나로 정리된다.

**인자와 기본값.** 매개변수의 기본값은 함수가 **정의될 때(def 문이 실행될 때) 한 번** 평가되어 함수 객체에 저장된다(`f.__defaults__`). 기본값이 가변 객체이면, 인자를 생략한 모든 호출이 그 객체 하나를 공유한다.

**가변 인자.** `def f(*args, **kwargs)`에서 `args`는 남는 위치 인자를 모은 tuple, `kwargs`는 남는 키워드 인자를 모은 dict다. 호출할 때 `f(*xs, **opts)`로 풀어 넘길 수도 있다. 래퍼 함수가 받은 인자를 그대로 넘길 때 쓰지만, 남용하면 어떤 인자가 허용되는지 시그니처만 보고 알 수 없게 된다. `def f(x, *, lr)`처럼 `*` 뒤에 둔 매개변수는 키워드로만 넘길 수 있어, `train(model, 0.001, 32)`처럼 숫자의 의미가 보이지 않는 호출을 막는다.

**scope와 closure.** 함수 안에서 이름을 찾는 순서는 지역(Local) → 감싸는 함수(Enclosing) → 모듈 전역(Global) → 내장(Built-in)이다(LEGB). 안쪽 함수가 바깥 함수의 변수를 쓰면, 안쪽 함수는 그 **변수 자체**를 붙잡는다. 이를 closure라 한다. 값이 아니라 변수를 붙잡으므로, 안쪽 함수가 **호출될 때** 그 변수의 현재 값을 읽는다. 이것이 늦은 바인딩(late binding)이다.

```python {exec="browser" session="pylang"}
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
```

```output
[1, 2] [1, 2]
기본값 객체: ([1, 2],)
[1] [2]
늦은 바인딩: [20, 20, 20]
고정한 뒤  : [0, 10, 20]
```

첫 줄이 `[1] [1, 2]`가 아니라 `[1, 2] [1, 2]`인 점에 주목한다. 두 호출이 **같은 리스트 객체**를 반환했고, `print`는 두 호출이 모두 끝난 뒤 그 객체를 두 번 출력했다. 고친 버전은 기본값으로 불변 객체 `None`을 두고, 호출할 때마다 새 리스트를 만든다.

::quiz{id="mutable-default-count"}

늦은 바인딩은 두 번째 결과를 설명한다. 세 lambda는 모두 같은 변수 `k`를 붙잡고, 호출 시점의 `k`는 반복이 끝난 뒤의 값 2다. 기본값은 정의 시점에 평가된다는 규칙을 거꾸로 이용해 `k=k`로 그 시점의 값을 고정하면 의도대로 동작한다. 같은 일을 `functools.partial(mul, k)`로도 할 수 있다.

바깥 함수의 변수에 **다시 할당**하려면 `nonlocal`을 선언해야 한다. 선언 없이 `count += 1`을 쓰면 `count`가 지역 변수로 취급되어 `UnboundLocalError`가 난다. 반면 붙잡은 리스트에 `append`하는 것은 재할당이 아니라서 선언 없이 된다. 이 비대칭도 "이름표를 옮기는가, 객체를 바꾸는가"의 차이다.

**연구에서의 의미.** 학습률 스케줄러, 콜백, 하이퍼파라미터 탐색 루프는 모두 함수를 만들어 나중에 호출하는 구조다. 반복문 안에서 lambda나 내부 함수를 만들어 리스트에 담는 코드를 보면 늦은 바인딩을 먼저 의심한다.

## 예외와 자원 관리

실험 코드는 중간에 실패한다. loss가 `nan`이 되거나, 디스크가 차거나, 사용자가 Ctrl-C를 누른다. 실패 자체보다 위험한 것은 실패한 뒤의 상태다. 파일이 반쯤 쓰인 checkpoint, 닫히지 않은 HDF5 파일, 잠긴 GPU 메모리가 다음 실행을 망친다.

**예외 처리.** `try` 블록에서 예외가 나면 맞는 `except`가 처리하고, `else`는 예외가 없을 때만, `finally`는 어떤 경우에도 실행된다. 원칙은 세 가지다.

- **처리할 수 있는 예외만 구체적으로 잡는다.** `except Exception: pass`는 버그를 숨긴다. 특히 `except:`(맨 except)는 Ctrl-C가 만드는 `KeyboardInterrupt`까지 삼킨다.
- **다시 던질 때 원인을 남긴다.** `raise ConfigError("lr이 음수") from e`는 원래 예외를 `__cause__`로 연결해 traceback에 두 예외가 모두 보인다.
- **조용한 실패보다 시끄러운 실패.** 예상하지 못한 상태(음수 분산, 빈 batch)를 만나면 그 자리에서 예외를 낸다. 실험이 끝난 뒤 이상한 숫자로 발견하는 것보다 비용이 훨씬 작다. 오류 처리 전략은 6.5장에서 자세히 다룬다.

**context manager.** `with` 문은 "확보 → 사용 → 해제"의 짝을 언어 수준에서 보장한다. `with open(path) as f:` 블록을 어떻게 빠져나가든(정상 종료, 예외, `return`) 파일은 닫힌다. `with` 문은 객체의 `__enter__`를 부르고, 블록을 떠날 때 `__exit__`를 부른다. `contextlib.contextmanager`를 쓰면 generator 하나로 context manager를 만들 수 있다. `yield` 앞이 확보, 뒤(`finally` 안)가 해제다.

```python {exec="browser" session="pylang"}
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
```

```output
[run-01] 자원 확보
[run-01] 자원 해제
밖에서 잡은 예외: loss가 nan이 됨
```

출력 순서가 핵심이다. 해제가 예외 처리**보다 먼저** 일어났다. 예외가 `with` 블록을 빠져나가는 길에 `finally`가 실행되고, 예외는 그대로 바깥으로 전달된다. `try`/`finally` 없이 `yield`만 쓰면 예외가 났을 때 해제 코드가 실행되지 않는다는 점에 주의한다.

**연구에서의 의미.** checkpoint는 임시 파일에 다 쓴 뒤 이름을 바꾸는 방식으로 저장하면, 쓰는 도중 실패해도 이전 checkpoint가 망가지지 않는다. 이런 "확보 → 사용 → 마무리 또는 되돌리기" 패턴을 context manager로 감싸 두면, 학습 루프 코드는 `with atomic_write(path) as f:` 한 줄로 안전해진다. 여러 자원을 동적으로 열어야 하면 `contextlib.ExitStack`을 쓴다.

## 반복과 지연 계산

**iterator.** `for x in obj`는 먼저 `iter(obj)`로 iterator를 얻고, `next()`를 반복해서 부르다가 `StopIteration` 예외가 나면 멈춘다. 리스트처럼 여러 번 순회할 수 있는 객체를 iterable, 한 번 순회하면 끝나는 상태 있는 객체를 iterator라 한다. 파일 객체, `zip`, `map`, generator는 모두 iterator다.

**generator.** 함수 본문에 `yield`가 있으면 generator 함수가 된다. 호출하면 본문을 실행하지 않고 generator 객체만 반환한다. `next()`를 부를 때마다 다음 `yield`까지 실행하고 멈춘다. 지역 변수와 실행 위치가 그대로 보존되므로, 상태를 가진 반복을 클래스 없이 쓸 수 있다. **지연 계산**(lazy evaluation)이란 값을 실제로 요청받을 때 만든다는 뜻이다.

**comprehension.** `[f(x) for x in xs if cond(x)]`는 리스트를, `{...}`는 set이나 dict를, 괄호 `(f(x) for x in xs)`는 generator를 만든다. 리스트 comprehension은 즉시 모든 원소를 만들고, generator 표현식은 요청할 때 하나씩 만든다.

```python {exec="browser" session="pylang"}
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
```

```output
list가 generator보다 1000배 이상 큰가: True
첫 번째 sum: 333328333350000
두 번째 sum: 0
[4, 4, 2]
```

`sys.getsizeof`는 컨테이너 자체의 크기만 재고 담긴 객체의 크기는 더하지 않으므로, 실제 메모리 차이는 이보다 더 크다. 리스트의 메모리는 원소 수 $n$에 비례하고($O(n)$), generator는 $n$과 무관하다($O(1)$). 대신 generator는 **한 번만** 순회할 수 있다. 두 번째 `sum`이 0인 것은 오류가 아니라 정의된 동작이어서 더 위험하다. 데이터셋을 generator로 만들어 두고 epoch 루프에서 재사용하면 두 번째 epoch부터 batch가 하나도 나오지 않는다. 학습 loss 곡선이 epoch 1 이후 갑자기 끊기면 이것을 의심한다.

::quiz{id="generator-exhausted"}

:::note[분야 예시]
유전체 서열 파일(FASTA)이나 기후 모델의 시간별 출력은 메모리에 한 번에 올리기 어려운 크기인 경우가 많다. 파일을 줄 단위로 읽는 generator로 레코드를 하나씩 만들고, 그 위에 필터와 batch 묶기를 generator로 쌓으면 전체 파이프라인이 한 번에 레코드 몇 개 분량의 메모리만 쓴다. 이런 streaming 처리의 계산 비용은 5.4장에서 다룬다.
:::

## 클래스

클래스는 데이터(속성)와 그 데이터에 대한 연산(메서드)을 묶는다. 연구 코드에서 클래스가 필요한 경우는 생각보다 적다. 상태가 없으면 함수로 충분하고, 값을 묶기만 하면 dataclass로 충분하다.

**dataclass.** `@dataclass`는 필드 선언(type hint가 붙은 클래스 변수)을 읽어 `__init__`, `__repr__`, `__eq__`를 자동으로 만든다. 실험 설정, 측정 레코드, 결과 요약처럼 "이름 붙은 값의 묶음"에 맞는다. `frozen=True`를 주면 인스턴스를 만든 뒤 필드를 바꿀 수 없어, 설정이 실행 도중 바뀌는 사고를 막는다.

```python {exec="browser" session="pylang"}
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
```

```output
MLPConfig(width=64, depth=2, tags=['baseline'])
MLPConfig(width=64, depth=4, tags=[])
값 비교 ==: True
ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory
MLPConfig(width='64', depth=2, tags=[])
```

dataclass는 함수의 가변 기본 인자와 같은 버그를 정의 시점에 막는다. 리스트·dict·set을 기본값으로 쓰면 `ValueError`를 내고 `default_factory`를 쓰라고 안내한다. 마지막 줄은 다음 절의 주제다. 필드에 `int`라고 적었지만 문자열 `"64"`가 그대로 들어갔다.

**magic method.** `__len__`, `__getitem__`, `__iter__`, `__enter__`/`__exit__`, `__call__`처럼 밑줄 두 개로 감싼 메서드는 언어의 문법과 내장 함수에 연결된다. `len(ds)`는 `ds.__len__()`을, `ds[i]`는 `ds.__getitem__(i)`를 부른다. PyTorch의 map-style `Dataset`이 `__len__`과 `__getitem__` 두 개만 구현하면 되는 것도 이 때문이다(17.4장). 연산자를 다시 정의하는 `__add__`, `__mul__` 등은 수학 객체(벡터, 다항식, 단위가 있는 물리량)를 표현할 때 유용하지만, 연산의 의미가 수학적으로 분명할 때만 쓴다.

**상속과 composition.** 상속(inheritance)은 "A는 B의 한 종류다"(is-a), composition은 "A는 B를 가지고 있다"(has-a)를 표현한다. 연구 코드에서 깊은 상속 계층은 읽기 어렵다. 모델의 동작을 이해하려면 부모 클래스를 몇 단계씩 거슬러 올라가야 하고, 부모를 고치면 모든 자식이 영향을 받는다. "학습기(Trainer)는 모델·optimizer·logger를 **가진다**"처럼 composition으로 조립하면 부품을 바꿔 끼우기 쉽고 테스트하기도 쉽다. 상속은 프레임워크가 요구할 때(`nn.Module`, `Dataset`)와 인터페이스를 공유할 때 주로 쓴다.

## 타입

type hint(타입 힌트)는 이름이 어떤 타입의 값을 가리킬 것으로 기대하는지 적는 주석이다. `def scale(x: float) -> float:`의 `: float`와 `-> float`가 그것이다. 앞 절의 마지막 출력처럼 **Python 인터프리터는 실행 중에 type hint를 검사하지 않는다.** hint는 함수 객체의 `__annotations__`에 저장될 뿐이다.

그런데도 type hint를 쓰는 이유는 세 가지다.

1. **정적 분석.** mypy, pyright 같은 정적 분석기(static type checker)는 코드를 실행하지 않고 읽어서 타입이 맞지 않는 호출을 찾는다. `scale("ab")`나 `None`일 수 있는 값에 `.shape`를 부르는 코드를 실행 전에 잡는다.
2. **문서.** 시그니처만 보고 무엇을 넘겨야 하는지 안다. 다른 사람(그리고 석 달 뒤의 자신)이 읽는 연구 코드에서 가장 싸고 정확한 문서다.
3. **편집기 지원.** 자동 완성과 이름 바꾸기 같은 refactoring이 정확해진다.

자주 쓰는 표기는 다음과 같다(Python 3.10 이상 기준).

```python
from collections.abc import Callable, Iterable, Sequence
import numpy as np

def mean_loss(losses: Sequence[float]) -> float: ...
def find(path: str, default: int | None = None) -> int | None: ...
def apply(fn: Callable[[float], float], xs: Iterable[float]) -> list[float]: ...
def normalize(x: np.ndarray) -> np.ndarray:        # shape는 주석으로: (N, D) -> (N, D)
    ...
```

`int | None`은 "정수이거나 None"이다. 인자 타입은 가능한 한 넓게(`list` 대신 `Sequence`나 `Iterable`), 반환 타입은 구체적으로 쓰면 호출하는 쪽의 자유도가 커진다. 배열의 shape와 dtype은 표준 type hint로 표현하기 어렵기 때문에, 이 책은 shape를 주석(`# (N, D)`)으로 적는 관례를 따른다.

**Protocol.** 상속 관계와 상관없이 "이런 메서드를 가진 객체면 된다"를 표현하려면 `typing.Protocol`을 쓴다. 구조적 부분타입(structural subtyping), 흔히 duck typing을 정적 분석기가 검사할 수 있게 만든 것이다.

```python
from typing import Protocol
import numpy as np

class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...

def evaluate(model: Model, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((model.predict(x) - y) ** 2))
```

`Model`을 상속하지 않았더라도 `predict` 메서드를 가진 모든 객체(scikit-learn 추정기, 직접 만든 물리 모델)를 `evaluate`에 넘길 수 있고, 정적 분석기는 `predict`가 없는 객체를 넘기는 호출을 오류로 표시한다.

**연구에서의 의미.** 정적 분석기를 CI(6.6장)에 넣으면 리팩토링 중에 생긴 인자 순서 실수 같은 버그를 실행 전에 잡는다. 그러나 정적 분석은 숫자 범위(학습률이 양수인가)나 배열 shape를 검사하지 못한다. 그런 조건은 함수 첫머리의 검사나 테스트로 확인해야 한다.

## 데코레이터

데코레이터(decorator)는 함수(또는 클래스)를 받아 함수(또는 클래스)를 돌려주는 함수다. 

```python
@decorator
def f(...): ...
```

는 `f = decorator(f)`와 정확히 같다. 앞에서 쓴 `@dataclass`, `@contextmanager`가 모두 데코레이터다. Python에서 함수는 값이므로 인자로 넘기고 반환할 수 있고, 이것이 데코레이터를 가능하게 한다.

**함수 감싸기.** 가장 흔한 형태는 원래 함수를 호출하는 안쪽 함수를 만들어 반환하는 것이다. 호출 횟수 세기, 실행 시간 기록, 입력 검사, 결과 캐시가 이런 식으로 구현된다. 감쌀 때는 `functools.wraps`를 붙여 원래 함수의 이름과 docstring을 보존해야 traceback과 문서 도구가 제대로 동작한다.

```python
import functools

def count_calls(fn):
    @functools.wraps(fn)                  # 이름·docstring을 원래 함수에서 복사
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@count_calls
def simulate(params): ...
# simulate(p)를 여러 번 부른 뒤 simulate.calls로 호출 횟수를 확인한다
```

표준 라이브러리의 `functools.lru_cache`(또는 `functools.cache`)는 같은 인자로 다시 호출하면 저장된 결과를 돌려주는 데코레이터다. 값비싼 순수 함수(같은 입력에 항상 같은 출력)를 반복 호출할 때 유용하지만, 인자가 hash 가능해야 하고(리스트·배열은 안 된다) 캐시가 메모리를 계속 차지한다는 점을 기억한다.

**등록 패턴(registry).** 데코레이터가 함수를 바꾸지 않고 어딘가에 **기록만** 하고 돌려주는 패턴이다. 설정 파일의 문자열(`model: mlp`)로 클래스를 고르는 코드에서 많이 쓴다.

```python
MODELS = {}

def register(name):
    def decorator(cls):
        MODELS[name] = cls                # 기록만 하고
        return cls                        # 클래스는 그대로 돌려준다
    return decorator

@register("mlp")
class MLP: ...

model_cls = MODELS[config["model"]]       # 설정 문자열로 클래스 선택
```

`register("mlp")`는 데코레이터를 **만드는** 함수다. 인자를 받는 데코레이터가 한 겹 더 감싸져 있는 이유다. 등록은 모듈이 **import될 때** 일어난다는 점도 기억한다. 모델 클래스를 정의한 모듈을 아무도 import하지 않으면 레지스트리는 비어 있고, `KeyError`가 난다. import가 언제 무엇을 실행하는지는 5.2장에서 다룬다.

:::note[분야 예시]
분자 시뮬레이션이나 배터리 열화 모델 연구에서는 같은 인터페이스를 가진 여러 물리 모델(힘장, 등가회로 모델)을 바꿔 가며 비교하는 일이 잦다. 각 모델 클래스에 `@register("lj")`, `@register("2rc")`처럼 이름을 붙여 두면, 설정 파일 한 줄로 실험할 모델을 고르고, 새 모델을 추가할 때 실험 스크립트를 고치지 않아도 된다.
:::

## 실습

이 장의 세 가지 대표 버그, 즉 **가변 기본 인자**, **리스트 공유 참조**, **closure의 늦은 바인딩**을 연구 코드의 모양으로 재현하고 고친다. 코드는 `labs/python-and-computing/python-language/three_bugs.py`다. 학습 기록 함수, 시뮬레이션 격자, epoch별 학습률 스케줄을 각각 버그 버전과 수정 버전으로 실행한다.

```output
1. 가변 기본 인자
  버그: run_b 기록 수 = 2, run_a is run_b = True
  수정: run_b 기록 수 = 1, run_a is run_b = False
2. 리스트 공유 참조
  버그: 값이 바뀐 칸 수 = 3, 서로 다른 행 객체 수 = 1
  수정: 값이 바뀐 칸 수 = 1, 서로 다른 행 객체 수 = 3
3. closure의 늦은 바인딩
  버그: epoch별 학습률 = [0.0001, 0.0001, 0.0001]
  수정: epoch별 학습률 = [0.01, 0.001, 0.0001]
덤: 얕은 복사로 만든 실험 설정
  얕은 복사: base의 lr = 0.01
  깊은 복사: base의 lr = 0.001
```

각 버그에 대해 다음을 순서대로 한다.

1. 코드를 실행하기 **전에** 버그 버전의 출력을 예측하고, 이름표와 객체를 화살표로 잇는 참조 그림을 그린다.
2. 실행해서 예측과 비교한다. 틀렸다면 그림의 어느 화살표가 틀렸는지 찾는다.
3. `is`와 `id()`로 그림을 확인한다(출력의 `run_a is run_b`, 서로 다른 행 객체 수가 그 예다).

이어서 두 연습문제를 푼다. 첫 번째는 실험 설정 병합에서 얕은 복사가 만드는 공유를 피하는 문제, 두 번째는 끝이 없는 스트림에서도 동작하는 지연 계산 generator를 만드는 문제다. 테스트가 바로 채점한다.

::exercise{id="merge-config"}

::exercise{id="lazy-batches"}

## 실패 사례

- **함수 기본값에 리스트·dict를 쓰는 것.** 기본값은 정의 시점에 한 번 만들어져 모든 호출이 공유한다. 학습 기록이 실행 사이에 누적되거나, 캐시가 의도치 않게 남는다. 기본값은 `None`으로 두고 함수 안에서 새 객체를 만든다. dataclass에서는 `field(default_factory=list)`를 쓴다.
- **복사한 줄 알았던 객체가 공유되는 것.** `b = a`는 복사가 아니고, `list(a)`, `a.copy()`, `dict(cfg)`는 얕은 복사다. 중첩된 설정 dict를 얕게 복사해 하이퍼파라미터 탐색의 각 실행에 넘기면, 한 실행이 바꾼 값이 다른 실행의 설정에 새어 든다. 중첩 구조를 독립적으로 바꿔야 하면 `copy.deepcopy`를 쓰고, 가능하면 설정을 `frozen=True` dataclass처럼 바꿀 수 없게 만든다.
- **값 비교에 `is`를 쓰는 것.** `if status is "done"`이나 `if n is 1000`은 구현의 캐시에 따라 맞기도 하고 틀리기도 한다. `is`는 `None`과 비교할 때만 쓴다.
- **generator를 두 번 순회하는 것.** 두 번째 순회는 오류 없이 아무것도 내보내지 않는다. 여러 epoch에 걸쳐 쓸 데이터는 iterable(리스트, 또는 `__iter__`가 매번 새 generator를 만드는 클래스)로 만든다.
- **`except Exception: pass`로 실패를 숨기는 것.** 수천 개의 시뮬레이션 중 일부가 조용히 실패하고, 결과 표에는 성공한 것만 남아 편향된 결론이 나온다. 잡은 예외는 기록하고, 실패한 실행의 수를 결과와 함께 보고한다.

## 통과 기준

- 주어진 짧은 코드(할당·복사·함수 호출이 섞인 10줄 안팎)의 이름–객체 참조 그림을 그리고, 실행하기 전에 출력과 버그를 예측한다. `three_bugs.py`의 세 버그에 대해 예측이 실행 결과와 일치한다.
- 연습문제 두 개(`merge-config`, `lazy-batches`)의 테스트를 통과한다.
- 자신이 쓴 함수 하나에 type hint를 달고, 정적 분석기로 잘못된 호출 하나를 잡아 그 경고를 확인한다.

## 참고 문헌

- Python Software Foundation. *The Python Language Reference* — Data model(객체의 정체·타입·값, 가변성), Execution model(이름과 바인딩). Python 공식 문서.
- Python Software Foundation. *The Python Tutorial* — More on Defining Functions의 Default Argument Values(기본값은 한 번만 평가된다는 경고). Python 공식 문서.
- PEP 484 — type hint의 문법과 의미. 표준 라이브러리 `typing` 모듈이 이 PEP를 근거로 한다.
- PEP 544 — Protocol을 이용한 구조적 부분타입(static duck typing).
