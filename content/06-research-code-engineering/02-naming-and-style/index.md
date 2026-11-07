---
id: research-code-engineering/naming-and-style
title: "네이밍·코딩 컨벤션"
title_en: "Naming and Coding Conventions"
tier: core
status: draft
prerequisites: [research-code-engineering/project-structure]
---

# 네이밍·코딩 컨벤션

연구 코드의 버그 중 상당수는 계산이 아니라 **해석**에서 생긴다. 밀리볼트를 볼트로 읽고, 분을 초로 읽고, 열벡터를 행벡터로 읽고, 입력의 마지막 시각을 예측의 첫 시각으로 읽는다. 이런 버그는 오류 없이 실행되고 그럴듯한 숫자를 낸다. 이름이 단위·shape·시간 기준을 드러내면, 코드를 읽는 사람이 계산을 따라가지 않고도 틀린 줄을 알아본다.

이 장은 일관된 이름 규칙, 의미를 드러내는 이름, 코드 스타일, 그리고 이 규칙을 사람의 기억이 아니라 도구로 강제하는 방법을 다룬다. 스타일 논쟁에 시간을 쓰지 않으려면 규칙을 정하고 도구에 맡겨야 한다.

## 학습 목표

- 일관된 이름 규칙을 적용한다
- shape·단위·시간 기준을 이름에 드러낸다
- 자동 도구로 스타일을 강제한다

## 이름 규칙

**왜 필요한가.** 같은 개념이 파일마다 `n_samples`, `num_samples`, `N`, `size`로 불리면, 읽는 사람은 이것들이 같은 것인지 매번 확인해야 한다. 이름 규칙은 "이 이름은 무엇의 종류인가"를 이름의 모양만으로 알려 준다.

**정의.** Python 코드는 PEP 8(Python의 공식 스타일 가이드)의 이름 관례를 기본으로 따른다. 연구 프로젝트에서는 파일·디렉터리·실험 이름까지 규칙을 넓힌다.

| 대상 | 규칙 | 예 |
|---|---|---|
| 변수 · 함수 · 메서드 | `snake_case` | `load_cycles`, `capacity_ah` |
| 클래스 | `CapWords` | `CycleDataset`, `SohRegressor` |
| 모듈 상수 | `UPPER_SNAKE_CASE` | `MAX_CYCLES`, `FARADAY_C_PER_MOL` |
| 내부용(비공개) | 앞에 밑줄 하나 | `_interpolate_gaps` |
| Python 모듈 · 패키지 | 짧은 `snake_case` | `soh_model/data/cycles.py` |
| 그 밖의 파일 · 디렉터리 | 연구실 규칙 하나(예: 소문자 `kebab-case`) | `configs/lstm-small.toml` |
| 실험 이름 | 구성 요소를 정해진 순서로 | `soh-lstm-small-seed0` |

Python 모듈 이름은 Python 식별자여야 import할 수 있다. `my-module.py`는 `import my-module`로 import할 수 없으므로 하이픈 대신 밑줄을 쓴다. 설정·문서처럼 import하지 않는 파일은 연구실이 정한 한 가지 규칙을 따른다.

**품사와 접두사.** 이름의 모양 외에 의미의 관례도 정한다.

- 함수는 동사로 시작한다: `load_`, `compute_`, `fit_`, `plot_`, `save_`.
- 변수는 명사다. 컬렉션은 복수형(`cells`), 원소는 단수형(`cell`)으로 쓴다.
- 개수는 `n_` 접두사(`n_cycles`), 인덱스는 `_idx` 접미사(`cycle_idx`), 참·거짓은 `is_`·`has_` 접두사(`is_valid`)를 쓴다.
- 분야에서 표준인 약어(`soc`, `soh`, `rmse`, `dft`)는 쓰되, 연구실 스타일 가이드의 약어 목록에 한 번 정의한다. 그 밖의 줄임말(`cnt`, `vlt`, `prc`)은 쓰지 않는다.

**실험 이름.** 실험 이름은 사람이 결과를 찾고 묶는 데 쓴다. 데이터·모델·주요 변형·seed처럼 구성 요소의 순서를 정해 두면 정렬과 검색이 쉬워진다. 다만 실험 이름은 요약일 뿐이다. 그 실험의 정확한 설정은 이름이 아니라 저장된 설정 파일(6.3장)이 원천이다. `lr3e-4`를 이름에 넣었는데 설정에서는 다른 학습률이 쓰였다면, 이름이 거짓말을 하게 된다.

## 의미를 드러내는 이름

이름 규칙이 이름의 모양을 정한다면, 이 절은 이름이 **무엇을 알려야 하는지**를 다룬다. 연구 코드에서 이름이 드러내야 할 정보는 세 가지다. 단위, shape, 시간 기준이다.

**단위.** 물리량을 담는 변수에는 단위 접미사를 붙인다. 이 책의 표기 규칙도 같다.

| 접미사 | 단위 | 접미사 | 단위 |
|---|---|---|---|
| `_v`, `_mv` | 볼트, 밀리볼트 | `_s`, `_ms`, `_min`, `_h` | 초, 밀리초, 분, 시간 |
| `_a`, `_ma` | 암페어, 밀리암페어 | `_c`, `_k` | 섭씨, 켈빈 |
| `_ah`, `_mah` | 암페어시, 밀리암페어시 | `_wh`, `_j` | 와트시, 줄 |

단위가 이름에 없으면 어떤 일이 생기는지 본다. 계측기가 전압을 mV로 저장했는데, 분석 코드는 그 배열을 그냥 `voltage`라고 부르며 V로 가정했다.

```python {exec="browser" session="naming"}
import numpy as np

# 정전류 방전 기록. 전압은 계측기가 mV로 저장했다
voltage = np.array([4150.0, 3900.0, 3700.0, 3500.0])   # (T,) 실제 단위는 mV
current, dt = 2.0, 600.0
energy = np.sum(voltage * current * dt) / 3600          # Wh라고 믿었다
print(f"단위를 모른 채 계산  : {energy:,.1f} Wh")

voltage_mv = voltage                                    # (T,)
voltage_v = voltage_mv / 1000.0                         # (T,)
current_a, dt_s = 2.0, 600.0
energy_wh = np.sum(voltage_v * current_a * dt_s) / 3600.0
print(f"단위를 이름에 쓴 계산: {energy_wh:.3f} Wh")
```

```output
단위를 모른 채 계산  : 5,083.3 Wh
단위를 이름에 쓴 계산: 5.083 Wh
```

첫 계산은 정확히 1000배 틀렸다. 단위를 이름에 쓰면 `voltage_mv * current_a`라는 식을 쓰는 순간 어색함이 보인다. 두 단위를 곱한 결과에 어떤 접미사를 붙여야 할지 답할 수 없기 때문이다. 이름이 계산을 검사하는 셈이다.

**shape.** 배열과 텐서에는 shape 주석을 단다. 축 이름은 이 책의 약어를 따른다: `B` batch, `T` time, `C` channel, `D` feature, `N` 샘플 수. 주석이 있으면 연산마다 결과 shape을 머릿속으로 확인할 수 있고, 어긋날 수 있는 곳에는 `assert`를 넣는다. 다음은 연구 코드에서 아주 흔한 broadcasting 버그다.

```python {exec="browser" session="naming"}
rng = np.random.default_rng(0)
y_true = rng.normal(size=5)                 # (N,)
y_pred = y_true[:, None] + 0.1              # (N, 1) 모델이 열벡터로 반환했다
err = y_pred - y_true                       # (N, 1) - (N,) → (N, N)
print("err.shape =", err.shape, " MSE =", round(float(np.mean(err**2)), 4))

y_pred = y_pred[:, 0]                       # (N,) shape을 명시적으로 맞춘다
assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
err = y_pred - y_true                       # (N,)
print("err.shape =", err.shape, "   MSE =", round(float(np.mean(err**2)), 4))
```

```output
err.shape = (5, 5)  MSE = 0.3032
err.shape = (5,)    MSE = 0.01
```

예측은 모든 점에서 정확히 0.1만큼 벗어났으므로 MSE는 $0.1^2 = 0.01$이어야 한다. 첫 계산은 `(N, 1)`과 `(N,)`이 `(N, N)`으로 broadcasting되어, 모든 예측과 모든 정답의 쌍을 비교한 무의미한 숫자를 냈다. 오류는 나지 않았다.

::quiz{id="broadcast-shape"}

**시간 기준.** 시계열 코드에서 가장 흔한 off-by-one은 "기준 시각"이 무엇인지 정하지 않아서 생긴다. 이름과 docstring에 다음을 정해 둔다.

- `t0`: 입력의 마지막 시각인가, 첫 예측 시각인가.
- `lookback`: 입력에 몇 스텝을 쓰는가. `horizon`: 몇 스텝 앞까지 예측하는가.
- 인덱스인가, 물리 시간인가: `t_idx`와 `t_s`를 구분한다.
- 시간대: 타임스탬프는 `_utc`처럼 기준을 붙이거나 처음부터 UTC로 통일한다.

```python {exec="browser" session="naming"}
series = np.arange(10.0)                    # (T,) 시각 t의 값이 t

def make_window(series, t0, lookback, horizon):
    """입력: 시각 t0-lookback+1 … t0 (t0 포함), 목표: t0+1 … t0+horizon"""
    x = series[t0 - lookback + 1 : t0 + 1]  # (lookback,)
    y = series[t0 + 1 : t0 + 1 + horizon]   # (horizon,)
    return x, y

x, y = make_window(series, t0=5, lookback=3, horizon=2)
print("입력 시각:", x, " 목표 시각:", y)
y_bad = series[5 : 5 + 2]                   # 흔한 실수: 목표를 t0부터 자른다
print("잘못 자른 목표:", y_bad, " 입력과 겹친 값:", np.intersect1d(x, y_bad))
```

```output
입력 시각: [3. 4. 5.]  목표 시각: [6. 7.]
잘못 자른 목표: [5. 6.]  입력과 겹친 값: [5.]
```

목표를 `t0`부터 자르면 입력의 마지막 값이 목표의 첫 값이 된다. 모델은 첫 스텝을 입력 복사로 맞히고, 검증 성능은 실제보다 좋게 나온다. 값을 시각과 같게 만든 장난감 시계열(`series[t] = t`)로 창을 출력해 보면 이런 겹침이 바로 보인다.

::quiz{id="forecast-window-leak"}

:::note[분야 예시]
유전체 좌표는 형식마다 기준이 다르다. BED 형식은 0부터 세고 끝 위치를 포함하지 않는(0-based, half-open) 구간을, VCF와 GFF 형식은 1부터 세는 위치를 쓴다. 두 형식을 함께 다루는 코드에서 `start`라는 이름만 쓰면, 한 염기씩 어긋난 구간이 조용히 섞인다. `start_0based`, `pos_1based`처럼 기준을 이름에 넣으면 변환이 필요한 곳이 이름에서 드러난다.
:::

## 코드 스타일

**PEP 8의 핵심.** 스타일 규칙의 목적은 읽는 사람이 코드의 모양이 아니라 내용에 집중하게 하는 것이다. 연구 코드에서 특히 중요한 것은 다음 몇 가지다.

- 들여쓰기는 공백 4칸.
- 한 줄의 길이를 제한한다. PEP 8은 79자를 권하고, 많은 formatter는 그보다 조금 긴 값을 기본으로 쓴다(집필 시점 예시: ruff의 기본값은 88자). 프로젝트에서 하나를 정해 도구 설정에 적는다.
- import는 표준 라이브러리, 서드파티, 프로젝트 내부 순서로 묶고, 묶음 사이에 빈 줄을 둔다. `from module import *`는 쓰지 않는다. 어떤 이름이 어디서 왔는지 알 수 없게 된다.

**함수 길이.** 함수 하나는 한 가지 일을 한다. 줄 수로 정한 규칙보다 "이 함수가 하는 일을 한 문장으로 말할 수 있는가"가 더 좋은 기준이다. "데이터를 읽고, 정규화하고, 모델을 학습하고, 그림을 그린다"처럼 "그리고"가 여러 번 들어가면 나눌 때다. 짧은 함수는 테스트하기 쉽고(6.6장), 이름이 곧 설명이 된다.

**주석과 docstring.** 주석은 코드가 **무엇을** 하는지가 아니라 **왜** 그렇게 하는지를 쓴다. `i += 1  # i를 1 증가`는 쓸모가 없고, `# 첫 사이클은 formation 단계라 용량이 불안정하므로 제외`는 코드만으로 알 수 없는 정보를 준다. 함수의 docstring에는 하는 일, 인자와 반환값의 shape·단위, 발생시키는 예외를 쓴다.

```python
def coulomb_count(current_a, dt_s, capacity_ah, soc0):
    """전류를 적분해 SOC(state of charge)를 추정한다.

    Args:
        current_a: (T,) 방전을 양수로 한 전류 [A]
        dt_s: 샘플 간격 [s]
        capacity_ah: 정격 용량 [Ah]
        soc0: 시작 SOC, 0–1
    Returns:
        (T,) 각 샘플 시각의 SOC, 0–1
    """
    return soc0 - np.cumsum(current_a) * dt_s / 3600.0 / capacity_ah
```

docstring에 무엇을 어떻게 쓸지는 6.7장에서 자세히 다룬다. 여기서는 shape과 단위가 docstring에 빠지지 않는 것이 핵심이다.

## 자동화 도구

**왜 필요한가.** 사람은 규칙을 일관되게 지키지 못하고, 코드 리뷰에서 공백과 따옴표를 지적하는 데 시간을 쓰면 정작 중요한 로직을 보지 못한다. 기계가 확인할 수 있는 규칙은 기계에 맡긴다. 도구는 세 종류다.

| 종류 | 하는 일 | 바꾸는가 | 예(집필 시점) |
|---|---|---|---|
| formatter | 공백·줄바꿈·따옴표 같은 모양을 통일 | 코드 모양을 고친다 | ruff format, black |
| linter | 쓰지 않는 import, 정의되지 않은 이름, 위험한 패턴, 이름 규칙 위반 | 보고만, 일부 자동 수정 | ruff check, flake8, pylint |
| type checker | type hint와 맞지 않는 호출 | 보고만 | mypy, pyright |

linter는 코드를 실행하지 않는다. 소스 코드를 **구문 트리**(abstract syntax tree, AST)로 읽고 규칙에 맞지 않는 모양을 찾는다. 표준 라이브러리 `ast`로 같은 일을 흉내 내 본다. 함수 이름·인자·대입되는 변수 이름 중 의미를 드러내지 않는 것을 찾는다.

```python {exec="browser" session="naming"}
import ast
import re

source = '''
def f(x, x2):
    tmp = x * 1000
    final_final = tmp + x2
    voltage_v = final_final / 1000
    return voltage_v
'''
BAD = re.compile(r"^([a-z]|[a-z]\d+|tmp\d*|final.*)$")
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.FunctionDef):
        names = [node.name] + [a.arg for a in node.args.args]
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        names = [node.id]
    else:
        continue
    for name in names:
        if BAD.match(name):
            print(f"{node.lineno}행 {name!r}: 의미를 드러내지 않는 이름")
```

```output
2행 'f': 의미를 드러내지 않는 이름
2행 'x': 의미를 드러내지 않는 이름
2행 'x2': 의미를 드러내지 않는 이름
3행 'tmp': 의미를 드러내지 않는 이름
4행 'final_final': 의미를 드러내지 않는 이름
```

실제 linter는 수백 개의 규칙을 같은 방식으로 검사한다. 실습 디렉터리의 `before_lint.py`는 일부러 규칙을 어긴 파일이다. 여기에 ruff의 linter를 실행한 결과는 다음과 같다(집필 시점 예시, ruff 0.16.9. 도구를 처음 쓸 때는 `ruff --version`으로 버전을, `ruff rule <코드>`로 규칙의 설명을 확인한다).

```bash
cd labs/research-code-engineering/naming-and-style
ruff check --select E,F,I,B,N --output-format concise before_lint.py
```

```output
before_lint.py:5:1: I001 [*] Import block is un-sorted or un-formatted
before_lint.py:5:8: F401 [*] `os` imported but unused
before_lint.py:9:5: N802 Function name `LoadCycles` should be lowercase
before_lint.py:9:28: B006 Do not use mutable data structures for argument defaults
before_lint.py:17:12: N803 Argument name `V` should be lowercase
before_lint.py:17:15: N803 Argument name `I` should be lowercase
before_lint.py:17:15: E741 Ambiguous variable name: `I`
before_lint.py:18:5: E741 Ambiguous variable name: `l`
Found 8 errors.
[*] 2 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

`--select`의 문자는 규칙 묶음이다. `E`는 pycodestyle(PEP 8 모양), `F`는 pyflakes(쓰지 않는 import, 정의되지 않은 이름), `I`는 import 정렬, `B`는 흔한 버그 패턴(flake8-bugbear), `N`은 이름 규칙(pep8-naming)에서 온 규칙이다. `B006`이 잡은 가변 기본값(`cache={}`)은 함수 호출 사이에 상태가 공유되는 실제 버그의 원인이다.

같은 파일에 formatter를 검사 모드로 실행하면, 고칠 모양을 diff로 보여 준다. 연산자 주변 공백처럼 의미와 상관없는 것만 바꾼다. 쓰지 않는 import는 그대로 둔다. 그것은 linter의 일이다.

```bash
ruff format --diff before_lint.py
```

```output
--- before_lint.py
+++ before_lint.py
@@ -2,6 +2,7 @@
 
 집필 시점 예시: ruff check --select E,F,I,B,N before_lint.py
 """
+
 import os
 import numpy as np
 
@@ -15,5 +16,5 @@
 
 
 def energy(V, I, t):
-    l = V*I*t
-    return l.sum()/3600
+    l = V * I * t
+    return l.sum() / 3600

1 file would be reformatted
```

type checker는 type hint를 근거로 호출을 검사한다. 실습 디렉터리의 `units_typed.py`는 `def to_volts(voltage_mv: float) -> float`에 문자열 `"3700"`을 넘긴다. mypy로 검사하면 다음과 같이 보고한다(집필 시점 예시, mypy 2.3.1).

```bash
mypy units_typed.py
```

```output
units_typed.py:5: error: Argument 1 to "to_volts" has incompatible type "str"; expected "float"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

::quiz{id="formatter-vs-linter"}

**설정.** 도구 설정은 `pyproject.toml`에 모은다. 다음은 ruff 설정의 예다(집필 시점 예시. 설정 키는 도구 문서에서 확인한다).

```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "N"]
```

**pre-commit.** pre-commit은 Git의 커밋 직전 hook에서 도구를 실행해 주는 프레임워크다. 저장소에 `.pre-commit-config.yaml`을 두고, 각자 한 번 `pre-commit install`을 실행하면 이후 커밋마다 검사가 돌아간다. 기존 파일 전체는 `pre-commit run --all-files`로 검사한다.

```yaml
# .pre-commit-config.yaml (집필 시점 예시)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: "<태그>"            # `pre-commit autoupdate`로 최신 태그를 채운다
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
```

hook의 `id`와 쓸 수 있는 태그는 hook 저장소의 `.pre-commit-hooks.yaml`과 태그 목록에서 확인한다. pre-commit은 각자 설치해야 동작하고 `git commit --no-verify`로 건너뛸 수 있다. 그래서 규칙을 **강제**하는 곳은 CI다. 모든 PR에서 `ruff check .`와 `ruff format --check .`를 실행해, 하나라도 실패하면 병합하지 않는다. CI 구성은 6.6장에서 다룬다.

## 연구실 스타일 가이드

PEP 8과 도구 기본값이 대부분을 정해 주지만, 연구실마다 정해야 하는 것이 남는다. 연구실 스타일 가이드에는 **PEP 8과 다른 점, 그리고 PEP 8이 말하지 않는 것**만 쓴다. 전체를 다시 쓰면 아무도 읽지 않는다.

- **단위 접미사 목록**과 단위가 없는 양의 표기(`_frac`, `_pct` 등)
- **shape 축 약어**(`B`, `T`, `C`, `D`, `N`)와 shape 주석 형식
- **분야 약어 목록**(`soc`, `soh`, `dft`, `sst` 등)과 그 정의
- **시간 기준**: 타임스탬프의 시간대, `t0`의 의미
- **파일·실험 이름 규칙**
- **도구와 설정**: 어떤 도구를 어떤 설정으로 쓰는지는 문서가 아니라 `pyproject.toml`과 pre-commit 설정이 원천이다. 문서에는 그 위치만 쓴다.

**예외 처리.** 규칙에는 정당한 예외가 있다. 수식을 옮긴 코드에서는 행렬을 `W`, `X`처럼 대문자로 쓰는 것이 논문과 대응하기 쉽다. 이런 예외는 범위를 좁혀 설정에 적는다.

```toml
[tool.ruff.lint.per-file-ignores]
"src/*/models/*.py" = ["N806"]   # 모델 코드에서는 행렬을 대문자 변수로 허용
```

한 줄만 예외로 할 때는 `# noqa: E741`처럼 규칙 코드를 명시하고, 왜 예외인지 짧은 주석을 붙인다. 코드 없는 `# noqa`나 파일 전체 무시는 다른 문제까지 숨긴다.

**기존 코드에 적용하기.** 규칙을 기존 저장소에 처음 적용할 때는 formatter가 바꾼 내용만 담은 커밋을 따로 만든다. 기능 변경과 섞으면 리뷰어가 수천 줄의 공백 변경 속에서 실제 변경을 찾아야 한다. `git blame`은 이런 대량 포매팅 커밋을 건너뛰는 옵션(`--ignore-rev`, `--ignore-revs-file`)을 지원하므로, 포매팅 커밋의 해시를 파일 하나에 모아 두면 줄마다의 변경 이력을 계속 추적할 수 있다. 사용법은 `git help blame`에서 확인한다.

:::note[분야 예시]
수치 시뮬레이션 코드를 논문의 수식과 함께 읽어야 하는 물리 연구실은 수식 기호를 변수 이름에 그대로 쓰는 예외를 둘 수 있다. 이때도 단위 접미사는 지킨다. 예를 들어 열전도 방정식의 확산 계수를 `alpha`로 쓰더라도 `alpha_m2_per_s`처럼 단위를 붙이면, 무차원화한 값과 물리 단위 값이 섞이는 일을 막는다.
:::

## 실습

기존 코드에 연구실 네이밍 규칙과 lint를 적용하고, 이름 변경만으로 발견된 버그를 기록한다.

1. 실습 디렉터리의 `before_lint.py`에 linter와 formatter를 실행해 결과를 읽는다. 각 규칙 코드의 설명을 `ruff rule <코드>`로 확인하고, 어떤 것이 실제 버그이고 어떤 것이 모양의 문제인지 나눈다.
2. 자신의 프로젝트(또는 6.1장의 템플릿 프로젝트)의 `pyproject.toml`에 formatter·linter 설정을, 저장소에 `.pre-commit-config.yaml`을 추가하고 `pre-commit run --all-files`를 실행한다. formatter의 변경은 별도 커밋으로 만든다.
3. 물리량 변수에 단위 접미사를, 배열에 shape 주석을, 시계열 함수에 `t0`·`horizon`의 정의를 붙인다. 이름을 바꾸다 단위·shape·시간 기준이 어긋난 곳을 발견하면 다음 형식으로 기록한다.

| 위치 | 바꾼 이름 | 발견한 문제 | 결과에 미친 영향 |
|---|---|---|---|
| `analysis/energy.py:42` | `v` → `voltage_mv` | V로 가정하고 계산 | 에너지 1000배 과대 |

이름만 바꿨는데 버그가 하나도 나오지 않았다면, 그것도 기록할 가치가 있는 결과다.

마지막으로 단위 접미사를 읽어 SI 단위로 통일하는 함수를 연습문제로 구현한다.

::exercise{id="units-to-si"}

## 실패 사례

- **`x`, `x2`, `tmp`, `final_final` 같은 이름.** 쓴 사람도 일주일 뒤에는 무엇인지 모른다. `results_final_v2_real.csv` 같은 파일 이름은 버전 관리를 이름으로 대신하려는 신호다. 버전은 Git과 run ID가 맡는다.
- **단위가 섞인 변수.** 같은 `voltage`가 어떤 파일에서는 V, 어떤 파일에서는 mV다. 계산은 오류 없이 돌고 결과는 몇 자릿수 틀린다. 섭씨와 켈빈처럼 더하기로 바뀌는 단위는 배율 착오보다 더 늦게 발견된다.
- **경고를 일괄로 끄는 것.** 처음 lint를 적용했을 때 경고가 많다고 규칙 묶음 전체를 끄거나 파일마다 `# noqa`를 붙이면, 도구를 도입한 효과가 사라진다. 범위를 좁힌 예외만 둔다.
- **대량 포매팅과 기능 변경을 한 커밋에 섞는 것.** 리뷰가 불가능해지고, 나중에 버그가 들어온 커밋을 찾기도 어렵다.

## 통과 기준

- 프로젝트에 formatter·linter·pre-commit을 설정하고, CI에서 `ruff check .`와 `ruff format --check .`(또는 연구실이 정한 같은 역할의 도구)가 통과한다.
- 기존 코드의 물리량·배열·시계열 변수에 단위 접미사·shape 주석·시간 기준을 붙이고, 그 과정에서 발견한 문제를 표로 기록한다.
- 연습문제 `units-to-si`의 테스트를 통과한다.
