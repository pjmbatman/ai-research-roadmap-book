---
id: research-code-engineering/testing-and-debugging
title: "테스트와 디버깅"
title_en: "Testing and Debugging"
tier: core
status: draft
prerequisites: [research-code-engineering/logging-and-errors, git-and-agents/github-collaboration]
---

# 테스트와 디버깅

연구 코드에 테스트를 쓰지 않는 이유로 흔히 "결과를 보면 맞는지 안다"가 꼽힌다. 그런데 연구 코드의 가장 위험한 버그는 결과를 보아도 모르는 버그다. 같은 시료의 측정값이 train과 test에 나뉘어 들어가면 test 점수가 **좋아진다.** batch 평균의 평균으로 metric을 계산하면 숫자가 **조금** 달라진다. checkpoint에서 재개한 학습이 처음부터 이어서 한 학습과 다르면 곡선은 **그럴듯하게** 이어진다. 결과가 기대보다 나쁠 때는 사람이 코드를 의심하지만, 기대보다 좋을 때는 의심하지 않는다.

이 장은 세 가지를 다룬다. 무엇을 먼저 테스트할지 고르는 기준(틀리면 결론이 바뀌는 것), 테스트가 실패했을 때 원인을 찾는 도구와 절차(디버거, 가설과 이분 탐색), 그리고 그 검증을 사람이 잊지 않도록 자동으로 돌리는 CI다. 명령과 옵션은 집필 시점의 예시다. 쓰는 도구의 `--help`와 공식 문서로 확인한다.

## 학습 목표

- 연구 결론을 바꿀 수 있는 실패를 우선 테스트한다
- 디버거로 버그를 추적한다
- CI로 최소 검증을 자동화한다

## 테스트의 종류

테스트(test)는 코드가 기대대로 동작하는지 확인하는 코드다. 손으로 확인하는 것과 다른 점은 **다시 실행할 수 있다**는 것이다. 코드를 고칠 때마다, 의존성을 올릴 때마다, 동료가 PR을 올릴 때마다 같은 확인을 몇 초 만에 반복한다. 범위에 따라 세 가지로 나눈다.

| 종류 | 확인하는 것 | 연구 코드의 예 | 실행 시간 |
|---|---|---|---|
| unit test | 함수 하나의 동작 | metric 함수, 데이터 분할 함수, 단위 변환 | 밀리초 |
| integration test | 여러 부분의 연결 | 데이터 로더 → 모델 → loss가 한 batch에서 맞물리는가, checkpoint 저장·복원 | 초 |
| smoke test | 전체가 끝에서 끝까지 돌아가는가 | 합성 데이터로 학습 몇 step, 평가, 결과 파일 생성 | 수십 초~몇 분 |

버그를 고친 뒤에는 그 버그를 재현하는 테스트를 하나 남긴다. 이것을 회귀 테스트(regression test)라 한다. 같은 버그가 다시 들어오면 바로 실패한다.

**pytest 기초.** Python에서 가장 널리 쓰이는 테스트 도구는 pytest다. 규칙은 단순하다. `test_`로 시작하는 파일의 `test_`로 시작하는 함수를 찾아 실행하고, 함수 안의 `assert`가 실패하거나 예외가 나면 그 테스트를 실패로 보고한다. pytest는 `assert` 문을 다시 써서(assertion rewriting) 실패할 때 비교한 양쪽의 값을 보여 준다. 자주 쓰는 기능은 다음과 같다(집필 시점 예시).

| 기능 | 쓰임 |
|---|---|
| `@pytest.fixture` | 여러 테스트가 공유하는 준비물(작은 데이터셋, 모델)을 함수로 만든다. 테스트 함수의 인자 이름으로 받는다 |
| `tmp_path` | pytest가 제공하는 fixture. 테스트마다 새 임시 디렉터리(`pathlib.Path`)를 준다. checkpoint·결과 파일 테스트에 쓴다 |
| `@pytest.mark.parametrize` | 같은 테스트를 여러 입력으로 반복한다. 실패하면 어느 입력에서 실패했는지 보인다 |
| `pytest.approx` | 부동소수 값을 허용오차 안에서 비교한다 |
| `pytest.raises` | 특정 예외가 나야 통과하는 테스트(잘못된 입력에 실패하는가) |
| `-x`, `-k 이름`, `--lf` | 첫 실패에서 멈춤, 이름으로 골라 실행, 지난번 실패한 것만 다시 실행 |

pytest가 하는 일의 핵심은 "test 함수를 부르고, `AssertionError`가 나는지 본다"이다. 브라우저에서도 실행되도록 그 최소판을 직접 만들어, 부동소수 비교를 테스트한다.

```python {exec="browser" session="test"}
import numpy as np

def standardize(x):                                # x: (N, D) → (N, D)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)

z = standardize(np.random.default_rng(0).normal(5.0, 3.0, size=(1000, 4)))   # (1000, 4)

def test_exact_equality():
    assert z.mean(axis=0).tolist() == [0.0, 0.0, 0.0, 0.0]

def test_allclose_default_tolerance():
    np.testing.assert_allclose(z.mean(axis=0), 0.0)        # rtol=1e-7, atol=0

def test_allclose_with_atol():
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-12)

def run(tests):                                    # pytest가 하는 일의 최소판
    for t in tests:
        try:
            t()
            print(f"PASSED {t.__name__}")
        except AssertionError as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else "assert 실패"
            print(f"FAILED {t.__name__}: {first}")

print("열 평균이 정확히 0:", bool(np.all(z.mean(axis=0) == 0)), "/ |열 평균| < 1e-12:", bool(np.all(np.abs(z.mean(axis=0)) < 1e-12)))
run([test_exact_equality, test_allclose_default_tolerance, test_allclose_with_atol])
```

```output
열 평균이 정확히 0: False / |열 평균| < 1e-12: True
FAILED test_exact_equality: assert 실패
FAILED test_allclose_default_tolerance: Not equal to tolerance rtol=1e-07, atol=0
PASSED test_allclose_with_atol
```

구현은 올바른데 앞의 두 테스트가 실패했다. 열 평균은 $10^{-16}$–$10^{-15}$ 수준의 반올림 오차를 가진다. `==` 비교는 이 오차를 허용하지 않는다. `np.testing.assert_allclose(actual, desired)`는

$$
\lvert \text{actual} - \text{desired} \rvert \le \text{atol} + \text{rtol} \cdot \lvert \text{desired} \rvert
$$

를 원소마다 검사하고, 기본값은 `rtol=1e-7`, `atol=0`이다. 기댓값이 0이면 오른쪽의 $\text{rtol} \cdot 0$이 0이 되어 허용오차가 정확히 0이 된다. **0과 비교할 때는 `atol`이 필요하다.**

허용오차는 근거를 가지고 정한다. 출발점은 자료형의 기계 정밀도(machine epsilon)다. `np.finfo(np.float64).eps`는 약 $2.2 \times 10^{-16}$, `np.finfo(np.float32).eps`는 약 $1.2 \times 10^{-7}$이다. 원소 $N$개를 더하는 계산의 오차는 대략 eps의 몇 배에서 $N$배 사이로 커질 수 있으므로, float64로 1,000개를 평균 낸 위 예에서 `atol=1e-12`는 넉넉한 값이다. GPU에서 float32나 혼합 정밀도로 계산한 결과를 CPU float64 결과와 비교한다면 `rtol=1e-4` 정도가 필요할 수 있다. 허용오차를 "통과할 때까지" 늘리지 않는다. 그러면 진짜 버그도 통과한다.

::quiz{id="allclose-zero-atol"}

## 연구 코드에서 테스트할 것

모든 것을 테스트할 시간은 없다. 우선순위의 기준은 하나다. **이 부분이 틀리면 결론이 바뀌는가, 그리고 틀려도 눈에 띄지 않는가.** 그림의 색은 틀리면 바로 보이고 결론을 바꾸지 않는다. 데이터 분할은 틀려도 보이지 않고 결론을 바꾼다. 연구 코드에서 이 기준을 통과하는 대표적인 불변조건(invariant)은 다음과 같다.

| 대상 | 불변조건 | 틀렸을 때 |
|---|---|---|
| shape | 각 단계의 출력 shape이 명세와 같다. 특히 broadcasting이 의도하지 않은 차원을 만들지 않는다 | `(B,)`와 `(B, 1)`의 차가 `(B, B)`가 되어 loss가 조용히 틀린다 |
| 단위 | 변환 전후의 물리량이 같다(왕복 변환이 항등). 입력 범위가 물리적으로 가능하다 | 섭씨와 켈빈, Å과 nm가 섞인다 |
| 데이터 분할 | train·val·test가 겹치지 않고, 빠진 행이 없고, 같은 그룹(시료·환자·분자)이 한쪽에만 있다. 시계열이면 test가 train보다 뒤다 | test 점수가 과대평가된다 |
| metric | batch 크기·순서와 무관하다. 손으로 계산한 작은 예와 같다 | 모델 비교가 뒤집힌다 |
| checkpoint | 저장·복원 후 이어서 학습한 결과가 중단 없이 학습한 결과와 같다 | 긴 실험의 후반부를 믿을 수 없다 |

분할 누수(leakage)와 metric 집계 오류를 직접 본다. 시료 20개를 5번씩 측정한 데이터(100행)를 행 단위로 무작위로 나누면 어떻게 되는지, batch 평균의 평균이 전체 평균과 언제 달라지는지 확인한다.

```python {exec="browser" session="test"}
groups = np.repeat(np.arange(20), 5)               # (100,) 시료 20개 × 반복 측정 5회

def split_rows(n, seed):                           # 행 단위 무작위 분할 (80/20)
    idx = np.random.default_rng(seed).permutation(n)
    return idx[20:], idx[:20]

def split_groups(seed):                            # 시료 단위 분할 (16/4 시료)
    test_g = np.random.default_rng(seed).choice(20, 4, replace=False)
    is_test = np.isin(groups, test_g)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)

def leaked(split):
    tr, te = split
    return len(np.intersect1d(groups[tr], groups[te]))

print(f"행 단위 분할: train·test 공통 시료 {leaked(split_rows(100, seed=0))}개")
print(f"시료 단위 분할: train·test 공통 시료 {leaked(split_groups(seed=0))}개")

err = np.abs(np.random.default_rng(1).normal(size=100))      # (100,) 표본별 절대오차
def mae_mean_of_batches(err, bs):                  # 흔한 버그: batch 평균의 평균
    return np.mean([err[s:s + bs].mean() for s in range(0, len(err), bs)])
for bs in [10, 32, 64]:
    print(f"batch {bs:2d}: batch 평균의 평균 {mae_mean_of_batches(err, bs):.4f}  전체 평균 {err.mean():.4f}")
```

```output
행 단위 분할: train·test 공통 시료 11개
시료 단위 분할: train·test 공통 시료 0개
batch 10: batch 평균의 평균 0.6567  전체 평균 0.6567
batch 32: batch 평균의 평균 0.6186  전체 평균 0.6567
batch 64: batch 평균의 평균 0.6578  전체 평균 0.6567
```

행 단위로 나누면 test에 들어간 시료 대부분(이 시드에서는 11개)이 train에도 들어 있다. 모델은 시료의 개별 특성을 외워서 test를 맞힐 수 있으므로, test 점수는 새 시료에 대한 성능을 과대평가한다. metric 쪽은 더 교묘하다. 100이 10으로 나누어떨어지는 batch 10에서는 모든 batch의 크기가 같아 두 계산이 일치한다. batch 32에서는 마지막 batch가 4개뿐인데 다른 batch와 같은 가중치를 받아 값이 달라진다. **버그가 특정 설정에서만 드러나므로**, 테스트는 나누어떨어지지 않는 batch 크기를 일부러 포함해야 한다.

이 두 가지를 테스트로 쓰면 다음과 같다. 실습의 `test_pipeline.py`에 있는 테스트다.

```python
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_has_no_group_leakage(groups, seed):
    tr, te = pipeline.split_by_group(groups, 0.2, seed)
    shared = np.intersect1d(groups[tr], groups[te])
    assert len(shared) == 0, f"같은 시료 {len(shared)}개가 train과 test에 나뉘어 있다: {shared[:5]}"


@pytest.mark.parametrize("batch_size", [1, 7, 32, 100, 1000])
def test_mae_does_not_depend_on_batch_size(batch_size):
    rng = np.random.default_rng(1)
    y, y_hat = rng.normal(size=100), rng.normal(size=100)          # (100,), (100,)
    expected = np.mean(np.abs(y - y_hat))                           # 독립적인 기준: 한 번에 계산
    np.testing.assert_allclose(pipeline.mae(y, y_hat, batch_size), expected, rtol=1e-12)
```

**기댓값은 구현과 독립적이어야 한다.** 테스트 안에서 구현과 같은 식을 다시 써서 비교하면, 식 자체가 틀렸을 때 구현과 테스트가 함께 틀려 통과한다. 독립적인 기댓값을 얻는 방법은 세 가지다. 손으로 계산할 수 있는 작은 예(오차 $[0, 0, 0]$과 $[1, -2, 3]$의 MAE는 2), 알려진 성질(오차가 모두 0이면 0, batch 크기와 무관), 그리고 느리지만 다른 방법의 구현(위의 "한 번에 계산"처럼)이다.

::quiz{id="test-duplicates-implementation"}

**작은 데이터 과적합 테스트.** 학습 코드 전체를 확인하는 가장 싼 방법은, 아주 작은 데이터(몇 개~수십 개 표본)에 모델이 거의 완벽하게 맞출 수 있는지 보는 것이다. 표현력이 충분한 모델이 표본 몇 개도 외우지 못하면, 모델 크기나 학습률 문제가 아니라 loss·gradient·데이터 연결 어딘가의 버그일 가능성이 높다. label이 섞였거나, `optimizer.zero_grad()`를 빠뜨렸거나, loss의 부호가 반대인 경우가 여기서 잡힌다. 실습의 `test_overfits_tiny_problem`이 이 테스트다.

**무작위 입력으로 불변식 확인하기.** 손으로 고른 몇 개의 입력은 작성자가 생각한 경우만 확인한다. property 기반 테스트(property-based testing)는 입력을 무작위로 많이 만들고, 출력이 만족해야 할 **성질**을 확인한다. Python에는 이를 위한 Hypothesis 라이브러리가 있다. 여기서는 원리만 손으로 흉내 낸다. 대상은 $\log\sum_i e^{x_i}$를 계산하는 logsumexp다. 이 함수는 두 성질을 만족해야 한다.

$$
\max_i x_i \le \operatorname{lse}(\mathbf{x}) \le \max_i x_i + \log n, \qquad \operatorname{lse}(\mathbf{x} + c) = \operatorname{lse}(\mathbf{x}) + c.
$$

$n$은 원소 수, $c$는 모든 원소에 더한 상수다. 그대로 계산하는 구현과, 최댓값 $m$을 빼고 계산한 뒤 다시 더하는 구현 $m + \log\sum_i e^{x_i - m}$을 비교한다.

```python {exec="browser" session="test"}
def logsumexp_naive(x):
    return np.log(np.sum(np.exp(x)))

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def check_properties(f, n_trials=400, seed=0):
    rng = np.random.default_rng(seed)
    for trial in range(n_trials):
        scale = 10.0 ** (trial // 100)             # 입력 크기를 1, 10, 100, 1000으로 키운다
        x = rng.normal(0.0, scale, size=rng.integers(1, 10))    # (n,)
        c = rng.normal(0.0, scale)
        with np.errstate(over="ignore"):
            lse, lse_shift = f(x), f(x + c)
        props = {"유한": np.isfinite(lse),
                 "max ≤ lse ≤ max+log n": x.max() - 1e-9 <= lse <= x.max() + np.log(len(x)) + 1e-9,
                 "f(x+c) = f(x)+c": np.isclose(lse_shift, lse + c, rtol=1e-12, atol=1e-9)}
        failed = [k for k, ok in props.items() if not ok]
        if failed:
            with np.errstate(over="ignore"):       # 반례 축소: 가장 큰 원소 하나만 남겨도 실패하는가
                small = f(x[[np.argmax(x)]])
            return f"trial {trial}에서 실패 {failed}\n        축소한 반례: f([{x.max():.1f}]) = {small}"
    return f"{n_trials}회 모두 통과"

print("naive :", check_properties(logsumexp_naive))
print("stable:", check_properties(logsumexp))
```

```output
naive : trial 300에서 실패 ['유한', 'max ≤ lse ≤ max+log n']
        축소한 반례: f([1532.1]) = inf
stable: 400회 모두 통과
```

그대로 계산한 구현은 입력 크기가 1–100인 처음 300번의 시도를 모두 통과하고, 크기가 1000이 되자마자 실패한다. float64에서 $e^{x}$는 $x$가 약 709를 넘으면 overflow로 무한대가 된다. 작은 입력만 손으로 골라 테스트했다면 이 버그는 남았을 것이다. 실패한 입력을 더 작은 반례로 줄이는 단계(shrinking)도 중요하다. 원소 9개짜리 무작위 배열보다 `[1532.1]` 하나가 원인을 훨씬 분명히 보여 준다. Hypothesis 같은 도구는 입력 생성과 반례 축소를 자동으로 한다.

::quiz{id="what-to-test-first"}

:::note[분야 예시]
단백질 구조나 기능을 예측하는 연구에서는 서열이 거의 같은 단백질(같은 family의 상동 단백질)이 train과 test에 나뉘어 들어가면 test 성능이 크게 과대평가될 수 있다. 그래서 서열 유사도로 군집을 만든 뒤 군집 단위로 나누는 관행이 있다. 기후·지구과학 데이터에서는 인접한 날짜나 인접한 격자점이 강하게 상관되어 있어, 무작위로 나누면 test 날짜의 바로 전날이 train에 들어간다. 이때는 시간 블록이나 공간 블록 단위로 나눈다. 어느 경우든 "그룹 ID가 양쪽에 동시에 나타나지 않는다"는 같은 형태의 테스트로 지킬 수 있다. 그룹을 무엇으로 정할지는 분야 지식이 정한다.
:::

## 디버깅

테스트가 실패했다. 이제 원인을 찾을 차례다.

**stack trace 읽기.** 예외가 나면 Python은 traceback을 출력한다. shape이 맞지 않는 행렬 곱의 예다(파일 경로는 줄였다).

```text
Traceback (most recent call last):
  File "train.py", line 18, in <module>
    train_step(w)
    ~~~~~~~~~~^^^
  File "train.py", line 14, in train_step
    return model(x, w)
  File "train.py", line 9, in model
    return x @ w                        # (B, D) @ (D, 1)
           ~~^~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 12 is different from 10)
```

- **아래에서 위로 읽는다.** 마지막 줄이 예외의 종류와 메시지다. 그 바로 위가 예외가 실제로 난 줄, 위로 갈수록 그 줄을 부른 쪽이다(most recent call last).
- **메시지를 끝까지 읽는다.** 여기서는 "size 12 is different from 10", 즉 `w`의 첫 차원(12)이 `x`의 특성 수(10)와 다르다는 것이다. 예외가 난 곳(`model`)이 아니라 `w`를 만든 곳(설정의 `in_dim = 12`)이 원인이다. 예외가 난 줄과 버그가 있는 줄은 다를 수 있다.
- **라이브러리 내부 frame은 건너뛰고 자기 코드의 마지막 frame을 찾는다.** PyTorch 같은 라이브러리의 traceback은 수십 줄의 내부 frame을 포함한다. 자기 프로젝트 파일이 나오는 가장 아래 frame이 출발점이다.
- **예외 체인을 구분한다.** "The above exception was the direct cause of the following exception"은 `raise ... from err`로 연결된 원인이고(6.5장), "During handling of the above exception, another exception occurred"는 예외를 처리하던 `except` 블록 안에서 또 다른 예외가 났다는 뜻이다. 처음 예외가 원인인 경우가 많다.

**pdb와 `breakpoint()`.** traceback은 어디서 실패했는지 알려 주지만 그때 변수의 값은 알려 주지 않는다. 값을 보려고 `print`를 넣으면, 볼 변수를 미리 정해야 하고 가설이 바뀔 때마다 코드를 고쳐 다시 실행해야 한다. 디버거(debugger)는 실행을 멈춘 상태에서 아무 변수나 보고, 한 줄씩 진행하고, 호출 스택을 오르내리게 해 준다. Python 표준 라이브러리의 디버거가 `pdb`다. 들어가는 방법은 세 가지다.

- 코드에 `breakpoint()`를 넣으면 그 줄에서 멈춘다. 환경변수 `PYTHONBREAKPOINT=0`으로 실행하면 모든 `breakpoint()`가 무시된다.
- `python -m pdb script.py`로 실행하면 처음에 멈추고, `c`로 계속 실행하다가 처리되지 않은 예외가 나면 그 자리에서 멈춘다(post-mortem). Jupyter에서는 예외가 난 직후 `%debug`가 같은 일을 한다.
- `pytest --pdb`는 실패하거나 예외가 난 테스트의 그 지점에서 디버거를 연다.

| 명령 | 뜻 |
|---|---|
| `p 식`, `pp 식` | 식의 값을 출력(`pp`는 보기 좋게) |
| `l`, `ll` | 현재 위치 주변 코드, 현재 함수 전체 |
| `n` | 다음 줄(함수 호출은 건너뜀) |
| `s` | 다음 줄(함수 안으로 들어감) |
| `c` | 다음 중단점까지 계속 |
| `w` | 호출 스택 출력 |
| `u`, `d` | 호출 스택을 한 단계 위(호출한 쪽)·아래로 이동 |
| `b 파일:줄` | 중단점 설정 |
| `q` | 종료 |

위 traceback이라면 `pytest --pdb`나 post-mortem으로 `model`에서 멈춘 뒤 `p x.shape, w.shape`로 shape을 보고, `u`로 `train_step`에 올라가 `x`가 어디서 왔는지 확인한다.

::quiz{id="pdb-on-failure"}

**IDE debugger와 원격 디버깅.** VS Code·PyCharm 같은 IDE의 debugger는 같은 일을 화면에서 한다. 줄 번호 옆을 눌러 중단점을 걸고, 변수 창에서 배열을 펼쳐 본다. 조건부 중단점("`loss`가 NaN일 때만 멈춤")은 수천 step 뒤에 생기는 문제에 유용하다. 학습은 대개 원격 GPU 서버에서 돌기 때문에 원격 디버깅이 필요하다. 두 가지 방법이 흔하다(집필 시점 예시).

- 편집기를 원격 서버에 붙인다(VS Code의 Remote-SSH 등). 편집기의 debugger가 서버에서 실행된다.
- 서버에서 디버깅 서버를 띄우고 로컬 편집기가 접속한다. 예: `python -m debugpy --listen 5678 --wait-for-client train.py`로 실행하고 SSH 포트 포워딩(2.3장)으로 5678 포트를 연결한다.

터미널만 있으면 `pdb`는 어디서든 동작한다는 점도 기억해 둔다. 몇 가지 환경 특유의 함정이 있다. 여러 프로세스로 데이터를 읽는 DataLoader(`num_workers > 0`)의 worker 안에서는 `breakpoint()`가 기대대로 동작하지 않으므로, 디버깅할 때는 `num_workers=0`으로 같은 프로세스에서 읽게 한다. CUDA 연산은 비동기라 오류가 엉뚱한 줄에서 보고될 수 있으므로 `CUDA_LAUNCH_BLOCKING=1`로 다시 실행한다(3.6장).

## 체계적 디버깅

디버깅이 오래 걸리는 이유는 대개 도구가 아니라 절차다. 의심 가는 곳을 바꿔 보고, 안 되면 다른 곳을 바꿔 보는 식으로 진행하면, 우연히 증상이 사라져도 원인을 모른다. 체계적 디버깅은 실험과 같은 구조를 따른다.

1. **관찰을 고정한다.** 실패를 매번 재현하는 명령 하나(가능하면 테스트 하나)를 만든다. 재현되지 않는 실패는 고쳤는지도 확인할 수 없다.
2. **가설을 세운다.** "batch 크기 32에서만 metric이 다르다 → 마지막 batch의 가중치 문제다"처럼, 틀렸을 때 확인할 수 있는 형태로 쓴다.
3. **예측하고 시험한다.** 가설이 맞다면 무엇이 관찰되어야 하는가(batch 크기 25에서는 일치해야 한다). 시험해서 가설을 버리거나 남긴다. 가설과 결과를 메모로 남기면 같은 시험을 반복하지 않는다.
4. **범위를 줄인다.** 최소 재현(3.6장)으로 코드와 데이터를 줄이거나, 아래의 이분 탐색으로 후보를 반씩 줄인다.
5. **고친 뒤 회귀 테스트를 남긴다.** 1의 재현 명령이 테스트가 되어 다시 들어오는 버그를 막는다.

**이분 탐색과 `git bisect`.** "지난달에는 이 결과가 나왔는데 지금은 다르다"는 문제는 이분 탐색으로 푼다. 통과하던 커밋과 실패하는 커밋 사이에서 가운데 커밋을 시험하고, 통과하면 뒤쪽 절반, 실패하면 앞쪽 절반만 남긴다. 후보가 $N$개면 약 $\log_2 N$번의 시험으로 첫 실패 커밋을 찾는다. 원리는 몇 줄이다.

```python {exec="browser" session="test"}
history = [f"c{i:02d}" for i in range(64)]        # 오래된 것부터. c00은 통과, c63은 실패
FIRST_BAD = 41                                     # 우리가 모르는 값
tested = []

def is_bad(i):                                     # git bisect run에 넘기는 테스트 스크립트에 해당
    tested.append(history[i])
    return i >= FIRST_BAD

def first_bad(n, is_bad):
    good, bad = 0, n - 1                           # 불변식: good은 통과, bad는 실패
    while bad - good > 1:
        mid = (good + bad) // 2
        if is_bad(mid):
            bad = mid
        else:
            good = mid
    return bad

k = first_bad(len(history), is_bad)
print("첫 실패 커밋:", history[k])
print(f"시험한 커밋 {len(tested)}개: {tested}")
```

```output
첫 실패 커밋: c41
시험한 커밋 6개: ['c31', 'c47', 'c39', 'c43', 'c41', 'c40']
```

커밋 64개 사이에서 6번($\log_2 64 = 6$)의 시험으로 첫 실패 커밋 `c41`을 찾았다. 이분 탐색이 동작하려면 두 가지가 필요하다. 하나는 "어느 시점 이전은 모두 통과, 이후는 모두 실패"라는 단조성이고, 다른 하나는 매 시험의 판정이 정확하다는 것이다. 무작위로 실패하는 테스트로 bisect를 하면 한 번의 잘못된 판정이 탐색을 엉뚱한 곳으로 보낸다.

`git bisect`는 이 과정을 커밋 그래프 위에서 해 준다. 판정을 스크립트에 맡기면 자동으로 진행된다(집필 시점 예시).

```bash
git bisect start
git bisect bad HEAD                    # 지금은 실패
git bisect good v0.3                   # 이 태그(또는 커밋)에서는 통과
git bisect run uv run pytest -x tests/test_metrics.py   # 종료 코드 0이면 good, 1–127(125 제외)이면 bad
git bisect reset                       # 끝나면 원래 브랜치로 돌아간다
```

판정 스크립트의 종료 코드가 0이면 good, 125면 "이 커밋은 시험할 수 없음"(빌드가 깨진 커밋 등, skip), 그 밖의 1–127이면 bad로 처리된다. 시험에 쓰는 테스트 파일이 과거 커밋에 없을 수 있으므로, 판정 스크립트를 작업 트리 밖에 복사해 두고 실행하는 방법도 쓴다. 의존성이 커밋마다 다르면 판정 스크립트 안에서 그 커밋의 lockfile로 환경을 맞춘다(3.6장).

::quiz{id="bisect-steps"}

:::note[분야 예시]
분자동역학이나 궤도 계산처럼 에너지가 보존되어야 하는 시뮬레이션에서는 "총에너지의 상대 변화가 정해진 한도 안에 있다"가 좋은 불변조건이다. 몇 달 동안 코드가 바뀐 뒤 에너지 drift가 커졌다면, 짧은 시뮬레이션을 돌려 drift를 한도와 비교하는 스크립트를 판정 기준으로 `git bisect run`을 돌린다. 적분기, 단위 변환, 경계 조건 처리 중 어느 커밋이 원인인지 수십 개의 커밋 사이에서도 몇 번의 실행으로 좁힐 수 있다.
:::

## CI

테스트는 실행되어야 의미가 있다. 사람은 바쁠 때 테스트를 건너뛴다. CI(continuous integration)는 push나 pull request마다 정해진 검증을 서버에서 자동으로 실행하고, 결과를 PR 화면에 표시한다(4.2장). 실패한 PR은 merge 전에 드러난다. GitHub를 쓴다면 GitHub Actions가 가장 가까운 선택지다. 저장소의 `.github/workflows/` 아래에 YAML 파일을 두면 된다. 연구 프로젝트의 최소 workflow는 설치·lint·테스트 세 단계다(집필 시점 예시. action의 버전은 각 action의 저장소에서 확인한다).

```yaml
# .github/workflows/test.yml
name: test
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install (lockfile 그대로)
        run: uv sync --locked
      - name: Lint
        run: uv run ruff check .
      - name: Unit and smoke tests
        run: uv run pytest -q -m "not slow"
```

CI 검증을 설계할 때의 원칙은 다음과 같다.

- **빠르게.** PR마다 도는 검증은 몇 분 안에 끝나야 한다. 오래 걸리면 사람들이 결과를 기다리지 않는다. 오래 걸리는 테스트는 `@pytest.mark.slow`처럼 marker를 붙여(사용자 정의 marker는 `pyproject.toml`에 등록한다) PR에서는 빼고, 매일 밤이나 merge 후에 따로 돌린다.
- **GPU 없이.** 기본 hosted runner에는 GPU가 없다. 테스트는 CPU에서 작은 합성 데이터로 돌도록 설계한다. GPU가 꼭 필요한 검증은 연구실 서버에 self-hosted runner를 두거나 수동 절차로 남긴다.
- **결정적으로.** 시드를 고정하고, 허용오차는 근거를 가지고 정한다. 가끔 실패하는 테스트(flaky test)는 CI의 신뢰를 무너뜨린다. "다시 돌리면 통과"가 습관이 되면 진짜 회귀도 그렇게 넘어간다.
- **lockfile로 설치.** CI가 매번 최신 버전을 설치하면, 코드가 바뀌지 않았는데 CI가 깨지는 날이 온다. 반대로 lockfile을 일부러 갱신하는 PR에서는 그 영향이 CI에서 드러난다(3.4장).
- **테스트 자체를 테스트한다.** 테스트가 아무것도 확인하지 않는데 통과만 하고 있을 수도 있다. 이 책의 저장소 CI(`.github/workflows/check.yml`)는 연습문제의 해답이 테스트를 통과하는지와 함께, **시작 코드가 테스트를 통과하지 못하는지**도 확인한다. 시작 코드가 통과한다면 그 테스트는 아무것도 확인하지 않는 것이다. 같은 생각을 연구 코드에 적용한 것이 실습의 "일부러 넣은 버그를 테스트가 잡는가" 확인이다.

::quiz{id="flaky-test-in-ci"}

## 실습

데이터 분할 누수, metric 집계 오류, checkpoint 복원 실패를 잡는 테스트를 작성하고, 각 테스트가 일부러 넣은 버그를 실제로 잡는지 확인한다. 코드는 `labs/research-code-engineering/testing-and-debugging/`에 있다.

- `pipeline.py`: 시료 단위 분할(`split_by_group`), batch 단위로 누적하는 MAE(`mae`), momentum SGD와 checkpoint 저장·복원(`Trainer`)이 있는 작은 파이프라인. 환경변수 `INJECT_BUG`에 `leak`, `metric`, `checkpoint`를 주면 각각의 버그가 들어간다.
- `test_pipeline.py`: 분할(분할이 서로소이고 빠짐없는가, 시료 누수가 없는가, 시드가 같으면 같은가), metric(batch 크기와 무관한가, 손 계산과 같은가), checkpoint(10 step + 저장·복원 + 10 step = 20 step인가), 작은 문제 과적합 smoke test. fixture, `tmp_path`, `parametrize`, `pytest.approx`를 쓴다.

버그 없이 실행하면 모두 통과하고, 버그를 하나씩 넣으면 그 버그를 지키는 테스트만 실패한다.

```bash
cd labs
uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
INJECT_BUG=leak uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
```

집필 환경에서 네 가지 경우를 실행한 결과를 정리하면 다음과 같다.

| `INJECT_BUG` | 결과 | 실패한 테스트 |
|---|---|---|
| (없음) | 15 passed | — |
| `leak` | 3 failed, 12 passed | `test_split_has_no_group_leakage[0]`, `[1]`, `[2]` |
| `metric` | 3 failed, 12 passed | `test_mae_does_not_depend_on_batch_size[7]`, `[32]`, `test_mae_known_value` |
| `checkpoint` | 1 failed, 14 passed | `test_resume_from_checkpoint_is_identical` |

metric 버그는 batch 크기 1, 100, 1000에서는 드러나지 않는다. batch가 모두 같은 크기이기 때문이다. parametrize에 나누어떨어지지 않는 크기(7, 32)를 넣은 것이 이 버그를 잡았다. checkpoint 버그의 실패 메시지는 다음과 같다(`--tb=short`, 일부).

```output
E   AssertionError: 
E   Not equal to tolerance rtol=1e-12, atol=1e-12
E   중간에 저장·복원한 학습이 이어서 학습한 결과와 다르다
E   Mismatched elements: 3 / 3 (100%)
...
E    ACTUAL: array([ 1.020181, -1.632355,  0.222266])
E    DESIRED: array([ 1.397266, -2.080566,  0.119609])
```

momentum 상태를 저장하지 않으면 재개한 학습은 속도 0에서 다시 출발한다. 파라미터만 저장하는 checkpoint는 흔한 실수이고, loss 곡선만으로는 거의 보이지 않는다.

할 일은 다음과 같다.

1. 테스트를 하나씩 지워 보고, 그때 어떤 `INJECT_BUG`가 통과하게 되는지 확인한다. 각 테스트가 무엇을 지키는지 한 줄로 쓴다.
2. 자신의 프로젝트에서 같은 세 가지(분할 누수, metric 집계, checkpoint 복원)를 테스트한다. 버그를 일부러 넣어 테스트가 잡는지 확인한 뒤 되돌린다.
3. 실패하는 테스트 하나를 `pytest --pdb`로 실행해, 실패 지점에서 `p`, `u`, `d`로 변수를 확인한다.
4. 앞의 workflow를 참고해 CI를 연결하고, 일부러 버그를 넣은 PR에서 CI가 실패하는 것을 확인한다.

이어서 연습문제를 푼다. 이번에는 함수가 아니라 **테스트**를 쓴다. 채점기가 올바른 구현 두 개와 버그가 있는 구현 여러 개를 넣어, 테스트가 올바른 구현은 통과시키고 버그는 모두 잡는지 확인한다.

::exercise{id="standardize-check"}

## 실패 사례

- **테스트하기 쉬운 것만 테스트하는 것.** 설정 객체의 필드, 유틸리티 함수의 문자열 포맷처럼 쓰기 쉬운 테스트만 쌓이고, 데이터 분할·metric·checkpoint처럼 결론을 바꾸는 부분은 "어려워서" 비어 있다. 커버리지 숫자는 높지만 중요한 버그는 지나간다. 우선순위는 "틀리면 결론이 바뀌는가"로 정한다.
- **무작위성 때문에 불안정한 테스트.** 시드를 고정하지 않았거나, 허용오차를 근거 없이 빡빡하게 잡았거나, 테스트 실행 순서에 따라 전역 상태가 달라지는 테스트는 가끔 실패한다. 이런 테스트는 실패해도 아무도 믿지 않아 진짜 회귀를 가린다. 시드를 고정하고, 허용오차를 수치 정밀도나 여러 시드에서의 변동에 근거해 정한다.
- **부동소수를 `==`로 비교하는 테스트.** 올바른 구현도 반올림 오차 때문에 실패하고, 계산 순서만 바꾼 리팩터링에도 깨진다. `np.testing.assert_allclose`나 `pytest.approx`로 비교하고, 0과 비교할 때는 `atol`을 준다.
- **테스트가 구현을 복제하는 것.** 기댓값을 구현과 같은 식으로 계산하면 식의 오류를 잡지 못한다. 손 계산, 알려진 성질, 다른 방법의 구현에서 기댓값을 얻는다.
- **`print`만으로 디버깅하는 것.** 가설이 바뀔 때마다 코드를 고치고 다시 실행하게 된다. 몇 시간짜리 학습이면 가설 하나에 몇 시간이 든다. 실패를 작은 테스트로 재현한 뒤 디버거로 들어간다.
- **증상만 고치는 것.** NaN이 나는 곳에 `np.nan_to_num`을 넣으면 테스트는 통과하지만, NaN을 만든 원인(0으로 나누기, 발산)은 그대로다. 원인을 찾은 뒤 고치고, 원인을 재현하는 회귀 테스트를 남긴다.

## 통과 기준

- 일부러 넣은 세 가지 버그(분할 누수, metric 집계 오류, checkpoint 복원 실패)를 자신의 테스트가 모두 잡아낸다.
- 프로젝트의 CI가 push와 PR마다 설치·lint·테스트를 실행하고, 버그를 넣은 PR에서 실패한다.
- 연습문제 `standardize-check`의 테스트를 통과한다.

## 참고 문헌

- pytest 공식 문서 (docs.pytest.org). — fixture, `parametrize`, `tmp_path`, `--pdb`, marker
- NumPy 공식 문서, `numpy.testing.assert_allclose`. — 허용오차 비교의 정의(`rtol`, `atol`)
- Python 공식 문서, *pdb — The Python Debugger* (docs.python.org/3/library/pdb.html). — 디버거 명령, `breakpoint()`, post-mortem
- Git 공식 문서, *git-bisect* (git-scm.com/docs/git-bisect). — `bisect run`과 종료 코드 규칙(125 = skip)
