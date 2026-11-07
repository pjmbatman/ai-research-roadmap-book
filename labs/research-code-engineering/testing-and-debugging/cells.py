"""6.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/testing-and-debugging/cells.py
"""
# %% cell 1 — 테스트 함수와 부동소수 허용오차
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

# %% cell 2 — 결론을 바꾸는 실패: 분할 누수와 metric 집계
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

# %% cell 3 — 무작위 입력으로 불변식 확인 (property 기반 테스트 흉내)
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

# %% cell 4 — 이분 탐색으로 첫 실패 커밋 찾기 (git bisect의 원리)
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
