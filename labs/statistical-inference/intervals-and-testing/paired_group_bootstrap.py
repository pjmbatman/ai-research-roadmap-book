"""12.4 실습: 두 모델의 성능 차이에 대한 부트스트랩 신뢰구간을 셀 단위 그룹 부트스트랩과 비교한다.

상황: 배터리 셀 20개 × 셀마다 50 사이클 = 시험 샘플 1000개에서 두 SOH 예측 모델의 절대오차 [%]를 잰다.
- 셀마다 예측이 어려운 정도(base)가 달라 두 모델의 오차가 함께 오르내린다(짝지은 비교가 필요한 이유).
- 셀마다 두 모델의 차이 자체도 다르다(셀 단위로 재표본화해야 하는 이유).
알고 싶은 것은 "새 셀들에 대해" 평균 오차 차이 δ = E[A - B]이다. 이 시뮬레이션에서 참값은 -0.10 %p다.

세 가지 95% 부트스트랩 구간을 비교한다.
  (1) 짝 없이 샘플 단위:  A와 B의 오차를 각각 따로 재표본화
  (2) 짝지은 샘플 단위:   샘플 인덱스를 재표본화하고 같은 샘플의 A-B를 쓴다
  (3) 짝지은 셀 단위:     셀을 재표본화하고 그 셀의 모든 샘플을 함께 쓴다

실행: uv run --project labs python labs/statistical-inference/intervals-and-testing/paired_group_bootstrap.py
"""
import numpy as np

N_CELLS, N_CYCLES = 20, 50
DELTA_TRUE = -0.10        # %p, 새 셀들에 대한 평균 오차 차이 (A - B)
B = 1000                  # 부트스트랩 반복


def simulate(rng):
    base = rng.normal(2.0, 0.5, size=(N_CELLS, 1))                  # 셀의 난이도, (C, 1)
    gap = rng.normal(DELTA_TRUE, 0.3, size=(N_CELLS, 1))            # 셀마다 다른 A-B 차이, (C, 1)
    err_a = base + gap + rng.normal(0, 0.3, size=(N_CELLS, N_CYCLES))   # (C, K)
    err_b = base + rng.normal(0, 0.3, size=(N_CELLS, N_CYCLES))         # (C, K)
    return err_a, err_b


def intervals(err_a, err_b, rng):
    a, b = err_a.ravel(), err_b.ravel()                             # (N,)
    n = a.size
    i, j = rng.integers(0, n, size=(2, B, n))
    unpaired = a[i].mean(axis=1) - b[j].mean(axis=1)                # (1)
    d = a - b
    paired = d[i].mean(axis=1)                                      # (2)
    cell_d = (err_a - err_b).mean(axis=1)                           # 셀마다 평균 차이, (C,)
    c = rng.integers(0, N_CELLS, size=(B, N_CELLS))
    grouped = cell_d[c].mean(axis=1)                                # (3) 셀 수가 같으므로 셀 평균의 평균
    return {name: np.percentile(v, [2.5, 97.5]) for name, v in
            (("짝 없이, 샘플 단위", unpaired), ("짝지은, 샘플 단위", paired), ("짝지은, 셀 단위", grouped))}


def main():
    rng = np.random.default_rng(0)
    err_a, err_b = simulate(rng)
    print(f"시험 샘플 {err_a.size}개 (셀 {N_CELLS}개 × {N_CYCLES} 사이클)")
    print(f"평균 절대오차  A {err_a.mean():.3f} %   B {err_b.mean():.3f} %   차이 {err_a.mean() - err_b.mean():+.3f} %p")
    print(f"같은 샘플에서 A와 B 오차의 상관계수 {np.corrcoef(err_a.ravel(), err_b.ravel())[0, 1]:.2f}")
    print("\n95% 부트스트랩 구간 (데이터셋 하나)")
    for name, (lo, hi) in intervals(err_a, err_b, rng).items():
        print(f"  {name:12s}  [{lo:+.3f}, {hi:+.3f}]  폭 {hi - lo:.3f}")

    n_rep = 1000
    hits = {}
    for _ in range(n_rep):
        for name, (lo, hi) in intervals(*simulate(rng), rng).items():
            hits[name] = hits.get(name, 0) + (lo <= DELTA_TRUE <= hi)
    print(f"\n참값 δ = {DELTA_TRUE:+.2f} %p를 포함한 비율 (새 데이터셋 {n_rep}개, 목표 0.95)")
    for name, h in hits.items():
        print(f"  {name:12s}  {h / n_rep:.3f}")


if __name__ == "__main__":
    main()
