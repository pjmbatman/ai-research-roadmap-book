"""22.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs --group torch python labs/experiment-operations/reproducibility/cells.py
cell 3은 PyTorch가 필요하다(exec="python"). 출력은 이 책의 실습 환경(CPU)에서 얻은 것이다.
"""
# %% cell 1 — 전역 RNG 하나 대 용도별 RNG: 코드 한 줄을 추가했을 때
import numpy as np

def setup(add_augmentation, per_purpose):
    """가중치 초기화와 데이터 순서를 만든다. add_augmentation이면 그 사이에 증강 난수를 쓴다."""
    if per_purpose:
        init_ss, data_ss, aug_ss = np.random.SeedSequence(0).spawn(3)
        w = np.random.default_rng(init_ss).normal(size=3)
        if add_augmentation:
            np.random.default_rng(aug_ss).uniform(size=5)
        order = np.random.default_rng(data_ss).permutation(6)
    else:
        np.random.seed(0)                              # 전역 상태 하나를 모두가 나눠 쓴다
        w = np.random.normal(size=3)
        if add_augmentation:
            np.random.uniform(size=5)                  # 새로 넣은 증강이 난수를 먼저 소비한다
        order = np.random.permutation(6)
    return w, order

for per_purpose in [False, True]:
    (w0, o0), (w1, o1) = setup(False, per_purpose), setup(True, per_purpose)
    name = "용도별 RNG" if per_purpose else "전역 RNG  "
    print(f"{name}: 초기화 같음 {np.array_equal(w0, w1)}, 데이터 순서 같음 {np.array_equal(o0, o1)}"
          f"  {o0} → {o1}")

# %% cell 2 — 부동소수점 덧셈의 순서: 같은 수를 다르게 나눠 더하면
import math

rng = np.random.default_rng(0)
x = (rng.normal(size=100_000) * 1000).astype(np.float32)        # (100000,)

def chunked_sum(x, n_chunks):
    """n개 스레드가 조각을 나눠 더한 뒤 합치는 것을 흉내 낸다(각 조각은 앞에서부터 차례로 더한다)"""
    parts = [np.cumsum(c)[-1] for c in np.array_split(x, n_chunks)]
    return np.cumsum(np.array(parts, dtype=np.float32))[-1]

exact = math.fsum(x.astype(np.float64))
for n in [1, 2, 4, 8]:
    s = chunked_sum(x, n)
    print(f"조각 {n}개: {float(s):.4f}   정확한 합과의 차이 {float(s) - exact:+.4f}")
print("같은 조각 수로 두 번 더하면 같은가?", chunked_sum(x, 4) == chunked_sum(x, 4))

# %% cell 3 — PyTorch: 같은 seed, 다른 스레드 수
import torch

def train(seed, threads, steps=100):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    X = torch.randn(4096, 16)                                     # (N, D)
    Y = torch.sin(X @ torch.randn(16, 16))                        # (N, D)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 16))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])   # (P,)

a, b = train(seed=0, threads=1), train(seed=0, threads=1)
c, d = train(seed=0, threads=2), train(seed=1, threads=1)
print("같은 seed·같은 스레드 수, 두 번: 비트 단위로 같음", torch.equal(a, b))
print("같은 seed·스레드 1개 대 2개:    비트 단위로 같음", torch.equal(a, c),
      f"  최대 차이 {(a - c).abs().max().item():.1e}")
print(f"seed 0 대 seed 1:              최대 차이 {(a - d).abs().max().item():.1e}")

# %% cell 4 — 재현 판정: 다른 기계의 결과가 보고된 변동 범위 안인가
from scipy import stats

reported = np.array([0.912, 0.905, 0.918, 0.909, 0.914])         # 논문: seed 5개의 정확도
reproduced = {"기계 B": np.array([0.910, 0.907, 0.913]),           # 다른 기계에서 seed 3개
              "기계 C": np.array([0.893, 0.889, 0.897])}

n, m_bar, s = len(reported), reported.mean(), reported.std(ddof=1)
print(f"보고: {m_bar:.4f} ± {s:.4f} (seed {n}개, 표준편차)")
for name, r in reproduced.items():
    m = len(r)
    half = stats.t.ppf(0.975, df=n - 1) * s * np.sqrt(1 / n + 1 / m)   # 새 seed m개 평균의 95% 예측구간
    inside = abs(r.mean() - m_bar) <= half
    print(f"{name}: 평균 {r.mean():.4f}, 허용 범위 {m_bar - half:.4f}–{m_bar + half:.4f} → 재현 {inside}")
