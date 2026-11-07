"""17.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python"). 나머지는 NumPy만 쓴다(exec="browser").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/neural-network-components/cells.py
"""
# %% cell 1 — 활성화 함수와 도함수
import math
import numpy as np

z = np.array([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])            # (6,) 활성화 이전 값
erf = np.vectorize(math.erf)
acts = {
    "ReLU": lambda z: np.maximum(0.0, z),
    "GELU": lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2))),   # z * Phi(z)
    "SiLU": lambda z: z / (1 + np.exp(-z)),                  # z * sigmoid(z)
    "tanh": np.tanh,
}
h = 1e-5
print("z       " + "".join(f"{v:8.1f}" for v in z))
for name, f in acts.items():
    d = (f(z + h) - f(z - h)) / (2 * h)                      # 중앙차분 도함수
    print(f"{name:5s}   " + "".join(f"{v:8.3f}" for v in f(z)))
    print("  d/dz  " + "".join(f"{v:8.3f}" for v in d))

# %% cell 2 — 깊이·폭과 파라미터 수
def mlp_params(sizes):
    """Linear 층(n_in -> n_out)마다 가중치 n_in*n_out개 + bias n_out개"""
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))

for sizes in [[10, 256, 1], [10, 64, 64, 64, 1], [10, 40] + [40] * 7 + [1], [10, 256, 256, 1]]:
    print(f"은닉층 {len(sizes) - 2}개, 폭 {sizes[1]:3d}: 파라미터 {mlp_params(sizes):6d}개")

# %% cell 3 — 초기화 scale과 층별 활성값 크기
rng = np.random.default_rng(0)
n, depth = 128, 20
x0 = rng.normal(size=(256, n))                              # (B, D) 표준화된 입력
inits = {"std 0.01": 0.01, "Xavier": np.sqrt(2 / (n + n)), "He": np.sqrt(2 / n)}
for act_name, act in [("tanh", np.tanh), ("ReLU", lambda z: np.maximum(0.0, z))]:
    for init_name, std in inits.items():
        h, stds = x0, []
        for _ in range(depth):
            W = rng.normal(size=(n, n)) * std               # (D_in, D_out)
            h = act(h @ W)                                  # (B, D)
            stds.append(h.std())
        print(f"{act_name:4s} {init_name:8s} 층 1·5·10·20의 활성값 std:",
              " ".join(f"{stds[k]:8.1e}" for k in [0, 4, 9, 19]))

# %% cell 4 — 정규화 층의 train/eval 동작
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(8, 4) * 3 + 5                              # (B, D) 평균 약 5, 표준편차 약 3
bn, ln, rms = nn.BatchNorm1d(4), nn.LayerNorm(4), nn.RMSNorm(4)
y_bn, y_ln, y_rms = bn(x), ln(x), rms(x)                   # 새로 만든 모듈은 train 모드
print("BN  특성별 평균 0, 분산 1:", y_bn.mean(0).abs().max().item() < 1e-6,
      torch.allclose(y_bn.var(0, unbiased=False), torch.ones(4), atol=1e-3))
print("LN  표본별 평균 0, 분산 1:", y_ln.mean(1).abs().max().item() < 1e-6,
      torch.allclose(y_ln.var(1, unbiased=False), torch.ones(8), atol=1e-3))
print("RMS 표본별 제곱평균 1    :", torch.allclose(y_rms.pow(2).mean(1), torch.ones(8), atol=1e-3))
print("BN running_mean       :", bn.running_mean.numpy().round(3))
print("0.1 * batch 평균        :", (0.1 * x.mean(0)).numpy().round(3))

bn.eval(); ln.eval()
with torch.no_grad():
    print("eval에서 출력이 바뀌는가  BN:", not torch.allclose(bn(x), y_bn),
          " LN:", not torch.allclose(ln(x), y_ln))
    print("BN(eval)은 표본 하나도 처리한다:", bn(x[:1]).numpy().round(2))

# %% cell 5 — residual connection과 gradient 흐름
rng = np.random.default_rng(1)
D, L = 64, 50
Ws = [rng.normal(size=(D, D)) * 0.5 / np.sqrt(D) for _ in range(L)]   # 작은 초기화
for residual in [False, True]:
    h = rng.normal(size=D)                                  # (D,)
    Js = []
    for W in Ws:
        a = np.tanh(W @ h)                                  # 블록 F(h)
        J = (1 - a**2)[:, None] * W                         # dF/dh (D, D)
        h = h + a if residual else a
        Js.append(np.eye(D) + J if residual else J)         # 층 전체의 Jacobian
    g = np.ones(D) / np.sqrt(D)                             # 출력 쪽 gradient, 노름 1
    for J in reversed(Js):
        g = J.T @ g                                         # 역전파
    print(f"residual={residual!s:5s}  50층 아래 입력의 |grad| = {np.linalg.norm(g):.2e}")

# %% cell 6 — inverted dropout의 기댓값 보존
p = 0.2
a = np.ones((10000, 8))                                     # (B, D) 모두 1인 활성값
mask = rng.random(a.shape) >= p                             # 확률 1-p로 유지
a_train = a * mask / (1 - p)                                # train: 끄고 1/(1-p)배
print(f"유지된 비율 {mask.mean():.3f}, 남은 값 {a_train.max():.2f}, train 평균 {a_train.mean():.3f}")
print("eval: 그대로 통과, 평균", a.mean())
