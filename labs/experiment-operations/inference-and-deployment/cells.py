"""22.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 1·2는 PyTorch가 필요하고(exec="python"), 나머지는 NumPy만 쓰는 브라우저 셀이다(exec="browser").
실행: uv run --project labs --group torch python labs/experiment-operations/inference-and-deployment/cells.py
"""
# %% cell 1 — torch.export로 내보내고 저장·불러온 뒤 수치 비교
import io
import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
for _ in range(5):
    model(torch.randn(16, 4))                      # train 모드 forward: BatchNorm 이동 평균이 갱신된다
model.eval()                                       # 내보내기 전에 평가 모드로

x = torch.randn(8, 4)                              # 예시 입력, (B, 4)
batch = torch.export.Dim("batch")                  # batch 축은 크기가 바뀔 수 있다고 알린다
ep = torch.export.export(model, (x,), dynamic_shapes=({0: batch},))
buf = io.BytesIO()
torch.export.save(ep, buf)                         # 파일 대신 메모리 버퍼에 저장
buf.seek(0)
deployed = torch.export.load(buf).module()

with torch.no_grad():
    for n in [8, 3, 100]:                          # 예시와 다른 batch 크기도 확인
        xt = torch.randn(n, 4)
        diff = (deployed(xt) - model(xt)).abs().max().item()
        print(f"batch {n:3d}: 최대 차이 {diff:.1e}  allclose: {torch.allclose(deployed(xt), model(xt), atol=1e-6)}")

# %% cell 2 — train 모드로 내보내면
model.train()
ep_train = torch.export.export(model, (x,))       # eval()을 빠뜨렸다
model.eval()
with torch.no_grad():
    a, b = ep_train.module()(x), model(x)
print("train 모드로 내보낸 모델과 평가 모드 모델의 최대 차이 > 1e-3:", (a - b).abs().max().item() > 1e-3)

# %% cell 3 — 대칭 int8 양자화: 텐서 하나의 scale과 행(채널)별 scale
import numpy as np

rng = np.random.default_rng(0)
row_scale = np.array([0.01, 0.1, 1.0, 5.0])[:, None]           # 출력 채널마다 크기가 다르다
W = rng.normal(size=(4, 64)) * row_scale                       # (out, in) 가중치
x = rng.normal(size=64)                                        # (in,) 입력

def quant_sym(w, axis=None):
    amax = np.abs(w).max(axis=axis, keepdims=axis is not None)
    scale = amax / 127                                         # 최댓값이 ±127로 간다
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale

for name, axis in [("텐서 하나", None), ("행별", 1)]:
    q, s = quant_sym(W, axis)
    W_hat = q * s                                              # 복원(dequantize)
    rel_row = np.linalg.norm(W_hat - W, axis=1) / np.linalg.norm(W, axis=1)
    rel_out = np.abs(W_hat @ x - W @ x) / np.abs(W @ x)
    print(f"{name:6s} 가중치 상대오차 {np.round(rel_row, 4)}  출력 상대오차 {np.round(rel_out, 4)}")
print(f"메모리: fp32 {W.size * 4} byte → int8 {W.size} byte + scale")

# %% cell 4 — 정수 행렬 곱: int8 × int8을 int32로 누적
qW, sW = quant_sym(W, 1)                                       # (4, 64) int8, (4, 1)
qx, sx = quant_sym(x)                                          # (64,) int8, 스칼라
acc32 = qW.astype(np.int32) @ qx.astype(np.int32)              # 정수 곱의 합, (4,)
y_int8 = acc32 * sW[:, 0] * sx                                 # 실수로 되돌린다
print("float 출력 :", np.round(W @ x, 4))
print("int8 출력  :", np.round(y_int8, 4))
worst = 64 * 127 * 127                                         # 길이 64 내적의 최악 누적값
print("최악의 누적값:", worst, " int16 최댓값:", np.iinfo(np.int16).max, " int32로 안전한 최대 길이:",
      np.iinfo(np.int32).max // (127 * 127))

# %% cell 5 — batch 크기와 지연·처리량 (가정한 비용 모형)
t_fixed_ms, t_per_sample_ms = 4.0, 0.05            # batch마다 고정 비용, 표본당 비용 (가정)
arrival_per_s = 2000                               # 초당 요청 수 (가정)
print("batch  계산 지연(ms)  모으는 대기(ms)  총 지연(ms)  처리량(표본/s)")
for B in [1, 8, 32, 128, 512]:
    compute_ms = t_fixed_ms + B * t_per_sample_ms
    wait_ms = (B - 1) / arrival_per_s * 1000 / 2   # batch를 채우는 동안 요청이 평균적으로 기다리는 시간
    print(f"{B:5d} {compute_ms:13.2f} {wait_ms:15.2f} {compute_ms + wait_ms:12.2f} {B / compute_ms * 1000:15.0f}")

# %% cell 6 — 추론 계약: 학습 때의 전처리를 그대로 쓰는가
rng = np.random.default_rng(1)
temp_c = rng.uniform(10, 40, size=200)             # 학습 데이터: 온도 (°C)
y_obs = 0.8 * temp_c + 3 + rng.normal(0, 0.5, size=200)
mu, sd = temp_c.mean(), temp_c.std()               # 학습 데이터로 정한 정규화 상수 — 계약의 일부
z = (temp_c - mu) / sd
a, b = np.polyfit(z, y_obs, 1)                     # 정규화된 입력에 대한 선형 모델

def predict(t, mean, std):
    return a * (t - mean) / std + b

new_c = np.array([15.0, 20.0, 25.0])               # 배포 후 들어온 요청 3개 (°C)
ref = predict(new_c, mu, sd)
print("학습 때의 상수      :", np.round(ref, 2))
print("요청 batch로 정규화 :", np.round(predict(new_c, new_c.mean(), new_c.std()), 2))
print("단위가 K로 들어옴   :", np.round(predict(new_c + 273.15, mu, sd), 2))
