"""0.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/orientation/how-to-read/cells.py
"""
# %% cell 1 — shape 주석 읽기
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 3))        # (N, D): 샘플 8개, 특징 3개
W = rng.normal(size=(3, 4))        # (D, H): 특징 3개 → 은닉 4개
Z = np.maximum(0.0, X @ W)         # (N, H)
print("X:", X.shape, " W:", W.shape, " Z:", Z.shape)

# %% cell 2 — 단위 접미사
temp_c = np.array([20.0, 25.0, 30.0])   # (3,) 섭씨
temp_k = temp_c + 273.15                # (3,) 켈빈
time_s = np.array([0.0, 60.0, 120.0])   # (3,) 초
rate_k_per_s = np.diff(temp_k) / np.diff(time_s)   # (2,)
print("temp_k       =", temp_k)
print("rate_k_per_s =", rate_k_per_s)
