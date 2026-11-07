"""6.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/naming-and-style/cells.py
"""
# %% cell 1 — 단위가 이름에 없을 때와 있을 때
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

# %% cell 2 — shape이 어긋난 오차 계산
rng = np.random.default_rng(0)
y_true = rng.normal(size=5)                 # (N,)
y_pred = y_true[:, None] + 0.1              # (N, 1) 모델이 열벡터로 반환했다
err = y_pred - y_true                       # (N, 1) - (N,) → (N, N)
print("err.shape =", err.shape, " MSE =", round(float(np.mean(err**2)), 4))

y_pred = y_pred[:, 0]                       # (N,) shape을 명시적으로 맞춘다
assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
err = y_pred - y_true                       # (N,)
print("err.shape =", err.shape, "   MSE =", round(float(np.mean(err**2)), 4))

# %% cell 3 — 시간 기준 t0와 horizon
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

# %% cell 4 — 구문 트리로 이름 검사하기
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
