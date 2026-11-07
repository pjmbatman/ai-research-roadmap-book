"""4.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/git-and-agents/ai-coding-tools/cells.py
"""
# %% cell 1 — shape: 조용한 broadcasting
import numpy as np

rng = np.random.default_rng(0)
y = rng.normal(size=4)                    # (N,)    정답
pred = y[:, None] + 0.1                   # (N, 1)  모델 출력이 열벡터로 나왔다
mse_wrong = ((pred - y) ** 2).mean()      # (N, 1) - (N,) → (N, N)
mse_right = ((pred[:, 0] - y) ** 2).mean()
print("pred - y 의 shape:", (pred - y).shape)
print(f"틀린 MSE = {mse_wrong:.4f}, 맞는 MSE = {mse_right:.4f}")

# %% cell 2 — gradient: 유한차분으로 검증
def loss(w, X, s):                        # s: ±1 라벨 (N,), X: (N, D)
    return np.mean(np.logaddexp(0.0, -s * (X @ w)))

def grad_ai(w, X, s):                     # AI가 쓴 gradient (검증 대상)
    m = s * (X @ w)                       # (N,)
    return -(X * (s / (1 + np.exp(-m)))[:, None]).mean(axis=0)   # (D,)

def grad_num(w, X, s, h=1e-6):            # 중앙 차분
    E = np.eye(w.size) * h                # (D, D)
    return np.array([(loss(w + e, X, s) - loss(w - e, X, s)) / (2 * h) for e in E])

def grad_fixed(w, X, s):                  # 손으로 유도해 고친 gradient
    m = s * (X @ w)                       # (N,)
    return -(X * (s / (1 + np.exp(m)))[:, None]).mean(axis=0)    # (D,)

X = rng.normal(size=(50, 3))              # (N, D)
s = np.sign(rng.normal(size=50))          # (N,)
w = rng.normal(size=3)                    # (D,)
g_num = grad_num(w, X, s)
rel = lambda g: np.linalg.norm(g - g_num) / np.linalg.norm(g_num)
print(f"AI gradient 상대 오차   {rel(grad_ai(w, X, s)):.1e}")
print(f"고친 gradient 상대 오차 {rel(grad_fixed(w, X, s)):.1e}")

# %% cell 3 — 데이터 분할: 전체 데이터로 고른 특징
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

X = rng.normal(size=(100, 5000))          # (N, D)  순수한 잡음
y = rng.integers(0, 2, size=100)          # (N,)    입력과 무관한 라벨

X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)    # 전체 데이터로 고른다
leaky = cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
honest = cross_val_score(pipe, X, y, cv=5).mean()           # fold 안에서 고른다
print(f"누수 있는 CV 정확도 {leaky:.2f}, 올바른 CV 정확도 {honest:.2f}")
