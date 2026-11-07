"""7.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/vector-spaces/cells.py
"""
# %% cell 1 — 다항식을 계수 벡터로
import numpy as np
from numpy.polynomial import polynomial as P

p = np.array([1.0, 2.0, 3.0])     # 1 + 2x + 3x^2  (상수항부터)
q = np.array([-1.0, 0.0, 1.0])    # -1 + x^2

x = np.linspace(-1, 1, 5)         # (5,) 함수를 확인할 점들
lhs = P.polyval(x, p) + 2 * P.polyval(x, q)   # 함수로 더하고 곱한다
rhs = P.polyval(x, p + 2 * q)                 # 계수 벡터로 더하고 곱한다
print("p + 2q 의 계수 :", p + 2 * q)
print("두 계산이 같은가:", np.allclose(lhs, rhs))

# %% cell 2 — basis와 좌표
B = np.array([[1.0, 1.0, 1.0],    # 열: 1, 1+x, 1+x+x^2 의 계수 벡터
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]])   # (3, 3)
c = np.linalg.solve(B, p)         # B c = p 를 푼다
print("단항식 basis 좌표:", p)
print("새 basis 좌표    :", c)
print("되돌리기         :", B @ c)

# %% cell 3 — 미분 연산자의 행렬
n = 3                                   # P_3: 차수 3 이하 다항식, 차원 4
D = np.diag(np.arange(1.0, n + 1), k=1) # (4, 4) j번째 열 = d/dx x^j 의 좌표
print(D)
r = np.array([1.0, 2.0, 3.0, 4.0])      # 1 + 2x + 3x^2 + 4x^3
print("D @ r     :", D @ r)
print("polyder   :", P.polyder(r))
print("D^4 = 0 ? :", np.allclose(np.linalg.matrix_power(D, 4), 0))

# %% cell 4 — 기저 변환과 닮음
T = np.diag([0.0, 1.0, 2.0])            # x d/dx 를 P_2 의 단항식 basis로
S = np.array([[1.0, 0.0, -0.5],         # 열: Legendre 다항식 1, x, (3x^2-1)/2
              [0.0, 1.0,  0.0],
              [0.0, 0.0,  1.5]])        # (3, 3)
T_new = np.linalg.solve(S, T @ S)       # S^{-1} T S
print(T_new)
v_new = np.array([2.0, -1.0, 1.0])      # 새 basis 좌표로 준 다항식
print("같은 결과?", np.allclose(S @ (T_new @ v_new), T @ (S @ v_new)))
print("trace:", np.trace(T), np.trace(T_new))
