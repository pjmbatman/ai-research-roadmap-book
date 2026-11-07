"""20.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reinforcement-learning/mdp-and-dp/cells.py
"""
# %% cell 1 — 미끄러지는 4×4 격자 세계 MDP
import numpy as np

H, W = 4, 4                                  # 4×4 격자, 상태 번호 s = 행 * W + 열
GOAL, PIT = 3, 7                             # (0,3) 목표 +1, (1,3) 함정 -1, 둘 다 종료 상태
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 행동 0..3 = 위, 오른쪽, 아래, 왼쪽
S, A = H * W, 4

def move(s, d):                              # 격자 밖으로 나가는 이동은 제자리
    r, c = divmod(s, W)
    r2, c2 = r + MOVES[d][0], c + MOVES[d][1]
    return r2 * W + c2 if 0 <= r2 < H and 0 <= c2 < W else s

P = np.zeros((S, A, S))                      # P[s, a, s'] = 전이확률, (16, 4, 16)
for s in range(S):
    for a in range(A):
        if s in (GOAL, PIT):
            P[s, a, s] = 1.0                 # 종료 상태는 흡수 상태로 둔다
            continue
        for d, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a + 3) % 4, 0.1)]:
            P[s, a, move(s, d)] += p         # 80% 의도대로, 10%씩 좌우로 미끄러짐
r_enter = np.zeros(S)
r_enter[GOAL], r_enter[PIT] = 1.0, -1.0      # 목표·함정에 들어갈 때 받는 보상
R = P @ r_enter                              # R[s, a] = 기대 즉시 보상, (16, 4)
R[[GOAL, PIT]] = 0.0                         # 종료 뒤에는 보상 없음
print("P", P.shape, " 모든 (s, a)에서 합 1:", np.allclose(P.sum(axis=2), 1))
print("상태 2(목표 왼쪽)에서 R[2, :] =", R[2].round(2))
print("상태 2에서 '오른쪽'의 다음 상태 분포:", {s2: float(p) for s2, p in enumerate(P[2, 1]) if p > 0})

# %% cell 2 — 무작위 정책의 가치: Bellman 기대 방정식을 선형계로 푼다
gamma = 0.95
pi_rand = np.full((S, A), 0.25)              # 정책 pi(a|s), (16, 4)
P_pi = np.einsum("sa,sat->st", pi_rand, P)   # 정책을 따를 때의 Markov chain, (16, 16)
r_pi = (pi_rand * R).sum(axis=1)             # (16,)
V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)    # V = r_pi + gamma P_pi V
Q = R + gamma * P @ V                        # Q[s, a] = R + gamma sum_s' P V, (16, 4)
print("무작위 정책의 V (행 = 격자의 행)")
print(V.reshape(H, W).round(3) + 0.0)          # + 0.0은 -0.을 0.으로 바꾼다
print("Bellman 잔차 < 1e-10:", np.abs(V - (r_pi + gamma * P_pi @ V)).max() < 1e-10)
print("V = sum_a pi Q:", np.allclose(V, (pi_rand * Q).sum(axis=1)))

# %% cell 3 — 가치 반복과 정책 반복
ARROW = np.array(list("↑→↓←"))
def show(pi):                                # 결정적 정책을 화살표 격자로 출력
    g = ARROW[pi].copy()
    g[GOAL], g[PIT] = "G", "X"
    print("\n".join(" ".join(row) for row in g.reshape(H, W)))
V, gaps = np.zeros(S), []
for k in range(1000):                        # 가치 반복: V <- max_a (R + gamma P V)
    V_new = (R + gamma * P @ V).max(axis=1)
    gaps.append(np.abs(V_new - V).max())
    V = V_new
    if gaps[-1] < 1e-8:
        break
pi_vi = (R + gamma * P @ V).argmax(axis=1)
ratio = max(g1 / g0 for g0, g1 in zip(gaps, gaps[1:]))
print(f"가치 반복: {k + 1}번에 수렴, 연속 오차 비의 최댓값 {ratio:.3f} <= gamma = {gamma}")

pi, n_improve = np.zeros(S, dtype=int), 0    # 정책 반복: 모두 '위'에서 출발
while True:
    P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]           # (16, 16), (16,)
    V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)          # 정책 평가(정확히)
    pi_new = (R + gamma * P @ V_pi).argmax(axis=1)                  # 탐욕적 개선
    n_improve += 1
    if np.array_equal(pi_new, pi):
        break
    pi = pi_new
print(f"정책 반복: 개선 {n_improve}번, 같은 정책: {np.array_equal(pi, pi_vi)},"
      f" 가치 차이 < 1e-6: {np.abs(V_pi - V).max() < 1e-6}")
print(f"시작 칸 (3,0)의 V* = {V[12]:.3f}")
show(pi_vi)

# %% cell 4 — 부분 관측: 복도에서의 belief 갱신
n_pos = 5                                    # 1차원 복도의 칸 0..4, 문은 칸 1과 3에 있다
door = np.array([0, 1, 0, 1, 0])
def sense_lik(z):                            # p(z | 위치): 센서가 90% 맞게 문을 감지
    return np.where(door == z, 0.9, 0.1)
T = np.zeros((n_pos, n_pos))                 # '오른쪽 한 칸' 행동의 전이, 끝 칸에서는 제자리
for i in range(n_pos):
    T[i, min(i + 1, n_pos - 1)] += 0.8       # 80% 한 칸 이동
    T[i, i] += 0.2                           # 20% 제자리

b = np.full(n_pos, 1 / n_pos)                # 처음에는 위치를 모른다
for t, z in enumerate([1, 0, 1]):            # 관측: 문, 벽, 문 (관측 사이에 오른쪽 이동)
    if t > 0:
        b = b @ T                            # 예측: b'(s') = sum_s b(s) T(s, s')
    b = b * sense_lik(z)                     # 갱신: Bayes 규칙
    b /= b.sum()
    print(f"관측 {t + 1} ({'문' if z else '벽'})  belief = {b.round(3)}")

# %% cell 5 — 보상 설계의 실패: 살아 있기만 해도 보상을 주면
for living in [0.0, -0.04, 0.1]:             # 종료 전 매 step에 더하는 보상
    R2 = R + living
    R2[[GOAL, PIT]] = 0.0
    V2 = np.zeros(S)
    for _ in range(2000):
        V2 = (R2 + gamma * P @ V2).max(axis=1)
    pi2 = (R2 + gamma * P @ V2).argmax(axis=1)
    d = np.zeros(S); d[12] = 1.0             # 시작 칸 (3,0)에서 정책을 100 step 따라간다
    for _ in range(100):
        d = d @ P[np.arange(S), pi2]
    alive = np.delete(d, [GOAL, PIT]).sum()  # 아직 종료하지 않은 확률
    print(f"step 보상 {living:+.2f}: 100 step 뒤 목표 {d[GOAL]:.3f}, 함정 {d[PIT]:.3f}, 진행 중 {alive:.3f}")
