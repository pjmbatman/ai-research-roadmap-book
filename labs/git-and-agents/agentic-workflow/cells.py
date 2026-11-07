"""4.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/git-and-agents/agentic-workflow/cells.py
"""
# %% cell 1 — test set을 보며 고르면 생기는 낙관 편향
import numpy as np

rng = np.random.default_rng(0)
n_configs, n_test, p_true = 50, 200, 0.70        # 모든 설정의 실제 정확도는 똑같이 0.70

test_acc = rng.binomial(n_test, p_true, size=n_configs) / n_test     # (n_configs,)
best = np.argmax(test_acc)                        # test 결과를 보고 가장 좋은 설정을 고른다
fresh = rng.binomial(n_test, p_true) / n_test     # 고른 설정을 새 데이터로 다시 평가
print(f"고른 설정의 test 정확도   {test_acc[best]:.3f}")
print(f"같은 설정, 새 데이터      {fresh:.3f}")
print(f"설정 50개 test 정확도 평균 {test_acc.mean():.3f}")

# %% cell 2 — 시도 횟수에 따른 편향의 크기 (반복 2000번 평균)
for k in [1, 5, 20, 100]:
    acc = rng.binomial(n_test, p_true, size=(2000, k)) / n_test      # (반복, 설정 수)
    print(f"설정 {k:3d}개 중 최고: 평균 {acc.max(axis=1).mean():.3f}")
