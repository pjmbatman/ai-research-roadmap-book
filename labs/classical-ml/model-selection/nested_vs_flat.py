"""16.6 실습: 같은 모델을 nested CV와 단순 CV로 평가해 성능 추정의 낙관 편향을 측정한다.

같은 분포에서 작은 데이터셋을 여러 번 뽑고, 데이터셋마다
  (1) 단순 CV: 하이퍼파라미터 격자를 5-fold CV로 탐색하고, 고른 설정의 CV 점수(best_score_)를 그대로 보고
  (2) nested CV: 바깥 5-fold의 학습 부분에서만 (1)을 하고, 고른 설정을 바깥 시험 fold에서 평가
  (3) 참값: 전체 데이터로 탐색해 고른 최종 모델을 같은 분포의 큰 독립 데이터(2만 개)에서 평가
를 계산해, (1)과 (2)가 (3)을 얼마나 잘 추정하는지 비교한다.

실행: uv run --project labs python labs/classical-ml/model-selection/nested_vs_flat.py
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_DATASETS = 30
GRID = {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [0.001, 0.01, 0.1, 1, 10]}   # 설정 25개


def sample(rng, n, d=10, n_informative=2):
    """특징 10개 중 2개에만 신호가 있는 이진 분류. 결정 경계는 원(비선형)."""
    X = rng.normal(size=(n, d))                                         # (n, d)
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    p = 1 / (1 + np.exp(-2.0 * (r2 - 1.4)))                             # 원 밖일수록 양성
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def one_dataset(seed, n):
    rng = np.random.default_rng(seed)
    X, y = sample(rng, n)
    X_big, y_big = sample(rng, 20000)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    outer = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), GRID, cv=inner)
    search.fit(X, y)
    flat = search.best_score_
    true = search.score(X_big, y_big)
    nested = cross_val_score(search, X, y, cv=outer).mean()
    return flat, nested, true


def main():
    print(f"데이터셋 {N_DATASETS}개씩, SVC(RBF) 격자 25개, 5-fold (nested는 5 × 5-fold)")
    print("  n    단순 CV   nested CV   참 정확도 | 단순 CV - 참   nested - 참   (데이터셋별 sd)")
    for n in (60, 120, 240):
        res = np.array([one_dataset(1000 * n + s, n) for s in range(N_DATASETS)])   # (데이터셋, 3)
        flat, nested, true = res.T
        print(f"{n:4d}   {flat.mean():.3f}     {nested.mean():.3f}      {true.mean():.3f}   |"
              f"   {np.mean(flat - true):+.3f}        {np.mean(nested - true):+.3f}"
              f"      ({np.std(flat - true, ddof=1):.3f}, {np.std(nested - true, ddof=1):.3f})")


if __name__ == "__main__":
    main()
