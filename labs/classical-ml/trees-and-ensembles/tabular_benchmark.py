"""16.4 실습: 표 데이터에서 gradient boosting과 신경망(MLP)을 같은 튜닝 예산으로 비교한다.

실행: OMP_NUM_THREADS=1 uv run --project labs python labs/classical-ml/trees-and-ensembles/tabular_benchmark.py
두 모델 모두 학습 데이터 안의 3-fold 교차검증으로 무작위 탐색 설정 N_ITER개를 시도하고,
가장 좋은 설정을 학습 데이터 전체로 다시 학습한 뒤 한 번만 시험 데이터에서 평가한다.
"""
import time

import numpy as np
from scipy.stats import loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_ITER = 15


def make_alloy(n, rng):
    """본문(cells.py)과 같은 가상의 합금 시편 데이터."""
    temp_c = rng.uniform(450, 650, n)
    time_h = rng.uniform(0.5, 12, n)
    cu = rng.uniform(0, 0.05, n)
    mg = rng.uniform(0, 0.02, n)
    line = rng.integers(0, 5, n)
    noise_x = rng.normal(size=n)
    lot_id = rng.integers(0, 1000, n)
    peak = (temp_c > 520) & (temp_c < 600) & (time_h > 3)
    y = (200 + 3000 * cu + 5000 * mg + 40 * peak + 15 * np.sin(time_h)
         + np.array([0, 10, -5, 15, 0])[line] + rng.normal(0, 10, n))
    return np.column_stack([temp_c, time_h, cu, mg, line, noise_x, lot_id]), y


def main():
    rng = np.random.default_rng(0)
    X_tr, y_tr = make_alloy(2000, rng)
    X_te, y_te = make_alloy(2000, rng)

    searches = {
        "gradient boosting": RandomizedSearchCV(
            GradientBoostingRegressor(n_estimators=2000, n_iter_no_change=20, validation_fraction=0.2, random_state=0),
            {"learning_rate": loguniform(0.01, 0.3), "max_depth": [2, 3, 4, 5, 6],
             "subsample": [0.5, 0.7, 1.0], "min_samples_leaf": [1, 5, 20]},
            n_iter=N_ITER, cv=3, scoring="neg_root_mean_squared_error", random_state=0, n_jobs=1),
        "MLP": RandomizedSearchCV(
            TransformedTargetRegressor(                       # 출력도 표준화한다(신경망에 공정하게)
                make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, early_stopping=True, random_state=0)),
                transformer=StandardScaler()),
            {"regressor__mlpregressor__hidden_layer_sizes": [(32,), (64,), (64, 64), (128, 64), (128, 128)],
             "regressor__mlpregressor__alpha": loguniform(1e-6, 1e-1),
             "regressor__mlpregressor__learning_rate_init": loguniform(1e-4, 1e-2)},
            n_iter=N_ITER, cv=3, scoring="neg_root_mean_squared_error", random_state=0, n_jobs=1),
    }
    sq_err = {}
    ridge = make_pipeline(StandardScaler(), RidgeCV()).fit(X_tr, y_tr)
    sq_err["Ridge"] = (ridge.predict(X_te) - y_te) ** 2
    print(f"Ridge (튜닝: RidgeCV)         시험 RMSE {np.sqrt(sq_err['Ridge'].mean()):5.1f} MPa")
    for name, search in searches.items():
        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        sq_err[name] = (search.predict(X_te) - y_te) ** 2
        print(f"{name:18s} 설정 {N_ITER}개 탐색  CV RMSE {-search.best_score_:5.1f}  "
              f"시험 RMSE {np.sqrt(sq_err[name].mean()):5.1f} MPa  ({time.perf_counter() - t0:5.0f} s)")
        best = [f"{k.split('__')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('__')[-1]}={v}"
                for k, v in search.best_params_.items()]
        print("    고른 설정:", ", ".join(best))

    d = sq_err["MLP"] - sq_err["gradient boosting"]          # 시험 샘플별 짝지은 차이, (N_te,)
    boot = np.random.default_rng(1).choice(d, (5000, d.size)).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"MSE 차이 (MLP − GB) {d.mean():+.1f} MPa², 95% 부트스트랩 CI [{lo:+.1f}, {hi:+.1f}]")


if __name__ == "__main__":
    main()
