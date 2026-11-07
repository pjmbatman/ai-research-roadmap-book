"""16.4 퀴즈(content/16-classical-ml/04-trees-and-ensembles/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/04-trees-and-ensembles/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_tree_no_scaling():
    rng = np.random.default_rng(0)
    temp_c, conc = rng.uniform(20, 80, 300), rng.uniform(0.01, 1, 300)
    y = np.sin(temp_c / 10) + np.log(conc) + rng.normal(0, 0.1, 300)
    a = DecisionTreeRegressor(max_depth=5, random_state=0).fit(np.column_stack([temp_c, conc]), y)
    b = DecisionTreeRegressor(max_depth=5, random_state=0).fit(np.column_stack([temp_c + 273.15, np.log(conc)]), y)
    t_c, c_q = rng.uniform(20, 80, 50), rng.uniform(0.01, 1, 50)
    pa = a.predict(np.column_stack([t_c, c_q]))
    pb = b.predict(np.column_stack([t_c + 273.15, np.log(c_q)]))
    assert np.mean(np.isclose(pa, pb)) > 0.95                       # 문턱이 중점이라 경계 근처 몇 점만 다를 수 있다
    assert correct("tree-no-scaling") == [0]


def test_oob_fraction():
    N = 100000
    assert np.isclose((1 - 1 / N) ** N, Q["oob-fraction"]["answer"], atol=0.001)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5000, 5000)
    assert abs(1 - len(np.unique(idx)) / 5000 - np.exp(-1)) < 0.02


def test_rf_decorrelation():
    sigma2, B = 1.0, 10**6
    for rho in (0.8, 0.3):
        var = rho * sigma2 + (1 - rho) * sigma2 / B
        assert np.isclose(var, rho * sigma2, atol=1e-5)               # B를 늘려도 ρσ²가 남는다
    assert correct("rf-decorrelation") == [0]


def test_boosting_residuals():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(200, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + rng.normal(0, 0.05, 200)
    F = np.full(200, y.mean())
    for _ in range(50):
        F += 0.1 * DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y - F).predict(X)
    gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
    assert abs(np.sqrt(np.mean((F - y) ** 2)) - np.sqrt(np.mean((gbr.predict(X) - y) ** 2))) < 0.01
    assert correct("boosting-residuals") == [0]


def test_tree_extrapolation():
    rng = np.random.default_rng(2)
    t = rng.uniform(0.5, 12, (400, 1))
    y = 300 - 5 * t[:, 0] + rng.normal(0, 2, 400)
    gbr = GradientBoostingRegressor(random_state=0).fit(t, y)
    p12, p24, p48 = gbr.predict([[12.0], [24.0], [48.0]])
    assert np.isclose(p24, p48) and abs(p24 - p12) < 3                # 범위 밖에서 상수
    assert correct("tree-extrapolation") == [0]


def make_alloy(n, rng):
    temp_c, time_h = rng.uniform(450, 650, n), rng.uniform(0.5, 12, n)
    cu, mg = rng.uniform(0, 0.05, n), rng.uniform(0, 0.02, n)
    line, noise_x, lot_id = rng.integers(0, 5, n), rng.normal(size=n), rng.integers(0, 1000, n)
    peak = (temp_c > 520) & (temp_c < 600) & (time_h > 3)
    y = (200 + 3000 * cu + 5000 * mg + 40 * peak + 15 * np.sin(time_h)
         + np.array([0, 10, -5, 15, 0])[line] + rng.normal(0, 10, n))
    return np.column_stack([temp_c, time_h, cu, mg, line, noise_x, lot_id]), y


def test_impurity_importance_bias():
    rng = np.random.default_rng(0)
    (X, y), (Xt, yt) = make_alloy(1000, rng), make_alloy(1000, rng)
    rf = RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=0, n_jobs=1).fit(X, y)
    imp = rf.feature_importances_
    assert imp[6] > 1.5 * imp[4]                                     # lot_id > line (impurity)
    perm = permutation_importance(rf, Xt, yt, n_repeats=5, random_state=0, n_jobs=1).importances_mean
    assert abs(perm[6]) < 0.01                                       # 시험 데이터에서는 0 근처
    assert correct("impurity-importance-bias") == [0]


def test_importance_not_causal():
    assert correct("importance-not-causal") == [0, 1]
