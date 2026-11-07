"""1.5 퀴즈(content/01-research-landscape/05-first-pipeline/quiz.toml) 정답과 본문 수치를 계산으로 검증한다."""
import importlib.util
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = Path(__file__).parents[3] / "content/01-research-landscape/05-first-pipeline/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

spec = importlib.util.spec_from_file_location("first_pipeline", HERE / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_period_formula_small_angle_limit():
    """타원적분 공식은 작은 진폭에서 2π√(L/g)로 돌아간다(데이터 생성의 검증)."""
    from scipy.special import ellipk
    L = 1.0
    small = 4 * np.sqrt(L / pipeline.G_MS2) * ellipk(np.sin(np.deg2rad(0.01) / 2) ** 2)
    assert np.isclose(small, 2 * np.pi * np.sqrt(L / pipeline.G_MS2), rtol=1e-8)


def test_read_the_table():
    df = pipeline.make_pendulum_data(300, 0)
    X, y = df[["length_m", "amp_deg"]].to_numpy(), df["period_s"].to_numpy()
    tr, va, te = pipeline.split_indices(300, 1)
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(X[tr], y[tr])
    phys = LinearRegression().fit(pipeline.physics_features(X[tr]), y[tr])
    mlp = pipeline.train_mlp(X[tr], y[tr], seed=0)
    r_lin = pipeline.rmse(lin.predict(X[te]), y[te])
    r_phys = pipeline.rmse(phys.predict(pipeline.physics_features(X[te])), y[te])
    r_mlp = pipeline.rmse(mlp(X[te]), y[te])
    assert r_phys < r_mlp < r_lin
    assert [round(v, 4) for v in (r_lin, r_phys, r_mlp)] == [0.0760, 0.0089, 0.0190]
    assert correct("read-the-table") == [1]


def test_mlp_param_count():
    D, H, O = 2, 32, 1
    assert D * H + H + H * O + O == Q["mlp-param-count"]["answer"]


def test_noise_only_rmse_spread():
    """테스트 60개에서 잡음만으로 생기는 RMSE의 흔들림(본문: 약 0.0009 s)."""
    rng = np.random.default_rng(0)
    r = np.sqrt(np.mean(rng.normal(0, 0.01, size=(20_000, 60)) ** 2, axis=1))   # (20000,)
    assert abs(r.std() - 0.01 / np.sqrt(2 * 60)) < 1e-4
    assert np.mean(r <= 0.0089) > 0.05                       # 0.0089 이하도 드물지 않다
    assert correct("suspicious-results") == [0, 1]


def test_tanh_derivative():
    a, h = np.linspace(-2, 2, 9), 1e-6
    numeric = (np.tanh(a + h) - np.tanh(a - h)) / (2 * h)
    assert np.allclose(numeric, 1 - np.tanh(a) ** 2, atol=1e-8)
    assert correct("tanh-derivative-factor") == [0]


def test_single_choice_answers():
    assert correct("standardize-train-only") == [1]
    assert correct("where-next") == [0]
