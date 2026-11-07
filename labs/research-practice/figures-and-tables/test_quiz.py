"""23.9 퀴즈(content/23-research-practice/09-figures-and-tables/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import importlib.util
import io
import runpy
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/23-research-practice/09-figures-and-tables/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run_cells():
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(HERE / "cells.py"))


def test_error_bar_overlap():
    g = run_cells()
    a, b, t4 = g["rmse_a"], g["rmse_b"], g["t4"]
    ha, hb = (t4 * r.std(ddof=1) / np.sqrt(5) for r in (a, b))
    assert a.mean() - ha < b.mean() + hb                                  # 방법별 95% CI가 겹친다
    d = b - a
    half = t4 * d.std(ddof=1) / np.sqrt(5)
    assert round(d.mean() - half, 3) == -0.028 and round(d.mean() + half, 3) == -0.015
    assert np.all(d < 0)                                                  # 다섯 seed 모두 B가 낫다
    assert correct("error-bar-overlap") == [1]


def test_truncated_axis_ratio():
    a, b, y0 = 0.912, 0.925, 0.905
    ratio = (b - y0) / (a - y0)
    assert abs(ratio - Q["truncated-axis-ratio"]["answer"]) <= Q["truncated-axis-ratio"]["tolerance"]
    lie = ((b - y0) - (a - y0)) / (a - y0) / ((b - a) / a)
    assert round(lie) == 130                                              # 해설의 lie factor


def test_rainbow_colormap():
    g = run_cells()
    import matplotlib
    x = np.linspace(0, 1, 256)
    L_jet = g["lightness"](matplotlib.colormaps["jet"](x)[:, :3])
    L_vir = g["lightness"](matplotlib.colormaps["viridis"](x)[:, :3])
    assert not np.all(np.diff(L_jet) > 0) and np.all(np.diff(L_vir) > 0)
    assert 0.1 < x[L_jet.argmax()] < 0.9                                  # jet의 가장 밝은 곳은 가운데
    assert correct("rainbow-colormap") == [1]


def test_log_axis_slope():
    h = np.logspace(-3, -1, 5)
    e = 0.1 * h**2                                                        # 1e-7 … 1e-3
    assert np.isclose(e[0], 1e-7) and np.isclose(e[-1], 1e-3)
    slope = np.polyfit(np.log10(h), np.log10(e), 1)[0]
    assert np.isclose(slope, 2.0)
    assert correct("log-axis") == [1]


def test_hand_edited_figure():
    assert correct("hand-edited-figure") == [2]


def test_sig_fig_mean():
    spec = importlib.util.spec_from_file_location("sol", HERE / "exercises/uncertainty-rounding/solution.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    s = sol.format_pm(2.71828, 0.0347, 2)
    assert s == "2.718 ± 0.035"
    assert float(s.split(" ± ")[0]) == Q["sig-fig-mean"]["answer"]
