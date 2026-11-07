import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc_cv_time = load().cc_cv_time
CELL = dict(q_ah=3.0, r_ohm=0.03, ocv_a_v=3.4, ocv_b_v=0.8, v_max_v=4.2)   # τ_CV = 3600·3·0.03/0.8 = 405 s


def test_cc_phase():
    t_cc, _ = cc_cv_time(6.0, soc0=0.1, i_cut_a=0.15, **CELL)
    # CC는 V = 3.4 + 0.8 s + 6·0.03 = 4.2 → s = 0.775에서 끝난다: (0.775 - 0.1)·3600·3 / 6
    assert np.isclose(t_cc, 1215.0, rtol=1e-2), f"CC 시간은 1215 s여야 합니다. 현재 {t_cc:.1f} s"


def test_cv_phase():
    _, t_cv = cc_cv_time(6.0, soc0=0.1, i_cut_a=0.15, **CELL)
    assert np.isclose(t_cv, 405.0 * np.log(6.0 / 0.15), rtol=1e-2), "CV에서 전류는 시상수 3600·Q·R/b로 지수적으로 줄어듭니다"


def test_higher_current_shorter_cc_not_cv():
    t_cc_lo, t_cv_lo = cc_cv_time(3.0, soc0=0.1, i_cut_a=0.15, **CELL)
    t_cc_hi, t_cv_hi = cc_cv_time(9.0, soc0=0.1, i_cut_a=0.15, **CELL)
    assert t_cc_hi < t_cc_lo / 3, "전류를 3배로 하면 CC 시간은 3배보다 더 줄어야 합니다(CC가 더 낮은 SOC에서 끝나므로)"
    assert t_cv_hi > t_cv_lo, "전류가 크면 CV를 더 높은 전류에서 시작하므로 CV 시간이 길어져야 합니다"


def test_starts_in_cv():
    t_cc, t_cv = cc_cv_time(6.0, soc0=0.9, i_cut_a=0.15, **CELL)
    i0 = (4.2 - 3.4 - 0.8 * 0.9) / 0.03
    assert t_cc == 0.0, "시작부터 전압 상한을 넘으면 CC 단계는 0초입니다"
    assert np.isclose(t_cv, 405.0 * np.log(i0 / 0.15), rtol=1e-2), "CV는 (V_max - OCV(soc0))/R의 전류에서 시작합니다"


def test_already_below_cutoff():
    t_cc, t_cv = cc_cv_time(6.0, soc0=0.999, i_cut_a=0.15, **CELL)
    assert t_cc == 0.0 and t_cv == 0.0, "CV 시작 전류가 이미 차단 전류 이하면 두 단계 모두 0초입니다"
