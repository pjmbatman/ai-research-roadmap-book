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


paired_verdict = load().paired_verdict


def test_skill_and_diff_by_hand():
    out = paired_verdict([3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0])
    assert np.isclose(out["skill"], 0.25), f"skill = 1 - 3/4 = 0.25여야 합니다. 현재 {out['skill']}"
    assert np.isclose(out["diff"], -1.0), f"diff = 평균(3 - 4) = -1이어야 합니다. 현재 {out['diff']}"
    assert np.allclose(out["ci"], (-1.0, -1.0)), "차이가 모두 같으면 구간의 폭은 0입니다"
    assert out["verdict"] == "이긴다"


def test_pairing_removes_shared_difficulty():
    # 단위마다 난이도가 크게 다르고(0.5–5), 새 모델은 모든 단위에서 약 0.1 낫다
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 5.0, 30)                         # (30,)
    model = base - 0.1 + rng.normal(0, 0.02, 30)
    out = paired_verdict(model, base)
    lo, hi = out["ci"]
    assert lo <= out["diff"] <= hi, "구간은 차이의 평균을 포함해야 합니다"
    assert hi - lo < 0.05, f"짝을 지으면 구간이 좁아야 합니다(폭 {hi - lo:.3f}). 두 오차를 같은 인덱스로 뽑았는지 확인하세요"
    assert out["verdict"] == "이긴다"


def test_loses():
    rng = np.random.default_rng(1)
    base = rng.uniform(1, 2, 25)
    out = paired_verdict(base + 0.3 + rng.normal(0, 0.05, 25), base)
    assert out["skill"] < 0, "baseline보다 나쁘면 skill score는 음수입니다"
    assert out["verdict"] == "진다"


def test_undecided():
    rng = np.random.default_rng(2)
    base = rng.uniform(1, 2, 12)
    out = paired_verdict(base + rng.normal(0, 0.5, 12), base)
    lo, hi = out["ci"]
    assert lo < 0 < hi
    assert out["verdict"] == "판정 불가", "구간이 0을 포함하면 판정 불가입니다"


def test_reproducible():
    rng = np.random.default_rng(3)
    a, b = rng.normal(1, 0.3, 15), rng.normal(1, 0.3, 15)
    assert paired_verdict(a, b, seed=7)["ci"] == paired_verdict(a, b, seed=7)["ci"], "같은 seed면 같은 결과여야 합니다"
