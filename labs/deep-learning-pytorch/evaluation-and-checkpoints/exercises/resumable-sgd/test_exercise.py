import importlib.util
import os
import pickle
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


MomentumSGD = load().MomentumSGD

_rng = np.random.default_rng(0)
X = _rng.normal(size=(64, 3))                                  # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * _rng.normal(size=64)


def run(opt, steps):
    for _ in range(steps):
        opt.step()
    return opt


def test_resume_matches_uninterrupted():
    straight = run(MomentumSGD(X, y, seed=0), 25)
    first = run(MomentumSGD(X, y, seed=0), 12)
    state = first.state_dict()
    resumed = MomentumSGD(X, y, seed=99)                       # 새 프로세스: 다른 시드로 만들어진 객체
    resumed.load_state_dict(state)
    run(resumed, 13)
    assert resumed.step_count == 25, "step_count를 복원해야 학습률 스케줄이 이어집니다"
    assert np.array_equal(resumed.velocity, straight.velocity), "momentum(velocity)이 다릅니다"
    assert np.array_equal(resumed.w, straight.w), "재개한 결과가 끊김 없이 학습한 결과와 비트 단위로 같아야 합니다"


def test_state_survives_serialization():
    first = run(MomentumSGD(X, y, seed=0), 7)
    state = pickle.loads(pickle.dumps(first.state_dict()))     # 파일에 썼다 읽은 것과 같다
    resumed = MomentumSGD(X, y, seed=5)
    resumed.load_state_dict(state)
    run(first, 5)
    run(resumed, 5)
    assert np.array_equal(first.w, resumed.w)


def test_saved_state_is_a_snapshot():
    opt = run(MomentumSGD(X, y, seed=0), 5)
    state = opt.state_dict()
    w_saved = state["w"].copy() if "w" in state else None
    run(opt, 5)                                                # 저장 뒤에 학습을 계속해도
    a = MomentumSGD(X, y, seed=1)
    a.load_state_dict(state)
    b = MomentumSGD(X, y, seed=2)
    b.load_state_dict(state)
    run(a, 3)                                                  # 불러온 쪽이 학습해도
    run(b, 3)
    assert np.array_equal(a.w, b.w), "같은 상태에서 재개한 두 객체가 다릅니다. 저장본이 원본과 메모리를 공유하지 않는지 확인하세요"
    if w_saved is not None:
        assert np.array_equal(state["w"], w_saved), "state_dict()가 반환한 배열이 이후 학습으로 바뀌었습니다. 복사본을 저장하세요"
