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


is_estimate = load().is_estimate


def test_same_policy_gives_sample_mean():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(50, 4))
    p = np.full((50, 4), 0.3)
    ois, wis, ess = is_estimate(rewards, p, p, gamma=1.0)
    assert np.isclose(ois, rewards.sum(axis=1).mean()), "두 정책이 같으면 가중치는 모두 1이고 추정은 이득의 표본 평균입니다"
    assert np.isclose(wis, ois)
    assert np.isclose(ess, 50.0), "가중치가 모두 같으면 ESS = N입니다"


def test_hand_example():
    rewards = np.array([[1.0, 1.0], [0.0, 1.0]])         # 궤적 2개, 2 step
    pb = np.array([[0.5, 0.5], [0.5, 0.5]])
    pe = np.array([[1.0, 1.0], [0.0, 1.0]])              # 두 번째 궤적은 평가 정책이 하지 않는 행동 포함
    ois, wis, ess = is_estimate(rewards, pb, pe, gamma=1.0)
    # 가중치: 첫 궤적 2 * 2 = 4, 두 번째 0 * 2 = 0. 이득: 2, 1
    assert np.isclose(ois, (4 * 2 + 0 * 1) / 2), "OIS = mean(w * G) = (4 * 2 + 0) / 2 = 4입니다"
    assert np.isclose(wis, 2.0), "WIS = sum(w G) / sum(w) = 8 / 4 = 2입니다"
    assert np.isclose(ess, 1.0), "가중치 하나만 0이 아니면 ESS = 1입니다"


def test_discounting():
    rewards = np.array([[1.0, 1.0, 1.0]])
    p = np.ones((1, 3))
    ois, _, _ = is_estimate(rewards, p, p, gamma=0.5)
    assert np.isclose(ois, 1 + 0.5 + 0.25), "이득은 sum_t gamma^t r_t입니다"


def test_unbiased_on_bandit_chain():
    rng = np.random.default_rng(1)
    N, H = 20_000, 3
    pb, pe = np.array([0.5, 0.5]), np.array([0.2, 0.8])
    a = rng.choice(2, size=(N, H), p=pb)
    ois, wis, ess = is_estimate(a.astype(float), pb[a], pe[a], gamma=1.0)
    assert abs(ois - 0.8 * H) < 0.1, "행동 1의 보상이 1이면 평가 정책의 가치는 0.8 * H입니다"
    assert abs(wis - 0.8 * H) < 0.05
    assert 0.0 < ess < N, "가중치가 고르지 않으면 ESS는 N보다 작습니다"
