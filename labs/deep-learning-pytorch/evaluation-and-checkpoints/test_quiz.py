"""17.6 퀴즈(content/17-deep-learning-pytorch/06-evaluation-and-checkpoints/quiz.toml) 정답을 계산으로 검증한다."""
import copy
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/06-evaluation-and-checkpoints/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_eval_does_not_disable_grad():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1)).eval()
    x = torch.randn(5, 3)
    y = model(x)
    assert y.requires_grad                                   # 기록은 켜져 있다
    with torch.no_grad():
        assert torch.equal(model(x), y.detach())             # 값은 같다
    assert correct("eval-does-not-disable-grad") == [0]


def test_no_grad_train_mode_updates_running_stats():
    torch = pytest.importorskip("torch")
    bn = torch.nn.BatchNorm1d(3).train()
    with torch.no_grad():
        bn(torch.randn(8, 3) + 5.0)
    assert bn.running_mean.min() > 0.1                       # 0에서 출발해 갱신되었다
    assert correct("no-grad-train-mode-batchnorm") == [0]


def test_mean_of_batch_means():
    sizes, maes = np.array([4, 4, 2]), np.array([1.0, 1.0, 4.0])
    overall = (sizes * maes).sum() / sizes.sum()
    assert overall == pytest.approx(Q["mean-of-batch-means"]["answer"])
    assert maes.mean() == pytest.approx(2.0)                 # 흔한 오답


def test_masked_average_definitions():
    err = np.array([[1., 1., 1., 1., 1.], [2., 2., 0., 0., 0.], [4., 0., 0., 0., 0.]])
    mask = (np.arange(5) < np.array([[5], [2], [1]])).astype(float)
    per_obs = (err * mask).sum() / mask.sum()
    per_series = ((err * mask).sum(1) / mask.sum(1)).mean()
    with_padding = err.mean()
    assert per_obs != pytest.approx(per_series)              # 두 정의는 다른 값이다
    assert with_padding < per_obs                            # padding을 세면 작아진다
    # batch 두 개로 나누면 batch 평균의 평균은 관측 단위 평균과 다르다
    b1, b2 = slice(0, 2), slice(2, 3)
    mob = np.mean([(err[b] * mask[b]).sum() / mask[b].sum() for b in (b1, b2)])
    assert mob != pytest.approx(per_obs)
    assert correct("masked-average-definitions") == [0, 1]


def test_resume_without_grad_is_exact():
    torch = pytest.importorskip("torch")

    def make():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
        o = torch.optim.AdamW(m.parameters(), lr=1e-2)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=10)
        return m, o, s

    def train(m, o, s, n):
        for _ in range(n):
            x = torch.randn(4, 2)
            loss = (m(x) - x.sum(1, keepdim=True)).pow(2).mean()
            o.zero_grad(); loss.backward(); o.step(); s.step()

    a = make(); train(*a, 10)
    b = make(); train(*b, 5)
    state = {"m": b[0].state_dict(), "o": b[1].state_dict(), "s": b[2].state_dict(), "rng": torch.get_rng_state()}
    state = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in state.items()}
    c = make()
    c[0].load_state_dict(state["m"]); c[1].load_state_dict(state["o"]); c[2].load_state_dict(state["s"])
    torch.set_rng_state(state["rng"])
    train(*c, 5)                                             # .grad는 저장하지 않았다
    for p, q in zip(a[0].parameters(), c[0].parameters()):
        assert torch.equal(p, q)
    assert correct("resume-checkpoint-contents") == [0, 1, 2, 3]


def test_best_selection_is_optimistic():
    rng = np.random.default_rng(0)
    epochs = np.arange(1, 61)
    true_val = 0.5 + 0.5 * np.exp(-epochs / 8) + 0.004 * epochs
    gaps = []
    for _ in range(500):
        val = true_val + rng.normal(0, 0.02, size=60)
        k = val.argmin()
        gaps.append(val[k] - true_val[k])
    assert np.mean(gaps) < -0.01                             # 고른 검증 점수는 평균적으로 낙관적
    assert correct("best-checkpoint-selection") == [0]
