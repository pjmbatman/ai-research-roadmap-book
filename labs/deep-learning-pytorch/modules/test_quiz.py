"""17.3 퀴즈(content/17-deep-learning-pytorch/03-modules/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/03-modules/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_list_not_registered():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(64, 64) for _ in range(4)]

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x

    m = Bad()
    assert len(list(m.parameters())) == 0
    x = torch.randn(2, 64)
    m(x).sum().backward()
    assert m.layers[0].weight.grad is not None            # gradient는 계산된다
    assert m.layers[0] is not m.layers[1]                 # 공유가 아니다
    assert correct("list-not-registered") == [0]


def test_call_not_forward():
    torch = pytest.importorskip("torch")
    from torch import nn
    m, calls = nn.Linear(2, 2), []
    m.register_forward_hook(lambda *a: calls.append(1))
    x = torch.randn(1, 2)
    m.forward(x)
    assert calls == []
    m(x)
    assert calls == [1]
    m.train()
    m(x)
    assert m.training                                      # __call__은 모드를 바꾸지 않는다
    assert correct("call-not-forward") == [0]


def test_batchnorm_buffers():
    torch = pytest.importorskip("torch")
    bn = torch.nn.BatchNorm1d(8)
    keys = list(bn.state_dict().keys())
    buffers = {n for n, _ in bn.named_buffers()}
    texts = [c["text"] for c in Q["batchnorm-buffers"]["choices"]]
    assert set(texts) == set(keys)
    expected = [i for i, t in enumerate(texts) if t in buffers]
    assert correct("batchnorm-buffers") == expected == [2, 3, 4]


def test_plain_tensor_attribute():
    torch = pytest.importorskip("torch")
    from torch import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.tensor([300.0, 1e5])
            self.fc = nn.Linear(2, 1)

    m = M()
    assert "mean" not in m.state_dict()
    m.double()                                            # .to와 같은 경로: 등록된 텐서만 바뀐다
    assert m.mean.dtype == torch.float32 and m.fc.weight.dtype == torch.float64
    assert correct("plain-tensor-attribute") == [0]


def test_state_dict_key_count():
    torch = pytest.importorskip("torch")
    from torch import nn
    m = nn.Sequential(nn.Linear(3, 16), nn.LayerNorm(16), nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Linear(16, 1))
    assert len(m.state_dict()) == Q["state-dict-key-count"]["answer"] == 13
    assert len(list(m.parameters())) == 10


def test_strict_false_silent():
    torch = pytest.importorskip("torch")
    from torch import nn
    src, dst = nn.Linear(3, 2), nn.Linear(3, 2)
    sd = {"module." + k: v for k, v in src.state_dict().items()}
    res = dst.load_state_dict(sd, strict=False)          # 오류 없음
    assert set(res.missing_keys) == {"weight", "bias"}
    assert not torch.equal(dst.weight, src.weight)        # 아무것도 로드되지 않았다
    assert correct("strict-false-silent") == [0]
