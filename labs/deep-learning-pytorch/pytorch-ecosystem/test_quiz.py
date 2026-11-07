"""17.10 퀴즈(content/17-deep-learning-pytorch/10-pytorch-ecosystem/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/10-pytorch-ecosystem/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_compile_graph_break():
    torch = pytest.importorskip("torch")
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def f(x):
        y = torch.sin(x) * 2
        if y.sum() > 0:
            return y + 1
        return y - 1

    cf = torch.compile(f, backend=backend)
    for seed in range(4):
        x = torch.randn(8, generator=torch.Generator().manual_seed(seed))
        assert torch.allclose(cf(x), f(x))                 # 결과는 eager와 같다
    assert len(graphs) >= 2                                # 분기에서 그래프가 나뉜다
    assert correct("compile-graph-break") == [0]


def test_vmap_per_sample():
    torch = pytest.importorskip("torch")
    from torch.func import functional_call, grad, vmap
    model = torch.nn.Linear(3, 2)
    params = {k: v.detach() for k, v in model.named_parameters()}
    x, y = torch.randn(6, 3), torch.randn(6, 2)

    def loss_one(p, xi, yi):
        return ((functional_call(model, p, (xi[None],)) - yi) ** 2).mean()

    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    assert g["weight"].shape == (6, 2, 3)
    full = torch.func.grad(lambda p: torch.stack([loss_one(p, x[i], y[i]) for i in range(6)]).mean())(params)
    assert torch.allclose(g["weight"].mean(0), full["weight"], atol=1e-6)   # 평균하면 batch gradient
    assert correct("vmap-per-sample") == [0]


def test_choice_keys():
    assert correct("compile-benchmark") == [0]
    assert correct("jax-pure-update") == [0]
    assert correct("library-selection") == [0, 1, 2]
