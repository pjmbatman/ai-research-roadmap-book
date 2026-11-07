"""28.1 실습: 같은 합성 분자 물성 데이터에서 지문 + Ridge와 작은 GNN의 데이터 효율을 비교한다.

데이터는 실제 분자가 아니라 원자가 규칙(C 4, N 3, O 2)을 지키는 무작위 분자 그래프(수소 생략, 고리 0–1개)다.
물성은 합성 규칙으로 정한다.
  - 국소 항: 원자마다 (원소, 차수)에 따른 기여의 합 (group contribution과 같은 꼴)
  - 비국소 항: 그래프 거리 3–4에 있는 헤테로원자(N·O) 쌍마다 +0.5 (분자 내 상호작용을 흉내 낸 항)
  - 잡음: 표준편차 0.1
지문은 반경 0–2 원자 환경의 개수를 1024칸에 해시한 count fingerprint다.

실행: uv run --project labs --group torch python labs/chemistry-and-materials/molecular-representations/fp_vs_gnn.py
"""
import zlib

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

ELEM = ["C", "N", "O"]
VAL = np.array([4, 3, 2])
P_ELEM = np.array([0.7, 0.15, 0.15])


def random_molecule(rng):
    n = int(rng.integers(6, 13))
    el = rng.choice(3, size=n, p=P_ELEM)
    el[0] = 0                                               # 첫 원자는 탄소
    A = np.zeros((n, n), dtype=int)
    for i in range(1, n):                                   # 무작위 트리: 원자가가 남은 원자에 붙인다
        free = [j for j in range(i) if A[j].sum() < VAL[el[j]]]
        if not free or VAL[el[i]] < 1:
            el[i] = 0
            free = [j for j in range(i) if A[j].sum() < VAL[el[j]]]
        j = int(rng.choice(free))
        A[i, j] = A[j, i] = 1
    if rng.random() < 0.5:                                  # 고리 하나 닫기(가능하면)
        cand = [(i, j) for i in range(n) for j in range(i + 2, n)
                if A[i, j] == 0 and A[i].sum() < VAL[el[i]] and A[j].sum() < VAL[el[j]]]
        if cand:
            i, j = cand[int(rng.integers(len(cand)))]
            A[i, j] = A[j, i] = 1
    return el, A


def shortest_paths(A):
    n = len(A)
    D = np.full((n, n), 99)
    np.fill_diagonal(D, 0)
    frontier = np.eye(n, dtype=bool)
    reach = frontier.copy()
    for k in range(1, n):
        frontier = (frontier.astype(int) @ A > 0) & ~reach
        D[frontier] = k
        reach |= frontier
    return D


C_LOCAL = np.random.default_rng(123).normal(size=(3, 5))   # (원소, 차수 0–4) 기여


def prop(el, A, rng):
    deg = A.sum(1)
    local = C_LOCAL[el, deg].sum()
    D = shortest_paths(A)
    het = el > 0
    pairs = np.triu((D >= 3) & (D <= 4) & het[:, None] & het[None, :], 1).sum()
    return local + 0.5 * pairs + 0.1 * rng.normal()


def count_fp(el, A, n_bits=1024, radius=2):
    ids = [f"{ELEM[e]}{d}" for e, d in zip(el, A.sum(1))]
    x = np.zeros(n_bits)
    for r in range(radius + 1):
        for f in ids:
            x[zlib.crc32(f"{r}:{f}".encode()) % n_bits] += 1
        ids = [ids[i] + "(" + ",".join(sorted(ids[j] for j in np.flatnonzero(A[i]))) + ")" for i in range(len(el))]
    return x


class GNN(torch.nn.Module):
    def __init__(self, d=64, layers=4):
        super().__init__()
        self.emb = torch.nn.Linear(8, d)                    # 원소 one-hot(3) + 차수 one-hot(5)
        self.msg = torch.nn.ModuleList(torch.nn.Linear(2 * d, d) for _ in range(layers))
        self.out = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.SiLU(), torch.nn.Linear(d, 1))

    def forward(self, X, A, batch, n_graphs):              # X (N, 8), A (N, N) 블록 대각, batch (N,)
        h = self.emb(X)
        for lin in self.msg:
            h = h + torch.nn.functional.silu(lin(torch.cat([h, A @ h], -1)))
        pooled = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)   # 합 readout (G, d)
        return self.out(pooled).squeeze(-1)


def to_batch(mols):
    Xs, blocks, batch = [], [], []
    for g, (el, A) in enumerate(mols):
        X = np.zeros((len(el), 8))
        X[np.arange(len(el)), el] = 1
        X[np.arange(len(el)), 3 + A.sum(1)] = 1
        Xs.append(X)
        blocks.append(A)
        batch += [g] * len(el)
    return (torch.tensor(np.vstack(Xs), dtype=torch.float32),
            torch.block_diag(*[torch.tensor(b, dtype=torch.float32) for b in blocks]),
            torch.tensor(batch), len(mols))


def train_gnn(mols, y, seed, epochs=300):
    torch.manual_seed(seed)
    model = GNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    yt = torch.tensor(y, dtype=torch.float32)
    mu, sd = yt.mean(), yt.std()
    idx = np.arange(len(mols))
    rng = np.random.default_rng(seed)
    batches = [to_batch([mols[i] for i in chunk]) + (chunk,) for chunk in np.array_split(idx, max(1, len(idx) // 64))]
    for ep in range(epochs):
        for k in rng.permutation(len(batches)):
            X, A, b, G, chunk = batches[k]
            loss = ((model(X, A, b, G) - (yt[chunk] - mu) / sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lambda ms: (model(*to_batch(ms)).detach() * sd + mu).numpy()


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    mols = [random_molecule(rng) for _ in range(4000)]
    y = np.array([prop(el, A, rng) for el, A in mols])
    F = np.array([count_fp(el, A) for el, A in mols])
    test = np.arange(3000, 4000)
    print(f"분자 {len(mols)}개, 물성 표준편차 {y.std():.3f}, 시험 {len(test)}개")
    print(f"{'학습 수':>6s}  {'지문+Ridge MAE':>14s}  {'GNN MAE':>8s}")
    for n in [100, 300, 1000, 3000]:
        tr = np.arange(n)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(F[tr], y[tr])
        mae_fp = np.abs(ridge.predict(F[test]) - y[test]).mean()
        pred = train_gnn([mols[i] for i in tr], y[tr], seed=0, epochs=max(60, 30000 // n))
        mae_gnn = np.abs(pred([mols[i] for i in test]) - y[test]).mean()
        print(f"{n:6d}  {mae_fp:14.3f}  {mae_gnn:8.3f}")


if __name__ == "__main__":
    main()
