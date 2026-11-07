"""18.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 3은 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/language-models/cells.py
"""
# %% cell 1 — 작은 말뭉치로 BPE 병합 규칙 학습
from collections import Counter

corpus = ("reaction reactor reactant react reacted "
          "1990 1995 2000 2005 2010 2015 2020 2020 2021").split()
words = Counter(tuple(w) + ("</w>",) for w in corpus)  # 단어(문자 튜플) → 빈도

def merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1]); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)

merges = []
for _ in range(12):
    pairs = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]):
            pairs[a, b] += n
    best = max(pairs, key=lambda p: (pairs[p], p))      # 빈도 최대, 동률은 사전순으로 고정
    merges.append(best)
    words = Counter({merge(w, best): n for w, n in words.items()})
print("병합 순서:", [a + b for a, b in merges])

def tokenize(word):
    w = tuple(word) + ("</w>",)
    for p in merges:
        w = merge(w, p)
    return list(w)

for w in ["reactors", "2020", "2019", "1920"]:
    print(f"{w:>9} → {tokenize(w)}")

# %% cell 2 — 다음 토큰 예측과 perplexity: n-gram 계수 모델
import numpy as np

rng = np.random.default_rng(0)
V, PERIOD, P_COPY = 4, 8, 0.9                          # A·C·G·T, 주기 8 반복, 돌연변이 10%
def make_seq(n):
    x = rng.integers(V, size=n)                         # (n,)
    for t in range(PERIOD, n):
        if rng.random() < P_COPY:
            x[t] = x[t - PERIOD]
    return x
train, test = make_seq(20_000), make_seq(5_000)

def ngram_nll(k, alpha=0.1):
    """앞 k개 토큰을 문맥으로 쓰는 계수 모델의 평균 음의 로그우도(nat/토큰)"""
    def ctx_ids(x):                                     # 문맥을 정수 하나로 부호화 (n-k,)
        return sum(x[i:len(x) - k + i] * V ** (k - 1 - i) for i in range(k)) if k else np.zeros(len(x), int)
    counts = np.zeros((V ** k, V))
    np.add.at(counts, (ctx_ids(train), train[k:]), 1)
    prob = (counts + alpha) / (counts + alpha).sum(1, keepdims=True)   # add-α 평활화
    nll = lambda x: -np.log(prob[ctx_ids(x), x[k:]]).mean()
    return nll(train), nll(test)

q = P_COPY + (1 - P_COPY) / V                           # 반복 위치의 정답 확률
floor = -(q * np.log(q) + (V - 1) * (1 - q) / V * np.log((1 - q) / V))
print(f"균등 분포 perplexity = {V},  자료 생성 과정의 하한 = {np.exp(floor):.3f}")
for k in [0, 1, 2, 4, 8]:
    tr, te = ngram_nll(k)
    print(f"문맥 {k}  train ppl {np.exp(tr):6.3f}   test ppl {np.exp(te):6.3f}")

# %% cell 3 — PyTorch: 문맥 8을 보는 작은 신경망 언어모델
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
V, PERIOD, P_COPY = 4, 8, 0.9
def make_seq(n):                                        # cell 2와 같은 자료 생성 과정
    x = rng.integers(V, size=n)
    for t in range(PERIOD, n):
        if rng.random() < P_COPY:
            x[t] = x[t - PERIOD]
    return torch.as_tensor(x)
train, test = make_seq(20_000), make_seq(5_000)

def windows(x, k=8):                                    # (n-k, k) 문맥, (n-k,) 다음 토큰
    return x.unfold(0, k, 1)[:-1], x[k:]
Xtr, ytr = windows(train); Xte, yte = windows(test)

model = nn.Sequential(nn.Embedding(V, 8), nn.Flatten(),  # (B, 8) → (B, 8, 8) → (B, 64)
                      nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, V))  # → (B, V) logit
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(1, 1501):
    idx = torch.randint(len(ytr), (256,))
    loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])   # 다음 토큰 예측
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 300, 1500):
        with torch.no_grad():
            ppl = [np.exp(nn.functional.cross_entropy(model(X), y).item()) for X, y in [(Xtr, ytr), (Xte, yte)]]
        print(f"step {step:4d}  train ppl {ppl[0]:.3f}  test ppl {ppl[1]:.3f}")

# %% cell 4 — temperature와 top-k: 같은 logit에서 다른 분포
def decode_dist(logits, T=1.0, k=None):
    z = np.asarray(logits, float) / T                   # (V,)
    if k is not None:
        z = np.where(z >= np.sort(z)[-k], z, -np.inf)   # 상위 k개만 남긴다
    p = np.exp(z - z.max())                             # 안정적인 softmax
    return p / p.sum()

logits = np.array([2.0, 1.0, 0.5, -1.0])                # 다음 토큰 후보 4개
H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()      # 엔트로피(nat)
for T, k in [(1.0, None), (0.5, None), (2.0, None), (1.0, 2)]:
    p = decode_dist(logits, T, k)
    print(f"T={T:3.1f} top-k={str(k):4}  p = {np.round(p, 3)}  H = {H(p):.3f}")

p = decode_dist(logits, 1.0)
draws = np.random.default_rng(1).choice(4, size=10_000, p=p)
print("표본 빈도 (T=1)       :", np.round(np.bincount(draws, minlength=4) / 10_000, 3))
