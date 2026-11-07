"""16.5 실습: 2차원 embedding에서 보이는 분리가 실제 하위 과제의 일반화와 연결되는지 검증한다.

합성 데이터: 측정 장비(batch) 3대가 특징 대부분에 큰 오프셋을 더하고, 예측하려는 label은
특징 두 개의 약한 조합에만 들어 있다. t-SNE 그림에서 뚜렷한 군집이 무엇에 대응하는지,
그 군집이 label 예측의 일반화와 관련이 있는지 독립적인 지표로 확인한다.

UMAP(umap-learn)은 실습 환경에 없으므로 같은 계열의 t-SNE를 쓴다. umap-learn을 설치했다면
`TSNE(...)`를 `umap.UMAP(...)`로 바꿔 같은 검증을 반복할 수 있다.

실행: uv run --project labs python labs/classical-ml/unsupervised-learning/embedding_check.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_PER_BATCH, N_FEAT = 200, 20


def make_data(rng):
    batch = np.repeat(np.arange(3), N_PER_BATCH)                     # (N,) 측정 장비
    n = len(batch)
    X = rng.normal(size=(n, N_FEAT))                                 # (N, D)
    offsets = rng.normal(0, 3.0, size=(3, N_FEAT))                   # 장비마다 다른 오프셋
    offsets[:, :2] = 0.0                                             # label 정보가 있는 특징에는 오프셋 없음
    X += offsets[batch]
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1]                            # label은 특징 0, 1에만 의존
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)  # (N,)
    return X, y, batch


def main():
    rng = np.random.default_rng(SEED)
    X, y, batch = make_data(rng)
    print(f"샘플 {len(y)}개, 특징 {N_FEAT}개, 장비 3대, label 양성 비율 {y.mean():.2f}")

    Xs = StandardScaler().fit_transform(X)
    E = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Xs)   # (N, 2)

    print("\n1. 그림의 군집은 무엇에 대응하는가 (silhouette, 2차원 embedding 기준)")
    print(f"   장비로 묶을 때 {silhouette_score(E, batch):+.2f}   label로 묶을 때 {silhouette_score(E, y):+.2f}")

    print("\n2. label 예측 정확도 (5-fold 층화 CV / 장비 단위 leave-one-group-out)")
    knn = KNeighborsClassifier(n_neighbors=15)
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gcv = GroupKFold(3)
    rows = [
        ("kNN, t-SNE 2차원 좌표", knn, E),
        ("kNN, 원래 특징 20개", make_pipeline(StandardScaler(), knn), X),
        ("로지스틱 회귀, 원래 특징 20개", logreg, X),
    ]
    for name, model, feats in rows:
        a = cross_val_score(model, feats, y, cv=cv).mean()
        g = cross_val_score(model, feats, y, cv=gcv, groups=batch).mean()
        print(f"   {name:24s}  층화 CV {a:.3f}   새 장비 {g:.3f}")
    print(f"   (다수 클래스만 예측할 때 {max(y.mean(), 1 - y.mean()):.3f})")

    print("\n3. 장비 오프셋을 장비별 중심화로 지운 뒤 다시 그린 embedding")
    Xc = X.copy()
    for b in range(3):
        Xc[batch == b] -= Xc[batch == b].mean(axis=0)               # 장비별 평균 제거
    Ec = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(
        StandardScaler().fit_transform(Xc))
    print(f"   장비로 묶을 때 {silhouette_score(Ec, batch):+.2f}   label로 묶을 때 {silhouette_score(Ec, y):+.2f}")


if __name__ == "__main__":
    main()
