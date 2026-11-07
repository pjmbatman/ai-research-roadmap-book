"""17.9 실습: 디버깅 체크리스트를 pipeline.py에 적용한다.

    uv run --project labs --group torch python labs/deep-learning-pytorch/debugging-training/checklist.py

BUGS 환경변수로 켤 버그를 고른다: all(기본), none, 또는 쉼표로 구분한 이름.
    BUGS=none uv run ... checklist.py
결과 끝에는 버그를 하나씩만 켰을 때 어느 검사가 실패하는지의 표를 출력한다.
"""
import os

import torch
from sklearn.linear_model import LogisticRegression

from pipeline import BUGS, N_CLASSES, build_model, chance_loss, evaluate, load_data, train, train_step


def check_input_scale(bugs):
    Xtr = load_data(bugs)[0]
    mean, std = Xtr.mean(0).abs().max().item(), Xtr.std(0)
    ratio = (std.max() / std.min()).item()
    return mean < 1 and ratio < 10, f"|평균| 최대 {mean:.3g}, 표준편차 비 {ratio:.3g}"


def check_label_baseline(bugs):
    Xtr, ytr = load_data(bugs)[:2]
    X = (Xtr - Xtr.mean(0)) / Xtr.std(0)                                   # 검사용으로만 표준화
    acc = LogisticRegression(max_iter=1000).fit(X.numpy(), ytr.numpy()).score(X.numpy(), ytr.numpy())
    majority = torch.bincount(ytr).max().item() / len(ytr)
    return acc > majority + 0.2, f"선형 baseline train 정확도 {acc:.3f} (최다 class 비율 {majority:.3f})"


def check_initial_loss(bugs):
    Xtr, ytr = load_data(bugs)[:2]
    model = build_model(bugs)
    model.eval()
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(Xtr), ytr).item()
    return abs(loss - chance_loss()) < 0.3, f"초기 손실 {loss:.3f} (ln {N_CLASSES} = {chance_loss():.3f})"


def check_overfit_one_batch(bugs):
    Xtr, ytr = load_data(bugs)[:2]
    xb, yb = Xtr[:16], ytr[:16]
    model = build_model(bugs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        train_step(model, opt, xb, yb, bugs)
    model.eval()                                                           # 검사 자체는 dropout 없이
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(xb), yb).item()
    return loss < 0.05, f"batch 16개를 300 step 학습한 뒤 손실 {loss:.3g}"


def check_grad_reset(bugs):
    Xtr, ytr = load_data(bugs)[:2]
    model = build_model(bugs)
    model[2].p = 0.0                                                       # 비교를 위해 dropout을 끈다
    opt = torch.optim.SGD(model.parameters(), lr=0.0)                      # 파라미터는 그대로 둔다
    grads = []
    for _ in range(2):
        train_step(model, opt, Xtr[:32], ytr[:32], bugs)
        grads.append(model[0].weight.grad.clone())
    rel = ((grads[1] - grads[0]).norm() / grads[0].norm()).item()
    return rel < 1e-6, f"같은 batch 두 번째 step의 gradient 상대 차이 {rel:.3g}"


def check_eval_deterministic(bugs):
    Xtr, ytr, Xva, yva = load_data(bugs)
    model = build_model(bugs)
    a, b = evaluate(model, Xva, yva, bugs), evaluate(model, Xva, yva, bugs)
    return a == b, f"같은 모델을 두 번 평가한 손실 {a[0]:.4f}, {b[0]:.4f}"


CHECKS = [("1 입력 스케일", check_input_scale), ("2 label 정합성(선형 baseline)", check_label_baseline),
          ("3 초기 손실 ≈ ln C", check_initial_loss), ("4 batch 하나 과적합", check_overfit_one_batch),
          ("5 gradient 초기화", check_grad_reset), ("6 평가의 결정성", check_eval_deterministic)]


def parse(env):
    if env in ("", "all"):
        return set(BUGS)
    return set() if env == "none" else set(env.split(","))


def main():
    bugs = parse(os.environ.get("BUGS", "all"))
    print(f"켠 버그: {sorted(bugs) or '없음'}")
    for name, fn in CHECKS:
        ok, evidence = fn(bugs)
        print(f"  {'통과' if ok else '실패'}  {name:28s} {evidence}")
    loss, acc = train(bugs)
    print(f"  학습 결과: 검증 손실 {loss:.3f}, 검증 정확도 {acc:.3f}")
    if os.environ.get("TABLE", "1") == "0":
        return

    print("\n버그를 하나씩만 켰을 때 실패하는 검사")
    for bug in BUGS:
        failed = [name.split()[0] for name, fn in CHECKS if not fn({bug})[0]]
        loss, acc = train({bug})
        print(f"  {bug:15s} 실패한 검사 {', '.join(failed) or '없음':8s}  검증 손실 {loss:.3f}  검증 정확도 {acc:.3f}")


if __name__ == "__main__":
    main()
