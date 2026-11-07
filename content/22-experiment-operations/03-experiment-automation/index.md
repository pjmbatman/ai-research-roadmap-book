---
id: experiment-operations/experiment-automation
title: "실험 자동화와 하이퍼파라미터 탐색"
title_en: "Experiment Automation and Hyperparameter Search"
tier: core
status: draft
prerequisites: [experiment-operations/reproducibility, linux-and-remote/shared-servers]
---

# 실험 자동화와 하이퍼파라미터 탐색

learning rate 세 개, weight decay 두 개, seed 세 개만 조합해도 run이 18개다. 모델 크기와 데이터 증강까지 넣으면 수백 개가 된다. 이 run들을 터미널에서 하나씩 손으로 띄우면 오타 하나로 엉뚱한 설정이 돌고, 어느 조합을 이미 돌렸는지 헷갈리고, 서버가 재부팅되면 어디까지 했는지 모른다.

이 장은 실험을 **자동으로, 예산 안에서, 실패에 강하게** 돌리는 방법을 다룬다. 설정 조합을 job 목록(manifest)으로 만들고, 탐색 알고리즘으로 시도할 설정을 고르고, 비교 대상끼리 같은 예산을 쓰게 하고, 실패와 중단이 있어도 결과가 온전히 모이게 한다. 22.1장의 run 기록과 22.2장의 재현성 설정 위에서, 2.5장의 job scheduler로 실제 자원을 쓰는 단계다.

## 학습 목표

- sweep을 자동화한다
- 하이퍼파라미터 탐색을 예산 안에서 수행한다
- 중단과 실패에 강한 실험을 운영한다

## sweep

**왜 필요한가.** 하이퍼파라미터(hyperparameter)는 학습이 정하지 않고 사람이 정하는 값이다. learning rate, batch size, weight decay, 모델 크기 같은 값이 결과를 크게 바꾸므로, 여러 값을 시도해 비교해야 한다. 이렇게 여러 설정으로 같은 실험을 반복 실행하는 것을 **sweep**이라 한다. sweep을 손으로 돌리면 실수가 생기고 기록이 남지 않는다.

**정의.** sweep은 두 단계로 나눈다.

1. **생성.** 탐색 공간(각 하이퍼파라미터가 가질 수 있는 값)에서 시도할 설정 목록을 만들고, 각 설정에 job ID를 붙여 **job manifest** 파일(한 줄에 job 하나인 JSON Lines 등)로 저장한다.
2. **실행.** runner가 manifest를 읽어 job을 하나씩(또는 scheduler에 병렬로) 실행하고, 결과를 job ID 이름의 파일로 남긴다.

생성과 실행을 나누면 "무엇을 돌리려 했는가"가 실행 전에 파일로 남는다. 실행이 중간에 끊겨도 manifest와 결과 파일을 비교하면 남은 job을 정확히 알 수 있다.

**job ID는 설정의 hash로.** job ID를 목록의 순번(0, 1, 2, …)으로 붙이면, 탐색 공간에 값 하나를 추가하거나 순서를 바꾸는 순간 "job 7"이 다른 설정을 가리킨다. 22.1장의 run ID처럼 설정 내용의 hash를 쓰면 같은 설정은 언제나 같은 ID를 갖는다.

```python {exec="browser" session="sweep"}
import hashlib
import itertools
import json

import numpy as np

def job_id(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

space = {"lr": [1e-4, 1e-3, 1e-2], "wd": [0.0, 1e-4], "seed": [0, 1, 2]}
grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
manifest = [{"job_id": job_id(c), **c} for c in grid]
print(len(manifest), "jobs, job_id 중복 없음:", len({j["job_id"] for j in manifest}) == len(manifest))
print(manifest[0])

# 같은 설정은 목록 순서와 무관하게 같은 job_id를 갖는다
space2 = {"lr": [1e-2, 1e-3, 1e-4, 3e-4], "wd": [0.0, 1e-4], "seed": [0, 1, 2]}   # 값 하나 추가, 순서 바꿈
ids2 = {job_id(dict(zip(space2, v))) for v in itertools.product(*space2.values())}
print("기존 18개가 새 sweep에도 같은 ID로 있다:", {j["job_id"] for j in manifest} <= ids2,
      " 새로 돌릴 job:", len(ids2 - {j["job_id"] for j in manifest}))
```

```output
18 jobs, job_id 중복 없음: True
{'job_id': 'e849abda', 'lr': 0.0001, 'wd': 0.0, 'seed': 0}
기존 18개가 새 sweep에도 같은 ID로 있다: True  새로 돌릴 job: 6
```

lr 값 하나를 추가하고 순서를 바꾼 새 sweep에서도 기존 18개 job은 같은 ID를 유지했다. runner는 결과 파일이 이미 있는 18개를 건너뛰고 새로 생긴 6개만 돌리면 된다. 순번 ID였다면 24개를 모두 다시 돌리거나, 결과를 엉뚱한 설정에 연결했을 것이다.

::quiz{id="manifest-job-id"}

**scheduler와의 연결.** manifest가 있으면 2.5장의 Slurm array job과 바로 연결된다. task 번호 `SLURM_ARRAY_TASK_ID`로 manifest의 몇 번째 줄을 읽을지 정하고, job은 자기 job ID로 run 디렉터리를 만든다. 실패한 job만 다시 제출할 때는 manifest와 결과 파일의 차이로 task 번호 목록을 만든다.

```bash
# 집필 시점 예시: manifest의 줄 수만큼 array task를 만들고, 동시에 8개까지 실행
N=$(wc -l < sweep/manifest.jsonl)
sbatch --array=0-$((N - 1))%8 run_job.sh sweep/manifest.jsonl
# run_job.sh 안: python -m mypkg.run_job --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
```

설정 조합을 만들고 실행하는 일을 대신해 주는 도구도 있다. Hydra의 multirun, W&B sweeps, Ray Tune 등이 설정 조합 생성·병렬 실행·결과 수집을 묶어 제공한다. 기능과 사용법은 자주 바뀌므로 도입할 때 공식 문서로 확인한다(집필 시점 예시). 어느 도구를 쓰든 위의 두 원칙, 즉 실행 전에 job 목록을 남기는 것과 설정에서 job ID를 정하는 것을 지키는지 확인한다.

## 탐색 알고리즘

**왜 필요한가.** 탐색 공간은 금방 커진다. 하이퍼파라미터 5개에 값을 5개씩 두면 조합이 $5^5 = 3125$개다. 모두 시도할 수 없으므로 어떤 설정을 시도할지 고르는 방법이 필요하다.

**grid search와 random search.** grid search는 각 축의 값 목록을 정하고 모든 조합을 시도한다. random search는 각 축의 분포(learning rate는 로그 스케일 균등분포 등)에서 설정을 독립적으로 뽑는다. Bergstra와 Bengio(2012)는 같은 시도 횟수에서 random search가 대개 grid search보다 좋은 설정을 찾는다고 보고했다. 이유는 **중요한 축이 소수**인 경우가 많기 때문이다. 시도 9번을 $3 \times 3$ grid로 쓰면 중요한 축에서는 값 3개만 시도하고, 나머지 6번은 중요하지 않은 축만 바꿔 같은 값을 반복한다. random search 9번은 중요한 축에서도 서로 다른 값 9개를 시도한다.

lr이 결과의 대부분을 정하는 가상의 문제 2000개에서 두 방법을 비교한다. 문제마다 최적 lr의 위치가 다르다.

```python {exec="browser" session="sweep"}
def val_loss(log_lr, log_wd, opt):
    """lr이 결과의 대부분을 정하고 wd는 조금만 영향을 준다"""
    return (log_lr - opt) ** 2 + 0.01 * (log_wd + 4.5) ** 2

g_lr, g_wd = np.meshgrid(np.linspace(-5, -1, 3), np.linspace(-6, -3, 3))   # (3, 3) 격자
rng = np.random.default_rng(0)
regret = {"grid": [], "random": []}
for _ in range(2000):                                # 최적 lr이 어디 있는지 모르는 문제 2000개
    opt = rng.uniform(-5, -1)
    regret["grid"].append(val_loss(g_lr, g_wd, opt).min())
    r_lr, r_wd = rng.uniform(-5, -1, 9), rng.uniform(-6, -3, 9)          # (9,), (9,)
    regret["random"].append(val_loss(r_lr, r_wd, opt).min())
print("시도한 서로 다른 lr 값: grid", len(np.unique(g_lr)), " random", 9)
for k, v in regret.items():
    print(f"{k:6s}: 최적과의 손실 차이 평균 {np.mean(v):.4f}, 90% 분위수 {np.quantile(v, 0.9):.4f}")
print("random이 grid보다 나은 문제의 비율:", np.mean(np.array(regret["random"]) < np.array(regret["grid"])))
```

```output
시도한 서로 다른 lr 값: grid 3  random 9
grid  : 최적과의 손실 차이 평균 0.3376, 90% 분위수 0.8100
random: 최적과의 손실 차이 평균 0.0944, 90% 분위수 0.2412
random이 grid보다 나은 문제의 비율: 0.791
```

같은 9번으로 random search는 최적과의 손실 차이가 평균 약 3.6배 작았고, 문제의 79%에서 grid보다 나은 설정을 찾았다. grid는 최적 lr이 격자 점 사이에 있으면 그 근처를 전혀 시도하지 못한다. 그렇다고 grid가 쓸모없지는 않다. 축이 하나나 둘뿐이고 결과를 그림으로 보고 싶을 때, 또는 범주형 선택지(옵티마이저 종류 등)를 빠짐없이 비교할 때는 grid가 해석하기 쉽다.

::quiz{id="grid-vs-random"}

**베이지안 최적화.** 앞선 시도 결과를 보고 다음 시도를 고르면 더 적은 시도로 좋은 설정에 다가갈 수 있다. 베이지안 최적화는 지금까지의 결과로 목적함수의 대리모델(surrogate model, 예: Gaussian process)을 만들고, 획득함수(acquisition function)가 가장 큰 곳을 다음에 시도한다. 원리와 구현은 10.7장과 13.7장에서 다뤘다. 하이퍼파라미터 탐색 도구에서는 Gaussian process 대신 TPE(Tree-structured Parzen Estimator)처럼 고차원·범주형 축을 다루기 쉬운 대리모델도 널리 쓰인다. 시도 결과를 순서대로 봐야 하므로 병렬성이 제한되고, 처음 몇 번은 random search와 다르지 않다는 점을 기억한다.

**조기 중단.** 하이퍼파라미터 탐색의 비용 대부분은 가망 없는 설정을 끝까지 학습하는 데서 나온다. 학습 곡선이 초반부터 나쁜 설정은 일찍 멈추고, 그 예산을 가망 있는 설정에 돌리면 같은 예산으로 훨씬 많은 후보를 볼 수 있다.

- **successive halving**(Jamieson과 Talwalkar, 2016): 후보 $n$개를 작은 예산 $b$로 학습하고, 상위 $1/\eta$만 남겨 예산을 $\eta$배로 늘리는 과정을 반복한다. 후보 27개, $\eta = 3$, 첫 예산 1 epoch이면 단계마다 27 × 1, 9 × 3, 3 × 9, 1 × 27로 같은 예산(27 epoch)을 쓴다.
- **Hyperband**(Li 외, 2018): successive halving은 "후보를 많이 보고 일찍 자를지, 적게 보고 오래 볼지"를 미리 정해야 한다. Hyperband는 이 비율이 다른 여러 묶음(bracket)을 함께 돌려 그 선택을 피한다.
- **ASHA**(Li 외, 2020): successive halving을 병렬 환경에 맞춘 비동기 버전이다. 단계가 끝나기를 기다리지 않고, 한 단계에 결과가 충분히 쌓이면 그중 상위 후보를 바로 다음 단계로 올린다. worker가 놀지 않는다.

같은 예산 108 epoch로 random search(후보 4개를 끝까지)와 successive halving(후보 27개)을 비교한다. 두 가지 학습 곡선을 쓴다. 하나는 모든 설정이 같은 속도로 내려가는 경우, 다른 하나는 lr이 클수록 빨리 내려가지만 최종 손실은 중간 lr이 가장 좋은 경우다.

```python {exec="browser" session="sweep"}
def curve(log_lr, t, lr_sets_speed):
    """최종 손실은 log_lr = -3 근처가 가장 낮다. lr_sets_speed이면 lr이 클수록 빨리 내려간다"""
    final = 0.2 + 0.05 * (log_lr + 3) ** 2
    speed = 0.3 * 10 ** (log_lr + 3) if lr_sets_speed else 0.3
    return final + (1.0 - final) * np.exp(-speed * t)

def successive_halving(cands, lr_sets_speed, min_budget=1, eta=3):
    used, budget = 0, min_budget
    while True:
        losses = curve(cands, budget, lr_sets_speed)   # 살아남은 후보를 budget epoch만큼 학습
        used += budget * len(cands)
        if len(cands) == 1:
            return cands[0], used
        cands = cands[np.argsort(losses)[: max(1, len(cands) // eta)]]
        budget *= eta

for lr_sets_speed in [False, True]:
    rng = np.random.default_rng(1)
    best_rs, best_sh = [], []
    for _ in range(500):
        cands = rng.uniform(-5, -1, 27)                # 후보 27개, 최대 27 epoch
        best_rs.append(curve(cands[:4], 27, lr_sets_speed).min())   # 4개를 끝까지: 4 × 27 = 108
        pick, used = successive_halving(cands, lr_sets_speed)
        best_sh.append(curve(pick, 27, lr_sets_speed))
    name = "큰 lr이 빨리 내려감" if lr_sets_speed else "내려가는 속도 같음 "
    print(f"{name}: 예산 {used} epoch, 최종 손실 평균  random {np.mean(best_rs):.4f}  SH {np.mean(best_sh):.4f}")
```

```output
내려가는 속도 같음 : 예산 108 epoch, 최종 손실 평균  random 0.2129  SH 0.2007
큰 lr이 빨리 내려감: 예산 108 epoch, 최종 손실 평균  random 0.2282  SH 0.2296
```

결과는 곡선의 모양에 달려 있다.

- **내려가는 속도가 같으면** 1 epoch 후의 순위가 최종 순위와 같다. successive halving은 후보 27개 중 최선을 거의 항상 찾고, 후보 4개만 본 random search보다 낫다(0.2007 대 0.2129).
- **큰 lr이 빨리 내려가면** 초반에는 큰 lr이 앞서므로, 최종적으로 더 좋은 중간 lr 설정이 첫 단계에서 탈락한다. 후보를 6배 넘게 봤는데도 결과가 random search보다 조금 나쁘다(0.2296 대 0.2282).

조기 중단은 "적은 예산에서의 순위가 최종 순위와 비슷하다"는 가정에 기댄다. 작은 learning rate, 강한 정규화, warm-up이 긴 schedule처럼 **늦게 좋아지는**(late bloomer) 설정은 이 가정을 깬다. 첫 단계 예산을 learning rate schedule이 의미 있게 진행될 만큼 잡고, 탈락한 설정 몇 개를 끝까지 학습해 가정이 맞는지 확인한다. Hyperband가 여러 bracket을 함께 돌리는 이유도 이것이다.

::quiz{id="sh-budget"}

::quiz{id="late-bloomer"}

**도구.** Optuna(Akiba 외, 2019)는 탐색 알고리즘(sampler)과 조기 중단(pruner)을 조합해 쓰는 오픈소스 라이브러리다. 학습 코드가 epoch마다 중간 결과를 보고하고, pruner가 멈추라고 하면 예외를 던지는 구조다(집필 시점 예시, 최신 API는 공식 문서에서 확인한다).

```python
# 집필 시점 예시: Optuna로 TPE 탐색 + successive halving 조기 중단
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    model, opt = build(lr, wd)
    for epoch in range(max_epochs):
        val_loss = train_one_epoch(model, opt)
        trial.report(val_loss, step=epoch)
        if trial.should_prune():                    # pruner가 가망 없다고 판단하면
            raise optuna.TrialPruned()
    return val_loss

study = optuna.create_study(direction="minimize", storage="sqlite:///hpo.db",
                            study_name="soh-v1", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=100)
```

`storage`에 데이터베이스를 주면 탐색 기록이 파일로 남아, 중단된 탐색을 이어 가거나 여러 worker가 같은 탐색에 참여할 수 있다. 비교 기준이 되는 random search는 `RandomSampler`와 조기 중단을 하지 않는 `NopPruner`로 같은 코드에서 돌릴 수 있다. 도구가 시도를 대신 골라 주더라도, 각 시도는 22.1장의 run 기록을 남기게 한다. 도구의 데이터베이스만으로는 run 디렉터리의 checkpoint와 설정으로 되돌아가기 어렵다.

## 탐색 예산

**왜 필요한가.** 하이퍼파라미터 탐색은 결과를 바꾼다. 그래서 두 방법을 비교할 때 탐색에 쓴 노력이 다르면, 차이의 일부 또는 전부가 방법이 아니라 튜닝에서 온다. 제안 방법은 100번 튜닝하고 baseline은 원 논문의 기본값으로 한 번 돌리는 비교가 흔하고, 그 결과는 믿기 어렵다.

**사례.** Melis 외(2018)는 언어 모델링에서 여러 구조를 같은 대규모 자동 탐색으로 다시 튜닝하자, 잘 정규화한 표준 LSTM이 더 새로운 구조들을 앞섰다고 보고했다. Lucic 외(2018)는 GAN 여러 종류를 같은 조건에서 비교해, 계산 예산을 충분히 주면 대부분의 방법이 비슷한 점수에 이른다는 것을 보였다. 두 연구 모두 "새 방법의 개선" 중 상당 부분이 튜닝 예산의 차이일 수 있음을 보여 준다.

**공정한 예산의 조건.**

- **같은 시도 횟수, 같은 계산량.** 방법마다 같은 수의 trial과 같은 GPU 시간을 준다. 한 방법의 trial이 훨씬 비싸다면 시간 기준으로 맞춘다.
- **같은 크기의 탐색 공간.** 제안 방법에만 넓은 범위를 주지 않는다. baseline의 범위가 경계에 걸리면(최적값이 범위 끝에 있으면) 범위를 넓혀 다시 탐색한다.
- **같은 선택 규칙.** 모두 검증 세트로 고르고, 시험 세트는 마지막에 한 번만 쓴다(15.4장).
- **선택 후 재평가.** 100번 시도 중 최고 점수는 그 설정의 실력에 운이 더해진 값이라 낙관적으로 치우친다. 고른 설정을 새로운 seed 여러 개로 다시 학습해 보고한다(22.2장).
- **예산의 보고.** 탐색 공간, 탐색 알고리즘, 시도 횟수, 총 계산 시간을 방법마다 표로 남긴다. 이것이 이 장의 통과 기준인 탐색 보고서의 뼈대다.

::quiz{id="fair-budget"}

:::note[분야 예시]
재료 물성 예측에서 새 그래프 신경망과 기존의 기술자(descriptor) 기반 gradient boosting 모델을 비교한다고 하자. 신경망에만 learning rate·층 수·hidden 크기를 탐색하고 gradient boosting은 라이브러리 기본값으로 두면, 표의 차이는 탐색 예산의 차이를 포함한다. 두 모델 모두에 같은 trial 수(예: 50회)와 같은 탐색 알고리즘을 쓰고, 고른 설정을 같은 분할·같은 seed들로 다시 학습해 비교한다. 전통적인 모델이 잘 튜닝되면 격차가 크게 줄어드는 일은 드물지 않다.
:::

## 견고한 실행

**왜 필요한가.** job 수백 개를 며칠 돌리면 노드 장애, 선점, 시간 상한, 저장소 오류가 반드시 섞인다(2.5장). runner가 이런 일에 약하면 결과가 빠지거나, 깨진 결과가 섞이거나, 사람이 매일 아침 실패한 job을 손으로 다시 띄워야 한다.

**네 가지 장치.**

1. **재시도는 일시적 실패에만.** 노드 장애·선점·네트워크 저장소의 일시적 오류는 다시 돌리면 성공할 수 있다. 특정 설정에서 매번 손실이 NaN이 되거나 설정 오타로 예외가 나는 것은 **결정적** 실패라 다시 돌려도 같다. 결정적 실패는 원인과 함께 기록하고 바로 다음 job으로 넘어간다. 재시도 횟수에는 상한을 둔다.
2. **원자적 저장.** 결과를 임시 파일에 다 쓴 뒤 `os.replace`로 이름을 바꾼다. 같은 파일 시스템 안에서 이름 바꾸기는 원자적이라, 다른 프로세스는 이전 파일 또는 완성된 새 파일만 본다. 결과 경로에 직접 쓰다 죽으면 반쯤 쓴 파일이 남고, 재개한 runner가 그것을 끝난 job으로 오인한다.
3. **중간 산출물.** 긴 job은 checkpoint를 주기적으로 저장해 끊긴 지점부터 이어 간다(2.5장, 17.6장). 탐색에서는 조기 중단 단계의 중간 결과(epoch별 검증 손실)도 남겨 둔다. 나중에 조기 중단 가정이 맞았는지 확인할 근거가 된다.
4. **재개(idempotent 실행).** runner를 몇 번 다시 실행해도 같은 최종 상태가 되게 만든다. 결과 파일이 있는 job은 건너뛰고 없는 job만 돌린다. 그러면 "runner가 죽으면 다시 실행한다"가 유일한 복구 절차가 된다.

18개 job을 돌리는 runner가 10번째 job 앞에서 죽었다가 다시 시작하는 상황을 흉내 낸다. job마다 30% 확률로 일시적 실패가 나고, lr = 0.01·wd = 0인 설정은 매번 발산한다.

```python {exec="browser" session="sweep"}
import os
import tempfile
from pathlib import Path

OUT = Path(tempfile.mkdtemp())
fail_rng = np.random.default_rng(2)                 # 노드 장애 같은 일시적 실패를 흉내 낸다

class Transient(Exception):
    pass

def run_job(job):
    if fail_rng.random() < 0.3:
        raise Transient("node lost")
    if job["lr"] == 1e-2 and job["wd"] == 0.0:
        raise FloatingPointError("loss is NaN")     # 설정 때문: 다시 돌려도 같다
    return {"val_loss": round(0.2 + 0.05 * (np.log10(job["lr"]) + 3) ** 2, 4)}

def save_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)                           # 이름 바꾸기는 원자적: 반쯤 쓴 파일이 보이지 않는다

def run_queue(manifest, max_attempts=3, stop_after=None):
    for k, job in enumerate(manifest):
        if k == stop_after:
            return                                  # runner 자체가 죽었다
        path = OUT / f"{job['job_id']}.json"
        if path.exists():
            continue                                # 결과가 있는 job은 건너뛴다
        for attempt in range(1, max_attempts + 1):
            try:
                result = {"status": "finished", **run_job(job)}
                break
            except Transient:
                result = {"status": "failed", "error": "transient"}
            except FloatingPointError as e:
                result = {"status": "failed", "error": str(e)}
                break                               # 결정적 실패는 재시도하지 않는다
        save_atomic(path, {**result, "attempts": attempt})

run_queue(manifest, stop_after=10)
print("중단 시점의 결과 파일:", len(list(OUT.glob("*.json"))))
run_queue(manifest)
results = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
summary = {}
for r in results:
    key = (r["status"], r.get("error", "-"))
    summary[key] = summary.get(key, 0) + 1
print("재개 후:", len(results), "개", dict(sorted(summary.items())))
print("시도 횟수 분포:", {a: sum(r["attempts"] == a for r in results) for a in (1, 2, 3)})
```

```output
중단 시점의 결과 파일: 10
재개 후: 18 개 {('failed', 'loss is NaN'): 3, ('failed', 'transient'): 1, ('finished', '-'): 14}
시도 횟수 분포: {1: 13, 2: 3, 3: 2}
```

runner가 죽은 뒤 다시 실행하자 결과가 있는 10개는 건너뛰고 나머지 8개만 돌려 18개가 모두 모였다. 발산하는 설정 3개(seed 3개)는 NaN이 나오자 더 시도하지 않고 원인과 함께 실패로 기록되었다(그중 하나는 그 전에 일시적 실패를 한 번 겪어 시도가 2번이다). 일시적 실패 때문에 두 번 이상 시도한 job은 5개였고, 그중 1개는 세 번 모두 실패해 `transient`로 남았다. 이 runner는 결과 파일이 있으면 무조건 건너뛰므로, 다음 재개 때도 그 job을 다시 시도하지 않는다. 실제 runner는 `transient` 실패 파일을 재개할 때 다시 시도하도록 하고, 결정적 실패만 확정 실패로 남긴다.

::quiz{id="atomic-save"}

::quiz{id="which-to-retry"}

:::note[분야 예시]
기후 모델 파라미터(구름 미세물리 상수 등)를 보정하는 연구에서는 수십에서 수백 개의 파라미터 조합으로 모델을 돌리는 perturbed parameter ensemble을 만든다. 구성원 하나가 클러스터에서 며칠씩 돌기 때문에 일부는 반드시 시간 상한이나 노드 장애로 끊긴다. 구성원마다 restart 파일을 주기적으로 쓰고, 조합의 hash로 출력 디렉터리를 정하고, 끝난 구성원을 건너뛰는 제출 스크립트를 쓰면, 앙상블 전체를 "다시 제출" 한 번으로 복구할 수 있다. 수치적으로 불안정한 조합은 재시도 대상이 아니라 그 자체로 보정 결과의 일부(그 영역은 물리적으로 허용되지 않는다)가 된다.
:::

## 실습

Optuna 등으로 조기 중단을 포함한 탐색을 수행하고, 같은 예산의 random search와 비교한다.

1. 22.1장 실습의 학습 스크립트에 epoch마다 검증 손실을 보고하는 hook을 단다.
2. 탐색 공간(lr, weight decay, 그리고 하나 더)을 정하고 문서로 남긴다. 로그 스케일로 탐색할 축을 표시한다.
3. 같은 총 epoch 예산으로 두 탐색을 돌린다: (a) TPE + successive halving pruner, (b) random sampler + 조기 중단 없음. 각각 탐색 seed를 3개 이상 바꿔 반복한다.
4. 각 탐색이 고른 설정을 새 seed 5개로 끝까지 다시 학습해 검증 손실의 평균과 퍼짐을 비교한다.
5. pruner가 초반에 탈락시킨 설정 몇 개를 끝까지 학습해, 조기 중단이 놓친 좋은 설정이 있었는지 확인한다.
6. 탐색 공간, 알고리즘, 시도 수, 총 epoch, 고른 설정과 재평가 결과를 탐색 보고서로 정리한다.

Optuna를 쓸 수 없는 환경이라면 셀 3의 `successive_halving`을 실제 학습 함수에 연결해도 된다. 그 함수를 연습문제로 직접 구현한다. 테스트는 예산 계산, 순위 결정, 그리고 늦게 좋아지는 후보를 놓치는 경우를 확인한다.

::exercise{id="successive-halving"}

## 실패 사례

- **제안 방법만 많이 튜닝하는 것.** baseline을 기본값으로 두고 제안 방법에만 넓은 탐색을 하면, 차이가 방법에서 온 것인지 튜닝에서 온 것인지 알 수 없다.
- **탐색 최고 점수를 그대로 보고하는 것.** 수십 번 시도 중 최고 점수는 운이 더해진 값이다. 고른 설정을 새 seed로 다시 학습해 보고한다.
- **시험 세트로 하이퍼파라미터를 고르는 것.** 시험 점수가 선택에 쓰이는 순간 더 이상 일반화 성능의 추정이 아니다(15.4장).
- **탐색 범위의 경계에서 최적값이 나왔는데 그대로 쓰는 것.** 더 좋은 값이 범위 밖에 있을 수 있다. 범위를 넓혀 다시 탐색한다.
- **결정적 실패를 무한히 재시도하는 것.** 발산하는 설정을 계속 다시 돌리면 계산만 낭비하고 실패 기록도 늦어진다.

## 통과 기준

- 탐색 공간·탐색 알고리즘·시도 수·총 계산 예산·결과를 문서화한 탐색 보고서를 작성하고, 비교한 모든 방법에 같은 예산을 썼음을 보고서의 표로 보인다.
- 고른 설정을 새 seed 여러 개로 다시 학습한 결과(평균과 퍼짐)를 보고서에 함께 쓴다.
- sweep runner를 중간에 강제로 끝낸 뒤 다시 실행해, 끝난 job은 건너뛰고 나머지만 돌려 결과가 빠짐없이 모이는 것을 확인한다.
- 연습문제 `successive-halving`의 테스트를 통과한다.

## 참고 문헌

- Bergstra, Bengio (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305. — 같은 시도 횟수에서 random search가 grid search보다 효율적인 이유
- Jamieson, Talwalkar (2016). Non-stochastic best arm identification and hyperparameter optimization. *Proceedings of AISTATS 2016*, PMLR 51, 240–248. — successive halving을 하이퍼파라미터 탐색에 적용
- Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(185), 1–52. — 예산 배분이 다른 successive halving 묶음의 조합
- Li 외 (2020). A system for massively parallel hyperparameter tuning. *Proceedings of Machine Learning and Systems 2 (MLSys 2020)*. — 비동기 successive halving(ASHA)
- Akiba, Sano, Yanase, Ohta, Koyama (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of KDD 2019*, 2623–2631. — sampler·pruner를 조합하는 탐색 프레임워크
- Melis, Dyer, Blunsom (2018). On the state of the art of evaluation in neural language models. *ICLR 2018*. — 같은 탐색 예산으로 다시 튜닝하면 표준 LSTM이 새 구조를 앞설 수 있음
- Lucic, Kurach, Michalski, Gelly, Bousquet (2018). Are GANs created equal? A large-scale study. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. — 충분한 탐색 예산에서 여러 GAN이 비슷한 점수에 이름
