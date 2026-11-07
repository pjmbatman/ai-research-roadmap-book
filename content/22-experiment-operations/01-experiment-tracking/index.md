---
id: experiment-operations/experiment-tracking
title: "실험 추적과 결과 관리"
title_en: "Experiment Tracking and Results Management"
tier: core
status: draft
prerequisites: [research-code-engineering/configuration, deep-learning-pytorch/dashboards]
---

# 실험 추적과 결과 관리

논문 표의 숫자 하나를 가리키며 "이 값은 어떤 코드로, 어떤 데이터로, 어떤 설정으로 나왔는가"라고 물었을 때 바로 답할 수 있어야 한다. 몇 주 뒤의 자신, 공동 저자, 심사위원이 모두 이 질문을 한다. 실험이 수십 개일 때는 기억과 노트로 버틸 수 있지만, 수백 개가 되면 기록이 체계적이지 않은 연구실은 같은 실험을 다시 돌리거나, 어느 run이 표의 숫자를 만들었는지 찾지 못하거나, 손으로 옮긴 숫자가 틀린 채로 논문을 낸다.

이 장은 실험을 **추적 가능하게** 만드는 방법을 다룬다. 6.3장이 run 하나의 설정을 기록하는 법을, 17.8장이 학습 곡선을 보는 법을 다뤘다면, 이 장은 run 수백 개를 하나의 체계로 묶는다. 무엇을 기록할지, 디스크에 어떻게 놓을지, 추적 도구를 어디에 쓸지, 결과 표를 어떻게 자동으로 만들지, 실패하고 중단된 run을 어떻게 다룰지가 차례로 나온다. 원리는 하나다. **결과에서 실행으로 되돌아가는 경로가 끊기지 않게 한다.**

## 학습 목표

- 실험을 추적 가능하게 기록한다
- 결과와 artifact를 연결한다

## 무엇을 기록하는가

**왜 필요한가.** 결과 수치는 여러 입력의 함수다. 코드, 설정, 데이터, 분할, 난수, 실행 환경 중 하나만 달라도 숫자가 바뀐다. 이 중 하나라도 기록되지 않았다면, 숫자가 달라졌을 때 원인을 가를 수 없고 같은 숫자를 다시 만들 수도 없다.

**정의.** 실험 추적(experiment tracking)은 실행(run) 하나마다 그 실행의 입력과 출력을 함께 남기고, 실행끼리 비교할 수 있게 모아 두는 일이다. run 하나가 남길 것은 다음과 같다.

| 항목 | 내용 | 다루는 곳 |
|---|---|---|
| run ID | 실행을 가리키는 고유한 이름. 시작 시각 + 설정 hash | 6.1장 |
| resolved config | 기본값·설정 파일·명령행 override를 모두 반영한 최종 설정 | 6.3장 |
| 코드 revision | 커밋 해시, 커밋되지 않은 변경 여부와 그 diff | 4부 |
| 데이터 버전 | 데이터셋 manifest 해시 | 15.5장 |
| 분할 | split manifest 해시(어느 샘플이 train·val·test인가) | 15.4–15.5장 |
| seed | 실험이 쓴 모든 난수 생성기의 seed | 22.2장 |
| 환경 | Python·주요 패키지 버전, 장치, 호스트, 컨테이너 image digest | 3.6장 |
| 실행 정보 | 명령행 전체, 시작·종료 시각, 상태(진행·완료·실패) | 이 장 |
| 출력 | epoch별 metric, 최종 metric, checkpoint, 예측값 | 이 장 |

설정·코드·데이터·분할·seed·환경은 **입력**이고, 이들이 같으면 결과도 같아야 한다(어디까지 같아야 하는지는 22.2장에서 다룬다). 그래서 입력은 실행이 시작되기 **전에** 기록한다. 실행이 중간에 죽어도 무엇을 돌리려 했는지는 남아야 하기 때문이다(6.3장).

**직관.** 기록의 방향은 두 가지다. "이 run은 무엇을 만들었나"(실행 → 결과)는 대부분 자연스럽게 남는다. 어려운 쪽은 반대 방향 "이 숫자는 어느 run에서 왔나"(결과 → 실행)다. 숫자가 run 디렉터리를 떠나 표·그림·슬라이드로 옮겨지는 순간 출처가 떨어져 나가기 쉽다. 그래서 결과가 이동할 때마다 run ID를 함께 옮긴다.

**코드.** run 하나를 시작하고 metric을 기록하는 최소 구현이다. 브라우저에서는 git을 쓸 수 없으므로 코드 revision과 데이터 해시는 인자로 받는다. 실제 코드에서는 `git rev-parse HEAD`와 `git status --porcelain`의 결과, 15.5장의 manifest 해시를 넣는다.

```python {exec="browser" session="track"}
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp())
IGNORED = {"out_dir", "log_level"}                 # 결과에 영향을 주지 않는 키

def config_hash(cfg):
    core = {k: v for k, v in cfg.items() if k not in IGNORED}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]

def start_run(root, cfg, started, code_rev, data_hash):
    run_id = f"{started}-{config_hash(cfg)}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)                    # 이미 있으면 오류: 덮어쓰지 않는다
    meta = {"run_id": run_id, "config": cfg, "code_rev": code_rev, "data_hash": data_hash}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "status").write_text("running")
    return run_dir

def log_metrics(run_dir, **values):                # 한 줄에 JSON 객체 하나(JSON Lines)
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(values) + "\n")

cfg = {"lr": 0.03, "seed": 0, "epochs": 5, "out_dir": "outputs"}
run_dir = start_run(ROOT / "demo", cfg, "20261107-1403", code_rev="3f2a9c1", data_hash="sha256:9e01")
log_metrics(run_dir, epoch=1, train_loss=0.912, val_loss=0.954)
log_metrics(run_dir, epoch=2, train_loss=0.611, val_loss=0.688)
print(run_dir.name, sorted(p.name for p in run_dir.iterdir()))
print((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
```

```output
20261107-1403-dc2dd4b6 ['metrics.jsonl', 'run.json', 'status']
{"epoch": 2, "train_loss": 0.611, "val_loss": 0.688}
```

세 가지 설계를 눈여겨본다.

- **run ID는 시작 시각과 설정 hash로 만든다.** hash는 키를 정렬해 직렬화한 뒤 계산하므로 키 순서와 무관하고, `out_dir`처럼 결과에 영향을 주지 않는 키는 뺀다(6.1장). seed는 결과를 바꾸므로 hash에 남는다.
- **`mkdir`이 이미 있는 디렉터리에서 실패하게 둔다.** 같은 run ID에 두 번 쓰면 이전 결과를 덮어쓰는 대신 오류가 난다.
- **metric은 한 줄에 JSON 객체 하나씩 덧붙인다.** 이 형식을 JSON Lines라 한다. 아래 "저장 구조"에서 이유를 본다.

::quiz{id="what-to-record"}

**연구에서의 의미.** 기록 항목 중 가장 자주 빠지는 것은 "커밋되지 않은 변경"과 "분할"이다. 커밋 해시만 남기고 작업 디렉터리의 수정을 남기지 않으면 그 코드는 복원할 수 없다. 분할을 코드와 seed로만 기록하면 데이터가 늘었을 때 같은 seed가 다른 분할을 만든다(15.5장). 두 항목은 run 시작 시 자동으로 확인해서, 변경이 있으면 diff를 run 디렉터리에 저장하거나 실행을 거부하게 만든다.

## 저장 구조

**왜 필요한가.** 기록 항목을 정했어도 그것들이 흩어져 있으면 소용이 없다. 설정은 한 폴더에, 로그는 다른 폴더에, checkpoint는 scratch에, 예측값은 노트북 옆에 있으면, run 하나를 온전히 모으는 데만 반나절이 걸린다.

**정의.** run 하나의 모든 기록과 산출물(artifact)을 `outputs/<run-id>/` 디렉터리 하나에 모은다(6.1장). 전형적인 구성은 다음과 같다.

```text
outputs/20261107-1500-61439ac5/
├── run.json          # run ID, resolved config, 코드 revision, 데이터·분할 해시, 명령행
├── env.json          # 환경 보고서(3.6장)
├── git.diff          # 커밋되지 않은 변경(있을 때만)
├── status            # running · finished · failed
├── train.log         # 사람이 읽는 로그(6.5장)
├── metrics.jsonl     # epoch·step별 metric, 한 줄에 하나
├── checkpoints/      # last.pt, best.pt (17.6장)
└── predictions/      # 평가 세트의 예측값: 표를 다시 계산하는 근거
```

**JSON Lines.** metric 파일을 리스트 전체를 담은 JSON 하나로 두고 epoch마다 다시 쓰면, 쓰는 도중 프로세스가 죽었을 때 파일 전체가 깨진 JSON이 되어 앞선 기록까지 읽을 수 없다. JSON Lines는 덧붙이기만 하므로, 죽더라도 잃는 것은 많아야 마지막 한 줄이다. 읽는 쪽은 줄 단위로 파싱하고, 마지막 줄이 잘려 있으면 그 줄만 버리면 된다. 학습이 도는 동안 다른 프로세스가 파일을 읽어 그래프를 그리기도 쉽다.

::quiz{id="jsonl-append"}

**artifact의 연결.** checkpoint와 예측값은 그것을 만든 run 디렉터리 안에 두는 것이 기본이다. 크기 때문에 다른 저장소(공유 스토리지, 객체 저장소)에 두어야 한다면, run 디렉터리에는 그 위치와 파일 해시를 적은 작은 파일을 남긴다. 반대로 artifact 쪽에도 run ID를 남긴다. checkpoint 파일 안에 run ID와 설정을 함께 저장하면(17.6장), 파일 하나만 전달받은 사람도 출처를 찾을 수 있다.

**한 번 쓰고 고치지 않는다.** run 디렉터리는 그 실행이 만든 기록이다. 나중에 평가를 다시 하거나 표를 다시 계산한 결과는 원래 run 디렉터리를 고치지 않고, 새 run(평가 run)으로 남기거나 분석 결과 디렉터리에 원본 run ID를 적어 둔다. 원본이 바뀌면 "그때 무엇이 나왔는가"를 다시는 확인할 수 없다.

:::note[분야 예시]
기후 모델 앙상블 실험에서는 구성원 하나가 수십 GB의 출력을 만든다. 출력은 고성능 파일 시스템에 두고, run 디렉터리에는 출력 경로·파일 해시·모델 설정(해상도, 물리 과정 선택, 강제력 시나리오)·초기 조건 번호를 적은 메타데이터만 둔다. 분석 그림을 만든 스크립트는 입력으로 쓴 구성원의 run ID 목록을 그림 파일 옆에 저장한다. 그러면 그림 속 앙상블 평균이 어느 구성원들의 평균인지 나중에도 확인할 수 있다.
:::

## 추적 도구

**왜 필요한가.** 디렉터리 구조만으로도 추적은 된다. 그러나 run이 수백 개가 되면 "lr이 0.1인 run 중 검증 손실이 가장 낮은 것"을 찾거나 학습 곡선 열 개를 겹쳐 보는 일을 매번 스크립트로 하기는 번거롭다. 추적 도구는 run의 기록을 색인하고, 검색·비교·시각화 화면을 제공한다.

**역할의 구분.** 자주 쓰이는 도구는 성격이 다르다. 기능과 요금 정책은 자주 바뀌므로 아래는 집필 시점의 일반적인 구분이고, 도입 전에 공식 문서로 확인한다.

| 도구 | 성격 | 잘하는 일 | 따져 볼 점 |
|---|---|---|---|
| TensorBoard | 로컬 event 파일 + 보기 도구 | 학습 곡선, 히스토그램, 이미지를 학습 중에 보기(17.8장) | run 간 설정 비교와 검색 기능은 제한적 |
| MLflow | 오픈소스 추적 서버(로컬 파일로도 동작) | 파라미터·metric·artifact 기록, run 검색과 비교, 모델 등록 | 서버를 직접 운영하면 백업·접근 권한 관리가 필요 |
| Weights & Biases(W&B) | 호스팅 서비스(SaaS) | 팀 공유, 대시보드, sweep 관리, 보고서 | 기록이 기관 밖 서버로 나간다. 데이터·코드 반출 정책 확인 |

같은 기록을 MLflow와 W&B로 남기는 코드는 다음과 같다(집필 시점 예시, 최신 API는 각 공식 문서에서 확인한다).

```python
# 집필 시점 예시: MLflow
import mlflow
mlflow.set_experiment("battery-soh")
with mlflow.start_run(run_name=run_id):
    mlflow.log_params(cfg)                          # resolved config
    mlflow.set_tag("code_rev", code_rev)
    for epoch, val_loss in history:
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_artifact(str(run_dir / "run.json"))

# 집필 시점 예시: W&B
import wandb
run = wandb.init(project="battery-soh", name=run_id, config=cfg)
for epoch, val_loss in history:
    wandb.log({"val_loss": val_loss}, step=epoch)
run.finish()
```

**원천은 파일, 도구는 색인.** 어느 도구를 쓰든 이 장의 run 디렉터리를 **원천**(source of truth)으로 두고 도구를 그 위의 색인·보기 층으로 쓰는 것을 권한다. 이유는 셋이다.

1. **도구는 바뀐다.** 연구실이 도구를 바꾸거나, 서비스 요금제가 바뀌거나, 서버가 사라져도 run 디렉터리는 남는다. 논문은 몇 년 뒤에도 재현을 요구받는다.
2. **도구 화면의 숫자는 가공된 숫자일 수 있다.** 학습 곡선의 smoothing, 다운샘플링된 점, 화면에 맞춘 반올림이 끼어든다. 표의 숫자는 원본 파일에서 뽑는다.
3. **오프라인·클러스터 환경.** 계산 노드에서 외부 네트워크가 막힌 클러스터(2.5장)에서는 기록을 먼저 파일로 남기고 나중에 동기화해야 한다. 많은 도구가 오프라인 모드를 지원하지만, 그 경우에도 파일이 먼저다.

::quiz{id="dashboard-vs-raw"}

**도입 판단.** 혼자 하는 작은 프로젝트라면 run 디렉터리와 결과 표 스크립트만으로 충분하고, 학습 곡선은 TensorBoard로 본다. 여러 사람이 같은 프로젝트의 run을 비교해야 하거나 sweep이 수백 개를 넘으면 MLflow나 W&B 같은 도구가 시간을 아낀다. 어느 쪽이든 run ID를 도구의 run 이름으로 써서, 도구 화면에서 본 run을 디스크의 디렉터리로 곧바로 찾아갈 수 있게 한다.

## 결과 표 만들기

**왜 필요한가.** 논문 표의 숫자를 대시보드에서 손으로 옮기면 세 종류의 오류가 생긴다. 다른 run의 숫자를 옮기고, 자릿수를 잘못 옮기고, 나중에 실험을 다시 돌렸을 때 표를 갱신하지 않는다. 계산 연구의 재현성 규칙을 정리한 Sandve 외(2013)도 "모든 결과가 어떻게 만들어졌는지 추적하라", "손으로 하는 데이터 조작 단계를 피하라"를 첫 두 규칙으로 꼽는다.

**정의.** 결과 표는 **스크립트가 run 디렉터리들을 읽어 만든다.** 스크립트는 (1) run을 모으고, (2) 상태를 확인하고, (3) 설정으로 묶어(seed는 묶음 키에서 뺀다) 평균과 퍼짐을 계산하고, (4) 표의 각 칸에 그 칸을 만든 run ID 목록을 함께 남긴다. 표를 다시 만드는 일은 명령 하나로 끝나야 한다.

학습률 세 개 × seed 세 개의 sweep을 돌린다. 학습은 설정과 seed로 정해지는 가짜 손실로 대신한다. 이 sweep에서 run 하나는 학습률이 커서 발산하고, 하나는 4번째 epoch에서 프로세스가 죽는다(노드 장애나 시간 상한을 흉내 낸 것이다).

```python {exec="browser" session="track"}
import numpy as np

def fake_train(cfg, run_dir, stop_at=None):
    """학습 대신: 설정과 seed로 정해지는 검증 손실을 epoch마다 기록한다"""
    rng = np.random.default_rng(cfg["seed"])
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == stop_at:
            return                                 # 프로세스가 죽었다: status를 고칠 기회가 없다
        val_loss = 0.30 + np.exp(-30 * cfg["lr"] * epoch) + 0.02 * rng.standard_normal()
        if cfg["lr"] >= 0.1 and rng.random() < 0.02:
            log_metrics(run_dir, epoch=epoch, val_loss=7.9)   # 발산한 손실이 기록된다
            (run_dir / "status").write_text("failed")
            return
        log_metrics(run_dir, epoch=epoch, val_loss=round(float(val_loss), 4))
    (run_dir / "status").write_text("finished")

SWEEP = ROOT / "sweep"
for lr in [0.01, 0.03, 0.1]:
    for seed in [0, 1, 2]:
        cfg = {"lr": lr, "seed": seed, "epochs": 5, "out_dir": "outputs"}
        d = start_run(SWEEP, cfg, "20261107-1500", code_rev="3f2a9c1", data_hash="sha256:9e01")
        fake_train(cfg, d, stop_at=4 if (lr, seed) == (0.03, 2) else None)

status = [(d / "status").read_text() for d in sorted(SWEEP.iterdir())]
print({s: status.count(s) for s in sorted(set(status))})
```

```output
{'failed': 1, 'finished': 7, 'running': 1}
```

끝난 run은 7개뿐이다. 이제 표를 만든다. 비교를 위해 상태를 보지 않고 각 run의 마지막 metric을 평균한 "단순 평균"도 함께 계산한다.

```python {exec="browser" session="track"}
import pandas as pd

def load_runs(root):
    rows = []
    for d in sorted(root.iterdir()):
        meta = json.loads((d / "run.json").read_text())
        last = json.loads((d / "metrics.jsonl").read_text().splitlines()[-1])
        rows.append({"run_id": meta["run_id"], "lr": meta["config"]["lr"],
                     "status": (d / "status").read_text(), "val_loss": last["val_loss"]})
    return pd.DataFrame(rows)

runs = load_runs(SWEEP)
naive = runs.groupby("lr")["val_loss"].mean()                       # 상태를 보지 않은 평균
done = runs[runs["status"] == "finished"]
table = done.groupby("lr")["val_loss"].agg(["mean", "std", "count"])
print("   lr   단순 평균 | 끝난 run 평균 ± 표준편차  n  빠짐")
for lr, r in table.iterrows():
    missing = int((runs["lr"] == lr).sum() - r["count"])
    print(f"{lr:5.2f}   {naive[lr]:8.4f} | {r['mean']:.4f} ± {r['std']:.4f}  {int(r['count'])}  {missing}")
```

```output
   lr   단순 평균 | 끝난 run 평균 ± 표준편차  n  빠짐
 0.01     0.5376 | 0.5376 ± 0.0236  3  0
 0.03     0.3295 | 0.3148 ± 0.0204  2  1
 0.10     2.8376 | 0.3065 ± 0.0012  2  1
```

같은 기록에서 두 표가 전혀 다른 결론을 낸다.

- **단순 평균**에서는 lr = 0.1이 가장 나빠 보인다(2.8376). 발산한 run이 마지막으로 남긴 손실 7.9가 평균에 섞였기 때문이다. lr = 0.03의 단순 평균(0.3295)에도 3번째 epoch까지만 기록하고 멈춘 run의 중간 손실이 섞여 있다.
- **끝난 run만 쓴 표**에서는 lr = 0.1이 가장 좋다(0.3065). 그러나 세 run 중 하나가 발산했다는 사실이 "빠짐" 열에 남는다. 이 열이 없으면 독자는 lr = 0.1이 늘 안정적이라고 읽는다.

lr = 0.1과 0.03의 차이(0.3065 대 0.3148)는 run 두 개씩으로 판정하기에는 작다. 차이가 seed 사이의 퍼짐과 비슷한 크기라면, 더 많은 seed를 돌리고 짝지은 비교(12.5장)를 쓴 뒤에 결론을 낸다.

이 셀에는 빈틈이 하나 남아 있다. 표의 행을 끝난 run에서 만들기 때문에, 모든 seed가 실패한 설정은 표에서 통째로 사라진다. 가장 불안정한 설정이 가장 조용히 숨는 셈이다. 실습의 연습문제에서 이 빈틈을 막는다.

**표에 run ID를 남긴다.** 논문에 들어가는 표는 보통 평균만 보이지만, 표를 만드는 스크립트는 칸마다 run ID 목록을 별도 파일(CSV나 JSON)로 함께 쓴다. 표를 LaTeX나 Markdown으로 내보내는 것도 같은 스크립트가 한다. 그러면 표의 어느 숫자에서든 run ID → run 디렉터리 → 설정·코드·데이터로 되돌아갈 수 있다. 실습에서 이 경로를 실제로 따라간다.

:::note[분야 예시]
신약 후보 분자의 결합 친화도를 예측하는 모델을 데이터 분할 방식(무작위, scaffold 기준) × 모델 세 종류 × seed 다섯 개로 비교한다고 하자. 90개 run의 결과를 손으로 표에 옮기면 어느 칸이 어느 분할에서 나왔는지 헷갈리기 쉽다. 스크립트가 run 기록의 분할 해시와 모델 이름으로 묶어 표를 만들면, 분할 방식이 같은지 다른지를 사람이 아니라 기록이 보장한다.
:::

## 실패와 중단

**왜 필요한가.** sweep을 크게 돌리면 실패와 중단은 예외가 아니라 일상이다. 학습이 발산하고, 메모리가 부족하고, 노드가 죽고, scheduler의 시간 상한에 걸린다(2.5장). 문제는 실패 자체보다 **실패가 기록에서 성공처럼 보이는 것**이다. 위의 단순 평균이 그 예다.

**상태의 생명주기.** run은 다음 상태 중 하나다.

| 상태 | 의미 | 누가 기록하나 |
|---|---|---|
| `running` | 시작했고 아직 끝나지 않았다 | 시작할 때 학습 코드 |
| `finished` | 정상적으로 끝났다 | 끝날 때 학습 코드 |
| `failed` | 오류로 끝났다. 원인(예외 메시지, traceback)을 함께 남긴다 | 예외 처리 코드 |
| 중단(stale) | `running`인데 더 이상 진행하지 않는다 | 학습 코드가 기록할 수 없다. 읽는 쪽이 판정한다 |

마지막 상태가 까다롭다. 프로세스가 SIGKILL로 죽거나 노드가 꺼지면 학습 코드는 상태를 고칠 기회가 없다. 셀 2의 중단된 run이 그렇게 `running`으로 남았다. 그래서 학습 코드는 주기적으로 **heartbeat**(마지막으로 살아 있던 시각)를 파일에 쓰고, 결과를 모으는 쪽은 heartbeat가 일정 시간 이상 갱신되지 않은 `running` run을 중단된 것으로 판정한다. scheduler를 쓴다면 job 상태(`sacct`)와 run 상태를 대조하는 것도 방법이다.

**재개와 run ID.** 중단된 run을 checkpoint에서 이어 갈 때(2.5장, 17.6장)는 두 가지 선택이 있다. 같은 run ID로 이어 쓰거나, 새 run ID를 만들고 `resumed_from`에 원래 run ID를 적는 것이다. 어느 쪽이든 재개했다는 사실과 재개한 step을 기록에 남긴다. 재개는 데이터 순서·난수 상태가 끊기지 않은 run과 미묘하게 다를 수 있기 때문이다(22.2장).

**실패를 지우지 않는다.** 실패한 run 디렉터리를 지우면 저장 공간은 늘지만, 그 설정이 실패했다는 정보가 사라진다. 같은 설정을 다시 돌려 같은 실패를 겪게 되고, 결과 표의 "빠짐" 열을 만들 근거도 없어진다. 큰 checkpoint만 지우고 기록 파일은 남긴다.

::quiz{id="failed-run-in-table"}

## 실습

논문 표의 숫자 하나에서 그 숫자를 만든 run·checkpoint·설정·데이터로 되돌아가는 경로를 만든다.

1. 학습 스크립트가 시작할 때 `outputs/<run-id>/`를 만들고 `run.json`(resolved config, 커밋 해시, 커밋되지 않은 변경 여부, 데이터·분할 해시, 명령행)과 `status`를 쓰게 한다. 커밋되지 않은 변경이 있으면 `git.diff`를 함께 저장한다.
2. metric을 `metrics.jsonl`에 덧붙이고, 끝날 때 `status`를 `finished`로, 예외가 나면 `failed`와 traceback으로 바꾼다.
3. 결과 표 스크립트가 run 디렉터리를 모아 끝난 run만으로 표를 만들고, 칸마다 run ID 목록과 빠진 run 수를 함께 쓰게 한다.
4. 표에서 숫자 하나를 골라, 기록만 보고 그 run을 다시 실행해 같은 숫자가 나오는지 확인한다.

4단계를 위 sweep에서 해 본다. 가장 좋은 칸(lr = 0.1)에서 run ID로, run ID에서 설정·코드·데이터로 가서, 저장된 설정으로 다시 실행한다.

```python {exec="browser" session="track"}
best_lr = table["mean"].idxmin()
members = done[done["lr"] == best_lr]
print(f"표의 칸 lr={best_lr}: {table.loc[best_lr, 'mean']:.4f} ← run {list(members['run_id'])}")
row = members.iloc[0]
meta = json.loads((SWEEP / row["run_id"] / "run.json").read_text())
print("code_rev:", meta["code_rev"], " data:", meta["data_hash"], " config:", meta["config"])

replay = start_run(ROOT / "replay", meta["config"], "20261108-0900", meta["code_rev"], meta["data_hash"])
fake_train(meta["config"], replay)
again = json.loads((replay / "metrics.jsonl").read_text().splitlines()[-1])["val_loss"]
print(f"기록 {row['val_loss']:.4f}, 다시 실행 {again:.4f}, 같은가? {again == row['val_loss']}")
```

```output
표의 칸 lr=0.1: 0.3065 ← run ['20261107-1500-4de51384', '20261107-1500-61439ac5']
code_rev: 3f2a9c1  data: sha256:9e01  config: {'lr': 0.1, 'seed': 1, 'epochs': 5, 'out_dir': 'outputs'}
기록 0.3073, 다시 실행 0.3073, 같은가? True
```

다시 실행한 값이 기록과 같다. 이 예에서는 학습이 설정과 seed만으로 정해지는 함수라서 비트 단위로 같지만, 실제 학습에서는 같은 설정으로 다시 돌려도 결과가 조금 다를 수 있다. 어디까지 같아야 하고 어떻게 다름을 통제하는지는 22.2장에서 다룬다.

마지막으로 결과 표를 만드는 함수를 연습문제로 구현한다. 끝나지 않은 run을 평균에서 빼되, 빠진 수를 함께 세야 한다.

::exercise{id="results-table"}

## 실패 사례

- **결과를 손으로 표에 옮기는 것.** 다른 run의 숫자를 옮기거나, 실험을 다시 돌린 뒤 표를 갱신하지 않는다. 표는 스크립트로 만들고, 표를 만든 명령을 논문 저장소에 남긴다.
- **커밋하지 않은 코드로 실험하는 것.** 커밋 해시가 같아도 작업 디렉터리의 수정은 기록되지 않는다. 그 run의 코드는 복원할 수 없다. 실행 전에 커밋하거나, 적어도 `git.diff`를 저장하고 "dirty" 표시를 남긴다.
- **실패한 run의 마지막 metric을 평균에 섞는 것.** status를 보지 않는 집계 스크립트는 발산한 run의 손실이나 중단된 run의 중간값을 결과처럼 쓴다.
- **실패한 run을 조용히 빼는 것.** 결과 표에 "빠짐" 열이 없으면 불안정한 설정이 안정적으로 보인다.
- **`outputs/model.pt`처럼 고정된 이름에 덮어쓰는 것.** 어느 실행이 지금의 파일을 만들었는지 알 수 없다(6.1장).

## 통과 기준

- 임의의 결과 수치에서 run ID → 설정·코드 revision·데이터 해시로 되돌아가, 그 실험을 다시 실행해 같은 수치(또는 22.2장에서 정한 변동 범위 안의 수치)를 얻는다.
- 결과 표를 명령 하나로 run 디렉터리들에서 다시 만들고, 표의 칸마다 run ID 목록과 끝나지 않은 run의 수가 함께 남는다.
- 연습문제 `results-table`의 테스트를 통과한다.

## 참고 문헌

- Sandve, Nekrutenko, Taylor, Hovig (2013). Ten simple rules for reproducible computational research. *PLoS Computational Biology*, 9(10), e1003285. — 결과의 출처 추적, 수작업 단계 배제, 그림의 원자료 보존 등 계산 연구 기록의 기본 규칙
