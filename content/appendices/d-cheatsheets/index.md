---
id: appendices/cheatsheets
title: "명령어 치트시트"
title_en: "Command Cheatsheets"
tier: reference
status: draft
prerequisites: []
---

# 명령어 치트시트

연구 시간의 상당 부분은 모델이 아니라 그 주변에서 쓰인다. 서버에 접속하고, 로그를 거르고, GPU가 비었는지 보고, 환경을 다시 만들고, 사고 친 Git 이력을 되살린다. 이 부록은 본문 장들이 실제로 쓴 명령을 상황별 표로 모으고, 각 명령을 설명한 장을 링크한다.

**모든 명령은 집필 시점 예시다.** 옵션 이름과 기본값은 도구 버전·배포판·클러스터 설정에 따라 다르다. 처음 쓰는 명령은 `man <명령>`, `<명령> --help`, 공식 문서로 지금 쓰는 버전의 동작을 확인한다. 특히 삭제·덮어쓰기·이력 재작성 명령은 표의 '주의' 표시를 읽고, 가능하면 dry-run으로 먼저 본다. `<PID>`, `<id>`, `gpu1`처럼 꺾쇠나 예시 이름으로 쓴 부분은 자신의 값으로 바꾼다.

## 학습 목표

- 연구 작업에서 자주 쓰는 명령을 상황별로 찾아 쓴다
- 되돌리기 어려운 명령(삭제·덮어쓰기·이력 재작성)을 실행 전에 확인하는 습관을 들인다
- 명령의 동작이 버전과 환경에 따라 다를 때 확인하는 절차를 안다

## Linux와 셸

**파일·검색·텍스트**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `ls -lh`, `ls -lt`, `ls -a` | 크기를 읽기 쉽게, 수정 시각 순으로, 숨김 파일까지 보기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `ln -s /data/shared/imagenet data` | 공유 데이터에 심볼릭 링크 걸기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md), [6.1](../../06-research-code-engineering/01-project-structure/index.md) |
| `find . -name '*.ckpt' -mtime -1` | 하루 안에 바뀐 checkpoint 찾기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `find /data -size +10G` | 10 GB보다 큰 파일 찾기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `grep -rn 'lr=' configs/` | 디렉터리 아래에서 문자열을 줄 번호와 함께 찾기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `rg 'TODO' src/` | ripgrep으로 빠르게 찾기(설치되어 있을 때) | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `grep -o 'loss=[0-9.]*' train.log \| tail -n 3` | 로그에서 마지막 세 개의 loss 값 뽑기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `awk '{print $3}' train.log \| sort \| uniq -c \| sort -rn` | 셋째 열(로그 수준)별 줄 수를 많은 순으로 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `head -n 20 file`, `tail -n 50 file`, `tail -f train.log` | 앞·뒤 몇 줄 보기, 늘어나는 로그 따라 보기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `wc -l`, `cut -d, -f2`, `sed 's/old/new/g'` | 줄 수 세기, CSV 둘째 열, 문자열 바꾸기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `tar -czf results.tar.gz results/` | 디렉터리를 gzip으로 묶기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `tar -tzf results.tar.gz \| head` | 풀기 전에 목록 확인 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `tar -xzf results.tar.gz -C /scratch/kim` | 지정한 위치에 풀기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `rsync -avhn --delete src/ dst/` | **dry-run**: 동기화하면 무엇이 바뀔지만 보기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `rsync -avh --progress src/ dst/` | 권한·시간을 보존하며 동기화. `src/`(끝의 `/`)는 **내용**을, `src`는 디렉터리 자체를 복사한다 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |

**리다이렉션·환경변수·스크립트**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `python train.py --config cfg.yaml 2>&1 \| tee logs/run_$(date +%Y%m%d_%H%M%S).log` | stdout·stderr를 화면과 파일에 함께 남기기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `cmd > log 2>&1` | stdout과 stderr를 모두 파일로. 순서를 바꾼 `cmd 2>&1 > log`는 stderr를 화면에 남긴다 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `some_noisy_tool 2>/dev/null` | 오류 출력 버리기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `export DATA_ROOT=/data/kim` | 자식 프로세스에 전달되는 환경변수 설정 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `env \| grep DATA_ROOT` | 현재 환경변수에서 찾기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `type -a python3`, `command -v python` | 실행될 프로그램이 PATH의 어디에 있는지 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `chmod 750 run.sh`, `chmod o-rwx file` | 권한 바꾸기 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `set -euo pipefail` | 스크립트 첫머리: 오류·미정의 변수·파이프 중간 실패에서 멈춤 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |
| `rm -rf "$OUTDIR/"` | **주의.** 변수는 반드시 따옴표로 감싸고 `set -u`와 함께 쓴다. 빈 변수면 `rm -rf /`가 된다 | [2.1](../../02-linux-and-remote/01-linux-shell/index.md) |

**프로세스·자원**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `ps -eo pid,ppid,user,stat,rss,etime,comm --sort=-rss \| head` | 메모리를 많이 쓰는 프로세스 순으로 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `ps -o pid,ppid,user,etime,cmd -p <PID>` | 특정 프로세스의 부모·실행 시간·명령줄 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `kill <PID>` → 안 되면 `kill -9 <PID>` | SIGTERM으로 정리할 기회를 준 뒤 강제 종료 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `free -h` | 메모리 사용량(buff/cache 포함) | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `df -h`, `df -i` | 디스크 용량, inode 수 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `du -sh * \| sort -h` | 현재 디렉터리에서 큰 것 찾기 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `iostat -x 1` | 디스크 I/O 대기 확인 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `dmesg -T \| grep -i -E "out of memory\|killed process"` | OOM killer가 프로세스를 죽였는지 확인 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `lsof +L1` | 지웠지만 열려 있어 용량을 차지하는 파일 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `python -u train.py` | 출력 버퍼링 끄기(로그가 늦게 찍힐 때) | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |

## 원격과 세션

| 명령 | 하는 일 | 장 |
|---|---|---|
| `ssh-keygen -t ed25519 -C "kim@laptop-2026"` | 키 쌍 만들기(암호문 설정) | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `ssh-copy-id -i ~/.ssh/id_ed25519.pub kim@gpu1.example.org` | 공개키를 서버의 `authorized_keys`에 추가 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `~/.ssh/config`의 `Host`·`HostName`·`User`·`ProxyJump`·`IdentitiesOnly yes` | 별칭과 jump host로 접속 단순화 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `ssh -N -L 16006:localhost:6006 gpu1` | 원격 6006 포트(TensorBoard 등)를 내 16006 포트로 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `scp results.csv gpu1:~/project/` | 파일 몇 개 복사 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `rsync -avP gpu1:~/project/outputs/ ./outputs/` | 원격 → 로컬 동기화, 중단되면 이어 받기 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `tar -cf - many_small_files/ \| ssh gpu1 'tar -xf - -C /scratch/kim'` | 작은 파일 수십만 개를 묶어서 흘려보내기 | [2.3](../../02-linux-and-remote/03-ssh/index.md) |
| `tmux new -s exp42` | 이름 있는 세션 만들기. `Ctrl-b d`로 빠져나와도 계속 돈다 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `tmux ls`, `tmux attach -t exp42` | 세션 목록, 다시 붙기 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `tmux kill-session -t exp42` | 세션 끝내기 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `nohup cmd &` | 접속이 끊겨도 계속 실행(tmux를 쓸 수 없을 때) | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |

**공유 클러스터(Slurm)**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `sbatch job.sh` | 배치 스크립트 제출. job 번호가 출력된다 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `#SBATCH --gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=1-00:00:00` | job 스크립트의 자원 요청 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `srun --pty bash` | 자원을 받아 대화형 셸 얻기 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `squeue -u $USER` | 내 job 상태(PD 대기, R 실행) | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `scancel <JOBID>` | job 취소. array의 한 task는 `<JOBID>_<번호>` | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `sacct -j <JOBID>` | 끝난 job의 상태·자원 사용 기록 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `sbatch --array=0-$((N - 1))%8 run_job.sh sweep/manifest.jsonl` | manifest의 줄마다 task를 만들고 동시에 8개까지 | [22.3](../../22-experiment-operations/03-experiment-automation/index.md) |
| `quota -s` | 내 저장 공간 할당량 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |

## GPU

| 명령 | 하는 일 | 장 |
|---|---|---|
| `nvidia-smi` | GPU·드라이버·프로세스 요약 표 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits` | 기계가 읽기 좋은 CSV로 상태 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | GPU를 쓰는 프로세스와 메모리 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `nvidia-smi dmon`, `watch -n 1 nvidia-smi` | 1초마다 사용률·전력·온도 보기 | [2.2](../../02-linux-and-remote/02-processes-and-monitoring/index.md) |
| `nvidia-smi topo -m` | GPU 사이 연결(NVLink·PCIe) 구조 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| `export CUDA_DEVICE_ORDER=PCI_BUS_ID` | GPU 번호를 `nvidia-smi`와 같은 순서로 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `CUDA_VISIBLE_DEVICES=2,3 python train.py` | 프로세스에 보일 GPU만 지정. 안에서는 0, 1로 보인다 | [2.5](../../02-linux-and-remote/05-shared-servers/index.md) |
| `cat /proc/driver/nvidia/version` | 설치된 드라이버 버전 | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| `nvcc --version` | 시스템 CUDA toolkit 버전(PyTorch wheel의 CUDA와 다를 수 있다) | [3.2](../../03-environments/02-ubuntu-and-cuda/index.md) |
| `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | PyTorch가 빌드된 CUDA 버전과 GPU 인식 여부 | [3.1](../../03-environments/01-software-stack/index.md) |
| `docker run --rm --gpus all lab/proj nvidia-smi` | 컨테이너에서 GPU가 보이는지 | [3.5](../../03-environments/05-docker/index.md) |

## 환경

**Python 환경과 의존성**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `python3 -m venv .venv` → `source .venv/bin/activate` | 가상환경 만들고 활성화 | [3.3](../../03-environments/03-python-environments/index.md) |
| `python -m pip install numpy` | 실행할 python을 명시해 설치 | [3.3](../../03-environments/03-python-environments/index.md) |
| `which python`, `python -c "import sys; print(sys.executable)"`, `pip --version` | 지금 어떤 python·pip을 쓰는지 | [3.3](../../03-environments/03-python-environments/index.md) |
| `uv init`, `uv add numpy` | `pyproject.toml` 만들기, 의존성 추가(lockfile과 `.venv` 함께 갱신) | [3.3](../../03-environments/03-python-environments/index.md) |
| `uv run python train.py` | 활성화 없이 프로젝트 환경으로 실행 | [3.3](../../03-environments/03-python-environments/index.md) |
| `uv lock`, `uv sync` | lockfile 풀기, `.venv`를 lockfile과 정확히 맞추기 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| `uv sync --locked` | lockfile이 `pyproject.toml`과 어긋나면 실패(CI·새 머신용) | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| `uv lock --upgrade-package numpy` | 한 패키지만 의도적으로 올리기 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| `uv export --format requirements-txt > requirements.txt` | pip 사용자를 위한 내보내기 | [3.4](../../03-environments/04-dependencies-and-packaging/index.md) |
| `uv tree --package torch` | 의존성 트리 보기 | [3.1](../../03-environments/01-software-stack/index.md) |
| `conda create -n proj -c conda-forge python=3.12 numpy` → `conda activate proj` | conda 환경 만들고 활성화 | [3.3](../../03-environments/03-python-environments/index.md) |
| `pip check`, `uv pip check` | 설치된 패키지의 의존성 충돌 검사 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `pip freeze`, `uv pip freeze` | 설치된 버전 목록 남기기 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |

**진단**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `python -VV`, `uname -r`, `cat /etc/os-release` | Python·커널·배포판 버전 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `python -c "import numpy; print(numpy.__file__, numpy.__version__)"` | 실제로 import되는 패키지의 위치와 버전 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `python -m torch.utils.collect_env` | PyTorch·CUDA·드라이버 정보 한 번에 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `ldd /path/to/failing_extension.so \| grep "not found"` | 확장 모듈이 찾지 못하는 공유 라이브러리 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `ldconfig -p \| grep libcudart` | 시스템이 아는 CUDA 런타임 라이브러리 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |
| `python -X faulthandler script.py` | segfault가 난 Python 위치 출력 | [3.6](../../03-environments/06-environment-diagnosis/index.md) |

**컨테이너**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `docker build -t lab/proj:"${GIT_SHA:0:8}" .` | 커밋 해시를 태그로 image 만들기 | [3.5](../../03-environments/05-docker/index.md) |
| `docker image inspect --format '{{json .RepoDigests}}' lab/proj:<태그>` | 기록할 digest 확인 | [3.5](../../03-environments/05-docker/index.md) |
| `docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/outputs":/outputs lab/proj ...` | 출력 파일이 root 소유가 되지 않게 실행 | [3.5](../../03-environments/05-docker/index.md) |
| `-v /data/shared/dataset-v3:/data:ro` | 데이터는 읽기 전용으로 마운트 | [3.5](../../03-environments/05-docker/index.md) |
| `docker run --rm --gpus '"device=0,1"' lab/proj python train.py` | 특정 GPU만 컨테이너에 | [3.5](../../03-environments/05-docker/index.md) |
| `apptainer build proj.sif docker://<registry>/lab/proj@sha256:<digest>` | HPC용 Apptainer image로 변환 | [3.5](../../03-environments/05-docker/index.md) |
| `apptainer exec --nv --bind /scratch/me/data:/data proj.sif python train.py` | GPU(`--nv`)와 데이터 경로를 붙여 실행 | [3.5](../../03-environments/05-docker/index.md) |

## Git

**일상 작업**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `git status`, `git diff`, `git diff --staged` | 작업 트리·index의 변경 확인 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git add -p configs/train.yaml` | 파일 안의 변경 덩어리(hunk)를 골라 올리기 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git commit -m "model: ..."` | commit 만들기 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git log --oneline --graph -n 10`, `git log --oneline --graph --all` | 이력을 그래프로 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git show HEAD` | 마지막 commit의 메시지와 diff | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git switch -c exp-loss` | 새 branch를 만들고 이동 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git fetch origin` → `git log --oneline main..origin/main` | 원격에만 있는 commit 먼저 보기 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git pull --ff-only` | fast-forward로만 받기(의도치 않은 merge commit 방지) | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git push -u origin feature` | branch를 원격에 올리고 추적 설정 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git remote add upstream <URL>` → `git switch -c fix-typo upstream/main` | fork에서 원래 저장소의 최신 main 위에 작업 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git rm --cached data/raw.csv` | 파일은 두고 추적만 중단 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git check-ignore -v <경로>` | 어느 `.gitignore` 규칙이 파일을 무시하는지 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git lfs install`, `git lfs track "*.ckpt"` | 큰 파일을 Git LFS로 관리 | [4.2](../../04-git-and-agents/02-github-collaboration/index.md) |
| `git worktree add ../exp-lr -b exp/lr`, `git worktree list`, `git worktree remove ../exp-lr` | 한 저장소에서 여러 branch를 별도 디렉터리로 동시에 | [4.6](../../04-git-and-agents/06-agentic-workflow/index.md) |
| `git rev-parse HEAD`, `git status --porcelain` | 실험 기록에 남길 commit과 미커밋 변경 여부 | [22.1](../../22-experiment-operations/01-experiment-tracking/index.md) |

**되돌리기와 복구**

| 명령 | 하는 일 | 장 |
|---|---|---|
| `git restore <파일>` | **주의.** 작업 트리의 변경을 버린다(복구 불가) | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git restore --staged <파일>` | index에서만 내리기(변경은 남는다) | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git commit --amend` | 마지막 commit 고치기. push한 뒤에는 쓰지 않는다 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git revert <commit>` | 되돌리는 새 commit 만들기(공유된 이력에 안전) | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git reset --hard HEAD~2` | **주의.** branch와 작업 트리를 두 commit 전으로. 미커밋 변경은 사라진다 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git reflog`, `git reflog main` | HEAD·branch가 거쳐 온 위치 기록 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git branch rescue <id>` | reflog에서 찾은 commit에 이름 붙여 살리기 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git reset --hard main@{1}` | 직전 위치로 branch 되돌리기 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git merge --abort`, `git rebase --abort` | 진행 중인 merge·rebase 취소 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git push --force-with-lease` | 원격이 내가 본 상태일 때만 덮어쓰기. `--force`보다 안전 | [4.1](../../04-git-and-agents/01-git-fundamentals/index.md) |
| `git bisect start` → `git bisect bad HEAD` → `git bisect good v0.3` → `git bisect run uv run pytest -x tests/test_metrics.py` | 회귀가 생긴 commit을 이분 탐색으로 자동 추적 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |

## Python과 NumPy

| 명령·코드 | 하는 일 | 장 |
|---|---|---|
| `python -m package.module` | 패키지 안 모듈을 import 경로 기준으로 실행 | [5.2](../../05-python-and-computing/02-modules-and-imports/index.md) |
| `rng = np.random.default_rng(0)` | 시드를 고정한 난수 생성기 객체 만들기 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `x.shape`, `x.dtype`, `x.strides` | shape·자료형·메모리 보폭 확인 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `np.shares_memory(a, b)` | 두 배열이 메모리를 공유하는지(view인지) | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `sub = data[:, :10].copy()` | slice는 view다. 원본을 지키려면 복사 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `X.sum(axis=0)`, `X.mean(axis=0, keepdims=True)` | 축을 따라 줄이기, 줄인 축을 길이 1로 남겨 broadcasting에 쓰기 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `y_pred[:, 0]` | `(N, 1)`과 `(N,)`의 뺄셈이 `(N, N)`으로 퍼지는 것 막기 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `np.einsum("ijk,ijk->ij", d, d)` | 축 이름으로 곱·합 표현 | [5.3](../../05-python-and-computing/03-scientific-python/index.md), [7.7](../../07-linear-algebra/07-tensors-and-einsum/index.md) |
| `np.allclose(a, b)` | 부동소수점 결과는 `==` 대신 허용오차로 비교 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `df.loc[df.flag, "temp_c"] = np.nan` | pandas에서 조건에 맞는 행의 열을 한 번에 쓰기(연쇄 인덱싱 대신) | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `df.groupby("station")["temp_c"].mean()` | 그룹별 집계 | [5.3](../../05-python-and-computing/03-scientific-python/index.md) |
| `python -m timeit "..."`, `%timeit` | 짧은 코드 조각의 실행 시간 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md) |
| `python -m cProfile -s cumtime script.py` | 함수별 누적 시간 프로파일 | [5.4](../../05-python-and-computing/04-algorithms-and-complexity/index.md) |
| `ruff check .`, `ruff format --check .`, `mypy units_typed.py` | lint, 포맷 검사, 정적 타입 검사 | [6.2](../../06-research-code-engineering/02-naming-and-style/index.md) |
| `uv run pytest -q`, `pytest --pdb` | 테스트 실행, 실패 지점에서 디버거 열기 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |
| `breakpoint()`, `python -m pdb script.py` | 코드 안·밖에서 디버거 시작 | [6.6](../../06-research-code-engineering/06-testing-and-debugging/index.md) |

## PyTorch

**shape와 디바이스**

| 코드 | 하는 일 | 장 |
|---|---|---|
| `x.shape`, `x.dtype`, `x.device` | 텐서의 세 가지 기본 속성. 버그의 대부분이 여기서 드러난다 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| `x.permute(0, 2, 1)` | `(B, T, D)` → `(B, D, T)`처럼 **축 순서** 바꾸기 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| `x.view(B, D, T)` | **주의.** 축을 바꾸지 않고 메모리를 다시 읽는다. 축 교환에 쓰면 오류 없이 원소가 뒤섞인다 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| `x.contiguous()` | 연속 메모리로 복사(필요할 때만) | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| `torch.from_numpy(a)`, `torch.tensor(a)` | NumPy 배열과 메모리 공유 / 복사 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |
| `x.to(device)`, `x.to(device, non_blocking=True)` | 디바이스 옮기기. pinned memory와 함께 비동기 복사 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md), [17.4](../../17-deep-learning-pytorch/04-datasets-and-dataloaders/index.md) |
| `loss.item()`, `t.detach()`, `t.detach().clone()` | 값만 꺼내기, 그래프에서 떼기, 떼고 복사 | [17.2](../../17-deep-learning-pytorch/02-tensors-and-autograd/index.md) |

**학습·평가 루프**

| 코드 | 하는 일 | 장 |
|---|---|---|
| `opt.zero_grad()` → `loss.backward()` → `opt.step()` | 한 step의 기본 순서. `zero_grad`를 빼면 gradient가 누적된다 | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| `nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)` | `backward()` 뒤, `step()` 앞에서 gradient 노름 자르기 | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| `model.train()`, `model.eval()` | dropout·BatchNorm의 **동작** 전환 | [17.1](../../17-deep-learning-pytorch/01-neural-network-components/index.md), [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) |
| `with torch.no_grad():`, `with torch.inference_mode():` | 그래프를 만들지 않는다. `eval()`과 별개이므로 평가에는 둘 다 쓴다 | [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) |
| `torch.autocast(device_type, dtype)`, `torch.amp.GradScaler` | mixed precision과 fp16 loss scaling | [17.5](../../17-deep-learning-pytorch/05-training-loop/index.md) |
| `torch.save(...)`, `model.load_state_dict(sd, strict=True)` | checkpoint 저장·불러오기. `strict=True`로 빠지거나 남는 키를 잡는다 | [17.3](../../17-deep-learning-pytorch/03-modules/index.md), [17.6](../../17-deep-learning-pytorch/06-evaluation-and-checkpoints/index.md) |
| `torch.compile(model)` | 연산 그래프를 잡아 연산을 합치고 하드웨어에 맞는 kernel로 바꾸기 | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |
| `torch.cuda.synchronize()` | 시간 측정 전에 GPU 작업이 끝나길 기다리기 | [22.4](../../22-experiment-operations/04-profiling-and-efficiency/index.md) |

**메모리**

| 코드 | 하는 일 | 장 |
|---|---|---|
| `torch.cuda.reset_peak_memory_stats()` → `torch.cuda.max_memory_allocated()` | 구간의 최대 할당 메모리 측정 | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| `torch.cuda.memory_allocated()` | 지금 살아 있는 텐서가 차지하는 메모리 | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| `torch.cuda.empty_cache()` | 쓰지 않는 캐시 블록을 돌려줄 뿐, 살아 있는 텐서는 줄이지 않는다 | [22.5](../../22-experiment-operations/05-memory-management/index.md) |
| `torch.utils.checkpoint.checkpoint(...)` | activation checkpointing: 활성값 대신 재계산 | [22.5](../../22-experiment-operations/05-memory-management/index.md) |

## 실험 관리

| 명령·코드 | 하는 일 | 장 |
|---|---|---|
| `uv run python run_experiment.py --exp configs/linear-baseline.toml --set train.seed=1` | 설정 파일 + 명령줄 덮어쓰기로 실행(실습 저장소의 예) | [6.3](../../06-research-code-engineering/03-configuration/index.md) |
| `uv run python soh_cli.py train --config configs/linear.toml --seed 1 --dry-run` | 실행하지 않고 해석된 설정만 확인 | [6.4](../../06-research-code-engineering/04-cli-design/index.md) |
| `random.seed(s)`, `np.random.seed(s)`, `torch.manual_seed(s)` | 세 난수원을 함께 고정 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| `torch.use_deterministic_algorithms(True)` | 비결정적 연산을 쓰면 오류를 내게 하기 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| `torch.backends.cudnn.benchmark = True` | 첫 호출 때 convolution 알고리즘을 시험해 빠른 것을 고른다. 재현성을 우선할 때는 끈다 | [22.2](../../22-experiment-operations/02-reproducibility/index.md) |
| `tensorboard --logdir runs --port 6006` | 기록한 metric 보기(원격이면 SSH 포트 포워딩과 함께) | [2.3](../../02-linux-and-remote/03-ssh/index.md), [17.8](../../17-deep-learning-pytorch/08-dashboards/index.md) |
| `torchrun --standalone --nproc-per-node=4 train.py --config configs/base.yaml` | 서버 한 대의 GPU 4개로 DDP 실행 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |
| `export TORCH_DISTRIBUTED_DEBUG=DETAIL`, `export NCCL_DEBUG=INFO` | 분산 학습의 collective 불일치·통신 초기화 진단 | [22.6](../../22-experiment-operations/06-distributed-training/index.md) |

## 실습

1. **자기 환경의 버전 기록 만들기.** '환경 › 진단' 표의 명령으로 지금 쓰는 서버의 OS·커널·Python·드라이버·PyTorch·CUDA 버전을 한 파일로 모은다. 같은 명령을 다른 서버에서 돌려 차이를 표시한다.
2. **dry-run 습관 들이기.** `rsync -avhn --delete`로 동기화 결과를 먼저 보고, 지워질 파일 목록을 확인한 뒤에 `-n`을 빼고 실행한다. 같은 원칙으로 `git clean`이나 `rm -rf`를 쓰기 전에 대상 목록을 `ls`나 `git status`로 먼저 출력한다.
3. **Git 사고 복구 연습.** 연습용 저장소에서 `git reset --hard HEAD~2`로 commit 두 개를 "잃은" 뒤, `git reflog`로 찾아 `git branch rescue <id>`로 살린다([4.1](../../04-git-and-agents/01-git-fundamentals/index.md)의 실습 스크립트를 쓴다).
4. **개인 치트시트로 바꾸기.** 이 표에서 자기 작업에 쓰는 명령만 골라, 자기 서버의 실제 경로와 별칭으로 바꾼 한 쪽짜리 치트시트를 만든다. 명령마다 확인한 도구 버전을 적는다.

## 실패 사례

- **치트시트의 명령을 확인 없이 붙여 넣는 것.** 옵션 이름은 도구 버전마다 바뀐다(예: `docker run --gpus`는 NVIDIA Container Toolkit이 있어야 하고, Slurm의 `--gres`와 `--gpus` 지원은 클러스터 설정에 따른다). 처음 쓰는 환경에서는 `--help`와 공식 문서로 확인한다.
- **경로 끝의 `/` 하나를 무시하는 것.** `rsync -a src dst/`는 `dst/src/`를 만들고, `rsync -a src/ dst/`는 내용을 `dst/` 바로 아래에 둔다. `--delete`와 함께 틀리면 데이터가 지워진다.
- **리다이렉션 순서를 바꾸는 것.** `cmd 2>&1 > log`는 stderr를 파일이 아닌 화면에 남긴다. 밤새 돈 학습의 오류 메시지가 로그에 없는 흔한 이유다.
- **`git reset --hard`·`git push --force`를 습관처럼 쓰는 것.** 앞의 것은 커밋하지 않은 변경을, 뒤의 것은 동료의 commit을 지운다. 되돌릴 수 있는 `git revert`, `git push --force-with-lease`를 먼저 고려하고, 사고가 나면 `git reflog`부터 본다.
- **`view`로 축을 바꾸는 것.** `x.view(B, D, T)`는 shape만 맞추고 원소를 섞는다. 축 교환은 `permute`다. 오류가 나지 않으므로 학습이 "조금 덜 되는" 형태로만 드러난다.
- **`empty_cache()`로 메모리 문제를 고치려는 것.** 살아 있는 텐서(예: 리스트에 쌓아 둔 `loss`)는 줄지 않는다. `loss.item()`으로 값만 남기고, `max_memory_allocated()`로 측정해 원인을 찾는다.

## 통과 기준

- 실습 1의 버전 기록 파일을 만들었고, 그 파일에 OS·Python·PyTorch·CUDA·드라이버 버전이 모두 있다.
- `git reflog`로 잃은 commit을 되살리는 과정을 연습 저장소에서 한 번 수행했다.
- 자기 작업용 치트시트에 적은 모든 명령을 실제 환경에서 한 번 이상 실행해 확인했다.
