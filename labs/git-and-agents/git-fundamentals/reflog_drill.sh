#!/usr/bin/env bash
# 4.1 실습: 일부러 사고를 낸 연습용 저장소를 만들고, 복구했는지 확인한다.
#
#   bash reflog_drill.sh setup <새 디렉터리>   # 사고 세 개가 난 저장소를 만든다
#   bash reflog_drill.sh check <그 디렉터리>   # 복구했는지 검사한다
#
# 사고 1. main에서 reset --hard HEAD~2로 commit 두 개를 잃었다.
# 사고 2. 실험 branch exp-lr을 merge하지 않은 채 branch -D로 지웠다.
# 사고 3. feature를 잘못된 기준으로 rebase해 commit 하나가 빠졌다.
set -euo pipefail

usage() { echo "사용: $0 {setup|check} <디렉터리>" >&2; exit 2; }
[[ $# -eq 2 ]] || usage
cmd=$1 dir=$2

commit() {  # commit <파일> <내용> <메시지>
  printf '%s\n' "$2" >> "$1"
  git add "$1"
  git commit -q -m "$3"
}

setup() {
  if [[ -e $dir ]]; then echo "이미 있는 경로입니다: $dir" >&2; exit 1; fi
  mkdir -p "$dir" && cd "$dir"
  git init -q -b main
  git config user.name "Drill" && git config user.email "drill@example.com"

  commit data.py   "def load(): ..."       "data: 원시 데이터 로더"
  commit model.py  "class MLP: ..."        "model: baseline MLP"
  commit train.py  "def train(): ..."      "train: 학습 루프"
  commit eval.py   "def evaluate(): ..."   "eval: 평가 스크립트"
  git reset -q --hard HEAD~2                                  # 사고 1

  git switch -q -c exp-lr
  commit config.py "lr = 3e-4"             "exp: 학습률 3e-4 실험"
  git switch -q main
  git branch -q -D exp-lr                                     # 사고 2

  git switch -q -c feature
  commit blocks.py "class ResBlock: ..."   "feature: residual block"
  commit regularize.py "dropout = 0.1"     "feature: dropout 추가"
  git rebase -q --onto main HEAD~1                            # 사고 3: 첫 commit이 빠진다
  git switch -q main

  echo "연습 저장소를 만들었습니다: $dir"
  echo "현재 main:";    git log --oneline main
  echo "현재 feature:"; git log --oneline feature
  echo "복구할 것: main의 'train:'·'eval:' commit, branch exp-lr, feature의 'residual block' commit"
}

check() {
  cd "$dir"
  local ok=1
  has() {  # has <branch> <메시지 일부>
    if git rev-parse -q --verify "refs/heads/$1" >/dev/null &&
       git log --format=%s "$1" | grep -qF -- "$2"; then
      echo "  통과  $1 에 '$2'"
    else
      echo "  실패  $1 에 '$2' 가 없음"; ok=0
    fi
  }
  has main    "train: 학습 루프"
  has main    "eval: 평가 스크립트"
  has exp-lr  "exp: 학습률 3e-4 실험"
  has feature "feature: residual block"
  has feature "feature: dropout 추가"
  if [[ $ok == 1 ]]; then echo "모두 복구했습니다."; else exit 1; fi
}

case $cmd in
  setup) setup ;;
  check) check ;;
  *) usage ;;
esac
