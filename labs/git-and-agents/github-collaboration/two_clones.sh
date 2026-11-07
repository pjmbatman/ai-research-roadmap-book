#!/usr/bin/env bash
# 4.2 실습: 원격 저장소 하나와 두 사람의 clone을 로컬에 만들어 PR 흐름과 conflict를 연습한다.
#
#   bash two_clones.sh setup <새 디렉터리>   # remote.git(원격 역할), alice/, bob/ 를 만든다
#   bash two_clones.sh check <그 디렉터리>   # 원격 main에 두 사람의 변경이 모두 들어갔는지 검사한다
#
# setup이 끝나면 alice의 PR(alice/lr)은 이미 main에 merge되어 있고,
# bob의 branch(bob/batch)는 같은 줄을 다르게 고친 채 push되어 있다. bob/ 에서 해결한다.
set -euo pipefail

usage() { echo "사용: $0 {setup|check} <디렉터리>" >&2; exit 2; }
[[ $# -eq 2 ]] || usage
cmd=$1 dir=$2

as() {  # as <이름> <git 인자...> — 해당 사람의 clone에서 git 실행
  local who=$1; shift
  git -C "$dir/$who" -c user.name="$who" -c user.email="$who@example.com" "$@"
}

setup() {
  if [[ -e $dir ]]; then echo "이미 있는 경로입니다: $dir" >&2; exit 1; fi
  mkdir -p "$dir"
  git init -q --bare -b main "$dir/remote.git"
  git clone -q "$dir/remote.git" "$dir/alice" 2>/dev/null
  mkdir -p "$dir/alice/configs"
  printf 'lr: 1.0e-3\nbatch_size: 32\nepochs: 10\n' > "$dir/alice/configs/train.yaml"
  as alice add configs/train.yaml
  as alice commit -q -m "config: 기본 학습 설정"
  as alice push -q -u origin main 2>/dev/null
  git clone -q "$dir/remote.git" "$dir/bob"

  # alice: 학습률을 낮추는 PR → 리뷰 후 main에 merge되었다고 가정
  as alice switch -q -c alice/lr
  sed -i 's/^lr: .*/lr: 3.0e-4/' "$dir/alice/configs/train.yaml"
  as alice commit -q -am "config: 학습률을 3e-4로 낮춤 (발산 방지)"
  as alice push -q -u origin alice/lr 2>/dev/null
  as alice switch -q main
  as alice merge -q --no-ff alice/lr -m "Merge PR: alice/lr"
  as alice push -q origin main 2>/dev/null

  # bob: 같은 시각, 오래된 main에서 batch를 키우며 학습률도 함께 바꿨다
  as bob switch -q -c bob/batch
  sed -i 's/^lr: .*/lr: 2.0e-3/; s/^batch_size: .*/batch_size: 64/' "$dir/bob/configs/train.yaml"
  as bob commit -q -am "config: batch 64, 학습률을 batch에 비례해 2e-3으로"
  as bob push -q -u origin bob/batch 2>/dev/null

  echo "준비했습니다: $dir/{remote.git, alice, bob}"
  echo "bob/ 에서 origin/main을 bob/batch에 합치고 conflict를 해결한 뒤, main에 merge해 push하세요."
}

check() {
  local r="$dir/remote.git" ok=1
  local main; main=$(git -C "$r" rev-parse main)
  for b in alice/lr bob/batch; do
    if git -C "$r" merge-base --is-ancestor "$b" "$main"; then
      echo "  통과  원격 main에 $b 가 들어 있음"
    else
      echo "  실패  원격 main에 $b 가 없음"; ok=0
    fi
  done
  local cfg; cfg=$(git -C "$r" show "$main:configs/train.yaml")
  if grep -qE '^(<<<<<<<|=======|>>>>>>>)' <<<"$cfg"; then
    echo "  실패  configs/train.yaml에 conflict 표식이 남아 있음"; ok=0
  else
    echo "  통과  conflict 표식 없음"
  fi
  echo "원격 main의 configs/train.yaml:"; sed 's/^/    /' <<<"$cfg"
  if [[ $ok == 1 ]]; then echo "형식 검사를 통과했습니다. 남은 값이 두 사람의 의도를 모두 반영하는지는 직접 판단합니다."; else exit 1; fi
}

case $cmd in
  setup) setup ;;
  check) check ;;
  *) usage ;;
esac
