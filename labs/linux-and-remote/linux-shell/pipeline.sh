#!/usr/bin/env bash
# 2.1 실습: 데이터 준비 → 검사 → 학습을 한 스크립트로 묶고, 어느 단계든 실패하면 즉시 멈춘다.
# 실행: bash labs/linux-and-remote/linux-shell/pipeline.sh [출력 디렉터리]
#       FAIL_AT=prepare bash .../pipeline.sh   # 준비 단계를 일부러 실패시킨다
set -euo pipefail

OUT_DIR="${1:-./run_$(date +%Y%m%d_%H%M%S)}"
FAIL_AT="${FAIL_AT:-}"                 # set -u에서도 안전하도록 기본값을 준다
PYTHON="${PYTHON:-python3}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
trap 'log "실패: ${BASH_COMMAND} (줄 ${LINENO}, exit $?)"' ERR

log "python = $(command -v "$PYTHON")"
mkdir -p "$OUT_DIR"

# 1. 데이터 준비: 실패를 흉내 내려면 FAIL_AT=prepare
log "1/3 데이터 준비"
if [[ "$FAIL_AT" == prepare ]]; then
    "$PYTHON" -c "raise SystemExit('원본 데이터를 찾을 수 없음(모의 실패)')"
fi
"$PYTHON" -c "
import random
random.seed(0)
with open('$OUT_DIR/data.csv', 'w') as f:
    f.write('x,y\n')
    for i in range(100):
        x = random.random()
        f.write(f'{x:.4f},{2 * x + 0.1 * random.random():.4f}\n')
"

# 2. 데이터 검사: 헤더 제외 행 수가 기대와 다르면 멈춘다
log "2/3 데이터 검사"
rows=$(tail -n +2 "$OUT_DIR/data.csv" | wc -l)
if [[ "$rows" -ne 100 ]]; then
    log "행 수가 100이 아님: $rows"
    exit 2
fi

# 3. 학습(여기서는 최소제곱 기울기 하나): 출력은 화면과 로그 파일에 함께 남긴다
log "3/3 학습"
"$PYTHON" -c "
import csv
xs, ys = zip(*[(float(r['x']), float(r['y'])) for r in csv.DictReader(open('$OUT_DIR/data.csv'))])
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
print(f'slope={slope:.3f}')
" | tee "$OUT_DIR/train.log"

log "완료: $OUT_DIR"
