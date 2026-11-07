#!/bin/bash
# 2.5 실습: seed가 다른 실험 다섯 개를 Slurm array job으로 제출한다. (집필 시점 예시)
# 제출: sbatch array_job.sh        상태: squeue -u $USER        기록: sacct -j <JOBID>
# partition 이름, GPU 요청 문법(--gres=gpu:1 또는 --gpus=1), 시간 상한은 클러스터 문서에 맞게 고친다.
#
# --array=0-4%2      task 0–4, 동시에 최대 2개
# --output           %x 작업 이름, %A array 작업 번호, %a task 번호 (logs/ 디렉터리는 미리 만들어 둔다)
# --signal=B:USR1@120  시간 상한 120초 전에 배치 셸(B)에 SIGUSR1
# --requeue          선점되면 다시 대기열로
#SBATCH --job-name=seed-sweep
#SBATCH --array=0-4%2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

set -euo pipefail
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} gpus=${CUDA_VISIBLE_DEVICES:-none}"

# 배치 셸이 받은 시그널을 학습 프로세스에 전달한다
python3 train_seed.py --out runs "$@" &
child=$!
trap 'kill -USR1 "$child"' USR1
trap 'kill -TERM "$child"' TERM
# trap이 wait를 깨우면 종료 코드가 128보다 크다. 그때는 자식이 저장을 마칠 때까지 다시 기다린다
status=0
wait "$child" || status=$?
if (( status > 128 )) && kill -0 "$child" 2>/dev/null; then
    status=0
    wait "$child" || status=$?
fi
echo "train_seed.py exit=$status"
exit "$status"
