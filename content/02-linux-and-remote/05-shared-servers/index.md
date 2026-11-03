---
id: linux-and-remote/shared-servers
title: "공유 서버와 클러스터 사용"
title_en: "Shared Servers and Clusters"
tier: core
status: outline
prerequisites: [linux-and-remote/processes-and-monitoring]
---

# 공유 서버와 클러스터 사용

## 학습 목표

- 공유 자원 사용 규칙을 지킨다
- job scheduler로 실험을 제출한다
- 저장소 계층을 구분한다

## 공유 서버 예절

- GPU 할당(CUDA_VISIBLE_DEVICES), 사용 시간·우선순위 규칙, 종료하지 않은 프로세스

## Job scheduler

- Slurm 기초: sbatch · srun · squeue · scancel, 자원 요청, array job

## 저장소 계층

- home · scratch · 공유 NAS · 로컬 SSD, quota, 백업 여부

## 장기 실험 운영

- checkpoint 주기, 선점(preemption)과 재시작

## 실습

- Slurm 스크립트로 seed가 다른 실험 다섯 개를 array job으로 제출하고 결과를 수집한다.

## 실패 사례

- 다른 사람의 GPU에 프로세스를 올리는 것
- 백업되지 않는 scratch에 유일한 결과를 두는 것

## 통과 기준

- 연구실 공유 자원 규칙을 설명하고 scheduler로 실험을 제출·회수한다.
