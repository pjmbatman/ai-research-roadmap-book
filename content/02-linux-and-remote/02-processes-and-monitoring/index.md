---
id: linux-and-remote/processes-and-monitoring
title: "프로세스·세션·자원 모니터링"
title_en: "Processes, Sessions, and Resource Monitoring"
tier: core
status: outline
prerequisites: [linux-and-remote/linux-shell]
---

# 프로세스·세션·자원 모니터링

## 학습 목표

- 프로세스와 시그널을 다룬다
- CPU·메모리·디스크·GPU 자원을 모니터링한다
- 끊기지 않는 작업 세션을 운영한다

## 프로세스와 시그널

- PID, foreground·background, nohup, kill과 시그널(SIGINT · SIGTERM · SIGKILL)

## 자원 모니터링 도구

- ps · top · htop · btop, free, df · du, iostat, lsof

## GPU 모니터링

- nvidia-smi 출력 읽기(사용률 · VRAM · 온도 · 전력 · 프로세스), nvidia-smi dmon, nvtop · gpustat

## 작업 세션 유지

- tmux(session · window · pane), screen, 로그 파일로 출력 남기기

## 장애 진단

- OOM killer, zombie process, 디스크 가득 참, I/O 병목, GPU 메모리를 점유한 유령 프로세스

## 실습

- 일부러 메모리 부족·디스크 부족·끊긴 세션 상황을 만들고 로그와 모니터링 도구로 원인을 추적한다.

## 실패 사례

- SSH 세션에서 직접 학습을 실행해 연결과 함께 종료되는 것
- nvidia-smi의 사용률만 보고 병목을 단정하는 것

## 통과 기준

- 학습이 느리거나 죽은 사례에서 원인을 자원 관점으로 설명하는 보고서를 쓴다.
