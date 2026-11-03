---
id: experiment-operations/reproducibility
title: "재현성"
title_en: "Reproducibility"
tier: core
status: outline
prerequisites: [experiment-operations/experiment-tracking]
---

# 재현성

## 학습 목표

- 무작위성의 출처를 통제한다
- 결정성과 성능의 trade-off를 이해한다
- 재현성의 수준을 구분한다

## 재현성의 수준

- 같은 환경 재실행 · 다른 환경 재현 · 독립 재구현

## 무작위성의 출처

- Python · NumPy · PyTorch RNG, DataLoader worker, CUDA 비결정 연산

## 결정적 실행

- deterministic 알고리즘 설정과 성능 비용

## 변동성의 보고

- 여러 seed, 분산 보고

## 환경 고정

- lockfile · 컨테이너 · 하드웨어 기록

## 실습

- 같은 실험을 결정적 설정과 기본 설정으로 각각 여러 번 실행해 결과 변동을 측정한다.

## 실패 사례

- seed 하나 고정으로 재현성이 보장된다고 생각하는 것

## 통과 기준

- 다른 사람이 다른 머신에서 결과를 보고된 변동 범위 안에서 재현한다.
