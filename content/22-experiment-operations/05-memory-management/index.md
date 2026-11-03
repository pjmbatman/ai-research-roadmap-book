---
id: experiment-operations/memory-management
title: "GPU 메모리 관리"
title_en: "GPU Memory Management"
tier: core
status: outline
prerequisites: [experiment-operations/profiling-and-efficiency]
---

# GPU 메모리 관리

## 학습 목표

- 학습 중 메모리 사용을 계산한다
- 메모리 절약 기법을 사용한다

## 메모리 회계

- 파라미터 · gradient · optimizer 상태 · 활성값 · 임시 버퍼

## 측정

- max_memory_allocated, memory snapshot

## 절약 기법

- activation checkpointing · gradient accumulation · mixed precision · offloading

## 큰 모델 다루기

- 8-bit optimizer · 양자화된 가중치의 개념

## OOM 대응

- 원인 분석과 단계적 대응

## 실습

- 모델의 메모리 사용량을 손으로 계산하고 측정값과 비교한 뒤, 절약 기법별 효과를 측정한다.

## 실패 사례

- batch 크기만 줄여 OOM을 해결하고 학습 설정이 바뀐 것을 놓치는 것

## 통과 기준

- 주어진 GPU에서 학습 가능한 최대 설정을 계산과 측정으로 결정한다.
