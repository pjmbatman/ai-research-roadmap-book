---
id: deep-learning-pytorch/evaluation-and-checkpoints
title: "평가·추론·체크포인트"
title_en: "Evaluation, Inference, and Checkpointing"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/training-loop]
---

# 평가·추론·체크포인트

## 학습 목표

- 평가 모드와 gradient 모드를 구분한다
- metric을 올바르게 집계한다
- 학습을 정확히 재개한다

## 평가 모드

- model.eval()과 no_grad · inference_mode의 역할 차이

## metric 집계

- batch 평균의 평균 문제, 유효 관측 수 기반 집계, mask 처리

## 체크포인트의 내용

- model · optimizer · scheduler · scaler · RNG 상태 · epoch · 설정

## best와 last

- 선택 기준, 저장 정책

## 재개와 추론

- 학습 재개와 가중치만 불러오는 추론의 차이

## inference contract

- 전처리 · 설정 · 정규화 통계까지 포함한 추론 명세

## 실습

- 학습 중간에 중단·재개한 결과가 끊김 없이 학습한 결과와 일치하는지 검증하고, 독립 평가 스크립트로 결과를 재현한다.

## 실패 사례

- eval 모드를 빠뜨려 dropout·BatchNorm이 평가에 영향을 주는 것
- 마지막 batch 크기를 무시한 평균

## 통과 기준

- checkpoint 복원 후 독립된 평가 스크립트로 같은 숫자를 재현한다.
