---
id: experiment-operations/inference-and-deployment
title: "추론 최적화와 배포"
title_en: "Inference Optimization and Deployment"
tier: reference
status: outline
prerequisites: [deep-learning-pytorch/evaluation-and-checkpoints]
---

# 추론 최적화와 배포

## 학습 목표

- 학습된 모델을 내보내고 추론 비용을 줄인다

## 내보내기

- torch.export · ONNX, 전처리 포함 여부

## 추론 최적화

- 양자화 · pruning · distillation의 개념

## 서빙

- batch 추론 · 온라인 추론, 임베디드·엣지 환경

## 추론 계약의 유지

- 학습과 추론의 전처리 일치

## 실습

- 모델을 ONNX로 내보내고 원본과 출력 일치·속도를 비교한다.

## 실패 사례

- 내보낸 모델의 수치 결과를 검증하지 않는 것

## 통과 기준

- 배포된 모델이 학습 시 평가와 같은 결과를 냄을 검증한다.
