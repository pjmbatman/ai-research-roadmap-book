---
id: generative-and-foundation-models/foundation-models
title: "사전학습·전이·Foundation Model"
title_en: "Pretraining, Transfer, and Foundation Models"
tier: advanced
status: outline
prerequisites: [generative-and-foundation-models/self-supervised-learning, information-and-learning-theory/modern-generalization]
---

# 사전학습·전이·Foundation Model

## 학습 목표

- foundation model의 조건을 이해한다
- 전이 방법을 선택한다
- 전이 효과와 실패 조건을 평가한다

## foundation model이란

- 다양한 데이터와 과제에 대한 사전학습, 큰 모델과의 구분

## 데이터 통합

- 이질적 데이터 · modality · 단위 · 프로토콜

## 스케일링

- 데이터 · 모델 · 계산 예산의 배분

## 전이 방법

- full fine-tuning · PEFT(LoRA · adapter) · frozen feature · prompting

## 도메인 적응과 부정적 전이

- domain adaptation, negative transfer 진단

## 평가

- task suite, 데이터 중복 검증, 작은 데이터에서의 전이 효과

## 실습

- 공개 사전학습 모델을 frozen·LoRA·full fine-tuning으로 전이해 데이터 크기별 성능을 비교한다.

## 실패 사례

- 큰 모델이라는 이유만으로 foundation model이라 부르는 것

## 통과 기준

- 전이 범위와 검증 과제를 명시해 foundation model 주장을 평가한다.
