---
id: scientific-ml/hybrid-and-surrogate-models
title: "하이브리드 모델과 대리모델"
title_en: "Hybrid and Surrogate Models"
tier: core
status: outline
prerequisites: [scientific-ml/physics-and-learning, bayesian-inference/gaussian-processes]
---

# 하이브리드 모델과 대리모델

## 학습 목표

- 물리–데이터 결합 방식을 비교한다
- 대리모델을 구축하고 검증한다

## 결합 방식

- 잔차 학습 · 파라미터 학습 · 물리 특징 · 구조 제약 · 직렬·병렬 결합

## 대리모델

- GP · 신경망 · 축소 모델(POD), 정확도·속도·불확실성

## 다중 충실도

- 저충실도 시뮬레이션과 고충실도 데이터 결합

## 능동 학습

- 다음 시뮬레이션·실험 선택

## 공정한 비교

- 데이터 전용 · 물리 전용 · 하이브리드를 같은 조건에서

## 실습

- 데이터 전용·물리 모델·하이브리드 모델을 동일한 문제 정의와 평가 조건에서 비교한다.

## 실패 사례

- 하이브리드 모델이 물리 모델의 오차를 데이터로 덮어 외삽에서 실패하는 것

## 통과 기준

- 결합 방식의 선택을 데이터 양·물리 지식의 신뢰도·외삽 요구로 정당화한다.
