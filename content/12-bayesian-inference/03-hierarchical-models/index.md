---
id: bayesian-inference/hierarchical-models
title: "계층 모델"
title_en: "Hierarchical Models"
tier: advanced
status: outline
prerequisites: [bayesian-inference/conjugate-models]
---

# 계층 모델

## 학습 목표

- 그룹 구조를 계층 모델로 표현한다
- partial pooling을 이해한다

## 그룹 구조의 데이터

- 셀 · 배치 · 장비 · 실험실 단위의 변동

## pooling의 세 가지

- complete pooling · no pooling · partial pooling

## 계층 모델의 구성

- 하이퍼사전, 수축(shrinkage)

## 계산 문제

- funnel 문제와 재매개변수화

## 실습

- 여러 셀의 열화율을 no pooling과 partial pooling으로 추정해 데이터가 적은 셀에서의 차이를 비교한다.

## 실패 사례

- 그룹 구조를 무시하고 모든 샘플을 독립으로 취급하는 것

## 통과 기준

- 그룹 구조가 있는 자신의 데이터에 계층 모델을 설계한다.
