---
id: scientific-discovery/simulation-based-inference
title: "시뮬레이션 기반 추론"
title_en: "Simulation-Based Inference"
tier: advanced
status: outline
prerequisites: [scientific-ml/inverse-problems-and-uq, generative-and-foundation-models/generative-modeling]
---

# 시뮬레이션 기반 추론

## 학습 목표

- 우도를 쓸 수 없는 시뮬레이터로 파라미터를 추론한다
- 신경망 기반 사후 추정을 평가한다

## 우도 없는 추론

- 시뮬레이터는 샘플을 만들 수 있지만 우도를 계산할 수 없다

## 고전적 방법

- ABC(approximate Bayesian computation), 요약 통계

## 신경망 기반 추론

- 신경 사후 추정 · 신경 우도 추정 · 비율 추정, amortized inference

## 진단

- 보정(coverage) 검사, 시뮬레이터 오차(model misspecification)

## 실습

- 간단한 확률적 시뮬레이터의 파라미터를 ABC와 신경 사후 추정으로 각각 추론하고 coverage를 검사한다.

## 실패 사례

- 시뮬레이터가 현실과 다를 때 사후분포를 그대로 믿는 것

## 통과 기준

- 추론 결과를 coverage 진단과 함께 보고한다.
