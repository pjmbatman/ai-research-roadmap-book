---
id: scientific-ml/inverse-problems-and-uq
title: "역문제와 불확실성 정량화"
title_en: "Inverse Problems and Uncertainty Quantification"
tier: advanced
status: outline
prerequisites: [scientific-ml/system-identification-and-estimation, bayesian-inference/mcmc]
---

# 역문제와 불확실성 정량화

## 학습 목표

- 역문제를 베이지안으로 정의한다
- 민감도 분석과 불확실성 전파를 수행한다

## 역문제

- ill-posedness, 정규화

## 베이지안 역문제

- 사전 · 우도 · 사후, 시뮬레이터 기반 추론(SBI)

## 민감도 분석

- 국소 민감도, 전역 민감도(Sobol 지수)

## 불확실성 전파

- 파라미터 불확실성이 예측에 미치는 영향

## 실습

- 시뮬레이터 파라미터를 측정 데이터로 베이지안 추정하고, 사후 불확실성을 예측으로 전파한다.

## 실패 사례

- 점추정 파라미터로 예측하고 불확실성을 보고하지 않는 것

## 통과 기준

- 역문제의 결과를 사후분포와 예측 구간으로 보고한다.
