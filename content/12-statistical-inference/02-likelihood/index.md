---
id: statistical-inference/likelihood
title: "우도와 최대우도"
title_en: "Likelihood and Maximum Likelihood"
tier: core
status: outline
prerequisites: [statistical-inference/estimators]
---

# 우도와 최대우도

## 학습 목표

- 확률과 우도를 구분한다
- MLE를 유도하고 계산한다
- 손실함수를 우도로 해석한다

## 확률과 우도

- 같은 식, 다른 변수: p(D|θ)를 θ의 함수로 보기

## 최대우도추정

- log-likelihood, NLL 최소화, 지수족에서의 MLE

## 손실함수의 확률적 해석

- MSE ↔ 가우시안 우도, cross-entropy ↔ categorical 우도, L1 ↔ Laplace 우도

## MLE의 성질과 한계

- 점근적 성질, 과적합, 소표본

## MAP와 정규화

- 사전분포로서의 정규화 항(베이지안 부와 연결)

## 실습

- 회귀 문제를 가우시안 우도로 정의하고 MLE를 수치 최적화로 구한 뒤 최소제곱 해와 비교한다.

## 실패 사례

- 우도를 θ의 확률로 해석하는 것

## 통과 기준

- MSE와 cross-entropy를 각각 가정한 확률모형에서 유도한다.
