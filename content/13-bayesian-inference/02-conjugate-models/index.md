---
id: bayesian-inference/conjugate-models
title: "켤레 모델과 베이지안 선형회귀"
title_en: "Conjugate Models and Bayesian Linear Regression"
tier: core
status: outline
prerequisites: [bayesian-inference/bayesian-thinking, probability/multivariate-gaussian]
---

# 켤레 모델과 베이지안 선형회귀

## 학습 목표

- 켤레 사전을 사용해 사후를 닫힌 형태로 구한다
- 베이지안 선형회귀의 예측 불확실성을 계산한다

## 켤레성

- Beta–Bernoulli, Gamma–Poisson, Normal–Normal, 지수족과 켤레

## 베이지안 선형회귀

- 가중치 사후분포, 예측분포, 데이터 밀도에 따른 불확실성

## evidence와 모델 선택

- marginal likelihood, 하이퍼파라미터 추정(empirical Bayes)

## 실습

- 베이지안 선형회귀를 구현하고 데이터가 없는 구간에서 예측 불확실성이 커지는 것을 확인한다.

## 실패 사례

- 관측 잡음 분산을 고정해 불확실성을 과소평가하는 것

## 통과 기준

- 베이지안 선형회귀의 예측분포를 유도하고 구현한다.
