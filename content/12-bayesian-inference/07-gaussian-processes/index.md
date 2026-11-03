---
id: bayesian-inference/gaussian-processes
title: "Gaussian Process"
title_en: "Gaussian Processes"
tier: core
status: outline
prerequisites: [probability/multivariate-gaussian, linear-algebra/function-spaces]
---

# Gaussian Process

## 학습 목표

- 함수에 대한 사전분포로서 GP를 이해한다
- GP 회귀를 구현한다
- 커널과 하이퍼파라미터를 선택한다

## 함수에 대한 분포

- 평균함수와 공분산 함수, 사전 샘플

## GP 회귀

- 사후 평균과 분산, 잡음 모델

## 커널

- RBF · Matérn · periodic · 커널 조합, 길이척도의 의미

## 하이퍼파라미터

- marginal likelihood 최대화

## 확장

- 계산 복잡도와 sparse GP, 비가우시안 우도, multi-output GP

## 실습

- GP 회귀를 직접 구현하고 커널·길이척도에 따른 예측과 불확실성을 비교한다.

## 실패 사례

- 입력 스케일을 맞추지 않고 등방성 커널을 쓰는 것
- 외삽 영역의 불확실성을 과신하는 것

## 통과 기준

- 소규모 공학 데이터에서 GP 대리모델을 만들고 예측 구간을 평가한다.
