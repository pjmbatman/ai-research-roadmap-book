---
id: generative-and-foundation-models/diffusion-models
title: "확산 모델과 score 기반 모델"
title_en: "Diffusion and Score-Based Models"
tier: core
status: outline
prerequisites: [generative-and-foundation-models/generative-modeling, probability/stochastic-processes]
---

# 확산 모델과 score 기반 모델

## 학습 목표

- forward·reverse 과정을 이해한다
- denoising 목적함수를 유도한다
- 샘플링 방법을 비교한다

## forward 과정

- 잡음 스케줄, 닫힌 형태의 주변분포

## reverse 과정과 학습

- denoising 목적, ε · x0 · v 예측

## score 관점

- score matching, SDE · ODE 표현

## 샘플링

- ancestral · DDIM · ODE solver, step 수와 품질

## 조건부 생성

- classifier-free guidance

## 과학 데이터에의 적용

- 시계열 · 물리장 생성, 역문제

## 실습

- 2차원과 1차원 시계열 데이터에 작은 확산 모델을 학습하고 샘플링 step 수에 따른 품질을 평가한다.

## 실패 사례

- 학습 시와 샘플링 시 잡음 스케줄 불일치

## 통과 기준

- 확산 모델의 학습 목적과 샘플링을 구현하고 각 단계를 수식과 대응시킨다.
