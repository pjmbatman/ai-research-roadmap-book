---
id: information-and-learning-theory/objective-design
title: "목적함수 설계의 원리"
title_en: "Principles of Objective Design"
tier: core
status: outline
prerequisites: [information-and-learning-theory/entropy-and-kl, statistical-inference/likelihood]
---

# 목적함수 설계의 원리

## 학습 목표

- 손실·우도·divergence의 관계로 목적함수를 설계한다
- 손실의 스케일과 집계 방식을 관리한다

## 손실 = 음의 로그우도

- 확률모형의 선택이 손실을 결정한다

## divergence 최소화로서의 학습

- MLE ↔ forward KL, 생성모델 목적함수의 분류

## ELBO

- 잠재변수 모델의 목적함수

## 대조적 목적함수

- contrastive objective의 기본 해석

## 다중 손실의 결합

- 가중치 설정, 불확실성 기반 가중치, 단위와 스케일

## 집계 방식

- sum · mean · mask 기반 평균, batch 크기·시퀀스 길이에 따른 효과

## 실습

- 다중 손실 학습에서 손실 스케일과 집계 방식을 바꾸어 학습 결과가 어떻게 달라지는지 실험한다.

## 실패 사례

- padding을 포함해 평균을 내는 것
- 단위가 다른 손실을 그대로 더하는 것

## 통과 기준

- 자신의 모델 손실을 확률모형으로 해석하고 집계 방식을 정당화한다.
