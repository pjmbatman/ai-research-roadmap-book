---
id: architectures/inductive-bias
title: "귀납적 편향과 아키텍처 선택"
title_en: "Inductive Bias and Architecture Choice"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/neural-network-components]
---

# 귀납적 편향과 아키텍처 선택

## 학습 목표

- 귀납적 편향의 개념을 이해한다
- 데이터 구조에 맞는 아키텍처를 선택한다

## 귀납적 편향

- 가정이 일반화를 만든다, 국소성 · 가중치 공유 · 순서 · 대칭성

## 대칭성

- 불변성과 등변성, 데이터 증강과 구조적 제약의 비교

## MLP

- 아무 구조도 가정하지 않는 모델의 장단점

## 선택 기준

- 데이터 구조 · 데이터 크기 · 계산 예산 · 해석 요구

## 실습

- 같은 데이터에 구조적 가정이 다른 세 모델을 학습해 데이터 크기에 따른 성능 곡선을 비교한다.

## 실패 사례

- 최신 구조가 항상 낫다고 가정하는 것

## 통과 기준

- 자신의 데이터 구조에 근거해 아키텍처를 선택하고 대안을 기각한 이유를 말한다.
