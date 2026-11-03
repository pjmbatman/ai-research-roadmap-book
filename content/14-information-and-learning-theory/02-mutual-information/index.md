---
id: information-and-learning-theory/mutual-information
title: "상호정보량과 표현"
title_en: "Mutual Information and Representations"
tier: advanced
status: outline
prerequisites: [information-and-learning-theory/entropy-and-kl]
---

# 상호정보량과 표현

## 학습 목표

- 상호정보량을 정의한다
- 표현학습 목적함수와 정보량의 관계를 이해한다

## 상호정보량

- 정의와 성질, data processing inequality

## 추정의 어려움

- 고차원 MI 추정, 하한(InfoNCE · MINE)

## 정보 병목

- information bottleneck의 아이디어와 논쟁

## 실습

- InfoNCE 하한으로 두 변수 사이의 MI를 추정하고 batch 크기에 따른 상한 효과를 확인한다.

## 실패 사례

- InfoNCE 값을 MI의 정확한 추정으로 해석하는 것

## 통과 기준

- contrastive learning 목적함수를 MI 하한으로 설명한다.
