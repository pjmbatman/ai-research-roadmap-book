---
id: generative-and-foundation-models/self-supervised-learning
title: "자기지도학습"
title_en: "Self-Supervised Learning"
tier: core
status: outline
prerequisites: [architectures/inductive-bias, information-and-learning-theory/mutual-information]
---

# 자기지도학습

## 학습 목표

- 주요 자기지도학습 방법을 이해한다
- pretext 과제와 증강을 설계한다
- 사전학습 효과를 공정하게 평가한다

## 방법 계열

- masked reconstruction · contrastive · non-contrastive · predictive(JEPA류)

## 설계 요소

- 증강, positive · negative 쌍, 마스킹 전략

## 실패 양상

- representation collapse, shortcut learning, pretext–downstream 간극

## 평가

- linear probe · fine-tuning · few-shot, 분할과 데이터 중복 점검

## 실습

- 사전학습 효과를 무작위 초기화·동일 데이터·동일 downstream 예산과 비교한다.

## 실패 사례

- 사전학습 데이터와 평가 데이터가 겹치는 것
- downstream 튜닝 예산이 다른 비교

## 통과 기준

- 자기지도 사전학습의 효과를 통제된 비교로 보인다.
