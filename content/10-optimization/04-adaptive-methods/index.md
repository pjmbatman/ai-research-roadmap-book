---
id: optimization/adaptive-methods
title: "적응적 방법과 학습 레시피"
title_en: "Adaptive Methods and Training Recipes"
tier: core
status: outline
prerequisites: [optimization/first-order-methods]
---

# 적응적 방법과 학습 레시피

## 학습 목표

- Adam 계열의 원리를 설명한다
- weight decay와 L2 정규화를 구분한다
- 학습률 스케줄과 실전 기법을 사용한다

## 적응적 방법

- AdaGrad · RMSProp · Adam · AdamW, bias correction

## weight decay

- L2 정규화와 decoupled weight decay의 차이

## 학습률 스케줄

- step · cosine · one-cycle, warm-up의 이유

## 실전 기법

- gradient clipping, gradient accumulation, batch size와 학습률의 관계

## 최신 optimizer의 위치

- Lion · Shampoo · Muon 등 새 방법을 평가하는 관점

## 공정한 optimizer 비교

- 각 optimizer의 학습률을 따로 튜닝해야 하는 이유

## 실습

- optimizer만 바꾼 실험과 학습률까지 튜닝한 비교 실험을 수행해 결론이 어떻게 달라지는지 보인다.

## 실패 사례

- Adam에 L2 정규화를 넣고 weight decay라 부르는 것
- 다른 optimizer를 같은 학습률로 비교하는 것

## 통과 기준

- 자신의 학습 설정에서 optimizer·학습률·스케줄을 근거를 갖고 선택한다.
