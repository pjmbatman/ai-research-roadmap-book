---
id: physics-and-engineering/turbulence-and-closure
title: "난류와 closure 모델 학습"
title_en: "Learning Turbulence and Closure Models"
tier: advanced
status: outline
prerequisites: [physics-and-engineering/surrogate-modeling, scientific-ml/hybrid-and-surrogate-models]
---

# 난류와 closure 모델 학습

## 학습 목표

- closure 문제를 학습 문제로 정의한다
- solver에 결합했을 때의 안정성을 평가한다

## closure 문제

- 해상하지 못하는 규모의 효과, RANS·LES의 모델링 항

## 학습 기반 closure

- 데이터 기반 보정 항, 물리 제약(대칭·실현성)

## solver와의 결합

- a priori 평가와 a posteriori 평가의 차이, 결합 후 불안정성

## 일반화

- 새 형상·Reynolds 수에서의 성능

## 실습

- 간단한 난류 모델의 보정 항을 학습하고, a priori 오차와 solver에 결합한 a posteriori 결과를 비교한다.

## 실패 사례

- a priori 오차가 작으면 결합 후에도 잘 동작한다고 가정하는 것

## 통과 기준

- closure 모델을 solver에 결합한 상태에서 안정성과 정확도를 보고한다.
