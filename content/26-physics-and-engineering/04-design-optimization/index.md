---
id: physics-and-engineering/design-optimization
title: "형상·설계 최적화와 역설계"
title_en: "Design Optimization and Inverse Design"
tier: advanced
status: outline
prerequisites: [physics-and-engineering/surrogate-modeling, optimization/special-optimization]
---

# 형상·설계 최적화와 역설계

## 학습 목표

- 설계 최적화 문제를 정의한다
- 대리모델 기반 최적화의 위험을 관리한다

## 설계 문제의 정의

- 설계 변수 · 목적 · 제약, 파라미터화

## gradient 기반 설계

- adjoint 방법, 미분가능 시뮬레이션

## 대리모델 기반 설계

- 대리모델 최적화와 검증 루프, 베이지안 최적화

## 생성모델 기반 역설계

- 조건부 생성, 제약 만족

## 검증

- 최적 설계를 고충실도 시뮬레이션으로 재검증

## 실습

- 대리모델로 찾은 최적 설계를 고충실도 시뮬레이션으로 재검증해 대리모델 오차가 최적화에 미친 영향을 측정한다.

## 실패 사례

- 대리모델의 오차를 이용한 '가짜 최적해'를 그대로 보고하는 것

## 통과 기준

- 최적화 결과를 고충실도 재검증과 함께 보고한다.
