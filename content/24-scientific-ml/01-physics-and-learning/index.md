---
id: scientific-ml/physics-and-learning
title: "물리 모델과 학습 문제"
title_en: "Physical Models as Learning Problems"
tier: core
status: outline
prerequisites: [numerical-methods/ode-solvers, research-landscape/problem-formulation]
---

# 물리 모델과 학습 문제

## 학습 목표

- 물리 모델의 변수를 학습 문제의 변수에 대응시킨다
- 오차의 출처를 분해한다

## 물리 모델의 구성

- 지배 방정식 · 구성 관계 · 경계·초기 조건

## 변수의 역할

- 상태 · 파라미터 · 입력 · 관측

## 문제의 종류

- forward · inverse · 대리모델 · 발견(discovery)

## 단위와 무차원화

- 단위 일관성, 무차원화와 스케일링

## 물리적 제약

- 보존 법칙 · 대칭성 · 양수성 · 단조성

## 오차의 분해

- 데이터 오차 · 모델 오차 · 수치 오차, 시뮬레이터–실험 불일치

## 실습

- 물리 모델 하나의 각 변수를 AI 모델의 입력·출력·파라미터에 대응시키고 오차 출처를 분석한다.

## 실패 사례

- 무차원화 없이 크기가 다른 물리량을 함께 학습하는 것

## 통과 기준

- 물리 기반 학습 문제를 forward·inverse·대리모델 중 하나로 명확히 정의한다.
