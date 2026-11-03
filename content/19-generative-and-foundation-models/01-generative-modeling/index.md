---
id: generative-and-foundation-models/generative-modeling
title: "생성모델의 공통 문제 정의"
title_en: "The Generative Modeling Problem"
tier: core
status: outline
prerequisites: [probability/distributions, information-and-learning-theory/entropy-and-kl]
---

# 생성모델의 공통 문제 정의

## 학습 목표

- 생성모델이 무엇의 분포를 학습하는지 정의한다
- 계열별 공통점과 차이를 비교한다

## 무엇의 분포인가

- p(x) · p(x|c), 무조건·조건부 생성

## 세 가지 능력

- likelihood 계산 · 샘플링 · 잠재 표현, 계열별로 무엇을 할 수 있는가

## 생성 · 재구성 · 예측

- 세 과제의 차이

## 계열 지도

- 자기회귀 · 잠재변수 · 적대적 · flow · 확산 · flow matching의 비교표

## 실습

- 2차원 장난감 분포에서 여러 계열의 생성모델 결과를 같은 평가로 비교한다.

## 실패 사례

- 보기 좋은 샘플을 분포 학습의 증거로 삼는 것

## 통과 기준

- 생성모델 계열을 세 가지 능력과 학습 목적함수로 비교하는 표를 작성한다.
