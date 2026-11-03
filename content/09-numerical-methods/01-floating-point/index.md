---
id: numerical-methods/floating-point
title: "부동소수점과 수치 오차"
title_en: "Floating Point and Numerical Error"
tier: core
status: outline
prerequisites: [python-and-computing/computer-architecture]
---

# 부동소수점과 수치 오차

## 학습 목표

- 부동소수점 표현과 정밀도를 이해한다
- 수치적으로 안정한 계산식을 작성한다

## 부동소수점 표현

- IEEE 754, 지수와 가수, machine epsilon
- fp64 · fp32 · tf32 · fp16 · bf16 · fp8의 범위와 정밀도

## 오차의 종류

- 반올림 오차 · 절단 오차 · 누적 오차, 절대·상대 오차

## 위험한 연산

- catastrophic cancellation, overflow · underflow, 비결합성(합의 순서)

## 안정한 계산식

- log-sum-exp, 안정한 softmax · sigmoid · log1p · expm1, 분산의 안정한 계산

## 실습

- naive softmax와 log-sum-exp 기반 softmax, 큰 수의 합을 여러 순서로 계산해 오차를 비교한다.

## 실패 사례

- fp16에서 gradient가 underflow하는 것
- 수학적으로 같은 식이 수치적으로도 같다고 가정하는 것

## 통과 기준

- NaN이 발생한 학습에서 수치적 원인을 찾아 안정한 식으로 고친다.
