---
id: numerical-methods/ode-solvers
title: "ODE 수치해법"
title_en: "Numerical Methods for ODEs"
tier: core
status: outline
prerequisites: [numerical-methods/nonlinear-equations]
---

# ODE 수치해법

## 학습 목표

- 명시적·암시적 방법을 구분한다
- stiffness를 진단한다
- solver 설정이 결과에 주는 영향을 점검한다

## 초기값 문제

- 존재와 유일성, 상태공간 표현

## 기본 방법

- Euler · Runge–Kutta, 국소·전역 오차와 차수

## 암시적 방법

- backward Euler · BDF, Newton 반복과의 결합

## stiffness

- 서로 다른 시간 척도, 안정 영역, stiff solver의 필요성

## 적응적 step과 tolerance

- rtol · atol의 의미, 이벤트 감지

## 경계값 문제

- shooting method 개요

## 실습

- stiff한 화학 반응 ODE를 explicit RK와 BDF로 풀어 step 수·시간·정확도를 비교한다.

## 실패 사례

- solver의 기본 tolerance를 그대로 믿는 것
- stiff 문제에 explicit 방법을 쓰는 것

## 통과 기준

- solver·tolerance를 바꾸어 결과가 수렴하는지 확인하는 절차를 수행한다.
