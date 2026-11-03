---
id: numerical-methods/nonlinear-equations
title: "비선형 방정식과 반복법"
title_en: "Nonlinear Equations and Iterative Methods"
tier: core
status: outline
prerequisites: [numerical-methods/conditioning-and-stability]
---

# 비선형 방정식과 반복법

## 학습 목표

- 근 찾기 알고리즘을 구현한다
- Newton 방법의 수렴과 실패를 이해한다

## 근 찾기

- bisection · secant · Newton

## 고정점 반복

- 수축 사상, 수렴 조건

## 다변수 Newton

- Jacobian 기반 Newton, Newton–Krylov, damping과 line search

## 실패 양상

- 초기값 의존성, 발산, 특이 Jacobian

## 실습

- 비선형 연립방정식을 Newton 방법으로 풀고 초기값에 따른 수렴 영역을 시각화한다.

## 실패 사례

- Newton 방법이 항상 빠르게 수렴한다고 가정하는 것

## 통과 기준

- implicit ODE solver 내부에서 Newton 방법이 하는 역할을 설명한다.
