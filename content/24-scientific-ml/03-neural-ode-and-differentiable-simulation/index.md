---
id: scientific-ml/neural-ode-and-differentiable-simulation
title: "Neural ODE와 미분가능 시뮬레이션"
title_en: "Neural ODEs and Differentiable Simulation"
tier: advanced
status: outline
prerequisites: [scientific-ml/physics-and-learning, numerical-methods/ode-solvers]
---

# Neural ODE와 미분가능 시뮬레이션

## 학습 목표

- ODE solver를 통한 미분 방법을 이해한다
- universal differential equation을 구성한다

## Neural ODE

- 연속시간 동역학의 학습

## solver를 통한 미분

- discretize-then-optimize와 optimize-then-discretize, adjoint 방법

## universal differential equation

- 알려진 물리 + 학습되는 항

## 미분가능 시뮬레이터

- 시뮬레이터 파라미터의 gradient 기반 추정

## 수치적 주의점

- stiffness, gradient 정확도, 메모리

## 실습

- 알려진 물리 항과 미지의 항을 가진 ODE를 universal differential equation으로 학습한다.

## 실패 사례

- stiff 시스템에 explicit solver로 학습하는 것

## 통과 기준

- solver 설정이 gradient와 학습 결과에 주는 영향을 보고한다.
