---
id: scientific-ml/pinn
title: "물리 정보 신경망(PINN)"
title_en: "Physics-Informed Neural Networks"
tier: advanced
status: outline
prerequisites: [scientific-ml/physics-and-learning, calculus-and-autodiff/automatic-differentiation]
---

# 물리 정보 신경망(PINN)

## 학습 목표

- PINN의 목적함수를 구성한다
- 학습 실패의 원인을 진단한다
- residual과 해 오차를 함께 평가한다

## 목적함수

- 데이터 손실 · 방정식 residual · 경계·초기 조건 손실

## 도함수 계산

- 자동미분으로 입력에 대한 미분 구하기

## 제약의 부과

- soft · hard constraint

## 학습의 어려움

- 손실 균형, collocation 샘플링, stiffness, spectral bias, 최적화 난이도

## 개선 기법

- 적응적 가중치 · 적응적 샘플링 · 좌표 스케일링 · L-BFGS 미세 최적화

## 평가

- residual 감소와 실제 해 오차의 관계

## 실습

- 정확해가 있는 문제에서 PINN을 학습하고 residual과 해 오차를 함께 추적한다.

## 실패 사례

- residual이 작으면 해가 맞다고 가정하는 것
- 고전 solver보다 느린 PINN을 장점처럼 서술하는 것

## 통과 기준

- PINN과 수치 solver를 정확도·계산 비용으로 공정하게 비교한다.
