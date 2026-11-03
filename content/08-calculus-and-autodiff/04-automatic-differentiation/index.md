---
id: calculus-and-autodiff/automatic-differentiation
title: "자동미분과 역전파"
title_en: "Automatic Differentiation and Backpropagation"
tier: core
status: outline
prerequisites: [calculus-and-autodiff/matrix-calculus]
---

# 자동미분과 역전파

## 학습 목표

- 계산 그래프와 forward·reverse mode를 설명한다
- 작은 자동미분 엔진을 구현한다
- gradient를 검증하고 디버깅한다

## 미분 계산의 세 방법

- symbolic · numerical · automatic differentiation

## 계산 그래프

- 노드와 엣지, chain rule의 그래프 표현

## forward mode와 reverse mode

- JVP와 VJP, 입력·출력 차원에 따른 비용

## 역전파

- reverse mode로서의 역전파, gradient 계산과 parameter update의 구분

## 메모리와 고차미분

- 중간값 저장, 계산 그래프 유지, Hessian-vector product

## gradient 검증

- 유한차분 gradient check, 상대오차, 수치 정밀도

## gradient 소실·폭발

- 깊은 합성에서의 Jacobian 곱, 진단 방법

## 실습

- 스칼라 자동미분 엔진(micrograd 수준)을 직접 구현하고, 수치미분·손 유도·PyTorch autograd 결과를 비교한다.

## 실패 사례

- 역전파를 학습 전체와 동일시하는 것
- float32로 유한차분 검증을 해 잘못된 결론을 내리는 것

## 통과 기준

- 자신이 만든 엔진으로 작은 MLP를 학습하고 gradient를 PyTorch와 일치시킨다.
