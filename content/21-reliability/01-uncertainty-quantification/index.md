---
id: reliability/uncertainty-quantification
title: "불확실성 정량화"
title_en: "Uncertainty Quantification"
tier: core
status: outline
prerequisites: [statistical-inference/distributional-formulation, bayesian-inference/bayesian-thinking]
---

# 불확실성 정량화

## 학습 목표

- aleatoric·epistemic 불확실성을 구분한다
- 딥러닝 불확실성 추정 방법을 구현한다

## 불확실성의 종류

- aleatoric · epistemic, 둘이 관측값에 함께 섞여 나타나는 문제

## 확률적 출력

- 분포 파라미터 head, 이분산 손실의 학습 불안정성

## 앙상블과 근사 베이지안

- deep ensemble · MC dropout · Laplace 근사 · SWAG · BNN

## 불확실성 전파

- 입력 불확실성의 전파, Monte Carlo 전파

## 실습

- 데이터 밀도·관측 잡음·도메인 변화가 다른 영역에서 앙상블과 MC dropout의 불확실성을 비교한다.

## 실패 사례

- softmax 확률을 불확실성으로 그대로 쓰는 것
- 앙상블 크기가 너무 작은 것

## 통과 기준

- 예측 불확실성을 aleatoric과 epistemic으로 분해해 보고한다.
