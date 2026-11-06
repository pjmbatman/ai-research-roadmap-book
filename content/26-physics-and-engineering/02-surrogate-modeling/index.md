---
id: physics-and-engineering/surrogate-modeling
title: "시뮬레이션 대리모델"
title_en: "Surrogate Models for Simulation"
tier: core
status: outline
prerequisites: [physics-and-engineering/simulation-fundamentals, scientific-ml/neural-operators]
---

# 시뮬레이션 대리모델

## 학습 목표

- 문제에 맞는 대리모델 구조를 선택한다
- 대리모델의 정확도와 속도 이득을 공정하게 측정한다

## 대리모델의 종류

- 스칼라 출력 회귀, 장(field) 예측, 시간 전개(rollout)

## 구조 선택

- 격자 기반 CNN, 메시 그래프 네트워크, neural operator, 축소 모델(POD)

## 다중 충실도

- 저해상도·고해상도 데이터의 결합

## 속도 이득의 측정

- 같은 정확도에서의 solver 대비 시간, 학습 비용 포함 여부

## 실습

- 같은 시뮬레이션 데이터셋에 POD 기반 모델과 신경망 대리모델을 학습해 정확도·추론 속도·학습 비용을 비교한다.

## 실패 사례

- solver의 정확도 설정을 낮춘 비교 없이 속도 이득을 주장하는 것

## 통과 기준

- 대리모델의 정확도–속도 trade-off를 solver와 같은 조건에서 보고한다.
