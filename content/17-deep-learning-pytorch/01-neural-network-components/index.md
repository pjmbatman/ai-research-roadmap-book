---
id: deep-learning-pytorch/neural-network-components
title: "신경망의 구성 요소"
title_en: "Building Blocks of Neural Networks"
tier: core
status: outline
prerequisites: [linear-algebra/affine-and-nonlinear, calculus-and-autodiff/automatic-differentiation]
---

# 신경망의 구성 요소

## 학습 목표

- 신경망의 각 구성 요소의 역할을 설명한다
- 초기화와 정규화가 학습에 주는 영향을 이해한다

## 층과 활성화

- 아핀 변환 + 활성화, ReLU · GELU · SiLU · tanh

## 깊이와 폭

- 파라미터 수 계산, 표현력과 학습 용이성

## 초기화

- Xavier · He 초기화, 분산 보존의 원리

## 정규화 층

- BatchNorm · LayerNorm · RMSNorm, train/eval 동작 차이

## residual connection과 dropout

- gradient 흐름, 정규화 효과

## output head

- 과제별 출력과 손실, multi-head · multi-task 구조

## 실습

- 같은 파라미터 예산에서 깊이·폭·활성화·초기화를 바꾸어 활성값과 gradient의 분포를 층별로 관찰한다.

## 실패 사례

- 초기화를 바꾸고 학습 실패를 모델 구조 탓으로 돌리는 것

## 통과 기준

- 층별 활성값·gradient 통계로 학습이 잘 되는 설정과 아닌 설정을 구분한다.
