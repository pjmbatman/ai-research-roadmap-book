---
id: architectures/recurrent-and-ssm
title: "순환 모델과 상태공간 모델"
title_en: "Recurrent and State-Space Models"
tier: core
status: outline
prerequisites: [architectures/inductive-bias, linear-algebra/eigen-and-symmetric]
---

# 순환 모델과 상태공간 모델

## 학습 목표

- RNN·LSTM·GRU의 동작을 이해한다
- 상태공간 모델 계열을 안다
- rollout 평가를 수행한다

## RNN

- hidden state, BPTT, gradient 소실과 폭발

## LSTM과 GRU

- gate 구조, 장기 의존성

## 시간 convolution

- TCN, causal convolution

## 상태공간 모델

- 선형 SSM, S4 · Mamba 계열의 아이디어, 병렬 학습과 순차 추론

## 학습과 추론 방식

- teacher forcing과 autoregressive rollout, exposure bias

## 불규칙 샘플링

- 결측 관측, 시간 간격 입력, 연속시간 모델과의 연결

## 실습

- 같은 모델을 one-step과 free rollout으로 평가하고 오차 누적을 분석한다.

## 실패 사례

- teacher forcing 성능만 보고 장기 예측 성능을 주장하는 것

## 통과 기준

- 시계열 모델의 평가를 one-step과 rollout으로 나누어 보고한다.
