---
id: statistical-inference/distributional-formulation
title: "분포 기반 문제 정의"
title_en: "Formulating Problems with Distributions"
tier: core
status: outline
prerequisites: [statistical-inference/likelihood]
---

# 분포 기반 문제 정의

## 학습 목표

- 문제에 필요한 분포가 무엇인지 정한다
- 점 예측을 분포에 대한 의사결정으로 이해한다
- 분포를 출력하는 모델을 설계한다

## 무엇의 분포가 필요한가

- p(x,y) · p(y|x) · p(x) · p(z|x) · p(θ|D)의 구분
- 점 예측 ŷ=f(x), 조건부 예측분포 p(y|x), 파라미터 사후분포 p(θ|D)

## 결정 이론

- 손실과 최적 결정: 제곱손실 → 조건부 평균, 절댓값손실 → 조건부 중앙값, pinball loss → 분위수

## 분포를 출력하는 모델

- 이분산(heteroscedastic) 회귀, 분위수 회귀, mixture density network, 분포 파라미터 head

## 적정 점수 규칙

- proper scoring rule의 개념(NLL · CRPS), 평가 부와의 연결

## 불확실성의 출처 구분

- 관측 잡음 · 파라미터 불확실성 · 모델 구조 불확실성

## 실습

- 같은 데이터에 MSE 회귀·이분산 가우시안 회귀·분위수 회귀를 학습하고 출력이 무엇을 의미하는지 비교한다.

## 실패 사례

- 다봉 분포에서 조건부 평균을 대표값으로 쓰는 것
- 예측 구간과 신뢰 구간을 혼동하는 것

## 통과 기준

- 자신의 연구 문제를 '어떤 분포를 학습하고 어떤 결정에 쓰는가'로 다시 쓴다.
