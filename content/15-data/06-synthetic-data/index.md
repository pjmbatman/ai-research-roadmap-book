---
id: data/synthetic-data
title: "합성 데이터와 시뮬레이션 데이터"
title_en: "Synthetic and Simulated Data"
tier: advanced
status: outline
prerequisites: [data/dataset-versioning, numerical-methods/ode-solvers]
---

# 합성 데이터와 시뮬레이션 데이터

## 학습 목표

- 시뮬레이션 데이터셋을 설계한다
- 실험 설계로 입력 공간을 덮는다
- sim-to-real 간극을 평가한다

## 시뮬레이션 데이터의 생성

- 파라미터 · 초기조건 · 입력 프로파일, seed와 설정 기록

## 실험 설계

- grid · random · Latin hypercube · Sobol 수열, 커버리지 평가

## 합성 데이터의 위험

- 시뮬레이터 가정의 학습, 실제 분포와의 차이

## sim-to-real

- 도메인 간극 측정, 혼합 학습

## 실습

- 시뮬레이터로 Latin hypercube 기반 데이터셋을 생성하고, 실제 데이터와의 분포 차이를 측정한다.

## 실패 사례

- 시뮬레이터의 기본 파라미터 하나로 생성한 데이터로 일반화를 주장하는 것

## 통과 기준

- 데이터셋 생성 설정을 문서화하고 다른 사람이 동일한 데이터셋을 다시 만든다.
