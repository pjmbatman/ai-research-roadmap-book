---
id: physics-and-engineering/physics-benchmarks
title: "시뮬레이션 학습의 벤치마크와 평가"
title_en: "Benchmarks and Evaluation for Learned Simulation"
tier: core
status: outline
prerequisites: [physics-and-engineering/surrogate-modeling, reliability/distribution-shift]
---

# 시뮬레이션 학습의 벤치마크와 평가

## 학습 목표

- 학습된 시뮬레이터를 다면적으로 평가한다
- 공정한 비교 조건을 설계한다

## 정확도 지표

- 장 오차, 적분량(힘·유량) 오차, 스펙트럼

## 장기 rollout

- 오차 누적, 안정성, 통계적 정상 상태

## 물리적 일관성

- 보존량, 경계조건 만족, 양수성

## 일반화

- 새 형상·파라미터·해상도

## 비용

- 학습·추론 비용, solver와의 공정 비교

## 실습

- 학습된 시뮬레이터를 정확도·장기 안정성·보존량·일반화·비용의 다섯 축으로 평가한 표를 만든다.

## 실패 사례

- 한 step 오차만으로 시뮬레이터를 평가하는 것

## 통과 기준

- 다섯 축 평가표로 두 방법을 비교하고 각각의 약점을 말한다.
