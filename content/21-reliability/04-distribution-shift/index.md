---
id: reliability/distribution-shift
title: "분포 이동과 OOD"
title_en: "Distribution Shift and Out-of-Distribution"
tier: core
status: outline
prerequisites: [reliability/uncertainty-quantification, data/splitting-and-leakage]
---

# 분포 이동과 OOD

## 학습 목표

- 분포 이동의 유형을 구분한다
- OOD 탐지와 일반화를 평가한다
- 운영 중 drift에 대응한다

## 분포 이동의 유형

- covariate shift · label shift · conditional shift

## OOD 탐지와 OOD 일반화

- 두 과제의 차이

## 일반화 전략

- domain generalization · domain adaptation, 불변 표현

## 조건별 평가

- 집단별 성능과 worst-group 성능

## 강건성

- 센서 결측 · 잡음 · 이상치에 대한 강건성 시험

## 운영 중 drift

- drift 탐지, 재학습 정책

## 실습

- 새로운 셀·온도·충방전 조건·측정 장비 각각에 대해 일반화를 따로 평가하는 실험을 설계한다.

## 실패 사례

- 어떤 종류의 일반화를 검증하는지 명시하지 않는 것

## 통과 기준

- 연구 주장에 포함된 일반화 대상을 명시하고 각각을 검증한다.
