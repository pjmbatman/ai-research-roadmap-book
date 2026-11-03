---
id: battery-ai/battery-models
title: "배터리 모델의 계층"
title_en: "The Hierarchy of Battery Models"
tier: core
status: outline
prerequisites: [battery-ai/battery-fundamentals, numerical-methods/pde-and-dae]
---

# 배터리 모델의 계층

## 학습 목표

- ECM부터 DFN까지 모델의 수준과 가정을 이해한다
- 시뮬레이션 도구를 사용한다

## 등가회로 모델(ECM)

- OCV · 저항 · RC 쌍, 파라미터 추정

## 전기화학 모델

- SPM · SPMe · DFN의 가정과 계산 비용

## 열 모델과 열화 모델

- 열 결합, 경험적·물리적 열화 모델

## DAE 구조

- 미분 상태와 대수 변수, 일관된 초기조건

## 시뮬레이션 도구

- PyBaMM 등, 파라미터 세트, solver 설정

## 실습

- 같은 운전 프로파일을 ECM·SPM·DFN으로 시뮬레이션해 정확도와 계산 시간을 비교한다.

## 실패 사례

- 모델의 적용 범위를 넘는 C-rate·온도에서 결과를 믿는 것

## 통과 기준

- 연구 목적에 맞는 배터리 모델 수준을 선택하고 근거를 말한다.
