---
id: experiment-operations/experiment-tracking
title: "실험 추적과 결과 관리"
title_en: "Experiment Tracking and Results Management"
tier: core
status: outline
prerequisites: [research-code-engineering/configuration, deep-learning-pytorch/dashboards]
---

# 실험 추적과 결과 관리

## 학습 목표

- 실험을 추적 가능하게 기록한다
- 결과와 artifact를 연결한다

## 무엇을 기록하는가

- run ID · 설정 · 코드 revision · 데이터 version · 분할 · seed · 환경

## 저장 구조

- outputs/<run-id>/ 아래 설정 · 로그 · checkpoint · 예측 · metric

## 추적 도구

- TensorBoard · MLflow · W&B의 역할 분담

## 결과 표 만들기

- 여러 run의 집계, 자동 생성되는 결과 표

## 실패와 중단

- 실패·중단·재개 실험의 상태 관리

## 실습

- 논문 표의 숫자 하나에서 해당 run·checkpoint·설정·데이터로 되돌아가는 경로를 만든다.

## 실패 사례

- 결과를 손으로 표에 옮기는 것
- 커밋하지 않은 코드로 실험하는 것

## 통과 기준

- 임의의 결과 수치에서 그 실험을 정확히 다시 실행한다.
