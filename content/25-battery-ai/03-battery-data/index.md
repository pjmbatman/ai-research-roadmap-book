---
id: battery-ai/battery-data
title: "배터리 데이터셋과 데이터 함정"
title_en: "Battery Datasets and Data Pitfalls"
tier: core
status: outline
prerequisites: [battery-ai/battery-fundamentals, data/splitting-and-leakage]
---

# 배터리 데이터셋과 데이터 함정

## 학습 목표

- 공개 배터리 데이터셋의 특성을 비교한다
- 배터리 데이터 특유의 함정을 피한다

## 공개 데이터셋

- 셀 · 화학 · 프로토콜 · 규모 · 라이선스 비교, 통합 스키마 데이터 허브

## 이질성

- 시험기관 · 장비 · 프로토콜 · 샘플링 주기 차이

## 분할

- 셀 단위 · 배치 단위 · 프로토콜 단위 분할, 사이클 단위 분할의 누수

## censoring과 중단

- 수명 끝에 도달하지 않은 셀

## 품질 문제

- 측정 오류 · 결측 사이클 · 단위 · 부호

## 실습

- 여러 공개 데이터셋을 하나의 스키마로 통합하고, 셀 단위 분할과 품질 점검 보고서를 만든다.

## 실패 사례

- 같은 셀의 사이클이 학습·테스트에 섞이는 것
- 데이터셋 간 SOH 정의 차이를 무시하는 것

## 통과 기준

- 통합 데이터셋의 schema·분할·품질 문서를 작성한다.
