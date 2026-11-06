---
id: life-sciences/bio-evaluation
title: "생명과학 AI의 평가와 누수"
title_en: "Evaluation and Leakage in Life-Science AI"
tier: core
status: outline
prerequisites: [life-sciences/sequence-models, data/splitting-and-leakage]
---

# 생명과학 AI의 평가와 누수

## 학습 목표

- 진화적 유사성에 의한 누수를 방지한다
- 계산 결과와 실험 검증의 관계를 이해한다

## 유사성 기반 누수

- 서열 동일성·구조 유사성 기반 분할, 군집 분할

## 벤치마크의 함정

- 시간 분할, 데이터베이스 중복, 사전학습 데이터와의 중복

## 실험 검증

- in silico 결과의 검증 경로, 성공률의 보고

## 재현성

- 데이터 버전, 소프트웨어 버전, 무작위성

## 실습

- 같은 과제를 무작위 분할과 서열 동일성 기준 군집 분할로 평가해 성능 차이를 측정한다.

## 실패 사례

- 서열 유사도를 고려하지 않은 분할로 일반화를 주장하는 것

## 통과 기준

- 유사성 기반 분할과 실험 검증 계획을 포함한 평가 설계를 제시한다.
