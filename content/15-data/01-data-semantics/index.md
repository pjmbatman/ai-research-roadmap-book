---
id: data/data-semantics
title: "데이터의 의미와 구조"
title_en: "Data Semantics and Structure"
tier: core
status: outline
prerequisites: [research-landscape/problem-formulation]
---

# 데이터의 의미와 구조

## 학습 목표

- sample·entity·group·trajectory를 정의한다
- label의 생성 과정을 추적한다
- 데이터 사전을 작성한다

## 데이터의 단위

- sample · entity · group · trajectory · timestamp, 무엇이 독립인가

## label의 생성 과정

- 직접 측정 · 계산된 값 · 사람의 판단, 측정 불확실성과 label 잡음

## schema와 metadata

- 컬럼 의미 · 단위 · 범위 · 결측 규칙, provenance

## 원본 보존

- raw · interim · processed의 구분, 원본은 수정하지 않는다

## 실습

- 연구 데이터셋의 데이터 사전(컬럼 의미 · 단위 · 출처 · 결측 규칙)과 label 생성 과정 문서를 작성한다.

## 실패 사례

- 계산된 label을 측정값처럼 취급하는 것
- 단위가 섞인 데이터를 병합하는 것

## 통과 기준

- 데이터 사전만으로 다른 연구자가 데이터를 올바르게 사용한다.
