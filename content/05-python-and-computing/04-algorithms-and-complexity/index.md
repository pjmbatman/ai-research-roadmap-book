---
id: python-and-computing/algorithms-and-complexity
title: "자료구조·알고리즘·계산 비용"
title_en: "Data Structures, Algorithms, and Cost"
tier: core
status: outline
prerequisites: [python-and-computing/python-language]
---

# 자료구조·알고리즘·계산 비용

## 학습 목표

- 자료구조별 연산 비용을 안다
- 데이터 크기에 따른 병목을 예측한다
- 간단한 profiling을 수행한다

## 자료구조

- list · tuple · dict · set · deque · heap의 연산 비용

## 복잡도

- 시간·공간 복잡도, amortized cost

## 대용량 처리 패턴

- 정렬 · 탐색 · 샘플링 · streaming · chunking

## 희소와 밀집

- sparse 표현이 유리한 경우

## profiling 기초

- time · timeit · cProfile · line profiler · memory profiler

## 실습

- 데이터 로더의 중복 탐색이나 반복 파일 읽기 병목을 profiling으로 찾아 개선한다.

## 실패 사례

- list에서 반복적으로 membership을 검사하는 것
- 측정 없이 최적화하는 것

## 통과 기준

- 개선 전후를 측정해 병목과 개선 효과를 숫자로 보고한다.
