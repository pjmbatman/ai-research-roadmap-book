---
id: python-and-computing/scientific-python
title: "NumPy·pandas·SciPy"
title_en: "Scientific Python: NumPy, pandas, SciPy"
tier: core
status: outline
prerequisites: [python-and-computing/python-language]
---

# NumPy·pandas·SciPy

## 학습 목표

- 배열의 shape·axis·broadcasting을 정확히 다룬다
- 벡터화로 계산을 표현한다
- 표 데이터와 시간 데이터를 처리한다

## NumPy 배열

- shape · axis · dtype · stride, view와 copy

## indexing

- slicing · fancy indexing · boolean mask

## broadcasting과 벡터화

- broadcasting 규칙, loop 제거, einsum

## pandas

- index · merge · groupby · pivot, 시간 인덱스와 resampling

## 대안 도구

- Polars · DuckDB · Arrow의 위치

## SciPy

- optimize · interpolate · integrate · sparse · signal · stats

## 실습

- 같은 계산을 Python loop와 벡터화로 작성해 시간·메모리·수치 결과를 비교한다.

## 실패 사례

- view를 수정해 원본이 바뀌는 것
- broadcasting으로 의도하지 않은 shape가 조용히 만들어지는 것

## 통과 기준

- shape 주석만으로 복잡한 배열 연산의 결과 shape를 예측한다.
