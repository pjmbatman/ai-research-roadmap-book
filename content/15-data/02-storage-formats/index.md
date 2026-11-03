---
id: data/storage-formats
title: "저장 형식과 대용량 데이터"
title_en: "Storage Formats and Large Data"
tier: core
status: outline
prerequisites: [data/data-semantics]
---

# 저장 형식과 대용량 데이터

## 학습 목표

- 데이터 형식별 장단점을 안다
- 메모리보다 큰 데이터를 처리한다

## 형식 비교

- CSV · Parquet · Arrow · HDF5 · Zarr · NPY/NPZ · MAT · JSON Lines

## 열 지향 저장과 압축

- 열 단위 읽기, 압축 코덱, partitioning

## 대용량 처리

- chunking · memory mapping · lazy evaluation, DuckDB · Polars로 out-of-core 질의

## 데이터 접근 패턴

- 학습 시 random access와 순차 접근, 작은 파일 문제

## 실습

- 같은 시계열 데이터를 CSV·Parquet·HDF5로 저장해 크기·읽기 속도·부분 읽기 성능을 비교한다.

## 실패 사례

- 수십 GB CSV를 매 실험마다 파싱하는 것
- 수백만 개의 작은 파일

## 통과 기준

- 연구 데이터에 맞는 저장 형식과 접근 방식을 근거와 함께 선택한다.
