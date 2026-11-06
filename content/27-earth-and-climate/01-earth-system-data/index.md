---
id: earth-and-climate/earth-system-data
title: "기상·기후 데이터의 기초"
title_en: "Foundations of Weather and Climate Data"
tier: core
status: outline
prerequisites: [data/storage-formats]
---

# 기상·기후 데이터의 기초

## 학습 목표

- 관측·재분석·모델 출력 데이터를 구분한다
- 구면 격자 데이터를 다룬다

## 데이터의 종류

- 관측(지상·위성) · 재분석 자료(예: ERA5) · 수치 예보·기후 모델 출력

## 격자와 좌표

- 위경도 격자, 면적 가중, 연직 좌표, 해상도

## 변수와 단위

- 기온·기압·바람·습도, 지위고도, 누적량과 순간량

## 저장 형식과 도구

- NetCDF·Zarr, xarray, 대용량 시공간 데이터 접근

## 실습

- 재분석 자료의 한 변수를 내려받아 면적 가중 전구 평균과 기후값(climatology)을 계산한다.

## 실패 사례

- 위경도 격자에서 면적 가중 없이 평균을 내는 것

## 통과 기준

- 기상 데이터셋의 출처·격자·변수·단위를 문서화하고 기본 통계를 올바르게 계산한다.
