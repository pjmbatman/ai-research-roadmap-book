---
id: python-and-computing/computer-architecture
title: "컴퓨터 구조와 병렬 계산"
title_en: "Computer Architecture and Parallelism"
tier: core
status: outline
prerequisites: [python-and-computing/algorithms-and-complexity]
---

# 컴퓨터 구조와 병렬 계산

## 학습 목표

- CPU·GPU·메모리 계층을 설명한다
- process·thread·GIL을 이해한다
- 병목의 종류를 구분한다

## 하드웨어 구성

- CPU · GPU · RAM · VRAM · storage, 메모리 계층과 대역폭

## GPU의 계산 모델

- SIMT, kernel, 메모리 대역폭과 연산 강도(roofline 직관)

## 병렬성과 동시성

- process · thread · GIL, multiprocessing, async의 위치

## 병목의 종류

- I/O · 연산 · 메모리 대역폭 · CPU–GPU 전송 · 동기화

## 수의 표현

- 정수·부동소수점 표현 개요(자세한 내용은 수치해석 부)

## 실습

- 'GPU 사용률이 낮다'는 현상을 데이터 공급·전송·연산·동기화 가설로 나누어 실험으로 검증한다.

## 실패 사례

- GPU가 항상 빠르다고 가정하는 것
- 비동기 GPU 연산의 시간을 잘못 측정하는 것

## 통과 기준

- 성능 문제를 병목 종류로 분류하고 측정 근거를 제시한다.
