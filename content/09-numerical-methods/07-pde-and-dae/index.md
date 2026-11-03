---
id: numerical-methods/pde-and-dae
title: "PDE·DAE와 시뮬레이터 검증"
title_en: "PDEs, DAEs, and Simulator Verification"
tier: advanced
status: outline
prerequisites: [numerical-methods/ode-solvers]
---

# PDE·DAE와 시뮬레이터 검증

## 학습 목표

- PDE 이산화의 기본 방법을 안다
- DAE의 구조와 일관된 초기조건을 이해한다
- 시뮬레이터 결과를 검증한다

## PDE 이산화

- 유한차분 · 유한체적 · 유한요소의 기본 개념, method of lines

## 안정성 조건

- CFL 조건, 경계조건의 처리

## DAE

- 미분 상태와 대수 변수, index의 개념, 일관된 초기조건

## 검증과 확인(V&V)

- 격자·시간 간격 수렴 연구, 해석해와의 비교, 이산화 오차

## 시뮬레이터를 데이터 생성기로 쓸 때

- 시뮬레이터 결과는 정답이 아니라 근사다

## 실습

- 1차원 확산 방정식을 method of lines로 풀고 격자 수렴 연구를 수행한다.

## 실패 사례

- 시뮬레이터 출력을 절대적 정답으로 취급하는 것
- DAE에 비일관 초기조건을 넣는 것

## 통과 기준

- 시뮬레이터 결과에 대해 solver·mesh·tolerance 영향을 점검한 검증 보고서를 쓴다.
