---
id: physics-and-engineering/simulation-fundamentals
title: "ML 연구자를 위한 연속체 역학과 시뮬레이션"
title_en: "Continuum Mechanics and Simulation for ML Researchers"
tier: core
status: outline
prerequisites: [numerical-methods/pde-and-dae]
---

# ML 연구자를 위한 연속체 역학과 시뮬레이션

## 학습 목표

- 대표적인 지배 방정식과 무차원수를 읽는다
- 시뮬레이션 데이터가 만들어지는 과정을 이해한다

## 지배 방정식

- 질량·운동량·에너지 보존, Navier–Stokes, 탄성, 열전달

## 무차원수와 영역

- Reynolds·Péclet 수 등, 흐름의 영역(층류·난류)

## 시뮬레이션 파이프라인

- 형상 · 메시 · 경계조건 · solver · 후처리, 계산 비용의 구조

## 시뮬레이션 데이터

- 해상도, 저장 형식, 수렴 여부 확인

## 실습

- 공개 CFD 예제 하나를 해상도를 바꿔 실행하고 결과와 계산 시간을 비교한다.

## 실패 사례

- 수렴하지 않은 시뮬레이션 결과를 학습 데이터로 쓰는 것

## 통과 기준

- 시뮬레이션 데이터셋의 생성 조건(방정식·경계조건·해상도·solver)을 문서화한다.
