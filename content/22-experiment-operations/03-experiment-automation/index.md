---
id: experiment-operations/experiment-automation
title: "실험 자동화와 하이퍼파라미터 탐색"
title_en: "Experiment Automation and Hyperparameter Search"
tier: core
status: outline
prerequisites: [experiment-operations/reproducibility, linux-and-remote/shared-servers]
---

# 실험 자동화와 하이퍼파라미터 탐색

## 학습 목표

- sweep을 자동화한다
- 하이퍼파라미터 탐색을 예산 안에서 수행한다
- 중단과 실패에 강한 실험을 운영한다

## sweep

- 설정 조합 생성, job manifest

## 탐색 알고리즘

- grid · random · 베이지안 최적화 · 조기 중단(ASHA · Hyperband), Optuna 등 도구

## 탐색 예산

- 비교 대상 간 공정한 예산

## 견고한 실행

- 실패 재시도, 원자적 저장, 중간 산출물, 재개

## 실습

- Optuna 등으로 조기 중단을 포함한 탐색을 수행하고, 같은 예산의 random search와 비교한다.

## 실패 사례

- 제안 방법만 많이 튜닝하는 것

## 통과 기준

- 탐색 공간·예산·결과를 문서화한 탐색 보고서를 작성한다.
