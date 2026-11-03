---
id: research-code-engineering/project-structure
title: "연구 프로젝트 구조"
title_en: "Research Project Structure"
tier: core
status: outline
prerequisites: [python-and-computing/modules-and-imports]
---

# 연구 프로젝트 구조

## 학습 목표

- 책임에 따라 디렉터리를 나눈다
- notebook과 source code의 역할을 구분한다
- 연구실 공통 템플릿을 사용한다

## 책임 분리

- 데이터 처리 · 모델 · loss · 학습 · 평가 · 시각화

## 표준 디렉터리 구조

- README.md · pyproject.toml · configs/ · src/<project>/ · scripts/ · tests/ · notebooks/ · data/ · outputs/ · docs/
- src/<project>/{data, models, training, evaluation, utils}

## 진입점

- scripts/train.py · scripts/eval.py · scripts/predict.py, 평가 코드를 독립 실행할 수 있어야 하는 이유

## notebook의 자리

- 탐색·분석은 notebook, 재사용 코드는 src로 옮기는 기준

## 데이터와 출력의 위치

- data/는 경로 연결, outputs/<run-id>/ 구조

## 추상화의 균형

- 중복과 과도한 추상화 사이, 실험 코드가 공통 모듈로 승격되는 기준

## 실습

- 연구실 공통 템플릿으로 프로젝트를 시작하고, 기존 notebook 코드를 구조에 맞게 옮긴다.

## 실패 사례

- 모든 것이 하나의 거대한 train.py에 있는 것
- 평가 코드가 학습 코드 안에서만 실행되는 것

## 통과 기준

- 학습과 독립적으로 평가 스크립트만 실행해 같은 결과를 재현한다.
