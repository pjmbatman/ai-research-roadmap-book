---
id: deep-learning-pytorch/trainer-design
title: "Trainer 설계: 직접 작성과 프레임워크"
title_en: "Designing Trainers: Hand-Written vs Frameworks"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/evaluation-and-checkpoints, research-code-engineering/configuration]
---

# Trainer 설계: 직접 작성과 프레임워크

## 학습 목표

- 재사용 가능한 trainer.py를 설계한다
- 학습 프레임워크를 언제 도입할지 판단한다

## trainer.py의 책임

- 학습 루프 · 검증 · 로깅 · 체크포인트 · 조기 종료, 모델·데이터와의 경계

## 확장 지점

- callback · hook, 과제별 step 함수

## eval.py의 분리

- 학습 코드 없이 평가할 수 있어야 하는 이유

## 프레임워크

- PyTorch Lightning · Hugging Face Accelerate · Trainer의 추상화 수준

## 도입 판단

- 숨겨진 동작, 디버깅 비용, 연구 유연성

## 실습

- 직접 작성한 trainer와 Lightning으로 같은 실험을 구현해 코드량·유연성·결과 일치를 비교한다.

## 실패 사례

- 이해하지 못한 프레임워크 기본값이 결과를 바꾸는 것
- 모든 것을 처리하는 거대한 trainer

## 통과 기준

- 연구실 템플릿의 trainer.py·eval.py를 설계하고 문서화한다.
