---
id: research-code-engineering/configuration
title: "설정 관리"
title_en: "Configuration Management"
tier: core
status: outline
prerequisites: [research-code-engineering/project-structure]
---

# 설정 관리

## 학습 목표

- 코드와 설정을 분리한다
- 설정의 기본값·override·검증을 설계한다
- 실제 적용된 설정을 저장한다

## 왜 설정을 분리하는가

- 코드 수정 없이 데이터 · 모델 · seed · 학습률 바꾸기

## 설정 형식

- YAML · TOML · JSON, config.yaml의 구조(data · model · train · eval)

## 설정 객체

- dataclass · pydantic으로 타입과 검증, 누락·충돌 처리

## 조합과 override

- 기본값 + 실험별 override, OmegaConf · Hydra의 개념

## 기록

- resolved config 저장, run ID, 코드 revision과의 연결

## 경로 처리

- 프로젝트 root, 상대 경로, 작업 디렉터리 독립성

## 실습

- 하나의 config.yaml과 명령행 override로 데이터셋·모델·seed·학습률을 바꾸어 실행하고, 각 실행의 resolved config를 저장한다.

## 실패 사례

- 코드 곳곳에 하드코딩된 숫자와 경로
- 저장된 설정과 실제 실행 설정이 다른 것

## 통과 기준

- 임의의 결과 파일에서 그 결과를 만든 정확한 설정을 복원한다.
