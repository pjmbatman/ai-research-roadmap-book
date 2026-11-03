---
id: environments/dependencies-and-packaging
title: "의존성과 패키징"
title_en: "Dependencies and Packaging"
tier: core
status: outline
prerequisites: [environments/python-environments]
---

# 의존성과 패키징

## 학습 목표

- 직접·전이 의존성과 버전 제약을 이해한다
- pyproject와 lockfile로 환경을 고정한다
- 의존성 충돌을 해결한다

## 의존성의 구조

- 직접 의존성과 전이 의존성, 의존성 그래프
- 버전 제약(==, >=, ~=)과 semantic versioning

## pyproject.toml

- project 메타데이터, dependencies, optional-dependencies, dependency groups

## lockfile과 재현

- requirements.txt · uv.lock · conda-lock의 역할, lock과 sync

## 배포 형식

- wheel과 sdist, 소스 빌드가 일어나는 경우, platform tag

## PyTorch 설치의 특수성

- CUDA별 index URL, 플랫폼별 wheel 선택

## 설치와 개발 모드

- editable install, src layout, development dependency

## 충돌 해결

- resolver 오류 읽기, 제약 완화, 최소 재현

## 실습

- 의도적으로 충돌하는 의존성 조합을 만들고 resolver 메시지를 해석해 해결한 뒤 lockfile을 커밋한다.

## 실패 사례

- pip freeze 결과를 그대로 의존성 선언으로 쓰는 것
- 버전 제약 없이 최신 버전을 설치해 결과가 달라지는 것

## 통과 기준

- pyproject와 lockfile만으로 새 머신에서 동일 환경을 재생성한다.
