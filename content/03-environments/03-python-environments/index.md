---
id: environments/python-environments
title: "Python 환경 관리: venv·pip·uv·conda"
title_en: "Python Environments: venv, pip, uv, conda"
tier: core
status: outline
prerequisites: [environments/software-stack]
---

# Python 환경 관리: venv·pip·uv·conda

## 학습 목표

- 각 도구의 역할과 범위를 비교한다
- 프로젝트별 격리 환경을 만든다
- 연구실 표준 도구를 정한다

## 왜 격리 환경인가

- 시스템 Python, 사용자 site-packages, 프로젝트 환경

## 도구별 역할

- venv + pip: 표준 라이브러리 기반의 최소 구성
- uv: Python 버전 관리 · 가상환경 · 의존성 해석 · lockfile을 하나로
- conda/mamba: Python 밖의 바이너리 의존성까지 관리하는 환경
- pyenv 등 Python 버전 관리자

## 일상 작업

- 생성 · 활성화 · 삭제, 어떤 python이 실행되는지 확인(which python, sys.executable)

## 도구 혼용의 문제

- conda 환경 안의 pip, 시스템 pip로 설치한 패키지

## 연구실 표준 정하기

- 기본 도구 하나와 예외 상황의 규칙

## 실습

- 같은 프로젝트를 venv+pip, uv, conda로 각각 구성하고 설치 시간·재현성·디스크 사용량을 비교한다.

## 실패 사례

- 활성화하지 않은 환경에 패키지를 설치하는 것
- base 환경에 모든 것을 설치하는 것

## 통과 기준

- 연구실 표준 도구로 새 프로젝트 환경을 만들고 다른 사람이 동일하게 재생성한다.
