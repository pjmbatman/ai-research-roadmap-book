---
id: python-and-computing/modules-and-imports
title: "모듈·패키지·import 시스템"
title_en: "Modules, Packages, and Imports"
tier: core
status: outline
prerequisites: [python-and-computing/python-language]
---

# 모듈·패키지·import 시스템

## 학습 목표

- import가 동작하는 방식을 설명한다
- src layout 패키지를 구성한다
- 실행 진입점을 설계한다

## import의 동작

- sys.path, module cache, 상대·절대 import

## 패키지 구성

- __init__.py, src layout과 flat layout

## 실행 방식

- python script.py와 python -m package.module의 차이, __main__

## 진입점

- console script entry point, 실행 스크립트와 라이브러리 코드의 분리

## 순환 import와 import 비용

- 순환 참조 해소, 무거운 import 지연

## 실습

- 'notebook에서는 되는데 스크립트에서는 import가 안 되는' 상황을 재현하고 src layout으로 해결한다.

## 실패 사례

- sys.path를 코드 안에서 조작하는 것
- 표준 라이브러리와 같은 이름의 파일을 만드는 것

## 통과 기준

- 설치 가능한 연구 패키지를 만들고 어디서든 import·실행한다.
