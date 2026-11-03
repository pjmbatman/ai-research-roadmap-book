---
id: python-and-computing/python-language
title: "Python 언어의 핵심"
title_en: "Core Python"
tier: core
status: outline
prerequisites: []
---

# Python 언어의 핵심

## 학습 목표

- 객체·참조·가변성을 정확히 이해한다
- 함수·클래스·제너레이터를 연구 코드에 맞게 사용한다
- type hint를 작성한다

## 객체와 참조

- 변수는 이름표, 가변·불변 객체, 얕은 복사와 깊은 복사

## 함수

- 인자 전달, 기본값, *args · **kwargs, scope와 closure

## 예외와 자원 관리

- 예외 처리, context manager(with)

## 반복과 지연 계산

- iterator · generator · comprehension

## 클래스

- class · dataclass · 상속과 composition, magic method

## 타입

- type hint, Protocol, 정적 분석과의 관계

## 데코레이터

- 함수 감싸기, 등록 패턴(registry)

## 실습

- mutable default argument, 리스트 공유 참조, closure의 늦은 바인딩 버그를 재현하고 수정한다.

## 실패 사례

- 함수 기본값에 리스트를 쓰는 것
- 복사한 줄 알았던 객체가 공유되는 것

## 통과 기준

- 주어진 코드의 객체 참조 관계를 그림으로 설명하고 버그를 예측한다.
