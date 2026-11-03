---
id: data/dataset-versioning
title: "데이터셋 버전·문서화·라이선스"
title_en: "Dataset Versioning, Documentation, and Licensing"
tier: core
status: outline
prerequisites: [data/splitting-and-leakage]
---

# 데이터셋 버전·문서화·라이선스

## 학습 목표

- 데이터 버전과 분할을 추적한다
- 데이터셋 문서를 작성한다
- 라이선스와 민감정보를 관리한다

## 버전 관리

- checksum · split manifest · 데이터 version, DVC 등 도구의 개념

## 데이터 통합

- 여러 출처의 schema·단위·프로토콜 통합, 통합 스키마 데이터 허브 활용

## 데이터셋 문서

- datasheet · data card, 수집 조건과 한계

## 라이선스와 이용 조건

- 상업적 이용 · 재배포 · 인용 의무

## 민감정보

- 개인정보 · 기업 기밀, 접근 권한

## 업데이트와 재실험

- 데이터 변경 시 재실험 범위 결정

## 실습

- 실험마다 사용한 데이터 version·분할 manifest·checksum을 기록하는 체계를 만들고, 데이터 카드를 작성한다.

## 실패 사례

- 데이터가 조용히 바뀌어 이전 결과를 재현하지 못하는 것
- 라이선스를 확인하지 않고 데이터를 공개하는 것

## 통과 기준

- 임의의 실험이 어떤 원본·전처리·분할을 사용했는지 추적한다.
