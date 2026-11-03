---
id: environments/environment-diagnosis
title: "환경 진단·재현·유지보수"
title_en: "Diagnosing and Maintaining Environments"
tier: core
status: outline
prerequisites: [environments/docker]
---

# 환경 진단·재현·유지보수

## 학습 목표

- 환경 문제를 체계적으로 진단한다
- 환경 보고서와 최소 재현 예제를 만든다
- 업그레이드와 rollback 정책을 세운다

## 증상별 진단

- GPU 인식 실패, CUDA error, missing shared library, import 오류, 버전 충돌

## 환경 정보 수집

- python -m torch.utils.collect_env 등 환경 보고서, 하드웨어 정보

## 최소 재현 예제

- 문제를 가장 작은 코드와 환경으로 줄이기

## 변경 관리

- 수정 전 상태 기록, 업그레이드 정책, rollback 절차

## 검증

- 새 환경에서 설치부터 smoke test까지

## 실습

- 고장 난 환경 세 가지를 진단하고, 각각 원인·근거·해결을 기록한 진단 보고서를 작성한다.

## 실패 사례

- 원인을 모른 채 재설치를 반복하는 것
- 문제 보고에 환경 정보를 빠뜨리는 것

## 통과 기준

- 설치 문서와 lockfile 또는 image로 새 사용자가 smoke test를 통과한다.
