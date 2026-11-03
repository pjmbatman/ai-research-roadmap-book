---
id: environments/ubuntu-and-cuda
title: "Ubuntu·드라이버·CUDA 설치와 버전 전략"
title_en: "Ubuntu, Drivers, and CUDA Installation"
tier: core
status: outline
prerequisites: [environments/software-stack]
---

# Ubuntu·드라이버·CUDA 설치와 버전 전략

## 학습 목표

- OS와 드라이버 버전을 선택하는 기준을 안다
- 공식 호환성 문서를 확인하는 절차를 따른다
- 여러 CUDA 버전을 공존시킨다

## OS 버전 선택

- LTS 릴리스, 지원 기간, 연구실 표준 정하기

## 드라이버 설치

- 배포판 패키지 · NVIDIA 저장소 · runfile 방식의 차이, Secure Boot와 커널 모듈
- 커널 업데이트 후 드라이버가 깨지는 원인과 복구

## CUDA toolkit 설치

- 필요한 경우와 필요 없는 경우, 여러 버전 공존과 PATH·LD_LIBRARY_PATH

## 호환성 확인 절차

- 버전 번호를 외우지 않고 공식 호환성 표를 확인하는 순서

## 설치 기록

- 설치 명령·버전·날짜를 남기는 서버 운영 일지

## 실습

- 테스트용 VM 또는 컨테이너에서 드라이버·toolkit 조합을 공식 문서로 검증하고 설치 절차서를 작성한다.

## 실패 사례

- 블로그의 오래된 설치 명령을 그대로 실행하는 것
- 운영 중인 공유 서버에서 드라이버를 기록 없이 업그레이드하는 것

## 통과 기준

- 연구실 표준 OS·드라이버·CUDA 조합과 그 근거, 설치·복구 절차를 문서화한다.
