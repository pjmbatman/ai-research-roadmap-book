---
id: environments/docker
title: "Docker와 컨테이너"
title_en: "Docker and Containers"
tier: core
status: outline
prerequisites: [environments/dependencies-and-packaging]
---

# Docker와 컨테이너

## 학습 목표

- image와 container를 구분한다
- GPU를 사용하는 연구용 Dockerfile을 작성한다
- 데이터와 결과를 컨테이너 밖에서 관리한다

## 기본 개념

- image · container · registry · tag · digest

## Dockerfile

- base image 선택(CUDA 이미지 계열), layer와 build cache, multi-stage build

## 데이터와 결과

- bind mount · volume, working directory, 컨테이너 안에 데이터를 넣지 않는 원칙

## GPU 사용

- NVIDIA Container Toolkit, --gpus, host driver와 컨테이너 CUDA의 관계

## 실무 이슈

- 사용자 권한(UID/GID), shared memory(--shm-size), 네트워크·포트

## 개발 컨테이너와 HPC

- devcontainer, Apptainer/Singularity로 클러스터에서 실행하기

## 재현 가능한 빌드

- tag 대신 digest 고정, 빌드 기록

## 실습

- 연구 프로젝트의 Dockerfile을 작성하고 로컬 가상환경과 컨테이너에서 같은 결과가 나오는지 확인한다.

## 실패 사례

- 컨테이너 안에서 생성한 결과가 컨테이너 삭제와 함께 사라지는 것
- DataLoader worker가 shared memory 부족으로 죽는 것

## 통과 기준

- 새 사용자가 image 하나로 smoke test를 실행한다.
