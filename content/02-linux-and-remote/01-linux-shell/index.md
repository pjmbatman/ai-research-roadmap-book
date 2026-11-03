---
id: linux-and-remote/linux-shell
title: "Linux와 셸의 기초"
title_en: "Linux and Shell Fundamentals"
tier: core
status: outline
prerequisites: []
---

# Linux와 셸의 기초

## 학습 목표

- 파일시스템과 권한을 이해한다
- 셸 환경변수와 startup file을 다룬다
- 반복 작업을 스크립트로 만든다

## 파일시스템과 경로

- 디렉터리 구조(/home, /tmp, /usr, /opt, 마운트된 storage), 절대·상대 경로
- symbolic link와 hard link

## 사용자·그룹·권한

- rwx, chmod, chown, sudo의 범위, 공유 디렉터리 권한

## 셸과 환경

- bash/zsh, 환경변수, PATH, .bashrc·.profile의 로딩 순서

## 파일 탐색과 텍스트 처리

- ls · find · rg · grep · head/tail · less · wc · sort · uniq · awk/sed 기초

## 파이프와 리다이렉션

- stdin · stdout · stderr, |, >, >>, 2>&1, tee

## 압축·동기화

- tar · zip · rsync

## Bash 스크립트

- 변수, 조건, 반복, exit code, set -euo pipefail

## 실습

- 데이터 준비부터 학습 실행까지의 반복 작업을 하나의 Bash 스크립트로 만들고 실패 시 즉시 멈추게 한다.

## 실패 사례

- PATH 순서 때문에 다른 python이 실행되는 것
- 실패한 명령 뒤에도 계속 진행되는 스크립트

## 통과 기준

- 새 서버에서 자신의 셸 환경을 재구성하고, 실패를 감지하는 스크립트를 작성한다.
