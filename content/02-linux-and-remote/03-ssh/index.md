---
id: linux-and-remote/ssh
title: "SSH와 원격 접속"
title_en: "SSH and Remote Access"
tier: core
status: outline
prerequisites: [linux-and-remote/linux-shell]
---

# SSH와 원격 접속

## 학습 목표

- 키 기반 인증과 SSH 설정을 구성한다
- 포트 포워딩으로 원격 서비스를 연다
- 원격 편집 환경을 구성한다

## 인증과 키

- 공개키 인증, ssh-keygen, authorized_keys, agent와 agent forwarding의 위험
- host key와 known_hosts

## ~/.ssh/config

- Host 별칭, User, Port, IdentityFile, ProxyJump(jump host)

## 포트 포워딩

- local·remote forwarding, 원격 Jupyter·TensorBoard 접속

## 파일 전송

- scp · rsync · sshfs, 대용량 전송의 주의점

## 원격 편집

- VS Code Remote-SSH 등 원격 IDE, 원격 확장의 실행 위치

## 연결 유지와 복구

- ServerAliveInterval, mosh, 끊긴 뒤 작업 세션 재접속

## 실습

- jump host를 거치는 GPU 서버 접속을 config 한 줄로 만들고, 원격 TensorBoard를 로컬 브라우저로 연다.

## 실패 사례

- 비밀번호 인증과 키 공유
- 개인키를 서버나 저장소에 복사하는 것

## 통과 기준

- SSH 연결이 끊긴 뒤에도 실행 중인 학습의 로그와 대시보드에 다시 접속한다.
