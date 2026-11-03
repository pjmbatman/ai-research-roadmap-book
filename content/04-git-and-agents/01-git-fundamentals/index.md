---
id: git-and-agents/git-fundamentals
title: "Git의 모델과 기본 작업"
title_en: "Git Fundamentals"
tier: core
status: outline
prerequisites: [linux-and-remote/linux-shell]
---

# Git의 모델과 기본 작업

## 학습 목표

- working tree·index·commit의 관계를 설명한다
- branch·merge·rebase를 다룬다
- 실수를 복구한다

## Git의 데이터 모델

- blob · tree · commit, commit DAG, HEAD와 ref

## 세 영역

- working tree · staging area(index) · repository

## 기본 작업

- add · commit · status · diff · log · show

## branch·merge·rebase

- fast-forward와 merge commit, rebase의 의미와 주의점, conflict 해결

## 복구

- restore · reset · revert · reflog, 잃어버린 commit 찾기

## 실습

- 일부러 잘못된 commit·삭제·잘못된 rebase를 만들고 reflog로 복구한다.

## 실패 사례

- 공유 branch에서 force push로 이력을 덮어쓰는 것
- 의미 없는 거대 commit

## 통과 기준

- 임의의 Git 상태를 그림으로 설명하고 실수를 복구한다.
