# AI 연구자 로드맵 (가제)

> 문제 정의부터 재현 가능한 논문까지, 독립 연구자를 위한 훈련 체계

연구실 신입 연구원이 **독립적으로 연구를 수행하는 연구자**로 성장하기 위한 디지털 책(웹 문서)의 작업 저장소다.
AI 지식의 목록이 아니라 이론 · 구현 · 연구 · 운영의 네 축을 병행하는 **교육 체계**를 목표로 한다.

- **전체 목차:** [OUTLINE.md](OUTLINE.md) — 7개 트랙 · 28개 부 · 190개 장
- **목차 설계 근거:** [editorial/structure-rationale.md](editorial/structure-rationale.md)
- **UI/UX 설계 노트(논의 중):** [editorial/ux-design.md](editorial/ux-design.md)

## 구성

| 트랙 | 부 | 내용 |
|---|---|---|
| 시작하기 | 0–1 | 책의 목표와 읽는 법, AI 연구의 지도, 첫 연구 파이프라인 |
| 연구 작업대 | 2–6 | Linux·SSH·원격, CUDA·uv·Docker, Git·AI 에이전트·MCP, Python, 연구 코드 컨벤션 |
| 수학적 기초 | 7–14 | 선형대수·미적분·수치해석·확률·통계·베이지안·최적화·학습이론 |
| 모델링 | 15–21 | 데이터·고전 ML·PyTorch·아키텍처·생성모델·강화학습·신뢰성 |
| 연구 실천 | 22–23 | 실험 관리·재현성·분산학습, 연구 방법론·논문·발표 |
| 도메인 | 24–25 | Scientific ML, 배터리 AI |
| 연구실 | 26 | handbook·리뷰 체계·종합 프로젝트 |

모든 장은 **학습 목표 → 개념 → 실습 → 실패 사례 → 통과 기준**의 같은 형식을 따르고, 등급(필수·심화·참조)과 선수 지식을 가진다.

## 저장소 구조

```
book.toml          책 메타데이터, 트랙·등급·상태 정의
OUTLINE.md         전체 목차 (생성 파일)
content/           본문 — 부/장 디렉터리, 프레임워크 중립 Markdown
labs/              장별 실습 코드
data/              용어집, 참고문헌
editorial/         편집 규칙과 설계 문서 (책에 포함되지 않음)
templates/         부·장 템플릿
scripts/book.py    검증·목차 생성 도구 (Python 3.11+, 외부 의존성 없음)
```

자세한 규칙은 [editorial/conventions.md](editorial/conventions.md), 문체는 [editorial/style-guide.md](editorial/style-guide.md)를 따른다.

## 작업 방법

```bash
python3 scripts/book.py check
```

```bash
python3 scripts/book.py outline
```

```bash
python3 scripts/book.py stats
```

```bash
python3 scripts/book.py new-chapter 07-linear-algebra matrix-exponential "행렬 지수함수" "Matrix Exponential"
```

- 장의 집필 상태는 frontmatter `status`로 관리한다: `outline` → `draft` → `review` → `done`.
- 목차 구조(장 추가·이동·이름 변경)를 바꾸면 `outline`을 다시 생성해 같은 커밋에 넣는다. CI가 `check`를 실행한다.

## 현재 상태

- [x] 저장소 구조와 컨벤션
- [x] 전체 상세 목차 (모든 장 `outline` 상태)
- [ ] UI/UX·웹 프레임워크 결정 → [ux-design.md](editorial/ux-design.md)
- [ ] 1단계 학습 경로의 필수 장 초고
