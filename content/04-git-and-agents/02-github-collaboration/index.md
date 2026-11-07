---
id: git-and-agents/github-collaboration
title: "GitHub 협업과 저장소 관리"
title_en: "Collaboration with GitHub"
tier: core
status: draft
prerequisites: [git-and-agents/git-fundamentals]
---

# GitHub 협업과 저장소 관리

4.1장의 Git은 한 컴퓨터 안의 이력을 다뤘다. 연구는 혼자 하지 않는다. 공동 연구자와 같은 코드를 고치고, 지도교수가 변경을 검토하고, 논문이 나간 뒤에는 모르는 사람이 저장소를 clone해 결과를 재현한다. 이 장은 저장소를 여러 사람과 여러 컴퓨터가 공유할 때 필요한 것을 다룬다. 원격 저장소와 주고받는 방법, 변경을 리뷰받고 합치는 Pull request, 그리고 공유 저장소에 **들어가면 안 되는 것**(데이터, checkpoint, 비밀정보)을 막는 규칙이다.

뒤의 세 가지가 특히 중요하다. 실수로 commit한 2 GB checkpoint는 지워도 이력에 남고, 실수로 push한 API key는 지워도 이미 누군가 복사했을 수 있다. 이런 실수는 되돌리는 비용이 예방하는 비용보다 훨씬 크다. 이 장의 예시는 GitHub를 쓰지만, Git 자체의 동작은 다른 호스팅 서비스(GitLab, Gitea, 학교·기관의 자체 서버)에서도 같다. 화면 구성과 메뉴 이름은 서비스마다 다르고 자주 바뀌므로, 이 장은 개념과 Git 명령을 중심으로 쓰고 화면 조작은 각 서비스의 공식 문서를 확인하도록 한다.

## 학습 목표

- remote와 PR 기반 협업을 수행한다
- .gitignore와 대용량 파일 정책을 세운다
- 비밀정보 유출을 예방한다

## remote 작업

**정의.** 원격 저장소(remote)는 다른 곳에 있는 같은 프로젝트의 저장소에 붙인 이름이다. `git clone <URL>`은 저장소 전체(모든 commit)를 복사하고, 원래 저장소를 `origin`이라는 이름의 remote로 등록한다. remote는 서버일 필요가 없다. 같은 컴퓨터의 다른 디렉터리도 remote가 될 수 있고, 이 장의 실습이 그렇게 한다.

Git에서 모든 clone은 동등하다. 서버에 있는 저장소가 특별한 이유는 기술이 아니라 **합의** 때문이다. "원격의 main이 공식 이력"이라고 모두가 약속한 것이다.

**원격 추적 branch.** clone하면 로컬에 `main`과 함께 `origin/main`이 생긴다. `origin/main`은 "마지막으로 확인했을 때 원격의 main이 가리키던 commit"을 기억하는 이름표다. 직접 commit할 수 없고, 원격과 통신할 때만 갱신된다.

| 명령 | 하는 일 | 로컬 branch | working tree |
|---|---|---|---|
| `git fetch` | 원격의 새 객체를 받고 `origin/*` 갱신 | 그대로 | 그대로 |
| `git pull` | fetch + 현재 branch에 합치기(merge 또는 rebase) | 바뀐다 | 바뀐다 |
| `git push` | 로컬 branch의 commit을 보내고 원격 branch를 옮긴다 | 그대로 | 그대로 |

fetch는 아무것도 바꾸지 않으므로 언제 해도 안전하다. 습관적으로 fetch한 뒤, 무엇이 새로 왔는지 보고 합친다.

```bash
# 집필 시점 예시
git fetch origin
git log --oneline main..origin/main    # 원격에만 있는 commit
git log --oneline origin/main..main    # 내 로컬에만 있는(아직 push하지 않은) commit
git merge origin/main                  # 확인했으면 합친다 (또는 git pull)
```

::quiz{id="fetch-vs-pull"}

**push가 거부될 때.** 원격 branch는 fast-forward로만 옮겨지는 것이 기본이다. 내가 마지막으로 fetch한 뒤 다른 사람이 push했다면, 원격 main은 내 main의 조상이 아니므로 push가 거부된다(`! [rejected] main -> main (fetch first)`). 대응은 항상 같다. fetch하고, 합치고, 테스트하고, 다시 push한다. 두 branch가 갈라진 상태에서 `git pull`을 하면, 설정에 따라 merge할지 rebase할지 정하라는 안내가 나오기도 한다. 아직 push하지 않은 내 commit만 있다면 `git pull --rebase`로 이력을 한 줄로 유지할 수 있다(4.1장의 rebase 규칙). 거부를 `--force`로 넘기는 것은 다른 사람의 commit을 원격에서 지우는 일이다.

::quiz{id="push-rejected"}

**upstream.** 로컬 branch가 어느 원격 branch와 짝인지를 upstream(추적 대상)이라 한다. 새 branch를 처음 push할 때 `git push -u origin feature`로 짝을 지어 두면, 이후에는 `git push`·`git pull`만으로 그 branch와 주고받고, `git status`가 "Your branch is ahead of 'origin/feature' by 2 commits"처럼 앞뒤 관계를 알려 준다.

**fork.** 쓰기 권한이 없는 저장소(다른 연구실의 공개 코드, 오픈소스 라이브러리)에 기여하려면 fork를 만든다. fork는 호스팅 서비스 위에 만든 내 계정의 사본이다. 관례상 내 fork를 `origin`, 원래 저장소를 `upstream`이라는 remote로 등록한다.

```bash
# 집필 시점 예시 — <...>는 자신의 값으로 바꾼다
git clone https://github.com/<내 계정>/<저장소>.git
cd <저장소>
git remote add upstream https://github.com/<원래 소유자>/<저장소>.git
git fetch upstream
git switch -c fix-typo upstream/main   # 원래 저장소의 최신 main에서 작업 branch를 만든다
```

연구실 안의 협업처럼 모두가 쓰기 권한을 가진 경우에는 fork 없이 한 저장소에서 branch만 나눠 쓰는 것이 보통이다.

## Pull request와 code review

**왜 필요한가.** 공식 이력(main)에 들어가는 변경은 한 사람의 판단만으로 들어가지 않게 하는 것이 좋다. 버그를 잡기 위해서이기도 하지만, 연구에서는 **두 번째 사람이 변경의 가정을 확인하는 절차**라는 의미가 더 크다. 데이터 분할을 바꾸는 변경이 리뷰 없이 들어가면, 그 뒤의 모든 결과가 다른 조건에서 나온 것이 된다.

**정의.** Pull request(PR, GitLab에서는 merge request)는 "내 branch를 저 branch에 merge해 달라"는 요청을 호스팅 서비스 위에 만든 것이다. Git 자체의 기능이 아니라 서비스의 기능이다. PR에는 변경의 diff, 설명, 리뷰 댓글, 자동 검사(CI) 결과가 한곳에 모인다.

**PR의 단위.** PR 하나는 **하나의 목적**을 갖는다. 리뷰어가 한 번에 집중해서 읽을 수 있는 크기여야 하고, 목적을 한 문장으로 쓸 수 있어야 한다. "데이터 로더 리팩터링 + 새 손실 함수 + 학습률 변경"은 세 개의 PR이다. 리팩터링(동작이 바뀌지 않아야 하는 변경)과 실험 조건 변경(결과가 바뀌는 변경)은 특히 분리한다. 섞이면 결과가 바뀐 이유를 리뷰어가 판단할 수 없다.

**설명 쓰기.** 좋은 PR 설명은 리뷰어의 질문에 미리 답한다.

- **무엇을, 왜.** 이 변경이 푸는 문제와 선택한 방법. 연결된 이슈 번호.
- **어떻게 확인했는가.** 실행한 테스트, 재현한 실험, 비교한 숫자.
- **리뷰어가 볼 곳.** 판단이 필요한 부분, 확신이 없는 부분.
- **영향.** 기존 결과·checkpoint·설정 파일과의 호환성.

이 책의 저장소는 PR 템플릿(`.github/PULL_REQUEST_TEMPLATE.md`)에 이 항목들을 체크리스트로 넣어 두었다. "무엇을, 왜", "검증"(검사 통과, 코드 출력은 실제 실행 결과, 퀴즈 정답 확인), "확인 필요 목록"(직접 확인하지 못한 사실)이다. 템플릿은 PR을 만들 때마다 설명란에 자동으로 채워진다. PR 설명에 `Closes #12`처럼 쓰면, PR이 기본 branch에 merge될 때 GitHub가 그 이슈를 닫는다.

**리뷰 요청과 응답.** 리뷰어는 다음 순서로 본다.

1. 설명을 읽고 목적을 이해한다. 목적이 불분명하면 코드를 읽기 전에 묻는다.
2. 테스트와 실행 결과를 확인한다. 직접 재현할 수 있으면 재현한다.
3. diff를 읽는다. 연구 코드에서는 특히 실험 조건(데이터 분할, seed, 전처리, metric 정의)이 바뀌었는지 본다.
4. 댓글은 **요청**(merge 전에 고쳐야 함), **질문**, **제안**(선택)을 구분해 쓴다.

작성자는 각 댓글에 고치거나, 고치지 않는 이유를 답한다. 리뷰를 반영한 commit은 같은 branch에 push하면 PR에 자동으로 더해진다. 리뷰가 진행 중인 PR의 branch를 rebase·force push하면 리뷰어가 이미 본 commit이 사라져 "무엇이 바뀌었는지" 추적이 어려워진다. 리뷰 중에는 commit을 더하고, 정리가 필요하면 merge 방식(아래)으로 해결한다.

**merge 방식.** 호스팅 서비스는 보통 세 가지 merge 방식을 제공한다. merge commit(4.1장의 merge 그대로), squash(PR의 commit을 하나로 합쳐 main에 하나의 commit으로), rebase(PR의 commit을 main 위에 다시 적용)다. 어느 것을 쓸지는 저장소마다 정하고, branch 보호 설정으로 "PR을 거치지 않은 push 금지", "리뷰 승인 필수", "CI 통과 필수", "force push 금지"를 걸 수 있다. 설정 화면의 위치와 이름은 서비스 문서에서 확인한다.

:::note[분야 예시]
여러 기관이 함께 쓰는 단백질 구조 예측 파이프라인에서, 한 기관이 "학습 데이터에서 저해상도 구조를 제외"하는 필터를 PR로 올렸다고 하자. 코드는 몇 줄이지만, 이 PR이 merge되면 모든 기관의 다음 학습이 다른 데이터로 돌아간다. 리뷰어는 코드의 정확성보다 먼저 "이 필터가 test set에도 적용되는가, 이전 결과와의 비교는 어떻게 할 것인가"를 묻는다. PR 설명에 제외되는 구조의 수와 비율을 적어 두면 이 논의가 빨라진다.
:::

## 커밋과 branch 규칙

여러 사람이 쓰는 저장소는 이름과 메시지의 규칙을 정해 둔다. 규칙의 내용보다 **모두가 같은 규칙을 쓰는 것**이 중요하다.

**커밋 메시지.** 널리 쓰이는 규칙 중 하나가 Conventional Commits다. 첫 줄을 `<type>[optional scope]: <description>` 꼴로 쓴다.

```
feat(data): HDF5 로더에 chunk 단위 읽기 추가
fix(metric): RMSE 계산에서 마스크된 격자점을 제외
docs: 재현 절차에 데이터 내려받기 단계 추가
```

명세는 `fix`(버그 수정)와 `feat`(새 기능)을 정의하고, 호환성을 깨는 변경은 type 뒤의 `!`나 `BREAKING CHANGE:` footer로 표시한다. 그 밖의 type(`docs`, `refactor`, `test` 등)은 허용하되 강제하지 않는다. 형식이 정해져 있으면 변경 기록(changelog)을 자동으로 만들거나 `git log --grep '^fix'`로 버그 수정만 모아 볼 수 있다.

이 책의 저장소는 같은 생각을 원고 작업에 맞게 바꿔 쓴다(`editorial/workflow.md`). `content(07.2): 아핀 구조 초고`, `quiz(12.3): 분위수 문항 추가`처럼 범위에 장 번호를 넣는다. 연구 저장소라면 `exp(ablation): ...`처럼 실험 변경을 따로 표시하는 type을 정해 두면, 결과에 영향을 주는 commit을 나중에 쉽게 찾을 수 있다.

**branch 이름.** `<종류>/<짧은-설명>`이 흔하다. 이 저장소는 `write/linear-algebra-affine-and-nonlinear`, `fix/broken-links`처럼 작업 종류를 앞에 둔다. 연구실이라면 `exp/<실험 이름>`, `fix/<버그>`, `<사람>/<주제>` 같은 규칙을 정한다. 규칙은 `CONTRIBUTING.md`나 이 책의 `editorial/workflow.md` 같은 문서에 한 번 적어 두고, 4.4장에서 다룰 에이전트 지침 파일에서도 그 문서를 가리키게 한다.

**한 작업 = 한 branch = 한 PR.** 이 저장소의 원칙이기도 하다. 한 branch에 여러 작업을 쌓으면 하나만 merge하거나 하나만 되돌릴 수 없다. 작업이 끝나 merge된 branch는 지운다.

## .gitignore

**왜 필요한가.** 연구 프로젝트 디렉터리에는 저장소에 들어가면 안 되는 것이 코드보다 훨씬 많다. 데이터, 실험 출력, 가상환경, 캐시, 비밀정보다. 하나하나 `git add`에서 빼는 방식은 언젠가 실패한다. `.gitignore`는 "이런 이름의 파일은 아직 추적하지 않았다면 무시하라"는 규칙을 저장소에 commit해 모두가 공유하게 한다.

**정의와 문법.** `.gitignore`의 각 줄은 경로 패턴이다.

| 패턴 | 뜻 |
|---|---|
| `*.ckpt` | 어느 디렉터리에 있든 이름이 `.ckpt`로 끝나는 파일 |
| `/outputs/` | 저장소 루트의 `outputs` 디렉터리(앞의 `/`가 루트에 고정) |
| `__pycache__/` | 어느 깊이에 있든 그 이름의 디렉터리 |
| `/data/*` + `!/data/README.md` | `data` 안의 모든 것을 무시하되 `README.md`는 예외 |

`!`로 시작하는 줄은 앞의 규칙을 취소한다. 다만 **디렉터리 자체가 무시되면 그 안의 파일은 `!`로 되살릴 수 없다.** 그래서 `/data/`가 아니라 `/data/*`(디렉터리 안의 항목들)를 무시하고 예외를 둔다. 어떤 규칙이 어떤 파일을 무시하는지 헷갈리면 `git check-ignore -v <경로>`로 확인한다.

**연구 프로젝트에서 무시할 것.**

| 범주 | 예 | 대신 추적할 것 |
|---|---|---|
| 데이터 | 원시·전처리 데이터 | 내려받기·전처리 스크립트, 데이터 README, 체크섬 |
| 실험 출력 | checkpoint, 로그, 그림 | 출력을 만드는 설정 파일과 명령 |
| 환경 | `.venv/`, conda 환경 | `pyproject.toml`, lockfile(3.3–3.4장) |
| 캐시 | `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/` | — |
| 비밀정보 | `.env`, 개인 키 | 키 이름만 담은 `.env.example` |

이 장의 실습 폴더에 위 표를 옮긴 예시 `labs/git-and-agents/github-collaboration/research.gitignore`가 있다. 같은 폴더의 테스트가 이 파일로 데이터·checkpoint·`.env`는 무시되고 `data/README.md`·`.env.example`·소스 코드는 추적되는지 실제 저장소에서 검사한다. 언어·도구별 템플릿은 GitHub의 공개 저장소 `github/gitignore`에도 모여 있다.

**이미 추적 중인 파일.** `.gitignore`는 **아직 추적하지 않는** 파일에만 적용된다. 이미 commit한 파일을 나중에 `.gitignore`에 넣어도 Git은 계속 추적한다. 추적을 끊으려면 index에서만 지운다.

```bash
# 집필 시점 예시
git rm --cached data/raw.csv          # index에서만 지운다. 디스크의 파일은 남는다
git rm -r --cached outputs/           # 디렉터리 전체
git commit -m "chore: 데이터·출력물 추적 중단"
```

이렇게 해도 **과거 commit에는 파일이 그대로 남는다.** 크기가 크거나 민감한 파일이라면 아래 두 절의 조치가 더 필요하다.

::quiz{id="gitignore-tracked-file"}

## 대용량 파일

**왜 Git에 넣지 않는가.** Git은 모든 clone이 모든 이력을 갖는 구조다. 텍스트 파일은 압축과 차분 저장(delta compression)이 잘 되지만, checkpoint나 이미지처럼 바이너리이고 매번 통째로 바뀌는 파일은 버전마다 거의 전체 크기가 쌓인다. 1 GB checkpoint를 열 번 갱신해 commit하면, 이후 모든 clone이 그 이력 전체를 받아야 한다. 그리고 4.1장에서 본 것처럼 commit은 지워도 이력에 남는다. 호스팅 서비스도 한도를 둔다. 예를 들어 GitHub 문서는 집필 시점에 50 MiB보다 큰 파일에 경고를, 100 MiB보다 큰 파일에 push 차단을 명시하고 있다(한도는 서비스와 시점마다 다르므로 공식 문서에서 확인한다).

::quiz{id="checkpoint-in-history"}

**Git LFS.** Git Large File Storage(LFS)는 지정한 파일을 저장소에 직접 넣지 않고, 작은 **포인터 파일**만 commit한 뒤 실제 내용은 별도의 LFS 서버에 올리는 Git 확장이다.

```bash
# 집필 시점 예시
git lfs install                  # 사용자 계정에 한 번
git lfs track "*.ckpt"           # .gitattributes에 규칙을 기록한다
git add .gitattributes model.ckpt
git commit -m "chore: checkpoint를 LFS로 관리"
```

commit에 들어가는 포인터 파일은 다음과 같은 세 줄이다(집필 시점의 `git lfs pointer` 출력).

```
version https://git-lfs.github.com/spec/v1
oid sha256:d29751f2649b32ff572b5e0a9f541ea660a50f94ff0beedfb0b692b924cc8025
size 1000000
```

LFS의 범위와 한계를 알고 써야 한다.

- **앞으로 add하는 파일에만 적용된다.** 이미 일반 blob으로 commit된 파일은 그대로다. 과거 commit까지 바꾸려면 이력을 다시 쓰는 migrate 작업이 필요하다.
- **저장 용량과 전송량에 별도 한도와 비용이 있다.** 호스팅 서비스마다 다르고 자주 바뀐다.
- **모든 clone이 LFS를 설치해야 한다.** 설치하지 않은 사람은 checkpoint 대신 포인터 파일을 받는다.
- **데이터 관리 도구가 아니다.** 어떤 데이터로 어떤 checkpoint를 만들었는지, 데이터의 버전과 출처는 무엇인지를 기록해 주지 않는다.

**Git 밖에서 관리하는 이유.** 그래서 연구 데이터와 checkpoint는 대개 Git 밖에 둔다. 데이터는 공유 storage, 기관 데이터 저장소, 데이터 허브에 두고 저장소에는 **내려받는 스크립트와 체크섬**을 둔다. checkpoint는 실험 추적 도구나 모델 저장소에 두고, 저장소에는 그것을 만든 commit id와 설정을 남긴다. 데이터 버전 관리는 15.5장, 실험 추적은 22.1장에서 다룬다. Git에는 "다시 만드는 방법"을 넣고, 다시 만들 수 있는 결과물은 넣지 않는다는 원칙이다.

:::note[분야 예시]
기상 재분석 자료처럼 수 TB에 이르는 데이터는 애초에 Git에 넣을 수 없다. 이런 연구의 저장소는 데이터 제공 기관의 내려받기 스크립트, 사용한 변수·기간·격자의 목록, 내려받은 파일의 체크섬만 담는다. 독자가 저장소를 clone하고 스크립트를 실행하면 같은 데이터가 같은 경로에 놓이고, 체크섬으로 그것이 같은 데이터인지 확인한다.
:::

## 비밀정보

**무엇이 비밀인가.** API key, 접근 토큰, 비밀번호, SSH 개인 키, 클라우드 자격 증명, 그리고 데이터베이스 접속 문자열처럼 그 자체로 권한을 주는 값이다. 연구에서는 여기에 **개인정보와 공개 전 데이터**가 더해진다. 환자 정보가 든 CSV, 공개 전 실험 데이터, 심사 중인 원고도 저장소에 들어가면 같은 문제가 된다.

**commit했을 때의 대응.** 순서가 중요하다.

1. **폐기와 교체가 먼저다.** 그 key를 발급한 서비스에서 폐기(revoke)하고 새 key를 발급(rotate)한다. push된 순간 이미 누군가 복사했다고 가정한다. GitHub의 민감 정보 제거 문서도 첫 단계로 폐기·교체를 들고, 폐기한 것만으로 충분할 수 있다고 쓴다. 폐기한 key는 이력에 남아 있어도 쓸 수 없다.
2. **피해를 확인한다.** 그 key로 이루어진 접근 기록(서비스의 감사 로그, 청구 내역)을 확인한다.
3. **필요하면 이력을 정리한다.** 개인정보처럼 폐기할 수 없는 정보라면 이력을 다시 써야 한다(`git filter-repo` 같은 도구). 이력을 다시 쓰면 commit id가 모두 바뀌므로 모든 협업자와 조율해야 하고, 이미 만들어진 다른 사람의 clone과 fork에서는 지울 수 없다. 호스팅 서비스에 남은 캐시는 서비스에 따로 요청해야 할 수 있다.

**파일을 지우는 commit은 대응이 아니다.** 4.1장에서 본 것처럼 그 commit 이전의 스냅숏은 바뀌지 않는다. 저장소를 비공개로 돌리는 것도 이미 생긴 사본에는 효과가 없다.

::quiz{id="leaked-key-first-step"}

**.env와 secret 관리.** 예방의 핵심은 비밀을 코드와 분리하는 것이다. 코드는 환경변수에서 값을 읽고, 값은 실행 환경이 주입한다.

```python
import os

api_key = os.environ["TRACKER_API_KEY"]   # 없으면 KeyError로 바로 실패한다
```

로컬에서는 값을 `.env` 파일에 두고 `.gitignore`로 무시한다. 대신 어떤 변수가 필요한지는 공유해야 하므로, 값 없이 이름만 담은 `.env.example`을 commit한다.

```bash
# .env.example — 이 파일은 commit한다. 실제 값은 .env에 쓴다(commit하지 않는다)
TRACKER_API_KEY=
DATA_ROOT=/path/to/shared/data
```

CI나 클라우드에서는 호스팅 서비스·클라우드의 secret 저장소에 값을 넣고, 실행 시 환경변수로 주입한다. 권한은 가능한 한 좁게 준다. 읽기만 필요한 작업에는 읽기 전용 토큰을, 한 저장소만 필요한 작업에는 그 저장소에만 유효한 토큰을 쓰고, 만료 기간을 둔다. 이 원칙은 4.5장에서 AI 에이전트에게 도구를 연결할 때 다시 나온다.

**pre-commit 검사.** 실수는 commit 시점에 막는 것이 가장 싸다. Git은 commit 직전에 `.git/hooks/pre-commit` 스크립트를 실행하고, 이 스크립트가 실패하면 commit을 중단한다. `.git/hooks/`는 저장소와 함께 공유되지 않으므로, 팀에서는 `pre-commit` 같은 hook 관리 도구로 설정 파일을 commit해 공유하는 경우가 많다. 이 hook에서 비밀 탐지 도구(예: gitleaks, detect-secrets — 집필 시점 예시)와 대용량 파일 검사를 실행한다. 호스팅 서비스에도 push된 내용에서 알려진 형식의 key를 찾는 비밀 스캔 기능이 있다. 다만 어느 도구도 모든 비밀을 찾지는 못한다. 특히 연구실 내부 서비스의 비밀번호나 개인정보는 알려진 형식이 없어 탐지되지 않는다. 도구는 보조 수단이고, `git diff --staged`를 읽는 습관이 기본이다.

::quiz{id="secret-handling"}

## 이슈와 프로젝트 관리

**왜 필요한가.** 연구 작업의 맥락은 쉽게 흩어진다. "왜 이 실험을 그만두었는가", "이 버그는 어떤 결과에 영향을 주었는가"가 메신저와 머릿속에만 있으면 몇 달 뒤에는 아무도 모른다. 이슈 트래커는 이 맥락을 코드 옆에, 검색할 수 있게 남긴다.

**이슈.** 이슈(issue)는 할 일 하나, 질문 하나, 버그 하나에 대한 기록이다. 연구 저장소의 이슈는 다음을 담는다.

- **무엇을, 왜.** 해결할 문제와 그 동기.
- **완료 기준.** 무엇이 되면 닫는가. "학습이 안정적이다"가 아니라 "seed 3개에서 validation loss가 발산하지 않는다".
- **영향.** 버그라면 영향을 받은 실험과 결과(commit id 범위).

이슈와 PR은 서로 연결한다. PR 설명의 `Closes #번호`는 merge될 때 이슈를 닫고, commit 메시지나 댓글의 `#번호`는 그 이슈에 링크를 남긴다.

**label과 milestone.** label은 이슈를 분류하는 태그다(`bug`, `experiment`, `data`, `paper`). milestone은 마감이 있는 목표에 이슈를 묶는다(예: "학회 투고", "1차 재현 완료"). 이 책의 저장소는 작업 종류를 `type:content`, `type:fix` 같은 label로, 마일스톤을 M0–M4로 관리한다(`editorial/workflow.md` §3, §6). 이슈 양식(template)을 만들어 두면 새 이슈가 필요한 항목을 빠뜨리지 않는다. 이 저장소는 "장 집필", "내용 오류", "새 분야 제안" 양식을 `.github/ISSUE_TEMPLATE/`에 두었다.

**연구에서의 의미.** 이슈 트래커는 실험 노트를 대신하지 않는다(실험 기록은 23.7장). 대신 **결정의 기록**을 남긴다. "데이터 분할을 무작위에서 시간 순서로 바꿨다(#34). 이유: 무작위 분할에서 같은 날의 측정이 train과 test에 섞였다"는 기록은, 논문 심사에서 "왜 이 분할을 썼는가"라는 질문에 답할 근거가 된다.

## 실습

두 사람이 같은 설정 파일을 수정한 PR을 만들고 conflict를 해결해 merge한다. 먼저 로컬에서 흐름을 연습하고, 그다음 실제 호스팅 서비스에서 같은 흐름을 반복한다.

**1단계: 로컬 연습.** `labs/git-and-agents/github-collaboration/two_clones.sh`는 지정한 디렉터리에 원격 역할을 하는 저장소(`remote.git`)와 두 사람의 clone(`alice/`, `bob/`)을 만든다. alice는 학습률을 3e-4로 낮추는 PR(`alice/lr`)을 이미 merge했다. bob은 그 사실을 모른 채 오래된 main에서 batch를 64로 키우고 학습률을 batch에 비례해 2e-3으로 올리는 branch(`bob/batch`)를 push했다.

```bash
# 집필 시점 예시 — 저장소 루트에서 실행
bash labs/git-and-agents/github-collaboration/two_clones.sh setup /tmp/collab
cd /tmp/collab/bob
git fetch origin
git log --oneline --graph --all
git merge origin/main                # bob/batch에 최신 main을 합친다 → conflict
```

집필 시점에 실행했을 때 conflict는 다음과 같았다. 두 사람이 서로 다른 줄(`lr`과 `batch_size`)을 고쳤는데도 두 줄이 붙어 있어 한 덩어리의 conflict가 되었다.

```
<<<<<<< HEAD
lr: 2.0e-3
batch_size: 64
=======
lr: 3.0e-4
batch_size: 32
>>>>>>> origin/main
epochs: 10
```

여기서 한쪽을 고르면 안 된다. alice의 의도는 "발산을 막기 위해 학습률을 낮춘다"이고, bob의 의도는 "batch를 키우면서 학습률을 batch에 비례해 키운다"이다. 두 의도를 모두 반영하면 batch 64, 학습률 $3 \times 10^{-4} \times 64/32 = 6 \times 10^{-4}$이다. 이 판단이 맞는지는 두 사람이 PR에서 합의하고, 합친 설정으로 짧은 학습을 돌려 확인한다. 해결 후 다음을 실행한다.

```bash
# 집필 시점 예시
git add configs/train.yaml && git commit --no-edit
git push origin bob/batch            # PR이 갱신된다
git switch main && git pull --ff-only
git merge --no-ff bob/batch -m "Merge PR: bob/batch"
git push origin main
bash <저장소 루트>/labs/git-and-agents/github-collaboration/two_clones.sh check /tmp/collab
```

`check`는 원격 main에 두 branch가 모두 들어갔는지와 conflict 표식이 남지 않았는지만 검사한다. 남은 값이 옳은지는 검사할 수 없다. 그것이 리뷰의 몫이다.

**2단계: 실제 호스팅 서비스.** 연습용 저장소를 하나 만들고 동료(또는 두 번째 계정)와 같은 흐름을 반복한다.

- `research.gitignore`를 `.gitignore`로 복사해 프로젝트에 맞게 고치고, `.env.example`과 데이터 README를 commit한다.
- main에 branch 보호를 걸어 PR을 거치지 않은 push와 force push를 막는다(설정 위치는 서비스 문서를 확인한다).
- 두 사람이 각자 이슈를 만들고, 같은 설정 파일을 다르게 고치는 PR을 하나씩 올린다. PR 설명에 무엇을·왜·어떻게 확인했는지를 쓰고 `Closes #번호`로 이슈를 연결한다.
- 서로의 PR을 리뷰한다. 요청·질문·제안을 구분해 댓글을 단다.
- 먼저 merge된 PR 뒤에 남은 PR에서 conflict를 해결하고, 해결의 근거를 PR 댓글로 남긴 뒤 merge한다.

## 실패 사례

- **데이터·checkpoint를 저장소에 commit하는 것.** `git add .`가 가장 흔한 경로다. 한 번 들어간 파일은 지워도 이력에 남아 모든 clone이 내려받고, 호스팅 서비스의 한도에 걸려 push가 막히기도 한다. 프로젝트를 시작할 때 `.gitignore`부터 commit하고, `git add`는 경로를 지정해서 쓰고, pre-commit hook으로 큰 파일을 막는다. 이미 들어갔다면 협업자와 조율해 이력을 정리할지, 그대로 두고 앞으로만 막을지 결정한다.
- **비밀정보 커밋을 삭제 commit으로만 처리하는 것.** key는 이전 commit에 그대로 있고, 유효한 상태로 남는다. 먼저 폐기·교체하고, 그다음 필요하면 이력을 정리한다.
- **리뷰를 형식으로 만드는 것.** 수천 줄짜리 PR에 "LGTM(looks good to me)"만 남기는 리뷰는 아무것도 확인하지 않는다. 리뷰할 수 없는 크기의 PR은 나눠 달라고 요청하는 것이 리뷰어의 일이다.
- **conflict를 "내 것"이나 "상대 것"으로 기계적으로 해결하는 것.** 실습의 학습률처럼 두 변경의 의도가 모두 유효할 수 있다. 의도를 모르면 상대에게 묻는다.

## 통과 기준

- 연구 프로젝트 저장소에 `.gitignore`(데이터·출력물·환경·캐시·비밀정보), `.env.example`, 데이터 README를 commit하고, `git check-ignore -v`로 데이터·checkpoint·`.env`가 무시되고 `.env.example`이 추적되는지 확인한다.
- `two_clones.sh`로 만든 저장소에서 conflict를 해결해 `check`의 세 항목을 통과하고, 남긴 학습률 값의 근거를 한 문단으로 설명한다.
- 실제 호스팅 서비스에서 branch 보호가 걸린 main에 대해 이슈 → branch → PR → 리뷰 → conflict 해결 → merge의 흐름을 한 번 수행하고, PR 링크를 남긴다.

## 참고 문헌

- Conventional Commits 1.0.0. https://www.conventionalcommits.org/en/v1.0.0/ — 커밋 메시지 형식 명세
- GitHub Docs, "Removing sensitive data from a repository". — 비밀정보 유출 시 폐기·교체를 먼저 하라는 권고와 이력 정리의 부작용
- Chacon, S., Straub, B. (2014). *Pro Git* (2nd ed.). Apress. — remote 작업과 분산 협업 흐름(2장, 5장)
