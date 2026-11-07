---
id: git-and-agents/git-fundamentals
title: "Git의 모델과 기본 작업"
title_en: "Git Fundamentals"
tier: core
status: draft
prerequisites: [linux-and-remote/linux-shell]
---

# Git의 모델과 기본 작업

실험 결과는 그 결과를 만든 코드와 설정이 있을 때만 근거가 된다. 석 달 전의 그림을 다시 그리려는데 그때의 코드가 어느 것인지 모르면, 그 그림은 재현할 수 없는 주장이 된다. Git은 이 문제를 푸는 도구다. 파일의 모든 상태에 지울 수 없는 이름을 붙이고, 그 상태들이 어떤 순서로 이어졌는지를 기록한다.

Git 명령은 많고 이름도 직관적이지 않다. 그래서 명령을 외우는 방식으로 배우면 처음 보는 상황에서 막힌다. 이 장은 반대 순서로 간다. 먼저 Git이 저장하는 **데이터 모델**(객체와 그래프)을 이해하고, 그 위에서 각 명령이 그래프의 무엇을 바꾸는지로 명령을 읽는다. 이렇게 보면 branch·merge·rebase·reset이 모두 "그래프에 점을 더하거나, 이름표를 옮기는 일"로 정리되고, 실수를 복구하는 방법도 그 그림에서 나온다.

## 학습 목표

- working tree·index·commit의 관계를 설명한다
- branch·merge·rebase를 다룬다
- 실수를 복구한다

## Git의 데이터 모델

**왜 필요한가.** "commit을 잃어버렸다", "rebase가 이력을 망가뜨렸다"는 말은 대부분 데이터 모델을 모를 때 나온다. Git은 한 번 저장한 내용을 거의 지우지 않는다. 사라진 것처럼 보이는 것은 대개 그것을 가리키던 이름표가 옮겨 간 것이다. 무엇이 저장되고 무엇이 이름표인지 구분하면 복구 방법이 보인다.

**정의.** Git 저장소(repository)의 핵심은 객체(object) 데이터베이스다. 기본 객체는 세 종류다.

| 객체 | 담는 것 | 비유 |
|---|---|---|
| blob | 파일 하나의 내용(이름·권한은 없음) | 파일 내용 |
| tree | 디렉터리 하나: 이름·권한과 각 항목이 가리키는 blob 또는 하위 tree | 디렉터리 목록 |
| commit | 최상위 tree 하나, parent commit(0개 이상), 작성자·시각, 메시지 | 프로젝트 전체의 스냅숏 |

commit은 변경분(diff)이 아니라 **프로젝트 전체의 스냅숏**을 가리킨다. 두 commit 사이의 diff는 저장된 것이 아니라 두 스냅숏을 비교해 그때그때 계산한 것이다. 바뀌지 않은 파일은 같은 blob을 다시 가리키므로 스냅숏이라도 저장 공간이 크게 늘지 않는다.

**내용 주소.** 모든 객체의 이름(id)은 객체 내용의 해시다. 같은 내용은 어느 컴퓨터에서 만들어도 같은 id를 갖고, 한 바이트라도 다르면 id가 달라진다. 이런 방식을 내용 주소 지정(content addressing)이라 한다. blob의 id는 `blob <바이트 수>`와 NUL 문자 한 개를 내용 앞에 붙인 뒤 SHA-1 해시를 계산한 값이다.

```python {exec="browser" session="git"}
import hashlib

def blob_id(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()     # "blob <바이트 수>" + NUL
    return hashlib.sha1(header + content).hexdigest()

print(blob_id(b"hello\n"))
print(blob_id(b"hello!\n"))
print(blob_id(b"lr = 0.001\n") == blob_id(b"lr = 0.001\n"))
```

```output
ce013625030ba8dba906f756967f9e9ca394464a
4effa19f4f75f846c3229b9dbdbad14eff362f32
True
```

첫 줄의 값은 터미널에서 `printf 'hello\n' | git hash-object --stdin`을 실행한 결과와 같다. 직접 확인해 본다. 느낌표 하나를 더하자 id가 완전히 달라졌다. 기본 저장소 형식은 SHA-1을 쓰고, 저장소를 만들 때 SHA-256 형식을 고를 수도 있다. 어느 쪽이든 원리는 같다.

commit도 객체이므로 id는 내용의 해시이고, commit의 내용에는 tree id와 **parent commit의 id**가 들어 있다. 실제 commit 객체를 열어 보면 다음과 같다(id와 시각은 저장소마다 다르다. 서명을 설정했다면 서명 줄이 더 붙는다).

```bash
# 집필 시점 예시 — 출력의 id는 저장소마다 다르다
$ git cat-file -p HEAD
tree e070ef709c36bc2bd7a06c05165c178f9758c1d4
parent 4e23ecc102440424273b2a228da74900ac09605a
author Drill <drill@example.com> 1794043069 +0000
committer Drill <drill@example.com> 1794043069 +0000

eval: 평가 스크립트

$ git cat-file -p 'HEAD^{tree}'
100644 blob c66f745ac25428a82d9b45e282f4028a5b32342f	data.py
100644 blob 0be6a42f4836a3e4f354d294b173fc53c4bf4452	eval.py
...
```

여기서 중요한 결론이 하나 나온다. 과거 commit의 파일 한 글자를 고치면 그 blob의 id가 바뀌고, 그것을 담은 tree의 id, 그 commit의 id가 바뀐다. 그 commit을 parent로 가리키던 다음 commit도 내용(parent 줄)이 바뀌므로 id가 바뀌고, 이 변화는 끝까지 전파된다. 그래서 **commit id 하나가 그 commit에 이르는 전체 이력을 보증한다.** 논문에 "commit `76b96d1`로 실험했다"고 적으면, 그 시점의 코드 전체와 이력이 하나로 정해진다.

**commit DAG.** commit은 parent를 가리키므로 commit들은 그래프를 이룬다. 화살표는 항상 과거를 향하고 순환이 없으므로 방향성 비순환 그래프(directed acyclic graph, DAG)다. 보통 commit은 parent가 하나, 첫 commit은 parent가 없고, merge commit은 parent가 둘 이상이다.

```
A ← B ← C ← F        ← main
     ↖
       D ← E          ← feature ← HEAD
```

**ref와 HEAD.** 40자리 id를 외울 수는 없으므로 Git은 commit에 이름표를 붙인다. 이것이 참조(ref)다. **branch는 commit 하나를 가리키는 움직이는 이름표**일 뿐이다. `refs/heads/main`은 commit id 하나를 담고 있고, 그 branch에서 commit하면 이름표가 새 commit으로 옮겨 간다. tag는 보통 움직이지 않는 이름표다. `HEAD`는 "지금 작업 중인 위치"로, 대개 branch를 가리키는 이름표의 이름표다.

```bash
# 집필 시점 예시
$ cat .git/HEAD
ref: refs/heads/main
$ git rev-parse main
76b96d1430bc8f0ebc45671d12b69001daf0e749
```

`HEAD`가 branch가 아니라 commit을 직접 가리키는 상태를 detached HEAD라 한다. 과거 commit을 `git switch --detach <id>`로 꺼내 볼 때 생긴다. 이 상태에서 만든 commit은 어떤 branch도 가리키지 않으므로, 다른 곳으로 이동하면 찾기 어려워진다. 남기고 싶으면 `git switch -c <새 branch>`로 이름표를 붙인다.

**연구에서의 의미.** 데이터 모델에서 세 가지가 나온다. 첫째, 실험 기록에는 branch 이름이 아니라 commit id를 적는다. branch는 움직이지만 commit id는 영원히 같은 스냅숏을 가리킨다. 둘째, 이력을 "고치는" 모든 작업(amend, rebase)은 사실 새 commit을 만드는 일이다. 셋째, 이름표를 잘못 옮겨도 객체는 한동안 남아 있으므로 대부분의 실수는 복구할 수 있다.

## 세 영역

**왜 필요한가.** `git commit`을 했는데 방금 고친 내용이 빠져 있거나, 넣을 생각이 없던 파일이 들어가 있다. 이런 혼란은 Git이 파일을 세 곳에 따로 두기 때문에 생긴다.

**정의.**

| 영역 | 무엇인가 | 어디 있는가 |
|---|---|---|
| working tree | 지금 편집기로 열어 보는 파일들 | 프로젝트 디렉터리 |
| index (staging area) | **다음 commit이 될 스냅숏**의 초안 | `.git/index` |
| repository | 만들어진 commit과 객체들 | `.git/objects` 등 |

`git add`는 working tree의 파일 내용을 blob으로 저장하고 index가 그 blob을 가리키게 한다. `git commit`은 **working tree가 아니라 index**로 tree를 만들어 commit한다. 그래서 `git add` 뒤에 파일을 또 고치면, 그 두 번째 수정은 다시 `add`하기 전까지 commit에 들어가지 않는다.

::quiz{id="staged-version-committed"}

**직관.** index는 "포장대"다. 작업대(working tree)에서 여러 가지를 고치더라도, 이번 상자(commit)에 넣을 것만 골라 포장대에 올린다. 한 파일 안에서도 일부 변경만 고를 수 있다(`git add -p`). 덕분에 버그 수정과 실험 설정 변경을 한 번에 고쳤더라도 두 개의 commit으로 나눠 담을 수 있다.

세 영역 사이의 차이를 보는 명령도 세 가지다.

| 명령 | 비교하는 것 |
|---|---|
| `git diff` | working tree ↔ index (아직 add하지 않은 변경) |
| `git diff --staged` | index ↔ `HEAD` (다음 commit에 들어갈 변경) |
| `git diff HEAD` | working tree ↔ `HEAD` (모든 변경) |

`git diff`가 아무것도 보여 주지 않는데 commit할 것이 있다면, 이미 전부 add했다는 뜻이다. commit 직전에는 `git diff --staged`를 읽는 습관을 들인다.

## 기본 작업

일상적인 작업은 다음 순서를 반복한다.

```bash
# 집필 시점 예시
git status                      # 세 영역의 차이를 요약: 무엇이 바뀌었고 무엇이 staged인가
git diff                        # 아직 add하지 않은 변경을 줄 단위로 확인
git add src/model.py            # 이번 commit에 넣을 파일만 index에 올린다
git add -p configs/train.yaml   # 파일 안에서 변경 덩어리(hunk)를 골라 올린다
git diff --staged               # 다음 commit의 내용을 마지막으로 확인
git commit -m "model: residual block의 정규화 위치를 pre-norm으로 변경"
git log --oneline --graph -n 10 # 최근 이력을 그래프로
git show HEAD                   # 방금 만든 commit의 메시지와 diff
```

`git status`의 출력은 세 영역 표와 같은 순서로 읽는다. "Changes to be committed"는 index에 있는 것, "Changes not staged for commit"은 working tree에만 있는 것, "Untracked files"는 Git이 아직 한 번도 추적하지 않은 파일이다.

**좋은 commit의 단위.** commit은 "하나의 논리적 변경"이어야 한다. 기준은 두 가지다.

- **되돌릴 수 있는가.** 이 commit 하나만 되돌렸을 때(`git revert`) 의미 있는 상태가 되는가. 버그 수정과 학습률 변경이 한 commit에 섞여 있으면 하나만 되돌릴 수 없다.
- **메시지 한 줄로 요약되는가.** "여러 수정"이나 "update"로밖에 요약할 수 없다면 commit이 너무 크다.

메시지 첫 줄은 무엇을 바꿨는지, 본문은 **왜** 바꿨는지를 쓴다. 코드의 diff는 무엇이 바뀌었는지는 보여 주지만 이유는 보여 주지 않는다. "정규화를 pre-norm으로 바꿨다. 깊이 24에서 post-norm이 첫 1000 step 안에 발산했기 때문이다" 같은 본문은 6개월 뒤의 자신에게 가장 쓸모 있는 실험 노트가 된다. 커밋 메시지의 형식 규칙은 4.2장에서 다룬다.

:::note[분야 예시]
기후 모델의 물리 파라미터화(parameterization)를 바꾸는 연구에서는 한 번의 실행이 수일씩 걸린다. 실행 로그에 commit id를 함께 남겨 두면, 결과가 이상할 때 `git diff <이전 실행의 id> <이번 실행의 id>`로 두 실행 사이에 바뀐 코드를 정확히 볼 수 있다. 실행 스크립트가 시작할 때 `git rev-parse HEAD`와 `git status --porcelain`의 출력을 로그 첫 줄에 쓰게 하면, 커밋하지 않은 변경이 섞인 채 실행했는지도 기록된다(22.2장의 재현성에서 다시 다룬다).
:::

## branch·merge·rebase

**왜 필요한가.** 연구 코드에서는 여러 시도를 동시에 진행한다. 새 손실 함수를 실험하는 동안 데이터 로더의 버그를 고쳐야 하고, 실패한 실험은 버려야 한다. branch는 이 시도들을 서로 섞이지 않게 분리하고, merge와 rebase는 성공한 시도를 다시 합친다.

**branch.** branch를 만드는 것은 이름표 하나를 만드는 일이므로 비용이 거의 없다. `git switch -c exp-loss`는 현재 commit을 가리키는 새 이름표를 만들고 `HEAD`를 그쪽으로 옮긴다. 이후 commit하면 `exp-loss` 이름표만 앞으로 간다.

**merge의 두 가지 경우.** main에서 `git merge feature`를 실행하면 Git은 먼저 두 branch의 관계를 본다.

- **fast-forward.** main의 끝 commit이 feature의 조상이면, main에는 feature에 없는 commit이 없다. 이때는 합칠 것이 없으므로 main 이름표를 feature의 끝으로 **옮기기만** 한다. 새 commit은 생기지 않는다.
- **merge commit.** 두 branch가 갈라져 각자 commit이 있으면, parent가 둘인 새 commit을 만든다. 이 commit의 tree는 아래의 3-way merge로 계산한다.

이 판정은 그래프의 조상 관계만으로 할 수 있다. 앞의 그림의 DAG로 계산해 본다.

```python {exec="browser" session="git"}
parents = {"A": [], "B": ["A"], "C": ["B"], "F": ["C"],   # main
           "D": ["B"], "E": ["D"]}                         # feature

def ancestors(c):
    """c에서 parent를 따라 도달할 수 있는 모든 commit (c 포함)."""
    seen, stack = set(), [c]
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack.extend(parents[x])
    return seen

def is_ancestor(a, b):
    return a in ancestors(b)

print("feature만 있는 commit:", sorted(ancestors("E") - ancestors("F")))
print("main만 있는 commit:   ", sorted(ancestors("F") - ancestors("E")))
print("main에서 merge feature가 fast-forward?", is_ancestor("F", "E"))
parents["M"] = ["F", "E"]                                  # merge commit
print("merge 뒤 feature에서 merge main이 fast-forward?", is_ancestor("E", "M"))
```

```output
feature만 있는 commit: ['D', 'E']
main만 있는 commit:    ['C', 'F']
main에서 merge feature가 fast-forward? False
merge 뒤 feature에서 merge main이 fast-forward? True
```

앞의 두 줄은 `git log main..feature`와 `git log feature..main`이 보여 주는 집합이다(`a..b`는 "b의 조상이지만 a의 조상은 아닌 commit"). 실제 저장소에서 조상 관계는 `git merge-base --is-ancestor F E`로 확인한다. 이 명령은 출력 대신 종료 코드(0이면 조상)로 답한다. fast-forward만 허용하고 싶으면 `git merge --ff-only`, fast-forward가 가능해도 merge commit을 남기고 싶으면 `git merge --no-ff`를 쓴다.

**3-way merge.** 두 branch를 합칠 때 Git은 두 끝 commit만 보지 않고, 둘의 **공통 조상**(merge-base)을 함께 본다. 위 그림에서 F와 E의 merge-base는 B다. 기준(base) B와 비교해 어느 쪽이 바꿨는지를 따지면 판정이 간단해진다.

| base | ours | theirs | 결과 |
|---|---|---|---|
| x | x | x | x |
| x | y | x | y (우리만 바꿈) |
| x | x | z | z (상대만 바꿈) |
| x | y | y | y (양쪽이 같게 바꿈) |
| x | y | z | **conflict** |

공통 조상 없이 두 파일만 비교하면, 한 줄이 다를 때 누가 바꿨는지 알 수 없다. 기준이 있어야 "우리는 그대로 두고 상대만 바꿨다"를 판단할 수 있다. 실험 설정 파일로 판정해 본다. 실제 Git은 diff로 변경 덩어리(hunk)를 맞춘 뒤 덩어리 단위로 판정하지만, 여기서는 줄 수가 같은 경우로 단순화한다.

```python {exec="browser" session="git"}
def merge3(base, ours, theirs):
    out, conflicts = [], 0
    for b, o, t in zip(base, ours, theirs):
        if o == t:
            out.append(o)                 # 양쪽이 같다
        elif o == b:
            out.append(t)                 # 우리는 그대로, 상대만 바꿨다
        elif t == b:
            out.append(o)                 # 상대는 그대로, 우리만 바꿨다
        else:
            out.append(f"<<< {o} | {b} | {t} >>>")
            conflicts += 1
    return out, conflicts

base   = ["lr = 1e-3", "batch = 32", "seed = 0", "epochs = 10"]
ours   = ["lr = 3e-4", "batch = 32", "seed = 0", "epochs = 10"]
theirs = ["lr = 1e-3", "batch = 64", "seed = 0", "epochs = 20"]
print(merge3(base, ours, theirs))
theirs[0] = "lr = 1e-2"
print(merge3(base, ours, theirs))
```

```output
(['lr = 3e-4', 'batch = 64', 'seed = 0', 'epochs = 20'], 0)
(['<<< lr = 3e-4 | lr = 1e-3 | lr = 1e-2 >>>', 'batch = 64', 'seed = 0', 'epochs = 20'], 1)
```

첫 merge는 conflict 없이 끝났다. 그런데 결과를 보면 학습률 3e-4, batch 64, epoch 20의 조합이다. 두 사람 누구도 이 조합으로 실험한 적이 없다. **conflict가 없다는 것은 텍스트가 겹치지 않았다는 뜻이지, 결과가 맞다는 뜻이 아니다.** merge 뒤에는 테스트와 최소 실험을 다시 돌려야 한다.

**conflict 해결.** 같은 곳을 양쪽이 다르게 바꾸면 Git은 merge를 멈추고 파일에 표식을 남긴다.

```
<<<<<<< HEAD
lr = 3e-4
||||||| d73b6a5
lr = 1e-3
=======
lr = 1e-2
>>>>>>> feature
```

가운데 `|||||||` 구역(base, 뒤의 문자열은 공통 조상 commit의 id)은 `git config merge.conflictStyle diff3`을 설정하면 보인다. base가 보이면 양쪽이 각각 무엇을 바꾸려 했는지 알 수 있으므로 설정해 두는 것이 좋다. 해결 절차는 다음과 같다.

1. `git status`로 conflict가 난 파일 목록을 본다.
2. 각 파일에서 표식을 지우고 올바른 내용으로 고친다. 둘 중 하나를 고르는 것이 아니라 **두 변경의 의도를 모두 반영한 내용**을 쓰는 것이 목표다. 의도를 모르면 상대에게 묻는다.
3. `git add <파일>`로 해결했음을 표시하고, 모두 끝나면 `git commit`(rebase 중이면 `git rebase --continue`).
4. 해결이 엉망이 되면 `git merge --abort`(또는 `git rebase --abort`)로 merge 전 상태로 돌아간다.

**rebase.** `git rebase main`(feature에서 실행)은 feature에만 있는 commit들(D, E)을 main의 끝 F 위에 **하나씩 다시 적용해 새 commit D′, E′를 만들고**, feature 이름표를 E′로 옮긴다.

```
rebase 전:  A ← B ← C ← F           ← main
                 ↖
                   D ← E             ← feature

rebase 후:  A ← B ← C ← F ← D′ ← E′  ← feature
                              ↑
                              main
```

D′는 D와 같은 변경을 담지만 parent가 다르고(B 대신 F) tree도 다르므로 **id가 다른 새 commit**이다. 원래의 D, E는 지워지지 않고, 이름표만 잃은 채 객체 데이터베이스에 남는다. 이제 main에서 `git merge feature`를 하면 fast-forward가 되어 이력이 한 줄로 이어진다.

::quiz{id="rebase-new-commits"}

merge와 rebase는 같은 문제를 다르게 푼다.

| | merge | rebase |
|---|---|---|
| 이력의 모양 | 갈라졌다 합쳐진 그대로 남는다 | 한 줄로 펴진다 |
| 기존 commit | 그대로 둔다 | 새 commit으로 **바꿔 쓴다** |
| conflict | 한 번에 모아서 해결 | commit마다 따로 해결할 수 있다 |
| 안전한 대상 | 공유된 branch를 포함한 모든 branch | 아직 아무와도 공유하지 않은 자기 commit |

rebase의 규칙은 하나다. Pro Git의 표현으로 **"저장소 밖에 존재하고 다른 사람이 그 위에 작업했을 수 있는 commit은 rebase하지 않는다."** 이미 push해 동료가 받은 commit을 rebase하면 동료의 저장소에는 D, E가, 원격에는 D′, E′가 있게 된다. 동료가 그대로 pull하면 같은 변경이 두 번 들어간 이력이 생긴다. 자기 로컬 branch를 정리할 때(push 전), 또는 혼자 쓰는 branch를 최신 main 위로 옮길 때 rebase를 쓰고, 공유 branch를 합칠 때는 merge를 쓴다.

## 복구

**왜 필요한가.** 실수는 반드시 한다. 중요한 것은 실수했을 때 상황을 그래프로 그리고, 잃은 것이 객체인지 이름표인지 판단하는 능력이다.

**되돌리는 명령들.** 무엇을 되돌리는지에 따라 명령이 다르다.

| 상황 | 명령 | 하는 일 |
|---|---|---|
| working tree의 수정을 버린다 | `git restore <파일>` | index의 내용으로 파일을 덮어쓴다 |
| add를 취소한다 | `git restore --staged <파일>` | index를 `HEAD`의 내용으로 되돌린다. 파일 수정은 남는다 |
| 마지막 commit 메시지·내용을 고친다(push 전) | `git commit --amend` | 새 commit으로 **바꿔 쓴다** |
| branch 이름표를 과거로 옮긴다(push 전) | `git reset <commit>` | 아래 세 모드 참고 |
| 공유된 commit의 효과를 취소한다 | `git revert <commit>` | 반대 변경을 담은 **새 commit**을 더한다 |

`git reset`은 세 가지 모드가 있고, 세 영역 중 어디까지 되돌리는지가 다르다.

| 모드 | branch 이름표 | index | working tree |
|---|---|---|---|
| `--soft` | 옮긴다 | 그대로 | 그대로 |
| `--mixed` (기본값) | 옮긴다 | 되돌린다 | 그대로 |
| `--hard` | 옮긴다 | 되돌린다 | **되돌린다** |

`--hard`만이 아직 commit하지 않은 수정을 지운다. commit하지 않은 수정은 객체로 저장된 적이 없을 수 있으므로, 이것이 Git에서 **정말로 잃을 수 있는 거의 유일한 경우**다. `git reset --hard` 전에는 `git status`로 남은 수정이 없는지 확인한다.

**revert와 reset의 구분.** reset은 이력을 고친다. 이름표를 과거로 옮기므로, 이미 다른 사람이 받은 commit에 쓰면 원격과 이력이 갈라져 force push가 필요해진다. revert는 이력을 고치지 않고 **더한다**. 문제의 commit은 이력에 그대로 남고, 그것을 취소하는 commit이 뒤에 붙는다. 공유된 branch에서는 revert를 쓴다.

**reflog.** reflog는 `HEAD`와 각 branch 이름표가 **어디를 거쳐 왔는지**를 로컬에 기록한 일지다. commit, reset, rebase, switch가 모두 기록된다. 이름표를 옮겨 "잃어버린" commit도 reflog에는 id가 남아 있다.

```bash
# 집필 시점 예시 (실습 저장소에서 실행한 출력의 일부)
$ git reflog main
59fe448 main@{0}: reset: moving to HEAD~2
76b96d1 main@{1}: commit: eval: 평가 스크립트
4e23ecc main@{2}: commit: train: 학습 루프
59fe448 main@{3}: commit: model: baseline MLP
```

`main@{1}`은 "main 이름표가 한 번 전에 가리키던 commit"이다. 따라서 `git reset --hard main@{1}`로 reset 이전으로 돌아갈 수 있다. reset, rebase, merge는 직전 위치를 `ORIG_HEAD`에도 남기므로, 그 직후라면 `git reset --hard ORIG_HEAD`도 같은 일을 한다.

::quiz{id="recover-after-reset-hard"}

reflog에는 한계가 있다.

- **로컬 기록이다.** push되지 않고, 다른 사람의 clone에는 없다. 새로 clone한 저장소에는 그 이전 기록이 없다.
- **branch를 지우면 그 branch의 reflog도 지워진다.** 지운 branch의 commit은 `HEAD`의 reflog(`git reflog`)에서 찾는다. 그 branch에서 commit했다면 `HEAD`가 그 commit을 지나간 기록이 남아 있다.
- **영원하지 않다.** 오래된 항목은 정리된다. Git 문서의 기본값은 90일이고, 현재 branch에서 도달할 수 없는 commit의 항목(amend나 rebase 이전의 commit 등)은 30일이다. 정리된 뒤 어떤 이름표에서도 도달할 수 없는 객체는 garbage collection으로 지워질 수 있다.

reflog에도 없으면 `git fsck --lost-found`가 어떤 이름표에서도 도달할 수 없는(dangling) commit과 blob을 찾아 준다. add만 하고 commit하지 않은 내용도 blob으로는 남아 있으므로 여기서 찾을 수 있는 경우가 있다.

**복구의 순서.** 실수했다고 느끼면 다음 순서를 따른다.

1. **멈춘다.** 추가로 reset·rebase·gc를 하지 않는다. 특히 force push를 하지 않는다.
2. `git status`와 `git log --oneline --graph --all`로 현재 그래프를 본다.
3. `git reflog`(또는 `git reflog <branch>`)에서 사고 직전의 commit id를 찾는다.
4. 먼저 그 commit에 **새 이름표**를 붙인다(`git branch rescue <id>`). 이름표가 생긴 commit은 정리되지 않으므로, 그다음에 천천히 원래 branch를 옮겨도 된다.

## 실습

일부러 사고 세 개를 낸 연습 저장소를 만들고 reflog로 복구한다. 스크립트는 `labs/git-and-agents/git-fundamentals/reflog_drill.sh`다. 이 스크립트는 지정한 새 디렉터리에 저장소를 만들고 다음 사고를 낸다.

1. main에서 `git reset --hard HEAD~2`로 commit 두 개(`train:`, `eval:`)를 잃는다.
2. 실험 branch `exp-lr`에서 commit한 뒤, merge하지 않고 `git branch -D`로 지운다.
3. feature에서 commit 두 개를 만든 뒤 잘못된 기준으로 rebase해(`git rebase --onto main HEAD~1`) 첫 commit이 빠지게 한다.

```bash
# 집필 시점 예시 — 저장소 루트에서 실행
bash labs/git-and-agents/git-fundamentals/reflog_drill.sh setup /tmp/drill
cd /tmp/drill
git log --oneline --graph --all     # 사고 뒤의 그래프
git reflog                          # HEAD가 거쳐 온 기록
git reflog main                     # main 이름표의 기록
git reflog feature                  # feature 이름표의 기록
```

과제는 다음과 같다.

- 복구 **전에** 현재 그래프와 복구 후의 목표 그래프를 종이에 그린다. 각 사고에서 잃은 것이 객체인지 이름표인지 적는다.
- 세 사고를 각각 복구한다. 사고 2에서는 `git reflog exp-lr`이 기록 대신 "unknown revision" 오류를 내는 이유를 설명한다.
- `bash labs/git-and-agents/git-fundamentals/reflog_drill.sh check /tmp/drill`로 검사한다. 다섯 항목이 모두 "통과"여야 한다.

집필 시점에 이 스크립트로 만든 저장소에서는 다음 세 명령으로 복구했다(commit id는 실행할 때마다 다르므로 자신의 reflog에서 찾는다).

```bash
# 집필 시점 예시
git reset --hard main@{1}            # 사고 1: main이 reset 직전 위치로
git branch exp-lr a1d0ab5            # 사고 2: HEAD reflog에서 찾은 'exp:' commit에 이름표
git branch -f feature feature@{1}    # 사고 3: rebase 직전의 feature로 (main에서 실행)
```

이어서 3-way merge의 기준이 되는 merge-base를 직접 계산한다. 테스트가 바로 채점한다.

::exercise{id="merge-base"}

## 실패 사례

- **공유 branch에서 force push로 이력을 덮어쓰는 것.** 로컬에서 rebase나 reset을 한 뒤 push가 거부되면 `git push --force`로 밀어붙이고 싶어진다. 그 사이 동료가 push한 commit이 있었다면 그 commit은 원격에서 사라지고, 동료는 자기 로컬 reflog 말고는 복구할 방법이 없다. 공유 branch(main 등)에서는 force push를 하지 않고 revert와 merge로 해결한다. 혼자 쓰는 branch를 정리하느라 꼭 필요하면 `git push --force-with-lease`를 쓴다. 이 옵션은 원격 branch가 내가 마지막으로 본 상태와 다르면 push를 거부하므로, 모르는 사이에 생긴 다른 사람의 commit을 덮어쓰지 않는다. 저장소 호스팅 서비스의 branch 보호 설정으로 main의 force push 자체를 막아 두는 것이 가장 확실하다(4.2장).
- **의미 없는 거대 commit.** 일주일치 작업을 "update" 한 번으로 commit하면, 버그가 언제 들어왔는지 찾을 수 없고(`git bisect`가 무력해진다), 실험 하나만 되돌릴 수도 없다. 성능이 갑자기 바뀐 원인을 찾을 때 commit 단위가 곧 원인 추적의 해상도다. `git add -p`로 변경을 나눠 담는다.
- **`git add .`로 데이터와 출력물까지 commit하는 것.** 수 GB의 checkpoint가 한 번 commit되면, 다음 commit에서 지워도 과거 commit에 영원히 남아 모든 clone이 그것을 내려받는다. `.gitignore`는 4.2장에서 다룬다.
- **detached HEAD에서 만든 commit을 잃는 것.** 과거 commit을 꺼내 실험하다 commit하고 다른 branch로 이동하면, 그 commit을 가리키는 이름표가 없다. Git이 이동할 때 경고를 출력하므로 읽고, `git branch <이름> <id>`로 이름표를 붙인다.

## 통과 기준

- 주어진 저장소의 `git log --oneline --graph --all` 출력을 보고, commit DAG와 branch·`HEAD` 이름표를 그림으로 그리고 각 명령(commit, merge, rebase, reset)이 그림을 어떻게 바꾸는지 설명한다.
- `reflog_drill.sh`로 만든 저장소의 사고 세 개를 복구해 `check`의 다섯 항목을 모두 통과한다.
- 연습문제 `merge-base`의 테스트를 통과한다.

## 참고 문헌

- Chacon, S., Straub, B. (2014). *Pro Git* (2nd ed.). Apress. — Git 내부 구조(10장)와 rebase의 위험(3.6절). 저자들이 웹에 무료로 공개하고 있다.
