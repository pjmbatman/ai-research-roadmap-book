---
id: linux-and-remote/linux-shell
title: "Linux와 셸의 기초"
title_en: "Linux and Shell Fundamentals"
tier: core
status: draft
prerequisites: []
---

# Linux와 셸의 기초

연구 계산은 대부분 GUI가 없는 Linux 서버에서 돌아간다. 데이터를 옮기고, 로그를 뒤지고, 학습을 다시 돌리는 일은 결국 셸(shell)에서 명령을 치는 일이다. 셸을 "검색해서 복사하는 주문"으로만 쓰면, 같은 명령이 어떤 서버에서는 되고 어떤 서버에서는 안 되는 이유를 설명할 수 없다. 이 장은 그 이유를 이루는 네 가지, 즉 **파일과 경로**, **권한**, **환경변수**, **종료 코드**를 정확히 이해하는 데 집중한다.

목표는 명령을 많이 외우는 것이 아니다. 새 서버에 처음 접속했을 때 자기 작업 환경을 다시 만들고, 실패를 숨기지 않는 스크립트로 반복 작업을 자동화할 수 있으면 이 장의 목적을 이룬 것이다. 이 장의 명령은 bash 기준이며, 옵션은 GNU coreutils가 깔린 일반적인 Linux 배포판을 가정한 **집필 시점 예시**다. 서버마다 다를 수 있으니 `man <명령>`이나 `<명령> --help`로 확인하는 습관을 함께 들인다.

## 학습 목표

- 파일시스템과 권한을 이해한다
- 셸 환경변수와 startup file을 다룬다
- 반복 작업을 스크립트로 만든다

## 파일시스템과 경로

Linux의 파일은 모두 `/`(root)에서 시작하는 하나의 나무 구조 안에 있다. Windows처럼 드라이브마다 따로 시작하지 않고, 디스크·네트워크 저장소·USB가 모두 이 나무의 어느 디렉터리에 **마운트**(mount)되어 붙는다. 연구 서버에서 자주 마주치는 위치는 다음과 같다.

| 경로 | 용도 | 연구에서 주의할 점 |
|---|---|---|
| `/home/<사용자>` (`~`) | 사용자 홈. 설정 파일·코드 | 서버에 따라 quota가 작다. 대용량 데이터를 두지 않는다 |
| `/tmp` | 임시 파일. 모든 사용자가 공유 | 재부팅이나 주기적 정리로 사라질 수 있다 |
| `/usr`, `/usr/bin` | 배포판이 설치한 프로그램 | 일반 사용자는 쓸 수 없다 |
| `/usr/local`, `/opt` | 관리자가 따로 설치한 프로그램(CUDA, 상용 소프트웨어 등) | 버전이 여러 개 공존하기도 한다 |
| `/data`, `/scratch`, `/mnt/...` | 마운트된 대용량 저장소(이름은 서버마다 다름) | 백업 여부·속도·quota가 제각각이다(2.5장) |

어느 디렉터리가 어떤 장치에 마운트되어 있는지는 `df -h`(용량)나 `findmnt`(마운트 구조)로 확인한다. 같은 `/data` 아래라도 하위 디렉터리마다 다른 저장소일 수 있다.

**경로.** `/`로 시작하는 **절대 경로**(absolute path)는 어디서 실행해도 같은 파일을 가리킨다. 그렇지 않은 **상대 경로**(relative path)는 현재 작업 디렉터리(`pwd`)를 기준으로 해석된다. `.`은 현재 디렉터리, `..`은 상위 디렉터리, `~`는 홈 디렉터리다. 스크립트가 "내 컴퓨터에서는 되는데 서버의 scheduler에서는 파일을 못 찾는" 문제의 상당수는 상대 경로가 다른 작업 디렉터리에서 해석되기 때문이다.

**링크.** 같은 데이터를 여러 이름으로 부르는 방법이 두 가지 있다.

- **hard link**: 디스크의 파일 실체(inode)에 이름을 하나 더 붙인다. 원래 이름과 새 이름은 완전히 대등하다. 이름 하나를 지워도 다른 이름이 남아 있으면 데이터는 그대로다. 같은 파일시스템 안에서만 만들 수 있고, 일반적으로 디렉터리에는 만들 수 없다.
- **symbolic link**(symlink): 다른 경로를 **문자열로** 담은 작은 파일이다. 파일시스템 경계를 넘을 수 있고 디렉터리도 가리킬 수 있다. 대상이 지워지거나 옮겨지면 깨진다. 상대 경로로 만든 symlink는 **링크가 있는 디렉터리** 기준으로 해석된다.

```bash
# 집필 시점 예시: 링크 두 종류를 만들고 inode 번호(첫 열)와 링크 수(셋째 열)를 본다
echo x > a.txt
ln a.txt h.txt          # hard link
ln -s a.txt s.txt       # symbolic link
ls -li a.txt h.txt s.txt
```

실행하면 `a.txt`와 `h.txt`는 같은 inode 번호와 링크 수 2를 보이고, `s.txt`는 다른 inode 번호에 `s.txt -> a.txt`로 표시된다. 연구에서는 symlink를 주로 쓴다. 큰 데이터셋을 공유 저장소에 한 번만 두고 프로젝트 디렉터리에서 `ln -s /data/shared/imagenet data`처럼 가리키면, 코드는 항상 `data/`라는 같은 경로를 쓰면서 실제 위치는 서버마다 다르게 둘 수 있다.

::quiz{id="symlink-vs-hardlink"}

## 사용자·그룹·권한

여러 사람이 한 서버를 쓰는 연구실에서 권한은 "남의 결과를 실수로 지우지 않게" 해 주는 장치이자, "동료에게 결과를 보여 주지 못하게" 막는 흔한 원인이다.

**정의.** 모든 파일에는 소유자(user)와 그룹(group)이 있고, 세 부류의 사용자 — 소유자(u), 그룹(g), 기타(o) — 에 대해 각각 읽기(r)·쓰기(w)·실행(x) 비트가 있다. `ls -l`의 첫 열 `-rwxr-x---`에서 첫 글자는 파일 종류(`-` 일반 파일, `d` 디렉터리, `l` symlink)이고, 나머지 아홉 글자가 세 글자씩 u·g·o의 권한이다.

**디렉터리에서의 의미는 다르다.** 디렉터리에서 r은 목록을 볼 권한, w는 그 안에 파일을 만들거나 **지울** 권한, x는 그 안으로 들어가 이름으로 파일에 접근할(통과할) 권한이다. 그래서 파일 자체가 `rw-r--r--`여도 경로 위의 어느 디렉터리에 x가 없으면 열 수 없다. 또 파일을 지우는 권한은 파일이 아니라 **디렉터리**의 w가 정한다.

**8진수 표기.** r = 4, w = 2, x = 1을 더해 세 부류를 각각 한 자리로 쓴다. `chmod 750 run.sh`는 소유자 rwx(7), 그룹 r-x(5), 기타 없음(0)이다.

```python {exec="browser" session="shell"}
def perm_to_octal(s):
    """'rwxr-x---' 같은 9글자 권한 문자열을 '750' 같은 8진수 문자열로 바꾼다."""
    digits = []
    for i in range(0, 9, 3):                 # 소유자, 그룹, 기타 순
        r, w, x = s[i:i + 3]
        digits.append(str(4 * (r == "r") + 2 * (w == "w") + 1 * (x == "x")))
    return "".join(digits)

for s in ["rwxr-xr-x", "rw-r--r--", "rw-------", "rwxrwx---"]:
    print(s, "→", perm_to_octal(s))
```

```output
rwxr-xr-x → 755
rw-r--r-- → 644
rw------- → 600
rwxrwx--- → 770
```

::quiz{id="chmod-octal-value"}

`chmod`는 기호로도 쓸 수 있다. `chmod g+rx dir`은 그룹에 r과 x를 더하고, `chmod o-rwx file`은 기타의 모든 권한을 뺀다. 기존 권한의 일부만 바꿀 때는 기호 표기가 안전하다.

**새 파일의 기본 권한: umask.** 프로그램은 새 파일을 보통 666(rw-rw-rw-), 새 디렉터리를 777로 **요청**하고, 셸의 `umask`에 켜진 비트가 그중에서 꺼진다. 결과는 비트 연산 `요청 & ~umask`다.

```python {exec="browser" session="shell"}
def apply_umask(base, umask):
    return base & ~umask                      # umask에 켜진 비트를 끈다

for umask in [0o022, 0o002, 0o077]:
    f = apply_umask(0o666, umask)             # 새 파일의 기본 요청
    d = apply_umask(0o777, umask)             # 새 디렉터리의 기본 요청
    print(f"umask {umask:03o}  파일 {f:03o}  디렉터리 {d:03o}")
```

```output
umask 022  파일 644  디렉터리 755
umask 002  파일 664  디렉터리 775
umask 077  파일 600  디렉터리 700
```

집필 시점에 `umask 022`와 `umask 077`로 실제 파일·디렉터리를 만들어 `stat -c '%a'`로 확인한 결과도 위 표와 같았다. 공동 작업 디렉터리에서 그룹원이 서로의 파일을 고쳐야 한다면 `umask 002`가 맞고, 개인 키처럼 아무에게도 보이면 안 되는 파일을 다루는 동안에는 077이 맞다.

**공유 디렉터리.** 연구실 공용 디렉터리에는 특수 비트 두 개가 자주 쓰인다.

- **setgid**(`chmod g+s` 또는 앞자리 2): 디렉터리에 걸면 그 안에 새로 만든 파일이 만든 사람의 기본 그룹이 아니라 **디렉터리의 그룹**을 물려받는다. `ls -l`에서 그룹 실행 자리가 `s`로 보인다(`drwxrwsr-x` = 2775).
- **sticky bit**(`chmod +t` 또는 앞자리 1): 디렉터리에 걸면 쓰기 권한이 있어도 **자기 파일만** 지울 수 있다. `/tmp`가 이 설정이다(`drwxrwxrwt` = 1777).

**chown과 sudo.** 파일의 소유자를 바꾸는 `chown`은 관리자(root)만 쓸 수 있다. 일반 사용자는 자기가 속한 그룹으로 `chgrp`만 할 수 있다. `sudo`는 관리자가 허락한 명령을 root 권한으로 실행하는 장치이고, 공유 서버에서 일반 사용자에게는 없는 것이 보통이다. 소프트웨어를 설치할 때 sudo가 없다고 막히면, 시스템 전체가 아니라 사용자 공간(홈 아래 conda 환경, 컨테이너 등)에 설치하는 방법을 찾는다(3장). 자기 권한 범위는 `id`(속한 그룹)와 `sudo -l`(허락된 명령)로 확인한다.

::quiz{id="directory-execute-bit"}

## 셸과 환경

**셸.** 셸은 명령을 읽어 프로그램을 실행하는 프로그램이다. Linux 서버의 기본은 대개 bash이고, macOS의 기본은 zsh다. 두 셸은 대부분의 문법이 같지만 설정 파일이 다르다. 지금 셸은 `echo $SHELL`(로그인 셸)이나 `ps -p $$`(현재 프로세스)로 확인한다.

**환경변수.** 셸 변수 중 `export`한 것이 **환경변수**(environment variable)이고, 셸이 실행하는 모든 자식 프로세스(Python, 학습 스크립트)에 복사되어 전달된다. 거꾸로, 자식 프로세스가 바꾼 환경변수는 부모 셸에 돌아오지 않는다. 연구에서 자주 쓰는 환경변수는 `PATH`, `HOME`, `CUDA_VISIBLE_DEVICES`(2.5장), `OMP_NUM_THREADS`, 각종 캐시 경로(`HF_HOME` 등) 등이다.

```bash
# 집필 시점 예시
MY_VAR=1                  # 셸 변수: 자식 프로세스에 전달되지 않는다
export DATA_ROOT=/data/kim   # 환경변수: 자식 프로세스에 전달된다
python -c 'import os; print(os.environ.get("DATA_ROOT"))'
env | grep DATA_ROOT      # 현재 환경변수 목록에서 찾기
```

**PATH.** 명령 이름(`python`)을 입력하면 셸은 `PATH`에 콜론으로 나열된 디렉터리를 **앞에서부터** 찾아 처음 나온 실행 파일을 실행한다. 같은 이름이 여러 곳에 있어도 경고하지 않는다. 그래서 PATH의 **순서**가 곧 "어느 python이 실행되는가"를 정한다. 아래는 두 가지 PATH에서 같은 명령이 어디로 가는지를 흉내 낸 것이다.

```python {exec="browser" session="shell"}
installed = {                                 # 디렉터리 → 들어 있는 실행 파일(가정)
    "/home/kim/.local/bin": {"jupyter"},
    "/opt/conda/envs/proj/bin": {"python", "python3", "pip", "jupyter"},
    "/usr/bin": {"python3", "ls"},
}

def which(cmd, path):
    for d in path.split(":"):                 # 앞에서부터 찾아 처음 나온 것을 쓴다
        if cmd in installed.get(d, set()):
            return f"{d}/{cmd}"
    return None

path_a = "/opt/conda/envs/proj/bin:/home/kim/.local/bin:/usr/bin"
path_b = "/home/kim/.local/bin:/usr/bin:/opt/conda/envs/proj/bin"
for cmd in ["python3", "jupyter", "pip"]:
    print(f"{cmd:8s} A: {which(cmd, path_a):34s} B: {which(cmd, path_b)}")
```

```output
python3  A: /opt/conda/envs/proj/bin/python3   B: /usr/bin/python3
jupyter  A: /opt/conda/envs/proj/bin/jupyter   B: /home/kim/.local/bin/jupyter
pip      A: /opt/conda/envs/proj/bin/pip       B: /opt/conda/envs/proj/bin/pip
```

PATH B에서는 `pip`만 conda 환경 것이 실행되고 `python3`은 시스템 것이 실행된다. 이 상태에서 `pip install`한 패키지를 `python3`이 찾지 못하는 일이 생긴다. 실제 서버에서는 `type -a python3`으로 후보 전체를 순서대로 보고, 실행된 인터프리터는 `python3 -c 'import sys; print(sys.executable)'`로 확인한다. 셸이 기억해 둔 경로를 비우려면 `hash -r`을 쓴다.

::quiz{id="path-first-match"}

**startup file의 로딩 순서.** 환경변수와 별칭(alias)을 매번 치지 않으려고 설정 파일에 적는다. 어느 파일이 읽히는지는 셸이 **login shell**인지, **interactive shell**인지에 따라 다르다. bash의 규칙(집필 시점 bash 매뉴얼의 "Bash Startup Files" 절 기준)은 다음과 같다.

| 상황 | 읽는 파일 |
|---|---|
| login shell (SSH 로그인, `bash -l`) | `/etc/profile`, 그다음 `~/.bash_profile`, `~/.bash_login`, `~/.profile` 중 **처음 발견한 하나** |
| login이 아닌 interactive shell (로그인 후 `bash`를 새로 띄움, 데스크톱 터미널 창의 다수) | `~/.bashrc` |
| non-interactive shell (스크립트 실행) | 위 파일을 읽지 않는다(`BASH_ENV`가 있으면 그 파일) |

그래서 흔히 쓰는 구성은 **설정은 `~/.bashrc`에 모으고, `~/.bash_profile`(또는 `~/.profile`)에서 `.bashrc`를 불러오는** 것이다. 많은 배포판의 기본 `~/.profile`이 이미 이렇게 되어 있지만, 누군가 `~/.bash_profile`을 새로 만들면 `~/.profile`은 더 이상 읽히지 않는다는 점에 주의한다.

```bash
# 집필 시점 예시: ~/.bash_profile
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
```

zsh는 `~/.zshenv`(모든 zsh), `~/.zprofile`(login), `~/.zshrc`(interactive) 순으로 읽는다. 맥에서 쓰던 `~/.zshrc`를 리눅스 서버의 bash에 그대로 복사하면 읽히지 않는다. 설정을 고친 뒤에는 새 셸을 열거나 `source ~/.bashrc`로 현재 셸에 다시 읽힌다.

::quiz{id="bashrc-login-shell"}

:::tip[연구 팁]
설정 파일(`.bashrc`, `.tmux.conf`, `.ssh/config`의 공개 가능한 부분)을 git 저장소 하나에 모아 두면(흔히 dotfiles라 부른다) 새 서버에서 몇 분 만에 같은 환경을 재구성할 수 있다. 비밀번호·토큰·개인키는 이 저장소에 절대 넣지 않는다(2.3장).
:::

## 파일 탐색과 텍스트 처리

학습 로그, 설정 파일, 실험 결과 CSV는 모두 텍스트다. 이것들을 Python으로 열기 전에 셸에서 빠르게 훑어보는 도구가 있다.

| 목적 | 명령 (집필 시점 예시) |
|---|---|
| 목록 보기 | `ls -lh`(크기를 읽기 쉽게), `ls -lt`(최근 수정 순), `ls -a`(숨김 파일 포함) |
| 파일 찾기 | `find . -name '*.ckpt' -mtime -1`(하루 안에 바뀐 checkpoint), `find /data -size +10G` |
| 내용 검색 | `grep -rn 'lr=' configs/`, `rg 'TODO' src/`(ripgrep, 설치되어 있다면) |
| 앞뒤 보기 | `head -n 20 file`, `tail -n 50 file`, `tail -f train.log`(늘어나는 로그 따라가기) |
| 넘겨 보기 | `less file`(`/`로 검색, `q`로 종료) |
| 세기·정렬 | `wc -l`, `sort`, `sort -n`(숫자 순), `uniq -c`(연속 중복 세기) |
| 열 뽑기·치환 | `cut -d, -f2`, `awk '{print $3}'`, `sed 's/old/new/g'` |

이 도구들은 한 줄씩 흘려 처리하므로 GB 단위 로그도 메모리에 다 올리지 않고 다룬다. `labs/linux-and-remote/linux-shell/sample_train.log`(11줄짜리 예시 로그)로 해 보면 다음과 같다.

```bash
# 로그 수준(셋째 열)별 줄 수: 열 뽑기 → 정렬 → 중복 세기 → 많은 순
awk '{print $3}' sample_train.log | sort | uniq -c | sort -rn
# 마지막 세 epoch의 loss
grep -o 'loss=[0-9.]*' sample_train.log | tail -n 3
```

집필 시점의 실행 결과는 다음과 같다.

```text
      6 INFO
      3 WARN
      2 ERROR
loss=1.322
loss=1.198
loss=1.105
```

`uniq`는 **연속된** 중복만 묶는다. 위에서 `sort`를 빼고 `awk '{print $3}' sample_train.log | uniq -c`를 실행하면 `2 INFO`, `1 WARN`, `1 INFO`, `1 WARN`, …처럼 끊어진 결과가 나온다. 그래서 `sort | uniq -c`는 한 묶음의 관용구로 외워 둔다. 로그에서 `ERROR` 두 줄이 `No space left on device`라는 것도 `grep ERROR sample_train.log` 한 줄로 바로 보인다. 이 오류의 원인 추적은 2.2장에서 다룬다.

## 파이프와 리다이렉션

**표준 스트림.** 모든 프로세스는 세 개의 스트림을 갖고 시작한다. 번호(file descriptor)로 0은 stdin(입력), 1은 stdout(정상 출력), 2는 stderr(오류·경고 출력)다. Python의 `print`는 stdout으로, 예외의 traceback과 `logging`의 기본 출력은 stderr로 간다.

| 문법 | 의미 |
|---|---|
| `a \| b` | a의 stdout을 b의 stdin으로 |
| `> f` | stdout을 파일 f로(덮어쓰기) |
| `>> f` | stdout을 파일 f 끝에 덧붙이기 |
| `2> f` | stderr를 파일 f로 |
| `2>&1` | stderr를 "지금 stdout이 가리키는 곳"으로 복제 |
| `< f` | 파일 f를 stdin으로 |
| `\| tee f` | 화면에 보이면서 파일에도 저장 |

**순서가 중요하다.** 리다이렉션은 왼쪽부터 처리된다. `cmd > log 2>&1`은 stdout을 파일로 돌린 **다음** stderr를 그곳으로 복제하므로 둘 다 파일에 간다. `cmd 2>&1 > log`는 stderr를 먼저 당시의 stdout(터미널)으로 복제하고 나서 stdout만 파일로 돌리므로, 오류는 터미널에 남는다. 집필 시점에 `ls /nonexist`로 두 형태를 실행해 보면 첫 형태는 오류 메시지가 파일에 담기고, 둘째 형태는 파일이 비고 메시지가 화면에 나온다.

```bash
# 집필 시점 예시: 학습 출력 전체를 파일로 남기면서 화면에서도 본다
python train.py --config cfg.yaml 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
# 필요 없는 출력 버리기
some_noisy_tool 2>/dev/null
```

학습이 죽었을 때 원인이 담긴 traceback은 stderr에 있다. stdout만 파일로 남겼다면 그 정보는 연결이 끊긴 터미널과 함께 사라진다(2.2장).

::quiz{id="redirect-order"}

## 압축·동기화

**tar.** 여러 파일을 하나로 묶는다(archive). 작은 파일이 수십만 개인 데이터셋은 그대로 옮기면 파일 하나마다 생기는 부가 비용 때문에 매우 느리므로, 묶어서 옮기는 편이 빠르다.

```bash
# 집필 시점 예시
tar -czf results.tar.gz results/        # c: 만들기, z: gzip 압축, f: 파일 이름
tar -tzf results.tar.gz | head          # t: 목록만 보기(풀기 전에 확인)
tar -xzf results.tar.gz -C /scratch/kim # x: 풀기, -C: 풀 위치
zip -r figures.zip figures/ && unzip -l figures.zip   # Windows 사용자와 주고받을 때
```

**rsync.** 두 위치(로컬끼리 또는 SSH 너머, 2.3장)를 동기화한다. 이미 같은 파일은 건너뛰므로, 중단된 전송을 다시 실행하면 이어서 진행된다.

```bash
# 집필 시점 예시
rsync -avh --progress src/ dst/            # a: 권한·시간 보존하며 재귀, v: 자세히, h: 읽기 쉬운 크기
rsync -avhn --delete src/ dst/             # n(--dry-run): 실제로 하지 않고 무엇이 바뀔지만 출력
```

**끝의 슬래시에 주의한다.** `rsync -a src/ dst/`는 src의 **내용**을 dst 안에 넣고, `rsync -a src dst/`는 `dst/src/`라는 디렉터리를 만든다. `--delete`는 src에 없는 파일을 dst에서 **지운다**. 방향이나 슬래시를 잘못 쓰면 결과를 날릴 수 있으므로 `--delete`를 쓸 때는 반드시 `-n`으로 먼저 확인한다. 정확한 동작은 `man rsync`의 사용 예시 부분으로 확인한다.

## Bash 스크립트

같은 명령 묶음을 세 번째 치고 있다면 스크립트로 만들 때다. 스크립트는 실험의 **절차를 기록**하는 문서이기도 하다.

**기본 문법.**

```bash
#!/usr/bin/env bash
# 집필 시점 예시: 변수, 조건, 반복, 함수
DATA_DIR="/data/kim/project"         # = 양옆에 공백이 없어야 한다
SEEDS=(0 1 2)                        # 배열

check_dir() {                        # 함수
    if [[ ! -d "$1" ]]; then         # 디렉터리가 아니면
        echo "없음: $1" >&2          # 오류 메시지는 stderr로
        return 1
    fi
}

check_dir "$DATA_DIR"
for seed in "${SEEDS[@]}"; do
    python train.py --seed "$seed" --data "$DATA_DIR" > "log_seed${seed}.txt" 2>&1
    echo "seed=$seed exit=$?"        # $? = 직전 명령의 종료 코드
done
```

변수는 항상 큰따옴표로 감싼다(`"$DATA_DIR"`). 감싸지 않으면 값에 공백이 있을 때 인자가 여러 개로 쪼개진다.

**종료 코드.** 모든 명령은 끝날 때 0–255의 정수를 남긴다. 0은 성공, 그 밖은 실패다. 셸의 `if`, `&&`, `||`는 모두 이 값을 본다. Python 스크립트는 처리되지 않은 예외로 끝나면 1을, `sys.exit(n)`이면 n을 남긴다. 그래서 학습 스크립트가 실패를 **예외나 0이 아닌 종료 코드로 알리는 것**이 자동화의 전제다. 오류를 `print`만 하고 정상 종료하는 스크립트는 셸에서 성공으로 보인다.

**`set -euo pipefail`.** bash는 기본적으로 명령이 실패해도 다음 줄로 넘어간다. 데이터 준비가 실패했는데 학습이 빈 데이터로 돌고, 결과가 덮어써지는 사고가 여기서 생긴다. 스크립트 맨 위에 다음 줄을 둔다.

- `-e`: 명령이 실패하면 스크립트를 즉시 멈춘다.
- `-u`: 정의되지 않은 변수를 쓰면 오류로 멈춘다. 오타 난 변수가 빈 문자열이 되어 `rm -rf "$OUTDIR/"`이 엉뚱한 곳을 지우는 사고를 막는다.
- `-o pipefail`: 파이프라인 중 하나라도 실패하면 파이프라인 전체를 실패로 본다.

`-e`에는 예외가 있다. `if`의 조건, `&&`·`||`로 이어진 목록의 앞부분에서 일어난 실패로는 멈추지 않는다. 또 `pipefail`이 없으면 파이프라인의 종료 코드는 마지막 명령의 것이라 `false | tee log`는 성공으로 처리된다. 집필 시점에 bash 5.2로 확인한 동작은 다음과 같다.

```bash
#!/bin/bash
set -e
false | true
echo "A: pipefail 없이는 파이프 앞의 실패가 묻혀 여기까지 온다"
set -o pipefail
false | true
echo "B: 출력되지 않는다"
```

```text
A: pipefail 없이는 파이프 앞의 실패가 묻혀 여기까지 온다
```

스크립트는 `B`를 출력하지 않고 종료 코드 1로 끝난다. 명령 치환을 대입하는 `x=$(false)`도 스크립트를 멈춘다.

::quiz{id="set-e-pitfalls"}

**trap.** 실패한 지점을 기록하거나 임시 파일을 지우려면 `trap`을 쓴다. `trap '명령' ERR`은 `set -e`로 멈추게 되는 실패가 일어났을 때, `trap '명령' EXIT`은 스크립트가 어떤 이유로든 끝날 때 실행된다.

:::note[분야 예시]
기후 모델 출력은 흔히 날짜별 NetCDF 파일 수천 개로 나온다. "파일 목록 만들기 → 각 파일에서 변수 하나 추출 → 월평균 계산"을 셸 스크립트로 묶을 때, 한 날짜 파일이 손상되어 추출이 실패했는데 스크립트가 계속 진행하면 월평균이 조용히 하루 적은 데이터로 계산된다. `set -euo pipefail`과 "처리한 파일 수 = 기대한 파일 수" 검사가 이런 오류를 결과가 나오기 전에 드러낸다.
:::

:::note[분야 예시]
단백질 구조 예측이나 분자 동역학처럼 외부 프로그램을 연달아 부르는 파이프라인(서열 검색 → 정렬 → 구조 계산)에서는 앞 단계 프로그램이 빈 출력 파일을 남기고 0으로 끝나기도 한다. 종료 코드만 믿지 말고 `[[ -s 파일 ]]`(파일이 존재하고 크기가 0보다 큼) 같은 출력 검사를 단계 사이에 둔다.
:::

## 실습

데이터 준비부터 학습까지를 하나의 스크립트로 묶고, 어느 단계든 실패하면 즉시 멈추게 한다. 예시 스크립트는 `labs/linux-and-remote/linux-shell/pipeline.sh`다. 세 단계(데이터 준비 → 행 수 검사 → 학습)로 되어 있고, 다음을 갖췄다.

- `set -euo pipefail`
- 모든 변수에 기본값(`${FAIL_AT:-}`)을 주어 `-u`에서도 안전하게
- `trap ... ERR`로 실패한 명령과 줄 번호를 기록
- 학습 출력은 `tee`로 화면과 로그 파일에 동시에
- 실행한 인터프리터의 경로(`command -v`)를 맨 처음 기록

정상 실행과, `FAIL_AT=prepare`로 준비 단계를 일부러 실패시킨 실행의 집필 시점 결과는 다음과 같다(시각은 실행할 때마다 다르다).

```text
$ bash pipeline.sh r1
[09:15:26] python = /usr/local/bin/python3
[09:15:26] 1/3 데이터 준비
[09:15:26] 2/3 데이터 검사
[09:15:26] 3/3 학습
slope=2.009
[09:15:26] 완료: r1

$ FAIL_AT=prepare bash pipeline.sh r2
[09:15:26] python = /usr/local/bin/python3
[09:15:26] 1/3 데이터 준비
원본 데이터를 찾을 수 없음(모의 실패)
[09:15:26] 실패: "$PYTHON" -c "raise SystemExit('원본 데이터를 찾을 수 없음(모의 실패)')" (줄 20, exit 1)
```

두 번째 실행은 종료 코드 1로 끝나고, `r2/` 디렉터리에는 아무 파일도 생기지 않았다. 이 스크립트를 자기 프로젝트의 실제 단계로 바꾼 뒤 다음을 확인한다.

1. 각 단계를 하나씩 일부러 실패시켜(존재하지 않는 경로, 잘못된 인자) 스크립트가 그 자리에서 멈추는지 본다.
2. `set -euo pipefail` 줄을 지우고 같은 실패를 다시 일으켜, 무엇이 달라지는지 기록한다.
3. 새 계정이나 새 서버에서 `.bashrc`·`.bash_profile`을 처음부터 구성하고, `type -a python`과 `echo $PATH`로 의도한 인터프리터가 먼저 잡히는지 확인한다.

권한 표기는 다음 연습문제로 익힌다.

::exercise{id="mode-to-octal"}

## 실패 사례

- **PATH 순서 때문에 다른 python이 실행되는 것.** conda 환경을 활성화했는데 `.bashrc` 뒷부분의 `export PATH=/usr/local/bin:$PATH`가 그 앞에 끼어들거나, `pip install --user`로 깔린 `~/.local/bin`의 도구가 먼저 잡힌다. 증상은 "설치했는데 `ModuleNotFoundError`"다. `type -a python`, `sys.executable`로 실제 실행 파일을 확인하기 전에는 패키지를 다시 설치하지 않는다.
- **실패한 명령 뒤에도 계속 진행되는 스크립트.** `set -e`가 없거나, 있어도 `pipefail`이 없어 `python prep.py | tee log`의 실패가 묻힌다. 빈 데이터로 학습이 돌고 이전 결과가 덮어써진다.
- **따옴표 없는 변수와 정의되지 않은 변수.** `rm -rf $OUT/*`에서 `OUT`이 비어 있으면 `rm -rf /*`가 된다. 변수는 `"$OUT"`으로 감싸고 `set -u`를 켠다.
- **`sort` 없이 `uniq -c`를 쓰는 것.** 연속하지 않은 중복은 따로 세어 잘못된 집계가 나온다.
- **권한 문제를 `chmod 777`로 푸는 것.** 공유 서버에서 모두에게 쓰기 권한을 주면 다른 사람의 실수로 결과가 지워질 수 있다. 막힌 지점(대개 경로 위 디렉터리의 x)을 찾아 필요한 비트만 연다.

## 통과 기준

- 새 서버에서 `.bashrc`·`.bash_profile`을 구성하고, login shell과 non-login shell 모두에서 의도한 `python`이 `type -a python`의 첫 줄로 나오는 것을 확인한다.
- 데이터 준비 → 검사 → 학습 스크립트를 `set -euo pipefail`로 작성하고, 각 단계를 일부러 실패시켰을 때 그 단계에서 0이 아닌 종료 코드로 멈추는 것을 보인다.
- 연습문제 `mode-to-octal`의 테스트를 통과한다.
