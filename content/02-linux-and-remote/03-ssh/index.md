---
id: linux-and-remote/ssh
title: "SSH와 원격 접속"
title_en: "SSH and Remote Access"
tier: core
status: draft
prerequisites: [linux-and-remote/linux-shell]
---

# SSH와 원격 접속

연구 계산의 대부분은 내 노트북이 아니라 연구실 GPU 서버, 학교 클러스터, 클라우드 인스턴스에서 일어난다. 그 모든 곳에 들어가는 문이 SSH(Secure Shell)다. 매번 IP 주소와 비밀번호를 치며 접속하고, 원격의 대시보드를 보려고 포트를 공개하고, 개인키를 이 서버 저 서버에 복사해 두는 습관은 불편할 뿐 아니라 연구실 전체의 보안 구멍이 된다.

이 장은 SSH를 "접속 명령"이 아니라 **인증·설정·터널**의 세 층으로 이해하는 것을 목표로 한다. 키 기반 인증과 `~/.ssh/config`로 어떤 서버든 `ssh gpu1` 한 단어로 들어가고, 포트 포워딩으로 원격의 TensorBoard·Jupyter를 내 브라우저에서 열고, 원격 편집기가 어디서 무엇을 실행하는지 구분한다. 명령과 옵션은 OpenSSH 클라이언트를 기준으로 한 **집필 시점 예시**이며, 설정 키의 정확한 의미는 `man ssh_config`, `man ssh`로 확인한다.

## 학습 목표

- 키 기반 인증과 SSH 설정을 구성한다
- 포트 포워딩으로 원격 서비스를 연다
- 원격 편집 환경을 구성한다

## 인증과 키

**왜 키인가.** 비밀번호 인증은 비밀번호가 네트워크를 건너 서버로 간다(암호화되어 있더라도 서버는 비밀번호를 받는다). 서버가 사칭된 것이라면 비밀번호를 넘겨준 셈이 되고, 추측 공격에도 노출된다. 그래서 많은 연구 서버는 비밀번호 로그인을 막고 공개키 인증만 허용한다.

**공개키 인증.** 키는 **쌍**으로 만든다. **개인키**(private key)는 내 컴퓨터에만 두고 절대 밖으로 보내지 않는다. **공개키**(public key)는 이름 그대로 공개해도 되며, 접속할 서버의 `~/.ssh/authorized_keys`에 한 줄로 등록한다. 접속할 때 서버는 무작위 값을 보내고, 클라이언트가 개인키로 서명해 돌려보내면 서버가 공개키로 서명을 확인한다. 개인키는 네트워크를 한 번도 건너지 않는다.

```bash
# 집필 시점 예시
ssh-keygen -t ed25519 -C "kim@laptop-2026"     # 키 쌍 생성. 암호문(passphrase)을 설정한다
# → ~/.ssh/id_ed25519 (개인키), ~/.ssh/id_ed25519.pub (공개키)
ssh-copy-id -i ~/.ssh/id_ed25519.pub kim@gpu1.example.org   # 공개키를 서버의 authorized_keys에 추가
ssh kim@gpu1.example.org                        # 이제 키로 인증된다
```

`ssh-copy-id`가 없는 환경에서는 공개키 파일의 내용을 서버의 `~/.ssh/authorized_keys` 끝에 한 줄로 붙인다. `-C`의 주석은 서버 관리자가 "이 키가 누구의 어느 기기 것인지" 알아보는 데 쓰인다. 기기마다 키를 따로 만들면, 노트북을 잃어버렸을 때 그 기기의 공개키 한 줄만 지우면 된다.

**권한.** ssh 클라이언트는 개인키 파일이 그룹이나 기타 사용자에게 조금이라도 열려 있으면 `Permissions 0644 ... are too open` 같은 경고와 함께 그 키를 쓰지 않는다. 서버의 sshd도 기본 설정(`StrictModes`)에서 홈 디렉터리, `~/.ssh`, `authorized_keys`가 다른 사람에게 쓰기 가능하면 키 인증을 거부한다. 2.1장의 8진수 표기로 보면 기준은 "그룹·기타 비트(077)가 꺼져 있는가"다.

```python {exec="browser" session="ssh"}
def too_open(mode):
    return mode & 0o077 != 0                  # 그룹(070)·기타(007) 비트 중 하나라도 켜져 있으면

for name, mode in [("id_ed25519", 0o600), ("id_ed25519", 0o644), ("id_ed25519", 0o400),
                   ("id_ed25519.pub", 0o644), ("~/.ssh", 0o700), ("~/.ssh", 0o775)]:
    print(f"{name:15s} {mode:04o}  그룹·기타 비트 {mode & 0o077:03o}  {'열림' if too_open(mode) else '닫힘'}")
```

```output
id_ed25519      0600  그룹·기타 비트 000  닫힘
id_ed25519      0644  그룹·기타 비트 044  열림
id_ed25519      0400  그룹·기타 비트 000  닫힘
id_ed25519.pub  0644  그룹·기타 비트 044  열림
~/.ssh          0700  그룹·기타 비트 000  닫힘
~/.ssh          0775  그룹·기타 비트 075  열림
```

개인키는 600(또는 400)이어야 한다. 공개키 `.pub`은 "열림"이어도 문제없다. 공개용이기 때문이다. `~/.ssh` 디렉터리는 700으로 두는 것이 안전하다. 775처럼 그룹에게 쓰기가 열려 있으면 다른 사람이 `authorized_keys`를 바꿔 넣을 수 있으므로 sshd가 거부한다. 정리하면 `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519 ~/.ssh/authorized_keys`다.

::quiz{id="private-key-permissions"}

**암호문과 ssh-agent.** 개인키 파일 자체는 암호문으로 암호화해 둔다. 노트북을 잃어버려도 키 파일만으로는 쓸 수 없게 하기 위해서다. 매번 암호문을 치는 불편은 **ssh-agent**가 덜어 준다. agent는 로그인 세션 동안 풀린 키를 메모리에 들고 있다가 서명 요청에 응답한다(`ssh-add ~/.ssh/id_ed25519`로 등록, `ssh-add -l`로 목록 확인).

**agent forwarding의 위험.** `ssh -A`나 config의 `ForwardAgent yes`는 접속한 원격 서버에서 내 agent에 서명을 요청할 수 있게 한다. 원격 서버에서 다시 GitHub이나 다른 서버로 접속할 때 편리하다. 그러나 접속해 있는 동안 **그 서버의 root(또는 그 서버를 장악한 침입자)도 같은 통로로 내 agent를 쓸 수 있다.** 개인키 파일은 넘어가지 않지만, 키를 쓸 수는 있다는 뜻이다. 여러 사람이 root를 공유하는 서버나 신뢰할 수 없는 서버에서는 켜지 않는다. 서버를 건너가는 목적이라면 뒤의 `ProxyJump`가 더 안전한 대안이다.

**host key와 known_hosts.** 인증은 양방향이다. 내가 서버에 나를 증명하듯, 서버도 자신의 **host key**로 자신을 증명한다. 처음 접속할 때 ssh는 서버의 fingerprint를 보여 주며 믿을지 묻고, 수락하면 `~/.ssh/known_hosts`에 기록한다. 이후에는 접속마다 같은 키인지 확인한다. 이 첫 질문에 습관적으로 `yes`를 치기 전에, 관리자가 공지한 fingerprint와 비교하는 것이 원칙이다.

키가 바뀌면 `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!`와 함께 접속이 거부된다. 서버를 재설치했을 수도 있고, 중간에서 누군가 서버를 사칭하고 있을 수도 있다. 내 쪽에서는 둘을 구분할 수 없으므로 관리자에게 확인한 뒤에만 `ssh-keygen -R <호스트>`로 그 호스트의 옛 기록을 지운다.

::quiz{id="host-key-changed"}

::quiz{id="agent-forwarding-risk"}

## ~/.ssh/config

**왜 필요한가.** `ssh -i ~/.ssh/id_lab -p 2222 -J kim@gate.example.org lab@10.0.3.21`을 매번 치는 대신 `ssh gpu1`로 끝내고 싶다. 게다가 `scp`, `rsync`, VS Code, git 같은 도구도 모두 같은 config를 읽으므로, 한 번 적어 두면 어디서나 같은 별칭이 통한다.

**구조.** config는 `Host` 줄로 시작하는 블록의 나열이다. `Host` 뒤의 이름(패턴)이 명령줄에서 쓴 호스트 이름과 맞으면, 그 블록의 설정이 적용된다. 연구 서버에서 자주 쓰는 키는 다음과 같다.

| 키 | 의미 |
|---|---|
| `HostName` | 실제 접속할 주소(도메인 또는 IP) |
| `User` | 원격 사용자 이름 |
| `Port` | SSH 포트(기본 22) |
| `IdentityFile` | 쓸 개인키 파일 |
| `IdentitiesOnly yes` | 지정한 키만 시도(agent의 다른 키를 차례로 들이밀다 거부당하는 것을 막음) |
| `ProxyJump` | 이 서버로 가기 전에 거쳐 갈 jump host |
| `ServerAliveInterval` | 일정 시간(초)마다 연결 확인 신호를 보냄 |

```bash
# 집필 시점 예시: ~/.ssh/config
Host bastion
    HostName gate.example.org
    User kim
    IdentityFile ~/.ssh/id_ed25519

# %h는 명령줄의 호스트 이름(gpu1, gpu2)으로 바뀐다. 주석은 줄 전체로 쓴다.
Host gpu1 gpu2
    HostName %h.lab.internal
    User kim
    ProxyJump bastion
    IdentityFile ~/.ssh/id_ed25519

Host *
    ServerAliveInterval 60
    IdentitiesOnly yes
```

**규칙 하나: 처음 얻은 값이 이긴다.** ssh는 config를 위에서 아래로 읽으며, 각 설정에 대해 **처음 얻은 값**을 쓰고 이후 값은 무시한다. 많은 설정 파일이 "뒤의 값이 덮어쓴다"는 것과 반대다. 그래서 구체적인 Host 블록을 위에, `Host *` 기본값을 맨 아래에 둔다. 아래는 이 규칙을 흉내 낸 최소 구현이다.

```python {exec="browser" session="ssh"}
from fnmatch import fnmatch

config = [                                    # (Host 패턴, {설정}) — 파일에 적힌 순서
    ("*",    {"user": "kim", "serveraliveinterval": "60"}),
    ("gpu1", {"hostname": "10.0.3.21", "user": "lab", "proxyjump": "bastion"}),
]

def resolve(host, blocks):
    result = {}
    for pattern, options in blocks:
        if fnmatch(host, pattern):
            for key, value in options.items():
                result.setdefault(key, value)  # 이미 정해진 값은 덮어쓰지 않는다
    return result

print("Host * 가 먼저:", resolve("gpu1", config))
print("gpu1 이 먼저: ", resolve("gpu1", config[::-1]))
```

```output
Host * 가 먼저: {'user': 'kim', 'serveraliveinterval': '60', 'hostname': '10.0.3.21', 'proxyjump': 'bastion'}
gpu1 이 먼저:  {'hostname': '10.0.3.21', 'user': 'lab', 'proxyjump': 'bastion', 'serveraliveinterval': '60'}
```

`Host *`를 위에 두면 gpu1의 `User lab`이 조용히 무시된다. 오류도 경고도 없다. 실제 서버에서는 `ssh -G gpu1`이 이 모든 규칙을 적용한 최종 설정을 출력하므로, 접속이 이상하면 `ssh -G gpu1 | grep -i -E '^(user|hostname|proxyjump) '`부터 확인한다. 예외도 있다. `IdentityFile`처럼 여러 값을 모으는 키는 맞는 값을 모두 차례로 시도한다. 정확한 목록은 `man ssh_config`에서 확인한다.

::quiz{id="ssh-config-order"}

**ProxyJump.** 연구실·학교 서버는 흔히 외부에 bastion(jump host, 관문 서버) 하나만 열어 두고, 그 뒤의 GPU 서버는 내부망에만 둔다. `ProxyJump bastion`(명령줄에서는 `ssh -J bastion gpu1`)은 bastion을 **통로로만** 쓴다. 내 컴퓨터에서 gpu1까지의 SSH 연결이 bastion을 지나는 터널 안에서 끝까지 암호화되고, gpu1에 대한 인증도 내 컴퓨터의 키로 직접 이루어진다. bastion에 개인키를 둘 필요도, agent forwarding을 켤 필요도 없다.

:::note[분야 예시]
대형 실험 시설(가속기, 망원경, 핵융합 장치 등)의 데이터 분석 서버는 보안 등급이 높아 관문 서버 두 단계를 거쳐야 하는 경우가 있다. `ProxyJump gate1,gate2`처럼 쉼표로 여러 jump host를 이어 쓰면 config 한 블록으로 표현된다. 시설마다 OTP(일회용 비밀번호)나 인증서 기반 SSH를 요구하기도 하니, 시설의 사용자 안내 문서를 따른다.
:::

## 포트 포워딩

**왜 필요한가.** 원격 서버에서 TensorBoard(6006), Jupyter(8888) 같은 웹 서비스를 띄우면, 그 서비스는 서버 자신(localhost)에게만 열려 있거나 방화벽에 막혀 내 브라우저에서 바로 열리지 않는다. 그렇다고 모든 네트워크에 공개하면 인증 없는 웹 서버를 세상에 여는 것이 된다. **포트 포워딩**(port forwarding)은 SSH 연결 안에 터널을 만들어, SSH 인증을 통과한 나만 그 서비스에 닿게 한다.

**local forwarding (`-L`).** `-L 로컬포트:목적지호스트:목적지포트`는 "내 컴퓨터의 로컬포트로 들어오는 연결을, 원격 서버가 본 목적지호스트:목적지포트로 넘긴다"는 뜻이다. 목적지의 `localhost`는 **원격 서버 자신**이다.

```bash
# 집필 시점 예시: 원격 TensorBoard를 내 브라우저로
# (원격, tmux 안에서) tensorboard --logdir runs --port 6006
ssh -N -L 16006:localhost:6006 gpu1     # -N: 원격 명령 없이 터널만
# 내 브라우저에서 http://localhost:16006
```

로컬 포트를 원격과 다른 번호(16006)로 고르면, 내 컴퓨터에서 이미 돌고 있는 TensorBoard와 헷갈리지 않는다. 자주 쓴다면 config에 `LocalForward 16006 localhost:6006`으로 적어 둔다.

::quiz{id="local-forward-tensorboard"}

**remote forwarding (`-R`).** 반대 방향이다. `-R 원격포트:목적지호스트:목적지포트`는 원격 서버의 원격포트로 들어오는 연결을 **내 쪽**의 목적지로 넘긴다. 예를 들어 내 컴퓨터에서만 접근할 수 있는 라이선스 서버나 사내 패키지 저장소를 원격 서버에서 잠깐 써야 할 때 쓴다. 원격에 내 쪽 서비스를 노출하는 일이므로 필요할 때만, 짧게 쓴다.

**Jupyter.** 원격에서 `jupyter lab --no-browser --port 8888`로 띄우면 접속 URL과 token이 출력된다. `ssh -N -L 8890:localhost:8888 gpu1`로 터널을 연 뒤, 출력된 URL의 포트만 8890으로 바꿔 내 브라우저에서 연다. token은 비밀번호와 같으니 채팅이나 저장소에 붙여 넣지 않는다.

::widget{name="ssh-tunnel"}

::quiz{id="local-forward-port"}

:::caution[함정]
포워딩이 "안 된다"는 문제의 상당수는 원격 서비스가 다른 포트에서 떴기 때문이다. Jupyter나 TensorBoard는 요청한 포트가 이미 쓰이고 있으면 다른 포트로 바꿔 뜨기도 한다. 원격에서 출력된 실제 포트 번호를 확인하고, 원격 서버에서 `curl -s localhost:<포트> | head`로 서비스가 응답하는지 먼저 본다.
:::

## 파일 전송

| 도구 | 쓰임 | 주의 |
|---|---|---|
| `scp` | 파일 몇 개를 한 번 복사 | 중단되면 처음부터. 비교·재개 기능이 없다 |
| `rsync -avP` | 디렉터리 동기화, 대용량 | 바뀐 파일만 보낸다. `-P`(= `--partial --progress`)로 중단된 파일도 이어 받는다 |
| `sshfs` | 원격 디렉터리를 로컬에 마운트 | 편집·탐색용. 학습 데이터 읽기에는 매우 느리다 |

```bash
# 집필 시점 예시
scp results.csv gpu1:~/project/                          # config의 별칭을 그대로 쓴다
rsync -avP gpu1:~/project/outputs/ ./outputs/             # 원격 → 로컬 동기화(끝의 / 주의, 2.1장)
rsync -avPn --delete ./src/ gpu1:~/project/src/           # -n: 무엇이 바뀔지 먼저 확인
tar -cf - many_small_files/ | ssh gpu1 'tar -xf - -C /scratch/kim'   # 작은 파일 수십만 개는 묶어서 흘려보낸다
```

**대용량 전송의 주의점.**

- **작은 파일이 많으면 느리다.** 파일 하나마다 생기는 부가 비용이 쌓인다. 한 덩어리로 묶어(tar) 보내거나, 애초에 데이터를 큰 파일 몇 개(HDF5, Parquet, WebDataset 같은 shard)로 저장한다.
- **끊김에 대비한다.** 몇 시간짜리 전송은 tmux 안에서 `rsync -P`로 한다. 끊기면 같은 명령을 다시 실행해 이어 간다.
- **도착한 것을 검증한다.** 전송 뒤 양쪽에서 파일 개수와 `sha256sum` 결과를 비교한다.
- **어디로 보내는지 확인한다.** 홈 디렉터리 quota가 작은 서버가 많다. 대용량 데이터는 scratch나 데이터 전용 저장소로 보낸다(2.5장).

## 원격 편집

**선택지.** 원격 서버의 코드를 고치는 방법은 셋이다. (1) 서버에서 터미널 편집기(vim, nano, emacs)를 쓴다. (2) 로컬에서 고치고 rsync·git으로 보낸다. (3) **원격 IDE**를 쓴다. VS Code의 Remote-SSH 확장, JetBrains의 원격 개발 기능 등이 여기에 해당한다(제품 이름과 기능은 집필 시점 예시이며 빠르게 바뀐다).

**원격 확장의 실행 위치.** VS Code Remote-SSH를 예로 들면, 연결할 때 원격 서버의 홈 디렉터리에 VS Code 서버 프로그램을 설치하고 그것을 실행한다. 내 컴퓨터의 창은 화면과 키보드 입력만 맡는다. 그 결과는 다음과 같다.

- 편집기 안의 **터미널**은 원격 셸이다.
- 파일 탐색, 검색, 언어 서버(자동 완성, 타입 검사), 디버거, Jupyter 커널 같은 **작업 공간 확장**은 원격에서 실행되고, 원격에 따로 설치된다.
- 테마, 키보드 단축키 같은 **UI 확장**만 로컬에서 실행된다.
- Python 인터프리터 선택은 **원격 서버의 경로**(예: 원격의 conda 환경)를 골라야 한다.

이 구조 덕분에 원격 IDE는 로컬처럼 빠르게 느껴지지만, 원격 서버 쪽에 상주 프로세스가 생긴다. 이 프로세스는 서버의 CPU·메모리를 쓰고, 큰 저장소를 열면 파일 감시와 인덱싱으로 부하가 커진다. 클러스터의 로그인 노드처럼 여러 사람이 함께 쓰는 곳에서는 사용을 제한하는 규칙이 있을 수 있다(2.5장). 또 IDE에 연결된 원격 Jupyter 커널이나 디버거에서 학습을 돌리면, IDE 창을 닫거나 연결이 끊길 때 함께 멈출 수 있다. 오래 걸리는 학습은 IDE 밖, tmux 안에서 돌리고 IDE는 편집과 짧은 실험에 쓴다.

::quiz{id="remote-extension-location"}

## 연결 유지와 복구

**왜 끊기는가.** 집·카페의 공유기, 학교 방화벽, NAT 장비는 한동안 아무 데이터도 오가지 않는 연결을 정리한다. 그러면 SSH 창이 갑자기 멈추고, 한참 뒤 `Broken pipe`로 끝난다.

**ServerAliveInterval.** 클라이언트가 지정한 초마다 서버에 확인 신호를 보내 연결을 "살아 있게" 하고, 응답이 `ServerAliveCountMax`번 연속 없으면 연결을 끊는다. 멈춘 창을 붙잡고 기다리지 않고 빨리 끊긴 것을 알게 해 준다는 점도 중요하다. 위 config 예시처럼 `Host *`에 `ServerAliveInterval 60`을 두는 구성이 흔하다. 기본값과 정확한 동작은 `man ssh_config`로 확인한다.

**mosh.** mosh(mobile shell)는 SSH로 처음 인증한 뒤 UDP로 통신하며, 노트북이 잠들었다 깨어나거나 와이파이에서 휴대폰 테더링으로 바뀌어도 세션을 이어 간다. 원격 서버에 mosh 서버가 설치되어 있고, 방화벽이 mosh가 쓰는 UDP 포트를 허용해야 한다. 연구실 서버에서는 둘 중 하나가 막혀 있는 경우가 많으니 관리자에게 확인한다.

**끊긴 뒤 재접속.** 어떤 설정도 연결이 끊기는 것을 완전히 막지 못한다. 그래서 핵심은 **끊겨도 잃지 않는 구조**다.

1. 학습은 처음부터 tmux 안에서 돌리고, 출력은 로그 파일로 남긴다(2.2장).
2. 재접속하면 `tmux attach -t <세션>`으로 화면에 돌아오고, `tail -f`로 로그를 이어 본다.
3. 포트 포워딩 터널은 SSH 연결과 함께 사라지므로 다시 연다. 원격의 TensorBoard·Jupyter는 tmux 안에서 돌고 있었다면 그대로 살아 있다.

:::note[분야 예시]
천문학 관측자나 현장 연구자는 이동 중 불안정한 회선으로 원격 분석 서버에 접속하는 일이 많다. 이때 mosh로 셸을 유지하고, 서버의 tmux 안에서 분석을 돌리고, 결과 그림은 rsync로 필요한 것만 가져오는 구성이 끊김에 강하다. 회선이 가장 약한 구간에는 대화형 화면보다 "작업 제출 → 나중에 결과 회수" 방식이 맞다(2.4장, 2.5장).
:::

## 실습

jump host를 거치는 GPU 서버 접속을 config로 정리하고, 원격 TensorBoard를 로컬 브라우저로 연다.

1. 기기 전용 키를 만든다: `ssh-keygen -t ed25519 -C "<이름>@<기기>"`. 암호문을 설정하고 `ssh-add`로 agent에 등록한다.
2. bastion과 GPU 서버의 `authorized_keys`에 공개키를 등록하고, `~/.ssh`와 키 파일의 권한을 700/600으로 확인한다(`stat -c '%a %n' ~/.ssh ~/.ssh/*`).
3. 이 장의 config 예시를 자기 서버에 맞게 고친다. 구체적인 Host를 위에, `Host *`를 아래에 둔다. `ssh -G gpu1`로 `user`, `hostname`, `proxyjump`가 의도대로 나오는지 확인한 뒤 `ssh gpu1` 한 단어로 접속한다.
4. 원격의 tmux 세션에서 짧은 학습과 TensorBoard를 띄운다. 로컬에서 `ssh -N -L 16006:localhost:6006 gpu1`을 열고 브라우저로 확인한다.
5. SSH 창과 터널을 강제로 닫았다가 다시 접속해, `tmux attach`로 학습 화면에 돌아오고 터널을 다시 열어 대시보드가 이어지는지 확인한다.

config 규칙은 다음 연습문제로 익힌다. 함수가 `ssh -G`가 하는 일의 핵심을 흉내 낸다.

::exercise{id="ssh-config-lookup"}

## 실패 사례

- **비밀번호 인증과 키 공유.** 연구실 계정 하나의 비밀번호나 개인키 하나를 여러 사람이 돌려 쓰면, 누가 무엇을 했는지 추적할 수 없고 한 사람이 나가도 키를 바꿀 수 없다. 사람마다, 기기마다 키를 만들고 공개키만 등록한다.
- **개인키를 서버나 저장소에 복사하는 것.** "서버에서 GitHub에 push하려고" 개인키를 서버에 올리거나, dotfiles 저장소에 `~/.ssh` 전체를 커밋한다. 서버의 다른 관리자, 저장소를 본 모든 사람이 내 신원을 쓸 수 있게 된다. 서버에서 쓸 키가 필요하면 그 서버에서 새 키를 만들어 필요한 곳에만 등록하고, 저장소에는 `.gitignore`로 키 파일을 막는다.
- **`StrictHostKeyChecking no`로 경고를 없애는 것.** 서버 사칭을 막는 유일한 장치를 끄는 것이다.
- **`Host *`를 config 맨 위에 두는 것.** 아래의 구체적인 설정이 조용히 무시된다.
- **대시보드를 모든 인터페이스에 공개하는 것.** 포트 포워딩으로 해결할 일을 방화벽 구멍으로 해결한다.

## 통과 기준

- 키 기반 인증과 `~/.ssh/config`로 jump host 뒤의 서버에 `ssh <별칭>` 한 단어로 접속하고, `ssh -G <별칭>`으로 적용된 설정을 설명한다.
- SSH 연결을 끊었다 다시 접속한 뒤, 실행 중인 학습의 tmux 화면과 로그, 포트 포워딩으로 연 원격 대시보드에 다시 접근한다.
- 연습문제 `ssh-config-lookup`의 테스트를 통과한다.
