# 기본값·설정 파일·명령행을 출처와 함께 병합하기

CLI는 세 층의 값을 하나의 설정으로 합친다. 우선순위는 **기본값 < 설정 파일 < 명령행**이다. 명령행 옵션의 `dest`를 설정 키와 같은 점 경로로 두면(`add_argument("--lr", dest="train.lr", default=None)`), `vars(args)`가 곧 점 경로 → 값의 dict가 된다.

```python
vars(args)
# {'command': 'train', 'config': 'exp.toml', 'train.lr': None, 'train.seed': 3}
```

함수 `resolve_config(defaults, file_cfg, cli_args)`를 완성해 `(config, sources)`를 반환한다.

- `defaults`, `file_cfg`: 중첩 dict. `file_cfg`의 값은 `defaults`의 같은 위치 값을 대체한다. 리스트도 통째로 대체한다.
- `cli_args`: `vars(args)` 같은 평평한 dict.
  - 키에 점이 없으면(`command`, `config` 등) 설정이 아니므로 무시한다.
  - 값이 `None`이면 "명령행에서 주지 않음"이므로 무시한다. 설정 파일의 값을 `None`으로 덮어쓰면 안 된다.
  - 점 경로의 값은 설정 파일보다 우선한다.
- `defaults`에 없는 키가 `file_cfg`나 `cli_args`(점 있는 키)에 있으면 `KeyError`를 내고, 메시지에 점 경로를 넣는다.
- `file_cfg`에서 섹션(dict) 자리에 값이 오거나 값 자리에 섹션이 오면 `TypeError`를 낸다.
- `sources`는 `defaults`의 **모든 말단 키**(점 경로)에 대해 그 값이 어디서 왔는지 `"default"`, `"file"`, `"cli"` 중 하나로 기록한 dict다. 결과를 저장할 때 함께 남기면, 나중에 어떤 값이 의도적으로 바뀐 것인지 알 수 있다.
- 입력을 바꾸지 않는다.

```python
defaults = {"model": {"width": 64}, "train": {"lr": 1e-3, "seed": 0}}
file_cfg = {"train": {"lr": 3e-4}}
cli_args = {"command": "train", "train.lr": None, "train.seed": 3}
resolve_config(defaults, file_cfg, cli_args)
# ({'model': {'width': 64}, 'train': {'lr': 0.0003, 'seed': 3}},
#  {'model.width': 'default', 'train.lr': 'file', 'train.seed': 'cli'})
```
