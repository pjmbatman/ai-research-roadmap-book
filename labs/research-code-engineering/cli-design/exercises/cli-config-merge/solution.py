import copy


def _leaves(d: dict, prefix: str = "") -> list[str]:
    out = []
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        out += _leaves(value, path) if isinstance(value, dict) else [path]
    return out


def _apply_file(cfg: dict, override: dict, sources: dict, prefix: str = "") -> None:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in cfg:
            raise KeyError(path)
        if isinstance(cfg[key], dict) != isinstance(value, dict):
            raise TypeError(f"{path}: 섹션과 값을 서로 덮어쓸 수 없습니다")
        if isinstance(value, dict):
            _apply_file(cfg[key], value, sources, path)
        else:
            cfg[key] = copy.deepcopy(value)
            sources[path] = "file"


def resolve_config(defaults: dict, file_cfg: dict, cli_args: dict) -> tuple[dict, dict]:
    """우선순위 defaults < file_cfg < cli_args로 최종 설정을 만들고, 말단 키마다 출처를 기록한다.

    defaults, file_cfg: 중첩 dict
    cli_args: vars(argparse.Namespace). 점 없는 키와 None 값은 무시한다.
    반환: (config, sources). sources[점 경로] ∈ {"default", "file", "cli"}
    """
    cfg = copy.deepcopy(defaults)
    sources = dict.fromkeys(_leaves(defaults), "default")
    _apply_file(cfg, file_cfg, sources)
    for dotted, value in cli_args.items():
        if "." not in dotted or value is None:
            continue
        if dotted not in sources:
            raise KeyError(dotted)
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        node[leaf] = copy.deepcopy(value)
        sources[dotted] = "cli"
    return cfg, sources
