import copy


def resolve_config(defaults: dict, file_cfg: dict, cli_args: dict) -> tuple[dict, dict]:
    """우선순위 defaults < file_cfg < cli_args로 최종 설정을 만들고, 말단 키마다 출처를 기록한다.

    defaults, file_cfg: 중첩 dict
    cli_args: vars(argparse.Namespace). 점 없는 키와 None 값은 무시한다.
    반환: (config, sources). sources[점 경로] ∈ {"default", "file", "cli"}
    """
    raise NotImplementedError("설정을 병합하세요")
