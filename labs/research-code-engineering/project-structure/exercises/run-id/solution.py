import copy
import hashlib
import json
from datetime import datetime


def make_run_id(config: dict, started: datetime, ignore=()) -> str:
    """'YYYYmmdd-HHMMSS-<설정 hash 8자리>' 형식의 run ID를 만든다.

    config: 중첩 dict 설정. 바꾸지 않는다.
    started: 실행 시작 시각
    ignore: hash에서 뺄 점 경로들, 예 ("paths.output_dir", "logging")
    """
    cfg = copy.deepcopy(config)
    for dotted in ignore:
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(leaf, None)
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]
    return f"{started:%Y%m%d-%H%M%S}-{digest}"
