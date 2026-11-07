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
    raise NotImplementedError("run ID를 만드세요")
