import math

import numpy as np


def summarize(runs, group_by, metric):
    """run 기록 리스트를 group_by 설정 키로 묶어 metric을 요약한다.

    runs: [{"run_id", "config", "status", "metrics"}, ...]
    group_by: 설정 키 리스트, 예 ["lr"]
    반환: {(값, ...): {"mean", "std", "n", "missing", "run_ids"}}
    """
    raise NotImplementedError("그룹마다 끝난 run만 모아 평균·표준편차·개수·빠진 수·run ID를 계산하세요")
