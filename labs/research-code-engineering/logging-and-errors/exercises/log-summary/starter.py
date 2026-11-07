import json
import math


def summarize(lines):
    """JSON Lines 로그를 요약한다.

    lines: 로그 파일의 줄(str) 리스트
    반환: {"levels": dict, "first_bad_step": int | None, "config": dict | None, "malformed": int}
    """
    raise NotImplementedError("한 줄씩 json.loads로 읽고, 실패한 줄은 malformed로 세세요")
