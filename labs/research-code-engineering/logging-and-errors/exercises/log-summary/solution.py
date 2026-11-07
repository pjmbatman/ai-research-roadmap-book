import json
import math


def summarize(lines):
    """JSON Lines 로그를 요약한다.

    lines: 로그 파일의 줄(str) 리스트
    반환: {"levels": dict, "first_bad_step": int | None, "config": dict | None, "malformed": int}
    """
    levels, first_bad_step, config, malformed = {}, None, None, 0
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(rec, dict):
            malformed += 1
            continue
        if "level" in rec:
            levels[rec["level"]] = levels.get(rec["level"], 0) + 1
        if config is None and rec.get("event") == "config":
            config = {k: v for k, v in rec.items() if k not in ("level", "event")}
        if first_bad_step is None and "loss" in rec:
            loss = rec["loss"]
            if loss is None or not math.isfinite(loss):
                first_bad_step = rec.get("step")
    return {"levels": levels, "first_bad_step": first_bad_step, "config": config, "malformed": malformed}
