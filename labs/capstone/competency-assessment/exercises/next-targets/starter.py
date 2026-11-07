def next_targets(rows, k=2):
    """역량 루브릭에서 다음 성장 목표로 삼을 항목 k개를 고른다.

    rows: dict의 list. 각 dict는 다음 키를 가진다.
        "item"      : 항목 이름 (str)
        "claimed"   : 스스로 매긴 수준 (0–5의 int)
        "evidenced" : 근거로 보일 수 있는 최고 수준 (0–5의 int)
        "needed"    : 자기 연구에 필요한 항목인가 (bool)
    k: 고를 항목 수

    반환: 항목 이름의 list. 연구에 필요한 항목 중 min(claimed, evidenced)가 낮은 순서로
          최대 k개. 수준이 같으면 rows의 순서를 따른다.
    수준이 0–5의 정수가 아니면 ValueError를 낸다.
    """
    raise NotImplementedError("근거 규칙을 적용한 수준으로 필요한 항목을 정렬하세요")
