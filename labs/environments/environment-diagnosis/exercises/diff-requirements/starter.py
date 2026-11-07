def diff_requirements(before, after):
    """freeze 형식 문자열 두 개를 비교한다.

    before, after: "numpy==2.1.3\\npandas==2.2.3\\n..." 형식의 문자열
    반환: {"added": {이름: 버전}, "removed": {이름: 버전}, "changed": {이름: (옛, 새)}}
    """
    raise NotImplementedError("두 목록을 파싱해 비교하세요")
