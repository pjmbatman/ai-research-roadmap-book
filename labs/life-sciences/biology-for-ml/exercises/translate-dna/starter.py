BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # 표준 유전 부호


def translate(dna, frame=0, strand="+"):
    """dna를 strand("+" 그대로, "-" 역상보)의 frame(0, 1, 2)부터 번역한다.

    정지 코돈에서 멈추고(정지 코돈은 넣지 않음), 끝에 남는 염기는 버린다.
    반환: 아미노산 한 글자 기호 문자열
    """
    raise NotImplementedError("역상보·읽기 틀·코돈 번역을 구현하세요")
