BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # 표준 유전 부호
CODON = {a + b + c: AA[16 * i + 4 * j + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}


def translate(dna, frame=0, strand="+"):
    """dna를 strand("+" 그대로, "-" 역상보)의 frame(0, 1, 2)부터 번역한다.

    정지 코돈에서 멈추고(정지 코돈은 넣지 않음), 끝에 남는 염기는 버린다.
    반환: 아미노산 한 글자 기호 문자열
    """
    seq = dna.upper()
    if strand == "-":
        seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    protein = []
    for s in range(frame, len(seq) - 2, 3):
        aa = CODON[seq[s:s + 3]]
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)
