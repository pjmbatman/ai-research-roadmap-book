import hashlib


def dataset_manifest(files):
    """파일 내용으로 manifest 텍스트와 데이터셋 해시를 만든다.

    files: {경로(str): 내용(bytes)}
    반환: (manifest, digest)
      manifest: 경로 순으로 정렬한 "<sha256 hex>  <byte 수>  <경로>\\n" 줄들을 이어 붙인 문자열
      digest: manifest를 UTF-8로 인코딩한 바이트의 sha256 hex
    """
    raise NotImplementedError("경로 순으로 줄을 만들고, manifest 전체를 다시 해시하세요")
