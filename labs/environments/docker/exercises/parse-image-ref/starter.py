def parse(ref):
    """image 참조를 registry·repository·tag·digest로 나눈다.

    ref: 예 "ubuntu", "python:3.12-slim", "ghcr.io/astral-sh/uv:latest", "img@sha256:..."
    반환: {"registry", "repository", "tag", "digest"} dict
    """
    raise NotImplementedError("digest, tag, registry 순서로 떼어 내세요")
