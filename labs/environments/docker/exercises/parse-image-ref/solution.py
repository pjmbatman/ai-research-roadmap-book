def parse(ref):
    """image 참조를 registry·repository·tag·digest로 나눈다.

    ref: 예 "ubuntu", "python:3.12-slim", "ghcr.io/astral-sh/uv:latest", "img@sha256:..."
    반환: {"registry", "repository", "tag", "digest"} dict
    """
    name, digest = (ref.split("@", 1) + [None])[:2]

    tag = None
    last = name.rsplit("/", 1)[-1]               # 마지막 '/' 뒤에서만 tag를 찾는다
    if ":" in last:
        name, tag = name.rsplit(":", 1)
    if tag is None and digest is None:
        tag = "latest"

    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = "docker.io", name
    if registry == "docker.io" and "/" not in repository:
        repository = "library/" + repository
    return {"registry": registry, "repository": repository, "tag": tag, "digest": digest}
