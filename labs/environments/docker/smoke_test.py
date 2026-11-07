"""3.5 실습: 로컬 가상환경과 컨테이너에서 같은 결과가 나오는지 확인하는 smoke test.

실행(로컬): uv run --project labs python labs/environments/docker/smoke_test.py
실행(컨테이너): README.md의 docker run 명령

- 고정된 시드로 작은 선형 시스템을 풀고, 결과를 expected.json의 값과 허용 오차 안에서 비교한다.
  (BLAS 구현·CPU에 따라 마지막 자리가 달라질 수 있으므로 해시가 아니라 허용 오차로 비교한다)
- 결과를 OUTPUT_DIR(기본값 ./outputs)에 JSON으로 쓴다. 컨테이너에서는 이 디렉터리를 호스트에 bind mount한다.
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def compute():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(256, 256))                 # (N, N)
    b = np.ones(256)                                # (N,)
    x = np.linalg.solve(A, b)                       # (N,)
    return {"x_head": [float(v) for v in x[:3]], "x_norm": float(np.linalg.norm(x)),
            "residual": float(np.linalg.norm(A @ x - b))}


def gpu_info():
    try:
        import torch
    except ImportError:
        return "torch 없음"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, GPU 사용 불가"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


if __name__ == "__main__":
    result = compute()
    expected = json.loads((HERE / "expected.json").read_text())
    ok = (np.allclose(result["x_head"], expected["x_head"], rtol=1e-9, atol=1e-12)
          and abs(result["x_norm"] - expected["x_norm"]) < 1e-9 * expected["x_norm"]
          and result["residual"] < 1e-9)
    print(f"python {platform.python_version()} / numpy {np.__version__} / {platform.machine()}")
    print(f"GPU: {gpu_info()}")
    print(f"x[:3] = {np.round(result['x_head'], 6)}, |x| = {result['x_norm']:.6f}, 잔차 = {result['residual']:.1e}")
    out_dir = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(f"결과 기록: {out_dir / 'smoke_result.json'}")
    print("SMOKE TEST", "통과" if ok else "실패")
    sys.exit(0 if ok else 1)
