"""2.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linux-and-remote/execution-topology/cells.py
"""
# %% cell 1 — 데이터를 옮기는 데 걸리는 최소 시간
size_bytes = {"code 50 MB": 50e6, "results 2 GB": 2e9, "dataset 2 TB": 2e12}
link_gbps = {"100 Mbps": 0.1, "1 Gbps": 1.0, "10 Gbps": 10.0}

def transfer_s(n_bytes, gbps):
    return n_bytes * 8 / (gbps * 1e9)            # 대역폭을 100% 쓴다고 가정한 하한

def human(t_s):
    if t_s < 1:
        return f"{t_s * 1000:.0f} ms"
    return f"{t_s:.1f} s" if t_s < 60 else f"{t_s / 60:.1f} min" if t_s < 3600 else f"{t_s / 3600:.1f} h"

print(f"{'':10s}" + "".join(f"{name:>15s}" for name in size_bytes))
for link, gbps in link_gbps.items():
    print(f"{link:10s}" + "".join(f"{human(transfer_s(b, gbps)):>15s}" for b in size_bytes.values()))

# %% cell 2 — 도구마다 다섯 질문의 답을 표로
import pandas as pd

tools = pd.DataFrame([
    # 도구,                코드,        연산,        데이터,      자격,            세션 유지
    ("노트북 Jupyter",      "laptop",   "laptop",   "laptop",   "나(로컬 계정)",   "laptop 커널"),
    ("VS Code Remote-SSH", "gpu1",     "gpu1",     "gpu1",     "내 SSH 키",      "SSH 연결"),
    ("gpu1의 tmux 학습",    "gpu1",     "gpu1",     "nas",      "내 SSH 키",      "gpu1의 tmux"),
    ("관리형 notebook",     "cloud",    "cloud",    "laptop",   "클라우드 계정",   "브라우저 탭"),
], columns=["도구", "코드", "연산", "데이터", "자격", "세션"])

tools["데이터 이동"] = tools["데이터"] != tools["연산"]          # 연산하는 곳으로 데이터를 옮겨야 하는가
tools["창 닫으면 멈춤"] = tools["세션"].isin(["SSH 연결", "브라우저 탭", "laptop 커널"])
print(tools[["도구", "연산", "데이터", "데이터 이동", "창 닫으면 멈춤"]].to_string(index=False))
