"""31.3 본문 코드 셀(루브릭 점수 집계). 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/capstone/competency-assessment/cells.py
"""
# %% cell 1 — 근거 규칙을 적용한 루브릭 집계와 처음 진단과의 비교
import numpy as np

items = ["개념 설명", "논문 연결", "유도→코드", "실패 원인 분리",
         "재현 가능성", "주장 판단", "한계 명시"]
axis = np.array(["지식", "지식", "기술", "기술", "기술", "연구 습관", "연구 습관"])
first = np.array([2, 1, 1, 1, 1, 1, 0])          # (7,) 0.4장 첫 진단
claimed = np.array([4, 3, 4, 3, 4, 3, 3])        # (7,) 지금 스스로 매긴 수준
evidenced = np.array([4, 3, 4, 1, 4, 3, 2])      # (7,) 근거 파일로 보일 수 있는 최고 수준

now = np.minimum(claimed, evidenced)             # (7,) 근거 규칙: 근거가 있는 수준까지만
for a in ["지식", "기술", "연구 습관"]:
    m = axis == a
    print(f"{a}: 평균 {now[m].mean():.2f}, 최저 {now[m].min()} (처음 최저 {first[m].min()})")
print("근거보다 높게 매긴 항목:", [items[i] for i in np.flatnonzero(claimed > evidenced)])
lowest = np.argsort(now, kind="stable")[:2]      # 같은 수준이면 표의 순서대로
print("다음 목표:", [items[i] for i in lowest])
print("전체 평균:", f"{now.mean():.2f}", "→ 평균만 보면 최저 칸이 가려진다")
