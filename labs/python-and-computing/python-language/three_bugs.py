"""5.1 실습: 연구 코드에서 자주 만나는 세 가지 참조 버그를 재현하고 고친다.

실행: uv run --project labs python labs/python-and-computing/python-language/three_bugs.py
각 버그마다 '버그' 버전과 '수정' 버전을 같은 입력으로 실행해 결과를 나란히 출력한다.
"""
import copy


# 1. 가변 기본 인자: 실행(run)마다 새 기록을 시작하려 했지만 기록이 누적된다
def log_metrics_buggy(step, loss, history=[]):
    history.append((step, loss))
    return history


def log_metrics_fixed(step, loss, history=None):
    if history is None:
        history = []
    history.append((step, loss))
    return history


# 2. 리스트 공유 참조: 격자 [[0]*nx]*ny는 같은 행 객체를 ny번 담는다
def make_grid_buggy(ny, nx):
    return [[0.0] * nx] * ny


def make_grid_fixed(ny, nx):
    return [[0.0] * nx for _ in range(ny)]


# 3. closure의 늦은 바인딩: epoch마다 다른 학습률을 돌려주려 했지만 마지막 값만 쓴다
def make_schedule_buggy(lrs):
    return [lambda: lr for lr in lrs]


def make_schedule_fixed(lrs):
    return [lambda lr=lr: lr for lr in lrs]


def main():
    print("1. 가변 기본 인자")
    for name, fn in [("버그", log_metrics_buggy), ("수정", log_metrics_fixed)]:
        run_a = fn(0, 1.0)
        run_b = fn(0, 0.5)          # 새 실행을 시작했다고 생각한다
        print(f"  {name}: run_b 기록 수 = {len(run_b)}, run_a is run_b = {run_a is run_b}")

    print("2. 리스트 공유 참조")
    for name, fn in [("버그", make_grid_buggy), ("수정", make_grid_fixed)]:
        grid = fn(3, 4)
        grid[0][0] = 1.0            # 경계 조건: 한 칸만 바꾼다
        changed = sum(row[0] == 1.0 for row in grid)
        print(f"  {name}: 값이 바뀐 칸 수 = {changed}, 서로 다른 행 객체 수 = {len({id(r) for r in grid})}")

    print("3. closure의 늦은 바인딩")
    for name, fn in [("버그", make_schedule_buggy), ("수정", make_schedule_fixed)]:
        schedule = fn([1e-2, 1e-3, 1e-4])
        print(f"  {name}: epoch별 학습률 = {[f() for f in schedule]}")

    print("덤: 얕은 복사로 만든 실험 설정")
    base = {"optimizer": {"lr": 1e-3}}
    for name, cp in [("얕은 복사", copy.copy), ("깊은 복사", copy.deepcopy)]:
        cfg = cp(base)
        cfg["optimizer"]["lr"] = 1e-2
        print(f"  {name}: base의 lr = {base['optimizer']['lr']}")
        base["optimizer"]["lr"] = 1e-3   # 다음 비교를 위해 되돌린다


if __name__ == "__main__":
    main()
