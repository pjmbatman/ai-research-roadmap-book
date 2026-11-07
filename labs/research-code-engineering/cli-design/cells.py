"""6.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/cli-design/cells.py
argparse의 오류 메시지 문구는 Python 버전에 따라 조금 다를 수 있다(집필 시점: Python 3.13).
"""
# %% cell 1 — argparse 기초: 위치 인자, 옵션, 타입, 선택지
import argparse

class Parser(argparse.ArgumentParser):
    def error(self, message):            # 기본 동작(메시지 출력 후 종료)을 예외로 바꿔 셀에서 확인한다
        raise ValueError(message)

p = Parser(prog="train", description="모델을 학습한다.")
p.add_argument("config", help="설정 파일 경로")                    # 위치 인자
p.add_argument("--lr", type=float, help="학습률")
p.add_argument("--epochs", type=int, default=50, help="epoch 수")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--dry-run", action="store_true", help="설정만 출력하고 끝낸다")

print(p.parse_args(["configs/base.toml", "--lr", "3e-4", "--dry-run"]))
for argv in [["configs/base.toml", "--lr", "fast"], ["configs/base.toml", "--device", "tpu"], []]:
    try:
        p.parse_args(argv)
    except ValueError as e:
        print("오류:", str(e).split(" (choose from")[0])   # 허용 목록의 표기는 Python 패치 버전마다 달라 앞부분만 찍는다

# %% cell 2 — 조용히 무시되거나 엉뚱하게 해석되는 인자
loose = argparse.ArgumentParser()
loose.add_argument("--epochs", type=int, default=50)
loose.add_argument("--lr-schedule", default="cosine")

argv = ["--epoch", "5", "--lr", "3e-4", "--learning-rate", "3e-4"]
args, unknown = loose.parse_known_args(argv)
print("해석:", vars(args), " 버려짐:", unknown)

strict = Parser(allow_abbrev=False)
strict.add_argument("--epochs", type=int, default=50)
strict.add_argument("--lr-schedule", default="cosine")
try:
    strict.parse_args(argv)
except ValueError as e:
    print("오류:", e)

# %% cell 3 — subcommand와 공유 옵션
def cmd_train(args):
    return f"train: config={args.config} seed={args.seed}"

def cmd_eval(args):
    return f"eval: ckpt={args.ckpt} split={args.split} seed={args.seed}"

common = argparse.ArgumentParser(add_help=False)       # 여러 subcommand가 공유하는 옵션
common.add_argument("--seed", type=int, default=0)

cli = Parser(prog="soh")
sub = cli.add_subparsers(dest="command", required=True)
t = sub.add_parser("train", parents=[common], help="모델을 학습한다")
t.add_argument("config")
t.set_defaults(func=cmd_train)
e = sub.add_parser("eval", parents=[common], help="checkpoint를 평가한다")
e.add_argument("ckpt")
e.add_argument("--split", choices=["val", "test"], default="test")
e.set_defaults(func=cmd_eval)

for argv in [["train", "configs/base.toml", "--seed", "1"], ["eval", "outputs/run1/model.npz"], ["predict"]]:
    try:
        args = cli.parse_args(argv)
        print(args.func(args))
    except ValueError as err:
        print("오류:", str(err).split(" (choose from")[0])

# %% cell 4 — 우선순위: 기본값 < 설정 파일 < 명령행
defaults = {"lr": 1e-3, "epochs": 50, "seed": 0}
file_cfg = {"lr": 3e-4, "epochs": 100}                 # 설정 파일에서 읽었다고 하자

bad = argparse.ArgumentParser()
bad.add_argument("--lr", type=float, default=1e-3)     # 기본값을 parser에도 적었다
bad.add_argument("--epochs", type=int, default=50)
good = argparse.ArgumentParser()
good.add_argument("--lr", type=float, default=None)    # 주지 않은 인자는 None
good.add_argument("--epochs", type=int, default=None)

argv = ["--epochs", "20"]
for name, parser in [("bad ", bad), ("good", good)]:
    cli_args = {k: v for k, v in vars(parser.parse_args(argv)).items() if v is not None}
    print(name, {**defaults, **file_cfg, **cli_args})
