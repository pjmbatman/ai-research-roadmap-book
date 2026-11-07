"""6.5 실습 시작 코드: print와 조용한 except로 쓴 작은 학습 스크립트.

온도(K)로 어떤 응답 y를 맞추는 1차 모델 y ≈ w * temp_k를 SGD로 학습한다.
데이터에는 읽을 수 없는 행과 물리적으로 불가능한 행(음의 절대온도)이 섞여 있다.
이 스크립트는 둘 다 아무 말 없이 처리한다. 무엇이 일어났는지 출력만 보고 알 수 있는가?

실행: uv run --project labs python labs/research-code-engineering/logging-and-errors/print_train.py
"""

ROWS = [
    "temp_k,y",
    "293.1,1.02",
    "301.4,1.10",
    "n/a,1.05",
    "288.0,0.97",
    "-5.0,0.90",
    "310.2,1.21",
]


def load(rows):
    data = []
    for line in rows[1:]:
        try:
            t, y = line.split(",")
            data.append((float(t), float(y)))
        except Exception:
            pass
    return data


def train(data, lr=1e-6, epochs=3):
    w = 0.0
    for epoch in range(epochs):
        loss = 0.0
        for temp_k, y in data:
            err = w * temp_k - y
            loss += err**2
            w -= lr * 2 * err * temp_k
        print("epoch", epoch, "loss", loss / len(data))
    return w


if __name__ == "__main__":
    data = load(ROWS)
    w = train(data)
    print("w =", w)
