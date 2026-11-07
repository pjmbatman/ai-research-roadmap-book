"""24.7 실습: 시뮬레이터 파라미터를 베이지안으로 추정하고 사후 불확실성을 예측으로 전파한다.

Michaelis–Menten 모델 v = vmax S / (km + S)의 파라미터를 측정 10개로 추정하고, 측정 범위 밖 S = 20 mM의 속도를 예측한다.
같은 실험을 다른 잡음으로 40번 반복해, 두 가지 보고 방식의 90% 구간이 참값을 포함하는 비율을 잰다.

- 점추정: 사후 표본 중 로그 사후밀도가 가장 높은 파라미터(MAP 근사)로 예측하고, 구간은 측정 잡음만으로 만든다.
- 사후 예측: 사후 표본마다 예측하고 측정 잡음을 더해 분위수로 구간을 만든다.

실행: uv run --project labs python labs/scientific-ml/inverse-problems-and-uq/posterior_predictive.py
"""
import numpy as np

NOISE, V_TRUE, K_TRUE, S_PRED = 0.005, 2.0, 5.0, 20.0


def rate(s_mm, vmax, km):
    return vmax * s_mm / (km + s_mm)


def mcmc(s_mm, v_obs, n_chain=16, n_step=3000, seed=0):
    """본문 cell 4와 같은 random-walk Metropolis. 표본과 로그 사후밀도를 함께 돌려준다."""
    r = np.random.default_rng(seed)

    def logpost(th):
        pred = rate(s_mm[None, :], np.exp(th[:, :1]), np.exp(th[:, 1:]))
        return -0.5 * np.sum((v_obs - pred) ** 2, axis=1) / NOISE**2 - 0.5 * np.sum(th**2, axis=1) / 3**2

    th = r.normal(0, 1, size=(n_chain, 2))
    lp = logpost(th)
    out, out_lp = [], []
    for i in range(n_step):
        prop = th + 0.1 * r.normal(size=th.shape)
        lp_prop = logpost(prop)
        acc = np.log(r.uniform(size=n_chain)) < lp_prop - lp
        th[acc], lp[acc] = prop[acc], lp_prop[acc]
        if i >= 1000:
            out.append(th.copy())
            out_lp.append(lp.copy())
    return np.exp(np.concatenate(out)), np.concatenate(out_lp)


def coverage(s_mm, n_rep=40):
    v_star = rate(S_PRED, V_TRUE, K_TRUE)
    hits_point, hits_post, widths = [], [], []
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        v_obs = rate(s_mm, V_TRUE, K_TRUE) + NOISE * rng.normal(size=s_mm.size)
        post, lp = mcmc(s_mm, v_obs, seed=rep)
        y_new = v_star + NOISE * rng.normal()                       # 새로 측정할 값
        vmax_hat, km_hat = post[np.argmax(lp)]
        pred_point = rate(S_PRED, vmax_hat, km_hat)
        hits_point.append(abs(y_new - pred_point) < 1.645 * NOISE)
        draws = rate(S_PRED, post[:, 0], post[:, 1]) + NOISE * rng.normal(size=len(post))
        lo, hi = np.percentile(draws, [5, 95])
        hits_post.append(lo < y_new < hi)
        widths.append(hi - lo)
    return np.mean(hits_point), np.mean(hits_post), np.mean(widths)


if __name__ == "__main__":
    print(f"S = {S_PRED} mM 예측의 90% 구간 포함률 (실험 40번 반복, 명목 0.90)")
    for name, s_mm in [("저농도만 0.1–0.5 mM", np.linspace(0.1, 0.5, 10)), ("넓게 0.1–20 mM", np.geomspace(0.1, 20, 10))]:
        cp, cq, wq = coverage(s_mm)
        print(f"  {name:<16}  점추정+잡음 {cp:.2f}   사후 예측 {cq:.2f} (평균 폭 {wq:.3f})")
