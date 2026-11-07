// seed마다 흔들리는 검증 점수 (data/widgets.toml: seed-variance).
// 1.5장 진자 주기 회귀(데이터 seed 0, 분할 seed 1, 학습 180 / 검증 60)에서 MLP의 초기화 seed만 바꿔 30번 학습한 검증 RMSE다.
// 값은 labs/research-landscape/first-pipeline/pipeline.py의 train_mlp(hidden, steps=3000, lr=0.05)를 seed 0–29로 실행해 얻었다
// (numpy 2.5.4, scipy 1.18.1). 은닉 32의 seed 0–4는 본문 표의 0.0232·0.0272·0.0307·0.0297·0.0275 s와 같다.
// 브라우저에서 MLP 60개를 다시 학습하면 몇 초가 걸려, 실행 결과를 그대로 싣고 통계(평균·표준편차·승패·겹침)만 여기서 계산한다.
import { frame, param, presets, readouts, el, C } from "../frame";

export const RMSE: Record<"h32" | "h16", number[]> = {
  h32: [.023206, .027243, .030732, .029716, .027532, .032721, .027967, .029170, .032697, .034172, .027924, .026948, .030253, .031924, .030932, .029125, .029927, .034155, .030529, .030391, .029862, .031201, .031191, .031798, .032120, .030527, .029353, .027530, .028359, .032637],
  h16: [.025292, .026900, .024602, .029122, .025693, .026102, .030793, .027081, .028707, .028180, .032928, .027245, .031644, .025256, .029793, .032556, .025180, .035345, .028455, .023277, .025975, .027699, .029012, .024569, .030409, .023420, .032304, .028233, .026168, .025544],
};
const NMAX = RMSE.h32.length;

/** seed 0..n−1의 요약: 평균, 표본 표준편차, 사분위수, 16이 이긴 seed 수, 최소–최대 범위의 겹침 비율 */
export function summarize(n: number) {
  const stat = (v: number[]) => {
    const s = [...v].sort((a, b) => a - b);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : NaN;
    const q = (p: number) => { const x = p * (s.length - 1), i = Math.floor(x); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (x - i); };
    return { mean, sd, min: s[0], max: s[s.length - 1], q1: q(0.25), med: q(0.5), q3: q(0.75) };
  };
  const a = stat(RMSE.h32.slice(0, n)), b = stat(RMSE.h16.slice(0, n));
  let wins16 = 0;
  for (let s = 0; s < n; s++) if (RMSE.h16[s] < RMSE.h32[s]) wins16++;
  const inter = Math.max(0, Math.min(a.max, b.max) - Math.max(a.min, b.min)), union = Math.max(a.max, b.max) - Math.min(a.min, b.min);
  const se = n > 1 ? Math.sqrt((a.sd ** 2 + b.sd ** 2) / n) : NaN;
  return { a, b, wins16, overlap: union > 0 ? inter / union : 0, diff: a.mean - b.mean, se };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "seed마다 흔들리는 검증 점수", "compare", narrow ? 0.95 : 0.5, "초기화 seed마다 학습한 두 MLP의 검증 RMSE 점 그림과 상자 그림, 그리고 seed별 승자");
  let n = 1;
  const X0 = 0.022, X1 = 0.036;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const L = narrow ? 62 : 82, R = 26, top = 28;
    const X = (v: number) => L + ((v - X0) / (X1 - X0)) * (W - L - R);
    const strip = narrow ? 76 : 54;                      // 아래 승자 띠의 높이
    const axisY = H - strip - 26;
    const rows = { h32: top + (axisY - top) * 0.27, h16: top + (axisY - top) * 0.75 };
    const half = Math.min(26, (axisY - top) * 0.16);
    const s = summarize(n);
    ctx.textBaseline = "alphabetic";

    // 축: 검증 RMSE [s]
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, axisY); ctx.lineTo(W - R, axisY); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let v = 0.022; v <= 0.0361; v += narrow ? 0.004 : 0.002) {
      ctx.beginPath(); ctx.moveTo(X(v), axisY); ctx.lineTo(X(v), axisY + 4); ctx.stroke();
      const t = v.toFixed(3);
      ctx.fillText(t, X(v) - ctx.measureText(t).width / 2, axisY + 16);
    }
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("검증 RMSE [s] — 분할 고정, 초기화 seed만 바꿈", L, top - 12);

    // seed 짝: 같은 seed로 학습한 두 모델을 잇는 선(가장 최근 seed는 코랄)
    for (let i = 0; i < n; i++) {
      const last = i === n - 1;
      ctx.strokeStyle = last ? C.edge : C.grid2; ctx.lineWidth = last ? 1.8 : 0.8;
      ctx.beginPath(); ctx.moveTo(X(RMSE.h32[i]), rows.h32); ctx.lineTo(X(RMSE.h16[i]), rows.h16); ctx.stroke();
    }
    const box = (key: "h32" | "h16", st: ReturnType<typeof summarize>["a"], col: string, name: string) => {
      const y = rows[key];
      ctx.fillStyle = col; ctx.fillRect(6, y - 4, 9, 9);
      ctx.fillStyle = C.ink; ctx.font = "700 13px Pretendard Variable, sans-serif";
      ctx.fillText(name, 19, y + 5);
      if (n >= 4) {
        ctx.strokeStyle = col; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(X(st.min), y); ctx.lineTo(X(st.q1), y); ctx.moveTo(X(st.q3), y); ctx.lineTo(X(st.max), y);
        ctx.moveTo(X(st.min), y - half * 0.4); ctx.lineTo(X(st.min), y + half * 0.4); ctx.moveTo(X(st.max), y - half * 0.4); ctx.lineTo(X(st.max), y + half * 0.4);
        ctx.stroke();
        ctx.strokeRect(X(st.q1), y - half, X(st.q3) - X(st.q1), 2 * half);
        ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(X(st.med), y - half); ctx.lineTo(X(st.med), y + half); ctx.stroke();
      }
      // 점: seed 하나에 하나. 같은 자리의 점이 겹치지 않게 seed 번호로 위아래로 조금 흩는다
      RMSE[key].slice(0, n).forEach((v, i) => {
        const last = i === n - 1;
        const jy = n === 1 ? 0 : (((i * 7) % 11) / 10 - 0.5) * half * 1.3;
        ctx.fillStyle = last ? C.edge : col;
        ctx.beginPath(); ctx.arc(X(v), y + jy, last ? 5 : 3.4, 0, 7); ctx.fill();
      });
      // 평균: 아래쪽 삼각형
      ctx.fillStyle = C.ink;
      const mx = X(st.mean), my = y + half + 5;
      ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(mx - 5, my + 8); ctx.lineTo(mx + 5, my + 8); ctx.closePath(); ctx.fill();
    };
    box("h32", s.a, C.ink2, "은닉 32");
    box("h16", s.b, C.mid, "은닉 16");

    // 아래: seed별 승자 띠. 칸 하나 = seed 하나, 색은 그 seed에서 검증 RMSE가 더 작은 모델
    const y0 = axisY + 38;
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("seed별 승자 (칸 하나 = seed 하나)", L, y0);
    const perRow = narrow ? 15 : NMAX;
    const cw = (W - L - R) / perRow, sq = Math.min(cw - 3, 16);
    for (let i = 0; i < NMAX; i++) {
      const cx = L + (i % perRow) * cw, cy = y0 + 8 + Math.floor(i / perRow) * (sq + 4);
      if (i < n) {
        ctx.fillStyle = RMSE.h16[i] < RMSE.h32[i] ? C.mid : C.ink2;
        ctx.fillRect(cx, cy, sq, sq);
        if (i === n - 1) { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.strokeRect(cx - 1, cy - 1, sq + 2, sq + 2); }
      } else {
        ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
        ctx.strokeRect(cx + 0.5, cy + 0.5, sq - 1, sq - 1);
      }
    }
  }

  const fmt = (v: number) => v.toFixed(4);
  function update() {
    const s = summarize(n);
    const better = s.diff > 0 ? "은닉 16" : "은닉 32";
    ro.set("a", n > 1 ? `은닉 32: ${fmt(s.a.mean)} ± ${fmt(s.a.sd)}` : `은닉 32: ${fmt(s.a.mean)}`);
    ro.set("b", n > 1 ? `은닉 16: ${fmt(s.b.mean)} ± ${fmt(s.b.sd)}` : `은닉 16: ${fmt(s.b.mean)}`);
    ro.set("wins", `16이 이긴 seed ${s.wins16}/${n}`);
    ro.set("overlap", `범위 겹침 ${Math.round(s.overlap * 100)}%`);
    // 평균 차가 표준오차의 두 배를 넘을 때만 순위를 말한다
    const clear = Math.abs(s.diff) > 2 * s.se;
    ro.set("verdict", n > 1 ? `평균 차 ${fmt(Math.abs(s.diff))} s, 표준오차 ${fmt(s.se)} → ${clear ? `${better}가 낫다` : "흔들림 안의 차이"}` : `seed 0 하나: ${better}가 낫다`, "hot");
    draw();
  }

  type P = "one" | "five" | "all";
  const PN: Record<P, number> = { one: 1, five: 5, all: NMAX };
  const pre = presets<P>([{ key: "one", label: "seed 0 하나" }, { key: "five", label: "본문의 5개" }, { key: "all", label: `${NMAX}개` }], (k) => { n = PN[k]; pn.set(n); update(); }, "one");
  pre.row.classList.add("segmented");
  const pn = param({ label: "seed 수", min: 1, max: NMAX, step: 1, value: n, fmt: (v) => String(v), aria: "평균에 넣을 초기화 seed 수(seed 0부터)", onInput: (v) => { n = v; pre.select(v === 1 ? "one" : v === 5 ? "five" : v === NMAX ? "all" : null); update(); } });
  pn.row.style.gridTemplateColumns = "4.4em minmax(0, 1fr) 2.4em";
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
