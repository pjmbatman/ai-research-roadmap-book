// POD 에너지와 외삽 (data/widgets.toml: pod-energy).
// 26.2장과 같은 문제: 점성 Burgers 해(T = 1.5), 학습 ν 16개(0.05–0.2 로그 간격), POD 모드 r개 + 계수의 log ν 3차 다항식 회귀.
// 위: 질의 ν의 solver 해·기저 투영(모드 r개로 표현할 수 있는 최선)·대리모델. 아래 왼쪽: ν에 따른 대리모델 오차와 기저 투영 오차.
// 아래 오른쪽: 모드 수 r에 따른 버린 에너지 1 − E_r. 모드를 늘려도 외삽 오차(계수 회귀 오차)는 줄지 않는다.
import { frame, param, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { NX, RMAX, NU_TRAIN, buildPOD, solve, surrogate, project, relErr, checks, type POD } from "./model";

const NUS = [0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15, 0.17, 0.2, 0.23, 0.26, 0.3, 0.35, 0.4, 0.45, 0.5];
const NU_LO = NU_TRAIN[0], NU_HI = NU_TRAIN[NU_TRAIN.length - 1];
const inside = (nu: number) => nu >= NU_LO - 1e-12 && nu <= NU_HI + 1e-12;
const pct = (v: number) => (v < 0.00005 ? "< 0.01%" : `${(100 * v).toFixed(2)}%`);

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "POD 에너지와 외삽", "manipulate", narrow ? 1.05 : 0.66, "질의 점성 ν에서 solver 해와 POD 대리모델, ν에 따른 오차 곡선, 모드 수에 따른 버린 에너지");
  let r = 4, ni = NUS.indexOf(0.02);
  const P: POD = buildPOD();
  const refs = NUS.map((nu) => solve(nu));
  let errS: number[] = [], errP: number[] = [];
  const recurve = () => { errS = NUS.map((nu, i) => relErr(surrogate(P, nu, r), refs[i])); errP = NUS.map((_, i) => relErr(project(P, refs[i], r), refs[i])); };
  recurve();

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const mono = "11px JetBrains Mono, monospace";
    ctx.font = mono;
    const nu = NUS[ni], ref = refs[ni], sur = surrogate(P, nu, r), prj = project(P, ref, r);
    // ── 위: 장 u(x)
    const PL = 30, PR = 10, t0 = 26, t1 = Math.round(H * 0.47);
    const X = (j: number) => PL + (j / NX) * (W - PL - PR);
    const Y = (v: number) => t1 - ((v + 0.7) / 2.4) * (t1 - t0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(PL, t0, W - PL - PR, t1 - t0);
    ctx.setLineDash([2, 4]); ctx.strokeStyle = C.mid;
    for (const b of [-0.5, 1.5]) { ctx.beginPath(); ctx.moveTo(PL, Y(b)); ctx.lineTo(W - PR, Y(b)); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.fillStyle = C.mid;
    for (const v of [-0.5, 0.5, 1.5]) { const s = v.toFixed(1); ctx.fillText(s, PL - 4 - ctx.measureText(s).width, Y(v) + 4); }
    const line = (u: Float64Array, col: string, w: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); for (let j = 0; j <= NX; j++) { const v = u[j % NX]; j ? ctx.lineTo(X(j), Y(v)) : ctx.moveTo(X(j), Y(v)); } ctx.stroke();
      ctx.setLineDash([]);
    };
    line(prj, C.ink2, 1.4);
    line(ref, C.ink, 1.6, [5, 4]);
    line(sur, C.edge, 2.4);
    // 범례
    let lx = PL;
    for (const [nm, col, dash] of [["solver", C.ink, true], ["기저 투영", C.ink2, false], ["대리모델", C.edge, false]] as [string, string, boolean][]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, 12); ctx.lineTo(lx + 14, 12); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, 16);
      lx += 30 + ctx.measureText(nm).width;
    }
    // ── 아래 왼쪽: ν에 따른 상대 오차(로그–로그)
    const b0 = t1 + 30, b1 = H - 22, split = Math.round(W * 0.6);
    const lnA = Math.log(NUS[0]), lnB = Math.log(NUS[NUS.length - 1]);
    const EX = (v: number) => PL + ((Math.log(v) - lnA) / (lnB - lnA)) * (split - PL - 4);
    const EY = (e: number) => b1 - ((Math.log10(Math.max(e, 1e-6)) + 6) / 6) * (b1 - b0);
    ctx.fillStyle = C.fill; ctx.fillRect(EX(NU_LO), b0, EX(NU_HI) - EX(NU_LO), b1 - b0);
    ctx.strokeStyle = C.faint; ctx.strokeRect(PL, b0, split - PL - 4, b1 - b0);
    ctx.fillStyle = C.mid;
    for (const [e, s] of [[1e-4, "0.01%"], [1e-2, "1%"], [1, "100%"]] as [number, string][]) {
      ctx.strokeStyle = alpha(C.mid, 0.18); ctx.beginPath(); ctx.moveTo(PL, EY(e)); ctx.lineTo(split - 4, EY(e)); ctx.stroke();
      ctx.fillText(s, PL + 3, EY(e) + 11);
    }
    for (const v of [0.01, 0.05, 0.2, 0.5]) { const s = String(v); ctx.fillText(s, Math.min(Math.max(EX(v) - ctx.measureText(s).width / 2, PL), split - 4 - ctx.measureText(s).width), b1 + 14); }
    ctx.fillText("오차", PL, b0 - 7);
    const s = "ν"; ctx.fillText(s, split - 4 - ctx.measureText(s).width, b0 - 7);
    const curve = (es: number[], col: string, w: number) => {
      ctx.strokeStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); es.forEach((e, i) => (i ? ctx.lineTo(EX(NUS[i]), EY(e)) : ctx.moveTo(EX(NUS[i]), EY(e)))); ctx.stroke();
    };
    curve(errP, C.ink2, 1.4);
    curve(errS, C.edge, 2.2);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(EX(nu), b0); ctx.lineTo(EX(nu), b1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(EX(nu), EY(errP[ni]), 3.5, 0, 7); ctx.fill();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(EX(nu), EY(errS[ni]), 4.5, 0, 7); ctx.fill();
    // ── 아래 오른쪽: 버린 에너지 1 − E_r (로그)
    const ex0 = split + 30, ex1 = W - PR, bw = (ex1 - ex0) / RMAX;
    const DY = (d: number) => b0 + (-Math.log10(Math.max(d, 1e-12)) / 12) * (b1 - b0);
    ctx.strokeStyle = C.faint; ctx.strokeRect(ex0, b0, ex1 - ex0, b1 - b0);
    ctx.fillStyle = C.mid;
    for (const [d, s] of [[1, "1"], [1e-6, "1e-6"], [1e-12, "1e-12"]] as [number, string][]) ctx.fillText(s, ex0 - 3 - ctx.measureText(s).width, Math.min(Math.max(DY(d) + 4, b0 + 9), b1));
    ctx.fillText("버린 에너지", ex0, b0 - 7);
    P.discard.forEach((d, k) => {
      const x = ex0 + k * bw + bw * 0.18, y = DY(d);
      ctx.fillStyle = k + 1 === r ? C.edge : k + 1 < r ? C.edgeSoft : alpha(C.mid, 0.25);
      ctx.fillRect(x, y, bw * 0.64, b1 - y);
      ctx.fillStyle = C.mid;
      if (k % 2 === 0 || bw > 18) { const t = String(k + 1); ctx.fillText(t, x + bw * 0.32 - ctx.measureText(t).width / 2, b1 + 14); }
    });
  }

  let raf = 0;
  function update(curves = false) {
    if (curves) recurve();
    const nu = NUS[ni];
    const sur = surrogate(P, nu, r);
    const eS = errS[ni], eP = errP[ni];
    const ch = checks(sur);
    const isIn = inside(nu);
    ro.set("energy", `모드 ${r}개 · 누적 에너지 ${(100 * P.cum[r - 1]).toFixed(4)}%`);
    ro.set("err", `ν ${nu.toFixed(3)} ${isIn ? "내삽" : "외삽"} · 대리모델 오차 ${pct(eS)} · 투영 ${pct(eP)}`, isIn ? "ok" : "hot");
    const bad = [!ch.bounds ? "최대·최소 원리" : "", ch.tv > 4 ? `총변동 ${ch.tv.toFixed(2)} > 4` : ""].filter(Boolean);
    ro.set("check", bad.length ? `물리 검사 위반: ${bad.join(" · ")}` : "물리 검사 통과");
    pageState.set("pod-energy", { r, nu, surrogateErr: eS, basisErr: eP }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pr = param({ label: "r", min: 1, max: RMAX, step: 1, value: r, fmt: (v) => `${v}`, aria: "POD 모드 수 r", onInput: (v) => { r = v; update(true); } });
  const pn = param({ label: "ν", min: 0, max: NUS.length - 1, step: 1, value: ni, fmt: (v) => NUS[v].toFixed(3), aria: "질의 점성 ν (학습 범위 0.05–0.2)", onInput: (v) => { ni = v; update(); } });
  pn.input.setAttribute("aria-valuetext", `ν ${NUS[ni]}`);
  pn.input.addEventListener("input", () => pn.input.setAttribute("aria-valuetext", `ν ${NUS[ni]}`));
  const grid = el("div", "param-grid");
  grid.append(pr.row, pn.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
