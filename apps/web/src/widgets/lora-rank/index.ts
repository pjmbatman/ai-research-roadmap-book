// LoRA 랭크 (data/widgets.toml: lora-rank).
// 19.9장의 예: 고정 가중치 W₀(64 × 64) 위에 과제의 변화 ΔW_true = (rank 2 큰 성분) + 0.02·(full-rank 잡음 성분),
// 입력 x ~ N(0, I), 출력 잡음 분산 0.25. 학습 표본 n개로 잔차 Y − XW₀ᵀ에 ΔW를 맞춘다.
// full fine-tuning = 최소제곱 C = (XᵀX)⁻¹XᵀY', rank r LoRA = reduced-rank 회귀(XC의 위 r개 오른쪽 특이벡터로 투영).
// 시험 MSE는 x ~ N(0, I)에서의 기댓값을 닫힌 식으로 쓴다: 0.25 + ‖ΔW − ΔW_true‖²_F / 64.
// 아래 그림은 적합값 XC/√n의 특이값(LoRA가 위 r개를 남긴다)과 ΔW_true의 특이값이다.
import { frame, param, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

export const D = 64, NMAX = 2000, NOISE_VAR = 0.25, RMAX = 16;
export const NS = [70, 80, 100, 150, 200, 300, 500, 1000, 2000];

/** 고정 시드 자료: ΔW_trueᵀ (d_in × d_out), 표본 풀 X (NMAX × D), 잔차 목표 Y' = XΔW_trueᵀ + ε */
export function makeTask(seed = 2022) {
  const r = rng(seed);
  const G = Array.from({ length: D * 2 }, () => r.normal()), Hm = Array.from({ length: 2 * D }, () => r.normal());
  const dWt = new Float64Array(D * D); // dWt[i * D + j] = ΔW_true[j][i] (i: 입력, j: 출력)
  for (let j = 0; j < D; j++) for (let i = 0; i < D; i++) {
    dWt[i * D + j] = (G[j * 2] * Hm[i] + G[j * 2 + 1] * Hm[D + i]) / Math.sqrt(D) + 0.02 * r.normal();
  }
  const X = new Float64Array(NMAX * D), Y = new Float64Array(NMAX * D);
  for (let k = 0; k < NMAX * D; k++) X[k] = r.normal();
  for (let n = 0; n < NMAX; n++) for (let j = 0; j < D; j++) {
    let s = 0; for (let i = 0; i < D; i++) s += X[n * D + i] * dWt[i * D + j];
    Y[n * D + j] = s + Math.sqrt(NOISE_VAR) * r.normal();
  }
  return { dWt, X, Y };
}

/** 대칭 행렬의 고유분해(순환 Jacobi). 고윳값 내림차순과 열 고유벡터 V(V[i * m + k]) */
export function eigSym(Ain: Float64Array, m: number) {
  const A = Float64Array.from(Ain), V = new Float64Array(m * m);
  for (let i = 0; i < m; i++) V[i * m + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++) off += A[p * m + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++) {
      const apq = A[p * m + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * m + q] - A[p * m + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < m; k++) {
        const akp = A[k * m + p], akq = A[k * m + q];
        A[k * m + p] = c * akp - s * akq; A[k * m + q] = s * akp + c * akq;
      }
      for (let k = 0; k < m; k++) {
        const apk = A[p * m + k], aqk = A[q * m + k];
        A[p * m + k] = c * apk - s * aqk; A[q * m + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < m; k++) {
        const vkp = V[k * m + p], vkq = V[k * m + q];
        V[k * m + p] = c * vkp - s * vkq; V[k * m + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: m }, (_, i) => i).sort((a, b) => A[b * m + b] - A[a * m + a]);
  const vals = order.map((i) => A[i * m + i]);
  const vecs = new Float64Array(m * m);
  order.forEach((src, k) => { for (let i = 0; i < m; i++) vecs[i * m + k] = V[i * m + src]; });
  return { vals, vecs };
}

/** n개 표본으로 rank 1..RMAX와 full의 시험 MSE, 적합값의 특이값 */
export function fit(task: ReturnType<typeof makeTask>, n: number) {
  const { X, Y, dWt } = task;
  const G = new Float64Array(D * D), B = new Float64Array(D * D);
  for (let s = 0; s < n; s++) {
    const xo = s * D;
    for (let i = 0; i < D; i++) {
      const xi = X[xo + i];
      for (let j = i; j < D; j++) G[i * D + j] += xi * X[xo + j];
      for (let j = 0; j < D; j++) B[i * D + j] += xi * Y[xo + j];
    }
  }
  for (let i = 0; i < D; i++) for (let j = 0; j < i; j++) G[i * D + j] = G[j * D + i];
  // Cholesky G = LLᵀ, C = G⁻¹B (d_in × d_out)
  const L = new Float64Array(D * D);
  for (let i = 0; i < D; i++) for (let j = 0; j <= i; j++) {
    let s = G[i * D + j];
    for (let k = 0; k < j; k++) s -= L[i * D + k] * L[j * D + k];
    L[i * D + j] = i === j ? Math.sqrt(s) : s / L[j * D + j];
  }
  const Cm = new Float64Array(D * D);
  for (let col = 0; col < D; col++) {
    const z = new Float64Array(D);
    for (let i = 0; i < D; i++) { let s = B[i * D + col]; for (let k = 0; k < i; k++) s -= L[i * D + k] * z[k]; z[i] = s / L[i * D + i]; }
    for (let i = D - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < D; k++) s -= L[k * D + i] * Cm[k * D + col]; Cm[i * D + col] = s / L[i * D + i]; }
  }
  // M = Cᵀ G C = (XC)ᵀ(XC): 그 고유벡터가 XC의 오른쪽 특이벡터
  const GC = new Float64Array(D * D);
  for (let i = 0; i < D; i++) for (let k = 0; k < D; k++) { const g = G[i * D + k]; if (g) for (let j = 0; j < D; j++) GC[i * D + j] += g * Cm[k * D + j]; }
  const M = new Float64Array(D * D);
  for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { let s = 0; for (let i = 0; i < D; i++) s += Cm[i * D + a] * GC[i * D + b]; M[a * D + b] = M[b * D + a] = s; }
  const { vals, vecs } = eigSym(M, D);
  const mse = (dW: Float64Array) => { let s = 0; for (let k = 0; k < D * D; k++) s += (dW[k] - dWt[k]) ** 2; return NOISE_VAR + s / D; };
  // rank r: C V_r V_rᵀ를 한 열씩 더해 간다
  const CV = new Float64Array(D * D); // CV[i * D + k] = (C v_k)_i
  for (let i = 0; i < D; i++) for (let k = 0; k < RMAX; k++) { let s = 0; for (let j = 0; j < D; j++) s += Cm[i * D + j] * vecs[j * D + k]; CV[i * D + k] = s; }
  const acc = new Float64Array(D * D), byRank: number[] = [];
  for (let k = 0; k < RMAX; k++) {
    for (let i = 0; i < D; i++) { const c = CV[i * D + k]; for (let j = 0; j < D; j++) acc[i * D + j] += c * vecs[j * D + k]; }
    byRank.push(mse(acc));
  }
  return { byRank, full: mse(Cm), zero: mse(new Float64Array(D * D)), sv: vals.map((v) => Math.sqrt(Math.max(v, 0) / n)) };
}

/** ΔW_true의 특이값 */
export function trueSv(task: ReturnType<typeof makeTask>) {
  const { dWt } = task, M = new Float64Array(D * D);
  for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { let s = 0; for (let i = 0; i < D; i++) s += dWt[i * D + a] * dWt[i * D + b]; M[a * D + b] = M[b * D + a] = s; }
  return eigSym(M, D).vals.map((v) => Math.sqrt(Math.max(v, 0)));
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "LoRA 랭크", "manipulate", narrow ? 0.9 : 0.58, "rank에 따른 시험 MSE 곡선과, 적합값의 특이값 중 LoRA가 남기는 위 r개");
  const task = makeTask();
  const tsv = trueSv(task);
  let ni = 2, rank = 2; // n = 100, rank 2 (장의 예)
  let res = fit(task, NS[ni]);
  const isFull = () => rank > RMAX;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 40, PR = 12, top = 24, mid = Math.round(H * 0.56), bottom = H - 22;
    // 위: rank별 시험 MSE (x = 1..16, full)
    const nx = RMAX + 2;
    const X = (i: number) => PL + ((i - 0.5) / (nx - 0.5)) * (W - PL - PR); // i = 1..16, full = 18
    const yMax = 1.2, b1 = mid - 22;
    const Y = (v: number) => b1 - (Math.min(v, yMax) / yMax) * (b1 - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (const v of [0, 0.4, 0.8, 1.2]) {
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(1), PL - 30, Y(v) + 4);
    }
    for (const i of [1, 2, 4, 8, 16]) { const s = String(i); ctx.fillText(s, X(i) - ctx.measureText(s).width / 2, b1 + 14); }
    ctx.fillText("full", Math.min(X(nx) - 12, W - PR - ctx.measureText("full").width), b1 + 14);
    ctx.fillText("시험 MSE", PL, top - 10);
    const zl = `ΔW = 0: ${res.zero.toFixed(2)}`;
    ctx.fillText(zl, W - PR - ctx.measureText(zl).width, top - 10);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(PL, Y(NOISE_VAR)); ctx.lineTo(W - PR, Y(NOISE_VAR)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("잡음 0.25", X(9.5), Y(NOISE_VAR) + 13);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); res.byRank.forEach((v, i) => (i ? ctx.lineTo(X(i + 1), Y(v)) : ctx.moveTo(X(i + 1), Y(v)))); ctx.stroke();
    const dot = (i: number, v: number, sel: boolean) => {
      ctx.fillStyle = sel ? C.edge : C.ink2;
      ctx.beginPath(); ctx.arc(X(i), Y(v), sel ? 5.5 : 3, 0, 7); ctx.fill();
      if (v > yMax) { ctx.fillStyle = C.mid; ctx.fillText("▲", X(i) - 4, Y(yMax) + 12); }
    };
    res.byRank.forEach((v, i) => dot(i + 1, v, !isFull() && rank === i + 1));
    dot(nx, res.full, isFull());
    // 아래: 특이값(로그 눈금)
    const t2 = mid + 18, b2 = bottom;
    const L0 = -2, L1 = 1.3;
    const Y2 = (v: number) => b2 - ((Math.max(L0, Math.min(L1, Math.log10(Math.max(v, 1e-6)))) - L0) / (L1 - L0)) * (b2 - t2);
    const NB = 24, bw = (W - PL - PR) / NB;
    ctx.fillStyle = C.mid;
    ctx.fillText("특이값: 막대 XC/√n · 점 ΔW_true", PL, t2 - 6);
    for (const e of [-2, -1, 0, 1]) {
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(PL, Y2(10 ** e)); ctx.lineTo(W - PR, Y2(10 ** e)); ctx.stroke();
      ctx.fillStyle = C.mid; ctx.fillText(e === 0 ? "1" : e === 1 ? "10" : `1e${e}`.replace("-", "−"), PL - 34, Y2(10 ** e) + 4);
    }
    const kept = isFull() ? D : rank;
    for (let k = 0; k < NB; k++) {
      const x = PL + k * bw + 1;
      ctx.fillStyle = k < kept ? C.edge : alpha(C.ink2, 0.35);
      ctx.fillRect(x, Y2(res.sv[k]), Math.max(1, bw - 2), b2 - Y2(res.sv[k]));
      ctx.fillStyle = C.ink;
      ctx.beginPath(); ctx.arc(x + (bw - 2) / 2, Y2(tsv[k]), 2.4, 0, 7); ctx.fill();
    }
  }

  let raf = 0;
  function update(refit: boolean) {
    if (refit) res = fit(task, NS[ni]);
    const n = NS[ni];
    const cur = isFull() ? res.full : res.byRank[rank - 1];
    const best = Math.min(...res.byRank, res.full);
    ro.set("cur", `${isFull() ? "full" : `rank ${rank}`} 시험 MSE ${cur.toFixed(3)}`, Math.abs(cur - best) < 1e-9 ? "ok" : "hot");
    ro.set("full", `full ${res.full.toFixed(3)} · ΔW = 0 ${res.zero.toFixed(2)}`);
    ro.set("params", `학습 파라미터 ${isFull() ? D * D : rank * 2 * D} / ${D * D}`);
    pageState.set("lora", { n, rank: isFull() ? D : rank, mse: cur }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pr = param({ label: "r", min: 1, max: RMAX + 1, step: 1, value: rank, fmt: (v) => (v > RMAX ? "full" : String(v)), aria: "LoRA rank r (맨 끝은 full fine-tuning)", onInput: (v) => { rank = v; update(false); } });
  const pn = param({ label: "n", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => String(NS[v]), aria: "학습 표본 수 n", onInput: (v) => { ni = v; update(true); } });
  pn.input.setAttribute("aria-valuetext", String(NS[ni]));
  pn.input.addEventListener("input", () => pn.input.setAttribute("aria-valuetext", String(NS[ni])));
  const grid = el("div", "param-grid");
  grid.append(pr.row, pn.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update(false);
}
