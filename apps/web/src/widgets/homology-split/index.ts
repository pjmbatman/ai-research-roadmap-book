// 상동성 기준 분할 (data/widgets.toml: homology-split).
// 29.5장의 합성 데이터를 같은 생성 과정으로 만든다: 상위 가족 15개 × 가족 4개 × 구성원 5개 = 서열 300개, 길이 80, 잔기 20종.
// 뿌리 → 가족 조상(자리마다 35% 확률로 무작위 잔기) → 구성원(15%). 서열 동일성 = 같은 자리 잔기가 같은 비율(정렬 없음).
// label: 상위 가족마다 0/1, 가족의 15%는 기능이 갈라진다(뒤집힘). 모델: 가장 비슷한 train 서열의 label을 옮긴다(1-NN).
// 분할: 무작위 20% / 단일 연결 군집(동일성 ≥ t인 쌍을 이은 그래프의 연결 성분) / 대표 서열 탐욕 군집.
// 군집 분할은 군집을 무작위 순서로 test에 넣어 서열 60개(20%) 이상이 되면 멈춘다(90개를 넘기는 군집은 건너뛴다).
// 분할 5번(시드 고정)의 test 결과를 모아 train과의 최대 동일성 분포와 1-NN 정확도를 낸다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

const NS = 300, LEN = 80, REPS = 5, TEST_MIN = 60, TEST_MAX = 90;
export type Method = "random" | "single" | "greedy";

export function makeData(seed = 29) {
  const r = rng(seed);
  const evolve = (s: Uint8Array, rate: number) => { const o = s.slice(); for (let i = 0; i < o.length; i++) if (r.u() < rate) o[i] = Math.floor(r.u() * 20); return o; };
  const seqs: Uint8Array[] = [], fam: number[] = [];
  for (let s = 0; s < 15; s++) {
    const root = Uint8Array.from({ length: LEN }, () => Math.floor(r.u() * 20));
    for (let fa = 0; fa < 4; fa++) {
      const anc = evolve(root, 0.35);
      for (let m = 0; m < 5; m++) { seqs.push(evolve(anc, 0.15)); fam.push(s * 4 + fa); }
    }
  }
  const ident = new Float64Array(NS * NS);
  for (let i = 0; i < NS; i++) for (let j = i; j < NS; j++) {
    let c = 0; for (let k = 0; k < LEN; k++) if (seqs[i][k] === seqs[j][k]) c++;
    ident[i * NS + j] = ident[j * NS + i] = c / LEN;
  }
  const supLabel = Array.from({ length: 15 }, () => (r.u() < 0.5 ? 0 : 1));
  const famLabel = Array.from({ length: 60 }, (_, fa) => (r.u() < 0.15 ? 1 - supLabel[Math.floor(fa / 4)] : supLabel[Math.floor(fa / 4)]));
  const y = fam.map((fa) => famLabel[fa]);
  const order = Array.from({ length: NS }, (_, i) => i);
  for (let i = NS - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  return { ident, y, order };
}
type Data = ReturnType<typeof makeData>;

/** 군집 번호 (NS,) */
export function clusters(d: Data, method: Method, t: number): Int32Array {
  const lab = new Int32Array(NS);
  if (method === "random") { for (let i = 0; i < NS; i++) lab[i] = i; return lab; }
  if (method === "single") {
    const par = Int32Array.from({ length: NS }, (_, i) => i);
    const find = (a: number): number => { while (par[a] !== a) { par[a] = par[par[a]]; a = par[a]; } return a; };
    for (let i = 0; i < NS; i++) for (let j = i + 1; j < NS; j++) if (d.ident[i * NS + j] >= t) { const a = find(i), b = find(j); if (a !== b) par[a] = b; }
    const ids = new Map<number, number>();
    for (let i = 0; i < NS; i++) { const rt = find(i); if (!ids.has(rt)) ids.set(rt, ids.size); lab[i] = ids.get(rt)!; }
    return lab;
  }
  // 탐욕적: 순서대로 보며, 기존 대표와 동일성 ≥ t이면 (처음 만난) 그 군집에, 아니면 새 대표
  const reps: number[] = [];
  for (const i of d.order) {
    let c = -1;
    for (let k = 0; k < reps.length; k++) if (d.ident[i * NS + reps[k]] >= t) { c = k; break; }
    if (c < 0) { reps.push(i); c = reps.length - 1; }
    lab[i] = c;
  }
  return lab;
}

export function evaluate(d: Data, method: Method, t: number) {
  const lab = clusters(d, method, t);
  const nCl = Math.max(...lab) + 1;
  const size = new Int32Array(nCl);
  for (const c of lab) size[c]++;
  const maxId: number[] = [], correct: boolean[] = [];
  let testTotal = 0, failed = 0;
  for (let rep = 0; rep < REPS; rep++) {
    const r = rng(1000 + rep);
    const perm = Array.from({ length: nCl }, (_, i) => i);
    for (let i = nCl - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const inTest = new Uint8Array(nCl);
    let n = 0;
    for (const c of perm) { if (n >= TEST_MIN) break; if (n + size[c] > TEST_MAX) continue; inTest[c] = 1; n += size[c]; }
    if (n < TEST_MIN / 2) { failed++; continue; }
    const test: number[] = [], train: number[] = [];
    for (let i = 0; i < NS; i++) (inTest[lab[i]] ? test : train).push(i);
    testTotal += test.length;
    for (const i of test) {
      let best = -1, nb = -1;
      for (const j of train) { const v = d.ident[i * NS + j]; if (v > best) { best = v; nb = j; } }
      maxId.push(best); correct.push(d.y[nb] === d.y[i]);
    }
  }
  const sorted = [...maxId].sort((a, b) => a - b);
  const acc = correct.length ? correct.filter(Boolean).length / correct.length : NaN;
  const above = maxId.filter((v) => v >= t).length / Math.max(1, maxId.length);
  return { nCl, largest: Math.max(...size), maxId, acc, median: sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] * 0.5 + sorted[Math.ceil((sorted.length - 1) / 2)] * 0.5 : NaN, max: sorted.length ? sorted[sorted.length - 1] : NaN, above, testMean: testTotal / Math.max(1, REPS - failed), failed };
}

export const TS = Array.from({ length: 41 }, (_, i) => 0.1 + i * 0.02);

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400, wide = !narrow; // 수업 시트(약 419px)에서도 넓은 배치
  const f = frame(root, "상동성 기준 분할", "explore", wide ? 0.56 : 0.95,
    "test 서열마다 train과의 최대 서열 동일성 히스토그램과, 동일성 기준값에 따른 1-최근접 이웃 정확도 곡선");
  const d = makeData();
  const rand = evaluate(d, "random", 0.3);
  const accCurve: Record<Exclude<Method, "random">, number[]> = {
    single: TS.map((t) => evaluate(d, "single", t).acc),
    greedy: TS.map((t) => evaluate(d, "greedy", t).acc),
  };
  let method: Method = "single", t = 0.3;
  let ev = evaluate(d, method, t);
  const NB = 25;
  const hist = (xs: number[]) => { const h = new Array(NB).fill(0); for (const v of xs) h[Math.min(NB - 1, Math.floor(v * NB))]++; return h.map((c) => c / Math.max(1, xs.length)); };
  const hRand = hist(rand.maxId);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const mono = "11px JetBrains Mono, monospace", sans = "600 12px Pretendard Variable, sans-serif";
    type Box = { x: number; y: number; w: number; h: number };
    let A: Box, B: Box;
    if (wide) {
      const lw = Math.round(W * 0.58);
      A = { x: 40, y: 26, w: lw - 40 - 10, h: H - 26 - 26 };
      B = { x: lw + 44, y: 26, w: W - lw - 44 - 12, h: H - 26 - 26 };
    } else {
      const ah = Math.round((H - 26 - 26 - 32) * 0.58);
      A = { x: 40, y: 24, w: W - 52, h: ah };
      B = { x: 40, y: 24 + ah + 44, w: W - 52, h: H - (24 + ah + 44) - 22 };
    }
    // ── 히스토그램
    const h = hist(ev.maxId);
    const ymax = Math.max(0.3, ...h, ...hRand) * 1.08;
    const X = (v: number) => A.x + v * A.w, Y = (p: number) => A.y + A.h - (p / ymax) * A.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(A.x + 0.5, A.y + 0.5, A.w - 1, A.h - 1);
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) ctx.fillText(v.toFixed(1), Math.min(Math.max(X(v), A.x + 8), A.x + A.w - 10), A.y + A.h + 14);
    ctx.textAlign = "right";
    for (const p of [0, 0.2]) ctx.fillText(p.toFixed(1), A.x - 5, Y(p) + 4);
    if (method !== "random") {
      for (let i = 0; i < NB; i++) {
        const lo = i / NB;
        ctx.fillStyle = lo + 1 / NB > t + 1e-9 ? alpha(C.edge, 0.8) : alpha(C.ink2, 0.7);
        ctx.fillRect(X(lo) + 1, Y(h[i]), X(1 / NB) - A.x - 2, A.y + A.h - Y(h[i]));
      }
    }
    // 무작위 분할의 분포(윤곽)
    ctx.strokeStyle = method === "random" ? C.ink2 : C.mid; ctx.lineWidth = method === "random" ? 2.2 : 1.4; ctx.setLineDash(method === "random" ? [] : [4, 3]);
    ctx.beginPath(); ctx.moveTo(X(0), Y(0));
    for (let i = 0; i < NB; i++) { ctx.lineTo(X(i / NB), Y(hRand[i])); ctx.lineTo(X((i + 1) / NB), Y(hRand[i])); }
    ctx.lineTo(X(1), Y(0)); ctx.stroke(); ctx.setLineDash([]);
    if (method !== "random") {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(X(t), A.y); ctx.lineTo(X(t), A.y + A.h); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.textAlign = X(t) > A.x + A.w - 50 ? "right" : "left"; ctx.font = mono;
      ctx.fillText(`t = ${t.toFixed(2)}`, X(t) + (ctx.textAlign === "right" ? -5 : 5), A.y + 14);
    }
    ctx.font = sans; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    ctx.fillText(A.w < 260 ? "train과 최대 동일성" : "test–train 최대 동일성", A.x, A.y - 9);
    if (method !== "random") { ctx.font = mono; ctx.textAlign = "right"; ctx.fillText(A.w < 260 ? "" : "점선: 무작위", A.x + A.w, A.y - 9); }
    // ── 정확도 곡선
    const AX = (v: number) => B.x + ((v - TS[0]) / (TS[TS.length - 1] - TS[0])) * B.w, AY = (a: number) => B.y + B.h - ((a - 0.3) / 0.75) * B.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(B.x + 0.5, B.y + 0.5, B.w - 1, B.h - 1);
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const a of [0.5, 0.75, 1]) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(B.x, AY(a)); ctx.lineTo(B.x + B.w, AY(a)); ctx.stroke(); ctx.fillText(a.toFixed(2), B.x - 5, AY(a) + 4); }
    ctx.textAlign = "center";
    for (const v of [0.1, 0.3, 0.5, 0.7, 0.9]) ctx.fillText(v.toFixed(1), Math.min(Math.max(AX(v), B.x + 8), B.x + B.w - 10), B.y + B.h + 14);
    ctx.setLineDash([4, 3]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(B.x, AY(rand.acc)); ctx.lineTo(B.x + B.w, AY(rand.acc)); ctx.stroke(); ctx.setLineDash([]);
    for (const k of ["greedy", "single"] as const) {
      const on = k === method;
      ctx.strokeStyle = on ? C.ink2 : alpha(C.mid, 0.55); ctx.lineWidth = on ? 2.2 : 1.3;
      ctx.beginPath();
      let started = false;
      TS.forEach((v, i) => { const a = accCurve[k][i]; if (!Number.isFinite(a)) { started = false; return; } if (started) ctx.lineTo(AX(v), AY(a)); else { ctx.moveTo(AX(v), AY(a)); started = true; } });
      ctx.stroke();
    }
    if (method !== "random" && Number.isFinite(ev.acc)) { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(AX(t), AY(ev.acc), 5.5, 0, 7); ctx.fill(); }
    ctx.font = sans; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    ctx.fillText(B.w < 260 ? "t에 따른 1-NN 정확도" : "기준값 t에 따른 1-NN 겉보기 정확도", B.x, B.y - 9);
    ctx.font = mono; ctx.textAlign = "right";
    ctx.fillText(`무작위 ${rand.acc.toFixed(2)}`, B.x + B.w - 6, AY(rand.acc) + 14);
  }

  function update() {
    ev = evaluate(d, method, t);
    if (method !== "random" && ev.failed === REPS) {
      ro.set("cl", `군집 ${ev.nCl}개(최대 ${ev.largest})`, "hot");
      ro.set("id", "분할 불가: 거대 군집 하나", "hot");
      ro.set("lk", "—"); ro.set("acc", "—");
    } else {
      ro.set("cl", method === "random" ? `무작위 20% · test ${ev.testMean.toFixed(0)}개` : `군집 ${ev.nCl}개(최대 ${ev.largest}) · test ${ev.testMean.toFixed(0)}개`);
      ro.set("id", `최대 동일성 중앙 ${ev.median.toFixed(2)} · 최고 ${ev.max.toFixed(2)}`);
      ro.set("lk", method === "random" ? `동일성 0.6 이상 ${(ev.maxId.filter((v) => v >= 0.6).length / ev.maxId.length * 100).toFixed(0)}%` : `t 이상 test ${(ev.above * 100).toFixed(0)}%`, method !== "random" && ev.above > 0 ? "hot" : "");
      ro.set("acc", `겉보기 정확도 ${ev.acc.toFixed(2)}`, "ok");
    }
    pageState.set("homologySplit", { method, t, clusters: ev.nCl, largest: ev.largest, median_max_identity: ev.median, max_identity: ev.max, frac_above_t: ev.above, accuracy: ev.acc }, root);
    draw();
  }

  const pt = param({ label: "기준 t", min: 0.1, max: 0.9, step: 0.02, value: t, fmt: (v) => v.toFixed(2), aria: "군집을 묶는 서열 동일성 기준값", onInput: (v) => { t = v; update(); } });
  pt.row.style.gridTemplateColumns = "3.6em minmax(0, 1fr) 3em";
  const pre = presets<Method>([{ key: "random", label: "무작위" }, { key: "single", label: "단일 연결" }, { key: "greedy", label: "대표 서열 탐욕" }], (k) => { method = k; pt.input.disabled = k === "random"; update(); }, method);
  pre.row.classList.add("segmented");
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 9px"));
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
