// 학습 곡선 진단 (data/widgets.toml: learning-curve-reader).
// 곡선 넷(과적합·과소적합·학습률 큼·학습률 작음)은 17.8장 "학습 곡선 해석" 셀의 합성식 그대로이고,
// 둘(누수·label 오류)은 17.9장 증상 표의 모양을 같은 방식으로 합성한 것이다. 잡음 0.01·N(0, 1)은 위젯의 고정 시드.
// 특징은 본문 셀과 같다: 마지막 train, val 최저 epoch, 마지막 gap, 마지막 10 epoch val 기울기(최소제곱), val 흔들림 std(diff(val)).
// 곡선은 이름을 숨긴 A–F로 고르고, 진단을 누르면 맞는지와 다음 실험을 readout으로 보인다.
import { frame, presets, readouts, C, rng, alpha } from "../frame";

export const EP = 50;
const LN3 = Math.log(3), PRIOR = 0.4;
type Dx = "over" | "under" | "lrhi" | "lrlo" | "leak" | "label";
const ep = Array.from({ length: EP }, (_, i) => i + 1);
const SHAPES: Record<Dx, (e: number) => [number, number]> = {
  over: (e) => [1.0 * Math.exp(-e / 8) + 0.05, 0.9 * Math.exp(-e / 8) + 0.25 + 0.004 * e],
  under: (e) => [0.6 + 0.2 * Math.exp(-e / 5), 0.62 + 0.2 * Math.exp(-e / 5)],
  lrhi: (e) => [0.4 + 0.3 * Math.abs(Math.sin(e)), 0.45 + 0.3 * Math.abs(Math.sin(e + 1))],
  lrlo: (e) => [1.0 - 0.006 * e, 1.02 - 0.006 * e],
  leak: (e) => [1.0 * Math.exp(-e / 8) + 0.05, 0.9 * Math.exp(-e / 8) + 0.01],
  label: (e) => [LN3 - 0.35 * (1 - Math.exp(-e / 25)), LN3 + 0.002 * e],
};
export const CURVES = (Object.keys(SHAPES) as Dx[]).reduce((acc, k, i) => {
  const r = rng(301 + i);
  const tr = ep.map((e) => SHAPES[k](e)[0] + 0.01 * r.normal());
  const va = ep.map((e) => SHAPES[k](e)[1] + 0.01 * r.normal());
  acc[k] = { tr, va };
  return acc;
}, {} as Record<Dx, { tr: number[]; va: number[] }>);

export function features(tr: number[], va: number[]) {
  let best = 0; va.forEach((v, i) => { if (v < va[best]) best = i; });
  const xs = ep.slice(-10), ys = va.slice(-10);
  const mx = xs.reduce((a, b) => a + b, 0) / 10, my = ys.reduce((a, b) => a + b, 0) / 10;
  const slope = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const d = va.slice(1).map((v, i) => v - va[i]);
  const md = d.reduce((a, b) => a + b, 0) / d.length;
  const wiggle = Math.sqrt(d.reduce((s, v) => s + (v - md) ** 2, 0) / d.length);
  return { lastTr: tr[EP - 1], bestEp: best + 1, bestVa: va[best], gap: va[EP - 1] - tr[EP - 1], slope, wiggle };
}

const DX: Record<Dx, { label: string; next: string; hint: (f: ReturnType<typeof features>) => string }> = {
  over: { label: "과적합", next: "weight decay·dropout·증강 강화, 모델 축소, 그동안 best checkpoint", hint: (f) => `gap ${f.gap.toFixed(3)}, val 최저 epoch ${f.bestEp} 뒤 상승` },
  under: { label: "과소적합", next: "모델 용량 확대, 입력 표현 개선", hint: (f) => `train ${f.lastTr.toFixed(2)}에서 평평, 기울기 ${f.slope.toFixed(4)}` },
  lrhi: { label: "학습률 큼", next: "학습률 축소, warm-up, gradient clipping", hint: (f) => `val 흔들림 ${f.wiggle.toFixed(3)}` },
  lrlo: { label: "학습률 작음", next: "학습률 확대 또는 학습 연장", hint: (f) => `아직 내려가는 중, 기울기 ${f.slope.toFixed(4)}` },
  leak: { label: "누수", next: "train–검증 중복, 정답을 담은 특징, 분할 단위 확인", hint: (f) => `val 최저 ${f.bestVa.toFixed(3)} ≪ 기존 최고 ${PRIOR}` },
  label: { label: "label 오류", next: "표본을 직접 보고 선형 baseline 확인, 입력–label 정렬 점검", hint: (f) => `val이 우연 수준 ln 3 = ${LN3.toFixed(3)} 근처에 머묾 (최저 ${f.bestVa.toFixed(3)})` },
};
/** 받침에 따라 은/는을 붙인다 */
const topic = (w: string) => { const c = w.charCodeAt(w.length - 1); return w + (c >= 0xac00 && c <= 0xd7a3 && (c - 0xac00) % 28 ? "은" : "는"); };
// 곡선 버튼은 진단과 순서를 섞어 이름을 숨긴다
const ORDER: Dx[] = ["lrhi", "over", "label", "lrlo", "leak", "under"];
const KEYS = ["A", "B", "C", "D", "E", "F"] as const;
type CK = (typeof KEYS)[number];

export function mount(root: HTMLElement) {
  const f = frame(root, "학습 곡선 진단", "manipulate", 0.5, "epoch에 따른 train 손실과 검증 손실, 우연 수준과 기존 최고 기준선");
  let cur: Dx = ORDER[0];
  let pick: Dx | null = null;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 48, top = 16, bottom = H - 22, Y1 = 1.4;
    const X = (e: number) => PL + ((e - 1) / (EP - 1)) * (W - PL - PR);
    const Y = (v: number) => bottom - (Math.min(Math.max(v, 0), Y1) / Y1) * (bottom - top);
    ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      ctx.strokeStyle = C.faint;
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      ctx.fillStyle = C.mid; ctx.fillText(v.toFixed(1), 4, Y(v) + 4);
    }
    for (const e of [1, 10, 20, 30, 40, 50]) { const t = String(e), tw = ctx.measureText(t).width; ctx.fillStyle = C.mid; ctx.fillText(t, X(e) - tw / 2, bottom + 14); }
    const { tr, va } = CURVES[cur];
    const path = (arr: number[], col: string, wdt: number) => {
      ctx.strokeStyle = col; ctx.lineWidth = wdt;
      ctx.beginPath(); arr.forEach((v, i) => (i ? ctx.lineTo(X(i + 1), Y(v)) : ctx.moveTo(X(i + 1), Y(v)))); ctx.stroke();
    };
    path(tr, C.ink2, 2);
    path(va, C.edge, 2.4);
    // 기준선: 우연 수준 ln 3, 기존 최고 검증 손실(가상)
    ctx.setLineDash([3, 4]);
    for (const [v, lab] of [[LN3, "ln 3"], [PRIOR, "기존 최고(가상)"]] as const) {
      ctx.strokeStyle = C.mid;
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      const tw = ctx.measureText(lab).width;
      ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(PL + 2, Y(v) - 15, tw + 6, 13);
      ctx.fillStyle = C.mid; ctx.fillText(lab, PL + 5, Y(v) - 5);
    }
    ctx.setLineDash([]);
    const fe = features(tr, va);
    ctx.fillStyle = alpha(C.edge, 0.9);
    ctx.beginPath(); ctx.arc(X(fe.bestEp), Y(fe.bestVa), 4, 0, 7); ctx.fill();
    // 끝 라벨(겹치면 벌린다)
    let yt = Y(tr[EP - 1]) + 4, yv = Y(va[EP - 1]) + 4;
    if (Math.abs(yt - yv) < 13) { const m = (yt + yv) / 2, s = yt < yv ? -1 : 1; yt = m + s * 6.5; yv = m - s * 6.5; }
    ctx.fillStyle = C.ink2; ctx.fillText("train", W - PR + 6, yt);
    ctx.fillStyle = C.edge; ctx.fillText("val", W - PR + 6, yv);
    ctx.fillStyle = C.mid; ctx.fillText("epoch", W - PR - 30, top + 2);
  }

  function update() {
    const { tr, va } = CURVES[cur];
    const fe = features(tr, va);
    ro.set("tr", `train ${fe.lastTr.toFixed(3)}`);
    ro.set("best", `val 최저 ${fe.bestVa.toFixed(3)} @${fe.bestEp}`);
    ro.set("gap", `gap ${fe.gap.toFixed(3)}`);
    ro.set("slope", `끝 10 epoch 기울기 ${fe.slope.toFixed(4)}`);
    ro.set("wig", `흔들림 ${fe.wiggle.toFixed(3)}`);
    if (!pick) ro.set("verdict", "진단을 고른다");
    else if (pick === cur) ro.set("verdict", `맞다: ${DX[cur].label} → ${DX[cur].next}`, "ok");
    else ro.set("verdict", `${topic(DX[pick].label)} 아니다. 단서: ${DX[cur].hint(fe)}`, "hot");
    draw();
  }

  const curveBtns = presets(KEYS.map((k) => ({ key: k, label: k })), (k: CK) => { cur = ORDER[KEYS.indexOf(k)]; pick = null; dxBtns.select(null); update(); }, "A");
  curveBtns.row.classList.add("segmented");
  const dxBtns = presets((Object.keys(DX) as Dx[]).map((k) => ({ key: k, label: DX[k].label })), (k) => { pick = k; update(); });
  const ro = readouts();
  curveBtns.row.setAttribute("aria-label", "곡선 A–F");
  dxBtns.row.setAttribute("role", "toolbar");   // 여섯 개라 줄바꿈되는 버튼 묶음으로 둔다
  dxBtns.row.setAttribute("aria-label", "진단");
  f.panel.append(curveBtns.row, dxBtns.row, ro.row);
  f.onResize(draw);
  update();
}
