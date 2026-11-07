// Dockerfile 레이어 캐시 (data/widgets.toml: docker-layer-cache).
// 3.5장의 두 순서(코드 먼저 · lockfile 먼저)에서, 무엇을 바꿨는지에 따라 어느 줄이 cache에서 재사용되고 어느 줄이 다시 실행되는지 본다.
// 규칙은 본문과 같다: 어떤 줄의 입력(명령 문자열, 복사하는 파일의 내용)이 바뀌면 그 줄과 그 뒤의 모든 줄을 다시 실행한다.
// 줄마다 걸리는 시간은 가상의 값이다(의존성 설치가 가장 길다는 자릿수만 보인다).
import { frame, presets, readouts, C, alpha, reducedMotion } from "../frame";

export type Change = "none" | "code" | "lock" | "base";
type Input = "base" | "lock" | "code";
export interface Line { text: string; inputs: Input[]; sec: number }
export type Order = "lock" | "code";

// 가상 시간(초): base pull 20, 파일 복사 1–2, 의존성 설치 180, 프로젝트 자신 설치 5
export const DOCKERFILES: Record<Order, { label: string; lines: Line[] }> = {
  lock: {
    label: "lockfile 먼저",
    lines: [
      { text: "FROM python:3.12-slim", inputs: ["base"], sec: 20 },
      { text: "COPY pyproject.toml uv.lock ./", inputs: ["lock"], sec: 1 },
      { text: "RUN uv sync --locked --no-install-project", inputs: [], sec: 180 },
      { text: "COPY . .", inputs: ["lock", "code"], sec: 2 },
      { text: "RUN uv sync --locked", inputs: [], sec: 5 },
    ],
  },
  code: {
    label: "코드 먼저",
    lines: [
      { text: "FROM python:3.12-slim", inputs: ["base"], sec: 20 },
      { text: "COPY . .", inputs: ["lock", "code"], sec: 2 },
      { text: "RUN uv sync --locked", inputs: [], sec: 185 },
    ],
  },
};
const CHANGES: { key: Change; label: string }[] = [
  { key: "code", label: "train.py 수정" },
  { key: "lock", label: "uv.lock 수정" },
  { key: "base", label: "base image 변경" },
  { key: "none", label: "변경 없음" },
];

/** 바뀐 입력에 대해: 처음 무효화되는 줄의 번호(없으면 -1)와 다시 실행되는 줄들, 빌드 시간 */
export function rebuild(order: Order, change: Change) {
  const lines = DOCKERFILES[order].lines;
  const first = change === "none" ? -1 : lines.findIndex((l) => l.inputs.includes(change));
  const rerun = lines.map((_, i) => first >= 0 && i >= first);
  const sec = lines.reduce((s, l, i) => s + (rerun[i] ? l.sec : 0), 0);
  const full = lines.reduce((s, l) => s + l.sec, 0);
  return { first, rerun, sec, full };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "Dockerfile 레이어 캐시", "compare", ((w) => (w < 400 ? 1 : w < 560 ? 0.85 : 0.5))(root.clientWidth || 600), "Dockerfile의 줄마다 cache 재사용 여부와 다시 실행에 걸리는 가상 시간을 막대로 그린 그림");
  let order: Order = "lock", change: Change = "code";
  let shownFrac: number[] = []; // 막대가 새 길이로 자라는 모습(한 프레임 안에서 끝나도록 짧게)

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const df = DOCKERFILES[order], r = rebuild(order, change);
    const narrow = W < 560;
    const pad = 14, top = 34;
    const n = df.lines.length + 1; // 마지막 줄은 합계
    const rowH = Math.min(narrow ? 62 : 46, (H - top - 10) / n);
    const textW = narrow ? W - 2 * pad : Math.min(340, W * 0.48);
    const barX = narrow ? pad + 62 : pad + textW + 16, barW = W - pad - barX - 62;
    const maxSec = Math.max(DOCKERFILES.lock.lines.reduce((s, l) => s + l.sec, 0), DOCKERFILES.code.lines.reduce((s, l) => s + l.sec, 0));
    const bx = (s: number) => (s / maxSec) * barW;

    ctx.textBaseline = "middle";
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid;
    ctx.fillText(`Dockerfile · ${df.label}`, pad, 16);
    ctx.textAlign = "right";
    ctx.fillText("다시 실행 시간(가상)", W - pad, 16);
    ctx.textAlign = "start";

    df.lines.forEach((l, i) => {
      const y = top + i * rowH;
      const re = r.rerun[i], isFirst = i === r.first;
      // 줄 배경: 무효화가 시작된 줄부터 아래로 옅은 강조
      if (re) { ctx.fillStyle = alpha(C.edge, isFirst ? 0.16 : 0.08); roundRect(pad - 6, y + 2, W - 2 * pad + 12, rowH - 4, 8); ctx.fill(); }
      const ty = narrow ? y + rowH * 0.3 : y + rowH / 2;
      const by = narrow ? y + rowH * 0.7 : y + rowH / 2;
      ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillStyle = C.mid;
      ctx.fillText(String(i + 1), pad, ty);
      ctx.font = `${isFirst ? "700" : "500"} 12px JetBrains Mono, monospace`;
      ctx.fillStyle = re ? C.ink : C.mid;
      ctx.fillText(fitText(l.text, (narrow ? W - 2 * pad : textW) - 18), pad + 18, ty);
      // 상태 표시
      ctx.font = "600 11px Pretendard Variable, sans-serif";
      const tag = re ? "다시 실행" : "CACHED";
      ctx.fillStyle = re ? C.edge : C.mid;
      if (narrow) ctx.fillText(tag, pad + 2, by);
      // 막대: 처음부터 빌드할 때의 길이(흐림) 위에 이번 빌드의 길이
      const full = bx(l.sec), cur = full * (shownFrac[i] ?? (re ? 1 : 0));
      const bh = Math.min(14, rowH * 0.32);
      ctx.fillStyle = C.faint;
      ctx.fillRect(barX, by - bh / 2, Math.max(full, 2), bh);
      if (cur > 0.5) { ctx.fillStyle = C.edge; ctx.fillRect(barX, by - bh / 2, Math.max(cur, 2), bh); }
      ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillStyle = re ? C.ink : C.mid;
      ctx.fillText(re ? `${l.sec} s` : "0 s", barX + Math.max(full, 2) + 6, by);
      if (!narrow) { ctx.font = "600 11px Pretendard Variable, sans-serif"; ctx.fillStyle = re ? C.edge : C.mid; ctx.textAlign = "right"; ctx.fillText(tag, barX - 8, by); ctx.textAlign = "start"; }
    });
    // 합계
    const y = top + df.lines.length * rowH;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, y + 2); ctx.lineTo(W - pad, y + 2); ctx.stroke();
    const by = y + rowH / 2 + 2, bh = Math.min(16, rowH * 0.36);
    ctx.font = "700 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.ink;
    ctx.fillText("빌드 합계", pad, by);
    ctx.fillStyle = C.faint;
    ctx.fillRect(barX, by - bh / 2, bx(r.full), bh);
    const tot = r.rerun.reduce((s, re, i) => s + df.lines[i].sec * (shownFrac[i] ?? (re ? 1 : 0)), 0);
    if (tot > 0.5) { ctx.fillStyle = C.edge; ctx.fillRect(barX, by - bh / 2, bx(tot), bh); }
    ctx.font = "600 11px JetBrains Mono, monospace";
    ctx.fillStyle = C.ink;
    ctx.fillText(`${r.sec} s`, barX + Math.max(bx(r.full), 2) + 6, by);
    ctx.textBaseline = "alphabetic";
  }
  function roundRect(x: number, y: number, w: number, h: number, rr: number) {
    const c = f.ctx; c.beginPath(); c.moveTo(x + rr, y); c.arcTo(x + w, y, x + w, y + h, rr); c.arcTo(x + w, y + h, x, y + h, rr); c.arcTo(x, y + h, x, y, rr); c.arcTo(x, y, x + w, y, rr); c.closePath();
  }
  function fitText(t: string, w: number) {
    if (f.ctx.measureText(t).width <= w) return t;
    let s = t;
    while (s.length > 4 && f.ctx.measureText(s + "…").width > w) s = s.slice(0, -1);
    return s + "…";
  }

  let anim = 0;
  function update(grow = true) {
    const r = rebuild(order, change), other = rebuild(order === "lock" ? "code" : "lock", change);
    const df = DOCKERFILES[order];
    const nRe = r.rerun.filter(Boolean).length;
    ro.set("lines", `다시 실행 ${nRe} / ${df.lines.length}줄`, nRe ? "hot" : "ok");
    ro.set("first", r.first >= 0 ? `무효화 시작: ${r.first + 1}번 줄` : "모든 줄 cache 재사용");
    ro.set("sec", `이번 빌드 ${r.sec} s (가상)`, "hot");
    ro.set("other", `${DOCKERFILES[order === "lock" ? "code" : "lock"].label}였다면 ${other.sec} s`);
    cancelAnimationFrame(anim);
    const target = r.rerun.map((re) => (re ? 1 : 0));
    if (!grow || reducedMotion()) { shownFrac = target; draw(); return; }
    shownFrac = target.map(() => 0);
    const t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / 420), e = 1 - Math.pow(1 - k, 3);
      shownFrac = target.map((v) => v * e);
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  const orderBtns = presets<Order>([{ key: "lock", label: "lockfile 먼저" }, { key: "code", label: "코드 먼저" }], (k) => { order = k; update(); }, order);
  orderBtns.row.setAttribute("aria-label", "Dockerfile 줄 순서");
  const changeBtns = presets<Change>(CHANGES, (k) => { change = k; update(); }, change);
  changeBtns.row.setAttribute("aria-label", "바꾼 것");
  const ro = readouts();
  f.panel.append(orderBtns.row, changeBtns.row, ro.row);
  f.onResize(draw);
  update(false);
}
