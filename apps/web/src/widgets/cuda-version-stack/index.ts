// 드라이버·CUDA·프레임워크 호환 (data/widgets.toml: cuda-version-stack).
// 3.1장 "버전 정보 읽기"와 "PyTorch 바이너리와 CUDA"의 규칙만 쓴다.
// ① 실행: PyTorch가 빌드된 CUDA(torch.version.cuda)를 드라이버가 지원하는 최대 CUDA(nvidia-smi)와 정수 tuple로 비교한다.
//    이하면 실행 가능, 같은 major에서 더 높으면 minor version compatibility(일부 기능 제한), 다른 major로 높으면 실행 불가.
// ② toolkit(nvcc)은 실행 판정에 쓰이지 않는다. custom CUDA extension을 빌드할 때만 필요하고, 그 major가 torch.version.cuda의 major와 같아야 한다.
// 버전 수치는 본문에 나온 것(11.8, 12.1, 12.4, 12.6, 13.0)만 쓴다.
import { frame, presets, readouts, el, C, alpha } from "../frame";

export const VERS = ["11.8", "12.1", "12.4", "12.6", "13.0"];
export const ver = (s: string) => s.split(".").map(Number);
const cmp = (a: number[], b: number[]) => (a[0] - b[0]) || (a[1] - b[1]);

/** 본문 judge()와 같은 규칙 */
export function judgeRun(driver: string, runtime: string | null): { tone: "ok" | "warn" | "bad" | "none"; text: string } {
  if (runtime === null) return { tone: "none", text: "CPU 전용 빌드: GPU를 쓰지 않는다" };
  const d = ver(driver), r = ver(runtime);
  if (cmp(r, d) <= 0) return { tone: "ok", text: "실행 가능" };
  if (r[0] === d[0]) return { tone: "warn", text: "같은 major: minor version compatibility (일부 기능 제한)" };
  return { tone: "bad", text: "실행 불가: 드라이버 업그레이드 필요" };
}
export function judgeBuild(task: Task, toolkit: string | null, runtime: string | null): { tone: "ok" | "bad" | "none"; text: string } {
  if (task === "run") return { tone: "none", text: "toolkit은 실행에 쓰이지 않는다" };
  if (runtime === null) return { tone: "none", text: "CPU 전용 빌드: 맞출 CUDA가 없다" };
  if (toolkit === null) return { tone: "bad", text: "extension 빌드 불가: nvcc 없음" };
  return ver(toolkit)[0] === ver(runtime)[0] ? { tone: "ok", text: `major 일치(${ver(toolkit)[0]}): extension 빌드 가능` } : { tone: "bad", text: `major 불일치(${ver(toolkit)[0]} ≠ ${ver(runtime)[0]}): 빌드가 멈춘다` };
}
type Task = "run" | "build";

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "드라이버·CUDA·프레임워크 호환", "manipulate", narrow ? 1.1 : 0.56, "소프트웨어 스택의 층과 층 사이 호환 판정, 그리고 드라이버가 실행할 수 있는 CUDA 버전 범위");
  let driver = "12.4", toolkit: string | null = "12.1", torch: string | null = "12.1", task: Task = "run";

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "middle";
    const run = judgeRun(driver, torch), build = judgeBuild(task, toolkit, torch);
    const tone = (t: string) => (t === "ok" ? C.good : t === "warn" ? C.ink2 : t === "bad" ? C.edge : C.mid);
    // 층(아래 → 위). 판정에 쓰이지 않는 층은 흐리게
    const layers = [
      { key: "code", name: "내 코드 · PyTorch", val: task === "build" ? "custom CUDA extension 빌드" : "학습·추론", used: true },
      { key: "rt", name: "PyTorch에 딸린 CUDA runtime", val: `torch.version.cuda = ${torch ?? "None"}`, used: true },
      { key: "tk", name: "CUDA toolkit (nvcc)", val: toolkit ? `nvcc release ${toolkit}` : "nvcc 없음", used: task === "build" && torch !== null },
      { key: "drv", name: "NVIDIA 드라이버", val: `nvidia-smi CUDA Version: ${driver}`, used: torch !== null },
      { key: "hw", name: "GPU", val: "", used: torch !== null },
    ];
    const lineH = narrow ? 92 : 86;
    const top = 14, stackH = H - top - lineH - 14;
    const lh = stackH / layers.length, gap = 6;
    const sx = 12, sw = narrow ? W - 24 - 112 : Math.min(W - 24 - 230, 420);
    const ys: Record<string, number> = {};
    layers.forEach((Ly, i) => {
      const y = top + i * lh;
      ys[Ly.key] = y + (lh - gap) / 2;
      ctx.globalAlpha = Ly.used ? 1 : 0.42;
      ctx.fillStyle = alpha(C.ink2, Ly.key === "rt" || Ly.key === "drv" ? 0.14 : 0.06);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(sx, y, sw, lh - gap, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.font = `700 ${narrow ? 12 : 13}px Pretendard Variable, sans-serif`;
      ctx.fillText(Ly.name, sx + 10, y + (lh - gap) * (Ly.val ? 0.34 : 0.5), sw - 20);
      if (Ly.val) { ctx.fillStyle = C.mid; ctx.font = "12px JetBrains Mono, monospace"; ctx.fillText(Ly.val, sx + 10, y + (lh - gap) * 0.7, sw - 20); }
      ctx.globalAlpha = 1;
    });
    // 층 사이 판정: 오른쪽 괄호와 판정 글. 실행 판정(runtime↔드라이버)은 바깥 괄호, 빌드 판정(toolkit↔runtime)은 안쪽 괄호
    const e0 = sx + sw + 3, x1 = sx + sw + 12, x2 = x1 + 14, tx = x2 + 9, maxW = W - tx - 6;
    const bracket = (x: number, ya: number, yb: number, col: string) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(e0, ya); ctx.lineTo(x, ya); ctx.lineTo(x, yb); ctx.lineTo(e0, yb); ctx.stroke();
    };
    const label = (text: string, y: number, col: string) => {
      ctx.fillStyle = col; ctx.font = "700 12px Pretendard Variable, sans-serif";
      const lines = text.split(": ");
      lines.forEach((t, i) => ctx.fillText(t, tx, y + (i - (lines.length - 1) / 2) * 15, maxW));
    };
    if (torch !== null) {
      bracket(x2, ys.rt - 6, ys.drv, tone(run.tone));
      const t = ({ ok: "실행 가능", warn: "minor 호환: 일부 기능 제한", bad: "실행 불가: 드라이버 업그레이드", none: "" } as const)[run.tone];
      label(narrow ? t.split(": ")[0] : t, (ys.tk + ys.drv) / 2 + 6, tone(run.tone));
    }
    if (task === "build" && torch !== null) {
      bracket(x1, ys.rt + 6, ys.tk, tone(build.tone));
      const t = build.tone === "ok" ? "major 일치: 빌드 가능" : toolkit ? "major 불일치: 빌드 멈춤" : "nvcc 없음: 빌드 불가";
      label(narrow ? t.split(": ")[0] : t, (ys.rt + ys.tk) / 2 - 4, tone(build.tone));
    }

    // 아래: 드라이버가 실행하는 범위(정수 tuple 순서로 늘어놓은 본문의 버전들)
    const ly = H - lineH + 30, L = 22, R2 = W - 22;
    const X = (i: number) => L + (i / (VERS.length - 1)) * (R2 - L);
    const d = ver(driver);
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText(`runtime 판정 (드라이버 최대 CUDA ${driver} 기준)`, L - 10, ly - 22);
    VERS.forEach((v, i) => {
      const r = ver(v), t = cmp(r, d) <= 0 ? "ok" : r[0] === d[0] ? "warn" : "bad";
      const x0 = i ? (X(i - 1) + X(i)) / 2 : L - 10, x1 = i < VERS.length - 1 ? (X(i) + X(i + 1)) / 2 : R2 + 10;
      ctx.fillStyle = alpha(tone(t), t === "ok" ? 0.28 : 0.22);
      ctx.fillRect(x0, ly - 8, x1 - x0 - 1, 16);
      const on = torch === v;
      ctx.fillStyle = on ? C.ink : C.mid; ctx.font = `${on ? "700 " : ""}12px JetBrains Mono, monospace`;
      ctx.fillText(v, X(i) - ctx.measureText(v).width / 2, ly + 22);
      if (on) { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.moveTo(X(i), ly + 8); ctx.lineTo(X(i) - 6, ly - 4); ctx.lineTo(X(i) + 6, ly - 4); ctx.closePath(); ctx.fill(); }
    });
  }

  function update() {
    const run = judgeRun(driver, torch), build = judgeBuild(task, toolkit, torch);
    ro.set("run", run.text, run.tone === "ok" ? "ok" : run.tone === "none" ? "" : "hot");
    ro.set("build", build.text, build.tone === "ok" ? "ok" : build.tone === "bad" ? "hot" : "");
    draw();
  }

  const label = (t: string) => { const s = el("span", "wcap"); s.textContent = t; s.style.minWidth = "4.6em"; return s; };
  const line = (t: string, g: HTMLElement) => { const w = el("div", "wframe-row"); w.append(label(t), g); return w; };
  const pd = presets(["11.8", "12.4", "13.0"].map((v) => ({ key: v, label: v })), (k) => { driver = k; update(); }, driver);
  const pk = presets(["none", "11.8", "12.1", "13.0"].map((v) => ({ key: v, label: v === "none" ? "없음" : v })), (k) => { toolkit = k === "none" ? null : k; update(); }, "12.1");
  const pt = presets(["cpu", ...VERS].map((v) => ({ key: v, label: v === "cpu" ? "CPU" : v })), (k) => { torch = k === "cpu" ? null : k; update(); }, "12.1");
  const pj = presets<Task>([{ key: "run", label: "학습·추론" }, { key: "build", label: "extension 빌드" }], (k) => { task = k; update(); }, task);
  pj.row.classList.add("segmented");
  // 좁은 화면에서는 버전 버튼을 한 줄에 담도록 여백을 줄인다
  if (narrow) for (const g of [pd, pk, pt]) g.row.querySelectorAll<HTMLButtonElement>("button").forEach((b) => { b.style.padding = "0 7px"; b.style.fontSize = "12px"; });
  const ro = readouts();
  f.panel.append(pj.row, line("드라이버", pd.row), line("toolkit", pk.row), line("torch", pt.row), ro.row);
  f.onResize(draw);
  update();
}
