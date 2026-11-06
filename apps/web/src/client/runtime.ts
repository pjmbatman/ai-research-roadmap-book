// 런타임 어댑터(ADR 0004). v1은 브라우저 안의 Pyodide만 구현한다.
// 페이지마다 Web Worker 하나에 Pyodide를 한 번만 띄우고, 세션 이름마다 별도의 Python 이름공간을 쓴다.
// 로컬 런타임(localhost 커널)은 같은 인터페이스로 v2에서 추가한다.

export interface RunResult { ok: boolean; stdout: string; error?: string; value?: unknown }
export interface Runtime {
  name: string;
  run(code: string, opts?: { files?: Record<string, string>; onStdout?: (s: string) => void }): Promise<RunResult>;
  reset(): void;
}

export const PYODIDE_VERSION = "0.29.5";
const DEFAULT_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

function indexURL(): string {
  const env = (import.meta as any).env?.PUBLIC_PYODIDE_URL as string | undefined;
  if (env) return env.startsWith("http") ? env : new URL(env, location.origin).href;
  return DEFAULT_INDEX;
}

export type RuntimeState = "idle" | "loading" | "ready" | "running" | "error";
type Listener = (state: RuntimeState, detail?: string) => void;
const listeners = new Set<Listener>();
let lastState: RuntimeState = "idle";
export function onRuntimeState(fn: Listener) { listeners.add(fn); fn(lastState); return () => listeners.delete(fn); }
function emit(state: RuntimeState, detail?: string) { lastState = state; listeners.forEach((f) => f(state, detail)); }

interface Pending { resolve: (r: RunResult) => void; out: string[]; onStdout?: (s: string) => void }
let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, Pending>();
let ready: Promise<void> | null = null;

function boot(): Promise<void> {
  if (ready) return ready;
  emit("loading", "브라우저 Python을 불러오는 중");
  worker = new Worker(new URL("./py.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent) => {
    const m = e.data;
    const p = pending.get(m.id);
    if (!p) return;
    if (m.type === "stdout") { p.out.push(m.text); p.onStdout?.(m.text); return; }
    pending.delete(m.id);
    p.resolve({ ok: m.type === "done", stdout: p.out.join(""), error: m.error, value: m.value });
    if (lastState !== "loading") emit(pending.size ? "running" : "ready");
  };
  ready = new Promise<void>((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve: (r) => (r.ok ? resolve() : reject(new Error(r.error))), out: [] });
    worker!.onerror = (ev) => { pending.delete(id); reject(new Error(ev.message || "worker error")); };
    worker!.postMessage({ id, type: "init", indexURL: indexURL() });
  }).then(
    () => emit("ready", "브라우저 Python 준비됨"),
    (err) => { emit("error", String(err?.message ?? err)); worker?.terminate(); worker = null; ready = null; throw err; },
  );
  return ready;
}

class Session implements Runtime {
  name = "pyodide";
  constructor(private ns: string) {}
  async run(code: string, opts: { files?: Record<string, string>; onStdout?: (s: string) => void } = {}): Promise<RunResult> {
    try { await boot(); } catch (err: any) {
      return { ok: false, stdout: "", error: `브라우저 Python을 불러오지 못했다. 네트워크 연결을 확인한 뒤 다시 실행한다. (${err?.message ?? err})` };
    }
    emit("running");
    return new Promise((resolve) => {
      const id = ++seq;
      pending.set(id, { resolve, out: [], onStdout: opts.onStdout });
      worker!.postMessage({ id, type: "run", ns: this.ns, code, files: opts.files ?? {} });
    });
  }
  reset() { if (worker) worker.postMessage({ id: ++seq, type: "reset", ns: this.ns }); }
}

const sessions = new Map<string, Session>();
export function session(name: string): Runtime {
  if (!sessions.has(name)) sessions.set(name, new Session(name));
  return sessions.get(name)!;
}
