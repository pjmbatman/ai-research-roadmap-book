/// <reference lib="webworker" />
// 브라우저 Python 실행기. 페이지와 분리된 스레드에서 돌아가므로 계산 중에도 화면이 멈추지 않는다.
// 세션(ns)마다 별도의 globals 딕셔너리를 써서 코드 셀끼리의 변수 공유 범위를 나눈다.
let pyodide: any = null;
let current = 0;
const spaces = new Map<string, any>();
const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function space(ns: string) {
  if (!spaces.has(ns)) {
    const d = pyodide.globals.get("dict")();
    d.set("__name__", "__main__");
    spaces.set(ns, d);
  }
  return spaces.get(ns);
}

self.onmessage = async (e: MessageEvent) => {
  const { id, type } = e.data;
  try {
    if (type === "init") {
      const mod = await import(/* @vite-ignore */ e.data.indexURL + "pyodide.mjs");
      pyodide = await mod.loadPyodide({ indexURL: e.data.indexURL });
      pyodide.setStdout({ batched: (s: string) => post({ id: current, type: "stdout", text: s + "\n" }) });
      pyodide.setStderr({ batched: (s: string) => post({ id: current, type: "stdout", text: s + "\n" }) });
      post({ id, type: "done" });
      return;
    }
    if (type === "reset") {
      spaces.get(e.data.ns)?.destroy?.();
      spaces.delete(e.data.ns);
      post({ id, type: "done" });
      return;
    }
    if (type === "run") {
      current = id;
      for (const [path, content] of Object.entries(e.data.files as Record<string, string>)) {
        const dir = path.split("/").slice(0, -1).join("/");
        if (dir) pyodide.FS.mkdirTree(dir);
        pyodide.FS.writeFile(path, content, { encoding: "utf8" });
      }
      // 실행 코드뿐 아니라 함께 쓴 파일(제출 코드·테스트)의 import도 보고 필요한 패키지를 불러온다.
      const sources = [e.data.code, ...Object.entries(e.data.files as Record<string, string>).filter(([p]) => p.endsWith(".py")).map(([, c]) => c)];
      await pyodide.loadPackagesFromImports(sources.join("\n"));
      const result = await pyodide.runPythonAsync(e.data.code, { globals: space(e.data.ns) });
      let value: unknown = undefined;
      if (result && typeof result.toJs === "function") { value = result.toJs({ dict_converter: Object.fromEntries }); result.destroy?.(); }
      else if (["string", "number", "boolean"].includes(typeof result)) value = result;
      post({ id, type: "done", value });
    }
  } catch (err: any) {
    post({ id, type: "error", error: String(err?.message ?? err) });
  }
};
