// Runs off the main thread: owns the ONNX session and the sampling loop, and
// streams every step back so the page can cache it for scrubbing.
import * as ort from "onnxruntime-web/webgpu";
import { sample, type ModelFn, type Schedule } from "./sampler";

// ponytail: the runtime's .wasm files come from the CDN; copy them into public/ if the site must work without it.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

export type Request =
  | { type: "load"; forceWasm: boolean }
  | {
      type: "generate";
      id: number;
      cls: number;
      guidance: number;
      steps: number;
      noise: Float32Array;
      init?: Float32Array;
      strength?: number;
      mask?: Float32Array;
    };

export type Reply =
  | { type: "ready"; backend: "webgpu" | "wasm"; bytes: number; schedule: Schedule }
  | { type: "error"; message: string }
  | { type: "step"; id: number; index: number; count: number; xt: Float32Array; x0: Float32Array; ms: number }
  | { type: "done"; id: number; image: Float32Array };

const post = (reply: Reply) => self.postMessage(reply);
let session: ort.InferenceSession;
let schedule: Schedule;
let current = 0; // id of the newest request; older runs see the mismatch and stop

const model: ModelFn = async (x, t, cls) => {
  const b = cls.length;
  const out = await session.run({
    x_t: new ort.Tensor("float32", x, [b, 3, 32, 32]),
    t: new ort.Tensor("int64", BigInt64Array.from(cls.map(() => BigInt(t))), [b]),
    cls: new ort.Tensor("int64", BigInt64Array.from(cls.map(BigInt)), [b]),
  });
  return out.out.data as Float32Array;
};

async function load(forceWasm: boolean) {
  const base = import.meta.env.BASE_URL + "model/";
  schedule = await (await fetch(base + "schedule.json")).json();
  const bytes = await (await fetch(base + "chisel.onnx")).arrayBuffer();
  let backend: "webgpu" | "wasm" = "wasm";
  if (!forceWasm && "gpu" in navigator) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      backend = "webgpu";
    } catch {
      // No usable GPU adapter, or an operator the WebGPU backend lacks: fall through to WASM.
    }
  }
  if (backend === "wasm") session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", backend, bytes: bytes.byteLength, schedule });
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const request = event.data;
  try {
    if (request.type === "load") return await load(request.forceWasm);
    const id = (current = request.id);
    const image = await sample(model, schedule, {
      ...request,
      cancelled: () => current !== id,
      onStep: ({ index, count, xt, x0, ms }) => post({ type: "step", id, index, count, xt, x0, ms }),
    });
    if (image) post({ type: "done", id, image });
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
