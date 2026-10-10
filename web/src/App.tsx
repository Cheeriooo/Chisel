import { applyPalette, GIFEncoder, quantize } from "gifenc";
import { useCallback, useEffect, useRef, useState } from "react";
import { fromRGBA, gaussianNoise, PIXELS, SIZE, toRGBA, type Schedule } from "./sampler";
import type { Reply, Request } from "./worker";

type Frame = { xt: Float32Array; x0: Float32Array };
type Mode = "generate" | "sketch" | "inpaint";
type Ready = { backend: "webgpu" | "wasm"; bytes: number; schedule: Schedule };

const ICONS = ["😀", "🧑", "🐾", "🍔", "✈️", "⚽", "💡", "🔣"];
const STEP_CHOICES = [5, 10, 20, 50];
const PALETTE = ["#1c1a17", "#ffffff", "#ffcc4d", "#f4900c", "#dd2e44", "#f4abba", "#77b255", "#55acee", "#aa8dd8", "#8a4b38"];
const EXPORT_SCALE = 8;

const query = new URLSearchParams(location.search);
const num = (key: string, fallback: number) => (query.has(key) && isFinite(Number(query.get(key))) ? Number(query.get(key)) : fallback);
const blankSketch = () => new Uint8ClampedArray(PIXELS * 4).fill(255);
const hexToRGB = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Nearest-neighbour upscale of a 32x32 RGBA image, for exports. */
function upscale(rgba: Uint8ClampedArray, k: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(PIXELS * 4 * k * k);
  for (let y = 0; y < SIZE * k; y++)
    for (let x = 0; x < SIZE * k; x++) {
      const from = (Math.floor(y / k) * SIZE + Math.floor(x / k)) * 4;
      out.set(rgba.subarray(from, from + 4), (y * SIZE * k + x) * 4);
    }
  return out;
}

function download(blob: Blob, name: string) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

export default function App() {
  const [ready, setReady] = useState<Ready | null>(null);
  const [error, setError] = useState("");
  const [cls, setCls] = useState(num("class", 0));
  const [guidance, setGuidance] = useState(num("guidance", 2));
  const [steps, setSteps] = useState(num("steps", 20));
  const [seed, setSeed] = useState(num("seed", 7));
  const [mode, setMode] = useState<Mode>("generate");
  const [strength, setStrength] = useState(0.6);
  const [color, setColor] = useState(PALETTE[2]);
  const [brush, setBrush] = useState(2);

  const [frames, setFrames] = useState<Frame[]>([]);
  const [final, setFinal] = useState<Float32Array | null>(null);
  const [pos, setPos] = useState(0);
  const [view, setView] = useState<"xt" | "x0">("xt");
  const [busy, setBusy] = useState(false);
  const [msPerStep, setMsPerStep] = useState(0);
  const [editing, setEditing] = useState(false); // true while the slab shows the user's drawing, not a result
  const [, redraw] = useState(0);

  const worker = useRef<Worker | null>(null);
  const runId = useRef(0);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sketch = useRef(blankSketch());
  const mask = useRef(new Float32Array(PIXELS));
  const base = useRef<Float32Array | null>(null); // image being repaired in inpaint mode
  const lastPoint = useRef<[number, number] | null>(null);

  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = ({ data }: MessageEvent<Reply>) => {
      if (data.type === "ready") setReady(data);
      else if (data.type === "error") { setError(data.message); setBusy(false); }
      else if (data.id !== runId.current) return; // a newer request superseded this one
      else if (data.type === "step") {
        setFrames((f) => [...f, { xt: data.xt, x0: data.x0 }]);
        setPos(data.index);
        if (data.index > 0) setMsPerStep((ms) => (ms ? ms * 0.7 + data.ms * 0.3 : data.ms)); // step 0 includes warm-up
      } else {
        setFinal(data.image);
        setPos((p) => p + 1);
        setBusy(false);
      }
    };
    w.postMessage({ type: "load", forceWasm: query.has("wasm") } satisfies Request);
    return () => w.terminate();
  }, []);

  const run = useCallback(
    (overrides: { seed?: number; cls?: number } = {}) => {
      if (!worker.current || !ready) return;
      const s = overrides.seed ?? seed;
      const c = overrides.cls ?? cls;
      const request: Request = { type: "generate", id: ++runId.current, cls: c, guidance, steps, noise: gaussianNoise(s) };
      if (mode === "sketch") Object.assign(request, { init: fromRGBA(sketch.current), strength });
      if (mode === "inpaint" && base.current) Object.assign(request, { init: base.current, mask: mask.current.slice() });
      setFrames([]); setFinal(null); setPos(0); setBusy(true); setEditing(false); setError("");
      worker.current.postMessage(request);
      // The link is a full description of a plain generation: DDIM adds no randomness of its own.
      const url = new URL(location.href);
      for (const [k, v] of Object.entries({ class: c, guidance, steps, seed: s })) url.searchParams.set(k, String(v));
      history.replaceState(null, "", url);
    },
    [ready, seed, cls, guidance, steps, mode, strength],
  );

  // First image as soon as the model is in.
  useEffect(() => { if (ready) run(); }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const shown = pos < frames.length ? frames[pos][view] : (final ?? frames.at(-1)?.x0 ?? null);

  // Paint the slab: the drawing while editing, otherwise the frame under the scrubber.
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    let rgba: Uint8ClampedArray<ArrayBuffer>;
    if (editing && mode === "sketch") rgba = sketch.current.slice();
    else if (editing && mode === "inpaint" && base.current) {
      rgba = toRGBA(base.current);
      for (let p = 0; p < PIXELS; p++)
        if (mask.current[p]) rgba.set(((p % SIZE) + Math.floor(p / SIZE)) % 2 ? [226, 64, 28, 255] : [236, 231, 223, 255], p * 4);
    } else if (shown) rgba = toRGBA(shown);
    else return;
    ctx.putImageData(new ImageData(rgba, SIZE, SIZE), 0, 0);
  });

  const paint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (mode === "generate" || (mode === "inpaint" && !final && !base.current)) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point: [number, number] = [((event.clientX - rect.left) / rect.width) * SIZE, ((event.clientY - rect.top) / rect.height) * SIZE];
    if (event.type === "pointerdown") {
      event.currentTarget.setPointerCapture(event.pointerId);
      if (!editing) {
        // First touch after a result: go back to the drawing (sketch), or start a fresh hole in the latest image (inpaint).
        if (mode === "inpaint") { base.current = final ?? base.current; mask.current.fill(0); }
        setEditing(true);
      }
      lastPoint.current = point;
    } else if (!lastPoint.current) return;
    const [x0, y0] = lastPoint.current!;
    const rgb = hexToRGB(color);
    const n = Math.ceil(Math.hypot(point[0] - x0, point[1] - y0) * 2) || 1;
    for (let i = 0; i <= n; i++) {
      const cx = x0 + ((point[0] - x0) * i) / n, cy = y0 + ((point[1] - y0) * i) / n;
      for (let y = Math.floor(cy - brush); y <= cy + brush; y++)
        for (let x = Math.floor(cx - brush); x <= cx + brush; x++) {
          if (x < 0 || y < 0 || x >= SIZE || y >= SIZE || Math.hypot(x + 0.5 - cx, y + 0.5 - cy) > brush) continue;
          if (mode === "sketch") sketch.current.set(rgb, (y * SIZE + x) * 4);
          else mask.current[y * SIZE + x] = 1;
        }
    }
    lastPoint.current = point;
    redraw((v) => v + 1);
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    if (next === "inpaint") { base.current = final; mask.current.fill(0); }
    setEditing(next === "sketch" || (next === "inpaint" && !!final));
  };

  const savePNG = () => {
    if (!shown) return;
    const out = document.createElement("canvas");
    out.width = out.height = SIZE * EXPORT_SCALE;
    out.getContext("2d")!.putImageData(new ImageData(upscale(toRGBA(shown), EXPORT_SCALE), out.width, out.height), 0, 0);
    out.toBlob((blob) => blob && download(blob, `chisel-${seed}.png`));
  };

  const saveGIF = () => {
    if (!final) return;
    const gif = GIFEncoder();
    const images = [...frames.map((f) => f[view]), final];
    images.forEach((image, i) => {
      const rgba = upscale(toRGBA(image), EXPORT_SCALE);
      const palette = quantize(rgba, 256);
      gif.writeFrame(applyPalette(rgba, palette), SIZE * EXPORT_SCALE, SIZE * EXPORT_SCALE, { palette, delay: i === images.length - 1 ? 1500 : Math.max(60, 1600 / images.length) });
    });
    gif.finish();
    download(new Blob([gif.bytes() as BlobPart], { type: "image/gif" }), `chisel-${seed}.gif`);
  };

  const reroll = () => { const s = Math.floor(Math.random() * 100000); setSeed(s); run({ seed: s }); };
  const count = frames.length + (final ? 1 : 0);
  const canDraw = mode !== "generate";

  return (
    <main className="shop">
      <header className="mast">
        <h1>Chisel</h1>
        <p>
          A {ready ? (ready.bytes / 2 ** 20).toFixed(1) + " MB" : "tiny"} diffusion model, trained from scratch on emoji. Every
          step below is computed in this tab. Drag the ruler to watch the image get carved out of noise.
        </p>
      </header>

      <section className="bench">
        <div className={"slab" + (canDraw ? " drawable" : "")}>
          <canvas
            ref={canvas} width={SIZE} height={SIZE} onPointerDown={paint} onPointerMove={paint}
            onPointerUp={() => (lastPoint.current = null)} onPointerCancel={() => (lastPoint.current = null)}
            role="img" aria-label={editing ? "Drawing surface" : `Generated ${ready?.schedule.classes[cls] ?? ""} emoji, step ${pos} of ${count - 1}`}
          />
          {!ready && !error && <div className="veil">loading model…</div>}
          {error && <div className="veil error">{error}</div>}
        </div>

        <div className="ruler">
          <input
            type="range" min={0} max={Math.max(0, count - 1)} value={Math.min(pos, Math.max(0, count - 1))} disabled={editing || count < 2}
            onChange={(e) => setPos(Number(e.target.value))} aria-label="Denoising step" style={{ "--ticks": Math.max(1, count - 1) } as React.CSSProperties}
          />
          <div className="ruler-ends"><span>noise</span><span>{editing ? "drawing" : busy ? "carving…" : `step ${Math.min(pos, count - 1) + 1} / ${count || steps}`}</span><span>emoji</span></div>
        </div>

        <div className="row">
          <div className="seg" role="group" aria-label="What to show">
            <button aria-pressed={view === "xt"} onClick={() => setView("xt")}>noisy image</button>
            <button aria-pressed={view === "x0"} onClick={() => setView("x0")}>model's guess</button>
          </div>
          <div className="seg">
            <button onClick={savePNG} disabled={!shown}>PNG</button>
            <button onClick={saveGIF} disabled={!final}>GIF</button>
          </div>
        </div>
        <p className="readout">
          {ready ? <>{ready.backend === "webgpu" ? "WebGPU" : "WASM"} · {(ready.bytes / 2 ** 20).toFixed(1)} MB · {msPerStep ? msPerStep.toFixed(0) + " ms/step" : "—"}</> : "…"}
        </p>
      </section>

      <section className="tools">
        <div className="seg modes" role="group" aria-label="Mode">
          <button aria-pressed={mode === "generate"} onClick={() => switchMode("generate")}>Generate</button>
          <button aria-pressed={mode === "sketch"} onClick={() => switchMode("sketch")}>Sketch</button>
          <button aria-pressed={mode === "inpaint"} onClick={() => switchMode("inpaint")}>Repair</button>
        </div>
        <p className="hint">
          {mode === "generate" && "Pick a class and a seed. The same link always carves the same emoji."}
          {mode === "sketch" && "Draw a rough, filled-in shape on the slab. The model noises it part-way, then finishes it."}
          {mode === "inpaint" && (final || base.current ? "Scrub out part of the image on the slab. The model refills only that part." : "Generate something first, then erase part of it.")}
        </p>

        <fieldset>
          <legend>Class</legend>
          <div className="chips">
            {(ready?.schedule.classes ?? []).map((name, i) => (
              <button key={name} aria-pressed={cls === i} onClick={() => { setCls(i); if (mode === "generate") run({ cls: i }); }} title={name}>
                <span aria-hidden>{ICONS[i]}</span> {name.split(" ")[0]}
              </button>
            ))}
          </div>
        </fieldset>

        {canDraw && (
          <fieldset>
            <legend>{mode === "sketch" ? "Brush" : "Eraser"}</legend>
            {mode === "sketch" && (
              <div className="palette">
                {PALETTE.map((c) => <button key={c} aria-pressed={color === c} aria-label={`Colour ${c}`} style={{ background: c }} onClick={() => setColor(c)} />)}
              </div>
            )}
            <label>size <output>{brush}</output><input type="range" min={1} max={5} step={0.5} value={brush} onChange={(e) => setBrush(Number(e.target.value))} /></label>
            <button className="quiet" onClick={() => { if (mode === "sketch") sketch.current = blankSketch(); else mask.current.fill(0); setEditing(true); redraw((v) => v + 1); }}>clear</button>
          </fieldset>
        )}

        <fieldset>
          <legend>Settings</legend>
          {mode === "sketch" && (
            <label>strength <output>{strength.toFixed(2)}</output>
              <input type="range" min={0.1} max={1} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
              <small>low keeps your drawing, high lets the model take over</small>
            </label>
          )}
          <label>guidance <output>{guidance.toFixed(1)}</output>
            <input type="range" min={0} max={6} step={0.5} value={guidance} onChange={(e) => setGuidance(Number(e.target.value))} />
            <small>0 ignores the class, high follows it at the cost of variety</small>
          </label>
          <div className="field">steps
            <div className="seg">{STEP_CHOICES.map((s) => <button key={s} aria-pressed={steps === s} onClick={() => setSteps(s)}>{s}</button>)}</div>
          </div>
          <label className="field">seed
            <input type="number" value={seed} min={0} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
          </label>
        </fieldset>

        <div className="actions">
          <button className="go" disabled={!ready || (mode === "inpaint" && !base.current)} onClick={() => run()}>
            {mode === "generate" ? "Carve" : mode === "sketch" ? "Complete my sketch" : "Fill the hole"}
          </button>
          <button className="quiet" disabled={!ready} onClick={reroll}>new seed</button>
        </div>
      </section>

      <footer>
        Emoji from Noto Emoji (Apache 2.0), Fluent Emoji (MIT) and Twemoji (CC BY 4.0). Built by Rakesh Sawant.
      </footer>
    </main>
  );
}
