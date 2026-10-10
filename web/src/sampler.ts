// The DDIM sampling loop. A line-for-line port of ddim_sample in train/diffusion.py;
// web/test/parity.test.ts checks that both turn the same noise into the same image.
//
// Images are Float32Array of length 3 * 32 * 32 in channel-first order (all red
// pixels, then all green, then all blue), values in [-1, 1].

export const SIZE = 32;
export const PIXELS = SIZE * SIZE;
export const LEN = 3 * PIXELS;

export type Schedule = {
  T: number;
  classes: string[];
  null_class: number;
  /** What the network outputs: predicted noise, or v (see v_target in train/diffusion.py). */
  pred: "eps" | "v";
  alpha_bar: number[]; // exported from Python, never recomputed here
};

/** One network call on a batch: x is B images back to back; returns B raw network outputs. */
export type ModelFn = (x: Float32Array, t: number, cls: number[]) => Promise<Float32Array>;

export type SampleOptions = {
  cls: number;
  guidance: number;
  steps: number;
  noise: Float32Array;
  /** SDEdit or inpainting source image. */
  init?: Float32Array;
  /** SDEdit only: 1 ignores `init`, 0.1 barely changes it. */
  strength?: number;
  /** Inpainting: one value per pixel (length 32 * 32), 1 where the model may paint, 0 where `init` is kept. */
  mask?: Float32Array;
  /** Called before each step with the current noisy image and the model's guess of the final one. */
  onStep?: (step: { index: number; count: number; t: number; xt: Float32Array; x0: Float32Array; ms: number }) => void;
  /** Checked before every network call; return true to abandon the run. */
  cancelled?: () => boolean;
};

/** `steps` timesteps evenly spaced from T - 1 down to 0. Integer maths, same as Python. */
export function ddimTimesteps(steps: number, T: number): number[] {
  if (steps === 1) return [T - 1];
  return Array.from({ length: steps }, (_, i) => Math.floor(((T - 1) * (steps - 1 - i) * 2 + (steps - 1)) / (2 * (steps - 1))));
}

/** Seeded standard-normal noise: mulberry32 for uniform numbers, Box-Muller to make them Gaussian. */
export function gaussianNoise(seed: number, length = LEN): Float32Array {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - uniform())); // 1 - u keeps log away from 0
    const angle = 2 * Math.PI * uniform();
    out[i] = r * Math.cos(angle);
    if (i + 1 < length) out[i + 1] = r * Math.sin(angle);
  }
  return out;
}

/** Runs the sampler. Returns the final image, or null if cancelled. */
export async function sample(model: ModelFn, schedule: Schedule, opts: SampleOptions): Promise<Float32Array | null> {
  const { cls, guidance, steps, noise, init, mask } = opts;
  const ab = schedule.alpha_bar;
  let ts = ddimTimesteps(steps, schedule.T);
  let x = Float32Array.from(noise);

  if (init && !mask) {
    // SDEdit: skip the noisiest steps and start from the image noised to that level.
    const keep = Math.max(1, Math.min(steps, Math.floor((opts.strength ?? 1) * steps + 0.5)));
    ts = ts.slice(ts.length - keep);
    const a = ab[ts[0]];
    for (let i = 0; i < LEN; i++) x[i] = Math.sqrt(a) * init[i] + Math.sqrt(1 - a) * noise[i];
  }

  for (let index = 0; index < ts.length; index++) {
    if (opts.cancelled?.()) return null;
    const t = ts[index];
    const a = ab[t];
    const aPrev = index + 1 < ts.length ? ab[ts[index + 1]] : 1;
    const started = performance.now();

    // Classifier-free guidance: conditional and unconditional in one batch of 2,
    // then step past the conditional answer, away from the unconditional one.
    let e: Float32Array;
    if (guidance === 1) {
      e = await model(x, t, [cls]);
    } else {
      const both = new Float32Array(2 * LEN);
      both.set(x, 0);
      both.set(x, LEN);
      const out = await model(both, t, [cls, schedule.null_class]);
      e = new Float32Array(LEN);
      for (let i = 0; i < LEN; i++) e[i] = out[LEN + i] + guidance * (out[i] - out[LEN + i]);
    }

    // A v-prediction network's output becomes predicted noise: eps = sqrt(a) * v + sqrt(1 - a) * x_t.
    if (schedule.pred === "v") for (let i = 0; i < LEN; i++) e[i] = Math.sqrt(a) * e[i] + Math.sqrt(1 - a) * x[i];

    const x0 = new Float32Array(LEN);
    const next = new Float32Array(LEN);
    for (let i = 0; i < LEN; i++) {
      // The clean image this noise prediction implies, kept inside the valid pixel range.
      x0[i] = Math.max(-1, Math.min(1, (x[i] - Math.sqrt(1 - a) * e[i]) / Math.sqrt(a)));
      const e2 = (x[i] - Math.sqrt(a) * x0[i]) / Math.sqrt(1 - a); // noise consistent with the clamped guess
      next[i] = Math.sqrt(aPrev) * x0[i] + Math.sqrt(1 - aPrev) * e2;
      if (mask && init) {
        // Inpainting: outside the hole, put back the original noised to this level.
        const m = mask[i % PIXELS];
        next[i] = m * next[i] + (1 - m) * (Math.sqrt(aPrev) * init[i] + Math.sqrt(1 - aPrev) * noise[i]);
      }
    }
    opts.onStep?.({ index, count: ts.length, t, xt: x, x0, ms: performance.now() - started });
    x = next;
  }
  return x;
}

/** [-1, 1] channel-first floats -> RGBA bytes for a canvas ImageData. */
export function toRGBA(image: Float32Array): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(PIXELS * 4);
  for (let p = 0; p < PIXELS; p++) {
    for (let c = 0; c < 3; c++) out[p * 4 + c] = (image[c * PIXELS + p] + 1) * 127.5;
    out[p * 4 + 3] = 255;
  }
  return out;
}

/** RGBA bytes from a 32x32 canvas -> [-1, 1] channel-first floats. */
export function fromRGBA(rgba: Uint8ClampedArray): Float32Array {
  const out = new Float32Array(LEN);
  for (let p = 0; p < PIXELS; p++) for (let c = 0; c < 3; c++) out[c * PIXELS + p] = rgba[p * 4 + c] / 127.5 - 1;
  return out;
}
