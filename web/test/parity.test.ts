// Does the TypeScript sampler agree with the Python one?
// export/export_onnx.py saves a starting noise tensor and the image Python's
// ddim_sample made from it. Same model, same noise, same settings here.
//
//   python export/export_onnx.py --run base
//   npm test                      (RUN=other-run npm test for a different run)
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { expect, test } from "vitest";
import { ddimTimesteps, gaussianNoise, LEN, sample, type ModelFn, type Schedule } from "../src/sampler";

const dir = resolve(import.meta.dirname, "../../runs", process.env.RUN ?? "base", "export");

test("ddimTimesteps matches Python", () => {
  expect(ddimTimesteps(20, 1000).slice(0, 3)).toEqual([999, 946, 894]);
  expect(ddimTimesteps(5, 1000)).toEqual([999, 749, 500, 250, 0]);
  expect(ddimTimesteps(1, 1000)).toEqual([999]);
});

test("gaussianNoise is seeded and roughly standard normal", () => {
  const a = gaussianNoise(7);
  expect(a).toEqual(gaussianNoise(7));
  expect(a).not.toEqual(gaussianNoise(8));
  const mean = a.reduce((s, v) => s + v, 0) / a.length;
  const variance = a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length;
  expect(Math.abs(mean)).toBeLessThan(0.08);
  expect(Math.abs(variance - 1)).toBeLessThan(0.1);
});

test.skipIf(!existsSync(resolve(dir, "parity.bin")))("20-step sample matches Python", async () => {
  const schedule: Schedule = JSON.parse(readFileSync(resolve(dir, "schedule.json"), "utf8"));
  const settings = JSON.parse(readFileSync(resolve(dir, "parity.json"), "utf8"));
  const fixture = new Float32Array(new Uint8Array(readFileSync(resolve(dir, "parity.bin"))).buffer);
  const noise = fixture.slice(0, LEN);
  const expected = fixture.slice(LEN);

  const session = await ort.InferenceSession.create(resolve(dir, "chisel.onnx"));
  const model: ModelFn = async (x, t, cls) => {
    const b = cls.length;
    const out = await session.run({
      x_t: new ort.Tensor("float32", x, [b, 3, 32, 32]),
      t: new ort.Tensor("int64", BigInt64Array.from(cls.map(() => BigInt(t))), [b]),
      cls: new ort.Tensor("int64", BigInt64Array.from(cls.map(BigInt)), [b]),
    });
    return out.out.data as Float32Array;
  };

  const got = await sample(model, schedule, { cls: settings.cls, guidance: settings.guidance, steps: settings.steps, noise });
  let worst = 0;
  for (let i = 0; i < LEN; i++) worst = Math.max(worst, Math.abs(got![i] - expected[i]));
  console.log(`max abs pixel diff, TypeScript vs Python sampler: ${worst.toExponential(2)}`);
  expect(worst).toBeLessThan(1e-2);
});
